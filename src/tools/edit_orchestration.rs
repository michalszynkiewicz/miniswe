//! Edit orchestration: the outer layer around `edit_file` and `write_file`
//! that captures pre-edit baselines, re-indexes changed files, and runs
//! `auto_check` (LSP + compiler fallback) to surface regressions.
//!
//! The inner model-driven editor in [`super::edit_file`] owns patch
//! generation; this module owns the post-edit gating / confirmation.

use anyhow::Result;
use serde_json::Value;

use crate::config::{Config, ModelRole};
use crate::knowledge::ProjectIndex;
use crate::knowledge::indexer;
use crate::llm::{ChatRequest, Message, ModelRouter};
use crate::logging::SessionLog;
use crate::lsp::LspClient;

use super::ToolResult;
use super::cargo_check::{extract_error_locations, read_source_context, run_check_with_timeout};
use super::edit_file;
use super::permissions::PermissionManager;

/// Execute `edit_file` through the same permission and post-edit path used by
/// file(write/replace). The inner fixer still owns patch generation and
/// candidate validation.
pub async fn execute_edit_file_tool(
    args: &Value,
    config: &Config,
    perms: &PermissionManager,
    router: &ModelRouter,
    lsp: Option<&LspClient>,
    cancelled: Option<&std::sync::atomic::AtomicBool>,
    log: Option<&SessionLog>,
) -> Result<ToolResult> {
    let path = args["path"].as_str().unwrap_or("");
    if let Err(e) = perms.resolve_and_check_path(path) {
        return Ok(ToolResult::err(e));
    }

    let baseline = capture_edit_baseline(path, config, lsp).await;
    // Pass the LSP error baseline into edit_file so its inner candidate
    // validation uses the same pre-edit count as the outer auto_check.
    // Without this, the inner check captures its own baseline at a later
    // moment — and on slow LSPs (e.g. rust-analyzer still settling) the
    // two snapshots can disagree, producing contradictory result lines
    // ("[lsp] OK 2 -> 1" alongside "[lsp] 2 errors (was 1 before)").
    let baseline_lsp_errors = if baseline.existed_before {
        Some(baseline.lsp_errors)
    } else {
        None
    };
    let mut result = edit_file::execute(
        args,
        config,
        router,
        lsp,
        cancelled,
        log,
        baseline_lsp_errors,
        Some(perms),
    )
    .await?;
    if result.success {
        finalize_file_edit(path, config, &mut result, lsp, baseline, Some(router)).await;
    }
    Ok(result)
}

/// State captured for a file *before* an edit runs. Used by `auto_check`
/// to distinguish errors the edit *introduced* from errors that were
/// already there (or that the file is brand new and we have no baseline
/// at all).
#[derive(Debug, Clone)]
pub(super) struct EditBaseline {
    /// LSP error count for the file before the edit. 0 if LSP is
    /// unavailable, the file is new, or the count couldn't be obtained.
    pub lsp_errors: usize,
    /// Whether the file existed on disk before the edit. False for files
    /// being created by write_file or by edit_file's new-file path.
    pub existed_before: bool,
    /// Original file content before the edit. Used to rollback if the
    /// outer model rejects an LSP-regressed edit.
    pub original_content: Option<String>,
}

impl EditBaseline {
    #[allow(dead_code)]
    pub(super) fn new_unknown() -> Self {
        Self {
            lsp_errors: 0,
            existed_before: false,
            original_content: None,
        }
    }
}

/// Capture pre-edit baseline state for a file. Returns LSP error count
/// (0 if unavailable) and whether the file existed on disk before this
/// edit. Called *before* dispatching to write_file / edit_file / replace
/// so that `auto_check` can tell brand-new files apart from edits.
pub(super) async fn capture_edit_baseline(
    path: &str,
    config: &Config,
    lsp: Option<&LspClient>,
) -> EditBaseline {
    let abs_path = config.project_root.join(path);
    let existed_before = abs_path.exists();
    let original_content = if existed_before {
        std::fs::read_to_string(&abs_path).ok()
    } else {
        None
    };
    let lsp_errors = if existed_before {
        lsp_error_count_inner(&abs_path, config, lsp).await
    } else {
        0
    };
    EditBaseline {
        lsp_errors,
        existed_before,
        original_content,
    }
}

pub(super) async fn finalize_file_edit(
    path: &str,
    config: &Config,
    result: &mut ToolResult,
    lsp: Option<&LspClient>,
    baseline: EditBaseline,
    router: Option<&ModelRouter>,
) {
    reindex_changed_file(path, config);
    auto_check(path, config, result, lsp, baseline, router).await;
}

/// Inner: query the LSP error count for an existing file path. Returns 0
/// if LSP is unavailable or the file isn't reachable.
async fn lsp_error_count_inner(
    abs_path: &std::path::Path,
    config: &Config,
    lsp: Option<&LspClient>,
) -> usize {
    let Some(lsp) = lsp else {
        return 0;
    };
    if !lsp.is_ready() || lsp.has_crashed() {
        return 0;
    }
    if lsp.notify_file_changed(abs_path).is_err() {
        return 0;
    }
    let timeout = std::time::Duration::from_millis(config.lsp.diagnostic_timeout_ms);
    let diags = lsp.get_diagnostics(abs_path, timeout).await;
    diags
        .iter()
        .filter(|d| d.severity == Some(lsp_types::DiagnosticSeverity::ERROR))
        .count()
}

/// Re-index a single changed file. Best-effort — doesn't fail the tool call.
fn reindex_changed_file(rel_path: &str, config: &Config) {
    let miniswe_dir = config.miniswe_dir();
    let abs_path = config.project_root.join(rel_path);

    let mut index = match ProjectIndex::load(&miniswe_dir) {
        Ok(idx) => idx,
        Err(_) => return, // no index yet, skip
    };

    indexer::reindex_file(rel_path, &abs_path, &mut index, &miniswe_dir);
}

/// Auto-run type checker after editing a source file. Appends output + source
/// context around errors to the tool result. Runs in a blocking thread to
/// avoid stalling the async runtime.
///
/// `baseline` carries the LSP error count for the file *before* the edit
/// was applied AND whether the file existed at all. The result is marked as
/// failure only if the edit caused the error count to grow on a pre-existing
/// file. Pre-existing errors that the edit didn't touch are surfaced as
/// informational; errors in brand-new files (where we have no baseline at
/// all) are surfaced as WARNINGs — non-blocking, but flagged loudly enough
/// that the agent doesn't shrug them off. This applies to both the LSP path
/// and the non-LSP fallback checkers (cargo check, py_compile, tsc, go vet,
/// mvn).
async fn auto_check(
    path: &str,
    config: &Config,
    result: &mut ToolResult,
    lsp: Option<&LspClient>,
    baseline: EditBaseline,
    router: Option<&ModelRouter>,
) {
    let baseline_errors = baseline.lsp_errors;
    let is_new_file = !baseline.existed_before;
    // Try LSP diagnostics first — ~200ms vs 2-5s for compiler check
    {
        if let Some(lsp) = lsp
            && lsp.is_ready()
            && !lsp.has_crashed()
        {
            let abs_path = config.project_root.join(path);
            if lsp.notify_file_changed(&abs_path).is_ok() {
                let timeout = std::time::Duration::from_millis(config.lsp.diagnostic_timeout_ms);
                let diags = lsp.get_diagnostics(&abs_path, timeout).await;
                // Always proceed — get_diagnostics already waited for the timeout
                {
                    let errors: Vec<&lsp_types::Diagnostic> = diags
                        .iter()
                        .filter(|d| d.severity == Some(lsp_types::DiagnosticSeverity::ERROR))
                        .collect();
                    // For brand-new files we have no baseline at all,
                    // so any errors are surfaced as informational —
                    // don't flip success on new helper scripts whose
                    // imports happen to be unresolved.
                    let regressed = !is_new_file && errors.len() > baseline_errors;
                    if errors.is_empty() {
                        result.content.push_str("\n[lsp] OK");
                    } else if is_new_file {
                        let capped = errors.len().min(5);
                        result.content.push_str(&format!(
                                "\n[lsp] WARNING: {} error(s) introduced by newly-created {path} — fix before relying on this file:\n",
                                errors.len()
                            ));
                        for diag in &errors[..capped] {
                            let line = diag.range.start.line + 1;
                            let col = diag.range.start.character + 1;
                            result.content.push_str(&format!(
                                "{}:{}:{}: error: {}\n",
                                path, line, col, diag.message
                            ));
                        }
                        if errors.len() > capped {
                            result.content.push_str(&format!(
                                "... and {} more errors\n",
                                errors.len() - capped
                            ));
                        }
                    } else if !regressed {
                        // Pre-existing errors weren't introduced by this
                        // edit — surface them as informational, don't
                        // flip success.
                        result.content.push_str(&format!(
                                "\n[lsp] OK ({} pre-existing error(s) in {path}, unchanged by this edit)",
                                errors.len()
                            ));
                    } else {
                        // LSP regression: errors increased after this edit.
                        // Build an error report and ask the outer model
                        // whether to keep or revert the changes.
                        let capped = errors.len().min(20);
                        let mut error_report = format!(
                            "{} error(s) in {path} (was {} before this edit):\n",
                            errors.len(),
                            baseline_errors
                        );
                        for diag in &errors[..capped] {
                            let line = diag.range.start.line + 1;
                            let col = diag.range.start.character + 1;
                            error_report.push_str(&format!(
                                "  {}:{}:{}: error: {}\n",
                                path, line, col, diag.message
                            ));
                        }
                        if errors.len() > capped {
                            error_report.push_str(&format!(
                                "  ... and {} more errors\n",
                                errors.len() - capped
                            ));
                        }

                        let accepted = if let Some(router) = router {
                            ask_accept_lsp_regression(router, path, &error_report).await
                        } else {
                            false
                        };

                        if accepted {
                            result.content.push_str(&format!(
                                "\n[lsp] WARNING: {error_report}Changes kept (accepted by model)."
                            ));
                        } else {
                            // Revert the file to its pre-edit content.
                            if let Some(ref original) = baseline.original_content {
                                let abs = config.project_root.join(path);
                                let _ = std::fs::write(&abs, original);
                                // Re-notify LSP so it sees the reverted state.
                                let _ = lsp.notify_file_changed(&abs);
                            }
                            result
                                .content
                                .push_str(&format!("\n[lsp] {error_report}Edit reverted."));
                            result.success = false;
                        }
                    }
                    return;
                }
            }
        }
    }
    // Fallback: cargo check
    // Determine which checker to run based on file extension and project markers
    let (cmd, args) = if path.ends_with(".rs") && config.project_root.join("Cargo.toml").exists() {
        ("cargo", vec!["check", "--message-format=short"])
    } else if (path.ends_with(".ts") || path.ends_with(".tsx"))
        && config.project_root.join("tsconfig.json").exists()
    {
        ("npx", vec!["tsc", "--noEmit", "--pretty", "false"])
    } else if path.ends_with(".go") && config.project_root.join("go.mod").exists() {
        ("go", vec!["vet", "./..."])
    } else if path.ends_with(".py") {
        ("python3", vec!["-m", "py_compile", path])
    } else if path.ends_with(".java") && config.project_root.join("pom.xml").exists() {
        ("mvn", vec!["compile", "-q"])
    } else if path.ends_with(".java") && config.project_root.join("build.gradle").exists() {
        ("gradle", vec!["compileJava", "-q"])
    } else {
        return; // no checker for this language
    };

    let project_root = config.project_root.clone();
    let project_root2 = project_root.clone();
    let cmd = cmd.to_string();
    let args: Vec<String> = args.into_iter().map(|s| s.to_string()).collect();

    // Run in a blocking thread with timeout to avoid stalling the async runtime
    // and to prevent pipe deadlock (same pattern as shell.rs).
    let check_result = tokio::task::spawn_blocking(move || {
        run_check_with_timeout(&cmd, &args, &project_root2, 30)
    })
    .await;

    let (success, stderr) = match check_result {
        Ok(Some(r)) => r,
        Ok(None) | Err(_) => return, // checker not available or panicked
    };

    let checker_name = if path.ends_with(".rs") {
        "cargo check"
    } else if path.ends_with(".ts") || path.ends_with(".tsx") {
        "tsc"
    } else if path.ends_with(".go") {
        "go vet"
    } else if path.ends_with(".py") {
        "py_compile"
    } else if path.ends_with(".java") {
        "mvn compile"
    } else {
        "check"
    };

    if success {
        result.content.push_str(&format!("\n[{checker_name}] OK"));
        return;
    }

    // Extract error/warning lines
    let relevant: Vec<&str> = stderr
        .lines()
        .filter(|l| l.contains("error") || l.contains("warning") || l.starts_with("  "))
        .take(30)
        .collect();

    if relevant.is_empty() {
        if is_new_file {
            // New file with no parseable error details — surface as a
            // warning so the agent doesn't shrug it off, but don't flip
            // success (the file was written as requested).
            result.content.push_str(&format!(
                "\n[{checker_name}] WARNING: errors reported for newly-created {path} but no details were captured — fix before relying on this file"
            ));
        } else {
            result
                .content
                .push_str(&format!("\n[{checker_name}] failed (no details captured)"));
            result.success = false;
        }
        return;
    }

    if is_new_file {
        // For brand-new files we have no baseline — the model is
        // creating a helper script whose imports may not yet resolve in
        // the project's environment, etc. Surface as a WARNING so the
        // agent treats it as a real signal, but do NOT flip success —
        // the file was written as requested.
        result.content.push_str(&format!(
            "\n[{checker_name}] WARNING: {} line(s) reported for newly-created {path} — fix before relying on this file\n",
            relevant.len()
        ));
        result.content.push_str(&relevant.join("\n"));
        return;
    }

    result.content.push_str(&format!("\n[{checker_name}]\n"));
    result.content.push_str(&relevant.join("\n"));

    // Parse error locations and include source context
    let locations = extract_error_locations(&stderr);
    if !locations.is_empty() {
        result.content.push_str("\n[source context]\n");
        for (file, line_num) in &locations {
            if let Some(ctx) = read_source_context(file, *line_num, &project_root) {
                result.content.push_str(&ctx);
            }
        }
    }

    // Interpret common errors into actionable hints
    let joined = relevant.join("\n");
    let mut hints = Vec::new();
    if joined.contains("expected") && joined.contains("argument") && joined.contains("found") {
        hints.push("ACTION: Function signature changed but call sites not updated. Use file(action='search', query='function_name') to find ALL callers and update them.");
    }
    if joined.contains("cannot find") {
        hints.push("ACTION: A symbol was renamed/removed but references remain. Search for the old name and update.");
    }
    if joined.contains("unclosed delimiter") || joined.contains("unexpected closing") {
        hints.push(match config.tools.edit_mode {
            crate::config::EditMode::Smart => "ACTION: Broken syntax (missing/extra bracket). Use edit_file for the structural repair; replace is unreliable for structural fixes.",
            crate::config::EditMode::Fast => "ACTION: Broken syntax (missing/extra bracket). Revert the offending rev and land the structural repair as a tighter replace_range over the enclosing block.",
        });
    }
    if joined.contains("mismatched types") {
        hints.push("ACTION: Type mismatch. Check the function signature and update the caller to pass the correct type.");
    }
    if !hints.is_empty() {
        result.content.push_str("\n[action needed]\n");
        for hint in &hints {
            result.content.push_str(hint);
            result.content.push('\n');
        }
    }

    result.success = false;
}

/// Ask the outer model whether to accept an edit that increased LSP errors.
/// Returns `true` for YES (keep changes), `false` for NO (revert).
async fn ask_accept_lsp_regression(router: &ModelRouter, path: &str, error_report: &str) -> bool {
    let prompt = format!(
        "edit_file wrote changes to {path}.\n\
         LSP diagnostics show new errors:\n\
         {error_report}\n\
         Are these errors expected (e.g. a callee/dependency updated in a later step), \
         or are they mistakes in the edit?\n\
         Answer YES to keep the changes, NO to revert them."
    );
    let request = ChatRequest {
        messages: vec![
            Message::system(
                "You are reviewing an edit. Answer YES or NO only. \
                 YES means the errors are expected and the edit should be kept. \
                 NO means the errors are unexpected and the edit should be reverted.",
            ),
            Message::user(&prompt),
        ],
        tools: None,
        tool_choice: None,
    };
    match router.chat(ModelRole::Fast, &request).await {
        Ok(response) => {
            let text = response
                .choices
                .first()
                .and_then(|c| c.message.content.as_deref())
                .unwrap_or("");
            let first_line = text.lines().next().unwrap_or("").trim().to_uppercase();
            first_line.contains("YES")
        }
        Err(_) => false, // On error, default to revert (safe choice)
    }
}
