//! LSP-backed code intelligence tools and pull-based project context tools.
//!
//! These back the `code` tool group's `goto_definition`, `find_references`,
//! `diagnostics`, `repo_map`, `project_info`, and `architecture_notes`
//! actions. Kept separate from the edit orchestration path so the fast-mode
//! tool surface can reuse them without pulling in the inner-model editor.

use anyhow::Result;

use crate::config::Config;
use crate::context::compress;
use crate::knowledge::graph::DependencyGraph;
use crate::knowledge::{ProjectIndex, repo_map};
use crate::lsp::LspClient;

use super::ToolResult;
use super::cargo_check::read_source_context;

/// Gather project-wide diagnostics from LSP.
pub(super) async fn lsp_project_diagnostics(
    config: &Config,
    lsp: &LspClient,
) -> Option<ToolResult> {
    // Trigger a full check by notifying on a project config file (forces re-analysis)
    let config_files = [
        "Cargo.toml",
        "tsconfig.json",
        "package.json",
        "go.mod",
        "pyproject.toml",
        "pom.xml",
        "build.gradle",
    ];
    for name in config_files {
        let path = config.project_root.join(name);
        if path.exists() {
            let _ = lsp.notify_file_changed(&path);
            break;
        }
    }

    let timeout = std::time::Duration::from_millis(config.lsp.diagnostic_timeout_ms);
    tokio::time::sleep(timeout).await;

    // Collect all diagnostics
    let mut output = String::new();
    let mut error_count = 0;
    let mut warning_count = 0;

    for entry in lsp.diagnostics_snapshot() {
        for diag in entry.1 {
            let severity = match diag.severity {
                Some(lsp_types::DiagnosticSeverity::ERROR) => {
                    error_count += 1;
                    "error"
                }
                Some(lsp_types::DiagnosticSeverity::WARNING) => {
                    warning_count += 1;
                    "warning"
                }
                _ => continue,
            };
            let line = diag.range.start.line + 1;
            let col = diag.range.start.character + 1;
            // Strip file:// prefix for readability
            let path = entry.0.strip_prefix("file://").unwrap_or(&entry.0);
            output.push_str(&format!(
                "{path}:{line}:{col}: {severity}: {}\n",
                diag.message
            ));
        }
    }

    if output.is_empty() {
        Some(ToolResult::ok("[lsp] No errors or warnings".into()))
    } else {
        // Cap output to first 10 errors — the model can't fix 290 at once
        let capped_output: String = output.lines().take(10).collect::<Vec<_>>().join("\n");
        let shown = capped_output.lines().count();
        let total = output.lines().count();

        let mut summary = format!("[lsp] {error_count} error(s), {warning_count} warning(s)");
        if total > shown {
            summary.push_str(&format!(" (showing first {shown})"));
        }
        summary.push('\n');
        summary.push_str(&capped_output);

        let success = error_count == 0;
        Some(ToolResult {
            content: summary,
            success,
        })
    }
}

/// goto_definition tool handler.
pub(super) async fn lsp_goto_definition(
    path: &str,
    line: u32,
    column: u32,
    config: &Config,
    lsp: Option<&LspClient>,
) -> Result<ToolResult> {
    let lsp = match lsp {
        Some(l) if l.is_ready() && !l.has_crashed() => l,
        _ => {
            return Ok(ToolResult::err(
                "LSP not available. Use file(action='search') instead.".into(),
            ));
        }
    };

    let abs_path = config.project_root.join(path);
    // Ensure file is open in LSP
    let _ = lsp.notify_file_changed(&abs_path);

    match lsp.goto_definition(&abs_path, line, column).await {
        Ok(locations) if locations.is_empty() => Ok(ToolResult::ok(
            "No definition found at this location.".into(),
        )),
        Ok(locations) => {
            let mut output = String::new();
            for loc in &locations {
                let def_path = crate::lsp::client::uri_to_path(&loc.uri);
                let def_line = loc.range.start.line + 1;
                let def_col = loc.range.start.character + 1;

                if let Some(ref p) = def_path {
                    // Make path relative to project root
                    let rel = p.strip_prefix(&config.project_root).unwrap_or(p);
                    output.push_str(&format!(
                        "Definition: {}:{}:{}\n",
                        rel.display(),
                        def_line,
                        def_col
                    ));

                    // Include source context
                    if let Some(ctx) = read_source_context(
                        &rel.to_string_lossy(),
                        def_line as usize,
                        &config.project_root,
                    ) {
                        output.push_str(&ctx);
                    }
                } else {
                    output.push_str(&format!(
                        "Definition: {}:{}:{}\n",
                        loc.uri.as_str(),
                        def_line,
                        def_col
                    ));
                }
            }
            Ok(ToolResult::ok(output))
        }
        Err(e) => Ok(ToolResult::err(format!("LSP error: {e}"))),
    }
}

/// find_references tool handler.
pub(super) async fn lsp_find_references(
    path: &str,
    line: u32,
    column: u32,
    config: &Config,
    lsp: Option<&LspClient>,
) -> Result<ToolResult> {
    let lsp = match lsp {
        Some(l) if l.is_ready() && !l.has_crashed() => l,
        _ => {
            return Ok(ToolResult::err(
                "LSP not available. Use file(action='search') instead.".into(),
            ));
        }
    };

    let abs_path = config.project_root.join(path);
    let _ = lsp.notify_file_changed(&abs_path);

    match lsp.find_references(&abs_path, line, column).await {
        Ok(locations) if locations.is_empty() => Ok(ToolResult::ok("No references found.".into())),
        Ok(locations) => {
            let mut output = format!("{} reference(s) found:\n", locations.len());
            for loc in &locations {
                let ref_path = crate::lsp::client::uri_to_path(&loc.uri);
                let ref_line = loc.range.start.line + 1;

                if let Some(ref p) = ref_path {
                    let rel = p.strip_prefix(&config.project_root).unwrap_or(p);
                    // Read the actual line for context
                    let abs = config.project_root.join(rel);
                    let line_content = std::fs::read_to_string(&abs)
                        .ok()
                        .and_then(|content| {
                            content
                                .lines()
                                .nth(ref_line as usize - 1)
                                .map(|l| l.trim().to_string())
                        })
                        .unwrap_or_default();
                    output.push_str(&format!(
                        "  {}:{}: {}\n",
                        rel.display(),
                        ref_line,
                        line_content
                    ));
                } else {
                    output.push_str(&format!("  {}:{}\n", loc.uri.as_str(), ref_line));
                }
            }
            Ok(ToolResult::ok(output))
        }
        Err(e) => Ok(ToolResult::err(format!("LSP error: {e}"))),
    }
}

// ── Context tools (pull-based) ────────────────────────────────────────

/// Return the PageRank-scored repo map, optionally filtered by keywords.
pub(super) fn context_tool_repo_map(keywords_str: &str, config: &Config) -> Result<ToolResult> {
    let miniswe_dir = config.miniswe_dir();
    let index = match ProjectIndex::load(&miniswe_dir) {
        Ok(idx) => idx,
        Err(_) => {
            return Ok(ToolResult::err(
                "No project index. Run `miniswe init` first.".into(),
            ));
        }
    };
    let graph = DependencyGraph::load(&miniswe_dir).unwrap_or_default();

    let keywords: Vec<&str> = if keywords_str.is_empty() {
        vec![]
    } else {
        keywords_str.split_whitespace().collect()
    };

    let map = repo_map::render(
        &index,
        &graph,
        config.context.repo_map_budget,
        &keywords,
        &config.project_root,
    );

    if map.is_empty() {
        Ok(ToolResult::ok(
            "Repo map is empty (no indexed symbols).".into(),
        ))
    } else {
        Ok(ToolResult::ok(format!(
            "Repo map ({} files indexed, {} symbols):\n{map}",
            index.total_files, index.total_symbols
        )))
    }
}

/// Return project profile, guide, and lessons.
pub(super) fn context_tool_project_info(config: &Config) -> Result<ToolResult> {
    let mut output = String::new();

    let profile_path = config.miniswe_path("profile.md");
    if let Ok(content) = std::fs::read_to_string(&profile_path) {
        output.push_str("[PROFILE]\n");
        output.push_str(&compress::compress_profile(&content));
        output.push('\n');
    }

    let guide_path = config.miniswe_path("guide.md");
    if let Ok(content) = std::fs::read_to_string(&guide_path)
        && (!content.contains("<!-- Add project-specific instructions")
            || content.lines().count() > 5)
    {
        output.push_str("\n[GUIDE]\n");
        output.push_str(&content);
        output.push('\n');
    }

    let lessons_path = config.miniswe_path("lessons.md");
    if let Ok(content) = std::fs::read_to_string(&lessons_path)
        && (!content.contains("<!-- Accumulated tips") || content.lines().count() > 5)
    {
        output.push_str("\n[LESSONS]\n");
        output.push_str(&content);
        output.push('\n');
    }

    if output.is_empty() {
        Ok(ToolResult::ok(
            "No project info available. Run `miniswe init` to generate.".into(),
        ))
    } else {
        Ok(ToolResult::ok(output))
    }
}

/// Return architecture notes from .ai/README.md.
pub(super) fn context_tool_architecture_notes(config: &Config) -> Result<ToolResult> {
    let path = config.project_root.join(".ai").join("README.md");
    match std::fs::read_to_string(&path) {
        Ok(content) => Ok(ToolResult::ok(crate::truncate_chars(
            &content,
            config.tool_output_budget_chars(),
        ))),
        Err(_) => Ok(ToolResult::ok(
            "No architecture notes found (.ai/README.md does not exist).".into(),
        )),
    }
}
