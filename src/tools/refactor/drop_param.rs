//! `drop_param`: remove a parameter from a function signature and the
//! corresponding argument from every callsite.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use anyhow::{Context, Result};
use serde_json::Value;

use crate::config::Config;
use crate::llm::ModelRouter;
use crate::logging::SessionLog;
use crate::lsp::LspClient;
use crate::tools::ToolResult;
use crate::tools::args;

use crate::tools::fast::RevisionStore;

use super::model_edit::{apply_rewrite, ask_rewrite_validated};
use super::sites::{
    StagedEdit, commit_staged, ensure_ready, extract_window, find_callsites,
    resolve_function_location,
};
use super::validation::{ArgSchema, validate};

const DROP_PARAM_EXAMPLE: &str =
    "change_signature(action=\"drop_param\", path=\"src/lib.rs\", name=\"assemble\", param=\"x\")";

const DROP_PARAM_SCHEMA: ArgSchema<'static> = ArgSchema {
    action: "drop_param",
    required_strings: &["path", "name", "param"],
    optional_strings: &[],
    optional_ints: &["line"],
    example: DROP_PARAM_EXAMPLE,
};

pub async fn execute(
    args: &Value,
    config: &Config,
    router: &ModelRouter,
    lsp: Option<&LspClient>,
    log: Option<&SessionLog>,
    revisions: Option<&RevisionStore>,
    cancelled: Option<&AtomicBool>,
) -> Result<ToolResult> {
    if let Err(e) = validate(args, &DROP_PARAM_SCHEMA) {
        return Ok(ToolResult::err(e));
    }
    let path_str = args::require_str(args, "path").expect("validated");
    let function_name = args::require_str(args, "name").expect("validated");
    let param = args::require_str(args, "param").expect("validated");
    let line_hint = args::opt_u64(args, "line").expect("validated");

    let Some(lsp) = lsp else {
        return Ok(ToolResult::err(
            "drop_param requires LSP support (no LSP client available for this project)".into(),
        ));
    };
    if let Err(e) = ensure_ready(lsp, Duration::from_secs(60)).await {
        return Ok(ToolResult::err(format!(
            "LSP not ready in time: {e}. Try again in a moment, or call code(diagnostics) first to warm it up."
        )));
    }

    let abs_path = config.project_root.join(path_str);
    let line_hint_0 = line_hint.map(|n| (n.saturating_sub(1)) as u32);

    let resolved = match resolve_function_location(lsp, &abs_path, function_name, line_hint_0).await
    {
        Ok(r) => r,
        Err(e) => {
            return Ok(ToolResult::err(format!("✗ drop_param: {e}")));
        }
    };
    let line_0 = resolved.line_0;
    let column_0 = resolved.column_0;
    let resolved_line_1 = line_0 + 1;

    let original = std::fs::read_to_string(&abs_path)
        .with_context(|| format!("read function file {path_str}"))?;

    // 1. Update the signature. Snippet starts at the function definition
    // so the model can't get confused about which construct to edit.
    let sig_window = extract_window(&original, line_0, 12);
    let sig_instruction = format!(
        "Remove the parameter `{param}` from the function whose signature starts at the FIRST line of the snippet below. \
         Drop the parameter declaration AND its trailing comma if any (or the leading comma if it was the last parameter). \
         Do not touch anything outside the parameter list."
    );
    if let Some(log) = log {
        log.tool_debug(
            "change_signature",
            &format!(
                "drop_param entry path={path_str} name={function_name} line_hint={line_hint:?} \
                 resolved=line_0={line_0} column_0={column_0} param={param:?}"
            ),
        );
    }
    let sig_rewrite = match ask_rewrite_validated(
        router,
        log,
        &format!("signature:{path_str}:{resolved_line_1}"),
        &sig_instruction,
        &sig_window.text,
        cancelled,
        |r| apply_rewrite(&original, r, line_0).map(|_| ()),
    )
    .await
    {
        Ok(Some(r)) => r,
        Ok(None) => unreachable!(),
        Err(e) => {
            return Ok(ToolResult::err(format!("signature rewrite failed: {e}")));
        }
    };
    let updated =
        apply_rewrite(&original, &sig_rewrite, line_0).expect("validator verified apply succeeds");

    let mut staged: BTreeMap<PathBuf, StagedEdit> = BTreeMap::new();
    staged.insert(
        abs_path.clone(),
        StagedEdit {
            original: original.clone(),
            updated,
        },
    );

    // 2. Find callsites and rewrite each one.
    let callsites = find_callsites(lsp, config, &abs_path, line_0, column_0)
        .await
        .context("find_callsites failed")?;

    if callsites.is_empty() {
        commit_staged(&staged, config, revisions, "change_signature.drop_param")?;
        return Ok(ToolResult::ok(format!(
            "✓ drop_param: signature updated. No callsites found via LSP.\n\
             - Edited: {} (signature)",
            path_str
        )));
    }

    let mut report = Vec::new();
    let mut callsite_failures = Vec::new();
    let mut side_effect_warnings = Vec::new();
    let old_signature = sig_rewrite.old.clone();
    let new_signature = sig_rewrite.new.clone();

    for site in &callsites {
        let rel = display_path(&site.path, config);
        let instruction = format!(
            "A function's signature was modified: parameter `{param}` was removed. \
             Update the call expression at the FIRST line of the snippet below to match the new \
             signature: drop the corresponding argument. Change ONLY that one call. \
             If the dropped argument is a function call or otherwise has visible side effects, \
             emit SKIP with the reason 'side-effecting expression in dropped slot' so the human can review.\n\
             \n\
             Old signature:\n{old_signature}\n\
             \n\
             New signature:\n{new_signature}",
            param = param,
        );

        let (original, src) = match staged.get(&site.path) {
            Some(edit) => (edit.original.clone(), edit.updated.clone()),
            None => match std::fs::read_to_string(&site.path) {
                Ok(s) => (s.clone(), s),
                Err(e) => {
                    callsite_failures.push(format!(
                        "{}:{}: read failed: {}",
                        rel,
                        site.line + 1,
                        e
                    ));
                    continue;
                }
            },
        };
        // Validator-aware retries: if the model produces an OLD/NEW that
        // can't be applied at the LSP-resolved anchor (e.g. paraphrased
        // input), retry with a fresh inference pass.
        let rewrite = match ask_rewrite_validated(
            router,
            log,
            &format!("callsite:{rel}:{}", site.line + 1),
            &instruction,
            &site.window,
            cancelled,
            |r| apply_rewrite(&src, r, site.line).map(|_| ()),
        )
        .await
        {
            Ok(Some(r)) => r,
            Ok(None) => unreachable!(),
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("side-effecting") {
                    side_effect_warnings.push(format!("{}:{}: {}", rel, site.line + 1, msg));
                } else {
                    callsite_failures.push(format!("{}:{}: {}", rel, site.line + 1, msg));
                }
                continue;
            }
        };
        // Validator already verified apply succeeds; re-run to produce
        // the final updated source.
        match apply_rewrite(&src, &rewrite, site.line) {
            Ok(updated) => {
                staged.insert(site.path.clone(), StagedEdit { original, updated });
                report.push(format!("  • {}:{} updated", rel, site.line + 1));
            }
            Err(e) => {
                callsite_failures.push(format!("{}:{}: {}", rel, site.line + 1, e));
            }
        }
    }

    commit_staged(&staged, config, revisions, "change_signature.drop_param")?;

    let total = callsites.len();
    let succeeded = report.len();
    let mut out = String::new();
    if callsite_failures.is_empty() && side_effect_warnings.is_empty() {
        out.push_str(&format!(
            "✓ COMPLETE — definition and all {total} callsites are now consistent.\n",
        ));
    } else if succeeded == 0 {
        out.push_str(&format!(
            "✗ drop_param FAILED: signature rewritten on disk, but 0 of {total} callsite(s) \
             could be updated. The project will NOT compile until callsites are fixed. \
             Either retry OR roll back via file(action=\"revert\", to_round=<this round>) \
             to discard the signature change.\n",
        ));
    } else {
        out.push_str(&format!(
            "✗ drop_param PARTIAL: signature rewritten, only {succeeded}/{total} callsite(s) updated. \
             Project will NOT compile until remaining callsite(s) are fixed manually \
             OR you roll back via file(action=\"revert\", to_round=<this round>).\n",
        ));
    }
    if !side_effect_warnings.is_empty() {
        out.push_str(&format!(
            "\n{} site(s) skipped due to side-effecting expressions in the dropped slot — \
             review these manually:\n",
            side_effect_warnings.len(),
        ));
        for w in &side_effect_warnings {
            out.push_str(&format!("  • {w}\n"));
        }
    }
    if !callsite_failures.is_empty() {
        out.push_str(&format!(
            "\n{} site(s) failed to rewrite — review these manually:\n",
            callsite_failures.len(),
        ));
        for f in &callsite_failures {
            out.push_str(&format!("  • {f}\n"));
        }
    }
    if !report.is_empty() {
        out.push_str("\nEdits:\n");
        for line in &report {
            out.push_str(line);
            out.push('\n');
        }
    }
    out.push_str(
        "\nNext: run code(diagnostics) or your build to confirm the project still compiles.",
    );

    Ok(
        if callsite_failures.is_empty() && side_effect_warnings.is_empty() {
            ToolResult::ok(out)
        } else {
            ToolResult::err(out)
        },
    )
}

fn display_path(p: &std::path::Path, config: &Config) -> String {
    p.strip_prefix(&config.project_root)
        .unwrap_or(p)
        .display()
        .to_string()
}
