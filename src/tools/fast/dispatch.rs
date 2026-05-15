//! Fast-mode tool dispatch.
//!
//! Entry point for fast-mode edit tools. Mirrors [`super::super::dispatch`]
//! but for the smaller, revision-backed fast-mode surface. Not yet wired
//! into `execute_tool` — callers that want fast mode will invoke this
//! directly once the session REPL gains a fast-mode branch.

use anyhow::{Result, bail};
use serde_json::Value;

use crate::config::Config;
use crate::lsp::LspClient;

use super::super::ToolResult;
use super::super::permissions::PermissionManager;
use super::revisions::RevisionStore;
use super::{check, insert_at, replace_range, revert, show_rev};

/// Dispatch a fast-mode tool by name. Valid names:
/// `"replace_range"`, `"insert_at"`, `"revert"`, `"show_rev"`, `"check"`.
/// `write_file` is not fast-mode-specific and is handled by the main
/// dispatcher.
///
/// `project_baseline_errors` is captured once at session start and
/// threaded through so each feedback block can report the delta.
pub async fn execute_fast_tool(
    name: &str,
    args: &Value,
    config: &Config,
    perms: &PermissionManager,
    lsp: Option<&LspClient>,
    revisions: &RevisionStore,
    project_baseline_errors: usize,
) -> Result<ToolResult> {
    let result = match name {
        "replace_range" => {
            replace_range::execute(args, config, perms, lsp, revisions, project_baseline_errors)
                .await
        }
        "insert_at" => {
            insert_at::execute(args, config, perms, lsp, revisions, project_baseline_errors).await
        }
        "revert" => {
            revert::execute(args, config, perms, lsp, revisions, project_baseline_errors).await
        }
        "show_rev" => return show_rev::execute(args, perms, revisions).await,
        "check" => return check::execute(args, config).await,
        _ => bail!("Unknown fast-mode tool: {name}"),
    };

    // Keep the tree-sitter symbol index in sync. The fast tools don't go
    // through `finalize_file_edit` (that's the smart-mode path), so without
    // this the repo-map / symbol lookups serve pre-edit structure for code
    // the model just changed. <1ms per file; best-effort.
    if result.as_ref().is_ok_and(|r| r.success)
        && let Some(path) = args.get("path").and_then(|p| p.as_str())
    {
        super::super::edit_orchestration::reindex_changed_file(path, config);
    }

    result
}
