//! `revert <path> <rev>`
//!
//! Restores `path` to a named prior revision. Always explicit — no
//! "undo last" shortcut, no step count. The model scans the revision
//! table (attached to every feedback block) and picks the row to restore.
//!
//! Linear history: reverting to `rev_N` truncates `rev_{N+1}..` and the
//! next edit becomes `rev_{N+1}`. No branching.

use anyhow::Result;
use serde_json::Value;

use crate::config::Config;
use crate::lsp::LspClient;

use super::super::ToolResult;
use super::super::permissions::PermissionManager;
use super::revisions::RevisionStore;

pub async fn execute(
    _args: &Value,
    _config: &Config,
    _perms: &PermissionManager,
    _lsp: Option<&LspClient>,
    _revisions: &RevisionStore,
) -> Result<ToolResult> {
    todo!(
        "revert — load rev content from RevisionStore, write to disk, \
         truncate history past that rev, build feedback reflecting \
         post-revert state"
    )
}
