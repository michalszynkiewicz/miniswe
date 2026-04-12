//! `replace_range <path> <start> <end> <content>`
//!
//! Replaces lines `[start..=end]` (1-based, inclusive) with `content`.
//! Empty `content` deletes the range.
//!
//! No OLD-block confirmation: wrong-line edits surface as broken AST or
//! new LSP errors in the next feedback block, and the model reverts.

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
        "replace_range — validate range (1-based, start<=end, within file), \
         apply edit, record revision, build feedback"
    )
}
