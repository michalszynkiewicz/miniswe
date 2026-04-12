//! `insert_at <path> <after_line> <content>`
//!
//! Inserts `content` after line `after_line` (1-based).
//! `after_line = 0` inserts at the top of the file.
//!
//! Kept distinct from `replace_range` so tiny models don't have to encode
//! insertion as `replace_range N N "<echoed line N>\nnew"` or as
//! `start > end` slice trickery. Both are recurring cognitive traps.

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
        "insert_at — validate after_line (0..=line_count), apply edit, \
         record revision, build feedback"
    )
}
