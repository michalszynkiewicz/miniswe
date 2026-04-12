//! Per-edit feedback builder: AST status + per-file LSP diagnostics +
//! project-wide error stat + revision table.
//!
//! Every fast-mode write-ish tool appends this block to its `ToolResult`
//! content so the model always sees where it is and what just broke.
//! Format is optimized for tiny-model pattern-matching; see
//! `docs/fast-mode-design.md` for the design rationale.

use crate::config::Config;
use crate::lsp::LspClient;

use super::revisions::RevisionStore;

/// Built feedback block, ready to append to a `ToolResult.content`.
pub struct EditFeedback {
    /// Rendered multi-line feedback text.
    pub text: String,
    /// AST parsed cleanly for this file.
    pub ast_ok: bool,
    /// LSP error count in this file right now.
    pub file_errors: usize,
    /// LSP error count across the project right now.
    pub project_errors: usize,
}

/// Build per-edit feedback for `rel_path` after a successful write.
/// Runs AST parse, polls LSP for current diagnostics, and renders the
/// revision table. All IO is best-effort: missing pieces degrade
/// gracefully (e.g. no LSP → AST-only feedback, still useful).
pub async fn build_feedback(
    _rel_path: &str,
    _new_content: &str,
    _config: &Config,
    _lsp: Option<&LspClient>,
    _revisions: &RevisionStore,
    _project_baseline_errors: usize,
) -> EditFeedback {
    todo!(
        "build_feedback — AST parse + per-file LSP diagnostics + project \
         error stat with (+delta) + revision table for this file"
    )
}

/// Run a tree-sitter parse on `content` for the file's detected language.
/// Returns `Ok(())` if the file parses, `Err(msg)` with a specific parse
/// error otherwise. Fast-mode's AST gate — independent of LSP.
pub fn ast_check(_rel_path: &str, _content: &str) -> Result<(), String> {
    todo!("ast_check — tree-sitter parse, return first parse error if any")
}

/// Poll LSP for diagnostics for a single file with a short timeout.
/// Returns 0 if LSP is absent or not ready. Kept local so fast-mode
/// doesn't reach into edit_orchestration internals.
pub async fn file_error_count(
    _abs_path: &std::path::Path,
    _config: &Config,
    _lsp: Option<&LspClient>,
) -> usize {
    todo!("file_error_count — LSP diagnostics poll for a single file")
}

/// Snapshot total LSP error count across the project. Used for the
/// `project_errors (+delta)` line in feedback and as a baseline.
pub async fn project_error_count(_lsp: Option<&LspClient>) -> usize {
    todo!("project_error_count — sum LSP diagnostics across project")
}
