//! Tool system for miniswe.
//!
//! Top-level tools for grouped file/code/web operations plus focused editors.
//! action-based dispatch for grouped tools. All file access is jailed
//! to the project root. Destructive actions require user permission.
//! After file edits, the index is incrementally updated.
//!
//! The module is split into focused submodules:
//! - [`dispatch`]: the `execute_tool` entry point and per-group dispatchers
//! - [`edit_orchestration`]: `execute_edit_file_tool` plus the post-edit
//!   baseline / reindex / auto_check / LSP-regression-confirmation pipeline
//! - [`cargo_check`]: compiler/checker subprocess helpers
//! - [`code_intel`]: LSP tools and pull-based project context tools
//!
//! [`ToolResult`] is the shared return type used across all submodules.

pub(crate) mod args;
mod cargo_check;
mod code_intel;
mod delete_file;
mod dispatch;
pub mod edit_file;
mod edit_orchestration;
pub mod fast;
pub mod plan;
mod read_file;
mod refactor;
mod search;
pub(crate) mod shell;
pub mod snapshots;
mod task_update;
mod web;
mod write_file;

pub mod definitions;
pub mod permissions;
pub use cargo_check::run_check_with_timeout;
pub use definitions::{fast_mode_tool_definitions, tool_definitions};
pub use dispatch::execute_tool;
pub use edit_orchestration::execute_edit_file_tool;
pub(crate) use edit_orchestration::reindex_project_incremental;
pub use fast::{RevisionStore, execute_fast_tool};
pub(crate) use fast::{RewindCandidate, find_rewind_candidate};
pub use permissions::PermissionManager;
pub use refactor::execute_refactor_tool;

/// Result of executing a tool.
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub content: String,
    pub success: bool,
    /// Structured facts behind certain failures, alongside `content` (not
    /// instead of it). `content` is always the complete, ready-to-use
    /// message for miniswe's own agent loop — some of it is probe-validated
    /// wording (see `refactor::add_param`) that must not be reworded.
    /// `detail` lets a *different* consumer (e.g. an MCP server exposing a
    /// different tool surface) render its own next-step guidance from the
    /// same facts instead of reusing miniswe-specific tool-name references
    /// (`file(...)`, `refactor(...)`) that don't apply to it.
    pub detail: Option<ToolDetail>,
}

/// See [`ToolResult::detail`].
#[derive(Debug, Clone)]
pub enum ToolDetail {
    /// An LSP-backed action was requested but no LSP client is available.
    LspUnavailable,
    /// A `refactor`-family action's arguments failed schema validation.
    InvalidArgs {
        action: &'static str,
        missing: Vec<String>,
        bad_type: Vec<String>,
        unknown: Vec<String>,
    },
    /// add_param/drop_param rewrote the signature but not every callsite
    /// could be updated.
    PartialSignatureChange {
        action: &'static str,
        total: usize,
        succeeded: usize,
        /// Per-callsite failure descriptions (`path:line: reason`).
        callsite_failures: Vec<String>,
        /// Per-callsite success descriptions, already consumer-agnostic
        /// (e.g. `"  • src/foo.rs:10 now passes \`None\`"`).
        callsite_report: Vec<String>,
    },
}

impl ToolResult {
    pub fn ok(content: String) -> Self {
        Self {
            content,
            success: true,
            detail: None,
        }
    }

    pub fn err(content: String) -> Self {
        Self {
            content,
            success: false,
            detail: None,
        }
    }

    /// Like [`Self::err`], plus structured facts for other consumers — see
    /// [`ToolDetail`].
    pub fn err_with_detail(content: String, detail: ToolDetail) -> Self {
        Self {
            content,
            success: false,
            detail: Some(detail),
        }
    }
}
