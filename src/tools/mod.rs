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

mod cargo_check;
mod code_intel;
mod delete_file;
mod dispatch;
pub mod edit_file;
mod edit_orchestration;
pub mod fast;
pub mod plan;
mod read_file;
mod search;
pub(crate) mod shell;
pub mod snapshots;
mod task_update;
mod web;
mod write_file;

pub mod definitions;
pub mod permissions;
pub use cargo_check::run_check_with_timeout;
pub use definitions::tool_definitions;
pub use dispatch::execute_tool;
pub use edit_orchestration::execute_edit_file_tool;
pub use permissions::PermissionManager;

/// Result of executing a tool.
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub content: String,
    pub success: bool,
}

impl ToolResult {
    pub fn ok(content: String) -> Self {
        Self {
            content,
            success: true,
        }
    }

    pub fn err(content: String) -> Self {
        Self {
            content,
            success: false,
        }
    }
}
