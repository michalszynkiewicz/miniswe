//! Fast-mode editing tools.
//!
//! An alternative to the inner-model `edit_file` planner. Fast mode
//! exposes primitive range- and position-based edits directly to the outer
//! agent, with per-edit AST + LSP feedback and an explicit revision table
//! for rollback. No sub-delegations, no hidden turns — every cognitive
//! step happens in the outer model's context.
//!
//! See `docs/fast-mode-design.md` for the full rationale.
//!
//! ## Tool surface
//!
//! - [`write_file`](super::write_file) (reused) — create or overwrite
//! - [`replace_range`] — replace lines `[start..=end]`
//! - [`insert_at`] — insert after a line (0 = top of file)
//! - [`revert`] — restore a named prior revision
//! - [`check`] — explicit cargo check / project-wide diagnostics
//!
//! No OLD-block confirmation on any edit. Mistakes surface as broken AST
//! or new LSP errors in the very next tool result; the model reverts.
//!
//! ## Status
//!
//! Fully implemented. Wired into the run loop when `tools.edit_mode = "fast"`.
//! `edit_file` is dropped from the tool list in fast mode; the four
//! primitives above are offered instead.

mod ast;
mod dispatch;
mod feedback;
mod lines;
mod revisions;

mod check;
mod insert_at;
mod replace_range;
mod revert;

pub use dispatch::execute_fast_tool;
pub use feedback::project_error_count;
pub use revisions::RevisionStore;
