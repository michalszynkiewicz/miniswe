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
//! Skeleton only — all public entry points are `todo!()`. Not yet wired
//! into `execute_tool`. Tracked by task #82.

mod dispatch;
mod feedback;
mod revisions;

mod check;
mod insert_at;
mod replace_range;
mod revert;

pub use dispatch::execute_fast_tool;
pub use revisions::RevisionStore;
