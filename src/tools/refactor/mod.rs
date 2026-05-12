//! Signature-changing refactors: `add_param`, `drop_param`, and a generic
//! LSP-driven `rename`.
//!
//! See [`dispatch`] for entry points, [`sites`] for callsite discovery,
//! and [`model_edit`] for the per-snippet model-call helper.

mod add_param;
mod dispatch;
mod drop_param;
mod model_edit;
mod rename;
mod sites;
mod validation;

pub use dispatch::execute_refactor_tool;
