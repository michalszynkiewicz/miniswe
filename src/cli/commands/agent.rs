//! Shared agent-loop helpers used by both the interactive REPL
//! (`repl.rs`) and the single-shot run loop (`run.rs`).
//!
//! Previously each of those files inlined its own copy of these helpers.
//! The copies drifted — at the time of extraction `summarize_args`
//! disagreed between the two on several cases (file/read line ranges,
//! fast-mode tool formatting). This module is the single source of truth.

pub mod display;
pub mod hints;
pub mod loop_detector;
pub mod permissions;
pub mod subagent;
