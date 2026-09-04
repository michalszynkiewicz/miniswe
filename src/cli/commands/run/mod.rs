//! Agent execution — the main LLM loop.
//!
//! Implements the core agent loop:
//! 1. Assemble context for the turn
//! 2. Call the LLM with tools
//! 3. Parse tool calls from the response
//! 4. Execute tools
//! 5. Apply observation masking (compress old tool results)
//! 6. Feed results back and repeat

mod main_loop;
mod skill_step;
mod support;

pub use main_loop::run;

use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::Mutex;

use anyhow::Result;

use crate::cli::commands::agent::debugger;
use crate::cli::commands::agent::display::summarize_args;
use crate::cli::commands::agent::hints::{
    PLAN_CHECKPOINT_AFTER_EDITS, PLAN_CHECKPOINT_WARNING, PLAN_PROGRESS_NUDGE,
    PREMATURE_EXIT_NUDGE, REPEATED_READ_ESCALATION, REPEATED_READ_NUDGE, cycle_loop_hint,
    is_file_write, is_prunable_refactor_failure, loop_detected_hint, truncated_tool_call_hint,
    visible_tool_defs,
};
use crate::cli::commands::agent::loop_detector::{
    cycle_period, is_mutating_call, key_is_file_edit, key_is_mutating, loop_call_key_tagged,
};
use crate::cli::commands::agent::prune_reads::prune_repeated_reads;
use crate::cli::commands::agent::spiral;
use crate::cli::commands::agent::stuck_check;
use crate::cli::commands::agent::validation;
use crate::config::{Config, EditMode, ModelRole};
use crate::context;
use crate::llm::{
    ChatRequest, Message, ModelRouter, TRUNCATED_CALL_ABORT_AFTER, is_context_exceeded_error,
    is_context_truncated_response, is_tool_call_args_cap_error, is_truncated_tool_call_error,
    sanitize_truncated_tool_calls, scrub_unparseable_tool_calls, truncated_args_info,
    truncated_args_tool_result,
};
use crate::logging::SessionLog;
use crate::lsp::LspClient;
use crate::mcp::{McpConfig, McpRegistry};
use crate::runtime::{
    LlmWorkerEvent, LlmWorkerHandle, ShellControl, ShellWorkerEvent, ToolWorkerPool,
};
use crate::tools;
use crate::tools::permissions::{Action, PermissionManager};
use crate::tui;

use skill_step::*;
use support::*;
