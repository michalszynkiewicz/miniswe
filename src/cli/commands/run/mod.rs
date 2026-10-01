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
mod support;
mod ui;

pub use main_loop::run;

use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::Mutex;

use anyhow::Result;

use crate::cli::commands::agent::turn;
use crate::cli::commands::agent::turn_state;
use crate::config::{Config, EditMode, ModelRole};
use crate::context;
use crate::llm::{Message, ModelRouter};
use crate::logging::SessionLog;
use crate::lsp::LspClient;
use crate::mcp::{McpConfig, McpRegistry};
use crate::runtime::{
    LlmWorkerEvent, LlmWorkerHandle, ShellControl, ShellWorkerEvent, ToolWorkerPool,
};
use crate::tools;
use crate::tools::permissions::PermissionManager;
use crate::tui;

use support::*;
use ui::*;
