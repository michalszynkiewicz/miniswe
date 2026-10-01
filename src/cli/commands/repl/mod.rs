//! Interactive REPL mode with ratatui TUI.

mod agent_loop;
mod explore;
mod session;
mod support;
#[cfg(test)]
mod tests;
mod tui_ui;

pub use session::run;

use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::Mutex;

use anyhow::Result;
use crossterm::ExecutableCommand;
use crossterm::event::{KeyCode, KeyModifiers};
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::Terminal;
use ratatui::backend::Backend;
use ratatui::backend::CrosstermBackend;
use tokio::sync::mpsc;

use crate::cli::commands::agent::hints::is_file_write;
use crate::cli::commands::agent::turn;
use crate::cli::commands::agent::turn_state;
use crate::cli::commands::agent::ui::AgentUi;
use crate::config::{CeremonyMode, Config, EditMode, ModelRole};
use crate::context;
use crate::llm::{ChatRequest, Message, ModelRouter};
use crate::logging::SessionLog;
use crate::lsp::LspClient;
use crate::mcp::{McpConfig, McpRegistry};
use crate::runtime::{
    LlmWorkerEvent, LlmWorkerHandle, ShellControl, ShellWorkerEvent, ToolWorkerPool,
};
use crate::tools;
use crate::tools::permissions::PermissionManager;
use crate::tui::app::{App, AppMode, LineStyle, PlanStepView};
use crate::tui::event::{self, AppEvent};
use crate::tui::ui;

// Used only by `tests` (below) via its `use super::*;` — the admit-span
// extraction into `agent::turn::call_gate` moved their one production
// call site out of this module.
#[cfg(test)]
use crate::cli::commands::agent::display::summarize_args;
#[cfg(test)]
use crate::cli::commands::agent::explore_gate::{explore_block_reason, shell_is_read_only};
#[cfg(test)]
use crate::cli::commands::agent::hints::loop_detected_hint;
#[cfg(test)]
use crate::cli::commands::agent::permissions::permission_action;
#[cfg(test)]
use crate::tools::permissions::Action;

use agent_loop::*;
use explore::*;
use support::*;
use tui_ui::*;
