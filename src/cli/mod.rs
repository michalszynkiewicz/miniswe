pub mod commands;

use std::path::PathBuf;

use clap::{Parser, Subcommand};

/// miniswe — A lightweight CLI coding agent for local LLMs
#[derive(Parser, Debug)]
#[command(name = "miniswe", version, about)]
pub struct Cli {
    /// Message to send to the agent (runs in single-shot mode)
    pub message: Option<String>,

    /// Continue from last session
    #[arg(long, short = 'c')]
    pub r#continue: bool,

    /// Non-interactive mode: auto-approve all permissions, no prompts
    #[arg(long, short = 'y')]
    pub yes: bool,

    /// Replay mode: seed the agent loop from a captured context fixture
    /// (a context.json from scripts/replay/extract-fixture.py) instead of a
    /// fresh assemble. The working tree should already be the matching state.
    #[arg(long, value_name = "CONTEXT_JSON")]
    pub replay_context: Option<PathBuf>,

    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Initialize project knowledge base
    Init,

    /// Show project info and index stats
    Info,

    /// Show/edit configuration
    Config,

    /// Plan-only mode (exploration, no edits)
    Plan {
        /// The planning question or task
        message: String,
    },
}
