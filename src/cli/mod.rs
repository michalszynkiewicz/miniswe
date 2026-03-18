pub mod commands;

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

    /// Manage documentation cache
    Docs {
        #[command(subcommand)]
        subcommand: DocsSubcommand,
    },
}

#[derive(Subcommand, Debug)]
pub enum DocsSubcommand {
    /// Add a documentation source
    Add {
        /// URL to fetch docs from (e.g., https://docs.astro.build/llms.txt)
        url: String,
    },
    /// List cached documentation
    List,
    /// Refresh all cached docs
    Refresh,
}
