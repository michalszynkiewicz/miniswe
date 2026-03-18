//! minime — A context-frugal CLI coding agent
//!
//! Designed for 64K context windows on consumer hardware.
//! Optimized for small LLMs (Devstral Small 2, etc.) running locally.

mod cli;
mod config;
mod context;
mod knowledge;
mod llm;
mod mcp;
mod tools;
mod tui;

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::EnvFilter;

use crate::cli::{Cli, Command};
use crate::config::Config;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Some(Command::Init) => {
            cli::commands::init::run().await?;
        }
        Some(Command::Info) => {
            cli::commands::info::run().await?;
        }
        Some(Command::Config) => {
            cli::commands::config::run().await?;
        }
        Some(Command::Plan { message }) => {
            let config = Config::load()?;
            cli::commands::run::run(config, &message, true).await?;
        }
        Some(Command::Docs { subcommand }) => {
            cli::commands::docs::run(subcommand).await?;
        }
        None => {
            // Interactive mode or single message
            let config = Config::load()?;
            if let Some(message) = cli.message {
                cli::commands::run::run(config, &message, false).await?;
            } else {
                cli::commands::repl::run(config).await?;
            }
        }
    }

    Ok(())
}
