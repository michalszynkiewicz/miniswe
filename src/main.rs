//! miniswe — A lightweight CLI coding agent for local LLMs

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::EnvFilter;

use miniswe::cli::{Cli, Command};
use miniswe::config::Config;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Some(Command::Init) => {
            miniswe::cli::commands::init::run().await?;
        }
        Some(Command::Info) => {
            miniswe::cli::commands::info::run().await?;
        }
        Some(Command::Config) => {
            miniswe::cli::commands::config::run().await?;
        }
        Some(Command::Plan { message }) => {
            let config = Config::load()?;
            miniswe::cli::commands::run::run(config, &message, true, cli.yes).await?;
        }
        None => {
            let config = Config::load()?;
            if let Some(message) = cli.message {
                // Single-shot mode
                miniswe::cli::commands::run::run(config, &message, false, cli.yes).await?;
            } else {
                // Interactive REPL
                miniswe::cli::commands::repl::run(config, cli.yes).await?;
            }
        }
    }

    Ok(())
}
