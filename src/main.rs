//! buzztr — A lightweight CLI coding agent for local LLMs

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::EnvFilter;

use buzztr::cli::{Cli, Command};
use buzztr::config::Config;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Some(Command::Init) => {
            buzztr::cli::commands::init::run().await?;
        }
        Some(Command::Info) => {
            buzztr::cli::commands::info::run().await?;
        }
        Some(Command::Config) => {
            buzztr::cli::commands::config::run().await?;
        }
        Some(Command::Plan { message }) => {
            let config = Config::load()?;
            buzztr::cli::commands::run::run(
                config,
                &message,
                true,
                cli.yes,
                cli.r#continue,
                None,
                None,
            )
            .await?;
        }
        None => {
            let config = Config::load()?;
            if let Some(replay) = cli.replay_context {
                // Replay mode: the captured context drives the loop; the message
                // arg is a placeholder (the real prompt is the trailing captured
                // gate message).
                let msg = cli.message.unwrap_or_else(|| "[replay]".to_string());
                buzztr::cli::commands::run::run(
                    config,
                    &msg,
                    false,
                    cli.yes,
                    cli.r#continue,
                    Some(replay),
                    cli.replay_apply,
                )
                .await?;
            } else if let Some(message) = cli.message {
                // Single-shot mode
                buzztr::cli::commands::run::run(
                    config,
                    &message,
                    false,
                    cli.yes,
                    cli.r#continue,
                    None,
                    None,
                )
                .await?;
            } else {
                // Interactive REPL
                buzztr::cli::commands::repl::run(config, cli.yes, cli.r#continue).await?;
            }
        }
    }

    Ok(())
}
