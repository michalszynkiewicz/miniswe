//! Tool system for minime.
//!
//! Implements the 10-tool surface described in the design:
//! - read_file, read_symbol, search, edit, shell, task_update, diagnostics
//! - web_search, web_fetch, docs_lookup

mod definitions;
mod edit;
mod read_file;
mod search;
mod shell;
mod task_update;
mod web;

pub use definitions::tool_definitions;

use anyhow::{Result, bail};
use serde_json::Value;

use crate::config::Config;

/// Result of executing a tool.
#[derive(Debug, Clone)]
pub struct ToolResult {
    /// The output content to return to the LLM.
    pub content: String,
    /// Whether the tool execution was successful.
    pub success: bool,
}

impl ToolResult {
    pub fn ok(content: String) -> Self {
        Self {
            content,
            success: true,
        }
    }

    pub fn err(content: String) -> Self {
        Self {
            content,
            success: false,
        }
    }
}

/// Execute a tool by name with the given arguments.
pub async fn execute_tool(
    name: &str,
    args: &Value,
    config: &Config,
) -> Result<ToolResult> {
    match name {
        "read_file" => read_file::execute(args, config).await,
        "read_symbol" => {
            // For now, read_symbol falls back to search + read_file
            // Full tree-sitter symbol lookup comes with the knowledge engine
            let symbol_name = args["name"].as_str().unwrap_or("");
            let search_args = serde_json::json!({
                "query": symbol_name,
                "scope": "symbols",
                "max_results": 5
            });
            search::execute(&search_args, config).await
        }
        "search" => search::execute(args, config).await,
        "edit" => edit::execute(args, config).await,
        "shell" => shell::execute(args, config).await,
        "task_update" => task_update::execute(args, config).await,
        "diagnostics" => {
            // Stub: try running the project's linter
            let _path = args["path"].as_str().unwrap_or(".");
            let shell_args = serde_json::json!({
                "command": format!("cd {} && cargo check --message-format=short 2>&1 | head -50",
                    config.project_root.display()),
                "timeout": 30
            });
            shell::execute(&shell_args, config).await
        }
        "web_search" => web::search(args, config).await,
        "web_fetch" => web::fetch(args, config).await,
        "docs_lookup" => web::docs_lookup(args, config).await,
        _ => bail!("Unknown tool: {name}"),
    }
}
