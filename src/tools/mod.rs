//! Tool system for miniswe.
//!
//! 12 built-in tools + MCP bridge. All file access is jailed to the project
//! root. Destructive actions (shell, web, MCP) require user permission.
//! After file edits, the index is incrementally updated.

mod edit;
mod read_file;
mod read_symbol;
mod search;
mod shell;
mod task_update;
mod web;
mod write_file;

pub mod definitions;
pub mod permissions;
pub use definitions::tool_definitions;
pub use permissions::PermissionManager;

use anyhow::{Result, bail};
use permissions::Action;
use serde_json::Value;

use crate::config::Config;
use crate::knowledge::ProjectIndex;
use crate::knowledge::indexer;

/// Result of executing a tool.
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub content: String,
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
/// All file paths are jail-checked. Destructive actions require permission.
/// After successful file mutations (edit, write_file), the index is updated.
pub async fn execute_tool(
    name: &str,
    args: &Value,
    config: &Config,
    perms: &PermissionManager,
) -> Result<ToolResult> {
    match name {
        "read_file" => {
            let path = args["path"].as_str().unwrap_or("");
            if let Err(e) = perms.resolve_and_check_path(path) {
                return Ok(ToolResult::err(e));
            }
            read_file::execute(args, config).await
        }
        "read_symbol" => read_symbol::execute(args, config).await,
        "search" => search::execute(args, config).await,
        "edit" => {
            let path = args["path"].as_str().unwrap_or("");
            if let Err(e) = perms.resolve_and_check_path(path) {
                return Ok(ToolResult::err(e));
            }
            let mut result = edit::execute(args, config).await?;
            if result.success {
                reindex_changed_file(path, config);
                auto_cargo_check(path, config, &mut result);
            }
            Ok(result)
        }
        "write_file" => {
            let path = args["path"].as_str().unwrap_or("");
            if let Err(e) = perms.resolve_and_check_path(path) {
                return Ok(ToolResult::err(e));
            }
            let mut result = write_file::execute(args, config).await?;
            if result.success {
                reindex_changed_file(path, config);
                auto_cargo_check(path, config, &mut result);
            }
            Ok(result)
        }
        "shell" => {
            let cmd = args["command"].as_str().unwrap_or("");
            if let Err(e) = perms.check(&Action::Shell(cmd.into())) {
                return Ok(ToolResult::err(e));
            }
            shell::execute(args, config).await
        }
        "task_update" => task_update::execute(args, config).await,
        "diagnostics" => {
            let shell_args = serde_json::json!({
                "command": format!("cd {} && cargo check --message-format=short 2>&1 | head -50",
                    config.project_root.display()),
                "timeout": 30
            });
            shell::execute(&shell_args, config).await
        }
        "web_search" => {
            let query = args["query"].as_str().unwrap_or("");
            if let Err(e) = perms.check(&Action::WebSearch(query.into())) {
                return Ok(ToolResult::err(e));
            }
            web::search(args, config).await
        }
        "web_fetch" => {
            let url = args["url"].as_str().unwrap_or("");
            if let Err(e) = perms.check(&Action::WebFetch(url.into())) {
                return Ok(ToolResult::err(e));
            }
            web::fetch(args, config).await
        }
        "docs_lookup" => web::docs_lookup(args, config).await,
        _ => bail!("Unknown tool: {name}"),
    }
}

/// Re-index a single changed file. Best-effort — doesn't fail the tool call.
fn reindex_changed_file(rel_path: &str, config: &Config) {
    let miniswe_dir = config.miniswe_dir();
    let abs_path = config.project_root.join(rel_path);

    let mut index = match ProjectIndex::load(&miniswe_dir) {
        Ok(idx) => idx,
        Err(_) => return, // no index yet, skip
    };

    indexer::reindex_file(rel_path, &abs_path, &mut index, &miniswe_dir);
}

/// Auto-run `cargo check` after editing a .rs file. Appends output to the tool result.
fn auto_cargo_check(path: &str, config: &Config, result: &mut ToolResult) {
    if !path.ends_with(".rs") {
        return;
    }
    // Only if this is a Rust project (Cargo.toml exists)
    if !config.project_root.join("Cargo.toml").exists() {
        return;
    }

    let output = std::process::Command::new("cargo")
        .args(["check", "--message-format=short"])
        .current_dir(&config.project_root)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output();

    match output {
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            // Extract just the error/warning lines, cap at 30 lines
            let relevant: Vec<&str> = stderr
                .lines()
                .filter(|l| {
                    l.contains("error") || l.contains("warning") || l.starts_with("  ")
                })
                .take(30)
                .collect();

            if o.status.success() {
                result.content.push_str("\n[cargo check] OK");
            } else if relevant.is_empty() {
                result.content.push_str("\n[cargo check] failed (no details captured)");
                result.success = false;
            } else {
                result.content.push_str("\n[cargo check]\n");
                result.content.push_str(&relevant.join("\n"));
                result.success = false;
            }
        }
        Err(_) => {
            // cargo not available or timed out — silently skip
        }
    }
}
