//! Top-level tool dispatch: resolves a tool name + JSON args into a
//! `ToolResult` by delegating to individual tool implementations or to the
//! edit-orchestration and code-intel submodules.
//!
//! `execute_tool` is the canonical entry point used by the REPL and run
//! loops for every tool except `edit_file` (which needs a router — see
//! [`super::edit_orchestration::execute_edit_file_tool`]).

use anyhow::{Result, bail};
use serde_json::Value;

use crate::config::Config;
use crate::lsp::LspClient;

use super::ToolResult;
use super::code_intel::{
    context_tool_architecture_notes, context_tool_project_info, context_tool_repo_map,
    lsp_find_references, lsp_goto_definition, lsp_project_diagnostics,
};
use super::definitions;
use super::edit_orchestration::{capture_edit_baseline, finalize_file_edit};
use super::permissions::{Action, PermissionManager};
use super::{delete_file, read_file, search, shell, task_update, web, write_file};

/// Execute a tool by name with the given arguments.
/// Grouped tools (file, code, web) are dispatched by their `action` parameter.
pub async fn execute_tool(
    name: &str,
    args: &Value,
    config: &Config,
    perms: &PermissionManager,
    lsp: Option<&LspClient>,
) -> Result<ToolResult> {
    match name {
        "file" => execute_file_tool(args, config, perms).await,
        "code" => execute_code_tool(args, config, lsp).await,
        "web" => execute_web_tool(args, config, perms).await,
        "write_file" => execute_write_file_tool(args, config, perms, lsp).await,
        "plan" => {
            let action = args["action"].as_str().unwrap_or("");
            if action == "help" {
                return Ok(ToolResult::ok(definitions::plan_help().into()));
            }
            if action == "scratchpad" {
                return task_update::execute(args, config).await;
            }
            // plan tool needs round number — caller (run.rs) handles this
            bail!("plan tool must be dispatched by caller with round context");
        }
        _ => bail!("Unknown tool: {name}"),
    }
}

// ── file tool group ──────────────────────────────────────────────────

async fn execute_file_tool(
    args: &Value,
    config: &Config,
    perms: &PermissionManager,
) -> Result<ToolResult> {
    let action = args["action"].as_str().unwrap_or("");

    match action {
        "help" => Ok(ToolResult::ok(
            definitions::file_help(config.tools.edit_mode).into(),
        )),

        "read" => {
            let path = args["path"].as_str().unwrap_or("");
            if let Err(e) = perms.resolve_and_check_path(path) {
                return Ok(ToolResult::err(e));
            }
            read_file::execute(args, config).await
        }

        "write" => Ok(ToolResult::err(
            "file(action='write') is no longer supported. Use write_file(path, content) to create or overwrite a file, or write_file(path) to create a new empty file.".into(),
        )),

        "delete" => {
            let path = args["path"].as_str().unwrap_or("");
            if let Err(e) = perms.resolve_and_check_path(path) {
                return Ok(ToolResult::err(e));
            }
            delete_file::execute(args, config).await
        }

        "replace" => {
            let partial_hint = match config.tools.edit_mode {
                crate::config::EditMode::Smart => {
                    "call edit_file with {\"path\":\"<file>\",\"task\":\"<what to change and why>\"} \
                     and let the planner produce the patch"
                }
                crate::config::EditMode::Fast => {
                    "use replace_range(path,start,end,content) or insert_at(path,after_line,content)"
                }
            };
            Ok(ToolResult::err(format!(
                "file(action='replace') is no longer supported. For partial edits, {partial_hint}. \
                 For full-file overwrites use write_file(path, content)."
            )))
        }

        "search" => search::execute(args, config).await,

        "shell" => {
            let cmd = args["command"].as_str().unwrap_or("");
            if let Err(e) = perms.check(&Action::Shell(cmd.into())) {
                return Ok(ToolResult::err(e));
            }
            shell::execute(args, config).await
        }

        "revert" => {
            // Revert is handled specially in run.rs (needs snapshot manager).
            // If called through execute_tool, it means snapshot system isn't available.
            Ok(ToolResult::err(
                "Revert must be called through the run loop (needs snapshot manager).".into(),
            ))
        }

        _ => Ok(ToolResult::err(format!(
            "Unknown file action: '{action}'. Use 'help' to see available actions."
        ))),
    }
}

async fn execute_write_file_tool(
    args: &Value,
    config: &Config,
    perms: &PermissionManager,
    lsp: Option<&LspClient>,
) -> Result<ToolResult> {
    let path = args["path"].as_str().unwrap_or("");
    if let Err(e) = perms.resolve_and_check_path(path) {
        return Ok(ToolResult::err(e));
    }
    let baseline = capture_edit_baseline(path, config, lsp).await;
    let mut result = write_file::execute(args, config).await?;
    if result.success {
        finalize_file_edit(path, config, &mut result, lsp, baseline, None).await;
    }
    Ok(result)
}

// ── code tool group ──────────────────────────────────────────────────

async fn execute_code_tool(
    args: &Value,
    config: &Config,
    lsp: Option<&LspClient>,
) -> Result<ToolResult> {
    let action = args["action"].as_str().unwrap_or("");

    match action {
        "help" => Ok(ToolResult::ok(definitions::code_help().into())),

        "goto_definition" => {
            let path = args["path"].as_str().unwrap_or("");
            let line = args["line"].as_u64().unwrap_or(1).saturating_sub(1) as u32;
            let column = args["column"].as_u64().unwrap_or(1).saturating_sub(1) as u32;
            lsp_goto_definition(path, line, column, config, lsp).await
        }

        "find_references" => {
            let path = args["path"].as_str().unwrap_or("");
            let line = args["line"].as_u64().unwrap_or(1).saturating_sub(1) as u32;
            let column = args["column"].as_u64().unwrap_or(1).saturating_sub(1) as u32;
            lsp_find_references(path, line, column, config, lsp).await
        }

        "diagnostics" => {
            if let Some(lsp) = lsp
                && lsp.is_ready()
                && !lsp.has_crashed()
                && let Some(result) = lsp_project_diagnostics(config, lsp).await
            {
                return Ok(result);
            }
            // Fallback: cargo check
            let shell_args = serde_json::json!({
                "command": format!("cd {} && cargo check --message-format=short 2>&1 | head -50",
                    config.project_root.display()),
                "timeout": 30
            });
            shell::execute(&shell_args, config).await
        }

        "repo_map" => {
            let keywords_str = args["keywords"].as_str().unwrap_or("");
            context_tool_repo_map(keywords_str, config)
        }

        "project_info" => context_tool_project_info(config),

        "architecture_notes" => context_tool_architecture_notes(config),

        _ => Ok(ToolResult::err(format!(
            "Unknown code action: '{action}'. Use 'help' to see available actions."
        ))),
    }
}

// ── web tool group ───────────────────────────────────────────────────

async fn execute_web_tool(
    args: &Value,
    config: &Config,
    perms: &PermissionManager,
) -> Result<ToolResult> {
    let action = args["action"].as_str().unwrap_or("");

    match action {
        "help" => Ok(ToolResult::ok(definitions::web_help().into())),

        "search" => {
            let query = args["query"].as_str().unwrap_or("");
            if let Err(e) = perms.check(&Action::WebSearch(query.into())) {
                return Ok(ToolResult::err(e));
            }
            web::search(args, config).await
        }

        "fetch" => {
            let url = args["url"].as_str().unwrap_or("");
            if let Err(e) = perms.check(&Action::WebFetch(url.into())) {
                return Ok(ToolResult::err(e));
            }
            web::fetch(args, config).await
        }

        _ => Ok(ToolResult::err(format!(
            "Unknown web action: '{action}'. Use 'help' to see available actions."
        ))),
    }
}
