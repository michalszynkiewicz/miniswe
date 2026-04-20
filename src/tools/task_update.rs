//! task_update tool — Rewrite the task scratchpad.

use anyhow::Result;
use serde_json::Value;
use std::fs;

use super::ToolResult;
use crate::config::Config;

pub async fn execute(args: &Value, config: &Config) -> Result<ToolResult> {
    let content = match super::args::require_str(args, "content") {
        Ok(c) => c,
        Err(e) => return Ok(ToolResult::err(e)),
    };
    if content.is_empty() {
        return Ok(ToolResult::err(
            "Parameter 'content' must not be empty".into(),
        ));
    }

    // Validate structure
    if !content.contains("## Current Task") {
        return Ok(ToolResult::err(
            "Scratchpad must contain a '## Current Task' section".into(),
        ));
    }

    if !content.contains("## Plan") {
        return Ok(ToolResult::err(
            "Scratchpad must contain a '## Plan' section".into(),
        ));
    }

    let scratchpad_path = config.miniswe_path("scratchpad.md");

    // Ensure .miniswe directory exists
    if let Some(parent) = scratchpad_path.parent() {
        fs::create_dir_all(parent)?;
    }

    fs::write(&scratchpad_path, content)?;

    Ok(ToolResult::ok("✓ Scratchpad updated".into()))
}
