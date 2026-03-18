//! task_update tool — Rewrite the task scratchpad.

use anyhow::Result;
use serde_json::Value;
use std::fs;

use crate::config::Config;
use super::ToolResult;

pub async fn execute(args: &Value, config: &Config) -> Result<ToolResult> {
    let content = args["content"].as_str().unwrap_or("");

    if content.is_empty() {
        return Ok(ToolResult::err("Missing required parameter: content".into()));
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

    Ok(ToolResult::ok(
        "✓ Scratchpad updated".into(),
    ))
}
