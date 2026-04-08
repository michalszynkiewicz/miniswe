//! delete_file tool — Delete an existing file.

use std::fs;

use anyhow::Result;
use serde_json::Value;

use crate::config::Config;

use super::ToolResult;

pub async fn execute(args: &Value, config: &Config) -> Result<ToolResult> {
    let path_str = args["path"].as_str().unwrap_or("");

    if path_str.is_empty() {
        return Ok(ToolResult::err(
            "Missing required parameter: path. Expected JSON arguments: {\"action\":\"delete\",\"path\":\"src/bin/hello.rs\"}.".into(),
        ));
    }

    let path = config.project_root.join(path_str);

    if !path.exists() {
        return Ok(ToolResult::err(format!(
            "File not found: {path_str}. Use file(action='read') to inspect files or file(action='search') to locate them."
        )));
    }

    if path.is_dir() {
        return Ok(ToolResult::err(format!(
            "{path_str} is a directory. file(action='delete') only deletes regular files."
        )));
    }

    fs::remove_file(&path)?;
    Ok(ToolResult::ok(format!("Deleted {path_str}")))
}
