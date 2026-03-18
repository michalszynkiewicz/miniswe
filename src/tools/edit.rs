//! edit tool — Search-and-replace file editing.

use anyhow::Result;
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

use crate::config::Config;
use super::ToolResult;

pub async fn execute(args: &Value, config: &Config) -> Result<ToolResult> {
    let path_str = args["path"].as_str().unwrap_or("");
    let old = args["old"].as_str().unwrap_or("");
    let new = args["new"].as_str().unwrap_or("");

    if path_str.is_empty() {
        return Ok(ToolResult::err("Missing required parameter: path".into()));
    }
    if old.is_empty() {
        return Ok(ToolResult::err("Missing required parameter: old".into()));
    }

    let path = resolve_path(path_str, config);

    // For new files, create them if old is empty and file doesn't exist
    if !path.exists() && old.is_empty() {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, new)?;
        return Ok(ToolResult::ok(format!("Created new file: {path_str}")));
    }

    if !path.exists() {
        return Ok(ToolResult::err(format!("File not found: {path_str}")));
    }

    let content = fs::read_to_string(&path)
        .map_err(|e| anyhow::anyhow!("Failed to read {path_str}: {e}"))?;

    // Count occurrences
    let count = content.matches(old).count();

    if count == 0 {
        return Ok(ToolResult::err(format!(
            "old_content not found in {path_str}. Make sure the text matches exactly (including whitespace)."
        )));
    }

    if count > 1 {
        return Ok(ToolResult::err(format!(
            "old_content matches {count} locations in {path_str}. Provide more context to make it unique."
        )));
    }

    // Perform the replacement
    let new_content = content.replacen(old, new, 1);

    // Write the file
    fs::write(&path, &new_content)?;

    // Show context around the edit
    let edited_lines: Vec<&str> = new_content.lines().collect();

    // Find where the edit occurred
    let new_lines: Vec<&str> = new.lines().collect();
    let mut edit_start = 0;
    for (i, line) in edited_lines.iter().enumerate() {
        if !new_lines.is_empty() && line.contains(new_lines[0]) {
            edit_start = i;
            break;
        }
    }

    // Show 3 lines of context around the edit
    let context_start = edit_start.saturating_sub(3);
    let context_end = (edit_start + new_lines.len() + 3).min(edited_lines.len());

    let mut output = format!("✓ Edited {path_str} (1 replacement)\n");
    for i in context_start..context_end {
        let marker = if i >= edit_start && i < edit_start + new_lines.len() {
            "+"
        } else {
            " "
        };
        output.push_str(&format!("{marker}{:>4}│{}\n", i + 1, edited_lines[i]));
    }

    Ok(ToolResult::ok(output))
}

fn resolve_path(path_str: &str, config: &Config) -> PathBuf {
    let path = PathBuf::from(path_str);
    if path.is_absolute() {
        path
    } else {
        config.project_root.join(path)
    }
}
