//! write_file tool — Whole-file rewrite.
//!
//! For quantized small models, whole-file output is more reliable than
//! search-and-replace diffs. The tradeoff is higher token cost, so files
//! should be kept small (~200 lines / ~4K chars).

use anyhow::Result;
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

use crate::config::Config;
use super::ToolResult;

/// Soft warning threshold for file size (lines).
const LARGE_FILE_WARNING: usize = 250;

pub async fn execute(args: &Value, config: &Config) -> Result<ToolResult> {
    let path_str = args["path"].as_str().unwrap_or("");
    let content = args["content"].as_str().unwrap_or("");

    if path_str.is_empty() {
        return Ok(ToolResult::err("Missing required parameter: path".into()));
    }
    if content.is_empty() {
        return Ok(ToolResult::err("Missing required parameter: content".into()));
    }

    let path = resolve_path(path_str, config);

    // Create parent directories if needed
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let is_new = !path.exists();
    let old_lines = if !is_new {
        fs::read_to_string(&path)
            .map(|c| c.lines().count())
            .unwrap_or(0)
    } else {
        0
    };

    // Write the file
    fs::write(&path, content)?;

    let new_lines = content.lines().count();
    let new_chars = content.len();

    let mut output = if is_new {
        format!("✓ Created {path_str} ({new_lines} lines, {new_chars} chars)\n")
    } else {
        let diff = if new_lines >= old_lines {
            format!("+{}", new_lines - old_lines)
        } else {
            format!("-{}", old_lines - new_lines)
        };
        format!("✓ Wrote {path_str} ({new_lines} lines, {new_chars} chars, {diff} lines)\n")
    };

    // Warn if file shrank dramatically — likely accidental truncation
    if !is_new && old_lines > 50 && new_lines < old_lines / 2 {
        output.push_str(&format!(
            "⚠ WARNING: File shrank from {old_lines} to {new_lines} lines (lost {}). \
             Did you include the COMPLETE file content? If not, use revert() to restore and try edit or replace_all instead.\n",
            old_lines - new_lines
        ));
    }

    // Warn if file is getting large
    if new_lines > LARGE_FILE_WARNING {
        output.push_str(&format!(
            "⚠ File is {new_lines} lines (>{LARGE_FILE_WARNING}). Consider splitting into smaller modules for better context efficiency.\n"
        ));
    }

    // Include the tail of the written file so the model can verify correctness
    // and make follow-up edits without a separate read_file call.
    let lines: Vec<&str> = content.lines().collect();
    let tail_start = lines.len().saturating_sub(30);
    output.push_str("[tail]\n");
    for (i, line) in lines[tail_start..].iter().enumerate() {
        let line_num = tail_start + i + 1;
        output.push_str(&format!("{line_num:>4}│{line}\n"));
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
