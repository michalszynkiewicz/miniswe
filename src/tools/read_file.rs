//! read_file tool — Read a file or line range with line numbers.

use anyhow::Result;
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

use crate::config::Config;
use super::ToolResult;

/// Maximum lines to return before truncation.
const MAX_LINES: usize = 200;

pub async fn execute(args: &Value, config: &Config) -> Result<ToolResult> {
    let path_str = args["path"]
        .as_str()
        .unwrap_or("");

    if path_str.is_empty() {
        return Ok(ToolResult::err("Missing required parameter: path".into()));
    }

    let path = resolve_path(path_str, config);

    if !path.exists() {
        return Ok(ToolResult::err(format!("File not found: {path_str}")));
    }

    if path.is_dir() {
        return Ok(ToolResult::err(format!("{path_str} is a directory, not a file. Use search or shell(\"ls\") instead.")));
    }

    let content = match fs::read_to_string(&path) {
        Ok(c) => c,
        Err(e) => return Ok(ToolResult::err(format!("Failed to read {path_str}: {e}"))),
    };

    let lines: Vec<&str> = content.lines().collect();
    let total_lines = lines.len();

    let start_line = args["start_line"]
        .as_u64()
        .map(|n| n as usize)
        .unwrap_or(1)
        .max(1);
    let end_line = args["end_line"]
        .as_u64()
        .map(|n| n as usize)
        .unwrap_or(total_lines)
        .min(total_lines);

    if start_line > total_lines {
        return Ok(ToolResult::err(format!(
            "start_line {start_line} exceeds file length ({total_lines} lines)"
        )));
    }

    let selected: Vec<&str> = lines[(start_line - 1)..end_line].to_vec();
    let truncated = selected.len() > MAX_LINES;
    let display_lines = if truncated {
        &selected[..MAX_LINES]
    } else {
        &selected
    };

    let mut output = String::new();
    output.push_str(&format!("[{path_str}: {total_lines} lines"));
    if start_line != 1 || end_line != total_lines {
        output.push_str(&format!(", showing L{start_line}-{end_line}"));
    }
    output.push_str("]\n");

    for (i, line) in display_lines.iter().enumerate() {
        let line_num = start_line + i;
        output.push_str(&format!("{line_num:>4}│{line}\n"));
    }

    if truncated {
        output.push_str(&format!(
            "\n... truncated ({} more lines). Use start_line/end_line to read specific ranges.",
            selected.len() - MAX_LINES
        ));
    }

    Ok(ToolResult::ok(output))
}

/// Resolve a path relative to the project root.
fn resolve_path(path_str: &str, config: &Config) -> PathBuf {
    let path = PathBuf::from(path_str);
    if path.is_absolute() {
        path
    } else {
        config.project_root.join(path)
    }
}
