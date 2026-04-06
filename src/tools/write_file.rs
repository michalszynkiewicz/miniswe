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

    // Block writes that would truncate >50% of an existing file.
    // This prevents catastrophic data loss when the model sends partial content.
    let new_lines = content.lines().count();
    if !is_new && old_lines > 50 && new_lines < old_lines / 2 {
        return Ok(ToolResult::err(format!(
            "BLOCKED: This would truncate {path_str} from {old_lines} to {new_lines} lines \
             (losing {} lines). This is almost certainly accidental — you probably forgot to \
             include the complete file content.\n\
             Options:\n\
             1. Use replace() to change only the specific lines you need\n\
             2. Use replace(all=true) for find-and-replace across the file\n\
             3. Use read_file() first, then write_file() with the COMPLETE content\n\
             4. If the file is already corrupted, use revert() to restore it",
            old_lines - new_lines
        )));
    }

    // Write the file
    fs::write(&path, content)?;

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
