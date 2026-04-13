//! write_file — Create, overwrite, or touch a file.
//!
//! If `content` is omitted, this creates a new empty file. For partial edits
//! to existing code, prefer edit_file (Smart mode) or replace_range/insert_at
//! (Fast mode) unless the model is deliberately rewriting the entire file.

use anyhow::Result;
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

use super::ToolResult;
use crate::config::Config;

/// Soft warning threshold for file size (lines).
const LARGE_FILE_WARNING: usize = 250;

pub async fn execute(args: &Value, config: &Config) -> Result<ToolResult> {
    let path_str = args["path"].as_str().unwrap_or("");
    let content = args["content"].as_str().unwrap_or("");
    let has_content = args.get("content").is_some();

    if path_str.is_empty() {
        return Ok(ToolResult::err(
            "Missing required parameter: path. Expected JSON arguments: {\"action\":\"write\",\"path\":\"src/bin/hello.rs\",\"content\":\"fn main() {\\n    println!(\\\"hello\\\");\\n}\\n\"}. Omit content only to create a new empty file.".into()
        ));
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

    let partial_edit_hint = match config.tools.edit_mode {
        crate::config::EditMode::Smart => "use edit_file",
        crate::config::EditMode::Fast => "use replace_range or insert_at",
    };

    // Block writes that would truncate >50% of an existing file.
    // This prevents catastrophic data loss when the model sends partial content.
    if !is_new && !has_content {
        return Ok(ToolResult::err(format!(
            "{path_str} already exists. Omit content only when creating a new empty file. For edits to an existing file, {partial_edit_hint} or provide the complete replacement content."
        )));
    }

    let new_lines = content.lines().count();
    if has_content && !is_new && old_lines > 50 && new_lines < old_lines / 2 {
        return Ok(ToolResult::err(format!(
            "BLOCKED: This would truncate {path_str} from {old_lines} to {new_lines} lines \
             (losing {} lines). This is almost certainly accidental — you probably forgot to \
             include the complete file content.\n\
             Options:\n\
             1. For a partial edit to an existing file, {partial_edit_hint}\n\
             2. Use file(action='read') first, then write_file with the COMPLETE content\n\
             3. If the file is already corrupted, use file(action='revert') to restore it",
            old_lines - new_lines
        )));
    }

    // Write the file
    fs::write(&path, content)?;

    let new_chars = content.len();

    let mut output = if is_new {
        if has_content {
            format!("✓ Created {path_str} ({new_lines} lines, {new_chars} chars)\n")
        } else {
            format!("✓ Created empty file {path_str}\n")
        }
    } else {
        let diff = if new_lines >= old_lines {
            format!("+{}", new_lines - old_lines)
        } else {
            format!("-{}", old_lines - new_lines)
        };
        format!("✓ Wrote {path_str} ({new_lines} lines, {new_chars} chars, {diff} lines)\n")
    };

    // Warn if file is getting large
    if has_content && new_lines > LARGE_FILE_WARNING {
        output.push_str(&format!(
            "⚠ File is {new_lines} lines (>{LARGE_FILE_WARNING}). Consider splitting into smaller modules for better context efficiency.\n"
        ));
    }

    // Include the tail of the written file so the model can verify correctness
    // and make follow-up edits without a separate read_file call.
    let lines: Vec<&str> = content.lines().collect();
    output.push_str("[tail]\n");
    if lines.is_empty() {
        output.push_str("(empty file)\n");
    } else {
        let tail_start = lines.len().saturating_sub(30);
        for (i, line) in lines[tail_start..].iter().enumerate() {
            let line_num = tail_start + i + 1;
            output.push_str(&format!("{line_num:>4}│{line}\n"));
        }
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
