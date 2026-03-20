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
        // Show what the file actually contains near where the match might be,
        // so the model can fix its edit without a separate read_file call.
        let file_lines: Vec<&str> = content.lines().collect();
        let mut err_msg = format!(
            "old_content not found in {path_str}. Make sure the text matches exactly (including whitespace).\n"
        );
        let first_old_line = old.lines().next().unwrap_or("").trim();
        if !first_old_line.is_empty() {
            for (i, line) in file_lines.iter().enumerate() {
                if line.contains(first_old_line) {
                    let ctx_start = i.saturating_sub(5);
                    let ctx_end = (i + 6).min(file_lines.len());
                    err_msg.push_str(&format!("[near match at L{}]\n", i + 1));
                    for j in ctx_start..ctx_end {
                        let marker = if j == i { ">" } else { " " };
                        err_msg.push_str(&format!("{marker}{:>4}│{}\n", j + 1, file_lines[j]));
                    }
                    break;
                }
            }
        }
        err_msg.push_str(&format!("[{path_str}: {} lines total]\n", file_lines.len()));
        return Ok(ToolResult::err(err_msg));
    }

    if count > 1 {
        // Show where the matches are so the model can include more context
        let mut match_lines = Vec::new();
        let mut search_from = 0;
        for _ in 0..count.min(5) {
            if let Some(pos) = content[search_from..].find(old) {
                let abs_pos = search_from + pos;
                let line_num = content[..abs_pos].chars().filter(|&c| c == '\n').count() + 1;
                match_lines.push(format!("L{line_num}"));
                search_from = abs_pos + 1;
            }
        }
        return Ok(ToolResult::err(format!(
            "old_content matches {count} locations in {path_str} (at {}).\n\
             Include more surrounding lines in 'old' to make the match unique, \
             or use write_file to rewrite the whole file.",
            match_lines.join(", ")
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

    // Show ±10 lines of context around the edit so the model has enough
    // surrounding code to attempt follow-up edits without re-reading.
    let context_start = edit_start.saturating_sub(10);
    let context_end = (edit_start + new_lines.len() + 10).min(edited_lines.len());

    let total_lines = edited_lines.len();
    let mut output = format!(
        "✓ Edited {path_str} (1 replacement, showing L{}-{})\n",
        context_start + 1,
        context_end
    );
    for i in context_start..context_end {
        let marker = if i >= edit_start && i < edit_start + new_lines.len() {
            "+"
        } else {
            " "
        };
        output.push_str(&format!("{marker}{:>4}│{}\n", i + 1, edited_lines[i]));
    }

    // Nudge model to use write_file for small files with multiple changes
    if total_lines < 200 {
        output.push_str(&format!(
            "\nNote: {path_str} is {total_lines} lines. For multiple changes, use write_file to rewrite the whole file in one call.\n"
        ));
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
