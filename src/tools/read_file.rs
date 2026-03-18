//! read_file tool — Read a file or line range with line numbers.
//!
//! Applies context compression: strips comments and std imports while
//! preserving line numbers, so the model sees accurate positions for edits.

use anyhow::Result;
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

use crate::config::Config;
use crate::context::compress;
use super::ToolResult;

/// Maximum lines to return before truncation.
const MAX_LINES: usize = 200;

/// File extensions that get compression applied.
const COMPRESSIBLE: &[&str] = &[
    "rs", "py", "js", "ts", "tsx", "jsx", "go", "java",
    "c", "cpp", "h", "hpp", "rb", "sh", "bash",
];

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

    // Check file size before reading (reject files > 10MB to avoid OOM)
    const MAX_FILE_SIZE: u64 = 10 * 1024 * 1024;
    if let Ok(meta) = fs::metadata(&path) {
        if meta.len() > MAX_FILE_SIZE {
            return Ok(ToolResult::err(format!(
                "{path_str} is too large ({:.1}MB). Use shell(\"head -n 200 {path_str}\") instead.",
                meta.len() as f64 / 1_048_576.0
            )));
        }
    }

    let content = match fs::read_to_string(&path) {
        Ok(c) => c,
        Err(e) => return Ok(ToolResult::err(format!("Failed to read {path_str}: {e}"))),
    };

    let total_lines = content.lines().count();

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

    // Determine if we should compress this file
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    let should_compress = COMPRESSIBLE.contains(&ext);

    let mut output = String::new();

    if should_compress {
        // Compressed output: strip comments + std imports, preserve line numbers
        let compressed = compress::compress_for_reading(&content, ext);

        // Count how many lines were stripped
        let stripped_count = compressed[start_line.saturating_sub(1)..end_line]
            .iter()
            .filter(|l| l.is_none())
            .count();
        let _visible_count = (end_line - start_line + 1) - stripped_count;

        output.push_str(&format!("[{path_str}: {total_lines} lines"));
        if start_line != 1 || end_line != total_lines {
            output.push_str(&format!(", showing L{start_line}-{end_line}"));
        }
        if stripped_count > 0 {
            output.push_str(&format!(", {stripped_count} comment/import lines stripped"));
        }
        output.push_str("]\n");

        let mut lines_shown = 0;
        for i in (start_line - 1)..end_line {
            if lines_shown >= MAX_LINES {
                let remaining = end_line - i;
                output.push_str(&format!(
                    "\n... truncated ({remaining} more lines). Use start_line/end_line to read specific ranges."
                ));
                break;
            }

            if let Some(Some(line)) = compressed.get(i) {
                output.push_str(&format!("{:>4}│{line}\n", i + 1));
                lines_shown += 1;
            }
            // None lines are simply skipped — line numbers stay correct
        }
    } else {
        // Non-code files: output raw with line numbers
        let lines: Vec<&str> = content.lines().collect();
        let selected = &lines[start_line.saturating_sub(1)..end_line];
        let truncated = selected.len() > MAX_LINES;
        let display_lines = if truncated {
            &selected[..MAX_LINES]
        } else {
            selected
        };

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
