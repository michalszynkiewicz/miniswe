//! replace_all tool — deterministic find-and-replace across all occurrences.
//!
//! Unlike `edit` which replaces one unique match, `replace_all` replaces
//! EVERY occurrence of `old` with `new` in a file. No LLM involved —
//! instant, deterministic, can't corrupt.

use anyhow::Result;
use serde_json::Value;

use crate::config::Config;
use super::ToolResult;

pub async fn execute(
    args: &Value,
    config: &Config,
) -> Result<ToolResult> {
    let path_str = args["path"].as_str().unwrap_or("");
    let old = args["old"].as_str().unwrap_or("");
    let new = args["new"].as_str().unwrap_or("");

    if path_str.is_empty() {
        return Ok(ToolResult::err("Missing required parameter: path".into()));
    }
    if old.is_empty() {
        return Ok(ToolResult::err("Missing required parameter: old".into()));
    }

    let path = config.project_root.join(path_str);
    if !path.exists() {
        return Ok(ToolResult::err(format!("File not found: {path_str}")));
    }

    let content = std::fs::read_to_string(&path)
        .map_err(|e| anyhow::anyhow!("Failed to read {path_str}: {e}"))?;

    let count = content.matches(old).count();

    if count == 0 {
        // Try whitespace-normalized matching
        let old_lines: Vec<&str> = old.lines().collect();

        if old_lines.len() == 1 {
            // Single-line: try trimmed matching
            let trimmed_old = old.trim();
            let trimmed_count = content.lines()
                .filter(|l| l.trim() == trimmed_old)
                .count();
            if trimmed_count > 0 {
                return Ok(ToolResult::err(format!(
                    "Exact text not found, but {trimmed_count} whitespace-similar match(es) exist. \
                     Check indentation — your 'old' text has different whitespace than the file."
                )));
            }
        }

        return Ok(ToolResult::err(format!(
            "'{old}' not found in {path_str}. Use read_file to check the exact content."
        )));
    }

    // Replace all occurrences
    let new_content = content.replace(old, new);
    std::fs::write(&path, &new_content)?;

    let new_lines = new_content.lines().count();
    let mut output = format!(
        "✓ Replaced {count} occurrence(s) in {path_str} ({new_lines} lines total)\n"
    );

    // Show first replacement with context
    if let Some(pos) = new_content.find(new) {
        let line_num = new_content[..pos].chars().filter(|&c| c == '\n').count() + 1;
        let lines: Vec<&str> = new_content.lines().collect();
        let start = line_num.saturating_sub(3);
        let end = (line_num + 3).min(lines.len());
        output.push_str(&format!("[first replacement at L{line_num}]\n"));
        for i in start..end {
            let marker = if i + 1 == line_num { ">" } else { " " };
            output.push_str(&format!("{marker}{:>4}│{}\n", i + 1, lines[i]));
        }
    }

    if count > 1 {
        output.push_str(&format!("({} more replacement(s) not shown)\n", count - 1));
    }

    Ok(ToolResult::ok(output))
}
