//! fix_file tool — LLM generates edit instructions, miniswe applies them.
//!
//! The model describes the task, miniswe sends file content to the LLM
//! and gets back a list of line-level edits. Edits are applied deterministically.
//! No full-file rewrite, no truncation risk.

use anyhow::Result;
use serde_json::Value;

use crate::config::{Config, ModelRole};
use crate::llm::{ChatRequest, Message, ModelRouter};
use super::ToolResult;

/// Max lines per window for reliable LLM recall.
const WINDOW_SIZE: usize = 800;
/// Overlap between windows to catch edits at boundaries.
const WINDOW_OVERLAP: usize = 100;

pub async fn execute(
    args: &Value,
    config: &Config,
    router: &ModelRouter,
) -> Result<ToolResult> {
    let path_str = args["path"].as_str().unwrap_or("");
    let task = args["task"].as_str().unwrap_or("");

    if path_str.is_empty() {
        return Ok(ToolResult::err("Missing required parameter: path".into()));
    }
    if task.is_empty() {
        return Ok(ToolResult::err("Missing required parameter: task".into()));
    }

    let path = config.project_root.join(path_str);
    if !path.exists() {
        return Ok(ToolResult::err(format!("File not found: {path_str}")));
    }

    let content = std::fs::read_to_string(&path)
        .map_err(|e| anyhow::anyhow!("Failed to read {path_str}: {e}"))?;

    let lines: Vec<&str> = content.lines().collect();
    let total_lines = lines.len();

    // Split into windows
    let windows = build_windows(total_lines, WINDOW_SIZE, WINDOW_OVERLAP);

    let mut all_edits: Vec<Edit> = Vec::new();
    let mut output = String::new();

    for (win_idx, (start, end)) in windows.iter().enumerate() {
        let window_content: String = lines[*start..*end].iter()
            .enumerate()
            .map(|(i, l)| format!("{:>4}│{}", start + i + 1, l))
            .collect::<Vec<_>>()
            .join("\n");

        let window_info = if windows.len() > 1 {
            format!("(window {}/{}, lines {}-{} of {})",
                win_idx + 1, windows.len(), start + 1, end, total_lines)
        } else {
            format!("({total_lines} lines)")
        };

        let prompt = format!(
            "You are editing {path_str} {window_info}.\n\
             Task: {task}\n\n\
             Output ONLY a list of edits in this exact format, one per change:\n\
             EDIT <line_number>\n\
             OLD: <exact line content to replace>\n\
             NEW: <replacement line content>\n\n\
             Rules:\n\
             - Output ONLY EDIT blocks, nothing else\n\
             - Line numbers must match the file\n\
             - OLD must match the line exactly\n\
             - If no changes needed in this section, output: NO_CHANGES\n\n\
             File content:\n{window_content}"
        );

        let request = ChatRequest {
            messages: vec![
                Message::system("You output edit instructions. No explanations, no markdown, just EDIT blocks."),
                Message::user(&prompt),
            ],
            tools: None,
            tool_choice: None,
        };

        let response = match router.chat(ModelRole::Fast, &request).await {
            Ok(r) => r,
            Err(e) => {
                output.push_str(&format!("Window {}: LLM error: {e}\n", win_idx + 1));
                continue;
            }
        };

        let text = match response.choices.first().and_then(|c| c.message.content.as_deref()) {
            Some(t) => t,
            None => continue,
        };

        if text.trim() == "NO_CHANGES" {
            continue;
        }

        let edits = parse_edits(text);
        output.push_str(&format!("Window {}: {} edit(s) found\n", win_idx + 1, edits.len()));
        all_edits.extend(edits);
    }

    if all_edits.is_empty() {
        return Ok(ToolResult::ok(format!(
            "No changes needed in {path_str} for task: {task}"
        )));
    }

    // Apply edits in reverse order (so line numbers stay valid)
    all_edits.sort_by(|a, b| b.line.cmp(&a.line));
    // Dedup by line number (overlapping windows might produce duplicates)
    all_edits.dedup_by(|a, b| a.line == b.line);

    let mut new_lines: Vec<String> = lines.iter().map(|l| l.to_string()).collect();
    let mut applied = 0;
    let mut failed = 0;

    for edit in &all_edits {
        let idx = edit.line.saturating_sub(1);
        if idx >= new_lines.len() {
            failed += 1;
            output.push_str(&format!("  L{}: out of range\n", edit.line));
            continue;
        }
        if new_lines[idx].trim() == edit.old.trim() {
            new_lines[idx] = edit.new.clone();
            applied += 1;
        } else {
            // Try exact match
            if new_lines[idx] == edit.old {
                new_lines[idx] = edit.new.clone();
                applied += 1;
            } else {
                failed += 1;
                output.push_str(&format!(
                    "  L{}: expected '{}' but found '{}'\n",
                    edit.line,
                    crate::truncate_chars(&edit.old, 50),
                    crate::truncate_chars(&new_lines[idx], 50)
                ));
            }
        }
    }

    // Write the file
    let new_content = new_lines.join("\n");
    let final_content = if content.ends_with('\n') && !new_content.ends_with('\n') {
        format!("{new_content}\n")
    } else {
        new_content
    };
    std::fs::write(&path, &final_content)?;

    output.push_str(&format!(
        "✓ Applied {applied} edit(s) to {path_str} ({} lines)",
        final_content.lines().count()
    ));
    if failed > 0 {
        output.push_str(&format!(", {failed} failed (line content mismatch)"));
    }
    output.push('\n');

    if applied == 0 && failed > 0 {
        Ok(ToolResult::err(output))
    } else {
        Ok(ToolResult::ok(output))
    }
}

pub struct Edit {
    pub line: usize,
    pub old: String,
    pub new: String,
}

/// Parse EDIT blocks from LLM output.
pub fn parse_edits(text: &str) -> Vec<Edit> {
    let mut edits = Vec::new();
    let mut lines = text.lines().peekable();

    while let Some(line) = lines.next() {
        let trimmed = line.trim();

        // Look for "EDIT <line_number>"
        if let Some(rest) = trimmed.strip_prefix("EDIT ") {
            let line_num: usize = match rest.trim().parse() {
                Ok(n) => n,
                Err(_) => continue,
            };

            // Next line should be "OLD: ..."
            let old = match lines.next() {
                Some(l) => {
                    let t = l.trim();
                    t.strip_prefix("OLD:").or(t.strip_prefix("OLD :"))
                        .unwrap_or(t).trim().to_string()
                }
                None => continue,
            };

            // Next line should be "NEW: ..."
            let new = match lines.next() {
                Some(l) => {
                    let t = l.trim();
                    t.strip_prefix("NEW:").or(t.strip_prefix("NEW :"))
                        .unwrap_or(t).trim().to_string()
                }
                None => continue,
            };

            edits.push(Edit { line: line_num, old, new });
        }
    }

    edits
}

/// Build window ranges for a file.
pub fn build_windows(total_lines: usize, window_size: usize, overlap: usize) -> Vec<(usize, usize)> {
    if total_lines <= window_size {
        return vec![(0, total_lines)];
    }

    let mut windows = Vec::new();
    let mut start = 0;
    while start < total_lines {
        let end = (start + window_size).min(total_lines);
        windows.push((start, end));
        if end >= total_lines {
            break;
        }
        start = end - overlap;
    }
    windows
}
