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
        // Try whitespace-normalized matching as a fallback
        let content_lines: Vec<&str> = content.lines().collect();
        let old_lines: Vec<&str> = old.lines().collect();

        if let Some(start) = find_normalized_match(&content_lines, &old_lines) {
            // Found a match ignoring whitespace — do the replacement using actual text
            let end = start + old_lines.len();
            let original_old = content_lines[start..end].join("\n");
            let new_content = content.replacen(&original_old, new, 1);
            fs::write(&path, &new_content)?;

            let edited_lines: Vec<&str> = new_content.lines().collect();
            let total_lines = edited_lines.len();
            let new_lines: Vec<&str> = new.lines().collect();
            let context_start = start.saturating_sub(10);
            let context_end = (start + new_lines.len() + 10).min(edited_lines.len());

            let mut output = format!(
                "✓ Edited {path_str} (1 replacement, showing L{}-{})\n",
                context_start + 1, context_end
            );
            for i in context_start..context_end {
                let marker = if i >= start && i < start + new_lines.len() { "+" } else { " " };
                output.push_str(&format!("{marker}{:>4}│{}\n", i + 1, edited_lines[i]));
            }
            output.push_str("Note: matched with fuzzy/normalized matching — your 'old' text didn't match exactly.\n");
            if total_lines < 200 {
                output.push_str(&format!(
                    "Note: {path_str} is {total_lines} lines. For multiple changes, use write_file to rewrite the whole file in one call.\n"
                ));
            }
            return Ok(ToolResult::ok(output));
        }

        // No match at all — show context to help the model
        let file_lines = &content_lines;
        let mut err_msg = format!(
            "old_content not found in {path_str}. Make sure the text matches exactly (including whitespace).\n"
        );
        let first_old_line = old.lines().next().unwrap_or("").trim();
        if !first_old_line.is_empty() {
            let mut matches_shown = 0;
            for (i, line) in file_lines.iter().enumerate() {
                if line.contains(first_old_line) {
                    let ctx_start = i.saturating_sub(5);
                    let ctx_end = (i + 6).min(file_lines.len());
                    err_msg.push_str(&format!("[near match at L{}]\n", i + 1));
                    for j in ctx_start..ctx_end {
                        let marker = if j == i { ">" } else { " " };
                        err_msg.push_str(&format!("{marker}{:>4}│{}\n", j + 1, file_lines[j]));
                    }
                    matches_shown += 1;
                    if matches_shown >= 3 { break; }
                }
            }
        }
        err_msg.push_str(&format!("[{path_str}: {} lines total]\n", file_lines.len()));
        err_msg.push_str("HINT: Copy the exact text from the line numbers shown above into 'old'. Or use write_file to rewrite the whole file.\n");
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

    // Detect function signature changes and nudge about call sites
    if old.contains("fn ") && new.contains("fn ") {
        if let Some(fn_name) = extract_fn_name(new) {
            output.push_str(&format!(
                "\nIMPORTANT: You changed a function signature. Use search(\"{fn_name}\") to find ALL call sites and update them.\n"
            ));
        }
    }

    Ok(ToolResult::ok(output))
}

/// Find a whitespace-normalized match of `old_lines` in `content_lines`.
///
/// Uses a layered matching strategy (inspired by Aider):
/// 1. Exact whitespace-trimmed match (existing behavior)
/// 2. Indentation-preserving match (same content, different indent level)
/// 3. Fuzzy match via line similarity (handles minor hallucinations)
fn find_normalized_match(content_lines: &[&str], old_lines: &[&str]) -> Option<usize> {
    if old_lines.is_empty() { return None; }
    let old_len = old_lines.len();
    let max = content_lines.len().saturating_sub(old_len.saturating_sub(1));

    // Layer 1: Exact trimmed match
    'exact: for i in 0..max {
        for (j, old_line) in old_lines.iter().enumerate() {
            if content_lines[i + j].trim() != old_line.trim() {
                continue 'exact;
            }
        }
        return Some(i);
    }

    // Layer 2: Indentation-preserving match — same stripped content but
    // with a consistent indent delta across all lines
    'indent: for i in 0..max {
        let mut delta: Option<isize> = None;
        for (j, old_line) in old_lines.iter().enumerate() {
            let content_stripped = content_lines[i + j].trim();
            let old_stripped = old_line.trim();
            if content_stripped != old_stripped {
                continue 'indent;
            }
            // Check indent consistency (skip blank lines)
            if !old_stripped.is_empty() {
                let content_indent = content_lines[i + j].len() - content_lines[i + j].trim_start().len();
                let old_indent = old_line.len() - old_line.trim_start().len();
                let d = content_indent as isize - old_indent as isize;
                match delta {
                    None => delta = Some(d),
                    Some(existing) if existing != d => continue 'indent,
                    _ => {}
                }
            }
        }
        // Layer 2 only matches if indent delta is non-zero (layer 1 already handles zero)
        if delta.unwrap_or(0) != 0 {
            return Some(i);
        }
    }

    // Layer 3: Fuzzy match — allow minor per-line differences (typos, small
    // hallucinations). Require ≥80% of lines to match exactly (trimmed) and
    // the remaining lines to be ≥60% similar by character overlap.
    if old_len >= 3 {
        let mut best_pos = None;
        let mut best_score: f64 = 0.0;
        let match_threshold = 0.80; // 80% of lines must be exact
        let similarity_threshold = 0.60; // non-exact lines must be ≥60% similar
        let overall_threshold = 0.90; // weighted score must be ≥90%

        for i in 0..max {
            let mut exact_count = 0;
            let mut sim_sum: f64 = 0.0;
            let mut all_above_threshold = true;

            for (j, old_line) in old_lines.iter().enumerate() {
                let content_trimmed = content_lines[i + j].trim();
                let old_trimmed = old_line.trim();

                if content_trimmed == old_trimmed {
                    exact_count += 1;
                    sim_sum += 1.0;
                } else {
                    let sim = line_similarity(content_trimmed, old_trimmed);
                    if sim < similarity_threshold {
                        all_above_threshold = false;
                        break;
                    }
                    sim_sum += sim;
                }
            }

            if !all_above_threshold {
                continue;
            }

            let exact_ratio = exact_count as f64 / old_len as f64;
            let overall_score = sim_sum / old_len as f64;

            if exact_ratio >= match_threshold && overall_score >= overall_threshold && overall_score > best_score {
                best_score = overall_score;
                best_pos = Some(i);
            }
        }

        if best_pos.is_some() {
            return best_pos;
        }
    }

    None
}

/// Character-level similarity between two strings (Dice coefficient on bigrams).
/// Returns 0.0 for completely different, 1.0 for identical.
fn line_similarity(a: &str, b: &str) -> f64 {
    if a == b { return 1.0; }
    if a.is_empty() || b.is_empty() { return 0.0; }
    if a.len() == 1 || b.len() == 1 {
        return if a == b { 1.0 } else { 0.0 };
    }

    let bigrams_a: Vec<(char, char)> = a.chars().zip(a.chars().skip(1)).collect();
    let bigrams_b: Vec<(char, char)> = b.chars().zip(b.chars().skip(1)).collect();

    let mut matches = 0;
    let mut used = vec![false; bigrams_b.len()];

    for ba in &bigrams_a {
        for (j, bb) in bigrams_b.iter().enumerate() {
            if !used[j] && ba == bb {
                matches += 1;
                used[j] = true;
                break;
            }
        }
    }

    (2.0 * matches as f64) / (bigrams_a.len() + bigrams_b.len()) as f64
}

/// Extract a function name from text containing "fn name(...)".
fn extract_fn_name(text: &str) -> Option<&str> {
    for line in text.lines() {
        let trimmed = line.trim();
        for prefix in &["pub async fn ", "pub fn ", "async fn ", "fn "] {
            if let Some(rest) = trimmed.strip_prefix(prefix) {
                let name = rest.split(|c: char| !c.is_alphanumeric() && c != '_').next()?;
                if !name.is_empty() {
                    return Some(name);
                }
            }
        }
    }
    None
}

fn resolve_path(path_str: &str, config: &Config) -> PathBuf {
    let path = PathBuf::from(path_str);
    if path.is_absolute() {
        path
    } else {
        config.project_root.join(path)
    }
}
