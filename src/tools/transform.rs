//! transform tool — LLM-powered code transformation.
//!
//! Finds all occurrences of a pattern in a file, extracts each with
//! surrounding context, sends each chunk to the LLM for transformation,
//! and splices the results back. Handles multi-line code, any language.

use anyhow::Result;
use serde_json::Value;

use crate::config::{Config, ModelRole};
use crate::llm::{ChatRequest, Message, ModelRouter};
use super::ToolResult;

/// Context lines before and after each match.
const CONTEXT_LINES: usize = 5;

pub async fn execute(
    args: &Value,
    config: &Config,
    router: &ModelRouter,
) -> Result<ToolResult> {
    let path_str = args["path"].as_str().unwrap_or("");
    let find = args["find"].as_str().unwrap_or("");
    let instruction = args["instruction"].as_str().unwrap_or("");

    if path_str.is_empty() {
        return Ok(ToolResult::err("Missing required parameter: path".into()));
    }
    if find.is_empty() {
        return Ok(ToolResult::err("Missing required parameter: find".into()));
    }
    if instruction.is_empty() {
        return Ok(ToolResult::err("Missing required parameter: instruction".into()));
    }

    let path = config.project_root.join(path_str);
    if !path.exists() {
        return Ok(ToolResult::err(format!("File not found: {path_str}")));
    }

    let content = std::fs::read_to_string(&path)
        .map_err(|e| anyhow::anyhow!("Failed to read {path_str}: {e}"))?;

    let lines: Vec<&str> = content.lines().collect();

    // Find all lines containing the pattern
    let match_lines: Vec<usize> = lines.iter()
        .enumerate()
        .filter(|(_, line)| line.contains(find))
        .map(|(i, _)| i)
        .collect();

    if match_lines.is_empty() {
        return Ok(ToolResult::err(format!(
            "Pattern '{find}' not found in {path_str}."
        )));
    }

    // Build non-overlapping chunks with context around each match
    let chunks = build_chunks(&lines, &match_lines, CONTEXT_LINES);

    let mut output = format!(
        "Transforming {} occurrence(s) of '{find}' in {path_str}...\n",
        match_lines.len()
    );

    // Transform each chunk via LLM
    let mut new_lines = lines.iter().map(|l| l.to_string()).collect::<Vec<_>>();
    let mut offset: i64 = 0; // Track line count changes from previous transforms

    for (chunk_idx, chunk) in chunks.iter().enumerate() {
        let start = chunk.start;
        let end = chunk.end;
        let adj_start = (start as i64 + offset) as usize;
        let adj_end = (end as i64 + offset) as usize;

        let chunk_text: String = new_lines[adj_start..adj_end]
            .iter()
            .enumerate()
            .map(|(i, l)| format!("{:>4}│{}", adj_start + i + 1, l))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "Apply this transformation to the code below:\n\
             INSTRUCTION: {instruction}\n\
             FIND: {find}\n\n\
             Return ONLY the transformed code lines, nothing else. \
             Keep all unchanged lines exactly as they are. \
             Do NOT include line numbers in your output.\n\n\
             Code:\n{chunk_text}"
        );

        let request = ChatRequest {
            messages: vec![
                Message::system("You are a precise code transformer. Return only the transformed code, no explanation."),
                Message::user(&prompt),
            ],
            tools: None,
            tool_choice: None,
        };

        let response = match router.chat(ModelRole::Fast, &request).await {
            Ok(r) => r,
            Err(e) => {
                output.push_str(&format!("  chunk {}: LLM error: {e}\n", chunk_idx + 1));
                continue;
            }
        };

        let transformed = match response.choices.first().and_then(|c| c.message.content.as_deref()) {
            Some(t) => t,
            None => {
                output.push_str(&format!("  chunk {}: empty LLM response\n", chunk_idx + 1));
                continue;
            }
        };

        // Strip any markdown code fences the LLM might add
        let cleaned = strip_code_fences(transformed);

        let transformed_lines: Vec<String> = cleaned.lines().map(|l| l.to_string()).collect();
        let old_len = adj_end - adj_start;
        let new_len = transformed_lines.len();

        // Splice transformed lines back
        new_lines.splice(adj_start..adj_end, transformed_lines);

        offset += new_len as i64 - old_len as i64;
        output.push_str(&format!(
            "  chunk {}: lines {}-{} → {} lines\n",
            chunk_idx + 1, start + 1, end, new_len
        ));
    }

    // Write the file
    let new_content = new_lines.join("\n");
    // Preserve trailing newline if original had one
    let final_content = if content.ends_with('\n') && !new_content.ends_with('\n') {
        format!("{new_content}\n")
    } else {
        new_content
    };

    std::fs::write(&path, &final_content)?;

    let total_lines = final_content.lines().count();
    output.push_str(&format!(
        "✓ Transformed {path_str} ({} chunks, {total_lines} lines total)\n",
        chunks.len()
    ));

    Ok(ToolResult::ok(output))
}

/// A chunk: a range of lines [start, end) to transform together.
struct Chunk {
    start: usize,
    end: usize,
}

/// Build non-overlapping chunks around match lines, merging overlaps.
fn build_chunks(lines: &[&str], match_lines: &[usize], context: usize) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let total = lines.len();

    for &line in match_lines {
        let start = line.saturating_sub(context);
        let end = (line + context + 1).min(total);

        // Merge with previous chunk if overlapping
        if let Some(last) = chunks.last_mut() {
            let last_chunk: &mut Chunk = last;
            if start <= last_chunk.end {
                last_chunk.end = end;
                continue;
            }
        }
        chunks.push(Chunk { start, end });
    }

    chunks
}

/// Strip markdown code fences if the LLM wrapped its output.
fn strip_code_fences(text: &str) -> &str {
    let trimmed = text.trim();
    if let Some(rest) = trimmed.strip_prefix("```") {
        // Skip the language identifier on the first line
        let after_lang = rest.find('\n').map(|i| &rest[i + 1..]).unwrap_or(rest);
        if let Some(content) = after_lang.strip_suffix("```") {
            return content.trim_end();
        }
        after_lang.trim_end()
    } else {
        trimmed
    }
}
