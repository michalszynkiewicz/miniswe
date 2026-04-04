//! Unified context compressor — single-pass timeline compression.
//!
//! Replaces separate tool masking + history compression with one system
//! that sees the entire message stream and produces a coherent narrative.
//!
//! Budget (fractions of context_window):
//! - Output headroom: 1/6
//! - Compressed summary: 1/6
//! - Raw recent: 1/4
//! - Work zone (system prompt + current): rest

use crate::config::{Config, ModelRole};
use crate::context::estimate_tokens;
use crate::llm::{ChatRequest, Message, ModelRouter};

/// Compress old messages when raw history exceeds budget.
///
/// Walks the messages, keeps the newest within the raw budget,
/// summarizes older messages via LLM into a narrative block,
/// and archives full content to `.miniswe/session_archive.md`.
pub async fn maybe_compress(
    messages: &mut Vec<Message>,
    config: &Config,
    router: &ModelRouter,
    tool_def_tokens: usize,
) {
    let context_window = config.model.context_window;
    // Subtract fixed overhead: tool definitions + output headroom
    let available = context_window.saturating_sub(tool_def_tokens).saturating_sub(context_window / 6);
    let raw_budget = available / 3;            // 1/3 of available for raw recent
    let summary_budget = available / 4;        // 1/4 of available for compressed summary

    // Count tokens in non-system messages (the conversation)
    let mut total_tokens = 0;
    let mut msg_tokens: Vec<usize> = Vec::new();
    for msg in messages.iter() {
        if msg.role == "system" {
            msg_tokens.push(0);
            continue;
        }
        let tokens = estimate_tokens(msg.content.as_deref().unwrap_or(""));
        // Add tool call tokens
        if let Some(tcs) = &msg.tool_calls {
            for tc in tcs {
                let tc_tokens = estimate_tokens(&tc.function.arguments) + 5;
                total_tokens += tc_tokens;
            }
        }
        total_tokens += tokens;
        msg_tokens.push(tokens);
    }

    // Only compress if we exceed the raw budget
    if total_tokens <= raw_budget {
        return;
    }

    // Find the split point: keep newest messages within raw_budget
    let mut kept_tokens = 0;
    let mut split_idx = messages.len();
    for i in (0..messages.len()).rev() {
        if messages[i].role == "system" {
            continue;
        }
        kept_tokens += msg_tokens[i];
        if kept_tokens > raw_budget {
            split_idx = i + 1;
            break;
        }
    }

    // Don't compress if there's nothing old enough
    if split_idx <= 1 {
        return;
    }

    // Find first non-system message to start compressing from
    let compress_start = messages.iter().position(|m| m.role != "system").unwrap_or(0);
    if compress_start >= split_idx {
        return;
    }

    // Check if there's already a summary message (from previous compression)
    let existing_summary_idx = messages[compress_start..split_idx].iter()
        .position(|m| {
            m.role == "user" && m.content.as_deref()
                .is_some_and(|c| c.starts_with("[Session summary"))
        })
        .map(|i| i + compress_start);

    // Clone messages to compress (need to release borrow before mutating)
    let to_compress: Vec<Message> = messages[compress_start..split_idx].iter()
        .filter(|m| m.role != "system")
        .cloned()
        .collect();

    if to_compress.is_empty() {
        return;
    }

    let existing_summary = existing_summary_idx
        .and_then(|i| messages[i].content.clone())
        .unwrap_or_default();

    let to_compress_refs: Vec<&Message> = to_compress.iter().collect();
    let summary = match llm_summarize_timeline(&to_compress_refs, &existing_summary, summary_budget, router).await {
        Some(s) => s,
        None => heuristic_summarize(&to_compress_refs),
    };

    // Archive full content
    archive_messages(&to_compress_refs, config);

    let compress_count = to_compress.len();

    // Replace compressed messages with summary
    let after_split: Vec<Message> = messages[split_idx..].to_vec();
    messages.truncate(compress_start);

    messages.push(Message::user(&format!(
        "[Your earlier work in this session ({compress_count} messages compressed)]\n{summary}\n[Continue from where you left off. Do not re-introduce yourself or restart the task.]"
    )));

    messages.extend(after_split);
}

/// Ask the LLM to summarize a timeline of messages into a narrative.
async fn llm_summarize_timeline(
    messages: &[&Message],
    existing_summary: &str,
    budget_tokens: usize,
    router: &ModelRouter,
) -> Option<String> {
    let max_prompt_chars = router.config_for(ModelRole::Fast).context_window * 3;

    let mut timeline = String::new();
    if !existing_summary.is_empty() {
        timeline.push_str(&format!("Previous summary:\n{existing_summary}\n\n"));
    }
    timeline.push_str("New messages to incorporate:\n");

    for msg in messages {
        let role = &msg.role;
        let content = msg.content.as_deref().unwrap_or("");

        // Skip existing summary messages
        if content.starts_with("[Your earlier work") || content.starts_with("[Session summary") {
            continue;
        }

        match role.as_str() {
            "user" => {
                let truncated = crate::truncate_chars(content, 200);
                timeline.push_str(&format!("USER: {truncated}\n"));
            }
            "assistant" => {
                if let Some(tcs) = &msg.tool_calls {
                    let calls: Vec<String> = tcs.iter()
                        .map(|tc| format!("{}({})", tc.function.name, crate::truncate_chars(&tc.function.arguments, 100)))
                        .collect();
                    timeline.push_str(&format!("ASSISTANT called: {}\n", calls.join(", ")));
                } else {
                    let truncated = crate::truncate_chars(content, 200);
                    timeline.push_str(&format!("ASSISTANT: {truncated}\n"));
                }
            }
            "tool" => {
                let truncated = crate::truncate_chars(content, 300);
                timeline.push_str(&format!("TOOL RESULT: {truncated}\n"));
            }
            _ => {}
        }

        if timeline.len() > max_prompt_chars {
            break;
        }
    }

    let prompt = format!(
        "Summarize what YOU did in this coding session so far. Write in first person past tense.\n\
         IMPORTANT: Preserve exact function signatures and type names you encountered.\n\
         For each file you read or changed, include the actual function signatures.\n\
         Example: 'I changed run.rs: pub async fn run(config: Config, msg: &str, plan_only: bool, headless: bool) to add system_prompt_override: Option<String>'\n\
         Include: what you changed, what's left to do, any errors.\n\
         Keep it under {} tokens.\n\n\
         {timeline}",
        budget_tokens
    );

    let request = ChatRequest {
        messages: vec![
            Message::system("You write concise session summaries for a coding agent. Include file paths, function names, and what worked/failed."),
            Message::user(&prompt),
        ],
        tools: None,
        tool_choice: None,
    };

    let response = router.chat(ModelRole::Fast, &request).await.ok()?;
    let text = response.choices.first()?.message.content.as_deref()?;
    eprintln!("[compressor] summarized {} messages into {} chars", messages.len(), text.len());
    Some(text.to_string())
}

/// Heuristic fallback when LLM summarization fails.
fn heuristic_summarize(messages: &[&Message]) -> String {
    let mut summary = String::new();
    let mut files_read = Vec::new();
    let mut files_edited = Vec::new();
    let mut errors = Vec::new();

    for msg in messages {
        let content = msg.content.as_deref().unwrap_or("");

        if msg.role == "tool" {
            if content.contains("[read:") || content.starts_with("[src/") {
                if let Some(path) = content.split(':').nth(1).and_then(|s| s.split('→').next()) {
                    files_read.push(path.trim().to_string());
                }
            }
            if content.contains("✓ Edited") || content.contains("✓ Wrote") {
                if let Some(path) = content.split_whitespace().nth(2) {
                    files_edited.push(path.to_string());
                }
            }
            if content.contains("error") && !content.contains("[cargo check] OK") {
                let first_error = content.lines()
                    .find(|l| l.contains("error"))
                    .unwrap_or("(error details lost)");
                errors.push(crate::truncate_chars(first_error, 100));
            }
        }
    }

    if !files_read.is_empty() {
        files_read.dedup();
        summary.push_str(&format!("Files read: {}\n", files_read.join(", ")));
    }
    if !files_edited.is_empty() {
        files_edited.dedup();
        summary.push_str(&format!("Files edited: {}\n", files_edited.join(", ")));
    }
    if !errors.is_empty() {
        summary.push_str(&format!("Errors: {}\n", errors.join("; ")));
    }
    if summary.is_empty() {
        summary.push_str("(earlier session activity — use read_file(\".miniswe/session_archive.md\") for details)");
    }

    summary
}

/// Archive compressed messages to `.miniswe/session_archive.md`.
fn archive_messages(messages: &[&Message], config: &Config) {
    let archive_path = config.miniswe_dir().join("session_archive.md");
    let mut archive = std::fs::read_to_string(&archive_path).unwrap_or_default();

    archive.push_str(&format!("\n## Compressed at round ~{}\n", messages.len()));
    for msg in messages {
        let role = &msg.role;
        let content = msg.content.as_deref().unwrap_or("");

        // Skip old summaries
        if content.starts_with("[Your earlier work") || content.starts_with("[Session summary") {
            continue;
        }

        match role.as_str() {
            "assistant" => {
                if let Some(tcs) = &msg.tool_calls {
                    for tc in tcs {
                        archive.push_str(&format!("→ {}({})\n",
                            tc.function.name,
                            crate::truncate_chars(&tc.function.arguments, 200)));
                    }
                } else {
                    archive.push_str(&format!("ASSISTANT: {}\n", crate::truncate_chars(content, 500)));
                }
            }
            "tool" => {
                archive.push_str(&format!("RESULT: {}\n", crate::truncate_chars(content, 500)));
            }
            "user" => {
                archive.push_str(&format!("USER: {}\n", crate::truncate_chars(content, 200)));
            }
            _ => {}
        }
    }

    let _ = std::fs::write(&archive_path, &archive);
}
