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
use crate::runtime::{LlmWorkerEvent, LlmWorkerHandle};

/// Check if compression is needed without doing it.
pub fn needs_compression(messages: &[Message], config: &Config, tool_def_tokens: usize) -> bool {
    let context_window = config.model.context_window;
    let available = context_window
        .saturating_sub(tool_def_tokens)
        .saturating_sub(context_window / 6);
    let raw_budget = available / 3;

    let total_tokens: usize = messages
        .iter()
        .filter(|m| m.role != "system")
        .map(|m| estimate_tokens(m.content.as_deref().unwrap_or("")))
        .sum();

    total_tokens > raw_budget
}

/// Compress old messages when raw history exceeds budget.
///
/// If plan tool is enabled, first asks the model to update its plan.
/// The actual compression uses the plan as an anchor for the summary.
pub async fn maybe_compress(
    messages: &mut Vec<Message>,
    config: &Config,
    router: &ModelRouter,
    llm_worker: &LlmWorkerHandle,
    tool_def_tokens: usize,
    plan_update_requested: &mut bool,
) {
    let context_window = config.model.context_window;
    // Subtract fixed overhead: tool definitions + output headroom
    let available = context_window
        .saturating_sub(tool_def_tokens)
        .saturating_sub(context_window / 6);
    let raw_budget = available / 3; // 1/3 of available for raw recent
    let summary_budget = available / 4; // 1/4 of available for compressed summary

    // If plan is enabled and we haven't asked for an update yet,
    // ask the model to update its plan before compressing
    if config.tools.plan && !*plan_update_requested {
        let total: usize = messages
            .iter()
            .filter(|m| m.role != "system")
            .map(|m| estimate_tokens(m.content.as_deref().unwrap_or("")))
            .sum();

        if total > raw_budget {
            // Inject plan update request instead of compressing
            messages.push(Message::user(
                "[Context is getting large. Before I compress, update your plan: \
                 call plan(action='check', step=N) for any completed steps, \
                 or plan(action='set') if the plan needs revision. \
                 Then I'll compress and continue.]",
            ));
            *plan_update_requested = true;
            return;
        }
    }

    // Reset the flag — plan was updated (or not needed), proceed with compression
    *plan_update_requested = false;

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
    let compress_start = messages
        .iter()
        .position(|m| m.role != "system")
        .unwrap_or(0);
    if compress_start >= split_idx {
        return;
    }

    // Check if there's already a summary message (from previous compression)
    let existing_summary_idx = messages[compress_start..split_idx]
        .iter()
        .position(|m| {
            m.role == "user"
                && m.content
                    .as_deref()
                    .is_some_and(|c| c.starts_with("[Session summary"))
        })
        .map(|i| i + compress_start);

    // Clone messages to compress (need to release borrow before mutating)
    let to_compress: Vec<Message> = messages[compress_start..split_idx]
        .iter()
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
    let summary = match llm_summarize_timeline(
        &to_compress_refs,
        &existing_summary,
        summary_budget,
        router,
        llm_worker,
    )
    .await
    {
        Some(s) => s,
        None => heuristic_summarize(&to_compress_refs),
    };

    // Archive full content
    archive_messages(&to_compress_refs, config);

    // Replace compressed messages with summary
    let after_split: Vec<Message> = messages[split_idx..].to_vec();
    messages.truncate(compress_start);

    messages.push(Message::user(&format!(
        "[Your earlier work in this session]\n{summary}\n[Details: read_file(\".miniswe/session_archive.md\"). Continue from where you left off.]"
    )));

    messages.extend(after_split);
}

/// Ask the LLM to summarize a timeline of messages into a narrative.
async fn llm_summarize_timeline(
    messages: &[&Message],
    existing_summary: &str,
    budget_tokens: usize,
    router: &ModelRouter,
    llm_worker: &LlmWorkerHandle,
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
                    let calls: Vec<String> = tcs
                        .iter()
                        .map(|tc| {
                            format!(
                                "{}({})",
                                tc.function.name,
                                crate::truncate_chars(&tc.function.arguments, 100)
                            )
                        })
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
        "List WHAT you accomplished, one line per file changed. Use this format:\n\
         - file.rs: what changed (include exact function signatures if modified)\n\
         - file.rs: ✗ attempted but failed — reason\n\
         End with: Still need: [what's left]\n\
         Keep it under {} tokens. No process narrative.\n\n\
         {timeline}",
        budget_tokens
    );

    let request = ChatRequest {
        messages: vec![
            Message::system(
                "List completed actions, one per line. Include exact signatures when functions were changed. No explanation.",
            ),
            Message::user(&prompt),
        ],
        tools: None,
        tool_choice: None,
    };

    let mut events = llm_worker.submit_non_streaming(ModelRole::Fast, request);
    let response = loop {
        match events.recv().await {
            Some(LlmWorkerEvent::Completed(Ok(response))) => break response,
            Some(LlmWorkerEvent::Completed(Err(_))) => return None,
            Some(LlmWorkerEvent::Token(_)) => {}
            None => return None,
        }
    };
    let text = response.choices.first()?.message.content.as_deref()?;
    eprintln!(
        "[compressor] summarized {} messages into {} chars",
        messages.len(),
        text.len()
    );
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
                let first_error = content
                    .lines()
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
                        archive.push_str(&format!(
                            "→ {}({})\n",
                            tc.function.name,
                            crate::truncate_chars(&tc.function.arguments, 200)
                        ));
                    }
                } else {
                    archive.push_str(&format!(
                        "ASSISTANT: {}\n",
                        crate::truncate_chars(content, 500)
                    ));
                }
            }
            "tool" => {
                archive.push_str(&format!(
                    "RESULT: {}\n",
                    crate::truncate_chars(content, 500)
                ));
            }
            "user" => {
                archive.push_str(&format!("USER: {}\n", crate::truncate_chars(content, 200)));
            }
            _ => {}
        }
    }

    let _ = std::fs::write(&archive_path, &archive);
}
