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

use crate::config::{CompactionStrategy, Config, ModelRole};
use crate::context::estimate_tokens;
use crate::llm::{ChatRequest, Message, ModelRouter};
use crate::runtime::{LlmWorkerEvent, LlmWorkerHandle};

/// Token cost of one message: content **plus** tool-call argument bytes.
/// Used for both the compression trigger and the keep/compress split so the
/// two agree — coding histories are dominated by large tool-call arg blobs,
/// and counting them in the trigger but not the split made compression keep
/// more raw history than budgeted.
fn msg_token_cost(msg: &Message) -> usize {
    let mut tokens = estimate_tokens(msg.content.as_deref().unwrap_or(""));
    if let Some(tcs) = &msg.tool_calls {
        for tc in tcs {
            tokens += estimate_tokens(&tc.function.arguments) + 5;
        }
    }
    tokens
}

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
        .map(msg_token_cost)
        .sum();

    total_tokens > raw_budget
}

/// Budget split (raw-recent, summary) in tokens, from context window minus
/// fixed overhead (tool definitions + output headroom). Shared by every
/// strategy so they all fire at the same `raw_budget` threshold.
fn budgets(config: &Config, tool_def_tokens: usize) -> (usize, usize) {
    let context_window = config.model.context_window;
    let available = context_window
        .saturating_sub(tool_def_tokens)
        .saturating_sub(context_window / 6);
    (available / 3, available / 4) // (raw recent, compressed summary)
}

/// Per-message token cost, with system messages counted as 0 (they are never
/// compressed). The keep/compress split sums this, so it must agree with the
/// compression trigger total.
fn per_msg_tokens(messages: &[Message]) -> Vec<usize> {
    messages
        .iter()
        .map(|m| {
            if m.role == "system" {
                0
            } else {
                msg_token_cost(m)
            }
        })
        .collect()
}

/// Total tokens of the non-system conversation history.
fn history_token_total(messages: &[Message]) -> usize {
    messages
        .iter()
        .filter(|m| m.role != "system")
        .map(msg_token_cost)
        .sum()
}

/// Split point that keeps the newest messages within `raw_budget`; everything
/// before it is old enough to compress/drop. Returns `messages.len()` when
/// nothing exceeds the budget.
fn find_split_idx(messages: &[Message], msg_tokens: &[usize], raw_budget: usize) -> usize {
    let mut kept = 0;
    let mut split_idx = messages.len();
    for i in (0..messages.len()).rev() {
        if messages[i].role == "system" {
            continue;
        }
        kept += msg_tokens[i];
        if kept > raw_budget {
            split_idx = i + 1;
            break;
        }
    }
    split_idx
}

/// First non-system message index (where compression begins).
fn first_history_idx(messages: &[Message]) -> usize {
    messages
        .iter()
        .position(|m| m.role != "system")
        .unwrap_or(0)
}

/// Recognize any in-context summary marker so the summarizer doesn't re-nest a
/// previous summary into a new one.
fn is_summary_marker(content: &str) -> bool {
    content.starts_with("[Your earlier work")
        || content.starts_with("[Session summary")
        || content.starts_with("[Summary of earlier conversation]")
}

/// One standardized stderr line per compaction event — grep'd by the
/// compaction benchmark driver. Goes to stderr (not tracing) so it is captured
/// regardless of tracing config, alongside the `[compressor] summarized…` line.
fn emit_compaction_metric(
    strategy: &str,
    before_tokens: usize,
    after_tokens: usize,
    msgs_before: usize,
    msgs_after: usize,
) {
    let elided = before_tokens.saturating_sub(after_tokens);
    eprintln!(
        "[compaction] strategy={strategy} before_tokens={before_tokens} \
         after_tokens={after_tokens} elided_tokens={elided} \
         msgs_before={msgs_before} msgs_after={msgs_after}"
    );
}

/// Compress old messages when raw history exceeds budget.
///
/// Dispatches to the configured [`CompactionStrategy`]. `Unified` is miniswe's
/// production behavior; the others are canonical baselines for benchmarking.
/// All strategies share the same `raw_budget` trigger (see [`budgets`]).
pub async fn maybe_compress(
    messages: &mut Vec<Message>,
    config: &Config,
    router: &ModelRouter,
    llm_worker: &LlmWorkerHandle,
    tool_def_tokens: usize,
    plan_update_requested: &mut bool,
) {
    match config.context.compaction {
        CompactionStrategy::Unified => {
            compact_unified(
                messages,
                config,
                router,
                llm_worker,
                tool_def_tokens,
                plan_update_requested,
            )
            .await
        }
        CompactionStrategy::RollingSummary => {
            compact_rolling_summary(messages, config, router, llm_worker, tool_def_tokens).await
        }
        CompactionStrategy::SlidingWindow => {
            compact_sliding_window(messages, config, tool_def_tokens)
        }
        CompactionStrategy::ObservationMasking => {
            compact_observation_masking(messages, config, tool_def_tokens)
        }
    }
}

/// miniswe production strategy: rolling LLM summary anchored on the plan, with
/// the full pre-compression text archived to disk and a pointer in the summary.
///
/// If the plan tool is enabled, first asks the model to update its plan; the
/// actual compression uses the plan as an anchor for the summary.
async fn compact_unified(
    messages: &mut Vec<Message>,
    config: &Config,
    router: &ModelRouter,
    llm_worker: &LlmWorkerHandle,
    tool_def_tokens: usize,
    plan_update_requested: &mut bool,
) {
    let (raw_budget, summary_budget) = budgets(config, tool_def_tokens);

    // If plan is enabled and we haven't asked for an update yet,
    // ask the model to update its plan before compressing
    if config.tools.plan && !*plan_update_requested && history_token_total(messages) > raw_budget {
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

    // Reset the flag — plan was updated (or not needed), proceed with compression
    *plan_update_requested = false;

    let msg_tokens = per_msg_tokens(messages);
    let total_tokens: usize = msg_tokens.iter().sum();

    // Only compress if we exceed the raw budget
    if total_tokens <= raw_budget {
        return;
    }

    let split_idx = find_split_idx(messages, &msg_tokens, raw_budget);

    // Don't compress if there's nothing old enough
    if split_idx <= 1 {
        return;
    }

    let compress_start = first_history_idx(messages);
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

    let msgs_before = messages.len();
    let to_compress_refs: Vec<&Message> = to_compress.iter().collect();
    let summary = match llm_summarize_timeline(
        &to_compress_refs,
        &existing_summary,
        summary_budget,
        router,
        llm_worker,
        SummaryStyle::Structured,
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
        "[Your earlier work in this session]\n{summary}\n[Details: file(action='read', path='.miniswe/session_archive.md'). Continue from where you left off.]"
    )));

    messages.extend(after_split);

    emit_compaction_metric(
        "unified",
        total_tokens,
        history_token_total(messages),
        msgs_before,
        messages.len(),
    );
}

/// Textbook rolling LLM summarization: summarize the old turns into a running
/// summary (carrying the previous summary forward) and keep recent turns raw.
/// No plan-anchor, no disk archive, neutral summarization prompt.
async fn compact_rolling_summary(
    messages: &mut Vec<Message>,
    config: &Config,
    router: &ModelRouter,
    llm_worker: &LlmWorkerHandle,
    tool_def_tokens: usize,
) {
    const MARKER: &str = "[Summary of earlier conversation]";
    let (raw_budget, summary_budget) = budgets(config, tool_def_tokens);

    let msg_tokens = per_msg_tokens(messages);
    let total_tokens: usize = msg_tokens.iter().sum();
    if total_tokens <= raw_budget {
        return;
    }

    let split_idx = find_split_idx(messages, &msg_tokens, raw_budget);
    if split_idx <= 1 {
        return;
    }
    let compress_start = first_history_idx(messages);
    if compress_start >= split_idx {
        return;
    }

    // Carry the previous running summary forward (textbook rolling summary).
    let existing_summary_idx = messages[compress_start..split_idx]
        .iter()
        .position(|m| {
            m.role == "user" && m.content.as_deref().is_some_and(|c| c.starts_with(MARKER))
        })
        .map(|i| i + compress_start);

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
        .map(|c| c.trim_start_matches(MARKER).trim().to_string())
        .unwrap_or_default();

    let msgs_before = messages.len();
    let to_compress_refs: Vec<&Message> = to_compress.iter().collect();
    let summary = match llm_summarize_timeline(
        &to_compress_refs,
        &existing_summary,
        summary_budget,
        router,
        llm_worker,
        SummaryStyle::Neutral,
    )
    .await
    {
        Some(s) => s,
        None => heuristic_summarize(&to_compress_refs),
    };

    // Replace compressed messages with the running summary (no disk archive).
    let after_split: Vec<Message> = messages[split_idx..].to_vec();
    messages.truncate(compress_start);
    messages.push(Message::user(&format!("{MARKER}\n{summary}")));
    messages.extend(after_split);

    emit_compaction_metric(
        "rolling_summary",
        total_tokens,
        history_token_total(messages),
        msgs_before,
        messages.len(),
    );
}

/// Pure truncation: drop the oldest turns, keep the most-recent turns within
/// budget. No summary, no LLM call, no archive — just a one-line marker so the
/// model knows history was elided (and to keep a clean user anchor).
fn compact_sliding_window(messages: &mut Vec<Message>, config: &Config, tool_def_tokens: usize) {
    let (raw_budget, _) = budgets(config, tool_def_tokens);

    let msg_tokens = per_msg_tokens(messages);
    let total_tokens: usize = msg_tokens.iter().sum();
    if total_tokens <= raw_budget {
        return;
    }

    let split_idx = find_split_idx(messages, &msg_tokens, raw_budget);
    if split_idx <= 1 {
        return;
    }
    let compress_start = first_history_idx(messages);
    if compress_start >= split_idx {
        return;
    }

    let msgs_before = messages.len();
    let after_split: Vec<Message> = messages[split_idx..].to_vec();
    messages.truncate(compress_start);
    messages.push(Message::user(
        "[Older conversation turns dropped to fit the context window.]",
    ));
    messages.extend(after_split);

    emit_compaction_metric(
        "sliding_window",
        total_tokens,
        history_token_total(messages),
        msgs_before,
        messages.len(),
    );
}

/// Observation masking: keep the full action trajectory (assistant messages,
/// tool calls, user turns) but replace old tool *observations* (results) with a
/// short placeholder, oldest-first, until back within budget — always keeping
/// the last `KEEP_RAW_OBS` observations in full. No LLM call.
fn compact_observation_masking(
    messages: &mut Vec<Message>,
    config: &Config,
    tool_def_tokens: usize,
) {
    const KEEP_RAW_OBS: usize = 3;
    const PLACEHOLDER: &str = "[earlier tool output elided to save context]";
    let (raw_budget, _) = budgets(config, tool_def_tokens);

    let total_tokens = history_token_total(messages);
    if total_tokens <= raw_budget {
        return;
    }

    // Tool-result message indices, oldest first.
    let tool_idxs: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter(|(_, m)| m.role == "tool")
        .map(|(i, _)| i)
        .collect();
    if tool_idxs.len() <= KEEP_RAW_OBS {
        return; // nothing old enough to mask
    }

    let maskable = &tool_idxs[..tool_idxs.len() - KEEP_RAW_OBS];
    let placeholder_tokens = estimate_tokens(PLACEHOLDER);
    let mut running = total_tokens;
    let mut masked_any = false;
    for &i in maskable {
        if running <= raw_budget {
            break;
        }
        let content = messages[i].content.as_deref().unwrap_or("");
        if content == PLACEHOLDER {
            continue; // already masked on a prior pass
        }
        let saved = msg_token_cost(&messages[i]).saturating_sub(placeholder_tokens);
        messages[i].content = Some(PLACEHOLDER.to_string());
        running = running.saturating_sub(saved);
        masked_any = true;
    }

    if !masked_any {
        return;
    }

    let msgs = messages.len();
    emit_compaction_metric(
        "observation_masking",
        total_tokens,
        history_token_total(messages),
        msgs,
        msgs, // masking preserves message count; only tool contents shrink
    );
}

/// Summarization prompt flavor. `Structured` is miniswe's production
/// per-file-changes format (anchored on actions); `Neutral` is the textbook
/// "summarize the conversation so far" prose used by the rolling-summary arm.
#[derive(Clone, Copy)]
enum SummaryStyle {
    Structured,
    Neutral,
}

/// Ask the LLM to summarize a timeline of messages into a narrative.
async fn llm_summarize_timeline(
    messages: &[&Message],
    existing_summary: &str,
    budget_tokens: usize,
    router: &ModelRouter,
    llm_worker: &LlmWorkerHandle,
    style: SummaryStyle,
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
        if is_summary_marker(content) {
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

    let (system_prompt, prompt) = match style {
        SummaryStyle::Structured => (
            "List completed actions, one per line. Include exact signatures when functions were changed. No explanation.",
            format!(
                "List WHAT you accomplished, one line per file changed. Use this format:\n\
                 - file.rs: what changed (include exact function signatures if modified)\n\
                 - file.rs: ✗ attempted but failed — reason\n\
                 End with: Still need: [what's left]\n\
                 Keep it under {budget_tokens} tokens. No process narrative.\n\n\
                 {timeline}"
            ),
        ),
        SummaryStyle::Neutral => (
            "You summarize an in-progress coding session so the assistant can continue from the summary alone. Be concise and faithful.",
            format!(
                "Summarize the conversation so far, preserving key decisions, the files \
                 changed and how, important findings, and what remains to be done. If a \
                 previous summary is given, update it to incorporate the new messages \
                 (do not drop still-relevant earlier facts). Keep it under {budget_tokens} \
                 tokens.\n\n\
                 {timeline}"
            ),
        ),
    };

    let request = ChatRequest {
        messages: vec![Message::system(system_prompt), Message::user(&prompt)],
        tools: None,
        tool_choice: None,
        max_tokens_override: None,
        chat_template_kwargs: Some(serde_json::json!({"enable_thinking": false})),
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
            if (content.contains("[read:") || content.starts_with("[src/"))
                && let Some(path) = content.split(':').nth(1).and_then(|s| s.split('→').next())
            {
                files_read.push(path.trim().to_string());
            }
            if (content.contains("✓ Edited") || content.contains("✓ Wrote"))
                && let Some(path) = content.split_whitespace().nth(2)
            {
                files_edited.push(path.to_string());
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
        summary.push_str("(earlier session activity — use file(action='read', path='.miniswe/session_archive.md') for details)");
    }

    summary
}

/// Archive compressed messages to `.miniswe/session_archive.md`.
fn archive_messages(messages: &[&Message], config: &Config) {
    let archive_path = config.miniswe_dir().join("session_archive.md");
    let mut archive = match std::fs::read_to_string(&archive_path) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => {
            // Read error on an existing archive — start from empty and warn.
            // The next atomic_write will overwrite the file, so the user
            // should know history was discarded before that happens.
            tracing::warn!(
                "Session archive read failed for {}; starting fresh and the next archive \
                 will overwrite the existing file: {e}",
                archive_path.display()
            );
            String::new()
        }
    };

    archive.push_str(&format!("\n## Compressed at round ~{}\n", messages.len()));
    for msg in messages {
        let role = &msg.role;
        let content = msg.content.as_deref().unwrap_or("");

        // Skip old summaries
        if content.starts_with("[Your earlier work") || content.starts_with("[Session summary") {
            continue;
        }

        // Full, untruncated content — the archive is the lossless record on
        // disk; the in-context summary is the lossy view.
        match role.as_str() {
            "assistant" => {
                if let Some(tcs) = &msg.tool_calls {
                    for tc in tcs {
                        archive.push_str(&format!(
                            "→ {}({})\n",
                            tc.function.name, tc.function.arguments
                        ));
                    }
                } else {
                    archive.push_str(&format!("ASSISTANT: {content}\n"));
                }
            }
            "tool" => {
                archive.push_str(&format!("RESULT: {content}\n"));
            }
            "user" => {
                archive.push_str(&format!("USER: {content}\n"));
            }
            _ => {}
        }
    }

    if let Err(e) = crate::atomic_write(&archive_path, archive.as_bytes()) {
        tracing::warn!(
            "Session archive write failed for {}: {e}",
            archive_path.display()
        );
    }
}

#[cfg(test)]
mod compaction_tests {
    use super::*;
    use crate::config::Config;

    // context_window=1200, tool_def_tokens=0 → available=1000, raw_budget=333.
    // A ~400-char message is ~100 tokens, so a handful of them blows the budget.
    fn cfg() -> Config {
        let mut c = Config::default();
        c.model.context_window = 1200;
        c
    }
    fn blob() -> String {
        "x".repeat(400) // 100 tokens
    }

    const SLIDING_MARKER: &str = "[Older conversation turns dropped to fit the context window.]";
    const OBS_PLACEHOLDER: &str = "[earlier tool output elided to save context]";

    #[test]
    fn sliding_window_drops_old_keeps_recent_and_marker() {
        let mut msgs = vec![Message::system("sys")];
        // 10 history messages of ~100 tokens each (total 1000 > raw_budget 333).
        for i in 0..10 {
            msgs.push(Message::user(&format!("{} msg{i}", blob())));
        }
        let newest_two: Vec<String> = msgs[msgs.len() - 2..]
            .iter()
            .map(|m| m.content.clone().unwrap())
            .collect();

        compact_sliding_window(&mut msgs, &cfg(), 0);

        // System preserved at front.
        assert_eq!(msgs[0].role, "system");
        // A single truncation marker, no summary, sits right after system.
        assert_eq!(msgs[1].role, "user");
        assert_eq!(msgs[1].content.as_deref(), Some(SLIDING_MARKER));
        // The newest turns are kept verbatim at the tail.
        let tail: Vec<String> = msgs[msgs.len() - 2..]
            .iter()
            .map(|m| m.content.clone().unwrap())
            .collect();
        assert_eq!(tail, newest_two, "newest turns must be preserved verbatim");
        // History is now within budget.
        assert!(history_token_total(&msgs) <= budgets(&cfg(), 0).0);
        // No LLM summary text leaked in.
        assert!(!msgs.iter().any(|m| {
            m.content
                .as_deref()
                .is_some_and(|c| c.starts_with("[Summary") || c.starts_with("[Your earlier"))
        }));
    }

    #[test]
    fn sliding_window_noop_under_budget() {
        let mut msgs = vec![
            Message::system("sys"),
            Message::user("hi"),
            Message::assistant("ok"),
        ];
        let before = msgs.clone();
        compact_sliding_window(&mut msgs, &cfg(), 0);
        assert_eq!(msgs.len(), before.len(), "under budget: no change");
    }

    #[test]
    fn observation_masking_elides_old_tools_keeps_last_three() {
        let mut msgs = vec![Message::system("sys")];
        // 6 (assistant tool-call, tool result) pairs. Tool results are large
        // (~100 tokens); assistant turns are tiny.
        for i in 0..6 {
            msgs.push(Message::assistant(&format!("call{i}")));
            msgs.push(Message::tool_result(
                &format!("id{i}"),
                &format!("{} out{i}", blob()),
            ));
        }
        let count_before = msgs.len();
        let tool_idxs: Vec<usize> = msgs
            .iter()
            .enumerate()
            .filter(|(_, m)| m.role == "tool")
            .map(|(i, _)| i)
            .collect();
        let last_three_raw: Vec<String> = tool_idxs[tool_idxs.len() - 3..]
            .iter()
            .map(|&i| msgs[i].content.clone().unwrap())
            .collect();

        compact_observation_masking(&mut msgs, &cfg(), 0);

        // Message count is preserved (trajectory intact); only tool contents shrink.
        assert_eq!(msgs.len(), count_before, "masking preserves message count");
        // The oldest tool observation is masked.
        assert_eq!(msgs[tool_idxs[0]].content.as_deref(), Some(OBS_PLACEHOLDER));
        // The last three observations are untouched.
        for (k, &i) in tool_idxs[tool_idxs.len() - 3..].iter().enumerate() {
            assert_eq!(
                msgs[i].content.as_ref().unwrap(),
                &last_three_raw[k],
                "last K observations must stay raw"
            );
        }
        // Assistant turns (the actions) are never masked.
        assert!(
            msgs.iter()
                .filter(|m| m.role == "assistant")
                .all(|m| m.content.as_deref().is_some_and(|c| c.starts_with("call")))
        );
    }

    #[test]
    fn observation_masking_noop_when_few_observations() {
        let mut msgs = vec![Message::system("sys")];
        for i in 0..3 {
            msgs.push(Message::assistant(&format!("call{i}")));
            msgs.push(Message::tool_result(&format!("id{i}"), &blob()));
        }
        let before = msgs.clone();
        compact_observation_masking(&mut msgs, &cfg(), 0);
        // Only 3 tool results (== KEEP_RAW_OBS) → nothing old enough to mask.
        for (a, b) in msgs.iter().zip(before.iter()) {
            assert_eq!(a.content, b.content, "≤ KEEP_RAW_OBS: untouched");
        }
    }
}
