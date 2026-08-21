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

/// Header line `compact_unified` writes in front of its summary. The
/// existing-summary search in `compact_unified` matches on this same constant
/// so writer and search can never drift apart again — they did once: the
/// search looked for "[Session summary", a header no writer has produced
/// since the outcome-focused-summaries rewrite (389bd4e), which silently
/// disabled carry-forward and dropped prior summary facts from context on
/// every compaction.
const UNIFIED_SUMMARY_HEADER: &str = "[Your earlier work in this session]";

/// Hard output cap (tokens) for the summarizer LLM call. A summary replaces a
/// handful of old messages; without this the call inherits the agent's full
/// `max_output_tokens` (8k) — enough for a repetition-loop runaway to GROW
/// context instead of shrinking it (nemotron 3.5: three runs in a row emitted
/// 25-36k-char fabricated changelogs, one growing history 16.8k→23k tokens).
const SUMMARY_MAX_TOKENS: u64 = 1024;

/// Recognize any in-context summary marker so the summarizer doesn't re-nest a
/// previous summary into a new one.
fn is_summary_marker(content: &str) -> bool {
    content.starts_with("[Your earlier work")
        || content.starts_with("[Session summary")
        || content.starts_with("[Summary of earlier conversation]")
}

/// Extract the carried-forward text from a previously injected summary
/// message: drop the header line and the trailing "[Details: …]" archive
/// pointer so only the summary content itself feeds the next summarization.
fn strip_summary_envelope(content: &str) -> String {
    content
        .lines()
        .filter(|l| !(is_summary_marker(l) || l.starts_with("[Details:")))
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
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
    // `elided_tokens` is clamped at 0 (existing parsers expect \d+), which
    // hides the pathological case where compaction GROWS context — the
    // signed `delta_tokens` field, appended so existing regexes keep
    // matching, is the one that can go negative.
    let delta = after_tokens as i64 - before_tokens as i64;
    eprintln!(
        "[compaction] strategy={strategy} before_tokens={before_tokens} \
         after_tokens={after_tokens} elided_tokens={elided} \
         msgs_before={msgs_before} msgs_after={msgs_after} delta_tokens={delta}"
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
    // Plan and scratchpad are no longer injected into the system prompt —
    // they're the only two pieces of context the agent itself mutates
    // mid-run, which made the system prompt go stale between refreshes
    // (see `config::ProvidersConfig`'s doc comment). Instead, the current
    // state is kept on the tail of the message list, refreshed here,
    // unconditionally, every time compression runs (i.e. every round).
    // This has to live in front of every compaction strategy's dispatch,
    // not be gated on "did plan/scratchpad change" or "is this strategy
    // about to actually do work": the risk isn't the content changing, it's
    // an unchanged block quietly aging until some strategy's cutoff reaches
    // past it and drops or summarizes it away. Refreshing every round keeps
    // it permanently at 0 rounds old, which every strategy here treats as
    // "too recent to touch" by construction.
    refresh_current_state(messages, config);

    match config.context.compaction {
        CompactionStrategy::Unified => {
            compact_unified(
                messages,
                config,
                router,
                llm_worker,
                tool_def_tokens,
                plan_update_requested,
                "unified",
            )
            .await
        }
        CompactionStrategy::RollingSummary => {
            compact_rolling_summary(
                messages,
                config,
                router,
                llm_worker,
                tool_def_tokens,
                "rolling_summary",
            )
            .await
        }
        CompactionStrategy::SlidingWindow => {
            compact_sliding_window(messages, config, tool_def_tokens)
        }
        CompactionStrategy::ObservationMasking => {
            compact_observation_masking(messages, config, tool_def_tokens)
        }
        // Tiered variants: mask → summary fallback. `rolling_cap` picks the
        // fallback (unified summary+archive vs rolling summary); TieredSmart
        // additionally adds a scratchpad nudge in build_system_prompt.
        CompactionStrategy::Tiered => {
            compact_tiered(
                messages,
                config,
                router,
                llm_worker,
                tool_def_tokens,
                plan_update_requested,
                "tiered",
                false,
            )
            .await
        }
        CompactionStrategy::TieredRolling => {
            compact_tiered(
                messages,
                config,
                router,
                llm_worker,
                tool_def_tokens,
                plan_update_requested,
                "tiered_rolling",
                true,
            )
            .await
        }
        CompactionStrategy::TieredSmart => {
            compact_tiered(
                messages,
                config,
                router,
                llm_worker,
                tool_def_tokens,
                plan_update_requested,
                "tiered_smart",
                false,
            )
            .await
        }
        // Reactive: never compact proactively. Compaction happens only via
        // `force_compress`, driven from the round loop when the server
        // itself signals context exhaustion. (refresh_current_state above
        // still ran — the current-state block stays on the tail regardless
        // of strategy.)
        CompactionStrategy::Lazy => {}
    }
}

/// Cap on consecutive `force_compress`-and-retry cycles for one failing
/// request. Two is enough for the realistic worst case (first pass frees
/// too little because one giant recent message survives the split; second
/// pass after the summary landed); beyond that, retrying is futile and the
/// original error should surface.
pub const FORCE_COMPRESS_MAX_RETRIES: usize = 2;

/// Estimated total prompt cost of a request as the round loop is about to
/// send it: every message (system included) plus the serialized tool
/// definitions. Used by the reactive-compaction gate to distinguish "the
/// model legitimately hit its output cap" from "the context window is
/// exhausted" — `finish_reason == "length"` alone can't tell these apart,
/// but a prompt sitting near the window can only mean the latter.
pub fn estimated_context_tokens(messages: &[Message], tool_def_tokens: usize) -> usize {
    messages.iter().map(msg_token_cost).sum::<usize>() + tool_def_tokens
}

/// Reactive compaction: the server itself signaled context exhaustion (a
/// rejected over-size request, or a generation truncated by the context
/// ceiling), so compact NOW and let the caller retry the round. Unlike
/// `maybe_compress` this bypasses `compact_unified`'s plan-update detour —
/// there is no "nudge first, compress next round" luxury when the request
/// can't even be sent — and `Lazy` (a no-op in `maybe_compress`) maps to
/// the `Unified` summary+archive action here.
///
/// Returns `true` if the message list actually shrank; callers must treat
/// `false` as "nothing could be freed" and fall through to their normal
/// error handling instead of retrying a request that will fail identically.
pub async fn force_compress(
    messages: &mut Vec<Message>,
    config: &Config,
    router: &ModelRouter,
    llm_worker: &LlmWorkerHandle,
    tool_def_tokens: usize,
) -> bool {
    let before = history_token_total(messages);
    // Bypass the plan-update nudge branch in the unified/tiered paths:
    // pretend the nudge already happened this cycle.
    let mut plan_nudge_done = true;
    match config.context.compaction {
        CompactionStrategy::Lazy | CompactionStrategy::Unified => {
            compact_unified(
                messages,
                config,
                router,
                llm_worker,
                tool_def_tokens,
                &mut plan_nudge_done,
                "lazy",
            )
            .await
        }
        CompactionStrategy::RollingSummary => {
            compact_rolling_summary(
                messages,
                config,
                router,
                llm_worker,
                tool_def_tokens,
                "lazy_rolling",
            )
            .await
        }
        CompactionStrategy::SlidingWindow => {
            compact_sliding_window(messages, config, tool_def_tokens)
        }
        CompactionStrategy::ObservationMasking => {
            compact_observation_masking(messages, config, tool_def_tokens)
        }
        CompactionStrategy::Tiered
        | CompactionStrategy::TieredRolling
        | CompactionStrategy::TieredSmart => {
            compact_tiered(
                messages,
                config,
                router,
                llm_worker,
                tool_def_tokens,
                &mut plan_nudge_done,
                "lazy_tiered",
                matches!(config.context.compaction, CompactionStrategy::TieredRolling),
            )
            .await
        }
    }
    history_token_total(messages) < before
}

/// Marker prefixing the current-state block appended to the tail of the
/// message list every round. Lets `refresh_current_state` find-and-strip
/// whatever it previously appended before appending a fresh copy, so at
/// most one live copy exists at a time, always on the last message.
const CURRENT_STATE_MARKER: &str = "\n\n[CURRENT STATE]\n";

/// Build the current-state block (plan + scratchpad), or `None` if both are
/// empty.
fn format_current_state_block(config: &Config) -> Option<String> {
    let plan = crate::tools::plan::load_plan(config);
    let scratchpad = std::fs::read_to_string(config.miniswe_path("scratchpad.md"))
        .ok()
        .filter(|s| !s.trim().is_empty());
    // Just-in-time skill-step re-inject: when a skill cursor is active,
    // append the CURRENT step's distilled instructions so the model executes
    // from the manual, not from priors (uds-mcp e2e: it drifted after one
    // read otherwise). One step at a time — nothing to rubber-stamp.
    // Computed BEFORE the empty-state check: an active cursor alone must
    // produce a block — the routed task points the model at [SKILL STEP]
    // from round 1, before any plan or scratchpad exists.
    let step_block = crate::cli::commands::agent::skill_cursor::active_step_block(config);
    if plan.is_none() && scratchpad.is_none() && step_block.is_none() {
        return None;
    }
    let mut block = String::from(CURRENT_STATE_MARKER);
    if let Some(p) = plan {
        block.push_str("[PLAN]\n");
        block.push_str(p.trim_end());
        block.push('\n');
    }
    if let Some(s) = scratchpad {
        block.push_str("[SCRATCHPAD]\n");
        block.push_str(s.trim_end());
        block.push('\n');
    }
    if let Some(step_block) = step_block {
        block.push_str(&step_block);
    }
    Some(block)
}

/// Strip any previously-appended current-state block from `messages`
/// (wherever it landed), then append a fresh one to the last message.
/// Unconditional and called every round (see `maybe_compress`'s call site)
/// so the block is never more than 0 rounds old — every compaction
/// strategy here treats the most recent message as too new to touch, so a
/// block re-attached every round can never be dropped or summarized away.
///
/// Appending to an EXISTING message's content (rather than inserting a new
/// message) is role-order-safe by construction: it never introduces a new
/// role transition, so it can't break strict chat templates.
fn refresh_current_state(messages: &mut [Message], config: &Config) {
    for m in messages.iter_mut() {
        if let Some(content) = &m.content
            && let Some(pos) = content.find(CURRENT_STATE_MARKER)
        {
            m.content = Some(content[..pos].to_string());
        }
    }
    let Some(block) = format_current_state_block(config) else {
        return;
    };
    if let Some(last) = messages.last_mut() {
        let content = last.content.get_or_insert_with(String::new);
        content.push_str(&block);
    }
}

#[cfg(test)]
mod current_state_tests {
    use super::{CURRENT_STATE_MARKER, format_current_state_block, refresh_current_state};
    use crate::config::Config;
    use crate::llm::Message;

    fn config_in(dir: &std::path::Path) -> Config {
        std::fs::create_dir_all(dir.join(".miniswe")).unwrap();
        let mut config = Config::default();
        config.project_root = dir.to_path_buf();
        config
    }

    #[test]
    fn no_block_when_both_empty() {
        let tmp = tempfile::TempDir::new().unwrap();
        let config = config_in(tmp.path());
        assert!(format_current_state_block(&config).is_none());
    }

    #[test]
    fn active_skill_cursor_alone_produces_a_block() {
        // Regression: the empty-state early return must not swallow the
        // [SKILL STEP] injection — before the model writes a plan or
        // scratchpad, the cursor is the only guidance the routed task
        // points at.
        use crate::cli::commands::agent::skill_cursor::{self, SkillCursor};
        use crate::cli::commands::agent::skill_router::SkillStep;
        let tmp = tempfile::TempDir::new().unwrap();
        let config = config_in(tmp.path());
        let mut cursor = SkillCursor::default();
        cursor.push_skill(
            "uds-package",
            tmp.path(),
            vec![SkillStep {
                name: "Create the package".into(),
                anchor: "## Create".into(),
            }],
        );
        skill_cursor::save(&config, &cursor);

        let block = format_current_state_block(&config).expect("cursor alone must produce a block");
        assert!(block.contains("[SKILL STEP]"), "{block}");
        assert!(block.contains("uds-package"), "{block}");
        assert!(!block.contains("[PLAN]"), "{block}");
    }

    #[test]
    fn appends_to_last_message_when_state_exists() {
        let tmp = tempfile::TempDir::new().unwrap();
        let config = config_in(tmp.path());
        std::fs::write(config.miniswe_path("plan.md"), "1. step one\n").unwrap();

        let mut msgs = vec![Message::user("do the task"), Message::assistant("ok")];
        refresh_current_state(&mut msgs, &config);

        let content = msgs.last().unwrap().content.as_deref().unwrap();
        assert!(content.contains("[CURRENT STATE]"));
        assert!(content.contains("[PLAN]"));
        assert!(content.contains("step one"));
        assert!(
            content.starts_with("ok"),
            "original content preserved: {content}"
        );
    }

    #[test]
    fn replaces_old_block_instead_of_accumulating() {
        let tmp = tempfile::TempDir::new().unwrap();
        let config = config_in(tmp.path());
        std::fs::write(config.miniswe_path("plan.md"), "1. step one\n").unwrap();

        let mut msgs = vec![Message::tool_result("call1", "first result")];
        refresh_current_state(&mut msgs, &config);
        assert_eq!(
            msgs[0]
                .content
                .as_deref()
                .unwrap()
                .matches("[PLAN]")
                .count(),
            1
        );

        // A later round: a fresh tool result gets pushed, plan changes.
        std::fs::write(config.miniswe_path("plan.md"), "1. [x] step one\n").unwrap();
        msgs.push(Message::tool_result("call2", "second result"));
        refresh_current_state(&mut msgs, &config);

        // Old block gone from msgs[0], new one only on the last message.
        assert!(
            !msgs[0]
                .content
                .as_deref()
                .unwrap()
                .contains(CURRENT_STATE_MARKER)
        );
        assert_eq!(msgs[0].content.as_deref().unwrap(), "first result");
        let last_content = msgs.last().unwrap().content.as_deref().unwrap();
        assert!(last_content.contains("[x] step one"));
        assert_eq!(last_content.matches("[PLAN]").count(), 1);
    }

    #[test]
    fn refreshes_every_round_even_when_state_is_unchanged() {
        // The bug this guards against: a block parked on an old message and
        // never moved would eventually be compacted away. Simulate several
        // "rounds" with unchanged plan content and confirm the block always
        // ends up on the newest message, never left behind on an old one.
        let tmp = tempfile::TempDir::new().unwrap();
        let config = config_in(tmp.path());
        std::fs::write(config.miniswe_path("plan.md"), "1. step one\n").unwrap();

        let mut msgs = vec![Message::tool_result("call1", "round 1 result")];
        refresh_current_state(&mut msgs, &config);

        for i in 2..=5 {
            msgs.push(Message::tool_result(
                &format!("call{i}"),
                &format!("round {i} result"),
            ));
            refresh_current_state(&mut msgs, &config); // unchanged plan content
        }

        for m in &msgs[..msgs.len() - 1] {
            assert!(
                !m.content.as_deref().unwrap().contains(CURRENT_STATE_MARKER),
                "block should not linger on an old message: {:?}",
                m.content
            );
        }
        let last_content = msgs.last().unwrap().content.as_deref().unwrap();
        assert!(last_content.contains("[PLAN]"));
    }

    #[test]
    fn no_op_when_no_state_and_nothing_to_strip() {
        let tmp = tempfile::TempDir::new().unwrap();
        let config = config_in(tmp.path());
        let mut msgs = vec![Message::tool_result("call1", "a result")];
        refresh_current_state(&mut msgs, &config);
        assert_eq!(msgs[0].content.as_deref().unwrap(), "a result");
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
    label: &str,
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
                    .is_some_and(|c| c.starts_with(UNIFIED_SUMMARY_HEADER))
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
        .map(|c| strip_summary_envelope(&c))
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
        "{UNIFIED_SUMMARY_HEADER}\n{summary}\n[Details: file(action='read', path='.miniswe/session_archive.md'). Continue from where you left off.]"
    )));

    messages.extend(after_split);

    emit_compaction_metric(
        label,
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
    label: &str,
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
        label,
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

/// Placeholder swapped in for an elided old tool observation.
const OBS_PLACEHOLDER: &str = "[earlier tool output elided to save context]";
/// Number of most-recent observations always kept raw.
const KEEP_RAW_OBS: usize = 3;

/// Markers identifying corrective/guard tool results that must NEVER be
/// masked. These carry loop-breaking guidance (auto-revert warnings,
/// post-revert smallest-edit hints, loop-detector interventions) — masking
/// them within a round or two of appearing was a direct contributor to fatal
/// edit↔revert oscillations in the 2026-07-03 bench: the model lost both the
/// medicine and the evidence it had already tried the same edit dozens of
/// times. Keep the list narrow — these messages are short, so exempting them
/// costs little budget.
const GUARD_MARKERS: &[&str] = &[
    "[auto-revert]",
    "[hint]",
    "You are in a loop",
    "You are in an edit↔revert loop",
    "same read/inspection call",
    // replace_range's no-op rejection (tools/fast/replace_range.rs). Its
    // loop-breaking power comes from the visible pile of repeated rejections,
    // not just the newest one — the 2026-07-15 warm-replay probe showed the
    // reworded guard escapes 7/8 with full history but 1/8 once masking eats
    // older rejections (KEEP_RAW_OBS alone doesn't protect them: rejections
    // interleave with reads and fall out of the newest-3 window).
    "already match the content you provided",
];

/// Guard exemption size cap. Guard texts ride on edit/revert results that
/// also carry the revisions table + LSP feedback (~1.5-2.5k chars total); a
/// much larger tool result containing a marker is almost certainly a file
/// READ of source code that contains the marker string as a literal (this
/// codebase does) — those must remain maskable or one read bloats the
/// window forever.
const GUARD_MAX_CHARS: usize = 4000;

/// True if this tool-result content carries corrective/guard guidance that
/// must survive observation masking.
fn is_guard_observation(content: &str) -> bool {
    content.len() <= GUARD_MAX_CHARS && GUARD_MARKERS.iter().any(|m| content.contains(m))
}

/// Mask old tool observations (oldest-first) down to [`OBS_PLACEHOLDER`] until
/// within `raw_budget`, always keeping the last [`KEEP_RAW_OBS`] raw and never
/// masking guard observations (see [`GUARD_MARKERS`]). Mutates in place;
/// returns true if it masked anything. Shared by `observation_masking` and the
/// tiered hybrid's cheap first tier.
fn mask_old_observations(messages: &mut [Message], raw_budget: usize) -> bool {
    let tool_idxs: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter(|(_, m)| m.role == "tool")
        .map(|(i, _)| i)
        .collect();
    if tool_idxs.len() <= KEEP_RAW_OBS {
        return false; // nothing old enough to mask
    }
    let maskable = &tool_idxs[..tool_idxs.len() - KEEP_RAW_OBS];
    let placeholder_tokens = estimate_tokens(OBS_PLACEHOLDER);
    let mut running = history_token_total(messages);
    let mut masked_any = false;
    for &i in maskable {
        if running <= raw_budget {
            break;
        }
        let content = messages[i].content.as_deref().unwrap_or("");
        if content == OBS_PLACEHOLDER {
            continue; // already masked on a prior pass
        }
        if is_guard_observation(content) {
            continue; // corrective guidance — must stay visible
        }
        let saved = msg_token_cost(&messages[i]).saturating_sub(placeholder_tokens);
        messages[i].content = Some(OBS_PLACEHOLDER.to_string());
        running = running.saturating_sub(saved);
        masked_any = true;
    }
    masked_any
}

/// Observation masking: keep the full action trajectory (assistant messages,
/// tool calls, user turns) but replace old tool *observations* (results) with a
/// short placeholder, oldest-first, until back within budget — always keeping
/// the last [`KEEP_RAW_OBS`] observations in full. No LLM call.
fn compact_observation_masking(
    messages: &mut Vec<Message>,
    config: &Config,
    tool_def_tokens: usize,
) {
    let (raw_budget, _) = budgets(config, tool_def_tokens);
    let total_tokens = history_token_total(messages);
    if total_tokens <= raw_budget {
        return;
    }
    if !mask_old_observations(messages, raw_budget) {
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

/// Tiered hybrid: mask old observations first (cheap, free); only if that
/// doesn't get under budget, fall through to the `Unified` summary + archive
/// (the hard cap). Avoids observation-masking's edit-heavy thrash — when the
/// bulk is in tool-call args (edit bodies) that masking can't touch, the summary
/// tier collapses them — while staying cheap when observations dominate.
///
/// `label` distinguishes the flavor in the metric stream. `rolling_cap` selects
/// the tier-2 fallback: `false` → `unified` (summary + disk archive), `true` →
/// `rolling_summary` (running summary, no archive).
#[allow(clippy::too_many_arguments)]
async fn compact_tiered(
    messages: &mut Vec<Message>,
    config: &Config,
    router: &ModelRouter,
    llm_worker: &LlmWorkerHandle,
    tool_def_tokens: usize,
    plan_update_requested: &mut bool,
    label: &str,
    rolling_cap: bool,
) {
    let (raw_budget, _) = budgets(config, tool_def_tokens);
    let before_tokens = history_token_total(messages);
    if before_tokens <= raw_budget {
        return;
    }

    // Tier 1: mask old observations (free, preserves the action trace).
    let masked = mask_old_observations(messages, raw_budget);

    if history_token_total(messages) <= raw_budget {
        // Masking alone capped it — no LLM call needed.
        if masked {
            let msgs = messages.len();
            emit_compaction_metric(
                label,
                before_tokens,
                history_token_total(messages),
                msgs,
                msgs,
            );
        }
        return;
    }

    // Tier 2: still over budget (e.g. edit bodies dominate) → a summary tier
    // caps it. The fallback emits its own (post-mask) metric under `label`.
    if rolling_cap {
        compact_rolling_summary(messages, config, router, llm_worker, tool_def_tokens, label).await;
    } else {
        compact_unified(
            messages,
            config,
            router,
            llm_worker,
            tool_def_tokens,
            plan_update_requested,
            label,
        )
        .await;
    }
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
    // The stated budget and the hard cap must agree — asking for "under
    // 11k tokens" while capping at 1k invites truncated-mid-line output.
    let budget_tokens = budget_tokens.min(SUMMARY_MAX_TOKENS as usize);

    let mut timeline = String::new();
    if !existing_summary.is_empty() {
        timeline.push_str(&format!("Previous summary:\n{existing_summary}\n\n"));
    }
    timeline.push_str("New messages to incorporate:\n");
    let timeline_header_len = timeline.len();

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

    // Empty window: every message was skipped (e.g. the window held only a
    // previous summary marker). Asking an LLM to "list what you
    // accomplished" over nothing coerces confabulation — probed live on
    // nemotron: an empty timeline yielded a fully fabricated changelog
    // (invented vm.rs/lexer.rs/parser.rs) 2/2 times. Skip the call: carry
    // the existing summary forward unchanged, or signal the caller to use
    // the heuristic fallback.
    if timeline.len() == timeline_header_len {
        eprintln!("[compressor] empty summarize window — skipping LLM call");
        if !existing_summary.is_empty() {
            return Some(existing_summary.to_string());
        }
        return None;
    }

    let (system_prompt, prompt) = match style {
        SummaryStyle::Structured => (
            "List completed actions, one per line. Include exact signatures when functions were changed. No explanation.",
            format!(
                "List WHAT you accomplished, one line per file changed. Use this format:\n\
                 - file.rs: what changed (include exact function signatures if modified)\n\
                 - file.rs: ✗ attempted but failed — reason\n\
                 End with: Still need: [what's left]\n\
                 Keep it under {budget_tokens} tokens. No process narrative.\n\
                 If no files were changed in these messages, output exactly: No completed actions.\n\n\
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
        // Hard cap, not just the prompt's "keep it under N" ask — a
        // repetition-looping model ignores the ask and burns the full
        // agent-level output budget otherwise.
        max_tokens_override: Some(SUMMARY_MAX_TOKENS),
        chat_template_kwargs: Some(serde_json::json!({"enable_thinking": false})),
        cache_prompt: None,
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
    // Degenerate-summary guard: a summary REPLACES `messages` (the old
    // summary message included), so one at least as large as its input is
    // never compression — it's a runaway (repetition loop / fabricated
    // changelog). Reject it; the caller falls back to heuristic_summarize,
    // which only extracts from real tool results and cannot invent.
    let input_tokens: usize = messages.iter().map(|m| msg_token_cost(m)).sum();
    let summary_tokens = estimate_tokens(text);
    if summary_tokens >= input_tokens {
        eprintln!(
            "[compressor] rejected degenerate summary ({summary_tokens} tokens >= \
             {input_tokens}-token input) — falling back to heuristic"
        );
        return None;
    }
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
    // OBS_PLACEHOLDER comes from super::*

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

    #[test]
    fn mask_helper_returns_true_only_when_it_masks() {
        // The tiered hybrid uses this bool to decide tier-1-only vs tier-2.
        // raw_budget at the test cfg ≈ 333; 6×100-tok tool results blow it.
        let mut msgs = vec![Message::system("sys")];
        for i in 0..6 {
            msgs.push(Message::assistant(&format!("call{i}")));
            msgs.push(Message::tool_result(&format!("id{i}"), &blob()));
        }
        let raw_budget = budgets(&cfg(), 0).0;
        assert!(
            mask_old_observations(&mut msgs, raw_budget),
            "should mask when >KEEP_RAW_OBS observations exceed budget"
        );
        // Idempotent-ish: a second pass with everything maskable already masked
        // (and now under budget) masks nothing more.
        assert!(
            !mask_old_observations(&mut msgs, raw_budget),
            "nothing left to mask on the second pass"
        );

        // Too few observations → never masks (tier-1 is a no-op, tier-2 decides).
        let mut few = vec![Message::system("sys")];
        for i in 0..3 {
            few.push(Message::tool_result(&format!("id{i}"), &blob()));
        }
        assert!(!mask_old_observations(&mut few, 1));
    }

    #[test]
    fn guard_observations_survive_masking() {
        let guard = format!(
            "revert f.rs → rev_3: restored\n[hint] Restored to a parsing state. \
             Make the SMALLEST possible edit.\n{}",
            "x".repeat(200)
        );
        let mut msgs = vec![Message::system("sys")];
        // Old guard result first, then plenty of plain old observations.
        msgs.push(Message::tool_result("id-guard", &guard));
        for i in 0..8 {
            msgs.push(Message::assistant(&format!("call{i}")));
            msgs.push(Message::tool_result(&format!("id{i}"), &blob()));
        }
        // Budget of 1 forces masking of every maskable message.
        assert!(mask_old_observations(&mut msgs, 1));
        let guard_msg = &msgs[1];
        assert_eq!(
            guard_msg.content.as_deref(),
            Some(guard.as_str()),
            "guard observation must never be masked"
        );
        // The plain old observations (all but the last KEEP_RAW_OBS) got masked.
        let masked = msgs
            .iter()
            .filter(|m| m.content.as_deref() == Some(OBS_PLACEHOLDER))
            .count();
        assert_eq!(
            masked,
            8 - KEEP_RAW_OBS,
            "non-guard old observations masked"
        );
    }

    #[test]
    fn oversized_marker_carrier_is_still_masked() {
        // A file READ of source code containing a marker literal must remain
        // maskable — only short, genuine guard messages are exempt.
        let big_read = format!("[auto-revert] as a source literal\n{}", "x".repeat(5000));
        let mut msgs = vec![Message::system("sys")];
        msgs.push(Message::tool_result("id-big", &big_read));
        for i in 0..8 {
            msgs.push(Message::assistant(&format!("call{i}")));
            msgs.push(Message::tool_result(&format!("id{i}"), &blob()));
        }
        assert!(mask_old_observations(&mut msgs, 1));
        assert_eq!(
            msgs[1].content.as_deref(),
            Some(OBS_PLACEHOLDER),
            "oversized marker-carrying read must be masked"
        );
    }

    #[test]
    fn is_guard_observation_matches_the_real_guard_texts() {
        assert!(is_guard_observation(
            "[auto-revert] Your last 3 edits to f.rs EACH left the syntax tree broken"
        ));
        assert!(is_guard_observation(
            "revert f.rs → rev_0: restored\n[hint] Restored to a parsing state."
        ));
        assert!(is_guard_observation(
            "ERROR: You are in a loop — this exact tool call has been repeated 3 times"
        ));
        assert!(is_guard_observation(
            "ERROR: You are in an edit↔revert loop — you have alternated between the SAME two tool calls"
        ));
        assert!(is_guard_observation(
            "You just made this same read/inspection call 3 times in a row."
        ));
        assert!(is_guard_observation(
            "replace_range: lines L36-45 of chart/values.yaml already match the content you \
             provided — nothing changed. The file ALREADY contains exactly this text."
        ));
        assert!(!is_guard_observation("[file] src/main.rs: 40 lines"));
    }
}

#[cfg(test)]
mod force_compress_tests {
    use super::{FORCE_COMPRESS_MAX_RETRIES, estimated_context_tokens, force_compress};
    use crate::config::{CompactionStrategy, Config};
    use crate::llm::{Message, ModelRouter};
    use crate::runtime::LlmWorkerHandle;
    use std::sync::Arc;

    /// Test config rooted in a temp dir. The endpoint points at a port
    /// nothing listens on and retries are disabled, so any LLM-based
    /// summarization fails instantly (connection refused) and falls back to
    /// the heuristic summary — tests never touch a live server.
    fn config_in(dir: &std::path::Path, strategy: CompactionStrategy) -> Config {
        std::fs::create_dir_all(dir.join(".miniswe")).unwrap();
        let mut config = Config::default();
        config.project_root = dir.to_path_buf();
        config.model.endpoint = "http://127.0.0.1:9".into();
        config.model.max_retries = 0;
        config.model.context_window = 60_000;
        config.context.compaction = strategy;
        config
    }

    /// A message list far over `raw_budget` (~16.7K tokens at a 60K window
    /// with tool_def_tokens=0): 20 tool results of 5K chars ≈ 25K tokens.
    fn over_budget_messages() -> Vec<Message> {
        let mut msgs = vec![
            Message::system("You are miniswe."),
            Message::user("do the task"),
        ];
        for i in 0..20 {
            msgs.push(Message::assistant(&format!("reading file {i}")));
            msgs.push(Message::tool_result(
                &format!("call{i}"),
                &"line of tool output\n".repeat(250),
            ));
        }
        msgs
    }

    #[tokio::test]
    async fn lazy_is_a_no_op_in_maybe_compress() {
        let tmp = tempfile::TempDir::new().unwrap();
        let config = config_in(tmp.path(), CompactionStrategy::Lazy);
        let router = Arc::new(ModelRouter::new(&config));
        let worker = LlmWorkerHandle::new(router.clone(), 1);

        let mut messages = over_budget_messages();
        let before_tokens = estimated_context_tokens(&messages, 0);
        let before_len = messages.len();

        let mut plan_flag = false;
        super::maybe_compress(&mut messages, &config, &router, &worker, 0, &mut plan_flag).await;

        // Far over budget, yet nothing was compacted — Lazy never fires
        // proactively. (No plan/scratchpad exists in the temp project, so
        // the current-state refresh is also a no-op here.)
        assert_eq!(messages.len(), before_len);
        assert_eq!(estimated_context_tokens(&messages, 0), before_tokens);
    }

    #[tokio::test]
    async fn force_compress_lazy_shrinks_via_unified_path() {
        let tmp = tempfile::TempDir::new().unwrap();
        let config = config_in(tmp.path(), CompactionStrategy::Lazy);
        let router = Arc::new(ModelRouter::new(&config));
        let worker = LlmWorkerHandle::new(router.clone(), 1);

        let mut messages = over_budget_messages();
        let before_tokens = estimated_context_tokens(&messages, 0);

        let freed = force_compress(&mut messages, &config, &router, &worker, 0).await;

        // The LLM summarizer can't be reached (dead endpoint, 0 retries) —
        // the heuristic fallback must still compact.
        assert!(freed, "force_compress should report freed tokens");
        assert!(
            estimated_context_tokens(&messages, 0) < before_tokens,
            "history should shrink"
        );
        // The unified path archives what it elided.
        assert!(config.miniswe_path("session_archive.md").exists());
    }

    #[tokio::test]
    async fn force_compress_sliding_window_shrinks_without_llm() {
        let tmp = tempfile::TempDir::new().unwrap();
        let config = config_in(tmp.path(), CompactionStrategy::SlidingWindow);
        let router = Arc::new(ModelRouter::new(&config));
        let worker = LlmWorkerHandle::new(router.clone(), 1);

        let mut messages = over_budget_messages();
        let before_tokens = estimated_context_tokens(&messages, 0);

        let freed = force_compress(&mut messages, &config, &router, &worker, 0).await;

        assert!(freed);
        assert!(estimated_context_tokens(&messages, 0) < before_tokens);
    }

    #[tokio::test]
    async fn force_compress_reports_false_when_nothing_to_free() {
        let tmp = tempfile::TempDir::new().unwrap();
        let config = config_in(tmp.path(), CompactionStrategy::Lazy);
        let router = Arc::new(ModelRouter::new(&config));
        let worker = LlmWorkerHandle::new(router.clone(), 1);

        // Tiny history — nothing old enough to compact. Callers rely on
        // `false` here to avoid resending a request that will fail
        // identically.
        let mut messages = vec![
            Message::system("You are miniswe."),
            Message::user("hi"),
            Message::assistant("hello"),
        ];
        let freed = force_compress(&mut messages, &config, &router, &worker, 0).await;
        assert!(!freed);
        assert_eq!(messages.len(), 3);
    }

    #[test]
    fn retry_cap_is_small_and_nonzero() {
        // The cap bounds consecutive futile retries of ONE failing request;
        // it must allow at least one retry and stay small enough that a
        // truly unfixable request fails fast.
        assert!((1..=3).contains(&FORCE_COMPRESS_MAX_RETRIES));
    }

    #[test]
    fn estimated_context_tokens_counts_system_and_tools() {
        let messages = vec![
            Message::system(&"s".repeat(400)), // ~100 tokens
            Message::user(&"u".repeat(400)),   // ~100 tokens
        ];
        let with_tools = estimated_context_tokens(&messages, 500);
        let without_tools = estimated_context_tokens(&messages, 0);
        assert_eq!(with_tools - without_tools, 500);
        // System message IS counted — unlike needs_compression's history
        // total, this estimates the full prompt as the server sees it.
        assert!(without_tools >= 200);
    }

    #[test]
    fn strip_summary_envelope_keeps_only_content() {
        let injected = format!(
            "{}\n- run.rs: threaded the new param\n- mod.rs: added flag\n\
             [Details: file(action='read', path='.miniswe/session_archive.md'). Continue from where you left off.]",
            super::UNIFIED_SUMMARY_HEADER
        );
        let stripped = super::strip_summary_envelope(&injected);
        assert_eq!(
            stripped,
            "- run.rs: threaded the new param\n- mod.rs: added flag"
        );
    }

    #[tokio::test]
    async fn unified_writes_the_marker_its_search_looks_for() {
        // Regression guard for the writer/search drift that silently killed
        // carry-forward for months: the message compact_unified injects must
        // start with the exact prefix its existing-summary search matches on.
        let tmp = tempfile::TempDir::new().unwrap();
        let config = config_in(tmp.path(), CompactionStrategy::Lazy);
        let router = Arc::new(ModelRouter::new(&config));
        let worker = LlmWorkerHandle::new(router.clone(), 1);

        let mut messages = over_budget_messages();
        force_compress(&mut messages, &config, &router, &worker, 0).await;

        let summary_msg = messages
            .iter()
            .find(|m| {
                m.role == "user"
                    && m.content
                        .as_deref()
                        .is_some_and(|c| c.starts_with(super::UNIFIED_SUMMARY_HEADER))
            })
            .expect("compact_unified should inject a summary the search can find");
        assert!(super::is_summary_marker(
            summary_msg.content.as_deref().unwrap()
        ));
    }

    #[tokio::test]
    async fn empty_summarize_window_short_circuits_without_llm() {
        // A window holding only a previous summary marker builds an empty
        // timeline; the summarizer must not ask an LLM to "list what you
        // accomplished" over nothing (probed on nemotron: 2/2 fabricated
        // changelogs). With an existing summary it is carried forward
        // verbatim; without one the caller falls back to the heuristic.
        let tmp = tempfile::TempDir::new().unwrap();
        let config = config_in(tmp.path(), CompactionStrategy::Lazy);
        let router = Arc::new(ModelRouter::new(&config));
        let worker = LlmWorkerHandle::new(router.clone(), 1);

        let blob = format!("{}\nold facts", super::UNIFIED_SUMMARY_HEADER);
        let only_marker = [Message::user(&blob)];
        let refs: Vec<&Message> = only_marker.iter().collect();

        // The dead endpoint (config_in) would return None if the LLM path
        // were reached; getting the existing summary back proves the
        // short-circuit fired before any request.
        let carried = super::llm_summarize_timeline(
            &refs,
            "earlier facts",
            1000,
            &router,
            &worker,
            super::SummaryStyle::Structured,
        )
        .await;
        assert_eq!(carried.as_deref(), Some("earlier facts"));

        let none = super::llm_summarize_timeline(
            &refs,
            "",
            1000,
            &router,
            &worker,
            super::SummaryStyle::Structured,
        )
        .await;
        assert_eq!(none, None);
    }
}
