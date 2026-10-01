//! Per-turn explore/coding intent router and the read-only EXPLORE
//! tool surface.

use super::*;

/// Per-turn intent router (REPL only). One isolated, tool-less LLM
/// round-trip — NOT added to conversation history — classifying the
/// user's message as code-editing vs read-only investigation.
///
/// Biased toward EXPLORE for *questions* (the prompt routes
/// explain/summarize/why/what/where/how to EXPLORE) so a plain question
/// doesn't tip into execution; CODING is reserved for a clear instruction
/// to change code. The one hard safety rule kept from the original: a
/// parse failure / empty / LLM error → `false` (CODING), so on genuine
/// uncertainty we never silently swallow an edit request into read-only
/// mode. Normal-path bias is the prompt's job; the error-path default is
/// CODING.
pub(super) async fn classify_is_explore(
    llm_worker: &LlmWorkerHandle,
    user_message: &str,
    cancelled: &Arc<AtomicBool>,
) -> bool {
    // Artifact-mutation framing (validated 2026-06-30 vs gemma, isolated
    // battery in scripts/classifier-prompt-probe family, 27 cases × 3 reps):
    // key on whether the user wants an artifact created/altered/dropped vs a
    // question answered. This generalizes to descriptive phrasings the earlier
    // verb-based prompt ("tells you to change/add/fix code") misrouted to
    // EXPLORE — "create an app …", "I want a CLI that …", "… sort it out" were
    // all 3/3 EXPLORE under the old prompt and are 0/17 dangerous under this
    // one, with 0/10 EXPLORE-side regressions. Don't trim without re-running.
    let sys = "Reply one word: CODING or EXPLORE. \
        Determine if this is a pure exploration task that leads to answering a \
        question (reply EXPLORE), or a task that creates, alters, or drops an \
        artifact — file, app, project, feature, etc. (reply CODING). \
        Default EXPLORE.";
    let request = ChatRequest {
        messages: vec![Message::system(sys), Message::user(user_message)],
        tools: None,
        tool_choice: None,
        max_tokens_override: Some(8),
        chat_template_kwargs: Some(serde_json::json!({"enable_thinking": false})),
        temperature_override: None,
        cache_prompt: None,
    };
    let mut events = llm_worker.submit(ModelRole::Default, request, cancelled.clone());
    let mut out = String::new();
    while let Some(ev) = events.recv().await {
        match ev {
            LlmWorkerEvent::Completed(Ok(r)) => {
                out = r
                    .choices
                    .first()
                    .and_then(|c| c.message.content.clone())
                    .unwrap_or_default();
                break;
            }
            LlmWorkerEvent::Completed(Err(_)) => break, // fail-safe → CODING
            _ => {}
        }
    }
    is_explore_reply(&out)
}

/// Fail-safe classifier parse: EXPLORE only on a clean leading
/// EXPLORE; everything else (incl. empty, prose-wrapped, CODING) →
/// false = CODING. The asymmetric bias is the safety property.
pub(super) fn is_explore_reply(s: &str) -> bool {
    s.trim().to_ascii_uppercase().starts_with("EXPLORE")
}

/// Read-only tool subset for EXPLORE turns: drop every writer/mutator
/// (so the model is never *offered* an edit tool) and the plan tool
/// (no planning in Q&A). Keeps file:read/search, code:* (LSP/repo
/// map), web, show_rev/check.
pub(super) fn read_only_tool_defs(
    all: &[crate::llm::ToolDefinition],
) -> Vec<crate::llm::ToolDefinition> {
    all.iter()
        .filter(|t| {
            let n = t.function.name.as_str();
            !(is_file_write(n) || matches!(n, "revert" | "delete_file" | "plan" | "spawn_agents"))
        })
        .cloned()
        .collect()
}

// `shell_is_read_only` / `explore_block_reason` moved to
// `agent::explore_gate` (shared with `admit`'s read-only gate); re-imported
// into `repl/mod.rs`'s glob namespace so call sites and tests here are
// unaffected.
