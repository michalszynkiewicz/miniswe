//! Real-history check for the read pruner, gated on `PRUNE_FIXTURE` pointing
//! at an `llm_dumps/req-*.json` from a bench run (the loop this was built for:
//! `benchmark_results/docker_20260826_163007__*Muse-Glimmer*/00_baseline/llm_dumps`).
//!
//! Synthetic tests can't prove the pruner survives a real chat template — the
//! invariant that matters is that every surviving `tool` result still has its
//! assistant `tool_calls` message immediately before it, on history the model
//! actually produced.

use miniswe::cli::commands::agent::prune_reads::prune_repeated_reads;
use miniswe::llm::Message;

#[test]
fn prunes_a_real_read_loop_without_orphaning_results() {
    let Ok(path) = std::env::var("PRUNE_FIXTURE") else {
        eprintln!("PRUNE_FIXTURE unset — skipping");
        return;
    };
    let raw = std::fs::read_to_string(&path).expect("fixture unreadable");
    let dump: serde_json::Value = serde_json::from_str(&raw).expect("fixture is not JSON");
    let mut messages: Vec<Message> =
        serde_json::from_value(dump["messages"].clone()).expect("messages did not deserialize");

    // Real histories can already contain a stranded result: compaction
    // summarizes away an assistant `tool_calls` message and leaves its result
    // behind (North-Mini-Code `docker_20260823_200353` opens with exactly
    // that — the loop nudge, stranded at index 2). Those are the run's
    // problem, not the pruner's; the invariant to prove is that pruning adds
    // no new ones. Count rather than compare ids: some models reuse
    // tool_call ids (North cycles through seven of them), so only POSITION
    // identifies a pair.
    let stranded_before = stranded(&messages);

    let before = messages.len();
    let chars_before = payload_chars(&messages);
    let outcome = prune_repeated_reads(&mut messages);
    let chars_after = payload_chars(&messages);
    eprintln!(
        "{path}: {before} -> {} messages ({} pairs dropped across {} keys), \
         {chars_before} -> {chars_after} chars, deepest {:?}",
        messages.len(),
        outcome.removed / 2,
        outcome.keys,
        outcome.deepest,
    );

    assert_eq!(before - messages.len(), outcome.removed);
    // `PRUNE_OUT` writes the pruned history back out so a replay probe can
    // send THIS code's output to a live model, instead of re-testing a
    // Python re-implementation of the same idea.
    if let Ok(out) = std::env::var("PRUNE_OUT") {
        let mut dump = dump.clone();
        dump["messages"] = serde_json::to_value(&messages).expect("messages did not serialize");
        std::fs::write(&out, serde_json::to_string(&dump).unwrap()).expect("PRUNE_OUT unwritable");
        eprintln!("wrote {out}");
    }
    let stranded_after = stranded(&messages);
    assert!(
        stranded_after <= stranded_before,
        "pruning stranded a tool result ({stranded_before} -> {stranded_after})"
    );
}

/// Number of `tool` results whose immediate predecessor is not an assistant
/// message carrying a call with the same id — i.e. results the chat template
/// has nothing to attach to.
fn stranded(messages: &[Message]) -> usize {
    messages
        .iter()
        .enumerate()
        .filter(|(_, m)| m.role == "tool")
        .filter(|(i, m)| {
            let Some(prev) = i.checked_sub(1).map(|p| &messages[p]) else {
                return true;
            };
            let Some(id) = m.tool_call_id.as_deref() else {
                return true;
            };
            !prev
                .tool_calls
                .as_ref()
                .is_some_and(|cs| cs.iter().any(|c| c.id == id))
        })
        .count()
}

/// Rough prompt size: message content plus tool-call argument bytes. Chars,
/// not tokens — the ratio is what matters here, and it avoids depending on
/// the tokenizer estimate.
fn payload_chars(messages: &[Message]) -> usize {
    messages
        .iter()
        .map(|m| {
            m.content.as_deref().map_or(0, str::len)
                + m.tool_calls.as_ref().map_or(0, |cs| {
                    cs.iter().map(|c| c.function.arguments.len()).sum::<usize>()
                })
        })
        .sum()
}
