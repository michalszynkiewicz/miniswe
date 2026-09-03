//! Prune repeated read/inspection call+result pairs out of the prompt tail.
//!
//! Compaction removes the WRONG END for a read loop. Every compaction
//! strategy summarises the OLDEST history, but a read loop lives in the
//! NEWEST messages, so each forced compaction leaves the repeats untouched
//! and *raises* their share of the prompt. Measured on Muse-Glimmer
//! `docker_20260826_163007`: the same `file(read tests/e2e_context.rs:435-441)`
//! pair went from 7% of the prompt at dump 101 to 24% at dump 130, while the
//! history stayed pinned at 14.5-15.7k tokens against a 15,667-token budget
//! for 112 consecutive rounds — the model was paying for a wall of identical
//! reads and being primed by it to issue one more.
//!
//! So prune surgically instead: for a read/inspection call that recurs
//! [`MIN_REPEATS`]+ times, keep the FIRST pair (the anchor: this is where the
//! content entered the conversation) and the LAST pair (the current truth,
//! and the one the next turn reasons from), and drop everything between.
//!
//! What this buys, and what it does NOT. Replay against the live loop
//! (`scripts/moments/tier1-glimmer-readloop-probe.py`, n=8/cell on moments
//! 101 and 130) puts EVERY arm at 0/16 escape: the read nudge, the
//! escalation and the `stuck_check` note, but equally the Python prune
//! sketch, this module's own output, and pruning combined with the
//! escalation. Once Glimmer is deep in a terminal read loop nothing short of
//! a real context reset turns it around, and pruning is not that lever — do
//! not reach for it as one.
//!
//! What it buys is cost. The loop runs at a fixed two copies instead of
//! inflating the prompt, so the rounds that detect it stop paying for a wall
//! of repeats. On the 12-run 2026-08-27 queue, North-Mini-Code went 0/6, 0/6
//! -> 3/6, 3/6 while its loop count went UP (89-115 -> 152-168): freed of the
//! per-fire tax it reached 509-561 rounds instead of 291-388 inside the same
//! 3400s wall. Runs that never loop are untouched — qwen 624s against a
//! 618-625s baseline, `prunes=0`.
//!
//! Two hard invariants:
//! - An assistant `tool_calls` message and its matching `tool_call_id`
//!   result are removed TOGETHER or the chat template breaks (an orphan
//!   result is a 400 on most servers, and llama.cpp's PEG parser rejects
//!   the whole request).
//! - Must run BEFORE `maybe_compress`. `refresh_current_state` appends the
//!   `[CURRENT STATE]` block onto an existing message's content, which can be
//!   a tool result inside a pruned pair — it reconciles unconditionally at the
//!   top of `maybe_compress`, so a dropped carrier is rebuilt in the same
//!   round. (Superseded older copies going out with the pruned pairs is a
//!   bonus, not a loss.)
//! - Only read-only inspection calls are eligible. Dropping repeats of a
//!   MUTATING call would misrepresent the history of the tree — and the
//!   edit-loop guards deliberately depend on the visible pile of repeated
//!   rejections, which this must never eat.

use crate::cli::commands::agent::loop_detector::loop_call_key;
use crate::context::compressor::is_guard_observation;
use crate::llm::Message;

/// Marker opening the synthetic note left behind on the surviving pair. Also
/// listed in the compressor's guard markers so observation masking can't eat
/// it a round later.
pub const PRUNE_NOTE_MARKER: &str = "[pruned]";

/// Occurrences of one read key needed before pruning starts. Four, not
/// three: read → edit → re-read → verify is a legitimate rhythm that lands
/// on three identical reads, and the middle one there is a real before/after
/// snapshot. At four the call has stopped being a rhythm and started being a
/// rut — the live loops this targets ran 30-115 deep.
const MIN_REPEATS: usize = 4;

/// True for calls whose identical re-issue is pure re-inspection: no tree
/// mutation, no side effect, and a result the model already holds verbatim.
/// Deliberately narrower than `is_mutating_call`'s read-only set — `check`
/// and `plan(check)` are read-only but their result genuinely changes as the
/// tree changes, and their repetition is evidence worth keeping.
fn is_prunable_read(name: &str, args: &serde_json::Value) -> bool {
    match name {
        "code" | "show_rev" => true,
        "file" => matches!(
            args.get("action").and_then(|v| v.as_str()),
            Some("read") | Some("search") | Some("help") | None
        ),
        _ => false,
    }
}

/// Short human-readable form of a call, for the note.
fn call_summary(name: &str, args: &serde_json::Value) -> String {
    let action = args.get("action").and_then(|v| v.as_str());
    let path = args
        .get("path")
        .or_else(|| args.get("file"))
        .and_then(|v| v.as_str());
    match (action, path) {
        (Some(a), Some(p)) => format!("{name}({a} {p})"),
        (Some(a), None) => format!("{name}({a})"),
        (None, Some(p)) => format!("{name}({p})"),
        (None, None) => name.to_string(),
    }
}

/// What one pruning pass did.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct PruneOutcome {
    /// Messages removed (always even — call+result pairs).
    pub removed: usize,
    /// Distinct call keys that were pruned.
    pub keys: usize,
    /// Summary of the deepest pruned key, for the status line.
    pub deepest: Option<String>,
}

impl PruneOutcome {
    pub fn is_empty(&self) -> bool {
        self.removed == 0
    }
}

/// One `(assistant tool_calls, tool result)` pair eligible for pruning.
struct Pair {
    /// Index of the assistant message; the result sits at `idx + 1`.
    idx: usize,
    key: String,
    summary: String,
}

/// Collect the prunable read pairs, in conversation order.
///
/// Only single-call assistant messages qualify: a multi-call message would
/// need every one of its calls to be a prunable read AND all of its results
/// removed with it, which is not worth the blast radius for a shape no
/// current model produces in a read loop.
fn prunable_pairs(messages: &[Message]) -> Vec<Pair> {
    let mut pairs = Vec::new();
    for idx in 0..messages.len().saturating_sub(1) {
        let msg = &messages[idx];
        if msg.role != "assistant" {
            continue;
        }
        let Some(calls) = msg.tool_calls.as_ref() else {
            continue;
        };
        let [call] = calls.as_slice() else {
            continue;
        };
        let result = &messages[idx + 1];
        if result.role != "tool" {
            continue;
        }
        // When the server echoes ids, insist they line up — a mismatch means
        // this is not the pair we think it is.
        if result
            .tool_call_id
            .as_deref()
            .is_some_and(|id| id != call.id)
        {
            continue;
        }
        let Ok(args) = serde_json::from_str::<serde_json::Value>(&call.function.arguments) else {
            continue;
        };
        if !is_prunable_read(&call.function.name, &args) {
            continue;
        }
        // Corrective guidance riding on a read result must stay visible.
        // Our OWN note is stripped first: it is guard-marked (so masking
        // can't eat it), but treating that as "this pair is protected" would
        // drop the survivor out of its own group and move the note somewhere
        // else on the next round.
        if is_guard_observation(&strip_note(result.content.as_deref().unwrap_or(""))) {
            continue;
        }
        pairs.push(Pair {
            idx,
            key: loop_call_key(&call.function.name, &args),
            summary: call_summary(&call.function.name, &args),
        });
    }
    pairs
}

/// A tool result with any note this module wrote removed.
fn strip_note(content: &str) -> String {
    if !content.contains(PRUNE_NOTE_MARKER) {
        return content.to_string();
    }
    content
        .lines()
        .filter(|l| !l.starts_with(PRUNE_NOTE_MARKER))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Append (or refresh) the pruning note on a tool result. Idempotent: any
/// note this function wrote on a previous round is stripped first, so the
/// count stays accurate instead of stacking up.
fn set_note(msg: &mut Message, dropped: usize, summary: &str) {
    let existing = msg.content.take().unwrap_or_default();
    let mut kept = strip_note(&existing);
    if !kept.is_empty() {
        kept.push('\n');
    }
    kept.push_str(&format!(
        "{PRUNE_NOTE_MARKER} {dropped} earlier identical `{summary}` calls were \
         dropped from context. You already have this output — repeating the \
         call cannot produce anything new."
    ));
    msg.content = Some(kept);
}

/// Prune repeated read/inspection pairs in place. Cheap (no LLM call, one
/// pass over the history) and safe to run every round — steady state for a
/// live loop is the key oscillating between [`MIN_REPEATS`] - 1 and
/// [`MIN_REPEATS`] occurrences.
pub fn prune_repeated_reads(messages: &mut Vec<Message>) -> PruneOutcome {
    let pairs = prunable_pairs(messages);
    if pairs.len() < MIN_REPEATS {
        return PruneOutcome::default();
    }

    // Group pair positions by key, preserving conversation order.
    let mut groups: Vec<(&str, Vec<usize>)> = Vec::new();
    for pair in &pairs {
        match groups.iter_mut().find(|(k, _)| *k == pair.key) {
            Some((_, positions)) => positions.push(pair.idx),
            None => groups.push((&pair.key, vec![pair.idx])),
        }
    }

    let mut drop = vec![false; messages.len()];
    let mut outcome = PruneOutcome::default();
    let mut deepest = 0usize;
    // (surviving assistant index, dropped count, summary)
    let mut notes: Vec<(usize, usize, String)> = Vec::new();

    for (_, positions) in &groups {
        if positions.len() < MIN_REPEATS {
            continue;
        }
        // Keep the first (anchor) and the last (current truth).
        for &idx in &positions[1..positions.len() - 1] {
            drop[idx] = true;
            drop[idx + 1] = true;
        }
        let dropped = positions.len() - 2;
        let last = *positions.last().expect("non-empty by the length check");
        let summary = pairs
            .iter()
            .find(|p| p.idx == last)
            .map(|p| p.summary.clone())
            .unwrap_or_default();
        if dropped > deepest {
            deepest = dropped;
            outcome.deepest = Some(summary.clone());
        }
        notes.push((last, dropped, summary));
        outcome.removed += dropped * 2;
        outcome.keys += 1;
    }

    if outcome.is_empty() {
        return PruneOutcome::default();
    }

    // Note goes on the surviving result, which is not itself dropped — write
    // it before the removal so the recorded index is still valid.
    for (idx, dropped, summary) in notes {
        set_note(&mut messages[idx + 1], dropped, &summary);
    }

    let mut i = 0;
    messages.retain(|_| {
        let keep = !drop[i];
        i += 1;
        keep
    });
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{FunctionCall, ToolCall};

    fn call(id: &str, name: &str, args: &str) -> Message {
        Message::assistant_tool_calls(vec![ToolCall {
            id: id.into(),
            r#type: "function".into(),
            function: FunctionCall {
                name: name.into(),
                arguments: args.into(),
            },
        }])
    }

    fn read_pair(id: &str, path: &str, body: &str) -> Vec<Message> {
        vec![
            call(
                id,
                "file",
                &format!(r#"{{"action":"read","path":"{path}"}}"#),
            ),
            Message::tool_result(id, body),
        ]
    }

    fn edit_pair(id: &str, path: &str) -> Vec<Message> {
        vec![
            call(
                id,
                "replace_range",
                &format!(r#"{{"path":"{path}","start":1,"end":2}}"#),
            ),
            Message::tool_result(id, "ok"),
        ]
    }

    fn history(pairs: usize, path: &str) -> Vec<Message> {
        let mut msgs = vec![Message::user("go")];
        for i in 0..pairs {
            msgs.extend(read_pair(&format!("c{i}"), path, "line one\nline two"));
        }
        msgs
    }

    #[test]
    fn under_threshold_is_untouched() {
        let mut msgs = history(MIN_REPEATS - 1, "src/a.rs");
        let before = msgs.len();
        let outcome = prune_repeated_reads(&mut msgs);
        assert!(outcome.is_empty());
        assert_eq!(msgs.len(), before);
    }

    #[test]
    fn keeps_first_and_last_drops_the_middle() {
        let mut msgs = history(30, "src/a.rs");
        let outcome = prune_repeated_reads(&mut msgs);
        assert_eq!(outcome.keys, 1);
        assert_eq!(outcome.removed, 28 * 2);
        // user + 2 surviving pairs
        assert_eq!(msgs.len(), 1 + 4);
        assert_eq!(msgs[1].tool_calls.as_ref().unwrap()[0].id, "c0");
        assert_eq!(msgs[3].tool_calls.as_ref().unwrap()[0].id, "c29");
    }

    #[test]
    fn every_surviving_result_still_has_its_call() {
        let mut msgs = history(12, "src/a.rs");
        prune_repeated_reads(&mut msgs);
        for (i, m) in msgs.iter().enumerate() {
            if m.role == "tool" {
                let prev = &msgs[i - 1];
                assert_eq!(prev.role, "assistant");
                assert_eq!(
                    prev.tool_calls.as_ref().unwrap()[0].id,
                    m.tool_call_id.clone().unwrap()
                );
            }
        }
    }

    #[test]
    fn note_lands_on_the_survivor_and_does_not_stack() {
        let mut msgs = history(10, "src/a.rs");
        prune_repeated_reads(&mut msgs);
        let note_lines = |m: &Message| {
            m.content
                .as_deref()
                .unwrap_or("")
                .lines()
                .filter(|l| l.starts_with(PRUNE_NOTE_MARKER))
                .count()
        };
        assert_eq!(note_lines(&msgs[4]), 1);
        assert!(msgs[4].content.as_deref().unwrap().contains("src/a.rs"));
        // Original content survives alongside the note.
        assert!(msgs[4].content.as_deref().unwrap().contains("line two"));

        // A second loop episode on the already-pruned history refreshes the
        // note in place rather than appending another one.
        for i in 100..104 {
            msgs.extend(read_pair(
                &format!("c{i}"),
                "src/a.rs",
                "line one\nline two",
            ));
        }
        let outcome = prune_repeated_reads(&mut msgs);
        assert!(!outcome.is_empty());
        let last = msgs.last().unwrap();
        assert_eq!(note_lines(last), 1);
    }

    #[test]
    fn mutating_calls_are_never_pruned() {
        let mut msgs = vec![Message::user("go")];
        for i in 0..20 {
            msgs.extend(edit_pair(&format!("e{i}"), "src/a.rs"));
        }
        let before = msgs.len();
        let outcome = prune_repeated_reads(&mut msgs);
        assert!(outcome.is_empty());
        assert_eq!(msgs.len(), before);
    }

    #[test]
    fn distinct_reads_are_grouped_independently() {
        let mut msgs = vec![Message::user("go")];
        for i in 0..10 {
            msgs.extend(read_pair(&format!("a{i}"), "src/a.rs", "aaa"));
            msgs.extend(read_pair(&format!("b{i}"), "src/b.rs", "bbb"));
        }
        let outcome = prune_repeated_reads(&mut msgs);
        assert_eq!(outcome.keys, 2);
        assert_eq!(outcome.removed, 2 * 8 * 2);
        let ids: Vec<&str> = msgs
            .iter()
            .filter_map(|m| m.tool_calls.as_ref())
            .map(|tcs| tcs[0].id.as_str())
            .collect();
        assert_eq!(ids, vec!["a0", "b0", "a9", "b9"]);
    }

    #[test]
    fn guard_observations_survive() {
        let mut msgs = vec![Message::user("go")];
        for i in 0..10 {
            let mut pair = read_pair(&format!("c{i}"), "src/a.rs", "body");
            if i == 5 {
                pair[1] = Message::tool_result("c5", "[hint] make the smallest edit");
            }
            msgs.extend(pair);
        }
        prune_repeated_reads(&mut msgs);
        let ids: Vec<&str> = msgs
            .iter()
            .filter_map(|m| m.tool_calls.as_ref())
            .map(|tcs| tcs[0].id.as_str())
            .collect();
        assert!(ids.contains(&"c5"), "guard pair was pruned: {ids:?}");
    }

    #[test]
    fn interleaved_reads_and_edits_keep_the_edits() {
        let mut msgs = vec![Message::user("go")];
        for i in 0..8 {
            msgs.extend(read_pair(&format!("r{i}"), "src/a.rs", "body"));
            msgs.extend(edit_pair(&format!("e{i}"), "src/a.rs"));
        }
        prune_repeated_reads(&mut msgs);
        let edits = msgs
            .iter()
            .filter_map(|m| m.tool_calls.as_ref())
            .filter(|tcs| tcs[0].function.name == "replace_range")
            .count();
        assert_eq!(edits, 8);
    }

    #[test]
    fn multi_call_assistant_messages_are_skipped() {
        let mut msgs = vec![Message::user("go")];
        for i in 0..10 {
            msgs.push(Message::assistant_tool_calls(vec![
                ToolCall {
                    id: format!("m{i}a"),
                    r#type: "function".into(),
                    function: FunctionCall {
                        name: "file".into(),
                        arguments: r#"{"action":"read","path":"src/a.rs"}"#.into(),
                    },
                },
                ToolCall {
                    id: format!("m{i}b"),
                    r#type: "function".into(),
                    function: FunctionCall {
                        name: "file".into(),
                        arguments: r#"{"action":"read","path":"src/b.rs"}"#.into(),
                    },
                },
            ]));
            msgs.push(Message::tool_result(&format!("m{i}a"), "aaa"));
            msgs.push(Message::tool_result(&format!("m{i}b"), "bbb"));
        }
        let before = msgs.len();
        assert!(prune_repeated_reads(&mut msgs).is_empty());
        assert_eq!(msgs.len(), before);
    }
}
