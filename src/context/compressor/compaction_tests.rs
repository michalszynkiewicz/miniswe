use super::*;
use crate::config::Config;

// context_window=1200, tool_def_tokens=0 → available=1000, raw_budget=333.
// A ~400-char message is ~100 tokens, so a handful of them blows the budget.
fn cfg() -> Config {
    let mut c = Config::default();
    c.model.context_window = Some(1200);
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
