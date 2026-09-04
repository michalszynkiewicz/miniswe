use super::{
    CURRENT_STATE_MARKER, StateRefresh, checkoff_only, find_current_state,
    format_current_state_block, refresh_current_state,
};
use crate::config::Config;
use crate::llm::Message;

fn config_in(dir: &std::path::Path) -> Config {
    std::fs::create_dir_all(dir.join(".miniswe")).unwrap();
    let mut config = Config::default();
    config.project_root = dir.to_path_buf();
    config.ensure_session_dir().unwrap();
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
    let mut config = config_in(tmp.path());
    config.skill_step_injection = true;
    let mut cursor = SkillCursor::default();
    cursor.push_skill(
        "pkg-package",
        tmp.path(),
        vec![SkillStep {
            name: "Create the package".into(),
            anchor: "## Create".into(),
        }],
    );
    // Only a distilled step renders; an undistilled one produces no block.
    cursor.cache("Run `pkg pack dev lint` on the generated package.".into());
    skill_cursor::save(&config, &cursor);

    let block = format_current_state_block(&config).expect("cursor alone must produce a block");
    assert!(block.contains("[SKILL STEP]"), "{block}");
    assert!(block.contains("pkg-package"), "{block}");
    assert!(!block.contains("[PLAN]"), "{block}");
}

#[test]
fn skill_step_injection_off_ignores_stale_cursor() {
    // The repl never sets skill_step_injection, so a cursor left behind
    // by a killed run must not inject a [SKILL STEP] block there — the
    // repl has no `skill` tool, so the block would demand an impossible
    // call with no way to advance or clear the cursor.
    use crate::cli::commands::agent::skill_cursor::{self, SkillCursor};
    use crate::cli::commands::agent::skill_router::SkillStep;
    let tmp = tempfile::TempDir::new().unwrap();
    let config = config_in(tmp.path());
    let mut cursor = SkillCursor::default();
    cursor.push_skill(
        "pkg-package",
        tmp.path(),
        vec![SkillStep {
            name: "Create the package".into(),
            anchor: "## Create".into(),
        }],
    );
    skill_cursor::save(&config, &cursor);

    assert!(
        format_current_state_block(&config).is_none(),
        "cursor must be inert when injection is off"
    );

    // A plan alongside the stale cursor still yields a block — just
    // without the [SKILL STEP] section.
    std::fs::write(config.session_path("plan.md"), "1. step one\n").unwrap();
    let block = format_current_state_block(&config).unwrap();
    assert!(block.contains("[PLAN]"), "{block}");
    assert!(!block.contains("[SKILL STEP]"), "{block}");
}

#[test]
fn appends_to_last_message_when_state_exists() {
    let tmp = tempfile::TempDir::new().unwrap();
    let config = config_in(tmp.path());
    std::fs::write(config.session_path("plan.md"), "1. step one\n").unwrap();

    let mut msgs = vec![Message::user("do the task"), Message::assistant("ok")];
    refresh_current_state(&mut msgs, &config, StateRefresh::Sticky);

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
    std::fs::write(config.session_path("plan.md"), "1. step one\n").unwrap();

    let mut msgs = vec![Message::tool_result("call1", "first result")];
    refresh_current_state(&mut msgs, &config, StateRefresh::Sticky);
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
    std::fs::write(config.session_path("plan.md"), "1. [x] step one\n").unwrap();
    msgs.push(Message::tool_result("call2", "second result"));
    refresh_current_state(&mut msgs, &config, StateRefresh::Sticky);

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
fn unchanged_block_stays_put_across_rounds() {
    // Case 1, the whole point of the stickiness: on a narrow-window model
    // an unchanged block must NOT be moved, because relocating it rewinds
    // the prompt past the model's reuse threshold and forces a full
    // re-prefill. Simulate several rounds with unchanged plan content and
    // confirm the block never leaves the message it was first anchored to,
    // and that exactly one copy exists throughout.
    let tmp = tempfile::TempDir::new().unwrap();
    let mut config = config_in(tmp.path());
    narrow_window(&mut config);
    write_plan(&config, &["step one", "step two"], 0);

    let mut msgs = vec![Message::tool_result("call1", "round 1 result")];
    refresh_current_state(&mut msgs, &config, StateRefresh::Sticky);
    let anchored = msgs[0].content.clone().unwrap();
    assert!(anchored.contains("[PLAN]"));

    for i in 2..=5 {
        msgs.push(Message::tool_result(
            &format!("call{i}"),
            &format!("round {i} result"),
        ));
        refresh_current_state(&mut msgs, &config, StateRefresh::Sticky); // unchanged plan content
    }

    assert_eq!(
        msgs[0].content.as_deref().unwrap(),
        anchored,
        "carrier message must be byte-identical across rounds"
    );
    for (i, m) in msgs.iter().enumerate().skip(1) {
        assert!(
            !m.content.as_deref().unwrap().contains(CURRENT_STATE_MARKER),
            "no second copy on message {i}: {:?}",
            m.content
        );
    }
}

#[test]
fn re_anchors_when_compaction_drops_the_carrier() {
    // The guarantee the old unconditional refresh provided: the block can
    // never be summarized away. Now it is provided by repair instead of
    // by relocation — if the message carrying the block disappears, the
    // next refresh notices the block is gone and re-appends it.
    let tmp = tempfile::TempDir::new().unwrap();
    let config = config_in(tmp.path());
    std::fs::write(config.session_path("plan.md"), "1. step one\n").unwrap();

    let mut msgs = vec![Message::tool_result("call1", "round 1 result")];
    refresh_current_state(&mut msgs, &config, StateRefresh::Sticky);
    assert!(msgs[0].content.as_deref().unwrap().contains("[PLAN]"));

    // Compaction eats the carrier and leaves a summary in its place.
    msgs[0] = Message::user("[summary of earlier rounds]");
    msgs.push(Message::tool_result("call2", "round 2 result"));
    refresh_current_state(&mut msgs, &config, StateRefresh::Sticky);

    let last = msgs.last().unwrap().content.as_deref().unwrap();
    assert!(last.contains("[PLAN]"), "block must be restored: {last}");
    assert!(!msgs[0].content.as_deref().unwrap().contains("[PLAN]"));
}

#[test]
fn re_anchors_when_the_carrier_content_was_rewritten() {
    // Weaker damage than a drop: the marker survives but the block text
    // was mangled (a summarizer folding it into prose). The byte-compare
    // must reject it and rebuild a clean copy on the newest message.
    let tmp = tempfile::TempDir::new().unwrap();
    let config = config_in(tmp.path());
    std::fs::write(config.session_path("plan.md"), "1. step one\n").unwrap();

    let mut msgs = vec![Message::tool_result("call1", "round 1 result")];
    refresh_current_state(&mut msgs, &config, StateRefresh::Sticky);

    let mangled = msgs[0].content.as_deref().unwrap().replace("step one", "…");
    msgs[0].content = Some(mangled);
    msgs.push(Message::tool_result("call2", "round 2 result"));
    refresh_current_state(&mut msgs, &config, StateRefresh::Sticky);

    assert_eq!(msgs[0].content.as_deref().unwrap(), "round 1 result");
    let last = msgs.last().unwrap().content.as_deref().unwrap();
    assert!(last.contains("step one"), "{last}");
    assert_eq!(last.matches("[PLAN]").count(), 1);
}

/// Pin the served model to one with a narrow attention window — the only
/// case in which stickiness engages at all.
fn narrow_window(config: &mut Config) {
    config.model.probed_model =
        Some("/home/x/models/Laguna-XS-2.1-GGUF/Laguna-XS-2.1-IQ4_XS.gguf".into());
}

/// Write a plan in the rendered checkbox form, with the first `ticked`
/// steps done. Ticking a step is the change that may be appended; editing
/// the step list is the change that must sweep.
fn write_plan(config: &Config, steps: &[&str], ticked: usize) {
    let body: String = steps
        .iter()
        .enumerate()
        .map(|(i, s)| format!("- [{}] {s}\n", if i < ticked { "x" } else { " " }))
        .collect();
    std::fs::write(config.session_path("plan.md"), body).unwrap();
}

#[test]
fn wide_window_model_re_anchors_every_round() {
    // The default path, and every unknown model: a block at the tail
    // rewinds by exactly its own size, which a model that can trim its KV
    // tail serves almost free. So keep the original behaviour — the block
    // follows the newest message and history stays at one copy.
    let tmp = tempfile::TempDir::new().unwrap();
    let config = config_in(tmp.path());
    std::fs::write(config.session_path("plan.md"), "1. step one\n").unwrap();

    let mut msgs = vec![Message::tool_result("call1", "round 1 result")];
    refresh_current_state(&mut msgs, &config, StateRefresh::Sticky);
    for i in 2..=4 {
        msgs.push(Message::tool_result(&format!("call{i}"), "result"));
        refresh_current_state(&mut msgs, &config, StateRefresh::Sticky);
    }

    assert_eq!(find_current_state(&msgs).len(), 1);
    assert_eq!(msgs[0].content.as_deref().unwrap(), "round 1 result");
    assert!(
        msgs.last()
            .unwrap()
            .content
            .as_deref()
            .unwrap()
            .contains("[PLAN]"),
        "block must ride the tail on a wide-window model"
    );
}

#[test]
fn ticking_a_step_appends_instead_of_rewinding() {
    // Case 2: the block parked, the conversation moved on, and now a step
    // got ticked off. Stripping the old copy would rewind past the cliff
    // and re-prefill everything; appending is a pure tail extension, and
    // the superseded copy still agrees about what the plan is.
    let tmp = tempfile::TempDir::new().unwrap();
    let mut config = config_in(tmp.path());
    narrow_window(&mut config);
    write_plan(&config, &["step one", "step two"], 0);

    let mut msgs = vec![Message::tool_result("call1", "round 1 result")];
    refresh_current_state(&mut msgs, &config, StateRefresh::Sticky);
    let carrier = msgs[0].content.clone().unwrap();

    msgs.push(Message::tool_result("call2", "round 2 result"));
    write_plan(&config, &["step one", "step two"], 1);
    refresh_current_state(&mut msgs, &config, StateRefresh::Sticky);

    assert_eq!(
        msgs[0].content.as_deref().unwrap(),
        carrier,
        "history before the tail must not be rewritten"
    );
    let last = msgs.last().unwrap().content.as_deref().unwrap();
    assert!(last.contains("[x] step one"), "new block appended: {last}");
    assert_eq!(
        find_current_state(&msgs).len(),
        2,
        "one superseded, one live"
    );
}

#[test]
fn editing_the_plan_sweeps_even_on_a_narrow_window() {
    // Case 3: the steps themselves changed, so the parked copy now
    // CONTRADICTS the live one. That is worth a full re-prefill — a stale
    // contradictory copy wins on primacy and no marker wording recovers
    // it.
    let tmp = tempfile::TempDir::new().unwrap();
    let mut config = config_in(tmp.path());
    narrow_window(&mut config);
    write_plan(&config, &["step one", "step two"], 0);

    let mut msgs = vec![Message::tool_result("call1", "round 1 result")];
    refresh_current_state(&mut msgs, &config, StateRefresh::Sticky);
    msgs.push(Message::tool_result("call2", "round 2 result"));
    write_plan(&config, &["step one", "a different second step"], 0);
    refresh_current_state(&mut msgs, &config, StateRefresh::Sticky);

    assert_eq!(
        find_current_state(&msgs).len(),
        1,
        "a contradicting copy must be swept, not left behind"
    );
    assert_eq!(msgs[0].content.as_deref().unwrap(), "round 1 result");
    let last = msgs.last().unwrap().content.as_deref().unwrap();
    assert!(last.contains("a different second step"), "{last}");
}

fn wide_window_block_is_stripped_rather_than_duplicated() {
    // Even a checkoff-only change consolidates on a wide-window model:
    // the rewind is served by trimming the KV tail, so history stays at
    // one copy and the block stays maximally recent.
    let tmp = tempfile::TempDir::new().unwrap();
    let config = config_in(tmp.path());
    std::fs::write(config.session_path("plan.md"), "1. step one\n").unwrap();

    let mut msgs = vec![Message::tool_result("call1", "round 1 result")];
    refresh_current_state(&mut msgs, &config, StateRefresh::Sticky);
    msgs.push(Message::tool_result("call2", "short"));
    std::fs::write(config.session_path("plan.md"), "1. [x] step one\n").unwrap();
    refresh_current_state(&mut msgs, &config, StateRefresh::Sticky);

    assert_eq!(find_current_state(&msgs).len(), 1);
    assert_eq!(msgs[0].content.as_deref().unwrap(), "round 1 result");
}

#[test]
fn checkoff_copies_accumulate_uncapped() {
    // Deliberately unbounded. A cap forces a sweep exactly when the block
    // has been stable longest, which is when the rewind back to the oldest
    // copy is largest — replayed over the benchmark corpus, a cap of 3
    // doubled Laguna's full re-prefills. Copies cost context instead, and
    // compaction reclaims them for free.
    let tmp = tempfile::TempDir::new().unwrap();
    let mut config = config_in(tmp.path());
    narrow_window(&mut config);
    let steps = ["one", "two", "three", "four", "five", "six"];

    let mut msgs = vec![Message::tool_result("call1", "round 1 result")];
    for (i, _) in steps.iter().enumerate() {
        write_plan(&config, &steps, i);
        refresh_current_state(&mut msgs, &config, StateRefresh::Sticky);
        msgs.push(Message::tool_result(&format!("call{i}"), "result"));
    }
    write_plan(&config, &steps, steps.len());
    refresh_current_state(&mut msgs, &config, StateRefresh::Sticky);

    let copies = find_current_state(&msgs);
    assert!(
        copies.len() > 3,
        "checkoffs must accumulate past the old cap, got {}",
        copies.len()
    );
    let &(i, pos) = copies.last().unwrap();
    let live = &msgs[i].content.as_deref().unwrap()[pos..];
    assert!(live.contains("[x] six"), "newest copy must be live: {live}");
}

#[test]
fn checkoff_only_distinguishes_progress_from_revision() {
    let ticked = "\n\n[CURRENT STATE]\n[PLAN]\n- [x] (round 3) build it\n- [ ] test it\n";
    let unticked = "\n\n[CURRENT STATE]\n[PLAN]\n- [ ] build it\n- [ ] test it\n";
    let edited = "\n\n[CURRENT STATE]\n[PLAN]\n- [ ] build it\n- [ ] ship it\n";
    let noted = "\n\n[CURRENT STATE]\n[PLAN]\n- [ ] build it\n- [ ] test it\n[SCRATCHPAD]\nhm\n";

    assert!(
        checkoff_only(unticked, ticked),
        "ticking a step is progress"
    );
    assert!(
        !checkoff_only(unticked, edited),
        "editing a step is revision"
    );
    assert!(
        !checkoff_only(unticked, noted),
        "a scratchpad edit is not a checkoff — it is unbounded, so it sweeps"
    );
    assert!(
        !checkoff_only(ticked, ticked),
        "an unchanged block is case 1, not case 2"
    );
}

fn reanchor_mode_sweeps_and_moves_unconditionally() {
    // What maybe_compress uses after a compaction that rewrote history:
    // the cached prefix is already dead, so consolidate for free.
    let tmp = tempfile::TempDir::new().unwrap();
    let mut config = config_in(tmp.path());
    narrow_window(&mut config);
    write_plan(&config, &["step one"], 0);

    let mut msgs = vec![Message::tool_result("call1", "round 1 result")];
    refresh_current_state(&mut msgs, &config, StateRefresh::Sticky);
    msgs.push(Message::tool_result("call2", "round 2 result"));
    refresh_current_state(&mut msgs, &config, StateRefresh::Sticky);
    assert!(msgs[0].content.as_deref().unwrap().contains("[PLAN]"));

    refresh_current_state(&mut msgs, &config, StateRefresh::Reanchor);
    assert_eq!(msgs[0].content.as_deref().unwrap(), "round 1 result");
    assert!(msgs[1].content.as_deref().unwrap().contains("[PLAN]"));
}

#[test]
fn active_skill_step_re_anchors_every_round() {
    // Carve-out: with a [SKILL STEP] injected, recency is load-bearing
    // (the model drifts back to its priors when the step isn't the
    // freshest thing in context), so those rounds keep relocating the
    // block and keep paying the re-prefill.
    use crate::cli::commands::agent::skill_cursor::{self, SkillCursor};
    use crate::cli::commands::agent::skill_router::SkillStep;
    let tmp = tempfile::TempDir::new().unwrap();
    let mut config = config_in(tmp.path());
    config.skill_step_injection = true;
    let mut cursor = SkillCursor::default();
    cursor.push_skill(
        "pkg-package",
        tmp.path(),
        vec![SkillStep {
            name: "Create the package".into(),
            anchor: "## Create".into(),
        }],
    );
    cursor.cache("Run `pkg pack dev lint` on the generated package.".into());
    skill_cursor::save(&config, &cursor);

    let mut msgs = vec![Message::tool_result("call1", "round 1 result")];
    refresh_current_state(&mut msgs, &config, StateRefresh::Sticky);
    assert!(msgs[0].content.as_deref().unwrap().contains("[SKILL STEP]"));

    msgs.push(Message::tool_result("call2", "round 2 result"));
    refresh_current_state(&mut msgs, &config, StateRefresh::Sticky); // unchanged step content

    assert_eq!(msgs[0].content.as_deref().unwrap(), "round 1 result");
    let last = msgs.last().unwrap().content.as_deref().unwrap();
    assert!(last.contains("[SKILL STEP]"), "{last}");
}

#[test]
fn no_op_when_no_state_and_nothing_to_strip() {
    let tmp = tempfile::TempDir::new().unwrap();
    let config = config_in(tmp.path());
    let mut msgs = vec![Message::tool_result("call1", "a result")];
    refresh_current_state(&mut msgs, &config, StateRefresh::Sticky);
    assert_eq!(msgs[0].content.as_deref().unwrap(), "a result");
}
