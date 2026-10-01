//! Golden-turn regression tests for the headless agent loop.
//!
//! These drive the REAL `run()` end-to-end against a scripted wiremock LLM
//! and assert on everything observable from outside: files the tools wrote,
//! the session log's lines, and the exact request sequence the loop sent
//! (including which tools were visible in each request). They are the
//! spanning safety net for the loop-unification refactor — drift in round
//! sequencing, tool dispatch, the no-tool-call nudge ladder, the ceremony
//! latch, or the LLM error ladder shows up here as a changed request
//! sequence or a changed log line.

mod helpers;

use std::fs;
use wiremock::matchers::{body_string_contains, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use miniswe::cli::commands::run::run;
use miniswe::config::Config;

/// Test project + config pointed at the mock server, with the LSP disabled
/// so nothing tries to spawn rust-analyzer in the throwaway directory.
fn golden_config(mock_uri: &str) -> (tempfile::TempDir, Config) {
    let (tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, mock_uri);
    config.lsp.enabled = false;
    (tmp, config)
}

/// Bodies of the POST /v1/chat/completions requests, in arrival order.
/// (The model probe and startup summary hit other paths — filtered out.)
async fn chat_requests(server: &MockServer) -> Vec<String> {
    server
        .received_requests()
        .await
        .expect("mock server records requests")
        .into_iter()
        .filter(|r| r.url.path() == "/v1/chat/completions")
        .map(|r| String::from_utf8_lossy(&r.body).into_owned())
        .collect()
}

/// Tool names advertised in one chat request body.
fn request_tool_names(body: &str) -> Vec<String> {
    let v: serde_json::Value = serde_json::from_str(body).expect("chat request body is JSON");
    v["tools"]
        .as_array()
        .map(|tools| {
            tools
                .iter()
                .filter_map(|t| t["function"]["name"].as_str())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// Contents of the (single) session log the run wrote.
fn session_log(config: &Config) -> String {
    let logs_dir = config.project_root.join(".miniswe/logs");
    let mut entries: Vec<_> = fs::read_dir(&logs_dir)
        .expect("run() should have created .miniswe/logs")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .collect();
    entries.sort();
    assert_eq!(entries.len(), 1, "expected exactly one session log");
    fs::read_to_string(&entries[0]).unwrap()
}

// ── Happy path: tool round, then text-only rounds through the exit gates ──

#[tokio::test]
async fn golden_read_tool_then_clean_finish() {
    let server = MockServer::start().await;

    // Round 1: the model reads a file.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(helpers::mock_sse_tool_call(
            "file",
            r#"{"action":"read","path":"hello.txt"}"#,
        ))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    // Rounds 2+: text-only replies. Round 2 stops before any plan was set,
    // which draws the one-shot no-plan exit nudge; round 3 ends the turn.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(helpers::mock_sse_text_response(
            "Done — the file says hello.",
        ))
        .mount(&server)
        .await;

    let (_tmp, config) = golden_config(&server.uri());
    fs::write(
        helpers::project_path(&config, "hello.txt"),
        "Hello from golden test!",
    )
    .unwrap();

    run(
        config.clone(),
        "Read hello.txt and summarize it",
        false,
        true,
        false,
        None,
        None,
    )
    .await
    .unwrap();

    let reqs = chat_requests(&server).await;
    assert_eq!(
        reqs.len(),
        3,
        "tool round + nudged text round + final text round"
    );
    // The tool result made it back into the next request's context.
    assert!(
        reqs[1].contains("Hello from golden test!"),
        "round-2 request should carry the file-read tool result"
    );
    // The premature-exit path fired exactly once, with the no-plan wording.
    assert!(
        reqs[2].contains("You returned no tool call before setting a plan"),
        "round-3 request should carry the no-plan exit nudge"
    );

    let log = session_log(&config);
    assert!(
        log.contains("[tool] ✓ file("),
        "file read should be logged as a successful tool call:\n{log}"
    );
    assert!(
        log.contains("[end] 3 rounds, status=ok"),
        "turn should end cleanly after 3 rounds:\n{log}"
    );
}

// ── Ceremony latch: plan(set) unlocks the edit tools, write lands on disk ──

#[tokio::test]
async fn golden_plan_latch_unlocks_write_file() {
    let server = MockServer::start().await;

    // Round 1: set a plan (the ceremony unlock).
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(helpers::mock_sse_tool_call(
            "plan",
            r#"{"action":"set","steps":[{"step":"create out.txt with the golden content"}]}"#,
        ))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    // Round 2: write the file.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(helpers::mock_sse_tool_call(
            "write_file",
            r#"{"path":"out.txt","content":"golden content\n"}"#,
        ))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    // Rounds 3+: text-only. Round 3 stops with an unchecked plan step,
    // drawing the premature-exit nudge; round 4 ends the turn.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(helpers::mock_sse_text_response("All steps are complete."))
        .mount(&server)
        .await;

    let (_tmp, config) = golden_config(&server.uri());
    run(
        config.clone(),
        "Create out.txt",
        false,
        true,
        false,
        None,
        None,
    )
    .await
    .unwrap();

    let disk = fs::read_to_string(helpers::project_path(&config, "out.txt")).unwrap();
    assert_eq!(disk, "golden content\n");

    let reqs = chat_requests(&server).await;
    assert_eq!(reqs.len(), 4, "plan + write + nudged stop + final stop");
    // The ceremony latch: edit tools hidden before plan(set), visible after.
    let before = request_tool_names(&reqs[0]);
    let after = request_tool_names(&reqs[1]);
    assert!(
        !before.iter().any(|n| n == "write_file"),
        "write_file must be hidden before a plan is set, got: {before:?}"
    );
    assert!(
        after.iter().any(|n| n == "write_file"),
        "write_file must be visible after plan(set), got: {after:?}"
    );
    // Stopping on an unchecked plan draws the standard premature-exit nudge.
    assert!(
        reqs[3].contains("Stopping. Are you sure? Check the plan"),
        "round-4 request should carry the premature-exit nudge"
    );

    let log = session_log(&config);
    assert!(log.contains("[tool] ✓ plan("), "plan set logged:\n{log}");
    assert!(log.contains("[tool] ✓ write_file("), "write logged:\n{log}");
    assert!(
        log.contains("[end] 4 rounds, status=ok"),
        "clean end after 4 rounds:\n{log}"
    );
}

// ── LLM error ladder: a fatal server error ends the turn as an error ──

#[tokio::test]
async fn golden_llm_fatal_error_marks_session_error() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("Internal Server Error"))
        .mount(&server)
        .await;

    let (_tmp, config) = golden_config(&server.uri());
    // run() itself still returns Ok — the error is recorded in the log.
    run(
        config.clone(),
        "Do something",
        false,
        true,
        false,
        None,
        None,
    )
    .await
    .unwrap();

    let log = session_log(&config);
    assert!(
        log.contains("[error:llm]"),
        "fatal LLM error should be logged:\n{log}"
    );
    assert!(
        log.contains("[end] 1 rounds, status=error"),
        "session should end in round 1 with status=error:\n{log}"
    );
}

// ── Loop detector: streak of 3 identical mutating calls → StopCalls ──
//
// agent/loop_detector.rs + agent/turn/call_gate.rs: after `plan(set)`, the
// model repeats the byte-identical `revert(notes.txt, rev=0)` call. The 3rd
// consecutive repeat is the first detection (nudge injected); the 6th is
// the second detection — with no check/skill cursor to recover from
// (`config.validation.command` empty, no skill cursor), call_gate's
// recovery ladder falls through every `recover_output` source to `None`
// and hard-stops the turn via `AdmitFlow::StopCalls { error: true }`.

#[tokio::test]
async fn golden_loop_detector_streak_stops_turn() {
    let server = MockServer::start().await;

    // R1: set a plan (ceremony unlock).
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(helpers::mock_sse_tool_call(
            "plan",
            r#"{"action":"set","steps":[{"step":"tweak notes.txt"}]}"#,
        ))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    // R2: a structural edit, seeding rev_0 (pristine) + rev_1.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(helpers::mock_sse_tool_call(
            "replace_range",
            r#"{"path":"notes.txt","start":2,"end":2,"content":"line2 updated\n"}"#,
        ))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    // R3-R8: the SAME revert call, six times in a row.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(helpers::mock_sse_tool_call(
            "revert",
            r#"{"path":"notes.txt","rev":0}"#,
        ))
        .up_to_n_times(6)
        .mount(&server)
        .await;

    let (_tmp, config) = golden_config(&server.uri());
    fs::write(
        helpers::project_path(&config, "notes.txt"),
        "line1\nline2\nline3\n",
    )
    .unwrap();

    run(
        config.clone(),
        "Tweak notes.txt",
        false,
        true,
        false,
        None,
        None,
    )
    .await
    .unwrap();

    let reqs = chat_requests(&server).await;
    assert_eq!(
        reqs.len(),
        8,
        "plan + edit + 3 reverts (1st detection) + 3 more reverts (hard stop)"
    );

    // First detection (3rd consecutive revert, round 5) pushes the
    // fast-mode loop hint into round 6's request, byte-for-byte.
    let loop_hint = "ERROR: You are in a loop — this exact tool call has been repeated 3 times in a row. Stop retrying it in this turn. If you were repeating replace_range/insert_at with the same args, the edit already landed (or was rejected) — inspect the revision table with show_rev before trying again. If you were repeating revert to the same rev, pick a different live rev or move on. For structural rewrites that keep failing line-by-line, switch to edit_file (semantic patch). Use file(action='read') to re-check current state.";
    assert!(
        reqs[5].contains(loop_hint),
        "round-6 request should carry the loop-detected hint verbatim:\n{}",
        reqs[5]
    );

    let log = session_log(&config);
    let loop_lines = log
        .matches("[loop] revert(notes.txt to rev_0) repeated 3x")
        .count();
    assert_eq!(
        loop_lines, 2,
        "the streak should fire exactly twice (1st detection + hard stop):\n{log}"
    );
    assert!(
        log.contains("[end] 8 rounds, status=error"),
        "second detection has no check/skill cursor to recover from and hard-stops the turn:\n{log}"
    );
}

// ── Done-gate block → debugger two-fire (no judge): distinct failure_key ──
//
// agent/turn/done_gate.rs + agent/debugger.rs. `config.validation.command`
// is a self-incrementing shell counter: it can't literally be "a file the
// test rewrites between rounds" (the harness drives the real `run()` as one
// opaque `.await`, so there is no seam to reach in from outside) but it
// produces the same effect — the check's own output changes across
// invocations, entirely synchronously inside `run_check_command`. Calls 1-2
// report "missing widget wiring"; calls 3+ report "missing gadget wiring" —
// a DIFFERENT `failure_key`, which is what lets the debugger fire a SECOND
// time (the key-collapse bug that used to suppress this stays fixed).
// `debugger_judge = false` keeps the verdict parser on the plain
// Report/Scrap path; `debugger_judge_rewind = false` is belt-and-braces
// (irrelevant here since `run_debugger` only computes a rewind candidate
// when `debugger_judge` is also true).

#[tokio::test]
async fn golden_done_gate_debugger_two_fire_report() {
    let server = MockServer::start().await;

    // Debugger sub-requests carry build_prompt's exclusive, literal phrase
    // ("A verification check is BLOCKING task completion") plus the
    // specific failure text — both are required to match, so each mock
    // serves exactly the fire it's scripted for regardless of mount order.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_string_contains(
            "A verification check is BLOCKING task completion",
        ))
        .and(body_string_contains("missing widget wiring"))
        .respond_with(helpers::mock_sse_text_response(
            "ROOT CAUSE: the widget is read but never wired into assemble().\nFIX: call wire_widget(cfg) inside assemble() instead of discarding the parsed value.",
        ))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_string_contains(
            "A verification check is BLOCKING task completion",
        ))
        .and(body_string_contains("missing gadget wiring"))
        .respond_with(helpers::mock_sse_text_response(
            "ROOT CAUSE: the gadget is read but never wired into assemble().\nFIX: call wire_gadget(cfg) inside assemble() instead of discarding the parsed value.",
        ))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    // Every main-loop round is text-only: no tool call, no plan — the
    // behavioral done-gate is what drives the whole turn.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(helpers::mock_sse_text_response(
            "I believe the task is complete.",
        ))
        .mount(&server)
        .await;

    let (_tmp, mut config) = golden_config(&server.uri());
    config.validation.command = r#"n=$(( $(cat gate_n.txt 2>/dev/null || echo 0) + 1 )); echo "$n" > gate_n.txt; if [ "$n" -le 2 ]; then echo "ERROR: missing widget wiring"; else echo "ERROR: missing gadget wiring"; fi; exit 1"#.to_string();
    config.tools.debugger_judge = false;
    config.tools.debugger_judge_rewind = false;

    run(
        config.clone(),
        "Wire the widget and gadget into assemble()",
        false,
        true,
        false,
        None,
        None,
    )
    .await
    .unwrap();

    let reqs = chat_requests(&server).await;
    assert_eq!(
        reqs.len(),
        7,
        "5 main rounds + 2 debugger sub-requests (one per fire)"
    );

    // Round 2 carries the no-plan premature-exit nudge (no plan was ever set).
    assert!(
        reqs[1].contains("You returned no tool call before setting a plan"),
        "round-2 request should carry the no-plan exit nudge:\n{}",
        reqs[1]
    );
    // Round 3 carries block #1's fallback message — 1 block is below
    // DEBUGGER_TRIGGER_BLOCKS, so no debugger fire yet.
    assert!(
        reqs[2].contains("[Verification failed — do NOT finish yet.")
            && reqs[2].contains("missing widget wiring"),
        "round-3 request should carry the plain verification-failed fallback:\n{}",
        reqs[2]
    );
    // reqs[4] is round 4's request, built after block #2 triggered the
    // FIRST debugger fire and injected its gate-report message.
    assert!(
        reqs[4].contains(
            "A read-only debugger with fresh eyes investigated the failing check and produced"
        ) && reqs[4].contains("ROOT CAUSE: the widget"),
        "round-4 request should carry debugger fire #1's report:\n{}",
        reqs[4]
    );
    // reqs[6] is round 5's request, built after block #3 (a DIFFERENT
    // failure_key: gadget, not widget) triggered the SECOND debugger fire.
    assert!(
        reqs[6].contains("ROOT CAUSE: the gadget"),
        "round-5 request should carry debugger fire #2's report:\n{}",
        reqs[6]
    );

    let log = session_log(&config);
    assert!(
        log.contains("[end] 5 rounds, status=ok"),
        "5 main rounds, clean end once the retry budget is exhausted (not an error):\n{log}"
    );

    // The raw gate-failure output the debugger's note points readers at is
    // the MOST RECENT (gadget) failure.
    let raw = fs::read_to_string(config.miniswe_path("last_gate_failure.txt")).unwrap();
    assert!(
        raw.contains("missing gadget wiring"),
        "last_gate_failure.txt should hold the most recent check output:\n{raw}"
    );
}

// ── Debugger judge SCRAP → whole-tree revert; 2nd SCRAP → already-reset ──
//
// Same rung as above, but `debugger_judge = true` so the debugger's verdict
// is parsed for SCRAP/REWIND/CONTINUE instead of treated as a plain report.
// `debugger_judge_rewind = false` forces `find_rewind_candidate` to be
// skipped entirely (REWIND is therefore unreachable by construction here —
// see the report for why a genuine REWIND scenario is out of scope).
// Fire #1 (widget failure) votes SCRAP → restart::scrap_restart whole-tree
// reverts to the round-0 snapshot, taken before round 1 ever ran, so
// marker.txt (written in round 2) does not survive. Fire #2 (gadget
// failure, a DIFFERENT failure_key so it's even allowed to fire again)
// also votes SCRAP, but `state.gate.restart_fired` is already true and is
// never reset by `scrap_restart` — so the ladder degrades to
// `SCRAP_ALREADY_RESET_MSG` instead of reverting a second time.

#[tokio::test]
async fn golden_debugger_judge_scrap_restarts_then_already_reset_on_second_vote() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_string_contains(
            "A verification check is BLOCKING task completion",
        ))
        .and(body_string_contains("missing widget wiring"))
        .respond_with(helpers::mock_sse_text_response(
            "DECISION: SCRAP\nREASON: marker.txt is off-path for the goal; revert and restart.",
        ))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_string_contains(
            "A verification check is BLOCKING task completion",
        ))
        .and(body_string_contains("missing gadget wiring"))
        .respond_with(helpers::mock_sse_text_response(
            "DECISION: SCRAP\nREASON: still off-path after the reset; revert again.",
        ))
        .up_to_n_times(1)
        .mount(&server)
        .await;

    // R1: set a plan.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(helpers::mock_sse_tool_call(
            "plan",
            r#"{"action":"set","steps":[{"step":"create marker.txt"}]}"#,
        ))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    // R2: write a file that should NOT survive the SCRAP revert.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(helpers::mock_sse_tool_call(
            "write_file",
            r#"{"path":"marker.txt","content":"should not survive\n"}"#,
        ))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    // Every other round is text-only.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(helpers::mock_sse_text_response(
            "I believe the task is complete.",
        ))
        .mount(&server)
        .await;

    let (_tmp, mut config) = golden_config(&server.uri());
    // The counter lives OUTSIDE project_root: fire #1's SCRAP does a
    // whole-tree `git` revert of project_root, which would otherwise wipe
    // an in-tree counter file and silently replay "widget" (not "gadget")
    // for the first two post-reset blocks too.
    let counter_dir = tempfile::tempdir().unwrap();
    let counter_path = counter_dir.path().join("gate_n.txt");
    config.validation.command = format!(
        r#"n=$(( $(cat {p} 2>/dev/null || echo 0) + 1 )); echo "$n" > {p}; if [ "$n" -le 2 ]; then echo "ERROR: missing widget wiring"; else echo "ERROR: missing gadget wiring"; fi; exit 1"#,
        p = counter_path.display()
    );
    config.tools.debugger_judge = true;
    config.tools.debugger_judge_rewind = false;

    run(
        config.clone(),
        "Set up the initial scaffold",
        false,
        true,
        false,
        None,
        None,
    )
    .await
    .unwrap();

    let reqs = chat_requests(&server).await;
    assert_eq!(
        reqs.len(),
        11,
        "9 main rounds + 2 debugger-judge sub-requests (widget SCRAP, gadget SCRAP-already-reset)"
    );

    // marker.txt was written in round 2, after the round-0 snapshot — the
    // whole-tree revert takes it back out of existence.
    assert!(
        !helpers::project_path(&config, "marker.txt").exists(),
        "SCRAP should whole-tree-revert the file written before it fired"
    );

    // reqs[5] is the first debugger sub-request (fire #1).
    assert!(
        reqs[5].contains("A verification check is BLOCKING task completion")
            && reqs[5].contains("missing widget wiring"),
        "debugger fire #1 should see the widget failure output:\n{}",
        reqs[5]
    );
    // reqs[6] is the first request built from the RE-ASSEMBLED context
    // after the SCRAP: messages were replaced and conversation_history
    // cleared, so no trace of the reverted marker.txt work remains, but the
    // original task (re-derived from `ctx.task`, not the wiped history) is
    // still there.
    assert!(
        !reqs[6].contains("marker"),
        "post-SCRAP context should carry no trace of the reverted work:\n{}",
        reqs[6]
    );
    assert!(
        reqs[6].contains("Set up the initial scaffold"),
        "post-SCRAP context should still carry the original task:\n{}",
        reqs[6]
    );
    // reqs[8] is the second debugger sub-request (fire #2) — a DISTINCT
    // failure_key (gadget, not widget) is what lets it fire again at all.
    assert!(
        reqs[8].contains("A verification check is BLOCKING task completion")
            && reqs[8].contains("missing gadget wiring"),
        "debugger fire #2 should see the DIFFERENT gadget failure output:\n{}",
        reqs[8]
    );

    // The second SCRAP vote finds the tree already reset this turn and
    // degrades to marching orders instead of a second revert. Copied
    // verbatim (same line-continuation the source uses) so the compiler
    // computes the identical joined string rather than hand-flattening it.
    let scrap_already_reset_msg = "[A fresh-context review voted to reset again, but the tree was already reset once this \
         turn. Keep going: read the current failure carefully and fix it directly.]";
    assert!(
        reqs[9].contains(scrap_already_reset_msg),
        "round after the 2nd SCRAP vote should carry SCRAP_ALREADY_RESET_MSG verbatim:\n{}",
        reqs[9]
    );

    // The session log lives at project_root/.miniswe/logs/<ts>.log — INSIDE
    // the tree SCRAP whole-tree-reverts. The shadow git excludes `/.miniswe/`
    // entirely (src/tools/snapshots.rs `info/exclude`) precisely so the
    // revert never rewinds the log file out from under `SessionLog`'s single
    // append handle (src/logging.rs). Regression guard: lines from BOTH
    // sides of the SCRAP survive — the round-2 plan call and the final end
    // marker. (This repo's own root .gitignore has `.miniswe/`, which hid
    // the bug from every bench run; the test project has no .gitignore.)
    let log = session_log(&config);
    assert!(
        log.contains("[tool] ✓ plan("),
        "pre-SCRAP log lines must survive the whole-tree revert:\n{log}"
    );
    assert!(
        log.contains("[end] 9 rounds, status=ok"),
        "post-SCRAP log lines must land in the same (never-reverted) file:\n{log}"
    );
}

// ── Auto-revert AST cascade: 3 consecutive broken-AST edits force-revert ──
//
// tools/fast/auto_revert.rs (CASCADE_THRESHOLD = 3) + tools/fast/dispatch.rs
// (gated on `auto_revert_ast_cascade && matches!(name, "replace_range" |
// "insert_at")`), both default-on under `tools.edit_mode = Fast` (also
// default). NOT the same mechanism as `spiral.rs::SPIRAL_REVERT_THRESHOLD`
// cited by the spec for this scenario — that's `tools.spiral_reset`
// (default OFF), a separate revert-loop counter scoped to explicit
// top-level `revert` calls. Uses tree-sitter only; no LSP involvement.

#[tokio::test]
async fn golden_auto_revert_cascade_force_reverts_broken_file() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(helpers::mock_sse_tool_call(
            "plan",
            r#"{"action":"set","steps":[{"step":"tweak code.rs"}]}"#,
        ))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    // R2, R3, R4: three structural edits, each dropping a closing brace on a
    // DIFFERENT line (so the no-op guard never short-circuits them) and
    // each leaving the AST broken.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(helpers::mock_sse_tool_call(
            "replace_range",
            r#"{"path":"code.rs","start":1,"end":1,"content":"pub fn one() -> i32 { 1\n"}"#,
        ))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(helpers::mock_sse_tool_call(
            "replace_range",
            r#"{"path":"code.rs","start":2,"end":2,"content":"pub fn two() -> i32 { 2\n"}"#,
        ))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(helpers::mock_sse_tool_call(
            "replace_range",
            r#"{"path":"code.rs","start":3,"end":3,"content":"pub fn three() -> i32 { 3\n"}"#,
        ))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    // R5+: text-only.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(helpers::mock_sse_text_response("Done."))
        .mount(&server)
        .await;

    let (_tmp, config) = golden_config(&server.uri());
    let pristine =
        "pub fn one() -> i32 { 1 }\npub fn two() -> i32 { 2 }\npub fn three() -> i32 { 3 }\n";
    fs::write(helpers::project_path(&config, "code.rs"), pristine).unwrap();

    run(
        config.clone(),
        "Tweak code.rs",
        false,
        true,
        false,
        None,
        None,
    )
    .await
    .unwrap();

    let reqs = chat_requests(&server).await;
    assert_eq!(
        reqs.len(),
        6,
        "plan + 3 breaking edits (3rd triggers the cascade) + nudged stop + final stop"
    );

    // The cascade message landed in the tool_result of the 3rd edit, which
    // reaches the model in round 5's request, worded exactly as the source
    // emits it (not the separate, differently-worded manual `revert` hint).
    assert!(
        reqs[4]
            .contains("EACH left the syntax tree broken — you were digging deeper, not recovering"),
        "round-5 request should carry the auto-revert cascade message:\n{}",
        reqs[4]
    );
    assert!(
        reqs[4].contains("STOP patching line-by-line")
            && reqs[4].contains(
                "a single replace_range over the whole enclosing block (with matching braces/brackets)"
            ),
        "round-5 request should carry the cascade's stop-digging guidance:\n{}",
        reqs[4]
    );

    // The file on disk is back to its pristine pre-edit content.
    let disk = fs::read_to_string(helpers::project_path(&config, "code.rs")).unwrap();
    assert_eq!(
        disk, pristine,
        "auto-revert should restore the file to its last AST-clean revision"
    );

    let log = session_log(&config);
    assert!(
        log.contains("[end] 6 rounds, status=ok"),
        "clean end after 6 rounds:\n{log}"
    );
}
