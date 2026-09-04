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
use wiremock::matchers::{method, path};
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
