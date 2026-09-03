//! End-to-end test of long-running shell promotion + the `jobs` tool.
//!
//! Drives the REAL headless agent loop (`cli::commands::run::run`) against a
//! scripted wiremock LLM. Everything else is genuine: the shell worker, the
//! 60s→(here 1s) check-in, ShellControl::Detach, the JobRegistry, `jobs`
//! dispatch, the permission gate on `check` probes.
//!
//! Scripted conversation:
//!   R1  model: file(shell "echo started; sleep 3; echo done-marker-xyz")
//!       loop:  command outlives shell.default_timeout_secs=1 → auto-promoted
//!              → tool result = promotion message for job 1
//!   R2  model: jobs(action=wait, id=1, secs=30, check="echo probe-ok-xyz")
//!       loop:  job exits (~2s later) → FINISHED result incl. done-marker,
//!              then the check probe runs → probe-ok in the same result
//!   R3  model: plain text, no tool calls → loop exits
//!
//! Assertions inspect the REQUESTS the loop sent back to the "model": the
//! promotion message must reach R2's context, and the finished-job +
//! probe output must reach R3's context.

mod helpers;

use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer};

use miniswe::config::CeremonyMode;

#[tokio::test(flavor = "multi_thread")]
async fn shell_promotion_and_jobs_wait_end_to_end() {
    let mock_server = MockServer::start().await;
    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());

    // Fast promotion: foreground check-in after 1s instead of 60s.
    config.shell.default_timeout_secs = 1;
    // Keep the scripted conversation free of planning ceremony.
    config.tools.ceremony = CeremonyMode::Off;
    config.tools.plan = false;
    config.lsp.enabled = false;
    config.model.max_retries = 0;

    // R1: kick off the long-running command.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(helpers::mock_sse_tool_call(
            "shell",
            r#"{"action":"run","command":"echo started; sleep 3; echo done-marker-xyz"}"#,
        ))
        .up_to_n_times(1)
        .mount(&mock_server)
        .await;
    // R2: wait on the promoted job with a check probe.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(helpers::mock_sse_tool_call(
            "shell",
            r#"{"action":"wait","id":1,"secs":30,"check":"echo probe-ok-xyz"}"#,
        ))
        .up_to_n_times(1)
        .mount(&mock_server)
        .await;
    // R3+: done (no tool calls → loop exits). Unlimited so any extra
    // nudge-induced round also terminates.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(helpers::mock_sse_text_response(
            "Deployment finished successfully.",
        ))
        .mount(&mock_server)
        .await;

    // The whole scripted run should take ~4s; a stuck promotion (the old
    // continue/kill stdin prompt) would hang forever — bound it.
    let run = miniswe::cli::commands::run::run(
        config,
        "run the fake deploy and confirm it completes",
        false, // plan_only
        true,  // headless — promotion path under test
        false, // continue_session
        None,
        None,
    );
    tokio::time::timeout(std::time::Duration::from_secs(60), run)
        .await
        .expect("agent loop hung — promotion did not prevent blocking")
        .expect("agent loop errored");

    let requests = mock_server.received_requests().await.unwrap();
    let chat_bodies: Vec<String> = requests
        .iter()
        .filter(|r| r.url.path().ends_with("/chat/completions"))
        .map(|r| String::from_utf8_lossy(&r.body).into_owned())
        .collect();
    assert!(
        chat_bodies.len() >= 3,
        "expected at least 3 model rounds, got {}",
        chat_bodies.len()
    );

    // R2's context: the promotion message (job id, no-rerun guidance).
    let r2 = &chat_bodies[1];
    assert!(
        r2.contains("promoted to background job 1"),
        "promotion message missing from round 2 context"
    );
    assert!(
        r2.contains("Do NOT re-run"),
        "promotion guidance missing from round 2 context"
    );

    // R3's context: the finished job result + the check probe's output.
    let r3 = &chat_bodies[2];
    assert!(
        r3.contains("FINISHED"),
        "finished-job report missing from round 3 context"
    );
    assert!(
        r3.contains("done-marker-xyz"),
        "job output missing from round 3 context"
    );
    assert!(
        r3.contains("probe-ok-xyz"),
        "check probe output missing from round 3 context"
    );
}

/// A `shell` call that puts the command text in `action` instead of the
/// literal verb must get the corrected call back verbatim, not a list of
/// verb names. Regression test for the schema collapse in
/// demo-e2e-task-demo-skills-miniswe-20260830-214306 attempt 0, where 32 of
/// 88 shell calls were malformed and the model responded to the old
/// "Use 'run' (command)" wording by adding `command` while leaving the
/// command text in `action`.
#[tokio::test]
async fn shell_action_holding_command_text_echoes_the_corrected_call() {
    let (_tmp, config) = helpers::create_test_project();
    let perms = miniswe::tools::permissions::PermissionManager::headless(&config);
    let registry = miniswe::tools::jobs::JobRegistry::default();

    // Quotes and shell metacharacters must survive into the example.
    let cmd = r#"ls -la chart/ && echo "done" 2>/dev/null"#;
    let result = miniswe::tools::jobs::execute(
        &serde_json::json!({ "action": cmd, "timeout": 10 }),
        &config,
        &perms,
        &registry,
        None,
    )
    .await;

    assert!(
        !result.success,
        "wrong action should fail: {}",
        result.content
    );
    let expected = serde_json::json!({ "action": "run", "command": cmd }).to_string();
    assert!(
        result.content.contains(&expected),
        "error must echo the corrected call {expected}, got: {}",
        result.content
    );

    // A missing `action` has no command to echo — it must not suggest an
    // empty one.
    let missing = miniswe::tools::jobs::execute(
        &serde_json::json!({ "timeout": 10 }),
        &config,
        &perms,
        &registry,
        None,
    )
    .await;
    assert!(!missing.success);
    assert!(
        !missing.content.contains(r#""command":"""#),
        "missing action must not suggest an empty command: {}",
        missing.content
    );
}
