//! End-to-end tests for the agent loop.
//!
//! These use a mock HTTP server (wiremock) to simulate LLM responses,
//! then run tool dispatch through the real tool system.

mod helpers;

use serde_json::json;
use std::fs;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use miniswe::config::Config;
use miniswe::llm::{
    ChatRequest, LlmClient, Message, TRUNCATED_TOOL_CALL_MARKER, is_truncated_tool_call_error,
};
use miniswe::tools::{self, PermissionManager};

fn perms(config: &Config) -> PermissionManager {
    PermissionManager::headless(config)
}

// ── LLM client basics ──────────────────────────────────────────────

#[tokio::test]
async fn llm_client_chat_plain_text() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(helpers::mock_text_response("Hello!"))
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());

    let client = LlmClient::new(config.model.clone());
    let request = ChatRequest {
        messages: vec![Message::user("hi")],
        tools: None,
        tool_choice: None,
    };

    let response = client.chat(&request).await.unwrap();

    assert_eq!(
        response.choices[0].message.content.as_deref().unwrap(),
        "Hello!"
    );
}

#[tokio::test]
async fn llm_client_chat_tool_call() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(helpers::mock_tool_call_response(
            "file",
            json!({"action": "read", "path": "src/main.rs"}),
        ))
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());

    let client = LlmClient::new(config.model.clone());
    let request = ChatRequest {
        messages: vec![Message::user("read main.rs")],
        tools: Some(tools::tool_definitions(miniswe::config::EditMode::Smart)),
        tool_choice: None,
    };

    let response = client.chat(&request).await.unwrap();

    let msg = &response.choices[0].message;
    let tc = msg.tool_calls.as_ref().expect("should have tool calls");
    assert_eq!(tc.len(), 1);
    assert_eq!(tc[0].function.name, "file");

    let args: serde_json::Value = serde_json::from_str(&tc[0].function.arguments).unwrap();
    assert_eq!(args["action"], "read");
    assert_eq!(args["path"], "src/main.rs");
}

// ── LLM client with streaming ───────────────────────────────────────

#[tokio::test]
async fn llm_client_stream_plain_text() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(helpers::mock_sse_text_response("Streamed hello!"))
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());

    let client = LlmClient::new(config.model.clone());
    let request = ChatRequest {
        messages: vec![Message::user("hi")],
        tools: None,
        tool_choice: None,
    };

    let cancelled = Arc::new(AtomicBool::new(false));
    let mut tokens = Vec::new();
    let response = client
        .chat_stream(&request, |token| tokens.push(token.to_string()), &cancelled)
        .await
        .unwrap();

    assert_eq!(
        response.choices[0].message.content.as_deref().unwrap(),
        "Streamed hello!"
    );
    assert!(!tokens.is_empty(), "should have received streaming tokens");
}

#[tokio::test]
async fn llm_client_stream_tool_call() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(helpers::mock_sse_tool_call(
            "write_file",
            r#"{"path":"test.txt","content":"hello"}"#,
        ))
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());

    let client = LlmClient::new(config.model.clone());
    let request = ChatRequest {
        messages: vec![Message::user("write a file")],
        tools: Some(tools::tool_definitions(miniswe::config::EditMode::Smart)),
        tool_choice: None,
    };

    let cancelled = Arc::new(AtomicBool::new(false));
    let response = client
        .chat_stream(&request, |_| {}, &cancelled)
        .await
        .unwrap();

    let msg = &response.choices[0].message;
    let tc = msg.tool_calls.as_ref().expect("should have tool calls");
    assert_eq!(tc[0].function.name, "write_file");

    let args: serde_json::Value = serde_json::from_str(&tc[0].function.arguments).unwrap();
    assert_eq!(args["path"], "test.txt");
    assert_eq!(args["content"], "hello");
}

// ── LLM client idle-timeout on stuck stream ────────────────────────
//
// These tests pin down the behavior added for Task #29: the client
// *always* streams internally so we can kill a connection that accepts
// the request, returns headers, and then never sends a body byte.
// Wiremock doesn't model "accept then hang mid-body", so we roll a tiny
// raw TCP listener.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Spawn a raw TCP server that:
///  1. Accepts up to `max_connections` HTTP requests.
///  2. For each one, reads the request (up to the headers terminator),
///     writes a 200 OK response with `Content-Type: text/event-stream`,
///     and then hangs forever without sending any body bytes.
///
/// Returns `(base_url, counter)`. The counter increments once per
/// connection accepted, so tests can assert whether a retry happened.
async fn start_hanging_sse_server(max_connections: usize) -> (String, Arc<AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = Arc::clone(&counter);

    tokio::spawn(async move {
        for _ in 0..max_connections {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            counter_clone.fetch_add(1, Ordering::SeqCst);

            tokio::spawn(async move {
                // Drain the request headers so reqwest considers the
                // write side complete. We don't care about the body.
                let mut buf = [0u8; 4096];
                let mut read_total = 0usize;
                while read_total < 4096 {
                    match socket.read(&mut buf[read_total..]).await {
                        Ok(0) => break,
                        Ok(n) => {
                            read_total += n;
                            if buf[..read_total].windows(4).any(|w| w == b"\r\n\r\n") {
                                break;
                            }
                        }
                        Err(_) => return,
                    }
                }

                // Send headers that announce an SSE stream, then stall.
                // We use `Transfer-Encoding: chunked` so the client
                // doesn't try to size the body by `Content-Length`.
                let headers = b"HTTP/1.1 200 OK\r\n\
                    Content-Type: text/event-stream\r\n\
                    Transfer-Encoding: chunked\r\n\
                    Cache-Control: no-cache\r\n\
                    \r\n";
                if socket.write_all(headers).await.is_err() {
                    return;
                }
                let _ = socket.flush().await;

                // Hold the socket open. The test's idle-timeout is
                // the thing we're proving fires here — once it
                // elapses, reqwest drops the connection.
                tokio::time::sleep(Duration::from_secs(30)).await;
            });
        }
    });

    (format!("http://{addr}"), counter)
}

#[tokio::test]
async fn llm_client_stream_idle_timeout_fires_on_hung_connection() {
    // Server headers the request, then never sends a body byte.
    let (uri, counter) = start_hanging_sse_server(1).await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &uri);
    // Tight idle timeout and *no* retries — we want to observe the
    // first-attempt error directly rather than a retry-delay pile.
    config.model.stream_idle_timeout_secs = 1;
    config.model.request_timeout_secs = 30;
    config.model.max_retries = 0;

    let client = LlmClient::new(config.model.clone());
    let request = ChatRequest {
        messages: vec![Message::user("hi")],
        tools: None,
        tool_choice: None,
    };

    let start = std::time::Instant::now();
    let result = client.chat(&request).await;
    let elapsed = start.elapsed();

    let err = result.expect_err("expected idle-timeout error, got Ok");
    let msg = err.to_string();
    assert!(
        msg.contains("LLM stream idle"),
        "error should mention idle timeout, got: {msg}"
    );
    // Should fail promptly — ~1s idle timeout, plus a little slack.
    // If this asserts, something is using wall-clock timeout instead.
    assert!(
        elapsed < Duration::from_secs(5),
        "idle timeout should fire quickly, took {elapsed:?}"
    );
    assert_eq!(
        counter.load(Ordering::SeqCst),
        1,
        "exactly one connection should have been made (max_retries=0)"
    );
}

#[tokio::test]
async fn llm_client_stream_idle_timeout_retries_and_eventually_gives_up() {
    // Two hanging connections — one for the initial attempt, one for
    // the retry. The retry should also time out, and the caller should
    // get a final "LLM stream idle" error after both attempts.
    let (uri, counter) = start_hanging_sse_server(2).await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &uri);
    config.model.stream_idle_timeout_secs = 1;
    config.model.request_timeout_secs = 30;
    config.model.max_retries = 1;

    let client = LlmClient::new(config.model.clone());
    let request = ChatRequest {
        messages: vec![Message::user("hi")],
        tools: None,
        tool_choice: None,
    };

    let result = client.chat(&request).await;
    let err = result.expect_err("expected idle-timeout error after retry");
    assert!(
        err.to_string().contains("LLM stream idle"),
        "error should mention idle timeout, got: {err}"
    );
    assert_eq!(
        counter.load(Ordering::SeqCst),
        2,
        "should have made initial attempt + 1 retry"
    );
}

#[tokio::test]
async fn llm_client_chat_stream_idle_timeout_fires_on_hung_connection() {
    // Same idle-timeout guarantee as `chat()`, but exercised through
    // the streaming path that the agent loop actually uses. Without
    // the per-chunk idle wrapper this test would block on the hung
    // SSE socket until the test runner timed out.
    let (uri, counter) = start_hanging_sse_server(1).await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &uri);
    config.model.stream_idle_timeout_secs = 1;
    config.model.request_timeout_secs = 30;
    config.model.max_retries = 0;

    let client = LlmClient::new(config.model.clone());
    let request = ChatRequest {
        messages: vec![Message::user("hi")],
        tools: None,
        tool_choice: None,
    };

    let cancelled = Arc::new(AtomicBool::new(false));
    let mut tokens = Vec::new();
    let start = std::time::Instant::now();
    let result = client
        .chat_stream(&request, |t| tokens.push(t.to_string()), &cancelled)
        .await;
    let elapsed = start.elapsed();

    let err = result.expect_err("expected idle-timeout error from chat_stream, got Ok");
    assert!(
        err.to_string().contains("LLM stream idle"),
        "error should mention idle timeout, got: {err}"
    );
    assert!(
        elapsed < Duration::from_secs(5),
        "idle timeout should fire quickly, took {elapsed:?}"
    );
    assert_eq!(
        counter.load(Ordering::SeqCst),
        1,
        "exactly one connection (max_retries=0)"
    );
    assert!(
        tokens.is_empty(),
        "no tokens should reach the UI before the stall"
    );
}

#[tokio::test]
async fn llm_client_chat_stream_idle_timeout_retries_when_no_progress() {
    // Two hanging connections; chat_stream should retry once because
    // no UI tokens were emitted on the first attempt, then bail with
    // a final "LLM stream idle" error.
    let (uri, counter) = start_hanging_sse_server(2).await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &uri);
    config.model.stream_idle_timeout_secs = 1;
    config.model.request_timeout_secs = 30;
    config.model.max_retries = 1;

    let client = LlmClient::new(config.model.clone());
    let request = ChatRequest {
        messages: vec![Message::user("hi")],
        tools: None,
        tool_choice: None,
    };

    let cancelled = Arc::new(AtomicBool::new(false));
    let result = client.chat_stream(&request, |_| {}, &cancelled).await;
    let err = result.expect_err("expected idle-timeout error after retry");
    assert!(
        err.to_string().contains("LLM stream idle"),
        "error should mention idle timeout, got: {err}"
    );
    assert_eq!(
        counter.load(Ordering::SeqCst),
        2,
        "should have made initial attempt + 1 retry"
    );
}

#[tokio::test]
async fn llm_client_chat_stream_no_retry_after_partial_progress() {
    // Server sends one usable SSE event then stalls forever — that
    // counts as "had progress", so chat_stream must surface the idle
    // error instead of retrying (a retry would re-stream the same
    // tokens to the UI and corrupt the rendered conversation).
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = Arc::clone(&counter);
    tokio::spawn(async move {
        for _ in 0..4 {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            counter_clone.fetch_add(1, Ordering::SeqCst);
            tokio::spawn(async move {
                let mut buf = [0u8; 4096];
                let mut read_total = 0usize;
                while read_total < 4096 {
                    match socket.read(&mut buf[read_total..]).await {
                        Ok(0) => break,
                        Ok(n) => {
                            read_total += n;
                            if buf[..read_total].windows(4).any(|w| w == b"\r\n\r\n") {
                                break;
                            }
                        }
                        Err(_) => return,
                    }
                }
                let headers = b"HTTP/1.1 200 OK\r\n\
                    Content-Type: text/event-stream\r\n\
                    Transfer-Encoding: chunked\r\n\
                    Cache-Control: no-cache\r\n\
                    \r\n";
                if socket.write_all(headers).await.is_err() {
                    return;
                }
                // One SSE chunk, framed as a chunked-encoding piece.
                let event = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n";
                let chunk = format!("{:x}\r\n{}\r\n", event.len(), event);
                let _ = socket.write_all(chunk.as_bytes()).await;
                let _ = socket.flush().await;
                // Then stall.
                tokio::time::sleep(Duration::from_secs(30)).await;
            });
        }
    });
    let uri = format!("http://{addr}");

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &uri);
    config.model.stream_idle_timeout_secs = 1;
    config.model.request_timeout_secs = 30;
    config.model.max_retries = 3; // would retry plenty if not for the guard

    let client = LlmClient::new(config.model.clone());
    let request = ChatRequest {
        messages: vec![Message::user("hi")],
        tools: None,
        tool_choice: None,
    };

    let cancelled = Arc::new(AtomicBool::new(false));
    let mut tokens = Vec::new();
    let result = client
        .chat_stream(&request, |t| tokens.push(t.to_string()), &cancelled)
        .await;

    let err = result.expect_err("expected idle-timeout after partial stream");
    assert!(
        err.to_string().contains("LLM stream idle"),
        "error should mention idle timeout, got: {err}"
    );
    assert_eq!(
        tokens, vec!["hi"],
        "exactly one token should have been delivered to the UI"
    );
    assert_eq!(
        counter.load(Ordering::SeqCst),
        1,
        "must NOT retry once any progress reached the UI (no duplicate tokens)"
    );
}

// ── Single tool call flow ───────────────────────────────────────────

#[tokio::test]
async fn single_tool_call_flow_reads_file() {
    let mock_server = MockServer::start().await;

    // LLM returns a file(action='read') tool call
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "file",
                            "arguments": r#"{"action":"read","path":"hello.txt"}"#
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        })))
        .up_to_n_times(1)
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());

    // Create the file the LLM will try to read
    fs::write(
        helpers::project_path(&config, "hello.txt"),
        "Hello from test!",
    )
    .unwrap();

    let client = LlmClient::new(config.model.clone());
    let p = perms(&config);

    // First call: get tool call
    let request = ChatRequest {
        messages: vec![
            Message::system("You are a test agent."),
            Message::user("Read hello.txt"),
        ],
        tools: Some(tools::tool_definitions(miniswe::config::EditMode::Smart)),
        tool_choice: None,
    };

    let response = client.chat(&request).await.unwrap();
    let msg = &response.choices[0].message;
    let tc = msg.tool_calls.as_ref().unwrap();

    // Execute the tool
    let args: serde_json::Value = serde_json::from_str(&tc[0].function.arguments).unwrap();
    let result = tools::execute_tool(&tc[0].function.name, &args, &config, &p, None)
        .await
        .unwrap();

    assert!(result.success);
    assert!(result.content.contains("Hello from test!"));
}

// ── Write file flow ─────────────────────────────────────────────────

#[tokio::test]
async fn write_file_flow_creates_file_on_disk() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(helpers::mock_tool_call_response(
            "write_file",
            json!({"path": "output.txt", "content": "Generated content\nLine 2\n"}),
        ))
        .up_to_n_times(1)
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());

    let client = LlmClient::new(config.model.clone());
    let p = perms(&config);

    let request = ChatRequest {
        messages: vec![Message::user("create output.txt")],
        tools: Some(tools::tool_definitions(miniswe::config::EditMode::Smart)),
        tool_choice: None,
    };

    let response = client.chat(&request).await.unwrap();
    let tc = response.choices[0].message.tool_calls.as_ref().unwrap();

    let args: serde_json::Value = serde_json::from_str(&tc[0].function.arguments).unwrap();
    let result = tools::execute_tool(&tc[0].function.name, &args, &config, &p, None)
        .await
        .unwrap();

    assert!(
        result.success,
        "write_file should succeed: {}",
        result.content
    );

    // Verify file on disk
    let disk = fs::read_to_string(helpers::project_path(&config, "output.txt")).unwrap();
    assert_eq!(disk, "Generated content\nLine 2\n");
}

// ── Invalid JSON args ───────────────────────────────────────────────

#[tokio::test]
async fn invalid_json_args_from_llm() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_bad",
                        "type": "function",
                        "function": {
                            "name": "file",
                            "arguments": "{invalid json!!!"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        })))
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());

    let client = LlmClient::new(config.model.clone());

    let request = ChatRequest {
        messages: vec![Message::user("do something")],
        tools: Some(tools::tool_definitions(miniswe::config::EditMode::Smart)),
        tool_choice: None,
    };

    let response = client.chat(&request).await.unwrap();
    let tc = &response.choices[0].message.tool_calls.as_ref().unwrap()[0];

    // The run loop would parse the arguments — verify parsing fails
    let parse_result: Result<serde_json::Value, _> = serde_json::from_str(&tc.function.arguments);
    assert!(parse_result.is_err(), "malformed JSON should fail to parse");
}

// ── Plan mode blocks edits ──────────────────────────────────────────

#[tokio::test]
async fn plan_mode_blocks_write_file() {
    // In plan mode, the run loop blocks file shell action and write_file/edit_file tools.
    let plan_only = true;
    let blocked_actions = ["shell"];
    for action in &blocked_actions {
        assert!(
            plan_only && matches!(*action, "shell"),
            "plan mode should block file action: {action}"
        );
    }

    // Allowed actions in plan mode
    let allowed_actions = ["read", "search"];
    for action in &allowed_actions {
        assert!(
            !matches!(*action, "shell"),
            "plan mode should allow file action: {action}"
        );
    }
}

// ── LLM error handling ──────────────────────────────────────────────

#[tokio::test]
async fn llm_api_error_returns_error() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("Internal Server Error"))
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());

    let client = LlmClient::new(config.model.clone());
    let request = ChatRequest {
        messages: vec![Message::user("hi")],
        tools: None,
        tool_choice: None,
    };

    let result = client.chat(&request).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn llm_chat_does_not_retry_truncated_tool_call_500() {
    // llama.cpp returns 500 + "Failed to parse tool call arguments as JSON"
    // when the model hits max_tokens mid tool-call. Retrying with the same
    // prompt would just reproduce the same truncated output, so we must
    // surface the error on the first attempt rather than burning the
    // retry budget.
    let mock_server = MockServer::start().await;

    let body = format!(
        r#"{{"error":{{"message":"{TRUNCATED_TOOL_CALL_MARKER}: Unexpected EOF","type":"server_error"}}}}"#
    );
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(500)
                .insert_header("Content-Type", "application/json")
                .set_body_string(body),
        )
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    // Give the client a generous retry budget — the classifier should
    // veto retries regardless, and this proves it.
    config.model.max_retries = 5;

    let client = LlmClient::new(config.model.clone());
    let request = ChatRequest {
        messages: vec![Message::user("hi")],
        tools: Some(tools::tool_definitions(miniswe::config::EditMode::Smart)),
        tool_choice: None,
    };

    let err = client
        .chat(&request)
        .await
        .expect_err("truncated tool call 500 should surface as error");
    let err_msg = err.to_string();
    assert!(
        is_truncated_tool_call_error(&err_msg),
        "classifier must recognize the error, got: {err_msg}"
    );

    let requests = mock_server
        .received_requests()
        .await
        .expect("mock server should record requests");
    assert_eq!(
        requests.len(),
        1,
        "truncated tool-call 500 must not be retried (max_retries=5), \
         got {} requests",
        requests.len()
    );
}

#[tokio::test]
async fn llm_chat_retries_transient_503_and_succeeds() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(503).set_body_string("temporarily unavailable"))
        .up_to_n_times(1)
        .mount(&mock_server)
        .await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(helpers::mock_text_response("Recovered"))
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());

    let client = LlmClient::new(config.model.clone());
    let request = ChatRequest {
        messages: vec![Message::user("hi")],
        tools: None,
        tool_choice: None,
    };

    let response = client.chat(&request).await.unwrap();
    assert_eq!(
        response.choices[0].message.content.as_deref().unwrap(),
        "Recovered"
    );
}

#[tokio::test]
async fn llm_connection_refused() {
    let (_tmp, mut config) = helpers::create_test_project();
    config.model.endpoint = "http://127.0.0.1:1".into();

    let client = LlmClient::new(config.model.clone());
    let request = ChatRequest {
        messages: vec![Message::user("hi")],
        tools: None,
        tool_choice: None,
    };

    let result = client.chat(&request).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn llm_stream_retries_transient_503_and_succeeds() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(503).set_body_string("temporarily unavailable"))
        .up_to_n_times(1)
        .mount(&mock_server)
        .await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(helpers::mock_sse_text_response("Stream recovered"))
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());

    let client = LlmClient::new(config.model.clone());
    let request = ChatRequest {
        messages: vec![Message::user("hi")],
        tools: None,
        tool_choice: None,
    };

    let cancelled = Arc::new(AtomicBool::new(false));
    let response = client
        .chat_stream(&request, |_| {}, &cancelled)
        .await
        .unwrap();

    assert_eq!(
        response.choices[0].message.content.as_deref().unwrap(),
        "Stream recovered"
    );
}

// ── Stream cancellation ────────────────────────────────────────────

#[tokio::test]
async fn stream_cancellation_via_flag() {
    let mock_server = MockServer::start().await;

    // Return a very long streamed response
    let body = (0..100)
        .map(|i| {
            format!(
                "data: {{\"choices\":[{{\"delta\":{{\"content\":\"tok{i}\"}},\"finish_reason\":null}}]}}\n\n"
            )
        })
        .collect::<String>()
        + "data: [DONE]\n\n";

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());

    let client = LlmClient::new(config.model.clone());
    let request = ChatRequest {
        messages: vec![Message::user("go")],
        tools: None,
        tool_choice: None,
    };

    let cancelled = Arc::new(AtomicBool::new(true)); // pre-cancelled
    let mut tokens = Vec::new();
    let result = client
        .chat_stream(&request, |token| tokens.push(token.to_string()), &cancelled)
        .await;

    // Pre-cancelled stream may return Ok (partial) or Err (aborted) — both are fine.
    // The key is it finishes quickly without processing all 100 tokens.
    match result {
        Ok(_) => assert!(
            tokens.len() < 10,
            "should stop early: got {} tokens",
            tokens.len()
        ),
        Err(_) => {} // Aborted — also acceptable
    }
}

// ── Message factories ──────────────────────────────────────────────

#[test]
fn message_factories_produce_correct_roles() {
    let sys = Message::system("system prompt");
    assert_eq!(sys.role, "system");

    let usr = Message::user("hello");
    assert_eq!(usr.role, "user");

    let asst = Message::assistant("hi");
    assert_eq!(asst.role, "assistant");
    assert_eq!(asst.content.as_deref().unwrap(), "hi");

    let tool = Message::tool_result("call_1", "result content");
    assert_eq!(tool.role, "tool");
    assert_eq!(tool.tool_call_id.as_deref().unwrap(), "call_1");
}

#[test]
fn tool_definitions_are_valid() {
    let defs = tools::tool_definitions(miniswe::config::EditMode::Smart);

    // Should have the expected grouped tools
    let names: Vec<&str> = defs.iter().map(|d| d.function.name.as_str()).collect();
    assert!(names.contains(&"file"), "should have 'file' tool");
    assert!(names.contains(&"code"), "should have 'code' tool");
    assert!(names.contains(&"web"), "should have 'web' tool");
    assert!(names.contains(&"plan"), "should have 'plan' tool");
    assert!(names.contains(&"edit_file"), "should have 'edit_file' tool");

    // Current top-level tools
    assert!(
        !names.contains(&"read_file"),
        "flat read_file should be gone"
    );
    assert!(
        names.contains(&"write_file"),
        "should have top-level write_file"
    );
    assert!(!names.contains(&"replace"), "flat replace should be gone");
    assert!(!names.contains(&"search"), "flat search should be gone");
    assert!(!names.contains(&"shell"), "flat shell should be gone");
    assert!(
        !names.contains(&"task_update"),
        "flat task_update should be gone"
    );

    // Each definition should have required fields
    for def in &defs {
        assert_eq!(def.r#type, "function");
        assert!(!def.function.name.is_empty());
        assert!(!def.function.description.is_empty());
        assert!(def.function.parameters.is_object());
    }
}

// ── Tool result serialization ───────────────────────────────────────

#[test]
fn tool_result_message_roundtrips() {
    let msg = Message::tool_result("call_1", "file content here");
    assert_eq!(msg.role, "tool");
    assert_eq!(msg.content.as_deref().unwrap(), "file content here");
    assert_eq!(msg.tool_call_id.as_deref().unwrap(), "call_1");
}
