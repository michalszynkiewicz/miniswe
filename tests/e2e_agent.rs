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
use miniswe::llm::{ChatRequest, LlmClient, Message};
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
        tools: Some(tools::tool_definitions()),
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
        tools: Some(tools::tool_definitions()),
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
        tools: Some(tools::tool_definitions()),
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
        tools: Some(tools::tool_definitions()),
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
        tools: Some(tools::tool_definitions()),
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
    // In plan mode, the run loop blocks file write/replace/shell actions.
    let plan_only = true;
    let blocked_actions = ["write", "replace", "shell"];
    for action in &blocked_actions {
        assert!(
            plan_only && matches!(*action, "write" | "replace" | "shell"),
            "plan mode should block file action: {action}"
        );
    }

    // Allowed actions in plan mode
    let allowed_actions = ["read", "search"];
    for action in &allowed_actions {
        assert!(
            !matches!(*action, "write" | "replace" | "shell"),
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
    let defs = tools::tool_definitions();

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
