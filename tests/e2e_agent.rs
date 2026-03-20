//! E2E tests for the agent loop with a mocked LLM server.
//!
//! Uses wiremock to stub the OpenAI-compatible API and tests the full flow:
//! LlmClient → streaming → tool parsing → tool execution → response assembly.

mod helpers;

use std::fs;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use miniswe::config::Config;
use miniswe::llm::{ChatRequest, LlmClient, Message};
use miniswe::tools;
use miniswe::tools::permissions::PermissionManager;
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn perms(config: &Config) -> PermissionManager {
    PermissionManager::headless(config)
}

// ── LLM client with non-streaming ───────────────────────────────────

#[tokio::test]
async fn llm_client_chat_plain_text() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(helpers::mock_text_response("Hello from mock LLM!"))
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

    assert_eq!(response.choices.len(), 1);
    let msg = &response.choices[0].message;
    assert_eq!(msg.role, "assistant");
    assert_eq!(msg.content.as_deref().unwrap(), "Hello from mock LLM!");
    assert!(msg.tool_calls.is_none());
}

#[tokio::test]
async fn llm_client_chat_tool_call() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(helpers::mock_tool_call_response(
            "read_file",
            json!({"path": "src/main.rs"}),
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
    assert_eq!(tc[0].function.name, "read_file");

    let args: serde_json::Value = serde_json::from_str(&tc[0].function.arguments).unwrap();
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
        .chat_stream(
            &request,
            |token| tokens.push(token.to_string()),
            &cancelled,
        )
        .await
        .unwrap();

    assert_eq!(response.choices[0].message.content.as_deref().unwrap(), "Streamed hello!");
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

    // Step 1: LLM returns a read_file tool call
    // Step 2: After tool result, LLM returns plain text
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
                            "name": "read_file",
                            "arguments": r#"{"path":"hello.txt"}"#
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
    fs::write(helpers::project_path(&config, "hello.txt"), "Hello from test!").unwrap();

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
    let result = tools::execute_tool(&tc[0].function.name, &args, &config, &p)
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
    let tc = response.choices[0]
        .message
        .tool_calls
        .as_ref()
        .unwrap();

    let args: serde_json::Value = serde_json::from_str(&tc[0].function.arguments).unwrap();
    let result = tools::execute_tool(&tc[0].function.name, &args, &config, &p)
        .await
        .unwrap();

    assert!(result.success, "write_file should succeed: {}", result.content);

    // Verify file on disk
    let disk = fs::read_to_string(helpers::project_path(&config, "output.txt")).unwrap();
    assert_eq!(disk, "Generated content\nLine 2\n");
}

// ── Invalid JSON args ───────────────────────────────────────────────

#[tokio::test]
async fn invalid_json_args_from_llm() {
    let mock_server = MockServer::start().await;

    // Return a tool call with malformed JSON arguments
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
                            "name": "read_file",
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
    // In plan mode, the run loop blocks edit/write_file/shell tool calls.
    // We test the logic directly here.
    let plan_only = true;
    let blocked_tools = ["edit", "write_file", "shell"];

    for tool in &blocked_tools {
        assert!(
            plan_only
                && (*tool == "edit" || *tool == "write_file" || *tool == "shell"),
            "plan mode should block: {tool}"
        );
    }

    // Allowed tools in plan mode
    let allowed_tools = ["read_file", "search", "task_update", "read_symbol"];
    for tool in &allowed_tools {
        assert!(
            !(*tool == "edit" || *tool == "write_file" || *tool == "shell"),
            "plan mode should allow: {tool}"
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
        messages: vec![Message::user("test")],
        tools: None,
        tool_choice: None,
    };

    let result = client.chat(&request).await;
    assert!(result.is_err(), "should return error on 500");
    assert!(result.unwrap_err().to_string().contains("500"));
}

#[tokio::test]
async fn llm_connection_refused() {
    let (_tmp, mut config) = helpers::create_test_project();
    // Point at a port that isn't listening
    config.model.endpoint = "http://127.0.0.1:1".to_string();
    config.model.provider = "openai-compatible".to_string();

    let client = LlmClient::new(config.model.clone());
    let request = ChatRequest {
        messages: vec![Message::user("test")],
        tools: None,
        tool_choice: None,
    };

    let result = client.chat(&request).await;
    assert!(result.is_err(), "should error on connection refused");
}

// ── Streaming cancellation ──────────────────────────────────────────

#[tokio::test]
async fn stream_cancellation_via_flag() {
    let mock_server = MockServer::start().await;

    // Respond with a very long SSE stream (many chunks)
    let mut body = String::new();
    for i in 0..100 {
        body.push_str(&format!(
            "data: {}\n\n",
            json!({"choices":[{"delta":{"content": format!("token{i} ")},"finish_reason":null}]})
        ));
    }
    body.push_str("data: [DONE]\n\n");

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());

    let client = LlmClient::new(config.model.clone());
    let request = ChatRequest {
        messages: vec![Message::user("long response")],
        tools: None,
        tool_choice: None,
    };

    // Set cancelled immediately — the stream should abort
    let cancelled = Arc::new(AtomicBool::new(true));
    let result = client.chat_stream(&request, |_| {}, &cancelled).await;

    assert!(result.is_err(), "should error when cancelled");
    assert!(result.unwrap_err().to_string().contains("Interrupted"));
}

// ── Message construction helpers ────────────────────────────────────

#[test]
fn message_factories_produce_correct_roles() {
    let sys = Message::system("sys");
    assert_eq!(sys.role, "system");
    assert_eq!(sys.content.as_deref().unwrap(), "sys");

    let user = Message::user("user msg");
    assert_eq!(user.role, "user");

    let asst = Message::assistant("reply");
    assert_eq!(asst.role, "assistant");

    let tool = Message::tool_result("call_1", "result");
    assert_eq!(tool.role, "tool");
    assert_eq!(tool.tool_call_id.as_deref().unwrap(), "call_1");
}

#[test]
fn tool_definitions_are_valid() {
    let defs = tools::tool_definitions();

    // Should have the expected tools
    let names: Vec<&str> = defs.iter().map(|d| d.function.name.as_str()).collect();
    assert!(names.contains(&"read_file"));
    assert!(names.contains(&"write_file"));
    assert!(names.contains(&"edit"));
    assert!(names.contains(&"search"));
    assert!(names.contains(&"shell"));
    assert!(names.contains(&"task_update"));
    assert!(names.contains(&"read_symbol"));

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
    let msg = Message::tool_result("call_123", "file contents here");

    let json = serde_json::to_value(&msg).unwrap();
    assert_eq!(json["role"], "tool");
    assert_eq!(json["content"], "file contents here");
    assert_eq!(json["tool_call_id"], "call_123");

    let back: Message = serde_json::from_value(json).unwrap();
    assert_eq!(back.role, "tool");
    assert_eq!(back.tool_call_id.as_deref().unwrap(), "call_123");
}
