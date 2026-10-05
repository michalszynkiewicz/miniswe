//! Live, wire-level tests for the provider dialects, driven through a real
//! [`LlmClient`] against a [`wiremock`] server. Unit-level dialect logic
//! (provider parsing/classification, endpoint rules, body shaping, the
//! retry-after marker roundtrip, secret inheritance) is covered closer to
//! the code in `kind.rs`, `endpoint.rs`, `shape.rs`, `auth.rs`,
//! `../errors.rs`, and `config::secrets`. This file is for genuinely
//! integration-level behavior: what actually goes out over HTTP (and comes
//! back) through [`LlmClient`] for each dialect. See
//! `docs/hosted-providers.md` for the decision table these assert against.

use serde_json::{Value, json};
use wiremock::matchers::{bearer_token, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::config::ModelConfig;
use crate::llm::{ChatRequest, LlmClient, Message};

/// A `ModelConfig` pointed at a mock server, with short timeouts so a
/// misconfigured test fails fast instead of hanging for the real defaults
/// (30s/120s/600s).
fn config_for(provider: &str, endpoint: &str) -> ModelConfig {
    ModelConfig {
        provider: provider.into(),
        endpoint: endpoint.into(),
        model: "test-model".into(),
        max_retries: 1,
        request_timeout_secs: 5,
        stream_idle_timeout_secs: 5,
        request_deadline_secs: 10,
        ..ModelConfig::default()
    }
}

fn request() -> ChatRequest {
    ChatRequest {
        messages: vec![Message::user("hi")],
        ..Default::default()
    }
}

/// A request with thinking-mode reasoning requested, same signal every
/// real call site threads through (`enable_thinking` in
/// `chat_template_kwargs`) — see `shape::thinking_requested`.
fn thinking_request() -> ChatRequest {
    ChatRequest {
        messages: vec![Message::user("hi")],
        chat_template_kwargs: Some(json!({"enable_thinking": true})),
        ..Default::default()
    }
}

fn mock_text_response(content: &str) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({
        "choices": [{
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
    }))
}

/// Bodies of every request the mock server received so far, parsed as JSON,
/// in arrival order.
async fn chat_request_bodies(server: &MockServer) -> Vec<Value> {
    server
        .received_requests()
        .await
        .expect("mock server records requests")
        .into_iter()
        .map(|r| serde_json::from_slice(&r.body).expect("request body is JSON"))
        .collect()
}

#[tokio::test]
async fn llama_cpp_body_is_unchanged() {
    // This is benchmarked code: the llama-cpp wire body must keep exactly
    // the field set it had before the provider dialect layer existed.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(mock_text_response("hi back"))
        .mount(&server)
        .await;

    let client = LlmClient::new(config_for("llama-cpp", &server.uri()));
    client.chat(&request()).await.expect("chat succeeds");

    let bodies = chat_request_bodies(&server).await;
    assert_eq!(bodies.len(), 1);
    let obj = bodies[0].as_object().unwrap();
    let mut keys: Vec<&str> = obj.keys().map(String::as_str).collect();
    keys.sort_unstable();
    let mut expected = vec!["max_tokens", "messages", "model", "stream", "temperature"];
    expected.sort_unstable();
    assert_eq!(
        keys, expected,
        "llama-cpp request body field set must stay byte-for-byte identical"
    );
}

#[tokio::test]
async fn llama_cpp_with_no_key_sends_no_authorization_header() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(mock_text_response("ok"))
        .mount(&server)
        .await;

    let client = LlmClient::new(config_for("llama-cpp", &server.uri()));
    client.chat(&request()).await.expect("chat succeeds");

    let reqs = server.received_requests().await.unwrap();
    assert!(
        reqs[0].headers.get("authorization").is_none(),
        "a local provider with no key configured must send no auth header"
    );
}

#[tokio::test]
async fn local_provider_with_explicit_key_still_sends_bearer_auth() {
    // Decision-table row: local dialects send bearer auth IF a key is
    // configured — not just hosted providers.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(bearer_token("local-secret"))
        .respond_with(mock_text_response("ok"))
        .mount(&server)
        .await;

    let mut config = config_for("llama-cpp", &server.uri());
    config.api_key = Some("local-secret".into());
    let client = LlmClient::new(config);
    client
        .chat(&request())
        .await
        .expect("bearer auth must reach the mock for the request to match");
}

#[tokio::test]
async fn openrouter_sends_reasoning_object_and_omits_stream_options() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(bearer_token("or-key"))
        .respond_with(mock_text_response("ok"))
        .mount(&server)
        .await;

    let mut config = config_for("openrouter", &server.uri());
    config.api_key = Some("or-key".into());
    config.thinking_effort = "high".into();
    let client = LlmClient::new(config);
    client
        .chat(&thinking_request())
        .await
        .expect("chat succeeds");

    let bodies = chat_request_bodies(&server).await;
    let body = &bodies[0];
    assert_eq!(body["reasoning"], json!({"effort": "high"}));
    assert!(body.get("stream_options").is_none());
    assert!(body.get("chat_template_kwargs").is_none());
    assert!(body.get("cache_prompt").is_none());
}

#[tokio::test]
async fn openai_sends_max_completion_tokens_and_never_sends_temperature() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(bearer_token("oa-key"))
        .respond_with(mock_text_response("ok"))
        .mount(&server)
        .await;

    let mut config = config_for("openai", &server.uri());
    config.api_key = Some("oa-key".into());
    let client = LlmClient::new(config);
    client
        .chat(&thinking_request())
        .await
        .expect("chat succeeds");

    let bodies = chat_request_bodies(&server).await;
    let body = &bodies[0];
    assert!(body.get("max_completion_tokens").is_some());
    assert!(body.get("max_tokens").is_none());
    assert!(body.get("temperature").is_none());
    assert_eq!(body["stream_options"], json!({"include_usage": true}));
    assert_eq!(body["reasoning_effort"], Value::String("medium".into()));
}

#[tokio::test]
async fn usage_is_parsed_and_accumulated_across_two_calls() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "total_tokens": 110,
                "prompt_tokens_details": {"cached_tokens": 40}
            }
        })))
        .mount(&server)
        .await;

    let client = LlmClient::new(config_for("llama-cpp", &server.uri()));
    client.chat(&request()).await.expect("first call");
    client.chat(&request()).await.expect("second call");

    let snap = client.usage_snapshot();
    assert_eq!(snap.prompt_tokens, 200);
    assert_eq!(snap.completion_tokens, 20);
    assert_eq!(snap.cached_tokens, 80);
    assert_eq!(snap.calls, 2);
}

#[tokio::test]
async fn missing_api_key_fails_fast_before_any_request() {
    // SAFETY: test-only env scrub of the real conventional var, restored
    // before returning so no other test (or the real shell) is affected.
    let saved = std::env::var("OPENAI_API_KEY").ok();
    unsafe {
        std::env::remove_var("OPENAI_API_KEY");
    }

    // Endpoint is never actually dialed: check_credentials must bail
    // before any request goes out.
    let client = LlmClient::new(config_for("openai", "http://127.0.0.1:1"));
    let err = client
        .check_credentials()
        .expect_err("no key configured anywhere for a hosted provider");
    assert!(err.to_string().contains("OPENAI_API_KEY"));

    if let Some(v) = saved {
        unsafe {
            std::env::set_var("OPENAI_API_KEY", v);
        }
    }
}

#[tokio::test]
async fn env_var_api_key_resolves_and_is_sent() {
    // SAFETY: test-only env var, unique name avoids collisions with other
    // tests running in parallel in this process.
    unsafe {
        std::env::set_var("MINISWE_TEST_PROVIDERS_LIVE_KEY", "env-secret");
    }

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(bearer_token("env-secret"))
        .respond_with(mock_text_response("ok"))
        .mount(&server)
        .await;

    let mut config = config_for("openrouter", &server.uri());
    config.api_key_env = Some("MINISWE_TEST_PROVIDERS_LIVE_KEY".into());
    let client = LlmClient::new(config);
    client
        .chat(&request())
        .await
        .expect("the env-resolved key must reach the mock for the request to match");

    unsafe {
        std::env::remove_var("MINISWE_TEST_PROVIDERS_LIVE_KEY");
    }
}

#[tokio::test]
async fn hosted_probe_looks_up_the_configured_model_instead_of_taking_the_first() {
    // A hosted /v1/models lists the whole catalogue; the first id is
    // arbitrary and must never become `probed_model` (it drives the
    // model-family carve-outs). The probe must find the configured model
    // in the list, and fail when it's absent.
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [
                {"id": "openai/gpt-oss-120b"},
                {"id": "test-model"},
            ]
        })))
        .mount(&server)
        .await;

    let mut config = config_for("openrouter", &server.uri());
    config.api_key = Some("or-key".into());
    let client = LlmClient::new(config);
    assert_eq!(client.probe_model().await.unwrap(), "test-model");

    let mut missing = config_for("openrouter", &server.uri());
    missing.api_key = Some("or-key".into());
    missing.model = "nope/not-there".into();
    let err = LlmClient::new(missing).probe_model().await.unwrap_err();
    assert!(err.to_string().contains("not listed"), "{err}");

    // A local server keeps the first-id behavior (llama-server reports
    // the GGUF path, not the config alias).
    let local = LlmClient::new(config_for("llama-cpp", &server.uri()));
    assert_eq!(local.probe_model().await.unwrap(), "openai/gpt-oss-120b");
}

#[tokio::test]
async fn llama_cpp_probe_reads_context_window_from_props() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [{"id": "test-model"}]
        })))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/props"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "default_generation_settings": {"n_ctx": 60000},
            "total_slots": 1
        })))
        .mount(&server)
        .await;

    let client = LlmClient::new(config_for("llama-cpp", &server.uri()));
    let probe = client.probe().await.expect("probe succeeds");
    assert_eq!(probe.model, "test-model");
    assert_eq!(probe.context_window, Some(60000));
}

#[tokio::test]
async fn llama_cpp_probe_tolerates_missing_props_endpoint() {
    // /props returning 404 (an older or stripped-down server) must not
    // fail the probe — the model identity is still useful without a
    // context window.
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [{"id": "test-model"}]
        })))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/props"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;

    let client = LlmClient::new(config_for("llama-cpp", &server.uri()));
    let probe = client
        .probe()
        .await
        .expect("probe succeeds despite /props 404");
    assert_eq!(probe.model, "test-model");
    assert_eq!(probe.context_window, None);
}

#[tokio::test]
async fn vllm_probe_reads_max_model_len_and_never_calls_props() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [{"id": "test-model", "max_model_len": 32768}]
        })))
        .mount(&server)
        .await;
    // Deliberately not mounted: vllm's /v1/models already carried a
    // window, so no /props request should ever go out.

    let client = LlmClient::new(config_for("vllm", &server.uri()));
    let probe = client.probe().await.expect("probe succeeds");
    assert_eq!(probe.context_window, Some(32768));

    let reqs = server.received_requests().await.unwrap();
    assert!(
        reqs.iter().all(|r| r.url.path() != "/props"),
        "vllm probe must not fall through to /props when /v1/models already has a window"
    );
}

#[tokio::test]
async fn openrouter_probe_matches_context_length_of_configured_model() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [
                {"id": "openai/gpt-oss-120b", "context_length": 131072},
                {"id": "test-model", "context_length": 200000},
            ]
        })))
        .mount(&server)
        .await;

    let mut config = config_for("openrouter", &server.uri());
    config.api_key = Some("or-key".into());
    let client = LlmClient::new(config);
    let probe = client.probe().await.expect("probe succeeds");
    assert_eq!(probe.context_window, Some(200000));
}

#[tokio::test]
async fn openai_probe_has_no_context_window() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [{"id": "test-model"}]
        })))
        .mount(&server)
        .await;

    let mut config = config_for("openai", &server.uri());
    config.api_key = Some("oa-key".into());
    let client = LlmClient::new(config);
    let probe = client.probe().await.expect("probe succeeds");
    assert_eq!(probe.context_window, None);
}

#[tokio::test]
async fn retry_after_429_then_200_succeeds() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("Retry-After", "1")
                .set_body_string("slow down"),
        )
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(mock_text_response("recovered"))
        .mount(&server)
        .await;

    let client = LlmClient::new(config_for("llama-cpp", &server.uri()));
    let resp = client
        .chat(&request())
        .await
        .expect("retries once after a 429 and succeeds");
    assert_eq!(
        resp.choices[0].message.content.as_deref(),
        Some("recovered")
    );
}

// ---------------------------------------------------------------------
// Anthropic native Messages client
// ---------------------------------------------------------------------

fn sse(events: &[Value]) -> ResponseTemplate {
    let body: String = events
        .iter()
        .map(|e| format!("event: {}\ndata: {e}\n\n", e["type"].as_str().unwrap()))
        .collect();
    ResponseTemplate::new(200).set_body_raw(body, "text/event-stream")
}

fn text_stream(text: &str) -> ResponseTemplate {
    sse(&[
        json!({"type": "message_start", "message": {"usage": {"input_tokens": 7}}}),
        json!({"type": "content_block_start", "index": 0,
               "content_block": {"type": "text", "text": ""}}),
        json!({"type": "content_block_delta", "index": 0,
               "delta": {"type": "text_delta", "text": text}}),
        json!({"type": "content_block_stop", "index": 0}),
        json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"},
               "usage": {"output_tokens": 3}}),
        json!({"type": "message_stop"}),
    ])
}

async fn mount_messages(server: &MockServer, resp: ResponseTemplate) {
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(resp)
        .mount(server)
        .await;
}

async fn mount_model_caps(server: &MockServer, adaptive: bool) {
    Mock::given(method("GET"))
        .and(path("/v1/models/test-model"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "capabilities": {"thinking": {"types": {"adaptive": {"supported": adaptive}}}}
        })))
        .mount(server)
        .await;
}

fn anthropic_config(server: &MockServer) -> ModelConfig {
    let mut c = config_for("anthropic", &server.uri());
    c.api_key = Some("an-key".into());
    c
}

async fn messages_bodies(server: &MockServer) -> Vec<Value> {
    server
        .received_requests()
        .await
        .unwrap()
        .into_iter()
        .filter(|r| r.method.as_str() == "POST")
        .map(|r| serde_json::from_slice(&r.body).unwrap())
        .collect()
}

#[tokio::test]
async fn anthropic_posts_to_messages_with_headers_and_betas() {
    let server = MockServer::start().await;
    mount_model_caps(&server, true).await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", "an-key"))
        .and(header("anthropic-version", "2023-06-01"))
        .and(header(
            "anthropic-beta",
            "thinking-binding-controls-2026-08-01",
        ))
        .respond_with(text_stream("ok"))
        .mount(&server)
        .await;
    let client = LlmClient::new(anthropic_config(&server));
    let resp = client.chat(&thinking_request()).await.expect("matches");
    assert_eq!(resp.choices[0].message.content.as_deref(), Some("ok"));
    assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("stop"));
    let snap = client.usage_snapshot();
    assert_eq!((snap.prompt_tokens, snap.completion_tokens), (7, 3));

    let b = &messages_bodies(&server).await[0];
    assert_eq!(b["thinking"]["type"], "adaptive");
    assert_eq!(b["output_config"], json!({"effort": "medium"}));
    assert_eq!(b["max_tokens"], 32_000);
    for banned in [
        "temperature",
        "stream_options",
        "chat_template_kwargs",
        "cache_prompt",
    ] {
        assert!(b.get(banned).is_none(), "{banned}");
    }
}

#[tokio::test]
async fn anthropic_capability_lookup_happens_once_per_client() {
    let server = MockServer::start().await;
    mount_model_caps(&server, true).await;
    mount_messages(&server, text_stream("ok")).await;
    let client = LlmClient::new(anthropic_config(&server));
    client.chat(&request()).await.unwrap();
    client.chat(&request()).await.unwrap();
    let gets = server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .filter(|r| r.method.as_str() == "GET")
        .count();
    assert_eq!(gets, 1);
}

#[tokio::test]
async fn anthropic_lookup_failure_assumes_adaptive() {
    let server = MockServer::start().await; // GET /models/.. -> 404
    mount_messages(&server, text_stream("ok")).await;
    let client = LlmClient::new(anthropic_config(&server));
    client.chat(&request()).await.unwrap();
    let b = &messages_bodies(&server).await[0];
    assert_eq!(b["thinking"]["type"], "adaptive");
    assert_eq!(b["output_config"], json!({"effort": "low"}));
}

#[tokio::test]
async fn anthropic_haiku_shapes_thinking_off_and_on() {
    let server = MockServer::start().await;
    mount_model_caps(&server, false).await;
    mount_messages(&server, text_stream("ok")).await;
    let mut cfg = anthropic_config(&server);
    cfg.max_output_tokens = 1000;
    cfg.thinking_budget_tokens = 2048;
    let client = LlmClient::new(cfg);
    client.chat(&request()).await.unwrap();
    client.chat(&thinking_request()).await.unwrap();
    let bodies = messages_bodies(&server).await;
    assert!(bodies[0].get("thinking").is_none());
    assert_eq!(bodies[0]["max_tokens"], 1000);
    assert_eq!(bodies[1]["thinking"]["type"], "enabled");
    assert_eq!(bodies[1]["thinking"]["budget_tokens"], 2048);
    assert_eq!(bodies[1]["max_tokens"], 3072);
    assert!(bodies[1].get("output_config").is_none());
}

#[tokio::test]
async fn anthropic_per_request_effort_and_fallbacks() {
    let server = MockServer::start().await;
    mount_model_caps(&server, true).await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(text_stream("ok"))
        .mount(&server)
        .await;
    let mut cfg = anthropic_config(&server);
    cfg.model = "claude-opus-5-5".into();
    Mock::given(method("GET"))
        .and(path("/v1/models/claude-opus-5-5"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;
    let client = LlmClient::new(cfg);
    let mut req = request();
    req.chat_template_kwargs = Some(json!({"reasoning_effort": "high"}));
    client.chat(&req).await.expect("both betas sent");
    let b = &messages_bodies(&server).await[0];
    assert_eq!(b["fallbacks"], "default");
    let post = server
        .received_requests()
        .await
        .unwrap()
        .into_iter()
        .find(|r| r.method.as_str() == "POST")
        .unwrap();
    assert_eq!(
        post.headers["anthropic-beta"].to_str().unwrap(),
        "thinking-binding-controls-2026-08-01,server-side-fallback-2026-07-01"
    );
    assert_eq!(b["output_config"], json!({"effort": "high"}));
}

#[tokio::test]
async fn anthropic_converts_system_tools_results_and_breakpoints() {
    let server = MockServer::start().await;
    mount_model_caps(&server, true).await;
    mount_messages(&server, text_stream("ok")).await;
    let client = LlmClient::new(anthropic_config(&server));
    let marker = crate::context::compressor::CURRENT_STATE_MARKER;
    let call = crate::llm::ToolCall {
        id: "t1".into(),
        r#type: "function".into(),
        function: crate::llm::FunctionCall {
            name: "read".into(),
            arguments: "{\"p\":1}".into(),
        },
    };
    let req = ChatRequest {
        messages: vec![
            Message::system("sys"),
            Message::user("go"),
            Message::assistant_tool_calls(vec![call]),
            Message::tool_result("t1", "contents"),
            Message::user(&format!("next{marker}plan")),
        ],
        tools: Some(vec![crate::llm::ToolDefinition {
            r#type: "function".into(),
            function: crate::llm::FunctionDefinition {
                name: "read".into(),
                description: "d".into(),
                parameters: json!({"type": "object"}),
            },
        }]),
        ..Default::default()
    };
    client.chat(&req).await.unwrap();
    let b = &messages_bodies(&server).await[0];
    assert_eq!(b["system"][0]["text"], "sys");
    assert_eq!(b["tools"][0]["input_schema"], json!({"type": "object"}));
    assert_eq!(b["tools"][0]["eager_input_streaming"], true);
    let msgs = b["messages"].as_array().unwrap();
    assert_eq!(
        msgs.len(),
        3,
        "tool result merges with the following user text"
    );
    let last = msgs[2]["content"].as_array().unwrap();
    assert_eq!(last[0]["type"], "tool_result");
    assert_eq!(last[1]["text"], "next");
    assert!(last[1].get("cache_control").is_some());
    assert!(last[2]["text"].as_str().unwrap().starts_with(marker));
    assert!(last[2].get("cache_control").is_none());
    assert_eq!(b.to_string().matches("cache_control").count(), 3);
}

#[tokio::test]
async fn anthropic_without_marker_puts_third_breakpoint_on_last_block() {
    let server = MockServer::start().await;
    mount_model_caps(&server, true).await;
    mount_messages(&server, text_stream("ok")).await;
    let client = LlmClient::new(anthropic_config(&server));
    let mut req = request();
    req.messages.insert(0, Message::system("sys"));
    client.chat(&req).await.unwrap();
    let b = &messages_bodies(&server).await[0];
    assert!(
        b["messages"][0]["content"][0]
            .get("cache_control")
            .is_some()
    );
    assert_eq!(b.to_string().matches("cache_control").count(), 2);
}

#[tokio::test]
async fn anthropic_tool_use_thinking_and_replay_verbatim() {
    let server = MockServer::start().await;
    mount_model_caps(&server, true).await;
    mount_messages(
        &server,
        sse(&[
            json!({"type": "message_start", "message": {"usage": {
                "input_tokens": 5, "cache_creation_input_tokens": 20,
                "cache_read_input_tokens": 100}}}),
            json!({"type": "content_block_start", "index": 0,
                   "content_block": {"type": "thinking", "thinking": ""}}),
            json!({"type": "content_block_delta", "index": 0,
                   "delta": {"type": "thinking_delta", "thinking": "plan"}}),
            json!({"type": "content_block_delta", "index": 0,
                   "delta": {"type": "signature_delta", "signature": "SIG"}}),
            json!({"type": "content_block_start", "index": 1,
                   "content_block": {"type": "tool_use", "id": "t9", "name": "read", "input": {}}}),
            json!({"type": "content_block_delta", "index": 1,
                   "delta": {"type": "input_json_delta", "partial_json": "{\"p\""}}),
            json!({"type": "content_block_delta", "index": 1,
                   "delta": {"type": "input_json_delta", "partial_json": ": 2}"}}),
            json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"},
                   "usage": {"output_tokens": 9}}),
            json!({"type": "message_stop"}),
        ]),
    )
    .await;
    let client = LlmClient::new(anthropic_config(&server));
    let resp = client.chat(&request()).await.unwrap();
    let msg = resp.choices[0].message.clone();
    assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("tool_calls"));
    assert_eq!(
        msg.tool_calls.as_ref().unwrap()[0].function.arguments,
        "{\"p\": 2}"
    );
    let snap = client.usage_snapshot();
    assert_eq!(snap.prompt_tokens, 125);
    assert_eq!(snap.cached_tokens, 100);
    assert_eq!(snap.cache_write_tokens, 20);

    let mut req = request();
    req.messages.push(msg);
    req.messages.push(Message::tool_result("t9", "file"));
    client.chat(&req).await.unwrap();
    let b = &messages_bodies(&server).await[1];
    assert_eq!(
        b["messages"][1]["content"],
        json!([
            {"type": "thinking", "thinking": "plan", "signature": "SIG"},
            {"type": "tool_use", "id": "t9", "name": "read", "input": {"p": 2}},
        ])
    );
}

#[tokio::test]
async fn anthropic_edited_tool_arguments_drop_the_thinking_on_replay() {
    let server = MockServer::start().await;
    mount_model_caps(&server, true).await;
    mount_messages(&server, text_stream("ok")).await;
    let client = LlmClient::new(anthropic_config(&server));
    let mut asst = Message::assistant_tool_calls(vec![crate::llm::ToolCall {
        id: "t".into(),
        r#type: "function".into(),
        function: crate::llm::FunctionCall {
            name: "read".into(),
            arguments: "{\"p\":999}".into(),
        },
    }]);
    asst.provider_blocks = Some(vec![
        json!({"type": "thinking", "thinking": "x", "signature": "S"}),
        json!({"type": "tool_use", "id": "t", "name": "read", "input": {"p": 1}}),
    ]);
    let mut req = request();
    req.messages.push(asst);
    req.messages.push(Message::tool_result("t", "r"));
    client.chat(&req).await.unwrap();
    let b = &messages_bodies(&server).await[0];
    assert_eq!(
        b["messages"][1]["content"],
        json!([{"type": "tool_use", "id": "t", "name": "read", "input": {"p": 999}}])
    );
}

#[tokio::test]
async fn anthropic_refusal_is_a_non_retried_error() {
    let server = MockServer::start().await;
    mount_model_caps(&server, true).await;
    mount_messages(
        &server,
        sse(&[
            json!({"type": "message_start", "message": {"usage": {"input_tokens": 1}}}),
            json!({"type": "message_delta",
                   "delta": {"stop_reason": "refusal", "stop_details": {"category": "cyber"}},
                   "usage": {"output_tokens": 0}}),
            json!({"type": "message_stop"}),
        ]),
    )
    .await;
    let client = LlmClient::new(anthropic_config(&server));
    let err = client.chat(&request()).await.unwrap_err().to_string();
    assert!(err.contains("refusal") && err.contains("cyber"), "{err}");
    assert_eq!(messages_bodies(&server).await.len(), 1, "no retry");
}

#[tokio::test]
async fn anthropic_stream_error_event_overloaded_is_retried() {
    let server = MockServer::start().await;
    mount_model_caps(&server, true).await;
    mount_messages(
        &server,
        sse(&[json!({"type": "error",
                     "error": {"type": "overloaded_error", "message": "Overloaded"}})]),
    )
    .await;
    let client = LlmClient::new(anthropic_config(&server));
    let err = client.chat(&request()).await.unwrap_err().to_string();
    assert!(err.contains("overloaded_error"), "{err}");
    assert_eq!(messages_bodies(&server).await.len(), 2, "1 try + 1 retry");
}

#[tokio::test]
async fn anthropic_529_is_retried() {
    let server = MockServer::start().await;
    mount_model_caps(&server, true).await;
    mount_messages(
        &server,
        ResponseTemplate::new(529).set_body_string("overloaded"),
    )
    .await;
    let client = LlmClient::new(anthropic_config(&server));
    assert!(client.chat(&request()).await.is_err());
    assert_eq!(messages_bodies(&server).await.len(), 2);
}

#[tokio::test]
async fn anthropic_max_tokens_stop_maps_to_length() {
    let server = MockServer::start().await;
    mount_model_caps(&server, true).await;
    mount_messages(
        &server,
        sse(&[
            json!({"type": "content_block_start", "index": 0,
                   "content_block": {"type": "text", "text": ""}}),
            json!({"type": "content_block_delta", "index": 0,
                   "delta": {"type": "text_delta", "text": "cut"}}),
            json!({"type": "message_delta", "delta": {"stop_reason": "max_tokens"},
                   "usage": {"output_tokens": 3}}),
            json!({"type": "message_stop"}),
        ]),
    )
    .await;
    let client = LlmClient::new(anthropic_config(&server));
    let resp = client.chat(&request()).await.unwrap();
    assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("length"));
}

#[tokio::test]
async fn anthropic_non_streaming_json_body_is_accepted() {
    let server = MockServer::start().await;
    mount_model_caps(&server, true).await;
    mount_messages(
        &server,
        ResponseTemplate::new(200).set_body_json(json!({
            "type": "message", "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "whole"}],
            "usage": {"input_tokens": 4, "output_tokens": 2}
        })),
    )
    .await;
    let client = LlmClient::new(anthropic_config(&server));
    let mut streamed = String::new();
    let resp = client
        .chat_stream(
            &request(),
            |t| streamed.push_str(t),
            &std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        )
        .await
        .unwrap();
    assert_eq!(resp.choices[0].message.content.as_deref(), Some("whole"));
    assert_eq!(streamed, "whole");
}
