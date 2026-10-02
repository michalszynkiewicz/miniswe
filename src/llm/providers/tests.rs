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
async fn openai_sends_max_completion_tokens_and_no_temperature_when_thinking() {
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
async fn anthropic_sends_thinking_object_and_full_auth_headers() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(bearer_token("an-key"))
        .and(header("x-api-key", "an-key"))
        .and(header("anthropic-version", "2023-06-01"))
        .respond_with(mock_text_response("ok"))
        .mount(&server)
        .await;

    let mut config = config_for("anthropic", &server.uri());
    config.api_key = Some("an-key".into());
    config.thinking_budget_tokens = 1024;
    let client = LlmClient::new(config);
    client
        .chat(&thinking_request())
        .await
        .expect("auth headers + body must reach the mock for the request to match");

    let bodies = chat_request_bodies(&server).await;
    let body = &bodies[0];
    assert!(body.get("temperature").is_none());
    assert_eq!(
        body["thinking"],
        json!({"type": "enabled", "budget_tokens": 1024})
    );
    assert_eq!(body["stream_options"], json!({"include_usage": true}));
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
