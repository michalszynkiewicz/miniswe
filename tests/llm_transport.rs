//! LLM transport: refused connections retry once with a readable cause,
//! and llama.cpp streams (which only report `timings`) still yield usage.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use buzztr::config::{Config, ModelRole};
use buzztr::llm::{ChatRequest, Message, ModelRouter};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn request() -> ChatRequest {
    ChatRequest {
        messages: vec![Message::user("hi")],
        tools: None,
        tool_choice: None,
        max_tokens_override: None,
        chat_template_kwargs: None,
        temperature_override: None,
        cache_prompt: None,
    }
}

fn router_for(endpoint: &str, provider: &str, max_retries: usize) -> ModelRouter {
    let mut config = Config::default();
    config.model.endpoint = endpoint.to_string();
    config.model.provider = provider.to_string();
    config.model.max_retries = max_retries;
    ModelRouter::new(&config)
}

fn closed_port_endpoint() -> String {
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    drop(l);
    format!("http://127.0.0.1:{port}")
}

#[tokio::test]
async fn refused_connection_non_streaming_retries_once_with_cause() {
    let router = router_for(&closed_port_endpoint(), "openai-compatible", 6);
    let t = Instant::now();
    let err = router
        .chat(ModelRole::Default, &request())
        .await
        .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("Failed to connect to LLM at"), "{msg}");
    assert!(msg.contains("connection refused"), "{msg}");
    assert!(t.elapsed() < Duration::from_secs(5), "{:?}", t.elapsed());
}

#[tokio::test]
async fn refused_connection_streaming_retries_once_with_cause() {
    let router = router_for(&closed_port_endpoint(), "openai-compatible", 6);
    let cancelled = Arc::new(AtomicBool::new(false));
    let t = Instant::now();
    let err = router
        .chat_stream(ModelRole::Default, &request(), |_| {}, &cancelled)
        .await
        .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("Failed to connect to LLM at"), "{msg}");
    assert!(msg.contains("connection refused"), "{msg}");
    assert!(t.elapsed() < Duration::from_secs(5), "{:?}", t.elapsed());
}

#[tokio::test]
async fn llama_cpp_stream_timings_feed_usage_totals() {
    let server = MockServer::start().await;
    let chunk1 = serde_json::json!({"choices":[{"delta":{"content":"hello"},"finish_reason":null,"index":0}]});
    let last = serde_json::json!({
        "choices":[{"finish_reason":"stop","index":0,"delta":{}}],
        "created":1,"id":"x","model":"m","object":"chat.completion.chunk",
        "timings":{"cache_n":3,"prompt_n":55,"prompt_ms":20.1,"prompt_per_token_ms":0.36,
            "prompt_per_second":2700.0,"predicted_n":20,"predicted_ms":200.0,
            "predicted_per_token_ms":10.0,"predicted_per_second":100.0}
    });
    let body = format!("data: {chunk1}\n\ndata: {last}\n\ndata: [DONE]\n\n");
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
        .mount(&server)
        .await;

    let router = router_for(&server.uri(), "llama-cpp", 0);
    let cancelled = Arc::new(AtomicBool::new(false));
    let resp = router
        .chat_stream(ModelRole::Default, &request(), |_| {}, &cancelled)
        .await
        .unwrap();
    assert_eq!(resp.usage.as_ref().unwrap().prompt_tokens, 58);
    let totals = router.usage_totals();
    assert_eq!(totals.prompt_tokens, 58);
    assert_eq!(totals.completion_tokens, 20);
    assert_eq!(totals.cached_tokens, 3);
    assert_eq!(totals.calls, 1);
}
