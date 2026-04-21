//! LLM interface — OpenAI-compatible API client.
//!
//! Supports llama.cpp server, Ollama, vLLM, and any OpenAI-compatible endpoint.
//! Handles streaming responses and tool call parsing.

pub mod router;
mod types;

pub use router::ModelRouter;
pub use types::*;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use futures::StreamExt;
use reqwest::Client;
use reqwest::StatusCode;
use serde_json::Value;

use crate::config::ModelConfig;

/// Client for communicating with an OpenAI-compatible LLM API.
pub struct LlmClient {
    client: Client,
    config: ModelConfig,
}

impl LlmClient {
    pub fn new(config: ModelConfig) -> Self {
        Self {
            client: Client::new(),
            config,
        }
    }

    /// Build the API URL based on provider type.
    fn chat_url(&self) -> String {
        let base = self.config.endpoint.trim_end_matches('/');
        match self.config.provider.as_str() {
            "ollama" => format!("{base}/api/chat"),
            _ => format!("{base}/v1/chat/completions"),
        }
    }

    /// Ask the server what model it's actually serving, via `/v1/models`
    /// (or `/api/tags` for Ollama). Returns the first id the server reports.
    /// Short timeout so a dead endpoint doesn't stall startup.
    ///
    /// Error messages are kept short and URL-free — the caller already
    /// displays the endpoint alongside the probe result, so we avoid
    /// repeating it.
    pub async fn probe_model(&self) -> Result<String> {
        let base = self.config.endpoint.trim_end_matches('/');
        let (url, ollama) = match self.config.provider.as_str() {
            "ollama" => (format!("{base}/api/tags"), true),
            _ => (format!("{base}/v1/models"), false),
        };
        let resp = match tokio::time::timeout(Duration::from_secs(3), self.client.get(&url).send())
            .await
        {
            Err(_) => bail!("timeout"),
            Ok(Err(e)) if e.is_connect() => bail!("unreachable"),
            Ok(Err(e)) => bail!("transport error ({e})"),
            Ok(Ok(r)) => r,
        };
        if !resp.status().is_success() {
            bail!("HTTP {}", resp.status().as_u16());
        }
        let body: Value = resp.json().await.map_err(|_| anyhow::anyhow!("bad JSON"))?;
        let first = if ollama {
            body["models"]
                .as_array()
                .and_then(|a| a.first())
                .and_then(|m| m["name"].as_str())
                .map(|s| s.to_string())
        } else {
            body["data"]
                .as_array()
                .and_then(|a| a.first())
                .and_then(|m| m["id"].as_str())
                .map(|s| s.to_string())
        };
        first.ok_or_else(|| anyhow::anyhow!("no models listed"))
    }

    pub fn endpoint(&self) -> &str {
        &self.config.endpoint
    }

    /// Send a chat completion request and return the full response.
    pub async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        self.chat_with_cancel(request, None).await
    }

    /// Send a chat request with optional cancellation. Internally streams
    /// the response so we can apply an idle-timeout (kill the request if
    /// no tokens have arrived for `stream_idle_timeout_secs`) and retry
    /// the whole call as a transient failure. The caller still receives
    /// a single non-streaming `ChatResponse` — they don't see tokens
    /// piecewise.
    pub async fn chat_with_cancel(
        &self,
        request: &ChatRequest,
        cancelled: Option<&AtomicBool>,
    ) -> Result<ChatResponse> {
        let url = self.chat_url();
        let retry_delays = [1u64, 2, 4, 8, 16, 32];
        let max_retries = self.config.max_retries.min(retry_delays.len());

        let mut body = serde_json::to_value(request)?;
        // Inject model config — we ALWAYS stream now so we can detect
        // idle connections, even though the public API returns a single
        // ChatResponse to the caller.
        body["model"] = Value::String(self.config.model.clone());
        body["temperature"] = Value::from(self.config.temperature);
        body["max_tokens"] = Value::from(self.config.max_output_tokens);
        body["stream"] = Value::Bool(true);
        let connect_timeout = Duration::from_secs(self.config.request_timeout_secs);
        let idle_timeout = Duration::from_secs(self.config.stream_idle_timeout_secs);

        let mut attempt = 0usize;
        let mut noop = |_: &str| {};
        loop {
            let result = self
                .stream_once_assembled(
                    &url,
                    &body,
                    connect_timeout,
                    idle_timeout,
                    cancelled,
                    &mut noop,
                )
                .await;

            match result {
                Ok(resp) => return Ok(resp),
                Err(err) if attempt < max_retries && is_retryable_llm_error(&err) => {
                    let delay = retry_delays[attempt];
                    attempt += 1;
                    match cancelled {
                        Some(flag) => {
                            tokio::select! {
                                _ = tokio::time::sleep(Duration::from_secs(delay)) => {}
                                _ = wait_for_cancel(flag) => bail!("Interrupted by user"),
                            }
                        }
                        None => tokio::time::sleep(Duration::from_secs(delay)).await,
                    }
                }
                Err(err) => return Err(err),
            }
        }
    }

    /// One streamed attempt: connect, drain SSE chunks (each chunk read
    /// must arrive within `idle_timeout` or we bail), assemble into a
    /// `ChatResponse`. `on_token` fires once per content delta — pass a
    /// no-op closure if the caller is not surfacing intermediate tokens
    /// to the UI. Used by both `chat_with_cancel` (no-op) and
    /// `chat_stream` (live UI callback).
    async fn stream_once_assembled<F: FnMut(&str)>(
        &self,
        url: &str,
        body: &Value,
        connect_timeout: Duration,
        idle_timeout: Duration,
        cancelled: Option<&AtomicBool>,
        on_token: &mut F,
    ) -> Result<ChatResponse> {
        // The initial connect/HTTP-handshake gets the wall-clock timeout —
        // we don't want to dial forever if the server is unreachable.
        let connect_future = async {
            self.client
                .post(url)
                .json(body)
                .send()
                .await
                .with_context(|| format!("Failed to connect to LLM at {url}"))
        };

        let response = match cancelled {
            Some(flag) => {
                tokio::select! {
                    result = connect_future => result?,
                    _ = tokio::time::sleep(connect_timeout) => {
                        bail!("LLM request timed out after {}s", self.config.request_timeout_secs);
                    }
                    _ = wait_for_cancel(flag) => bail!("Interrupted by user"),
                }
            }
            None => tokio::time::timeout(connect_timeout, connect_future)
                .await
                .map_err(|_| {
                    anyhow::anyhow!(
                        "LLM request timed out after {}s",
                        self.config.request_timeout_secs
                    )
                })??,
        };

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            bail!("LLM API error ({status}): {text}");
        }

        // Servers that honor `stream: true` return text/event-stream;
        // some servers (and our test mocks) ignore the flag and return
        // a single application/json body. We dispatch on Content-Type
        // and handle both — the idle-timeout still applies to whichever
        // chunk reader we end up using.
        let is_sse = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|ct| ct.contains("event-stream"))
            .unwrap_or(false);

        if !is_sse {
            return self
                .read_non_streaming_body(response, idle_timeout, cancelled, on_token)
                .await;
        }

        let mut stream = response.bytes_stream();
        let mut full_content = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut current_tool_call_parts: std::collections::HashMap<
            usize,
            (String, String, String),
        > = std::collections::HashMap::new();
        let mut sse_buf = String::new();
        // Warn at most once per stream if the server omits tool_call index;
        // a broken server would otherwise spam a line per delta.
        let mut warned_missing_index = false;

        loop {
            // Wrap each chunk read in an idle-timeout. If the model has
            // not produced any tokens (or even keep-alives) within the
            // idle window, treat the connection as stuck and bail with
            // a retryable error.
            let next_chunk = async {
                if let Some(flag) = cancelled {
                    tokio::select! {
                        chunk = stream.next() => Ok(chunk),
                        _ = wait_for_cancel(flag) => Err(anyhow::anyhow!("Interrupted by user")),
                    }
                } else {
                    Ok(stream.next().await)
                }
            };

            let chunk_opt = match tokio::time::timeout(idle_timeout, next_chunk).await {
                Ok(Ok(chunk_opt)) => chunk_opt,
                Ok(Err(e)) => return Err(e),
                Err(_) => bail!(
                    "LLM stream idle: no tokens received for {}s",
                    self.config.stream_idle_timeout_secs
                ),
            };

            let Some(chunk) = chunk_opt else {
                break; // stream ended
            };
            let chunk = chunk.context("Stream read error")?;
            sse_buf.push_str(&String::from_utf8_lossy(&chunk));

            // Drain complete SSE events from the buffer (each event
            // ends with a `\n\n`). Hold any partial trailing event for
            // the next iteration so we don't truncate JSON mid-chunk.
            while let Some(idx) = sse_buf.find("\n\n") {
                let event = sse_buf[..idx].to_string();
                sse_buf.drain(..idx + 2);

                let mut done = false;
                for line in event.lines() {
                    let line = line.trim();
                    if line == "data: [DONE]" {
                        done = true;
                        break;
                    }
                    let Some(data) = line.strip_prefix("data: ") else {
                        continue;
                    };
                    let Ok(parsed) = serde_json::from_str::<StreamChunk>(data) else {
                        continue;
                    };
                    if let Some(choice) = parsed.choices.first() {
                        if let Some(content) = &choice.delta.content {
                            on_token(content);
                            full_content.push_str(content);
                        }
                        if let Some(tc_deltas) = &choice.delta.tool_calls {
                            for tc_delta in tc_deltas {
                                // Per OpenAI spec every tool-call delta carries `index`.
                                // Guessing 0 would silently corrupt parallel calls by
                                // merging stray deltas into call #0. Skip instead.
                                let Some(idx) = tc_delta.index else {
                                    if !warned_missing_index {
                                        tracing::warn!(
                                            "LLM stream: tool_call delta missing `index`; skipping. \
                                             The server emitted a non-spec-compliant SSE chunk — \
                                             if you see this often, the upstream tool call may be incomplete."
                                        );
                                        warned_missing_index = true;
                                    }
                                    continue;
                                };
                                let entry =
                                    current_tool_call_parts.entry(idx).or_insert_with(|| {
                                        (
                                            tc_delta.id.clone().unwrap_or_default(),
                                            String::new(),
                                            String::new(),
                                        )
                                    });
                                if let Some(id) = &tc_delta.id
                                    && !id.is_empty()
                                {
                                    entry.0 = id.clone();
                                }
                                if let Some(func) = &tc_delta.function {
                                    if let Some(name) = &func.name {
                                        entry.1.push_str(name);
                                    }
                                    if let Some(args) = &func.arguments {
                                        entry.2.push_str(args);
                                    }
                                }
                            }
                        }
                    }
                }
                if done {
                    break;
                }
            }
        }

        // Assemble tool calls from accumulated parts
        let mut indices: Vec<usize> = current_tool_call_parts.keys().copied().collect();
        indices.sort();
        for idx in indices {
            let Some((id, name, arguments)) = current_tool_call_parts.remove(&idx) else {
                continue;
            };
            tool_calls.push(ToolCall {
                id,
                r#type: "function".into(),
                function: FunctionCall { name, arguments },
            });
        }

        Ok(ChatResponse {
            choices: vec![Choice {
                message: Message {
                    role: "assistant".into(),
                    content: if full_content.is_empty() {
                        None
                    } else {
                        Some(full_content)
                    },
                    tool_calls: if tool_calls.is_empty() {
                        None
                    } else {
                        Some(tool_calls)
                    },
                    tool_call_id: None,
                    name: None,
                },
                finish_reason: Some("stop".into()),
            }],
            usage: None,
        })
    }

    /// Drain a non-streamed JSON response body chunk-by-chunk so the
    /// idle-timeout still applies (we don't want a hanging body to wedge
    /// the request indefinitely just because the server returned
    /// `application/json` instead of `text/event-stream`). After the
    /// body finishes we parse it as a single `ChatResponse` and forward
    /// any assistant text to `on_token` so streaming-style callers still
    /// get one final UI update.
    async fn read_non_streaming_body<F: FnMut(&str)>(
        &self,
        response: reqwest::Response,
        idle_timeout: Duration,
        cancelled: Option<&AtomicBool>,
        on_token: &mut F,
    ) -> Result<ChatResponse> {
        let mut stream = response.bytes_stream();
        let mut buf: Vec<u8> = Vec::new();

        loop {
            let next_chunk = async {
                if let Some(flag) = cancelled {
                    tokio::select! {
                        chunk = stream.next() => Ok(chunk),
                        _ = wait_for_cancel(flag) => Err(anyhow::anyhow!("Interrupted by user")),
                    }
                } else {
                    Ok(stream.next().await)
                }
            };

            let chunk_opt = match tokio::time::timeout(idle_timeout, next_chunk).await {
                Ok(Ok(chunk_opt)) => chunk_opt,
                Ok(Err(e)) => return Err(e),
                Err(_) => bail!(
                    "LLM stream idle: no tokens received for {}s",
                    self.config.stream_idle_timeout_secs
                ),
            };

            let Some(chunk) = chunk_opt else {
                break;
            };
            let chunk = chunk.context("Stream read error")?;
            buf.extend_from_slice(&chunk);
        }

        let resp: ChatResponse =
            serde_json::from_slice(&buf).context("Failed to parse LLM response")?;
        if let Some(content) = resp
            .choices
            .first()
            .and_then(|c| c.message.content.as_deref())
            .filter(|c| !c.is_empty())
        {
            on_token(content);
        }
        Ok(resp)
    }

    /// Send a streaming chat request. Calls `on_token` for each content
    /// delta and returns the final assembled response. The `cancelled`
    /// flag can be set from another task (e.g., Ctrl+C handler) to abort
    /// mid-stream.
    ///
    /// Internally shares the SSE / non-SSE / idle-timeout machinery with
    /// `chat_with_cancel` via [`Self::stream_once_assembled`]. Connect
    /// failures and idle-timeout errors are retried up to `max_retries`,
    /// but only if no tokens have been delivered yet on the current
    /// attempt — once the UI has seen partial content, retrying would
    /// duplicate it, so we surface the error instead.
    pub async fn chat_stream<F>(
        &self,
        request: &ChatRequest,
        mut on_token: F,
        cancelled: &Arc<AtomicBool>,
    ) -> Result<ChatResponse>
    where
        F: FnMut(&str),
    {
        let url = self.chat_url();
        let retry_delays = [1u64, 2, 4, 8, 16, 32];
        let max_retries = self.config.max_retries.min(retry_delays.len());

        let mut body = serde_json::to_value(request)?;
        body["model"] = Value::String(self.config.model.clone());
        body["temperature"] = Value::from(self.config.temperature);
        body["max_tokens"] = Value::from(self.config.max_output_tokens);
        body["stream"] = Value::Bool(true);
        let connect_timeout = Duration::from_secs(self.config.request_timeout_secs);
        let idle_timeout = Duration::from_secs(self.config.stream_idle_timeout_secs);

        let mut attempt = 0usize;
        loop {
            let mut had_progress = false;
            let result = {
                let mut wrapped = |token: &str| {
                    had_progress = true;
                    on_token(token);
                };
                self.stream_once_assembled(
                    &url,
                    &body,
                    connect_timeout,
                    idle_timeout,
                    Some(cancelled.as_ref()),
                    &mut wrapped,
                )
                .await
            };

            match result {
                Ok(resp) => return Ok(resp),
                Err(err)
                    if !had_progress && attempt < max_retries && is_retryable_llm_error(&err) =>
                {
                    let delay = retry_delays[attempt];
                    attempt += 1;
                    tokio::select! {
                        _ = tokio::time::sleep(Duration::from_secs(delay)) => {}
                        _ = wait_for_cancel(cancelled) => bail!("Interrupted by user"),
                    }
                }
                Err(err) => return Err(err),
            }
        }
    }
}

async fn wait_for_cancel(cancelled: &AtomicBool) {
    while !cancelled.load(Ordering::Relaxed) {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// Marker text llama.cpp includes in its 500 response when the model's
/// tool-call arguments couldn't be parsed as JSON — typically because the
/// model hit `max_tokens` mid-generation and the string was truncated.
///
/// Source: llama.cpp's server emits this in `common_chat_parse` when the
/// OAI tool-call path fails to parse arguments. If llama.cpp rewords this
/// in a future version, the detection here falls back to "retryable 500"
/// and the REPL will surface the raw error — adjust the marker if you
/// see the behavior regress.
pub const TRUNCATED_TOOL_CALL_MARKER: &str = "Failed to parse tool call arguments as JSON";

/// True if the LLM error came back as "Failed to parse tool call arguments
/// as JSON" (see [`TRUNCATED_TOOL_CALL_MARKER`]). Same prompt + same model
/// would re-emit the same truncated output, so this is *not* retryable —
/// the caller should surface a hint to the agent instead.
pub fn is_truncated_tool_call_error(err_msg: &str) -> bool {
    err_msg.contains(TRUNCATED_TOOL_CALL_MARKER)
}

fn is_retryable_llm_error(err: &anyhow::Error) -> bool {
    let msg = err.to_string();
    if is_truncated_tool_call_error(&msg) {
        // Retrying with the same prompt will just produce the same
        // truncated tool call. Bubble the error up so the caller can
        // synthesize a hint and let the agent try a different approach.
        return false;
    }
    msg.contains("Failed to connect to LLM")
        || msg.contains("LLM request timed out")
        || msg.contains("LLM stream idle")
        || msg.contains("Stream read error")
        || msg.contains("connection reset")
        || msg.contains("connection closed")
        || retryable_status_from_message(&msg).is_some()
}

fn retryable_status_from_message(msg: &str) -> Option<StatusCode> {
    for code in [
        StatusCode::INTERNAL_SERVER_ERROR,
        StatusCode::BAD_GATEWAY,
        StatusCode::SERVICE_UNAVAILABLE,
        StatusCode::GATEWAY_TIMEOUT,
    ] {
        if msg.contains(&format!("LLM API error ({code})")) {
            return Some(code);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncated_tool_call_error_detected() {
        let msg = r#"LLM API error (500 Internal Server Error): {"error":{"message":"Failed to parse tool call arguments as JSON: Unexpected EOF","type":"server_error"}}"#;
        assert!(is_truncated_tool_call_error(msg));
    }

    #[test]
    fn truncated_tool_call_error_not_retryable() {
        let err = anyhow::anyhow!(
            "LLM API error (500 Internal Server Error): Failed to parse tool call arguments as JSON"
        );
        assert!(!is_retryable_llm_error(&err));
    }

    #[test]
    fn plain_500_still_retryable() {
        let err =
            anyhow::anyhow!("LLM API error (500 Internal Server Error): upstream unavailable");
        assert!(is_retryable_llm_error(&err));
        assert!(!is_truncated_tool_call_error(&err.to_string()));
    }

    #[test]
    fn other_llm_errors_unaffected() {
        let err = anyhow::anyhow!("LLM request timed out after 60s");
        assert!(is_retryable_llm_error(&err));

        let err = anyhow::anyhow!("Failed to connect to LLM at http://localhost:8080");
        assert!(is_retryable_llm_error(&err));
    }
}
