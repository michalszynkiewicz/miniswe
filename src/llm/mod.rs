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
        loop {
            let result = self
                .stream_once_assembled(&url, &body, connect_timeout, idle_timeout, cancelled)
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
    /// `ChatResponse`. Used by both `chat_with_cancel` and any future
    /// caller that wants a single-shot try without retries.
    async fn stream_once_assembled(
        &self,
        url: &str,
        body: &Value,
        connect_timeout: Duration,
        idle_timeout: Duration,
        cancelled: Option<&AtomicBool>,
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
                .map_err(|_| anyhow::anyhow!("LLM request timed out after {}s", self.config.request_timeout_secs))??,
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
                .read_non_streaming_body(response, idle_timeout, cancelled)
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
                            full_content.push_str(content);
                        }
                        if let Some(tc_deltas) = &choice.delta.tool_calls {
                            for tc_delta in tc_deltas {
                                let idx = tc_delta.index.unwrap_or(0);
                                let entry =
                                    current_tool_call_parts.entry(idx).or_insert_with(|| {
                                        (
                                            tc_delta.id.clone().unwrap_or_default(),
                                            String::new(),
                                            String::new(),
                                        )
                                    });
                                if let Some(id) = &tc_delta.id {
                                    if !id.is_empty() {
                                        entry.0 = id.clone();
                                    }
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
    /// body finishes we parse it as a single `ChatResponse`.
    async fn read_non_streaming_body(
        &self,
        response: reqwest::Response,
        idle_timeout: Duration,
        cancelled: Option<&AtomicBool>,
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

        let resp: ChatResponse = serde_json::from_slice(&buf)
            .context("Failed to parse LLM response")?;
        Ok(resp)
    }

    /// Send a chat completion request with streaming. Calls `on_token` for each
    /// content delta and returns the final assembled response.
    /// Send a streaming chat request. The `cancelled` flag can be set from
    /// another task (e.g., Ctrl+C handler) to abort mid-stream.
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

        let mut attempt = 0usize;
        let response = loop {
            let result = self
                .client
                .post(&url)
                .json(&body)
                .send()
                .await
                .with_context(|| format!("Failed to connect to LLM at {url}"));

            match result {
                Ok(response) if response.status().is_success() => break response,
                Ok(response) => {
                    let status = response.status();
                    let text = response.text().await.unwrap_or_default();
                    let err = anyhow::anyhow!("LLM API error ({status}): {text}");
                    if attempt < max_retries && is_retryable_llm_error(&err) {
                        let delay = retry_delays[attempt];
                        attempt += 1;
                        tokio::select! {
                            _ = tokio::time::sleep(Duration::from_secs(delay)) => {}
                            _ = wait_for_cancel(cancelled) => bail!("Interrupted by user"),
                        }
                        continue;
                    }
                    return Err(err);
                }
                Err(err) => {
                    if attempt < max_retries && is_retryable_llm_error(&err) {
                        let delay = retry_delays[attempt];
                        attempt += 1;
                        tokio::select! {
                            _ = tokio::time::sleep(Duration::from_secs(delay)) => {}
                            _ = wait_for_cancel(cancelled) => bail!("Interrupted by user"),
                        }
                        continue;
                    }
                    return Err(err);
                }
            }
        };

        let mut stream = response.bytes_stream();
        let mut full_content = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut current_tool_call_parts: std::collections::HashMap<
            usize,
            (String, String, String),
        > = std::collections::HashMap::new();

        while let Some(chunk) = stream.next().await {
            if cancelled.load(Ordering::Relaxed) {
                bail!("Interrupted by user");
            }
            let chunk = chunk.context("Stream read error")?;
            let text = String::from_utf8_lossy(&chunk);

            // SSE format: lines starting with "data: "
            for line in text.lines() {
                let line = line.trim();
                if line == "data: [DONE]" {
                    break;
                }
                if let Some(data) = line.strip_prefix("data: ") {
                    if let Ok(chunk) = serde_json::from_str::<StreamChunk>(data) {
                        if let Some(choice) = chunk.choices.first() {
                            // Handle content deltas
                            if let Some(content) = &choice.delta.content {
                                on_token(content);
                                full_content.push_str(content);
                            }
                            // Handle tool call deltas
                            if let Some(tc_deltas) = &choice.delta.tool_calls {
                                for tc_delta in tc_deltas {
                                    let idx = tc_delta.index.unwrap_or(0);
                                    let entry =
                                        current_tool_call_parts.entry(idx).or_insert_with(|| {
                                            (
                                                tc_delta.id.clone().unwrap_or_default(),
                                                String::new(),
                                                String::new(),
                                            )
                                        });
                                    if let Some(id) = &tc_delta.id {
                                        if !id.is_empty() {
                                            entry.0 = id.clone();
                                        }
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
}

async fn wait_for_cancel(cancelled: &AtomicBool) {
    while !cancelled.load(Ordering::Relaxed) {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

fn is_retryable_llm_error(err: &anyhow::Error) -> bool {
    let msg = err.to_string();
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
