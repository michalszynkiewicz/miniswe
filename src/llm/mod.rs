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

    /// Send a non-streaming chat request with optional cancellation.
    pub async fn chat_with_cancel(
        &self,
        request: &ChatRequest,
        cancelled: Option<&AtomicBool>,
    ) -> Result<ChatResponse> {
        let url = self.chat_url();
        let retry_delays = [1u64, 2, 4, 8, 16, 32];
        let max_retries = self.config.max_retries.min(retry_delays.len());

        let mut body = serde_json::to_value(request)?;
        // Inject model config
        body["model"] = Value::String(self.config.model.clone());
        body["temperature"] = Value::from(self.config.temperature);
        body["max_tokens"] = Value::from(self.config.max_output_tokens);
        body["stream"] = Value::Bool(false);
        let timeout = Duration::from_secs(self.config.request_timeout_secs);

        let mut attempt = 0usize;
        loop {
            let request_future = async {
                let response = self
                    .client
                    .post(&url)
                    .json(&body)
                    .send()
                    .await
                    .with_context(|| format!("Failed to connect to LLM at {url}"))?;

                if !response.status().is_success() {
                    let status = response.status();
                    let text = response.text().await.unwrap_or_default();
                    bail!("LLM API error ({status}): {text}");
                }

                let resp: ChatResponse = response
                    .json()
                    .await
                    .context("Failed to parse LLM response")?;

                Ok(resp)
            };

            let result = match cancelled {
                Some(flag) => {
                    tokio::select! {
                        result = request_future => result,
                        _ = tokio::time::sleep(timeout) => {
                            bail!("LLM request timed out after {}s", self.config.request_timeout_secs)
                        }
                        _ = wait_for_cancel(flag) => {
                            bail!("Interrupted by user")
                        }
                    }
                }
                None => match tokio::time::timeout(timeout, request_future).await {
                    Ok(result) => result,
                    Err(_) => bail!("LLM request timed out after {}s", self.config.request_timeout_secs),
                },
            };

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
