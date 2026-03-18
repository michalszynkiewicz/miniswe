//! LLM interface — OpenAI-compatible API client.
//!
//! Supports llama.cpp server, Ollama, vLLM, and any OpenAI-compatible endpoint.
//! Handles streaming responses and tool call parsing.

mod types;

pub use types::*;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, bail};
use futures::StreamExt;
use reqwest::Client;
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
        let url = self.chat_url();

        let mut body = serde_json::to_value(request)?;
        // Inject model config
        body["model"] = Value::String(self.config.model.clone());
        body["temperature"] = Value::from(self.config.temperature);
        body["max_tokens"] = Value::from(self.config.max_output_tokens);
        body["stream"] = Value::Bool(false);

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

        let mut body = serde_json::to_value(request)?;
        body["model"] = Value::String(self.config.model.clone());
        body["temperature"] = Value::from(self.config.temperature);
        body["max_tokens"] = Value::from(self.config.max_output_tokens);
        body["stream"] = Value::Bool(true);

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

        let mut stream = response.bytes_stream();
        let mut full_content = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut current_tool_call_parts: std::collections::HashMap<usize, (String, String, String)> =
            std::collections::HashMap::new();

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
                                    let entry = current_tool_call_parts
                                        .entry(idx)
                                        .or_insert_with(|| {
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
