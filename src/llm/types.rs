//! Types for OpenAI-compatible chat API.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A chat completion request.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChatRequest {
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    /// Per-request override for `max_tokens`, used by callers that need
    /// more output budget than the model config's default. Skipped from
    /// serialization — the LLM client merges it into the request body
    /// directly so we don't need a custom JSON shape.
    #[serde(skip)]
    pub max_tokens_override: Option<u64>,
    /// Server-specific arguments passed to the chat template. Used to
    /// disable thinking-mode reasoning on Gemma 4 (and similar models)
    /// where reasoning eats `max_tokens` and never reaches the answer.
    /// Skipped from default serialization; the LLM client merges this
    /// into the body under the well-known `chat_template_kwargs` key.
    #[serde(skip)]
    pub chat_template_kwargs: Option<Value>,
    /// Per-request override for `temperature`, used by thinking-mode
    /// requests: reasoning traces need a higher temperature than code-task
    /// sampling (see `ModelConfig::thinking_temperature`). `None` = the
    /// model config's `temperature`. Merged into the body by the client,
    /// like `max_tokens_override`.
    #[serde(skip)]
    pub temperature_override: Option<f64>,
    /// Per-request `cache_prompt` override for llama.cpp. `Some(false)` forces
    /// a fresh (cold) prompt eval instead of reusing the slot's KV cache.
    /// `None` = server default (reuse). Merged into the body directly
    /// (skipped from serde).
    ///
    /// Set by the tool-call-leak retry in `llm::mod` — a leaked call is
    /// evidence the cached prefix itself is bad, so the resend must not reuse
    /// it. The agent loop no longer forces cold prefills on loop detection: a
    /// corpus audit of 680 forced prefills found no break-rate benefit and a
    /// large cost (a ~40k-token re-prefill is ~58s, up to ~250s on
    /// Mistral-Small-4). See the `window_edit_fires` field doc on
    /// `cli::commands::agent::loop_detector::LoopTracker`.
    #[serde(skip)]
    pub cache_prompt: Option<bool>,
}

/// A chat message (system, user, assistant, or tool).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The assistant turn's raw Anthropic content blocks (thinking, text,
    /// tool_use, anything unknown), verbatim, so the next request can replay
    /// them with their signatures. `serde(skip)` keeps it out of every
    /// OpenAI-shaped body; the Anthropic request builder only trusts it
    /// while it still matches `content` / `tool_calls`.
    #[serde(skip)]
    pub provider_blocks: Option<Vec<Value>>,
}

impl Message {
    pub fn system(content: &str) -> Self {
        Self {
            role: "system".into(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            provider_blocks: None,
        }
    }

    pub fn user(content: &str) -> Self {
        Self {
            role: "user".into(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            provider_blocks: None,
        }
    }

    pub fn assistant(content: &str) -> Self {
        Self {
            role: "assistant".into(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            provider_blocks: None,
        }
    }

    pub fn assistant_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: "assistant".into(),
            content: None,
            tool_calls: Some(tool_calls),
            tool_call_id: None,
            name: None,
            provider_blocks: None,
        }
    }

    pub fn tool_result(tool_call_id: &str, content: &str) -> Self {
        Self {
            role: "tool".into(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
            name: None,
            provider_blocks: None,
        }
    }

    /// True if this message has any content or any tool calls.
    /// Empty assistant messages (no content, no tool_calls) trigger 400s
    /// from most chat APIs, so callers should drop them before pushing
    /// to history.
    pub fn is_meaningful(&self) -> bool {
        self.content.as_deref().is_some_and(|s| !s.is_empty())
            || self.tool_calls.as_deref().is_some_and(|tc| !tc.is_empty())
    }
}

/// A tool call from the assistant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub function: FunctionCall,
}

/// Function call details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

/// Tool definition in OpenAI format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub r#type: String,
    pub function: FunctionDefinition,
}

/// Function definition for a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

/// A chat completion response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub choices: Vec<Choice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// A response choice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub message: Message,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// Token usage information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    /// Breakdown of `prompt_tokens`, e.g. how many were served from a
    /// provider-side cache. OpenAI and OpenRouter send this; absent on
    /// llama.cpp and Ollama.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
}

/// Breakdown of [`Usage::prompt_tokens`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: usize,
    /// Tokens written to a provider-side cache on this call (Anthropic
    /// `cache_creation_input_tokens`; OpenRouter spells it the same).
    #[serde(default)]
    pub cache_write_tokens: usize,
}

impl Usage {
    /// How many of `prompt_tokens` were served from a provider-side cache,
    /// or `0` when the server didn't report a breakdown.
    pub fn cached_tokens(&self) -> usize {
        self.prompt_tokens_details
            .as_ref()
            .map(|d| d.cached_tokens)
            .unwrap_or(0)
    }

    /// How many of `prompt_tokens` were written to a provider-side cache.
    pub fn cache_write_tokens(&self) -> usize {
        self.prompt_tokens_details
            .as_ref()
            .map(|d| d.cache_write_tokens)
            .unwrap_or(0)
    }
}

/// A streaming chunk from SSE.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    /// Defaulted: the final usage-bearing chunk from some servers carries
    /// no `choices` field at all, and must still parse.
    #[serde(default)]
    pub choices: Vec<StreamChoice>,
    /// llama.cpp (and OpenAI with `include_usage`) attach token usage to
    /// the final chunk. Captured when present so the assembled response
    /// carries real numbers; absent chunks leave it `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// A streaming choice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChoice {
    pub delta: StreamDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// A streaming delta (partial message).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

/// A partial tool call in streaming mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<FunctionCallDelta>,
}

/// A partial function call delta.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}
