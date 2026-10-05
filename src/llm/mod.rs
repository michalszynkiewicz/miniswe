//! LLM interface — OpenAI-compatible API client plus a native Anthropic
//! Messages client.
//!
//! Supports llama.cpp server, Ollama, vLLM, and any OpenAI-compatible
//! endpoint, plus the hosted providers OpenRouter, OpenAI, and Anthropic
//! (via its native Messages API) — see `providers` and
//! `docs/hosted-providers.md`. Handles streaming responses and tool call
//! parsing.

mod client;
mod errors;
mod normalize;
mod openai_stream;
pub mod providers;
pub mod router;
#[cfg(test)]
mod tests;
pub mod tool_call_repair;
mod types;
mod wire;
mod xml_tool_calls;

pub use client::{LlmClient, ProbeResult};
pub use errors::{
    TRUNCATED_TOOL_CALL_MARKER, is_context_exceeded_error, is_context_truncated_response,
    is_truncated_tool_call_error,
};
pub use providers::Provider;
pub use router::ModelRouter;
pub use tool_call_repair::{
    TOOL_CALL_ARGS_CAP_MARKER, TRUNCATED_CALL_ABORT_AFTER, is_tool_call_args_cap_error,
    sanitize_truncated_tool_calls, scrub_unparseable_tool_calls, tool_call_args_cap,
    truncated_args_info, truncated_args_tool_result,
};
pub use types::*;
