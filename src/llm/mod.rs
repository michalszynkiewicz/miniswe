//! LLM interface — OpenAI-compatible API client.
//!
//! Supports llama.cpp server, Ollama, vLLM, and any OpenAI-compatible endpoint.
//! Handles streaming responses and tool call parsing.

mod client;
mod errors;
mod normalize;
pub mod router;
#[cfg(test)]
mod tests;
pub mod tool_call_repair;
mod types;
mod xml_tool_calls;

pub use client::LlmClient;
pub use errors::{
    TRUNCATED_TOOL_CALL_MARKER, is_context_exceeded_error, is_context_truncated_response,
    is_truncated_tool_call_error,
};
pub use router::ModelRouter;
pub use tool_call_repair::{
    TOOL_CALL_ARGS_CAP_MARKER, TRUNCATED_CALL_ABORT_AFTER, is_tool_call_args_cap_error,
    sanitize_truncated_tool_calls, scrub_unparseable_tool_calls, tool_call_args_cap,
    truncated_args_info, truncated_args_tool_result,
};
pub use types::*;
