//! Provider dialect layer.
//!
//! Every provider miniswe speaks to except Anthropic shares the OpenAI
//! chat-completions wire protocol at the top level; this module is the single place that decides
//! the per-provider differences (default endpoint, auth headers, which
//! fields go in the request body, output-cap field name, thinking/reasoning
//! translation, usage reporting). See `docs/hosted-providers.md` for the
//! full decision table. Anthropic has its own wire format (`anthropic`).
//! Type definitions and re-exports only — logic lives
//! in the named submodules below.

pub mod anthropic;
mod auth;
mod endpoint;
mod kind;
mod probe;
mod shape;
pub mod usage;

#[cfg(test)]
mod tests;

pub use auth::{api_key_source, apply_auth, resolve_api_key};
pub use endpoint::{chat_url, effective_endpoint, models_url, props_url};
pub use probe::{context_window_from_models, context_window_from_props};
pub use shape::build_body;
pub use usage::{UsageSnapshot, UsageTotals};

/// Which wire dialect to speak for a `[model] provider` string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    /// Local llama.cpp server.
    LlamaCpp,
    /// Local/remote Ollama.
    Ollama,
    /// Local/remote vLLM.
    Vllm,
    /// Any other OpenAI-compatible endpoint (the catch-all, and the
    /// fallback for an unrecognized `provider` string).
    OpenAiCompatible,
    /// OpenRouter (`https://openrouter.ai/api/v1`).
    OpenRouter,
    /// OpenAI (`https://api.openai.com/v1`).
    OpenAi,
    /// Anthropic via its native Messages API (`https://api.anthropic.com/v1`).
    Anthropic,
}
