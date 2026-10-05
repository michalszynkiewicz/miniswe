//! Native Anthropic Messages API client pieces: request conversion and
//! cache breakpoints (`request`), SSE events to `ChatResponse` (`stream`),
//! and the per-model capability lookup (`caps`). Type definitions and
//! re-exports only — see `docs/hosted-providers.md`.

mod caps;
mod request;
mod stream;

pub use caps::lookup_caps;
pub use request::build_request;
pub use stream::AnthropicStream;

use serde_json::Value;

/// What a model supports, read from `GET /v1/models/{id}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelCaps {
    /// `capabilities.thinking.types.adaptive.supported`. False on Haiku 4.5
    /// and older, which keep the `enabled` + `budget_tokens` thinking form.
    pub adaptive: bool,
}

/// A Messages request ready to send: the JSON body plus the
/// `anthropic-beta` header value it needs, if any.
#[derive(Debug, Clone)]
pub struct AnthropicRequest {
    pub body: Value,
    pub beta: Option<String>,
}
