//! The seam between the shared connect / status / idle-timeout / cancel /
//! SSE-splitting loop in `client.rs` and the per-wire-format parsers (the
//! OpenAI chat-completions one in `openai_stream.rs`, Anthropic's in
//! `providers::anthropic::stream`).

use anyhow::Result;

use super::types::ChatResponse;

/// Turns one response's events into a [`ChatResponse`].
pub(crate) trait StreamParser: Sized {
    /// Feed one SSE event (the raw lines between two blank lines). Calls
    /// `on_token` per visible text delta. Returns `true` once the stream
    /// signalled its end.
    fn feed_event(&mut self, event: &str, on_token: &mut dyn FnMut(&str)) -> Result<bool>;

    /// The stream ended: assemble the response from what was fed.
    fn finish(self) -> Result<ChatResponse>;

    /// The server ignored `stream: true` and returned one whole JSON body.
    fn finish_body(self, body: &[u8]) -> Result<ChatResponse>;
}
