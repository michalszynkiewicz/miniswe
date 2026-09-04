//! Classification of LLM failures: which errors are retryable, which
//! signal context exhaustion, and which are truncated tool calls.

use reqwest::StatusCode;

use super::types::ChatResponse;

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

/// True if the server rejected the request outright because the PROMPT
/// alone exceeds the context window. llama.cpp returns a 400 whose body
/// carries `"type":"exceed_context_size_error"` (verified empirically:
/// `{"error":{"code":400,"message":"request (70017 tokens) exceeds the
/// available context size (60160 tokens)...","type":
/// "exceed_context_size_error",...}}`), which `stream_once_assembled`
/// folds into the error message verbatim. Not retryable as-is — the same
/// request fails identically — but recoverable by compacting the message
/// list and resending (see `compressor::force_compress`).
pub fn is_context_exceeded_error(err_msg: &str) -> bool {
    err_msg.contains("exceed_context_size_error")
        || err_msg.contains("exceeds the available context size")
}

/// True if a *successful* response was silently cut off by the context
/// ceiling rather than finishing on its own or legitimately hitting the
/// requested output cap. llama.cpp does NOT error in this case (verified
/// empirically): it returns 200 with `finish_reason: "length"` and stops
/// generation the instant `prompt_tokens + completion_tokens` reaches
/// `n_ctx`, regardless of the requested `max_tokens`. Since
/// `finish_reason: "length"` is the same value used for a legitimate
/// max-tokens stop, the tell is the completion landing well SHORT of the
/// requested cap: a legitimate cap-stop generates ~exactly `max_tokens`.
/// Uses real `usage` numbers when the server sent them, otherwise a
/// chars/4 estimate of the generated output; the 3/4 margin absorbs
/// estimation error. Callers should additionally gate on the prompt
/// actually being near the window (`compressor::estimated_context_tokens`)
/// before treating this as context exhaustion.
pub fn is_context_truncated_response(resp: &ChatResponse, requested_max_tokens: usize) -> bool {
    let Some(choice) = resp.choices.first() else {
        return false;
    };
    if choice.finish_reason.as_deref() != Some("length") {
        return false;
    }
    let completion_tokens = match &resp.usage {
        Some(u) => u.completion_tokens,
        None => {
            let msg = &choice.message;
            let content_chars = msg.content.as_deref().map_or(0, str::len);
            let args_chars: usize = msg
                .tool_calls
                .iter()
                .flatten()
                .map(|tc| tc.function.arguments.len())
                .sum();
            (content_chars + args_chars) / 4
        }
    };
    completion_tokens < requested_max_tokens * 3 / 4
}

/// True if any tool-call argument string contains a chat-template token
/// that should never appear in valid JSON arguments (`[TOOL_CALLS]`,
/// `[ARGS]`). When this happens the model has bled control tokens into
/// its own output — usually transient, KV-cache-state dependent.
pub(super) fn has_tool_call_leak(resp: &ChatResponse) -> bool {
    for choice in &resp.choices {
        let Some(tcs) = &choice.message.tool_calls else {
            continue;
        };
        for tc in tcs {
            let args = &tc.function.arguments;
            if args.contains("[TOOL_CALLS]") || args.contains("[ARGS]") {
                return true;
            }
        }
    }
    false
}

pub(super) fn is_retryable_llm_error(err: &anyhow::Error) -> bool {
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
