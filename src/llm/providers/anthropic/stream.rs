//! Anthropic Messages SSE events -> `ChatResponse`.

use std::collections::BTreeMap;

use anyhow::{Result, anyhow, bail};
use serde_json::{Value, json};

use crate::llm::tool_call_repair::{TOOL_CALL_ARGS_CAP_MARKER, tool_call_args_cap};
use crate::llm::types::{
    ChatResponse, Choice, FunctionCall, Message, PromptTokensDetails, ToolCall, Usage,
};
use crate::llm::wire::StreamParser;

/// One content block being accumulated.
struct Block {
    /// The block as announced by `content_block_start` (type, id, name, ...).
    base: Value,
    text: String,
    signature: String,
    /// `input_json_delta` fragments (tool_use only).
    json: String,
}

#[derive(Default)]
pub struct AnthropicStream {
    blocks: BTreeMap<usize, Block>,
    input_tokens: usize,
    cache_creation: usize,
    cache_read: usize,
    output_tokens: usize,
    stop_reason: Option<String>,
    stop_category: Option<String>,
    transformations: Vec<Value>,
    saw_stop: bool,
}

impl AnthropicStream {
    pub fn new() -> Self {
        Self::default()
    }

    fn take_usage(&mut self, usage: &Value) {
        // `message_start` carries the input side; `message_delta` carries
        // cumulative output (and, on newer API versions, repeats the input
        // fields). Only overwrite with fields that are present.
        let field = |name: &str| usage.get(name).and_then(Value::as_u64).map(|n| n as usize);
        if let Some(n) = field("input_tokens") {
            self.input_tokens = n;
        }
        if let Some(n) = field("cache_creation_input_tokens") {
            self.cache_creation = n;
        }
        if let Some(n) = field("cache_read_input_tokens") {
            self.cache_read = n;
        }
        if let Some(n) = field("output_tokens") {
            self.output_tokens = n;
        }
    }

    fn take_transformations(&mut self, v: &Value) {
        if let Some(arr) = v.get("input_transformations").and_then(Value::as_array) {
            self.transformations = arr.clone();
        }
    }

    fn take_stop(&mut self, v: &Value) {
        if let Some(r) = v.get("stop_reason").and_then(Value::as_str) {
            self.stop_reason = Some(r.to_string());
        }
        if let Some(c) = v.pointer("/stop_details/category").and_then(Value::as_str) {
            self.stop_category = Some(c.to_string());
        }
    }

    fn handle(&mut self, data: &str, on_token: &mut dyn FnMut(&str)) -> Result<bool> {
        let Ok(ev) = serde_json::from_str::<Value>(data) else {
            return Ok(false);
        };
        match ev["type"].as_str().unwrap_or_default() {
            "message_start" => {
                self.take_usage(&ev["message"]["usage"]);
                self.take_transformations(&ev["message"]);
            }
            "content_block_start" => {
                let idx = ev["index"].as_u64().unwrap_or(0) as usize;
                self.blocks.insert(
                    idx,
                    Block {
                        base: ev["content_block"].clone(),
                        text: String::new(),
                        signature: String::new(),
                        json: String::new(),
                    },
                );
            }
            "content_block_delta" => {
                let idx = ev["index"].as_u64().unwrap_or(0) as usize;
                let delta = &ev["delta"];
                let Some(block) = self.blocks.get_mut(&idx) else {
                    return Ok(false);
                };
                match delta["type"].as_str().unwrap_or_default() {
                    "text_delta" => {
                        if let Some(t) = delta["text"].as_str() {
                            on_token(t);
                            block.text.push_str(t);
                        }
                    }
                    "thinking_delta" => {
                        if let Some(t) = delta["thinking"].as_str() {
                            block.text.push_str(t);
                        }
                    }
                    "signature_delta" => {
                        if let Some(s) = delta["signature"].as_str() {
                            block.signature.push_str(s);
                        }
                    }
                    "input_json_delta" => {
                        if let Some(p) = delta["partial_json"].as_str() {
                            block.json.push_str(p);
                            let name = block.base["name"].as_str().unwrap_or_default();
                            if let Some(cap) = tool_call_args_cap(name)
                                && block.json.len() > cap
                            {
                                bail!(
                                    "{TOOL_CALL_ARGS_CAP_MARKER}: `{name}` arguments \
                                     reached {} chars (cap {cap}) — generation aborted",
                                    block.json.len()
                                );
                            }
                        }
                    }
                    _ => {}
                }
            }
            "message_delta" => {
                self.take_stop(&ev["delta"]);
                self.take_usage(&ev["usage"]);
                self.take_transformations(&ev["delta"]);
                self.take_transformations(&ev);
            }
            "message_stop" => {
                self.saw_stop = true;
                return Ok(true);
            }
            "error" => {
                let kind = ev["error"]["type"].as_str().unwrap_or("error");
                let msg = ev["error"]["message"].as_str().unwrap_or_default();
                bail!("LLM API error (stream): {kind}: {msg}");
            }
            // ping and anything newer
            _ => {}
        }
        Ok(false)
    }

    fn assemble(self) -> Result<ChatResponse> {
        if self.stop_reason.as_deref() == Some("refusal") {
            bail!(
                "LLM refused the request (stop_reason=refusal, category={})",
                self.stop_category.as_deref().unwrap_or("unknown")
            );
        }

        self.log_transformations();

        let mut content = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut raw: Vec<Value> = Vec::new();
        let mut raw_ok = true;
        for block in self.blocks.values() {
            match block.base["type"].as_str().unwrap_or_default() {
                "text" => {
                    content.push_str(&block.text);
                    raw.push(json!({"type": "text", "text": block.text}));
                }
                "thinking" => raw.push(json!({
                    "type": "thinking",
                    "thinking": block.text,
                    "signature": block.signature,
                })),
                "tool_use" => {
                    let args = if block.json.trim().is_empty() {
                        "{}".to_string()
                    } else {
                        block.json.clone()
                    };
                    let id = block.base["id"].as_str().unwrap_or_default().to_string();
                    let name = block.base["name"].as_str().unwrap_or_default().to_string();
                    match serde_json::from_str::<Value>(&args) {
                        Ok(input @ Value::Object(_)) => raw.push(json!({
                            "type": "tool_use", "id": id, "name": name, "input": input,
                        })),
                        // Truncated / invalid arguments: they still reach the
                        // harness' repair path, but can't be replayed raw.
                        _ => raw_ok = false,
                    }
                    tool_calls.push(ToolCall {
                        id,
                        r#type: "function".into(),
                        function: FunctionCall {
                            name,
                            arguments: args,
                        },
                    });
                }
                // redacted_thinking and anything unknown: keep as announced.
                _ => raw.push(block.base.clone()),
            }
        }

        if self.stop_reason.is_none() && !self.saw_stop {
            bail!("Stream read error: stream ended before message_stop");
        }

        let finish_reason = match self.stop_reason.as_deref() {
            Some("tool_use") => "tool_calls",
            Some("max_tokens" | "model_context_window_exceeded") => "length",
            _ => "stop",
        };

        let prompt = self.input_tokens + self.cache_creation + self.cache_read;
        let usage = Usage {
            prompt_tokens: prompt,
            completion_tokens: self.output_tokens,
            total_tokens: prompt + self.output_tokens,
            prompt_tokens_details: Some(PromptTokensDetails {
                cached_tokens: self.cache_read,
                cache_write_tokens: self.cache_creation,
            }),
        };

        Ok(ChatResponse {
            choices: vec![Choice {
                message: Message {
                    role: "assistant".into(),
                    content: (!content.is_empty()).then_some(content),
                    tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
                    tool_call_id: None,
                    name: None,
                    provider_blocks: (raw_ok && !raw.is_empty()).then_some(raw),
                },
                finish_reason: Some(finish_reason.into()),
            }],
            usage: Some(usage),
        })
    }

    /// The API drops thinking blocks whose signature no longer matches the
    /// prefix (`drop_block`); surface that once per response.
    fn log_transformations(&self) {
        let dropped: Vec<&Value> = self
            .transformations
            .iter()
            .filter(|t| t["type"] == "thinking_dropped")
            .collect();
        if dropped.is_empty() {
            return;
        }
        let mut reasons: Vec<&str> = dropped
            .iter()
            .filter_map(|t| t["reason"].as_str())
            .collect();
        reasons.sort_unstable();
        reasons.dedup();
        tracing::info!(
            count = dropped.len(),
            reasons = %reasons.join(","),
            "anthropic dropped thinking blocks from the request"
        );
    }
}

impl StreamParser for AnthropicStream {
    fn feed_event(&mut self, event: &str, on_token: &mut dyn FnMut(&str)) -> Result<bool> {
        for line in event.lines() {
            let Some(data) = line.trim().strip_prefix("data:") else {
                continue;
            };
            if self.handle(data.trim(), on_token)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn finish(self) -> Result<ChatResponse> {
        self.assemble()
    }

    fn finish_body(mut self, body: &[u8]) -> Result<ChatResponse> {
        let v: Value = serde_json::from_slice(body)
            .map_err(|e| anyhow!("Stream read error: unparseable Messages body: {e}"))?;
        if v["type"] == "error" {
            let kind = v["error"]["type"].as_str().unwrap_or("error");
            let msg = v["error"]["message"].as_str().unwrap_or_default();
            bail!("LLM API error (stream): {kind}: {msg}");
        }
        self.take_usage(&v["usage"]);
        self.take_stop(&v);
        self.take_transformations(&v);
        for (i, b) in v["content"].as_array().into_iter().flatten().enumerate() {
            let mut block = Block {
                base: b.clone(),
                text: String::new(),
                signature: String::new(),
                json: String::new(),
            };
            match b["type"].as_str().unwrap_or_default() {
                "text" => block.text = b["text"].as_str().unwrap_or_default().into(),
                "thinking" => {
                    block.text = b["thinking"].as_str().unwrap_or_default().into();
                    block.signature = b["signature"].as_str().unwrap_or_default().into();
                }
                "tool_use" => block.json = b["input"].to_string(),
                _ => {}
            }
            self.blocks.insert(i, block);
        }
        // A whole body is complete by construction.
        self.saw_stop = true;
        self.assemble()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(events: &[Value]) -> Result<ChatResponse> {
        let mut s = AnthropicStream::new();
        for e in events {
            if s.feed_event(&format!("event: x\ndata: {e}"), &mut |_| {})? {
                break;
            }
        }
        s.finish()
    }

    fn start(usage: Value) -> Value {
        json!({"type": "message_start", "message": {"usage": usage}})
    }

    fn stop(reason: &str, out: u64) -> Vec<Value> {
        vec![
            json!({"type": "message_delta", "delta": {"stop_reason": reason},
                   "usage": {"output_tokens": out}}),
            json!({"type": "message_stop"}),
        ]
    }

    #[test]
    fn text_tool_use_and_thinking_assemble_with_raw_blocks() {
        let mut ev = vec![
            start(
                json!({"input_tokens": 10, "cache_creation_input_tokens": 100,
                          "cache_read_input_tokens": 1000, "output_tokens": 1}),
            ),
            json!({"type": "content_block_start", "index": 0,
                   "content_block": {"type": "thinking", "thinking": ""}}),
            json!({"type": "content_block_delta", "index": 0,
                   "delta": {"type": "thinking_delta", "thinking": "hm"}}),
            json!({"type": "content_block_delta", "index": 0,
                   "delta": {"type": "signature_delta", "signature": "SIG"}}),
            json!({"type": "content_block_start", "index": 1,
                   "content_block": {"type": "text", "text": ""}}),
            json!({"type": "content_block_delta", "index": 1,
                   "delta": {"type": "text_delta", "text": "hel"}}),
            json!({"type": "content_block_delta", "index": 1,
                   "delta": {"type": "text_delta", "text": "lo"}}),
            json!({"type": "content_block_start", "index": 2,
                   "content_block": {"type": "tool_use", "id": "t1", "name": "read", "input": {}}}),
            json!({"type": "content_block_delta", "index": 2,
                   "delta": {"type": "input_json_delta", "partial_json": "{\"a\":"}}),
            json!({"type": "content_block_delta", "index": 2,
                   "delta": {"type": "input_json_delta", "partial_json": " 1}"}}),
            json!({"type": "ping"}),
        ];
        ev.extend(stop("tool_use", 42));
        let resp = run(&ev).unwrap();
        let c = &resp.choices[0];
        assert_eq!(c.finish_reason.as_deref(), Some("tool_calls"));
        assert_eq!(c.message.content.as_deref(), Some("hello"));
        let tc = &c.message.tool_calls.as_ref().unwrap()[0];
        assert_eq!((tc.id.as_str(), tc.function.name.as_str()), ("t1", "read"));
        assert_eq!(tc.function.arguments, "{\"a\": 1}");
        assert_eq!(
            c.message.provider_blocks.as_ref().unwrap(),
            &vec![
                json!({"type": "thinking", "thinking": "hm", "signature": "SIG"}),
                json!({"type": "text", "text": "hello"}),
                json!({"type": "tool_use", "id": "t1", "name": "read", "input": {"a": 1}}),
            ]
        );
        let u = resp.usage.unwrap();
        assert_eq!(u.prompt_tokens, 1110);
        assert_eq!(u.completion_tokens, 42);
        assert_eq!(u.total_tokens, 1152);
        assert_eq!(u.cached_tokens(), 1000);
        assert_eq!(u.cache_write_tokens(), 100);
    }

    #[test]
    fn stop_reasons_map_to_finish_reasons() {
        for (reason, want) in [
            ("end_turn", "stop"),
            ("stop_sequence", "stop"),
            ("max_tokens", "length"),
            ("model_context_window_exceeded", "length"),
            ("pause_turn", "stop"),
        ] {
            let resp = run(&stop(reason, 1)).unwrap();
            assert_eq!(resp.choices[0].finish_reason.as_deref(), Some(want));
        }
    }

    #[test]
    fn refusal_is_an_error_naming_the_category() {
        let ev = vec![
            json!({"type": "message_delta",
                   "delta": {"stop_reason": "refusal", "stop_details": {"category": "cyber"}},
                   "usage": {"output_tokens": 0}}),
            json!({"type": "message_stop"}),
        ];
        let err = run(&ev).unwrap_err().to_string();
        assert!(err.contains("stop_reason=refusal"));
        assert!(err.contains("category=cyber"));
    }

    #[test]
    fn error_event_bails_and_is_retryable_when_overloaded() {
        let ev = vec![json!({"type": "error",
            "error": {"type": "overloaded_error", "message": "Overloaded"}})];
        let err = run(&ev).unwrap_err();
        assert!(err.to_string().contains("overloaded_error: Overloaded"));
        assert!(crate::llm::errors::is_retryable_llm_error(&err));
    }

    #[test]
    fn stream_ending_without_stop_is_a_retryable_read_error() {
        let ev = vec![start(json!({"input_tokens": 1}))];
        let err = run(&ev).unwrap_err();
        assert!(crate::llm::errors::is_retryable_llm_error(&err));
    }

    #[test]
    fn truncated_tool_json_still_yields_a_call_but_no_raw_blocks() {
        let mut ev = vec![
            json!({"type": "content_block_start", "index": 0,
                   "content_block": {"type": "tool_use", "id": "t", "name": "x", "input": {}}}),
            json!({"type": "content_block_delta", "index": 0,
                   "delta": {"type": "input_json_delta", "partial_json": "{\"a\": \"unfin"}}),
        ];
        ev.extend(stop("max_tokens", 5));
        let resp = run(&ev).unwrap();
        let m = &resp.choices[0].message;
        assert_eq!(
            m.tool_calls.as_ref().unwrap()[0].function.arguments,
            "{\"a\": \"unfin"
        );
        assert!(m.provider_blocks.is_none());
    }

    #[test]
    fn empty_tool_input_is_an_empty_object() {
        let mut ev = vec![json!({"type": "content_block_start", "index": 0,
            "content_block": {"type": "tool_use", "id": "t", "name": "x", "input": {}}})];
        ev.extend(stop("tool_use", 1));
        let resp = run(&ev).unwrap();
        assert_eq!(
            resp.choices[0].message.tool_calls.as_ref().unwrap()[0]
                .function
                .arguments,
            "{}"
        );
    }

    #[test]
    fn tokens_stream_through_on_token() {
        let mut s = AnthropicStream::new();
        let mut seen = String::new();
        for e in [
            json!({"type": "content_block_start", "index": 0,
                   "content_block": {"type": "text", "text": ""}}),
            json!({"type": "content_block_delta", "index": 0,
                   "delta": {"type": "text_delta", "text": "ab"}}),
        ] {
            s.feed_event(&format!("data: {e}"), &mut |t| seen.push_str(t))
                .unwrap();
        }
        assert_eq!(seen, "ab");
    }

    #[test]
    fn whole_body_fallback_parses_a_messages_object() {
        let body = json!({
            "type": "message", "stop_reason": "end_turn",
            "content": [
                {"type": "thinking", "thinking": "t", "signature": "S"},
                {"type": "text", "text": "done"},
                {"type": "tool_use", "id": "u", "name": "n", "input": {"k": "v"}},
            ],
            "usage": {"input_tokens": 3, "output_tokens": 4,
                      "cache_read_input_tokens": 2},
        });
        let resp = AnthropicStream::new()
            .finish_body(body.to_string().as_bytes())
            .unwrap();
        let m = &resp.choices[0].message;
        assert_eq!(m.content.as_deref(), Some("done"));
        assert_eq!(
            m.tool_calls.as_ref().unwrap()[0].function.arguments,
            "{\"k\":\"v\"}"
        );
        assert_eq!(m.provider_blocks.as_ref().unwrap().len(), 3);
        assert_eq!(resp.usage.unwrap().prompt_tokens, 5);
    }

    #[test]
    fn whole_body_error_object_bails() {
        let body = json!({"type": "error", "error": {"type": "api_error", "message": "x"}});
        assert!(
            AnthropicStream::new()
                .finish_body(body.to_string().as_bytes())
                .is_err()
        );
    }
}
