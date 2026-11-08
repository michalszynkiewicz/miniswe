//! SSE parsing for the OpenAI chat-completions wire format (every provider
//! except Anthropic's native client).

use std::collections::HashMap;

use anyhow::{Context, Result, bail};

use crate::config::ToolCallFormat;

use super::normalize::normalize_xml_tool_calls;
use super::tool_call_repair::{TOOL_CALL_ARGS_CAP_MARKER, tool_call_args_cap};
use super::types::*;
use super::wire::StreamParser;

pub(super) struct OpenAiStream {
    tool_call_format: ToolCallFormat,
    full_content: String,
    /// index -> (id, name, arguments)
    current_tool_call_parts: HashMap<usize, (String, String, String)>,
    // Warn at most once per stream if the server omits tool_call index;
    // a broken server would otherwise spam a line per delta.
    warned_missing_index: bool,
    // Real finish_reason/usage from the stream (the final chunks carry
    // them). finish_reason matters downstream: "length" is the only
    // signal that a generation was cut off by the context ceiling
    // rather than finishing on its own — see
    // `is_context_truncated_response`.
    finish_reason: Option<String>,
    usage: Option<Usage>,
    /// Fallback when the server sends no `usage` (llama.cpp).
    timings_usage: Option<Usage>,
}

impl OpenAiStream {
    pub(super) fn new(tool_call_format: ToolCallFormat) -> Self {
        Self {
            tool_call_format,
            full_content: String::new(),
            current_tool_call_parts: HashMap::new(),
            warned_missing_index: false,
            finish_reason: None,
            usage: None,
            timings_usage: None,
        }
    }
}

impl StreamParser for OpenAiStream {
    fn feed_event(&mut self, event: &str, on_token: &mut dyn FnMut(&str)) -> Result<bool> {
        for line in event.lines() {
            let line = line.trim();
            if line == "data: [DONE]" {
                return Ok(true);
            }
            let Some(data) = line.strip_prefix("data: ") else {
                continue;
            };
            let Ok(parsed) = serde_json::from_str::<StreamChunk>(data) else {
                continue;
            };
            if let Some(u) = parsed.usage {
                self.usage = Some(u);
            }
            if let Some(t) = &parsed.timings {
                self.timings_usage = Some(t.to_usage());
            }
            if let Some(choice) = parsed.choices.first() {
                if let Some(fr) = &choice.finish_reason {
                    self.finish_reason = Some(fr.clone());
                }
                if let Some(content) = &choice.delta.content {
                    on_token(content);
                    self.full_content.push_str(content);
                }
                if let Some(tc_deltas) = &choice.delta.tool_calls {
                    for tc_delta in tc_deltas {
                        // Per OpenAI spec every tool-call delta carries `index`.
                        // Guessing 0 would silently corrupt parallel calls by
                        // merging stray deltas into call #0. Skip instead.
                        let Some(idx) = tc_delta.index else {
                            if !self.warned_missing_index {
                                tracing::warn!(
                                    "LLM stream: tool_call delta missing `index`; skipping. \
                                     The server emitted a non-spec-compliant SSE chunk — \
                                     if you see this often, the upstream tool call may be incomplete."
                                );
                                self.warned_missing_index = true;
                            }
                            continue;
                        };
                        let entry = self.current_tool_call_parts.entry(idx).or_insert_with(|| {
                            (
                                tc_delta.id.clone().unwrap_or_default(),
                                String::new(),
                                String::new(),
                            )
                        });
                        if let Some(id) = &tc_delta.id
                            && !id.is_empty()
                        {
                            entry.0 = id.clone();
                        }
                        if let Some(func) = &tc_delta.function {
                            if let Some(name) = &func.name {
                                entry.1.push_str(name);
                            }
                            if let Some(args) = &func.arguments {
                                entry.2.push_str(args);
                                // Anchor-only tools never need more
                                // than a few hundred chars; a call
                                // growing past the cap is the model
                                // pasting code into an anchor field.
                                // Abort now (the server cancels the
                                // slot on disconnect) instead of
                                // burning minutes until the context
                                // ceiling truncates it anyway.
                                if let Some(cap) = tool_call_args_cap(&entry.1)
                                    && entry.2.len() > cap
                                {
                                    bail!(
                                        "{TOOL_CALL_ARGS_CAP_MARKER}: `{}` arguments \
                                         reached {} chars (cap {cap}) — generation aborted",
                                        entry.1,
                                        entry.2.len()
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(false)
    }

    fn finish(mut self) -> Result<ChatResponse> {
        // Assemble tool calls from accumulated parts
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut indices: Vec<usize> = self.current_tool_call_parts.keys().copied().collect();
        indices.sort();
        for idx in indices {
            let Some((id, name, arguments)) = self.current_tool_call_parts.remove(&idx) else {
                continue;
            };
            tool_calls.push(ToolCall {
                id,
                r#type: "function".into(),
                function: FunctionCall { name, arguments },
            });
        }

        let mut resp = ChatResponse {
            choices: vec![Choice {
                message: Message {
                    role: "assistant".into(),
                    content: if self.full_content.is_empty() {
                        None
                    } else {
                        Some(self.full_content)
                    },
                    tool_calls: if tool_calls.is_empty() {
                        None
                    } else {
                        Some(tool_calls)
                    },
                    tool_call_id: None,
                    name: None,
                    provider_blocks: None,
                },
                // Real finish_reason from the stream when the server sent
                // one; "stop" preserves the old behavior for servers/mocks
                // that never emit it.
                finish_reason: Some(self.finish_reason.unwrap_or_else(|| "stop".into())),
            }],
            usage: self.usage.or(self.timings_usage),
        };
        normalize_xml_tool_calls(&mut resp, self.tool_call_format);
        Ok(resp)
    }

    fn finish_body(self, body: &[u8]) -> Result<ChatResponse> {
        let mut resp: ChatResponse =
            serde_json::from_slice(body).context("Failed to parse LLM response")?;
        if resp.usage.is_none()
            && let Ok(v) = serde_json::from_slice::<serde_json::Value>(body)
            && let Some(t) = v.get("timings")
            && let Ok(t) = serde_json::from_value::<LlamaTimings>(t.clone())
        {
            resp.usage = Some(t.to_usage());
        }
        normalize_xml_tool_calls(&mut resp, self.tool_call_format);
        Ok(resp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TIMINGS: &str = r#""timings":{"cache_n":3,"prompt_n":55,"prompt_ms":20.1,"predicted_n":20,"predicted_ms":200.0,"extra":1}"#;

    fn run(events: &[String]) -> ChatResponse {
        let mut s = OpenAiStream::new(ToolCallFormat::default());
        for e in events {
            s.feed_event(e, &mut |_| {}).unwrap();
        }
        s.finish().unwrap()
    }

    fn ev(extra: &str) -> String {
        format!(r#"data: {{"choices":[{{"finish_reason":"stop","index":0,"delta":{{}}}}]{extra}}}"#)
    }

    #[test]
    fn timings_only_synthesizes_usage() {
        let u = run(&[ev(&format!(",{TIMINGS}"))]).usage.unwrap();
        assert_eq!(u.prompt_tokens, 58);
        assert_eq!(u.completion_tokens, 20);
        assert_eq!(u.total_tokens, 78);
        assert_eq!(u.prompt_tokens_details.unwrap().cached_tokens, 3);
    }

    #[test]
    fn real_usage_wins_over_timings() {
        let usage = r#","usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":9}"#;
        let u = run(&[ev(&format!("{usage},{TIMINGS}"))]).usage.unwrap();
        assert_eq!(u.prompt_tokens, 7);
        assert_eq!(u.completion_tokens, 2);
        assert!(u.prompt_tokens_details.is_none());
    }

    #[test]
    fn neither_usage_nor_timings_is_none() {
        assert!(run(&[ev("")]).usage.is_none());
    }

    #[test]
    fn non_streaming_body_falls_back_to_timings() {
        let body = format!(
            r#"{{"choices":[{{"message":{{"role":"assistant","content":"x"}},"finish_reason":"stop"}}],{TIMINGS}}}"#
        );
        let s = OpenAiStream::new(ToolCallFormat::default());
        let u = s.finish_body(body.as_bytes()).unwrap().usage.unwrap();
        assert_eq!(u.prompt_tokens, 58);
    }
}
