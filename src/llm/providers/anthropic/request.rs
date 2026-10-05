//! `ChatRequest` (OpenAI-shaped history) -> Anthropic Messages request body.

use anyhow::Result;
use serde_json::{Map, Value, json};

use super::{AnthropicRequest, ModelCaps};
use crate::config::ModelConfig;
use crate::context::compressor::CURRENT_STATE_MARKER;
use crate::llm::providers::shape::{Thinking, thinking_requested};
use crate::llm::types::{ChatRequest, Message, ToolDefinition};

/// Thinking counts against `max_tokens`, so adaptive models never get less.
const ADAPTIVE_MAX_TOKENS_FLOOR: u64 = 32_000;
/// Room left above `budget_tokens` for the visible answer (the API requires
/// `max_tokens > budget_tokens`; a bare `+1` would leave no room for a reply).
const ANSWER_HEADROOM: u64 = 1024;

const BETA_THINKING_BINDING: &str = "thinking-binding-controls-2026-08-01";
const BETA_SERVER_FALLBACK: &str = "server-side-fallback-2026-07-01";

/// Models whose safety classifiers can decline a request; `fallbacks:
/// "default"` re-runs a declined request server-side on another model.
const FALLBACK_MODEL_PREFIXES: [&str; 4] = [
    "claude-fable-5-1",
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-sonnet-5-5",
];

/// One Anthropic message under construction.
struct Turn {
    role: &'static str,
    blocks: Vec<Value>,
}

fn ephemeral() -> Value {
    json!({"type": "ephemeral"})
}

fn text_block(text: &str) -> Value {
    json!({"type": "text", "text": text})
}

/// Build the Messages body for `request`. Never carries `temperature`,
/// `cache_prompt`, `chat_template_kwargs` or `stream_options`; a
/// `tool_choice` is passed through only if the request sets one.
pub fn build_request(
    config: &ModelConfig,
    request: &ChatRequest,
    caps: &ModelCaps,
) -> Result<AnthropicRequest> {
    let (system, mut turns) = convert_messages(&request.messages);
    mark_state_breakpoint(&mut turns);

    let mut body = Map::new();
    body.insert("model".into(), Value::String(config.model.clone()));
    body.insert("stream".into(), Value::Bool(true));

    if let Some(system) = system {
        body.insert(
            "system".into(),
            json!([{"type": "text", "text": system, "cache_control": ephemeral()}]),
        );
    }
    if let Some(tools) = request.tools.as_deref().filter(|t| !t.is_empty()) {
        body.insert("tools".into(), Value::Array(convert_tools(tools)));
    }
    body.insert(
        "messages".into(),
        Value::Array(
            turns
                .into_iter()
                .map(|t| json!({"role": t.role, "content": t.blocks}))
                .collect(),
        ),
    );
    if let Some(choice) = &request.tool_choice {
        body.insert("tool_choice".into(), choice.clone());
    }

    // --- thinking + output cap ---
    let thinking = thinking_requested(request);
    let requested = request
        .max_tokens_override
        .unwrap_or(config.max_output_tokens as u64);
    let binding = json!({"prefix_mismatch_behavior": "drop_block"});
    let mut betas: Vec<&str> = Vec::new();
    let max_tokens = if caps.adaptive {
        let effort = match &thinking {
            Thinking::On {
                effort: Some(effort),
            } => effort.clone(),
            Thinking::On { effort: None } => config.thinking_effort.clone(),
            Thinking::Off => "low".to_string(),
        };
        body.insert(
            "thinking".into(),
            json!({"type": "adaptive", "block_binding": binding}),
        );
        body.insert("output_config".into(), json!({"effort": effort}));
        betas.push(BETA_THINKING_BINDING);
        requested.max(ADAPTIVE_MAX_TOKENS_FLOOR)
    } else if matches!(thinking, Thinking::On { .. }) {
        let budget = config.thinking_budget_tokens as u64;
        body.insert(
            "thinking".into(),
            json!({"type": "enabled", "budget_tokens": budget, "block_binding": binding}),
        );
        betas.push(BETA_THINKING_BINDING);
        requested.max(budget + ANSWER_HEADROOM)
    } else {
        requested
    };
    body.insert("max_tokens".into(), Value::from(max_tokens));

    // --- server-side fallback on classifier declines ---
    if config.anthropic_fallbacks
        && FALLBACK_MODEL_PREFIXES
            .iter()
            .any(|p| config.model.starts_with(p))
    {
        body.insert("fallbacks".into(), Value::String("default".into()));
        betas.push(BETA_SERVER_FALLBACK);
    }

    Ok(AnthropicRequest {
        body: Value::Object(body),
        beta: (!betas.is_empty()).then(|| betas.join(",")),
    })
}

/// OpenAI tool definitions -> Anthropic tools; the breakpoint goes on the
/// last one, which caches the whole tool list.
fn convert_tools(tools: &[ToolDefinition]) -> Vec<Value> {
    let mut out: Vec<Value> = tools
        .iter()
        .map(|t| {
            json!({
                "name": t.function.name,
                "description": t.function.description,
                "input_schema": t.function.parameters,
                "eager_input_streaming": true,
            })
        })
        .collect();
    if let Some(last) = out.last_mut() {
        last["cache_control"] = ephemeral();
    }
    out
}

/// Split off the leading `system` messages (joined into one string) and
/// convert the rest into role-alternating Anthropic turns.
fn convert_messages(messages: &[Message]) -> (Option<String>, Vec<Turn>) {
    let lead = messages.iter().take_while(|m| m.role == "system").count();
    let system = messages[..lead]
        .iter()
        .filter_map(|m| m.content.as_deref())
        .filter(|c| !c.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");

    let mut turns: Vec<Turn> = Vec::new();
    for m in &messages[lead..] {
        let (role, mut blocks) = match m.role.as_str() {
            "assistant" => ("assistant", assistant_blocks(m)),
            "tool" => (
                "user",
                vec![json!({
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id.as_deref().unwrap_or_default(),
                    "content": m.content.as_deref().unwrap_or_default(),
                })],
            ),
            // `user`, and a `system` message that isn't leading (unsupported
            // mid-conversation): both become user text.
            _ => ("user", text_blocks(m.content.as_deref())),
        };
        // The API rejects empty (and whitespace-only) text blocks, including
        // ones inside replayed assistant turns.
        blocks.retain(|b| !is_blank_text(b));
        if blocks.is_empty() {
            continue;
        }
        match turns.last_mut() {
            Some(t) if t.role == role => t.blocks.extend(blocks),
            _ => turns.push(Turn { role, blocks }),
        }
    }
    ((!system.is_empty()).then_some(system), turns)
}

fn text_blocks(text: Option<&str>) -> Vec<Value> {
    text.filter(|t| !t.trim().is_empty())
        .map(|t| vec![text_block(t)])
        .unwrap_or_default()
}

fn is_blank_text(block: &Value) -> bool {
    block["type"] == "text" && block["text"].as_str().is_none_or(|t| t.trim().is_empty())
}

/// `arguments` as a JSON object; anything else is carried as `{"_raw": ..}`
/// so the call still round-trips instead of failing the request.
fn parse_arguments(arguments: &str) -> Value {
    if arguments.trim().is_empty() {
        return json!({});
    }
    match serde_json::from_str::<Value>(arguments) {
        Ok(v @ Value::Object(_)) => v,
        _ => json!({"_raw": arguments}),
    }
}

fn assistant_blocks(m: &Message) -> Vec<Value> {
    if let Some(blocks) = &m.provider_blocks
        && blocks_match_message(blocks, m)
    {
        return blocks.clone();
    }
    let mut out = text_blocks(m.content.as_deref());
    for tc in m.tool_calls.iter().flatten() {
        out.push(json!({
            "type": "tool_use",
            "id": tc.id,
            "name": tc.function.name,
            "input": parse_arguments(&tc.function.arguments),
        }));
    }
    out
}

/// Are the raw blocks still a faithful record of `m`? The harness edits
/// assistant messages after the fact (tool-call repair, pruning); replaying
/// stale blocks would contradict the visible message, so those turns are
/// rebuilt instead (losing their thinking, which the API then never sees).
fn blocks_match_message(blocks: &[Value], m: &Message) -> bool {
    let text: String = blocks
        .iter()
        .filter(|b| b["type"] == "text")
        .filter_map(|b| b["text"].as_str())
        .collect();
    if text != m.content.as_deref().unwrap_or_default() {
        return false;
    }
    let uses: Vec<&Value> = blocks.iter().filter(|b| b["type"] == "tool_use").collect();
    let calls = m.tool_calls.as_deref().unwrap_or_default();
    uses.len() == calls.len()
        && uses.iter().zip(calls).all(|(u, tc)| {
            u["id"].as_str() == Some(tc.id.as_str())
                && u["name"].as_str() == Some(tc.function.name.as_str())
                && u["input"] == parse_arguments(&tc.function.arguments)
        })
}

/// Put the third cache breakpoint just before the `[CURRENT STATE]` block,
/// which the harness appends to the last message every round. Everything
/// ahead of it is byte-stable between rounds; the state block itself is
/// split into its own trailing text block so it stays out of the cached
/// prefix. Without a marker the breakpoint goes on the last block.
fn mark_state_breakpoint(turns: &mut [Turn]) {
    let Some(ti) = turns.len().checked_sub(1) else {
        return;
    };
    let li = turns[ti].blocks.len() - 1;
    let mut target = Some((ti, li));

    if let Some(tail) = take_state_tail(&mut turns[ti].blocks[li]) {
        let blocks = &mut turns[ti].blocks;
        if payload_is_blank(&blocks[li]) {
            // Nothing before the marker to cache: step back one block. A
            // text block that is now empty is dropped; a tool_result stays
            // (the tool_use it answers needs it).
            if blocks[li]["type"] == "text" {
                blocks.pop();
            }
            target = match (li, ti) {
                (0, 0) => None,
                (0, _) => Some((ti - 1, turns[ti - 1].blocks.len() - 1)),
                _ => Some((ti, li - 1)),
            };
        }
        turns[ti].blocks.push(text_block(&tail));
    }

    if let Some((t, b)) = target
        && let Some(block) = turns[t].blocks.get_mut(b)
    {
        // A cache breakpoint is not accepted on thinking blocks.
        if !matches!(
            block["type"].as_str(),
            Some("thinking" | "redacted_thinking")
        ) {
            block["cache_control"] = ephemeral();
        }
    }
}

/// If this text / tool_result block contains the state marker, cut the
/// block there and return the marker-onwards text.
fn take_state_tail(block: &mut Value) -> Option<String> {
    let field = match block["type"].as_str()? {
        "text" => "text",
        "tool_result" => "content",
        _ => return None,
    };
    let full = block[field].as_str()?;
    let pos = full.find(CURRENT_STATE_MARKER)?;
    let (head, tail) = (full[..pos].to_string(), full[pos..].to_string());
    block[field] = Value::String(head);
    Some(tail)
}

fn payload_is_blank(block: &Value) -> bool {
    let field = if block["type"] == "text" {
        "text"
    } else {
        "content"
    };
    block[field].as_str().is_none_or(|s| s.trim().is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::{FunctionCall, FunctionDefinition, ToolCall};

    fn config(model: &str) -> ModelConfig {
        ModelConfig {
            provider: "anthropic".into(),
            model: model.into(),
            ..ModelConfig::default()
        }
    }

    fn adaptive() -> ModelCaps {
        ModelCaps { adaptive: true }
    }

    fn call(id: &str, name: &str, args: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            r#type: "function".into(),
            function: FunctionCall {
                name: name.into(),
                arguments: args.into(),
            },
        }
    }

    fn req(messages: Vec<Message>) -> ChatRequest {
        ChatRequest {
            messages,
            ..Default::default()
        }
    }

    fn thinking(effort: Option<&str>) -> Option<Value> {
        Some(match effort {
            Some(e) => json!({"reasoning_effort": e}),
            None => json!({"enable_thinking": true}),
        })
    }

    #[test]
    fn leading_system_is_hoisted_and_late_system_becomes_user_text() {
        let (system, turns) = convert_messages(&[
            Message::system("a"),
            Message::system("b"),
            Message::user("hi"),
            Message::assistant("ok"),
            Message::system("late"),
        ]);
        assert_eq!(system.as_deref(), Some("a\n\nb"));
        assert_eq!(turns.len(), 3);
        assert_eq!(turns[2].role, "user");
        assert_eq!(turns[2].blocks, vec![text_block("late")]);
    }

    #[test]
    fn tool_results_collapse_into_one_user_message_with_following_user_text() {
        let mut asst = Message::assistant_tool_calls(vec![
            call("t1", "read", r#"{"p":1}"#),
            call("t2", "read", r#"{"p":2}"#),
        ]);
        asst.content = Some("looking".into());
        let (_, turns) = convert_messages(&[
            Message::user("go"),
            asst,
            Message::tool_result("t1", "r1"),
            Message::tool_result("t2", "r2"),
            Message::user("next"),
        ]);
        assert_eq!(turns.len(), 3);
        assert_eq!(turns[1].role, "assistant");
        assert_eq!(turns[1].blocks[0], text_block("looking"));
        assert_eq!(turns[1].blocks[2]["input"], json!({"p": 2}));
        assert_eq!(turns[2].role, "user");
        assert_eq!(
            turns[2].blocks,
            vec![
                json!({"type": "tool_result", "tool_use_id": "t1", "content": "r1"}),
                json!({"type": "tool_result", "tool_use_id": "t2", "content": "r2"}),
                text_block("next"),
            ]
        );
    }

    #[test]
    fn empty_text_never_emitted_and_empty_messages_are_skipped() {
        let (_, turns) = convert_messages(&[
            Message::user("a"),
            Message::assistant(""),
            Message::user("b"),
        ]);
        // The empty assistant turn vanishes, so the two users merge.
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].blocks, vec![text_block("a"), text_block("b")]);
    }

    #[test]
    fn unparseable_arguments_are_sent_as_raw() {
        let mut m = Message::assistant_tool_calls(vec![call("t", "x", "{not json")]);
        m.content = None;
        let blocks = assistant_blocks(&m);
        assert_eq!(blocks[0]["input"], json!({"_raw": "{not json"}));
        let m = Message::assistant_tool_calls(vec![call("t", "x", "[1,2]")]);
        assert_eq!(assistant_blocks(&m)[0]["input"], json!({"_raw": "[1,2]"}));
    }

    #[test]
    fn consistent_provider_blocks_are_replayed_verbatim() {
        let blocks = vec![
            json!({"type": "thinking", "thinking": "hmm", "signature": "sig"}),
            json!({"type": "text", "text": "hello"}),
            json!({"type": "tool_use", "id": "t1", "name": "read", "input": {"a": 1}}),
        ];
        let mut m = Message::assistant_tool_calls(vec![call("t1", "read", r#"{"a": 1}"#)]);
        m.content = Some("hello".into());
        m.provider_blocks = Some(blocks.clone());
        assert_eq!(assistant_blocks(&m), blocks);
    }

    #[test]
    fn edited_tool_call_arguments_rebuild_the_turn_without_thinking() {
        let blocks = vec![
            json!({"type": "thinking", "thinking": "hmm", "signature": "sig"}),
            json!({"type": "tool_use", "id": "t1", "name": "read", "input": {"a": 1}}),
        ];
        let mut m = Message::assistant_tool_calls(vec![call("t1", "read", r#"{"a": 2}"#)]);
        m.provider_blocks = Some(blocks);
        let rebuilt = assistant_blocks(&m);
        assert_eq!(
            rebuilt,
            vec![json!({"type": "tool_use", "id": "t1", "name": "read", "input": {"a": 2}})]
        );
        // Edited text is just as inconsistent.
        let mut m = Message::assistant("changed");
        m.provider_blocks = Some(vec![
            json!({"type": "thinking", "thinking": "x", "signature": "s"}),
            json!({"type": "text", "text": "original"}),
        ]);
        assert_eq!(assistant_blocks(&m), vec![text_block("changed")]);
    }

    fn state_msg(prefix: &str) -> String {
        format!("{prefix}{CURRENT_STATE_MARKER}plan: x")
    }

    fn cc_positions(turns: &[Turn]) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for (t, turn) in turns.iter().enumerate() {
            for (b, block) in turn.blocks.iter().enumerate() {
                if block.get("cache_control").is_some() {
                    out.push((t, b));
                }
            }
        }
        out
    }

    #[test]
    fn state_block_is_split_off_the_last_user_text() {
        let (_, mut turns) = convert_messages(&[
            Message::user("task"),
            Message::assistant("a"),
            Message::user(&state_msg("continue")),
        ]);
        mark_state_breakpoint(&mut turns);
        let last = &turns[2].blocks;
        assert_eq!(last.len(), 2);
        assert_eq!(last[0]["text"], "continue");
        assert_eq!(last[0]["cache_control"], ephemeral());
        assert_eq!(last[1]["text"], state_msg("")); // marker onwards
        assert!(last[1].get("cache_control").is_none());
        assert_eq!(cc_positions(&turns), vec![(2, 0)]);
    }

    #[test]
    fn state_block_is_split_off_a_tool_result() {
        let (_, mut turns) = convert_messages(&[
            Message::user("task"),
            Message::assistant_tool_calls(vec![call("t1", "read", "{}")]),
            Message::tool_result("t1", &state_msg("output")),
        ]);
        mark_state_breakpoint(&mut turns);
        let last = &turns[2].blocks;
        assert_eq!(last[0]["type"], "tool_result");
        assert_eq!(last[0]["content"], "output");
        assert_eq!(last[0]["cache_control"], ephemeral());
        assert_eq!(last[1]["type"], "text");
        assert_eq!(last[1]["text"], state_msg(""));
    }

    #[test]
    fn empty_prefix_before_marker_moves_the_breakpoint_back() {
        let (_, mut turns) = convert_messages(&[
            Message::user("task"),
            Message::assistant("a"),
            Message::user(&state_msg("")),
        ]);
        mark_state_breakpoint(&mut turns);
        // Only the state block remains in the last message, uncached; the
        // breakpoint lands on the assistant turn before it.
        assert_eq!(turns[2].blocks.len(), 1);
        assert_eq!(turns[2].blocks[0]["text"], state_msg(""));
        assert_eq!(cc_positions(&turns), vec![(1, 0)]);
    }

    #[test]
    fn no_marker_puts_the_breakpoint_on_the_last_block_and_only_last_is_split() {
        let (_, mut turns) = convert_messages(&[
            Message::user(&state_msg("old")),
            Message::assistant("a"),
            Message::user("plain"),
        ]);
        mark_state_breakpoint(&mut turns);
        assert_eq!(cc_positions(&turns), vec![(2, 0)]);
        assert_eq!(turns[0].blocks.len(), 1, "older marker copy left alone");
    }

    #[test]
    fn body_has_three_breakpoints_and_native_tool_shape() {
        let mut request = req(vec![
            Message::system("sys"),
            Message::user("task"),
            Message::assistant("a"),
            Message::user(&state_msg("next")),
        ]);
        request.tools = Some(vec![
            ToolDefinition {
                r#type: "function".into(),
                function: FunctionDefinition {
                    name: "one".into(),
                    description: "d1".into(),
                    parameters: json!({"type": "object"}),
                },
            },
            ToolDefinition {
                r#type: "function".into(),
                function: FunctionDefinition {
                    name: "two".into(),
                    description: "d2".into(),
                    parameters: json!({"type": "object"}),
                },
            },
        ]);
        let b = build_request(&config("claude-haiku-4-5"), &request, &adaptive())
            .unwrap()
            .body;
        assert_eq!(b["system"][0]["cache_control"], ephemeral());
        assert_eq!(b["tools"][0]["input_schema"], json!({"type": "object"}));
        assert_eq!(b["tools"][0]["eager_input_streaming"], true);
        assert!(b["tools"][0].get("cache_control").is_none());
        assert_eq!(b["tools"][1]["cache_control"], ephemeral());
        let text = b.to_string();
        assert_eq!(text.matches("cache_control").count(), 3);
        assert_eq!(b["stream"], true);
        for banned in [
            "temperature",
            "stream_options",
            "cache_prompt",
            "chat_template_kwargs",
            "tool_choice",
        ] {
            assert!(b.get(banned).is_none(), "{banned}");
        }
    }

    #[test]
    fn no_tools_omits_the_tools_field() {
        let b = build_request(&config("m"), &req(vec![Message::user("x")]), &adaptive())
            .unwrap()
            .body;
        assert!(b.get("tools").is_none());
        assert!(b.get("system").is_none());
    }

    #[test]
    fn adaptive_thinking_on_off_and_per_request_effort() {
        let mut cfg = config("claude-sonnet-4-6");
        cfg.thinking_effort = "high".into();
        let mut r = req(vec![Message::user("x")]);

        let off = build_request(&cfg, &r, &adaptive()).unwrap();
        assert_eq!(off.body["output_config"], json!({"effort": "low"}));
        assert_eq!(
            off.body["thinking"],
            json!({"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "drop_block"}})
        );
        assert_eq!(
            off.beta.as_deref(),
            Some("thinking-binding-controls-2026-08-01")
        );

        r.chat_template_kwargs = thinking(None);
        let on = build_request(&cfg, &r, &adaptive()).unwrap();
        assert_eq!(on.body["output_config"], json!({"effort": "high"}));

        r.chat_template_kwargs = thinking(Some("medium"));
        let per = build_request(&cfg, &r, &adaptive()).unwrap();
        assert_eq!(per.body["output_config"], json!({"effort": "medium"}));
    }

    #[test]
    fn max_tokens_floor_for_adaptive_and_budget_headroom_for_enabled() {
        let mut cfg = config("m");
        cfg.max_output_tokens = 4096;
        let mut r = req(vec![Message::user("x")]);
        let b = build_request(&cfg, &r, &adaptive()).unwrap().body;
        assert_eq!(b["max_tokens"], 32_000);
        r.max_tokens_override = Some(64_000);
        let b = build_request(&cfg, &r, &adaptive()).unwrap().body;
        assert_eq!(b["max_tokens"], 64_000);

        let haiku = ModelCaps { adaptive: false };
        cfg.max_output_tokens = 1000;
        cfg.thinking_budget_tokens = 2048;
        r.max_tokens_override = None;
        r.chat_template_kwargs = thinking(None);
        let b = build_request(&cfg, &r, &haiku).unwrap();
        assert_eq!(b.body["max_tokens"], 3072);
        assert_eq!(
            b.body["thinking"],
            json!({"type": "enabled", "budget_tokens": 2048,
                   "block_binding": {"prefix_mismatch_behavior": "drop_block"}})
        );
        assert!(b.body.get("output_config").is_none());
        assert_eq!(
            b.beta.as_deref(),
            Some("thinking-binding-controls-2026-08-01")
        );
    }

    #[test]
    fn haiku_with_thinking_off_has_no_thinking_fields_and_no_beta() {
        let mut cfg = config("claude-haiku-4-5");
        cfg.max_output_tokens = 1000;
        let r = req(vec![Message::user("x")]);
        let b = build_request(&cfg, &r, &ModelCaps { adaptive: false }).unwrap();
        assert!(b.body.get("thinking").is_none());
        assert!(b.body.get("output_config").is_none());
        assert!(b.body.get("fallbacks").is_none());
        assert_eq!(b.body["max_tokens"], 1000);
        assert_eq!(b.beta, None);
    }

    #[test]
    fn fallbacks_follow_model_prefix_and_the_config_switch() {
        let r = req(vec![Message::user("x")]);
        for model in [
            "claude-fable-5-1",
            "claude-opus-5-5",
            "claude-opus-5",
            "claude-sonnet-5-5",
        ] {
            let b = build_request(&config(model), &r, &adaptive()).unwrap();
            assert_eq!(b.body["fallbacks"], "default", "{model}");
            assert_eq!(
                b.beta.as_deref(),
                Some("thinking-binding-controls-2026-08-01,server-side-fallback-2026-07-01")
            );
        }
        let b = build_request(&config("claude-sonnet-4-6"), &r, &adaptive()).unwrap();
        assert!(b.body.get("fallbacks").is_none());
        let mut off = config("claude-opus-5-5");
        off.anthropic_fallbacks = false;
        let b = build_request(&off, &r, &adaptive()).unwrap();
        assert!(b.body.get("fallbacks").is_none());
        assert_eq!(
            b.beta.as_deref(),
            Some("thinking-binding-controls-2026-08-01")
        );
        // Haiku (thinking off) + fallback model: only the fallback beta.
        let b =
            build_request(&config("claude-opus-5"), &r, &ModelCaps { adaptive: false }).unwrap();
        assert_eq!(b.beta.as_deref(), Some("server-side-fallback-2026-07-01"));
    }

    #[test]
    fn tool_choice_is_passed_through_only_when_set() {
        let mut r = req(vec![Message::user("x")]);
        r.tool_choice = Some(json!({"type": "any"}));
        let b = build_request(&config("m"), &r, &adaptive()).unwrap().body;
        assert_eq!(b["tool_choice"], json!({"type": "any"}));
    }
}
