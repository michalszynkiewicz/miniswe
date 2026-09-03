//! Repair of tool calls whose `arguments` never became valid JSON.
//!
//! A streamed tool call can be cut off by `max_tokens` or by the context
//! ceiling (llama.cpp stops at `n_ctx` with `finish_reason: "length"`).
//! The half-built `arguments` string then lands in the assistant message.
//! If that message is persisted verbatim, EVERY later request fails: the
//! server re-renders the history through its Jinja chat template, which
//! parses each assistant tool call's arguments and raises
//! "Failed to parse tool call arguments as JSON" (HTTP 500) — observed
//! live as a 436-round spin on Devstral Small 2 (2026-08-23 bench). The
//! invariant enforced here: an assistant message is never pushed to the
//! conversation with unparseable tool-call arguments.
//!
//! Two layers:
//! - [`sanitize_truncated_tool_calls`] rewrites the broken call in place to a
//!   small, valid stub tagged with [`TRUNCATED_ARGS_KEY`], so the tool loop
//!   can answer it with a proper `tool_result` ("not executed, re-issue
//!   smaller") and the template keeps seeing a well-formed call/result pair.
//! - [`tool_call_args_cap`] bounds generation for tools whose arguments are
//!   by construction short (anchors and identifiers): the stream assembler
//!   aborts the request once such a call's arguments blow past the cap,
//!   instead of letting the model paste a function body into `position` for
//!   nine minutes until the context ceiling ends it.

use serde_json::{Value, json};

use super::types::Message;

/// Key of the stub object that replaces truncated arguments.
pub const TRUNCATED_ARGS_KEY: &str = "__truncated_arguments__";

/// How many leading characters of the broken arguments the stub keeps, so
/// the model (and the log reader) can tell which call it was.
const HEAD_CHARS: usize = 120;

/// Marker the stream assembler puts in its error when a capped tool's
/// arguments outgrow [`tool_call_args_cap`]. Not retryable: the same prompt
/// regenerates the same flood.
/// Consecutive tool-call argument failures (server-side parse errors or our
/// own streaming size cap) after which the agent loop abandons the turn.
/// History is scrubbed of unparseable calls from the second failure on, so
/// reaching this means the model itself keeps producing them.
pub const TRUNCATED_CALL_ABORT_AFTER: usize = 4;

pub const TOOL_CALL_ARGS_CAP_MARKER: &str = "tool call arguments exceeded the size cap";

/// Per-tool hard cap on streamed `arguments` length, in chars. Only tools
/// whose every field is an anchor, identifier or short expression are
/// capped — a legitimate call never comes close, while a model pasting
/// code into one of them burns minutes of generation and then truncates.
/// Code-carrying tools (`replace_range`, `insert_at`, `write_file`, ...)
/// are deliberately uncapped; `max_tokens` bounds those.
pub fn tool_call_args_cap(tool_name: &str) -> Option<usize> {
    match tool_name {
        "refactor" | "file" | "code" => Some(4096),
        _ => None,
    }
}

/// True if the LLM error came from the stream assembler hitting a
/// [`tool_call_args_cap`].
pub fn is_tool_call_args_cap_error(err_msg: &str) -> bool {
    err_msg.contains(TOOL_CALL_ARGS_CAP_MARKER)
}

/// What the stub remembers about the original, broken arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TruncatedArgs {
    pub original_chars: usize,
    pub head: String,
}

fn truncated_stub(original: &str) -> String {
    let head: String = original.chars().take(HEAD_CHARS).collect();
    json!({
        TRUNCATED_ARGS_KEY: {
            "original_chars": original.chars().count(),
            "head": head,
        }
    })
    .to_string()
}

/// Rewrite every tool call in `msg` whose `arguments` do not parse as JSON
/// into the truncated stub. Empty arguments (some servers emit `""` for a
/// no-argument call) become `{}` rather than a stub — nothing was cut off.
/// Returns the number of calls rewritten as truncated.
pub fn sanitize_truncated_tool_calls(msg: &mut Message) -> usize {
    let Some(calls) = msg.tool_calls.as_mut() else {
        return 0;
    };
    let mut repaired = 0;
    for tc in calls.iter_mut() {
        let args = &tc.function.arguments;
        if args.trim().is_empty() {
            tc.function.arguments = "{}".into();
            continue;
        }
        if serde_json::from_str::<Value>(args).is_ok() {
            continue;
        }
        tc.function.arguments = truncated_stub(args);
        repaired += 1;
    }
    repaired
}

/// If `args` is a stub produced by [`sanitize_truncated_tool_calls`],
/// return what it remembers about the original call.
pub fn truncated_args_info(args: &Value) -> Option<TruncatedArgs> {
    let stub = args.get(TRUNCATED_ARGS_KEY)?;
    Some(TruncatedArgs {
        original_chars: stub
            .get("original_chars")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize,
        head: stub
            .get("head")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
    })
}

/// Defensive sweep over an existing message list: stub out any assistant
/// tool call that still carries unparseable arguments (history that was
/// built before this invariant existed, or any path that bypasses the
/// per-message sanitizer). Returns the number of calls rewritten.
pub fn scrub_unparseable_tool_calls(messages: &mut [Message]) -> usize {
    messages
        .iter_mut()
        .filter(|m| m.role == "assistant")
        .map(sanitize_truncated_tool_calls)
        .sum()
}

/// Tool-result text answering a stubbed call: tells the model the call was
/// NOT executed and why, without echoing the flood back into the context.
pub fn truncated_args_tool_result(tool_name: &str, info: &TruncatedArgs) -> String {
    format!(
        "ERROR: the arguments of this `{tool_name}` call were cut off by the output limit after \
         {} chars — the call was NOT executed and its arguments were discarded (they began with: \
         {:?}). Re-issue it with a much smaller payload: edit one function or one hunk per call, \
         and never paste code into anchor fields such as `position`, `name` or `path`.",
        info.original_chars, info.head
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::{FunctionCall, ToolCall};

    fn assistant_with(args: &[&str]) -> Message {
        Message {
            role: "assistant".into(),
            content: None,
            tool_calls: Some(
                args.iter()
                    .enumerate()
                    .map(|(i, a)| ToolCall {
                        id: format!("call_{i}"),
                        r#type: "function".into(),
                        function: FunctionCall {
                            name: "refactor".into(),
                            arguments: (*a).to_string(),
                        },
                    })
                    .collect(),
            ),
            tool_call_id: None,
            name: None,
        }
    }

    #[test]
    fn valid_arguments_are_left_alone() {
        let mut msg = assistant_with(&[r#"{"action":"help"}"#]);
        assert_eq!(sanitize_truncated_tool_calls(&mut msg), 0);
        assert_eq!(
            msg.tool_calls.unwrap()[0].function.arguments,
            r#"{"action":"help"}"#
        );
    }

    #[test]
    fn truncated_arguments_become_a_valid_stub_that_remembers_the_head() {
        let flood = format!(r#"{{"action":"add_param","position":"{}"#, "x".repeat(500));
        let mut msg = assistant_with(&[&flood]);
        assert_eq!(sanitize_truncated_tool_calls(&mut msg), 1);
        let repaired = &msg.tool_calls.as_ref().unwrap()[0].function.arguments;
        let parsed: Value = serde_json::from_str(repaired).expect("stub must be valid JSON");
        let info = truncated_args_info(&parsed).expect("stub must be recognised");
        assert_eq!(info.original_chars, flood.chars().count());
        assert!(info.head.starts_with(r#"{"action":"add_param""#));
        assert_eq!(info.head.chars().count(), HEAD_CHARS);
        assert!(
            repaired.len() < 400,
            "stub must be small: {}",
            repaired.len()
        );
    }

    #[test]
    fn empty_arguments_are_repaired_to_an_empty_object_not_a_stub() {
        let mut msg = assistant_with(&["", "   "]);
        assert_eq!(sanitize_truncated_tool_calls(&mut msg), 0);
        for tc in msg.tool_calls.unwrap() {
            assert_eq!(tc.function.arguments, "{}");
        }
    }

    #[test]
    fn head_is_char_safe_on_multibyte_input() {
        let flood = format!(r#"{{"name":"{}"#, "ł".repeat(300));
        let mut msg = assistant_with(&[&flood]);
        sanitize_truncated_tool_calls(&mut msg);
        let parsed: Value =
            serde_json::from_str(&msg.tool_calls.unwrap()[0].function.arguments).unwrap();
        let info = truncated_args_info(&parsed).unwrap();
        assert_eq!(info.head.chars().count(), HEAD_CHARS);
    }

    #[test]
    fn truncated_args_info_ignores_ordinary_arguments() {
        assert!(truncated_args_info(&json!({"action":"help"})).is_none());
        assert!(truncated_args_info(&json!("text")).is_none());
    }

    #[test]
    fn scrub_only_touches_assistant_messages_and_counts_repairs() {
        let mut messages = vec![
            Message::user("hi"),
            assistant_with(&[r#"{"ok":true}"#, r#"{"broken"#]),
            Message::tool_result("call_1", r#"{"broken"#),
            assistant_with(&[r#"{"also broken":"#]),
        ];
        assert_eq!(scrub_unparseable_tool_calls(&mut messages), 2);
        // Tool results are never rewritten, even when they echo bad JSON.
        assert_eq!(messages[2].content.as_deref(), Some(r#"{"broken"#));
        for m in messages.iter().filter(|m| m.role == "assistant") {
            for tc in m.tool_calls.iter().flatten() {
                assert!(serde_json::from_str::<Value>(&tc.function.arguments).is_ok());
            }
        }
        assert_eq!(scrub_unparseable_tool_calls(&mut messages), 0);
    }

    #[test]
    fn args_cap_covers_anchor_only_tools_and_leaves_code_tools_alone() {
        assert!(tool_call_args_cap("refactor").is_some());
        assert!(tool_call_args_cap("file").is_some());
        assert!(tool_call_args_cap("code").is_some());
        for code_tool in [
            "replace_range",
            "insert_at",
            "write_file",
            "edit_file",
            "shell",
        ] {
            assert!(tool_call_args_cap(code_tool).is_none(), "{code_tool}");
        }
        assert!(is_tool_call_args_cap_error(&format!(
            "{TOOL_CALL_ARGS_CAP_MARKER}: `refactor` arguments reached 4097 chars"
        )));
        assert!(!is_tool_call_args_cap_error("LLM API error (500): boom"));
    }

    #[test]
    fn tool_result_wording_names_the_tool_and_the_size_without_echoing_the_flood() {
        let info = TruncatedArgs {
            original_chars: 11234,
            head: "{\"action\":\"add_param\"".into(),
        };
        let text = truncated_args_tool_result("refactor", &info);
        assert!(text.contains("`refactor`"));
        assert!(text.contains("11234 chars"));
        assert!(text.contains("NOT executed"));
        assert!(text.len() < 600);
    }
}
