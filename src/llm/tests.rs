use std::time::Duration;

use serde_json::Value;

use crate::config::ToolCallFormat;

use super::client::{
    COLD_PREFILL_IDLE_SECS, MIN_PREFILL_TOKENS_PER_SEC, PROMPT_BYTES_PER_TOKEN,
    attempt_idle_timeout,
};
use super::errors::{has_tool_call_leak, is_retryable_llm_error};
use super::normalize::normalize_xml_tool_calls;
use super::*;

/// A body whose `messages` serialize to roughly `tokens` tokens.
fn body_of_size(tokens: u64, cache_prompt: Option<bool>) -> Value {
    let content = "x".repeat((tokens * PROMPT_BYTES_PER_TOKEN) as usize);
    let mut body = serde_json::json!({"messages": [{"role": "user", "content": content}]});
    if let Some(cp) = cache_prompt {
        body["cache_prompt"] = Value::Bool(cp);
    }
    body
}

#[test]
fn cold_prefill_widens_idle_window() {
    // cache_prompt=false → widened to the cold-prefill floor.
    let body = serde_json::json!({"cache_prompt": false});
    assert_eq!(
        attempt_idle_timeout(&body, 30, 600),
        Duration::from_secs(COLD_PREFILL_IDLE_SECS)
    );
}

#[test]
fn normal_request_keeps_configured_idle_window() {
    // cache_prompt=true or absent, no prompt → configured value, untouched.
    for body in [
        serde_json::json!({"cache_prompt": true}),
        serde_json::json!({}),
    ] {
        assert_eq!(
            attempt_idle_timeout(&body, 30, 600),
            Duration::from_secs(30)
        );
    }
}

#[test]
fn warm_request_widens_for_a_large_prompt() {
    // The regression this fix exists for: a *warm* request (cache_prompt
    // absent/true) carrying a big prompt used to get the flat configured
    // window and die mid-prefill. 12k tokens at MIN_PREFILL_TOKENS_PER_SEC
    // needs 300s, well past a 120s floor.
    let body = body_of_size(12_000, Some(true));
    assert_eq!(
        attempt_idle_timeout(&body, 120, 600),
        Duration::from_secs(12_000 / MIN_PREFILL_TOKENS_PER_SEC)
    );
}

#[test]
fn small_prompt_does_not_shrink_the_configured_window() {
    // A fast-role helper call must not get a *tighter* window than configured.
    let body = body_of_size(1_000, None);
    assert_eq!(
        attempt_idle_timeout(&body, 120, 600),
        Duration::from_secs(120)
    );
}

#[test]
fn idle_window_never_exceeds_the_absolute_deadline() {
    // Past the deadline the request is dead anyway — a larger idle window
    // would be a number that can never be reached.
    let body = body_of_size(100_000, Some(true));
    assert_eq!(
        attempt_idle_timeout(&body, 120, 600),
        Duration::from_secs(600)
    );
}

#[test]
fn cold_prefill_never_shrinks_a_larger_configured_window() {
    // A generous configured timeout wins over the floor (max, not set).
    let body = serde_json::json!({"cache_prompt": false});
    assert_eq!(
        attempt_idle_timeout(&body, 120, 600),
        Duration::from_secs(120)
    );
}

#[test]
fn truncated_tool_call_error_detected() {
    let msg = r#"LLM API error (500 Internal Server Error): {"error":{"message":"Failed to parse tool call arguments as JSON: Unexpected EOF","type":"server_error"}}"#;
    assert!(is_truncated_tool_call_error(msg));
}

#[test]
fn truncated_tool_call_error_not_retryable() {
    let err = anyhow::anyhow!(
        "LLM API error (500 Internal Server Error): Failed to parse tool call arguments as JSON"
    );
    assert!(!is_retryable_llm_error(&err));
}

#[test]
fn plain_500_still_retryable() {
    let err = anyhow::anyhow!("LLM API error (500 Internal Server Error): upstream unavailable");
    assert!(is_retryable_llm_error(&err));
    assert!(!is_truncated_tool_call_error(&err.to_string()));
}

#[test]
fn other_llm_errors_unaffected() {
    let err = anyhow::anyhow!("LLM request timed out after 60s");
    assert!(is_retryable_llm_error(&err));

    let err = anyhow::anyhow!("Failed to connect to LLM at http://localhost:8080");
    assert!(is_retryable_llm_error(&err));
}

#[test]
fn context_exceeded_error_detected() {
    // Verbatim llama.cpp 400 body, captured empirically against the
    // real server (a ~70K-token request into a 60K window).
    let msg = r#"LLM API error (400 Bad Request): {"error":{"code":400,"message":"request (70017 tokens) exceeds the available context size (60160 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":70017,"n_ctx":60160}}"#;
    assert!(is_context_exceeded_error(msg));
    // Not blindly retryable — same request fails identically; recovery
    // is compact-and-resend, driven by the round loop.
    assert!(!is_retryable_llm_error(&anyhow::anyhow!("{msg}")));
}

#[test]
fn unrelated_errors_are_not_context_exceeded() {
    assert!(!is_context_exceeded_error(
        "LLM API error (400 Bad Request): invalid model"
    ));
    assert!(!is_context_exceeded_error(
        "LLM request timed out after 30s"
    ));
    assert!(!is_context_exceeded_error(
        "LLM API error (500 Internal Server Error): Failed to parse tool call arguments as JSON"
    ));
}

fn resp_with_finish(
    finish_reason: &str,
    content: Option<&str>,
    usage: Option<Usage>,
) -> ChatResponse {
    ChatResponse {
        choices: vec![Choice {
            message: Message {
                role: "assistant".into(),
                content: content.map(str::to_string),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            finish_reason: Some(finish_reason.into()),
        }],
        usage,
    }
}

#[test]
fn context_truncated_response_detected_via_usage() {
    // The empirically observed shape: finish_reason="length" with the
    // completion stopping at n_ctx (210 tokens) despite max_tokens=2000.
    let resp = resp_with_finish(
        "length",
        Some("partial output"),
        Some(Usage {
            prompt_tokens: 59_950,
            completion_tokens: 210,
            total_tokens: 60_160,
        }),
    );
    assert!(is_context_truncated_response(&resp, 2000));
}

#[test]
fn legitimate_max_tokens_stop_is_not_context_truncation() {
    // finish_reason="length" with the completion AT the requested cap
    // is the model legitimately running to max_tokens.
    let resp = resp_with_finish(
        "length",
        Some("long output"),
        Some(Usage {
            prompt_tokens: 5_000,
            completion_tokens: 7_950,
            total_tokens: 12_950,
        }),
    );
    assert!(!is_context_truncated_response(&resp, 8000));
}

#[test]
fn normal_stop_is_not_context_truncation() {
    let resp = resp_with_finish("stop", Some("done"), None);
    assert!(!is_context_truncated_response(&resp, 8000));
}

#[test]
fn context_truncation_estimates_when_usage_absent() {
    // Streaming servers may omit usage — a ~1000-char (~250-token)
    // completion against an 8000-token cap still classifies via the
    // chars/4 estimate.
    let content = "x".repeat(1000);
    let resp = resp_with_finish("length", Some(&content), None);
    assert!(is_context_truncated_response(&resp, 8000));
}

#[test]
fn empty_choices_is_not_context_truncation() {
    let resp = ChatResponse {
        choices: vec![],
        usage: None,
    };
    assert!(!is_context_truncated_response(&resp, 8000));
}

fn resp_with_args(args: &str) -> ChatResponse {
    ChatResponse {
        choices: vec![Choice {
            message: Message {
                role: "assistant".into(),
                content: None,
                tool_calls: Some(vec![ToolCall {
                    id: "x".into(),
                    r#type: "function".into(),
                    function: FunctionCall {
                        name: "change_signature".into(),
                        arguments: args.into(),
                    },
                }]),
                tool_call_id: None,
                name: None,
            },
            finish_reason: Some("stop".into()),
        }],
        usage: None,
    }
}

#[test]
fn detects_tool_calls_token_leak() {
    let r = resp_with_args(r#"{"action":"add_param"}[TOOL_CALLS]"#);
    assert!(has_tool_call_leak(&r));
}

#[test]
fn detects_args_token_leak() {
    let r = resp_with_args(r#"[ARGS]{"action":"add_param"}"#);
    assert!(has_tool_call_leak(&r));
}

#[test]
fn clean_response_has_no_leak() {
    let r = resp_with_args(r#"{"action":"add_param","position":"after:foo"}"#);
    assert!(!has_tool_call_leak(&r));
}

fn resp_with_content(content: &str) -> ChatResponse {
    ChatResponse {
        choices: vec![Choice {
            message: Message {
                role: "assistant".into(),
                content: Some(content.into()),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            finish_reason: Some("stop".into()),
        }],
        usage: None,
    }
}

#[test]
fn normalize_lifts_xml_in_auto_mode_when_no_json_calls() {
    let mut r = resp_with_content(
        "Let me check.\n<file>\n<parameter=action>shell</parameter>\n<parameter=command>ls</parameter>\n</file>",
    );
    normalize_xml_tool_calls(&mut r, ToolCallFormat::Auto);
    let tcs = r.choices[0].message.tool_calls.as_ref().unwrap();
    assert_eq!(tcs.len(), 1);
    assert_eq!(tcs[0].function.name, "file");
    assert!(tcs[0].function.arguments.contains("\"action\":\"shell\""));
    assert!(tcs[0].function.arguments.contains("\"command\":\"ls\""));
    // Surrounding prose survives, XML block is gone.
    assert_eq!(
        r.choices[0].message.content.as_deref(),
        Some("Let me check.")
    );
}

#[test]
fn normalize_auto_keeps_existing_json_tool_calls() {
    // Auto mode must not overwrite real OpenAI tool_calls just because
    // the content happens to contain XML-looking text.
    let mut r = resp_with_args(r#"{"a":1}"#);
    r.choices[0].message.content =
        Some("<shell>\n<parameter=command>ls</parameter>\n</shell>".into());
    normalize_xml_tool_calls(&mut r, ToolCallFormat::Auto);
    let tcs = r.choices[0].message.tool_calls.as_ref().unwrap();
    assert_eq!(tcs.len(), 1);
    assert_eq!(tcs[0].function.name, "change_signature");
}

#[test]
fn normalize_xml_mode_always_replaces() {
    // In Xml mode we trust content even if tool_calls is populated.
    let mut r = resp_with_args(r#"{"a":1}"#);
    r.choices[0].message.content =
        Some("<shell>\n<parameter=command>ls</parameter>\n</shell>".into());
    normalize_xml_tool_calls(&mut r, ToolCallFormat::Xml);
    let tcs = r.choices[0].message.tool_calls.as_ref().unwrap();
    assert_eq!(tcs.len(), 1);
    assert_eq!(tcs[0].function.name, "shell");
}

#[test]
fn normalize_repairs_xml_leaked_into_args() {
    // Real shape captured from a Qwen3-Coder-Next bench dump.
    let mut r =
        resp_with_args(r#"{"action":"shell>\n<parameter=command>\ncd /work && grep -n foo"}"#);
    // Set tool name to "file" to mirror what llama-server emits.
    r.choices[0].message.tool_calls.as_mut().unwrap()[0]
        .function
        .name = "file".into();
    normalize_xml_tool_calls(&mut r, ToolCallFormat::Auto);
    let tc = &r.choices[0].message.tool_calls.as_ref().unwrap()[0];
    assert_eq!(tc.function.name, "file");
    let args: serde_json::Value = serde_json::from_str(&tc.function.arguments).unwrap();
    assert_eq!(args["action"], "shell");
    assert_eq!(args["command"], "cd /work && grep -n foo");
}

#[test]
fn normalize_json_mode_is_noop() {
    let mut r = resp_with_content("<shell>\n<parameter=command>ls</parameter>\n</shell>");
    normalize_xml_tool_calls(&mut r, ToolCallFormat::Json);
    assert!(r.choices[0].message.tool_calls.is_none());
    assert!(
        r.choices[0]
            .message
            .content
            .as_deref()
            .unwrap()
            .contains("<shell>")
    );
}
