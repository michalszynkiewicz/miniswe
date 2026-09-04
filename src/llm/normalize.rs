//! Promotion of XML-format tool calls (and XML leaked into JSON args)
//! into the OpenAI `tool_calls` shape.

use crate::config::ToolCallFormat;

use super::types::{ChatResponse, FunctionCall, ToolCall};
use super::xml_tool_calls;

/// Promote Anthropic-style XML tool calls embedded in `content` into the
/// JSON `tool_calls` array so the rest of the pipeline doesn't need to
/// know which wire format the model used.
///
/// Behaviour by [`ToolCallFormat`]:
///
/// * `Json`: no-op. We trust the OpenAI `tool_calls` field.
/// * `Xml`: always parse content as the source; replace `tool_calls`.
/// * `Auto`: keep existing `tool_calls` if non-empty; otherwise look for
///   XML tool-call blocks in content and lift them.
///
/// When XML calls are lifted, the matching XML blocks are *stripped* from
/// content so the surviving text is just the model's prose (thinking
/// commentary). If stripping leaves content empty, content is cleared so
/// downstream "empty assistant" checks still fire correctly.
pub(super) fn normalize_xml_tool_calls(resp: &mut ChatResponse, format: ToolCallFormat) {
    if matches!(format, ToolCallFormat::Json) {
        return;
    }
    // Pass 1: repair XML-leaked-into-args corruption (the dominant failure
    // for Qwen3-Coder-Next on llama-server, which doesn't yet have a
    // qwen3_coder tool-call parser — see llama.cpp issue #15012). Each tool
    // call's args string is re-checked; if a field contains `<parameter=`,
    // we lift the leaked XML into proper JSON fields in-place.
    for choice in &mut resp.choices {
        if let Some(tcs) = choice.message.tool_calls.as_mut() {
            for tc in tcs.iter_mut() {
                if let Some(repaired) = xml_tool_calls::repair_leaked_args(&tc.function.arguments) {
                    tc.function.arguments = repaired;
                }
            }
        }
    }

    // Pass 2: lift pure-content XML tool calls (the case where the model
    // emits the XML in `content` and tool_calls is empty).
    for choice in &mut resp.choices {
        let msg = &mut choice.message;
        let already_has_calls = msg.tool_calls.as_ref().is_some_and(|tcs| !tcs.is_empty());
        if matches!(format, ToolCallFormat::Auto) && already_has_calls {
            continue;
        }
        let Some(content_ref) = msg.content.as_deref() else {
            continue;
        };
        let parsed = xml_tool_calls::parse(content_ref);
        if parsed.is_empty() {
            continue;
        }

        let stripped = strip_xml_tool_blocks(content_ref);
        let synthesized: Vec<ToolCall> = parsed
            .into_iter()
            .enumerate()
            .map(|(i, p)| ToolCall {
                id: format!("xml_{i}"),
                r#type: "function".into(),
                function: FunctionCall {
                    name: p.name,
                    arguments: p.arguments.to_string(),
                },
            })
            .collect();

        msg.tool_calls = Some(synthesized);
        msg.content = if stripped.is_empty() {
            None
        } else {
            Some(stripped)
        };
    }
}

/// Remove `<NAME>...</NAME>` blocks that contain `<parameter=...>` from
/// the input, leaving surrounding thinking text untouched.
fn strip_xml_tool_blocks(content: &str) -> String {
    let bytes = content.as_bytes();
    let mut out = String::with_capacity(content.len());
    let mut cursor = 0;

    while cursor < bytes.len() {
        let rest = &content[cursor..];
        let Some(lt_off) = rest.find('<') else {
            out.push_str(rest);
            break;
        };
        out.push_str(&rest[..lt_off]);
        let lt_pos = cursor + lt_off;
        let after_lt = lt_pos + 1;

        // Skip closing tags and <parameter=...> (we only strip outer tool tags).
        if content[lt_pos..].starts_with("</") || content[lt_pos..].starts_with("<parameter=") {
            out.push('<');
            cursor = after_lt;
            continue;
        }

        // Read tag name.
        let mut name_end = after_lt;
        while name_end < bytes.len()
            && (bytes[name_end].is_ascii_alphanumeric() || bytes[name_end] == b'_')
        {
            name_end += 1;
        }
        if name_end == after_lt || bytes.get(name_end) != Some(&b'>') {
            out.push('<');
            cursor = after_lt;
            continue;
        }
        let name = &content[after_lt..name_end];
        let inner_start = name_end + 1;
        let close_pat = format!("</{name}>");
        let Some(close_off) = content[inner_start..].find(&close_pat) else {
            out.push('<');
            cursor = after_lt;
            continue;
        };
        let close_pos = inner_start + close_off;
        let inner = &content[inner_start..close_pos];

        if inner.contains("<parameter=") {
            // Tool-call block: drop it entirely.
            cursor = close_pos + close_pat.len();
            // Trim a trailing newline we likely inherited.
            if out.ends_with("\n\n") {
                out.pop();
            }
        } else {
            // Non-tool tag: keep verbatim.
            out.push_str(&content[lt_pos..close_pos + close_pat.len()]);
            cursor = close_pos + close_pat.len();
        }
    }

    out.trim().to_string()
}
