//! Parser for Anthropic-style XML tool calls embedded in assistant content.
//!
//! Some models (Qwen3-Coder-Next, Hermes fine-tunes, anything trained on
//! Claude transcripts) emit tool invocations as XML blocks inside the
//! `content` field instead of populating the OpenAI `tool_calls` array.
//! This module converts that XML back into the same internal `ToolCall`
//! shape so the dispatcher doesn't need to care which wire format the
//! model used.
//!
//! Two dialects we've observed in the wild and both handle here:
//!
//!   Flat (Anthropic / Hermes):
//!     <shell>
//!       <parameter=command>cargo check</parameter>
//!     </shell>
//!
//!   Grouped (a Qwen fallback):
//!     <file>
//!       <parameter=action>shell</parameter>
//!       <parameter=command>cargo check</parameter>
//!     </file>
//!
//! Output is normalized to a single representation: a tool name plus a JSON
//! object of arguments. The caller maps that into whatever ToolCall struct
//! the rest of the pipeline expects.
//!
//! Parameter values that parse as valid JSON (`[...]`, `{...}`, numbers,
//! booleans) are stored as JSON; everything else stays as a string. This
//! matters for tools like `spawn_agents` whose `agents` parameter is a
//! JSON array even when delivered as XML.

use serde_json::{Map, Value};

/// A tool call extracted from XML content.
#[derive(Debug, Clone, PartialEq)]
pub struct XmlToolCall {
    pub name: String,
    pub arguments: Value,
}

/// Walk `content` and pull out every well-formed XML tool-call block.
///
/// A block qualifies if it has the shape `<NAME>...<parameter=...>...</parameter>...</NAME>`
/// — i.e. an outer tag that contains at least one `<parameter=...>` child.
/// Tags without parameters are ignored (likely thinking-text XML, e.g.
/// `<thinking>...</thinking>`).
///
/// Returns blocks in the order they appear. Malformed regions are skipped.
pub fn parse(content: &str) -> Vec<XmlToolCall> {
    let mut out = Vec::new();
    let bytes = content.as_bytes();
    let mut cursor = 0;

    while cursor < bytes.len() {
        let Some(open_lt) = find_tool_tag_open(content, cursor) else {
            break;
        };
        let after_lt = open_lt + 1;

        // Read the tag name: ascii letters/digits/underscore.
        let name_end = scan_tag_name(bytes, after_lt);
        if name_end == after_lt {
            cursor = after_lt;
            continue;
        }
        // Tag must close with '>' immediately after the name (we don't
        // support attributes on the outer tag).
        if bytes.get(name_end) != Some(&b'>') {
            cursor = name_end;
            continue;
        }
        let name = &content[after_lt..name_end];
        let inner_start = name_end + 1;

        // Find the matching closing tag.
        let close_pat = format!("</{name}>");
        let Some(close_off) = content[inner_start..].find(&close_pat) else {
            cursor = inner_start;
            continue;
        };
        let close_pos = inner_start + close_off;
        let inner = &content[inner_start..close_pos];

        if let Some(args) = extract_parameters(inner) {
            out.push(XmlToolCall {
                name: name.to_string(),
                arguments: Value::Object(args),
            });
        }

        cursor = close_pos + close_pat.len();
    }

    out
}

/// Find the next `<` that introduces a potential tool tag — skipping over
/// `</...` closing tags and `<parameter=...` (those are children, not
/// outer tool tags).
fn find_tool_tag_open(content: &str, from: usize) -> Option<usize> {
    let bytes = content.as_bytes();
    let mut i = from;
    while i < bytes.len() {
        if bytes[i] != b'<' {
            i += 1;
            continue;
        }
        let rest = &content[i..];
        if rest.starts_with("</") || rest.starts_with("<parameter=") {
            i += 1;
            continue;
        }
        // Next char must look like a tag name start.
        match bytes.get(i + 1) {
            Some(&c) if is_name_start(c) => return Some(i),
            _ => {
                i += 1;
            }
        }
    }
    None
}

fn scan_tag_name(bytes: &[u8], from: usize) -> usize {
    let mut i = from;
    while i < bytes.len() && is_name_char(bytes[i]) {
        i += 1;
    }
    i
}

fn is_name_start(c: u8) -> bool {
    c.is_ascii_alphabetic() || c == b'_'
}

fn is_name_char(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

/// Repair tool-call arguments that have an XML body leaked into one of
/// their string fields. This is the corruption pattern that happens when
/// llama-server's chat template tries to fit Qwen3-Coder's XML tool-call
/// format into an OpenAI JSON tool_call: the leading `<` of the outer tag
/// is dropped, the closing tags are often truncated, and the XML body
/// ends up sitting in (typically) the `action` field as a literal string.
///
/// Example input we've seen in the wild:
///
/// ```text
/// {"action":"shell>\n<parameter=command>\ncd /work && grep ..."}
/// ```
///
/// The model meant `<shell><parameter=command>cd /work && grep ...</parameter></shell>`.
/// We strip the leak: replace `action` with the clean tool name (`shell`) and
/// merge the extracted parameter(s) into the JSON arg map. Existing
/// well-formed JSON keys win — we only fill in fields that aren't there.
///
/// Returns `Some(repaired_json)` if a leak was detected and repaired,
/// `None` if the args looked fine.
pub(crate) fn repair_leaked_args(args_str: &str) -> Option<String> {
    let mut args: Value = serde_json::from_str(args_str).ok()?;
    let obj = args.as_object_mut()?;

    // Locate a string field whose value contains `<parameter=` — that's the
    // signature of a leak.
    let leaked_key = obj.iter().find_map(|(k, v)| {
        v.as_str()
            .filter(|s| s.contains("<parameter="))
            .map(|_| k.clone())
    })?;
    let leaked_value = obj.get(&leaked_key)?.as_str()?.to_string();

    // The dropped leading `<` means the value begins with the tool/action
    // name followed by `>`. Pull that off.
    let gt_pos = leaked_value.find('>')?;
    let prefix = leaked_value[..gt_pos].trim();
    if prefix.is_empty() {
        return None;
    }
    let body = &leaked_value[gt_pos + 1..];

    obj.insert(leaked_key.clone(), Value::String(prefix.to_string()));

    // Extract `<parameter=NAME>VALUE</parameter>` pairs (or a final
    // unclosed `<parameter=NAME>VALUE...` chunk — the body is often
    // truncated) and merge them in without clobbering pre-existing keys.
    let params = extract_parameters_tolerant(body);
    for (pk, pv) in params {
        obj.entry(&pk).or_insert(pv);
    }

    serde_json::to_string(obj).ok()
}

/// Walk `inner` (the text between `<tool>` and `</tool>`) and collect every
/// `<parameter=NAME>VALUE</parameter>` into a JSON map. Returns `None` if
/// no parameters are found — the caller treats that as "not a tool call".
fn extract_parameters(inner: &str) -> Option<Map<String, Value>> {
    let mut params = Map::new();
    let mut cursor = 0;

    while let Some(p_off) = inner[cursor..].find("<parameter=") {
        let abs = cursor + p_off;
        let name_start = abs + "<parameter=".len();
        let Some(gt_off) = inner[name_start..].find('>') else {
            break;
        };
        let name_end = name_start + gt_off;
        let name = inner[name_start..name_end].trim();
        if name.is_empty() {
            cursor = name_end + 1;
            continue;
        }

        let value_start = name_end + 1;
        let Some(close_off) = inner[value_start..].find("</parameter>") else {
            break;
        };
        let value_end = value_start + close_off;
        let raw = inner[value_start..value_end].trim();

        // If the value parses as JSON, store it as JSON. This handles array
        // and object parameters (e.g. spawn_agents.agents). Everything else
        // is a plain string.
        let value = serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()));
        params.insert(name.to_string(), value);

        cursor = value_end + "</parameter>".len();
    }

    if params.is_empty() {
        None
    } else {
        Some(params)
    }
}

/// Like `extract_parameters` but tolerant of truncated bodies: if the last
/// `<parameter=NAME>` has no matching `</parameter>`, take the value as
/// "everything from the open marker to the next `<parameter=` or
/// end-of-string". This is the shape we see in leaked-into-args corruption
/// where the trailing close tags are dropped.
fn extract_parameters_tolerant(body: &str) -> Map<String, Value> {
    let mut params = Map::new();
    let mut cursor = 0;

    while let Some(p_off) = body[cursor..].find("<parameter=") {
        let abs = cursor + p_off;
        let name_start = abs + "<parameter=".len();
        let Some(gt_off) = body[name_start..].find('>') else {
            break;
        };
        let name_end = name_start + gt_off;
        let name = body[name_start..name_end].trim();
        if name.is_empty() {
            cursor = name_end + 1;
            continue;
        }
        let value_start = name_end + 1;

        // Value ends at the first of: </parameter>, next <parameter=, or EOS.
        let close_off = body[value_start..].find("</parameter>");
        let next_open_off = body[value_start..].find("<parameter=");
        let value_end = match (close_off, next_open_off) {
            (Some(c), Some(n)) => value_start + c.min(n),
            (Some(c), None) => value_start + c,
            (None, Some(n)) => value_start + n,
            (None, None) => body.len(),
        };
        let raw = body[value_start..value_end].trim();
        let value = serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()));
        params.insert(name.to_string(), value);

        cursor = if body[value_end..].starts_with("</parameter>") {
            value_end + "</parameter>".len()
        } else {
            value_end
        };
    }

    params
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn flat_dialect_single_param() {
        let calls = parse("<shell>\n<parameter=command>cargo check</parameter>\n</shell>");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(calls[0].arguments, json!({"command": "cargo check"}));
    }

    #[test]
    fn grouped_dialect_with_action() {
        let calls = parse(
            "<file>\n<parameter=action>shell</parameter>\n<parameter=command>ls tests/</parameter>\n</file>",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "file");
        assert_eq!(
            calls[0].arguments,
            json!({"action": "shell", "command": "ls tests/"})
        );
    }

    #[test]
    fn thinking_text_around_tool_call() {
        let content = "Let me check the tests.\n\n<shell>\n<parameter=command>cargo test</parameter>\n</shell>\n\nThen I'll know.";
        let calls = parse(content);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
    }

    #[test]
    fn ignores_tags_without_parameters() {
        // `<thinking>` is a common Claude-style narration tag — no parameters,
        // so the parser should skip it.
        let calls = parse("<thinking>I should run cargo check first.</thinking>");
        assert!(calls.is_empty());
    }

    #[test]
    fn multiple_tool_calls() {
        let content = "<shell>\n<parameter=command>ls</parameter>\n</shell>\n<shell>\n<parameter=command>pwd</parameter>\n</shell>";
        let calls = parse(content);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].arguments, json!({"command": "ls"}));
        assert_eq!(calls[1].arguments, json!({"command": "pwd"}));
    }

    #[test]
    fn json_array_parameter_preserved() {
        // spawn_agents.agents is a JSON array; the parser should preserve
        // that structure rather than stringify it.
        let content = "<spawn_agents>\n<parameter=agents>[{\"label\":\"a\",\"prompt\":\"do x\"}]</parameter>\n</spawn_agents>";
        let calls = parse(content);
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0].arguments,
            json!({"agents": [{"label": "a", "prompt": "do x"}]})
        );
    }

    #[test]
    fn multiline_value() {
        let content =
            "<shell>\n<parameter=command>cat <<EOF\nhello\nworld\nEOF</parameter>\n</shell>";
        let calls = parse(content);
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0].arguments,
            json!({"command": "cat <<EOF\nhello\nworld\nEOF"})
        );
    }

    #[test]
    fn unmatched_closing_tag_skipped() {
        // Malformed: no </shell>. Parser shouldn't loop forever.
        let calls = parse("<shell>\n<parameter=command>ls</parameter>\n");
        assert!(calls.is_empty());
    }

    #[test]
    fn empty_input() {
        assert!(parse("").is_empty());
    }

    #[test]
    fn no_xml_returns_empty() {
        let calls = parse("This is just regular assistant chatter, no tool calls here.");
        assert!(calls.is_empty());
    }

    #[test]
    fn repair_leaked_args_basic() {
        // The exact corruption we see in the bench dumps for Qwen3-Coder-Next.
        let leaked = r#"{"action":"shell>\n<parameter=command>\ncd /work && grep -n foo"}"#;
        let repaired = repair_leaked_args(leaked).expect("should repair");
        let parsed: serde_json::Value = serde_json::from_str(&repaired).unwrap();
        assert_eq!(parsed["action"], "shell");
        assert_eq!(parsed["command"], "cd /work && grep -n foo");
    }

    #[test]
    fn repair_leaked_args_with_extra_json_key() {
        // Sometimes a real JSON key survives alongside the leaked XML in another
        // field. The repair should not clobber it.
        let leaked =
            r#"{"action":"search>\n<parameter=path>\nsrc/cli/run.rs","pattern":"SessionLog"}"#;
        let repaired = repair_leaked_args(leaked).expect("should repair");
        let parsed: serde_json::Value = serde_json::from_str(&repaired).unwrap();
        assert_eq!(parsed["action"], "search");
        assert_eq!(parsed["path"], "src/cli/run.rs");
        assert_eq!(parsed["pattern"], "SessionLog");
    }

    #[test]
    fn repair_leaked_args_multiple_parameters() {
        let leaked = r#"{"action":"file_edit>\n<parameter=path>\nfoo.rs</parameter>\n<parameter=line>\n42"}"#;
        let repaired = repair_leaked_args(leaked).expect("should repair");
        let parsed: serde_json::Value = serde_json::from_str(&repaired).unwrap();
        assert_eq!(parsed["action"], "file_edit");
        assert_eq!(parsed["path"], "foo.rs");
        assert_eq!(parsed["line"], 42);
    }

    #[test]
    fn repair_leaked_args_clean_input_returns_none() {
        // No leak — repair should return None so the caller can leave the
        // args untouched.
        let clean = r#"{"action":"shell","command":"ls"}"#;
        assert!(repair_leaked_args(clean).is_none());
    }

    #[test]
    fn repair_leaked_args_non_object_returns_none() {
        assert!(repair_leaked_args("\"just a string\"").is_none());
        assert!(repair_leaked_args("[1, 2, 3]").is_none());
    }

    #[test]
    fn repair_leaked_args_existing_key_wins() {
        // If a real JSON key for `command` exists, the extracted XML
        // parameter for `command` should NOT overwrite it.
        let leaked =
            r#"{"action":"shell>\n<parameter=command>\nXML_VALUE","command":"REAL_VALUE"}"#;
        let repaired = repair_leaked_args(leaked).expect("should repair");
        let parsed: serde_json::Value = serde_json::from_str(&repaired).unwrap();
        assert_eq!(parsed["action"], "shell");
        assert_eq!(parsed["command"], "REAL_VALUE");
    }
}
