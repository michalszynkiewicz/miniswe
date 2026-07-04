//! Canonical keying for tool-call loop detection.
//!
//! The agent loop tracks the last few `(tool_name, args)` pairs it has
//! dispatched. When the same key shows up 3× in a row, the
//! `loop_detected_hint` injection fires. Keys must be stable under
//! irrelevant JSON differences (object key ordering, insignificant
//! whitespace), which is what `canonical_json` provides.

pub fn loop_call_key(tool_name: &str, args: &serde_json::Value) -> String {
    format!("{tool_name}:{}", canonical_json(args))
}

/// Repetitions of an alternating A/B pair that constitute a period-2 loop —
/// the detection window is `2 * PERIOD2_REPS` calls (A,B,A,B,A,B).
pub const PERIOD2_REPS: usize = 3;

/// Detect a period-2 cycle at the END of the call-key history: the last
/// `2 * PERIOD2_REPS` keys alternate A,B,A,B,A,B with A != B.
///
/// The consecutive-identical detector is structurally blind to this shape —
/// an alternating pair resets its streak to 1 every round. Observed killing
/// real bench runs: a byte-identical `replace_range` that breaks the AST,
/// followed by `revert` to the same rev, re-issued 130+ times until the
/// wall-clock died (observation-masking arm, 2026-07-03 matrix).
pub fn is_period2_cycle(history: &[String]) -> bool {
    let need = 2 * PERIOD2_REPS;
    if history.len() < need {
        return false;
    }
    let tail = &history[history.len() - need..];
    let (a, b) = (&tail[0], &tail[1]);
    if a == b {
        return false; // period-1 streak — the consecutive detector owns that
    }
    tail.iter().step_by(2).all(|k| k == a) && tail.iter().skip(1).step_by(2).all(|k| k == b)
}

/// Whether a stored call key (from `loop_call_key`) refers to a mutating
/// call. Used to judge a period-2 cycle by BOTH its members — an edit↔read
/// alternation is still a harmful loop even when the current call is the
/// read half.
pub fn key_is_mutating(key: &str) -> bool {
    let (name, json) = key.split_once(':').unwrap_or((key, "{}"));
    let args: serde_json::Value = serde_json::from_str(json).unwrap_or(serde_json::json!({}));
    is_mutating_call(name, &args)
}

/// True if the tool call mutates state (file contents, revision table,
/// plan, scratchpad, etc.). Three identical mutating calls in a row are a
/// real loop worth aborting. Three identical read-only calls are just
/// wasted tokens — worth a nudge, not a kill.
pub fn is_mutating_call(tool_name: &str, args: &serde_json::Value) -> bool {
    match tool_name {
        // Top-level read-only inspection tools
        "show_rev" | "check" => false,

        // Top-level mutators
        "replace_range"
        | "insert_at"
        | "revert"
        | "edit_file"
        | "write_file"
        | "delete_file"
        | "task_update"
        | "spawn_agents"
        | "mcp_use"
        | "add_function_param"
        | "drop_function_param"
        | "rename_symbol" => true,

        // Grouped tools — split by action.
        "file" => {
            // file(action='read'|'search'|'help') is read-only;
            // file(action='shell') runs arbitrary commands → treat as mutating.
            !matches!(
                args.get("action").and_then(|v| v.as_str()),
                Some("read") | Some("search") | Some("help") | None
            )
        }
        "code" => {
            // All code(action=*) variants today are read-only.
            false
        }
        "plan" => {
            // plan(action='set'|'check'|'refine') changes plan.md.
            // plan(action='show'|'help') is read-only.
            matches!(
                args.get("action").and_then(|v| v.as_str()),
                Some("set") | Some("check") | Some("refine") | Some("scratchpad")
            )
        }
        "web" => {
            // Web fetches/searches don't mutate local state, but they're
            // expensive and externally visible. Treat as read-only here —
            // the per-session approval gate already covers cost concerns.
            false
        }

        // Anything we don't recognize — be conservative and treat as
        // mutating so the detector still bails on potentially destructive
        // behavior. Read-only additions can opt out by name.
        _ => true,
    }
}

pub fn canonical_json(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::String(s) => serde_json::to_string(s).unwrap_or_else(|_| "\"\"".into()),
        serde_json::Value::Array(items) => {
            let inner = items
                .iter()
                .map(canonical_json)
                .collect::<Vec<_>>()
                .join(",");
            format!("[{inner}]")
        }
        serde_json::Value::Object(map) => {
            let mut entries = map.iter().collect::<Vec<_>>();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            let inner = entries
                .into_iter()
                .map(|(k, v)| {
                    let key = serde_json::to_string(k).unwrap_or_else(|_| "\"\"".into());
                    format!("{key}:{}", canonical_json(v))
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("{{{inner}}}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn object_key_order_does_not_affect_canonical_form() {
        let a = canonical_json(&json!({ "b": 1, "a": 2 }));
        let b = canonical_json(&json!({ "a": 2, "b": 1 }));
        assert_eq!(a, b);
    }

    #[test]
    fn loop_key_combines_name_and_canonical_args() {
        let key = loop_call_key("write_file", &json!({ "path": "x.rs" }));
        assert_eq!(key, "write_file:{\"path\":\"x.rs\"}");
    }

    #[test]
    fn mutating_top_level_tools_are_mutating() {
        for name in [
            "replace_range",
            "insert_at",
            "revert",
            "edit_file",
            "write_file",
            "delete_file",
            "task_update",
            "spawn_agents",
            "mcp_use",
        ] {
            assert!(
                is_mutating_call(name, &json!({})),
                "{name} should be mutating"
            );
        }
    }

    #[test]
    fn read_only_top_level_tools_are_not_mutating() {
        for name in ["show_rev", "check"] {
            assert!(
                !is_mutating_call(name, &json!({})),
                "{name} should be read-only"
            );
        }
    }

    #[test]
    fn file_action_read_search_help_are_read_only() {
        for action in ["read", "search", "help"] {
            assert!(
                !is_mutating_call("file", &json!({"action": action})),
                "file({action}) should be read-only"
            );
        }
    }

    #[test]
    fn file_action_shell_is_mutating() {
        assert!(is_mutating_call(
            "file",
            &json!({"action": "shell", "command": "ls"})
        ));
    }

    #[test]
    fn plan_show_is_read_only_set_check_refine_are_mutating() {
        assert!(!is_mutating_call("plan", &json!({"action": "show"})));
        assert!(is_mutating_call("plan", &json!({"action": "set"})));
        assert!(is_mutating_call("plan", &json!({"action": "check"})));
        assert!(is_mutating_call("plan", &json!({"action": "refine"})));
    }

    #[test]
    fn code_actions_are_read_only() {
        for action in ["repo_map", "diagnostics", "project_info"] {
            assert!(!is_mutating_call("code", &json!({"action": action})));
        }
    }

    #[test]
    fn unknown_tool_treated_as_mutating_for_safety() {
        assert!(is_mutating_call("brand_new_tool", &json!({})));
    }

    fn keys(seq: &[&str]) -> Vec<String> {
        seq.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn period2_cycle_detected_on_alternating_pair() {
        let h = keys(&["a", "b", "a", "b", "a", "b"]);
        assert!(is_period2_cycle(&h));
    }

    #[test]
    fn period2_detects_at_tail_of_longer_history() {
        let h = keys(&["x", "y", "z", "a", "b", "a", "b", "a", "b"]);
        assert!(is_period2_cycle(&h));
    }

    #[test]
    fn period2_not_detected_below_three_repetitions() {
        let h = keys(&["a", "b", "a", "b"]);
        assert!(!is_period2_cycle(&h));
    }

    #[test]
    fn period2_rejects_identical_pair_as_period1() {
        // a,a,a,a,a,a is a streak, not a cycle — the consecutive detector owns it.
        let h = keys(&["a", "a", "a", "a", "a", "a"]);
        assert!(!is_period2_cycle(&h));
    }

    #[test]
    fn period2_rejects_broken_alternation() {
        let h = keys(&["a", "b", "a", "b", "b", "a"]);
        assert!(!is_period2_cycle(&h));
    }

    #[test]
    fn key_is_mutating_parses_stored_keys() {
        assert!(key_is_mutating(&loop_call_key(
            "replace_range",
            &json!({"path": "x.rs", "start": 1, "end": 2, "content": "y"})
        )));
        assert!(key_is_mutating(&loop_call_key(
            "revert",
            &json!({"path": "x.rs", "rev": 3})
        )));
        assert!(!key_is_mutating(&loop_call_key(
            "file",
            &json!({"action": "read", "path": "x.rs"})
        )));
    }
}
