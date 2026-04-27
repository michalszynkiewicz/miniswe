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

/// True if the tool call mutates state (file contents, revision table,
/// plan, scratchpad, etc.). Three identical mutating calls in a row are a
/// real loop worth aborting. Three identical read-only calls are just
/// wasted tokens — worth a nudge, not a kill.
pub fn is_mutating_call(tool_name: &str, args: &serde_json::Value) -> bool {
    match tool_name {
        // Top-level read-only inspection tools
        "show_rev" | "check" => false,

        // Top-level mutators
        "replace_range" | "insert_at" | "revert" | "edit_file" | "write_file" | "delete_file"
        | "task_update" | "spawn_agents" | "mcp_use" => true,

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
}
