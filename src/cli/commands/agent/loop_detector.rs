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

/// `loop_call_key` extended with the active skill-step tag, so a step change
/// resets streak/period detection. Every site that stores a key later
/// compared against a streak key must build it through this helper — a
/// tagged compare against an untagged record silently never matches.
pub fn loop_call_key_tagged(
    tool_name: &str,
    args: &serde_json::Value,
    step_tag: Option<&str>,
) -> String {
    let mut key = loop_call_key(tool_name, args);
    if let Some(tag) = step_tag {
        key.push('@');
        key.push_str(tag);
    }
    key
}

/// Repetitions of a cycle that constitute a loop — the detection window
/// for a period-`p` cycle is `p * CYCLE_REPS` calls (A,B,A,B,A,B for p=2).
pub const CYCLE_REPS: usize = 3;
/// Kept for callers/tests that speak of the period-2 case specifically.
pub const PERIOD2_REPS: usize = CYCLE_REPS;

/// Longest cycle the detector looks for. The call-key window in the agent
/// loop holds 12 keys, which fits `CYCLE_REPS` repetitions of a period-4
/// cycle exactly; anything longer needs a wider window, not a bigger
/// constant here.
pub const MAX_CYCLE_PERIOD: usize = 4;

/// Detect a short cycle at the END of the call-key history: the last
/// `p * CYCLE_REPS` keys repeat a pattern of length `p` for some
/// `2 <= p <= MAX_CYCLE_PERIOD`. Returns the SHORTEST such `p`, so a
/// period-2 alternation is never reported as period-4 and a plain streak
/// (period 1 — the consecutive detector owns that) is never reported at all.
///
/// The consecutive-identical detector is structurally blind to every one of
/// these shapes — any change of call resets its streak to 1. Observed
/// killing real bench runs:
/// - period 2: a byte-identical `replace_range` that breaks the AST,
///   followed by `revert` to the same rev, re-issued 130+ times until the
///   wall-clock died (observation-masking arm, 2026-07-03 matrix);
/// - period 3: the same edit → `revert` → `plan(check)` ("already checked")
///   → the same edit, 20× over 36 minutes with the model narrating "let me
///   take a completely different approach" every time (Devstral Small 2,
///   `docker_20260823_114957`). The window detector saw it 16× but at the
///   time its only action was a cold prompt eval, which a temp-0.2 model
///   reproduces straight through; it now escalates to a forced compaction.
pub fn cycle_period(history: &[String]) -> Option<usize> {
    for period in 2..=MAX_CYCLE_PERIOD {
        let need = period * CYCLE_REPS;
        if history.len() < need {
            return None; // longer periods need even more history
        }
        let tail = &history[history.len() - need..];
        let pattern = &tail[..period];
        // A pattern that is itself periodic (a,a / a,b,a,b) would have been
        // caught at the shorter period already (or is a period-1 streak).
        if pattern.iter().all(|k| k == &pattern[0]) {
            continue;
        }
        if tail
            .iter()
            .enumerate()
            .all(|(i, k)| k == &pattern[i % period])
        {
            return Some(period);
        }
    }
    None
}

/// Detect a period-2 cycle at the END of the call-key history: the last
/// `2 * PERIOD2_REPS` keys alternate A,B,A,B,A,B with A != B.
pub fn is_period2_cycle(history: &[String]) -> bool {
    cycle_period(history) == Some(2)
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

/// Whether a stored call key is a FILE EDIT (or a revert of one): the calls
/// whose byte-identical recurrence is never legitimate. Repeating the same
/// read, `cargo test` or `plan(check)` between different edits is a normal
/// rhythm; re-issuing the exact same `replace_range`/`revert`/`refactor`
/// four times in a dozen calls is a rut, whatever sits in between.
pub fn key_is_file_edit(key: &str) -> bool {
    let name = key.split_once(':').map(|(n, _)| n).unwrap_or(key);
    matches!(
        name,
        "replace_range"
            | "insert_at"
            | "revert"
            | "edit_file"
            | "write_file"
            | "delete_file"
            | "refactor"
            | "add_function_param"
            | "drop_function_param"
            | "rename_symbol"
    )
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
        assert_eq!(cycle_period(&h), None);
    }

    #[test]
    fn period3_cycle_detected_after_three_repetitions() {
        // The Devstral r3 shape: edit → revert → plan(check) → edit …
        let h = keys(&["e", "r", "p", "e", "r", "p", "e", "r", "p"]);
        assert_eq!(cycle_period(&h), Some(3));
        assert!(!is_period2_cycle(&h));
        // Two repetitions are not enough — the third edit is what proves
        // the "different approach" narration wrong.
        let h = keys(&["e", "r", "p", "e", "r", "p"]);
        assert_eq!(cycle_period(&h), None);
    }

    #[test]
    fn period3_detected_at_tail_of_longer_history() {
        let h = keys(&["x", "y", "e", "r", "p", "e", "r", "p", "e", "r", "p"]);
        assert_eq!(cycle_period(&h), Some(3));
    }

    #[test]
    fn period4_cycle_needs_twelve_keys_and_is_detected() {
        let h = keys(&["a", "b", "c", "d", "a", "b", "c", "d", "a", "b", "c"]);
        assert_eq!(cycle_period(&h), None);
        let h = keys(&["a", "b", "c", "d", "a", "b", "c", "d", "a", "b", "c", "d"]);
        assert_eq!(cycle_period(&h), Some(4));
    }

    #[test]
    fn cycle_period_reports_the_shortest_period() {
        // a,b ×6 is period 2, not period 4 — even though it also matches
        // the period-4 pattern a,b,a,b.
        let h = keys(&["a", "b", "a", "b", "a", "b", "a", "b", "a", "b", "a", "b"]);
        assert_eq!(cycle_period(&h), Some(2));
        // a,a,b ×3: period 3 (a,a,b is not itself periodic).
        let h = keys(&["a", "a", "b", "a", "a", "b", "a", "a", "b"]);
        assert_eq!(cycle_period(&h), Some(3));
    }

    #[test]
    fn cycle_period_ignores_plain_streaks() {
        let h = keys(&["a"; 12]);
        assert_eq!(cycle_period(&h), None);
    }

    #[test]
    fn cycle_period_rejects_a_cycle_that_broke_at_the_end() {
        let h = keys(&["e", "r", "p", "e", "r", "p", "e", "r", "x"]);
        assert_eq!(cycle_period(&h), None);
    }

    #[test]
    fn tagged_key_is_stable_and_differs_from_untagged() {
        // The recovery ladder compares a stored failure key against the
        // current streak key; both must come from the same constructor.
        let args = json!({"action": "shell", "command": "zarf package create ."});
        let tag = Some("uds-package/create");
        assert_eq!(
            loop_call_key_tagged("file", &args, tag),
            loop_call_key_tagged("file", &args, tag)
        );
        assert_ne!(
            loop_call_key_tagged("file", &args, tag),
            loop_call_key("file", &args)
        );
    }

    #[test]
    fn tagged_key_without_tag_equals_plain_key() {
        let args = json!({"path": "x.rs"});
        assert_eq!(
            loop_call_key_tagged("write_file", &args, None),
            loop_call_key("write_file", &args)
        );
    }

    #[test]
    fn key_is_file_edit_covers_edit_family_only() {
        assert!(key_is_file_edit(&loop_call_key(
            "replace_range",
            &json!({"path": "x.rs", "start": 1, "end": 2, "content": "y"})
        )));
        assert!(key_is_file_edit(&loop_call_key(
            "revert",
            &json!({"path": "x.rs", "rev": 0})
        )));
        assert!(key_is_file_edit(&loop_call_key(
            "refactor",
            &json!({"action": "add_param", "position": "fn f("})
        )));
        assert!(!key_is_file_edit(&loop_call_key(
            "file",
            &json!({"action": "shell", "command": "cargo test"})
        )));
        assert!(!key_is_file_edit(&loop_call_key(
            "plan",
            &json!({"action": "check"})
        )));
        assert!(!key_is_file_edit(&loop_call_key_tagged(
            "file",
            &json!({"action": "read", "path": "x.rs"}),
            Some("step")
        )));
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
