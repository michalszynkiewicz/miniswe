//! Injected hint messages and plan-checkpoint constants used across the
//! agent loop.

use crate::config::EditMode;

// ── Plan-checkpoint thresholds and nudges ────────────────────────────

pub const PLAN_CHECKPOINT_AFTER_EDITS: u32 = 5;

/// True if `content` is a `refactor` validator-shaped failure that's safe
/// to drop from history. We rewind the assistant message + tool results
/// and replace with a user-role corrective so the model isn't primed by its
/// own bad-shape arguments. Only fires for *schema* failures (missing keys
/// or malformed `position`), not for downstream LSP / inner-rewrite errors.
pub fn is_prunable_refactor_failure(content: &str, success: bool) -> bool {
    !success
        && (content.starts_with("✗ refactor(") || content.starts_with("✗ change_signature("))
        && (content.contains("missing required parameter") || content.contains("is malformed"))
}

/// True if the tool call writes to a source file.
///
/// Drives the plan-checkpoint counter, the `PLAN_PROGRESS_NUDGE`
/// appended to results, and the stall detector's edit-progress reset.
/// Both `run` and `repl` agent loops consult this — keep the two call
/// sites in sync by funnelling through this helper.
///
/// Excludes `revert` (undo, not new progress) and read-only tools.
pub fn is_file_write(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "edit_file"
            | "write_file"
            | "refactor"
            | "replace_range"
            | "insert_at"
            | "add_function_param"
            | "drop_function_param"
            | "rename_symbol"
    )
}

/// Hide write-tools from the model's tool list when no plan exists yet.
/// The runtime would otherwise reject these calls with "Create a plan
/// first" — a schema↔runtime mismatch that costs models 1-2 rounds to
/// learn through rejection. By dropping the tools entirely until a plan
/// is set, the model only sees what it can actually use.
///
/// The runtime gate stays in place as defense-in-depth.
///
/// Tools field is a per-request OpenAI parameter, so swapping mid-session
/// is supported. The transition fires once (on first plan(action='set'))
/// per session — that's a single prompt-cache miss, not per-turn churn.
pub fn visible_tool_defs(
    all: &[crate::llm::ToolDefinition],
    plan_exists: bool,
) -> Vec<crate::llm::ToolDefinition> {
    if plan_exists {
        return all.to_vec();
    }
    all.iter()
        .filter(|t| !is_file_write(&t.function.name))
        .cloned()
        .collect()
}

pub const PLAN_PROGRESS_NUDGE: &str = "\
PLAN STATUS: If this edit completed one of your current plan steps, mark it now with plan(action='check', step=N). If the work split changed, use plan(action='refine') or plan(action='set').";

pub const PLAN_CHECKPOINT_WARNING: &str = "\
PLAN CHECKPOINT: You have made 5 edits since the last successful plan action. Before making many more edits, review the plan: use plan(action='check') for completed steps, plan(action='refine' or 'set') if direction changed, or plan(action='show') if no step is complete yet.";

/// Pushed once per turn when the assistant returns no tool calls but the
/// plan still has unchecked steps. Intentionally short and open-ended —
/// the model decides whether to continue or really stop.
pub const PREMATURE_EXIT_NUDGE: &str = "\
Stopping. Are you sure? Check the plan — if steps remain, continue.";

/// Pushed when a read/inspection tool repeats 3× with identical args.
/// Doesn't end the round — the call has no side effects, the model is
/// just wasting tokens. Surface it once so the model notices.
pub const REPEATED_READ_NUDGE: &str = "\
You just made this same read/inspection call 3 times in a row. The result hasn't changed. What specifically are you looking for? Try a narrower search, a different range, or move on to making an edit.";

/// Escalation when the nudge failed: the model re-entered an identical read
/// loop after a [`REPEATED_READ_NUDGE`]. Tail wording alone is inert here
/// (warm-replay probe, 2026-07-15: 7-8/8 kept looping regardless of message
/// — the rut lives in the cache-hot prompt prefix), so the agent loop also
/// forces a context compaction before the next request, which broke the loop
/// 8/8. Contains the "same read/inspection call" GUARD_MARKERS phrase so
/// compaction never masks this message itself.
pub const REPEATED_READ_ESCALATION: &str = "\
You made this same read/inspection call 3 times in a row AGAIN — the result STILL has not changed, and the answer is not in this output. Older history has been compacted so you can re-approach. Re-orient from the plan and current state, then take a DIFFERENT action now: make an edit, run a check or validator, or consult the docs.";

// ── Error-recovery hints ─────────────────────────────────────────────

/// Injected as a user-role message after the model repeats the same
/// tool call 3× in a row. Fast-mode models also loop on `replace_range`
/// (same range + same bytes) and on `revert` (same rev), so fast mode
/// points the model at the revision-table tools (`show_rev`, a different
/// `rev`) rather than the smart-mode edit surface.
pub fn loop_detected_hint(edit_mode: EditMode) -> &'static str {
    match edit_mode {
        EditMode::Smart => {
            "ERROR: You are in a loop — this exact tool call has been repeated 3 times in a row. Stop retrying it in this turn. Try a different approach: use file(action='search'), file(action='read'), code(action='repo_map'), code(action='diagnostics'), or edit_file for semantic edits."
        }
        EditMode::Fast => {
            "ERROR: You are in a loop — this exact tool call has been repeated 3 times in a row. Stop retrying it in this turn. If you were repeating replace_range/insert_at with the same args, the edit already landed (or was rejected) — inspect the revision table with show_rev before trying again. If you were repeating revert to the same rev, pick a different live rev or move on. For structural rewrites that keep failing line-by-line, switch to edit_file (semantic patch). Use file(action='read') to re-check current state."
        }
    }
}

/// Injected when the short-cycle detector fires: the model cycles through
/// the SAME two to four calls (typically an edit that breaks the file, a
/// revert that undoes it, and sometimes a `plan(check)` in between).
/// Distinct from `loop_detected_hint` because the calls are not
/// consecutive-identical — telling the model "this exact call repeated 3
/// times" would be false and confusing here.
pub fn cycle_loop_hint(period: usize) -> String {
    format!(
        "ERROR: You are in an edit↔revert loop — you have cycled through the SAME {period} tool \
         calls several times in a row. The edit you keep re-submitting is what breaks the file; \
         submitting it again UNCHANGED will break it the same way, and reverting just resets for \
         another failure. Stop this cycle now: re-read the exact lines, check the project-wide \
         errors for the REAL blocker (it may be in a different file), and write a DIFFERENT, \
         smaller edit — do not re-issue any of the {period} calls you have been cycling through."
    )
}

/// Guidance for a tool call whose arguments were cut off by the output
/// limit. Reaches the model two ways: as a user-role hint when the request
/// itself failed (server-side "Failed to parse tool call arguments as
/// JSON" — `crate::llm::TRUNCATED_TOOL_CALL_MARKER` — or our own streaming
/// size cap), and appended to the tool_result of a call that arrived
/// truncated and was stubbed by `crate::llm::sanitize_truncated_tool_calls`
/// instead of being persisted (the server re-parses every historical
/// call on every later request; one broken call failed them all).
pub fn truncated_tool_call_hint(edit_mode: EditMode) -> &'static str {
    match edit_mode {
        EditMode::Smart => {
            "\
Your previous tool call was rejected because the server could not parse its arguments as JSON — \
most likely the generation hit max_tokens mid-string and the JSON got truncated. \
Try a smaller operation: prefer edit_file over write_file for existing files, \
break large writes into multiple smaller tool calls, \
and avoid embedding very long literals in a single argument."
        }
        EditMode::Fast => {
            "\
Your previous tool call was rejected because the server could not parse its arguments as JSON — \
most likely the generation hit max_tokens mid-string and the JSON got truncated. \
Try a smaller operation: prefer replace_range or insert_at over write_file for existing files, \
break large writes into multiple smaller tool calls, \
and avoid embedding very long literals in a single argument."
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loop_hint_smart_mentions_edit_file() {
        let hint = loop_detected_hint(EditMode::Smart);
        assert!(hint.contains("edit_file"));
    }

    #[test]
    fn loop_hint_fast_mentions_revision_table_tools() {
        let hint = loop_detected_hint(EditMode::Fast);
        assert!(hint.contains("show_rev"));
        assert!(hint.contains("revert"));
        // Fast mode now exposes edit_file too, so the loop hint can
        // suggest it as a structural-rewrite escape hatch.
        assert!(hint.contains("edit_file"));
    }

    #[test]
    fn prunable_failure_recognizes_validator_shapes() {
        // Missing keys.
        assert!(is_prunable_refactor_failure(
            "✗ refactor(add_param): missing required parameter(s): name, callsite_fill_in\nRequired: ...",
            false,
        ));
        // Malformed position.
        assert!(is_prunable_refactor_failure(
            "✗ refactor(add_param): the 'position' value you sent (\"bogus\") is malformed.",
            false,
        ));
        // Combined error.
        assert!(is_prunable_refactor_failure(
            "✗ refactor(add_param): missing required parameter(s): name\n\nAlso: the 'position' value you sent (\"x\") is malformed.",
            false,
        ));
        // Backward compat: still match the old change_signature prefix in
        // case `--continue` carries history from a pre-rename session.
        assert!(is_prunable_refactor_failure(
            "✗ change_signature(add_param): missing required parameter(s): name",
            false,
        ));
    }

    #[test]
    fn prunable_failure_rejects_non_validator_failures() {
        // Tool succeeded → never prune.
        assert!(!is_prunable_refactor_failure(
            "✓ COMPLETE — definition and all 2 callsites are now consistent.",
            true,
        ));
        // Different tool prefix.
        assert!(!is_prunable_refactor_failure(
            "✗ rename: target symbol not found",
            false,
        ));
        // refactor with a downstream LSP / inner-rewrite error (not a
        // schema-shape failure) — keep these so the agent sees the real cause.
        assert!(!is_prunable_refactor_failure(
            "refactor error: apply signature rewrite to src/x.rs (model output didn't match)",
            false,
        ));
    }

    #[test]
    fn is_file_write_covers_smart_and_fast_edit_tools() {
        for tool in [
            "edit_file",
            "write_file",
            "refactor",
            "replace_range",
            "insert_at",
        ] {
            assert!(is_file_write(tool), "{tool} should count as a file write");
        }
        // Excluded: read-only or undo tools, and the deprecated standalone
        // names that have been folded into `refactor`.
        for tool in [
            "file",
            "code",
            "plan",
            "revert",
            "show_rev",
            "check",
            "change_signature",
            "rename",
        ] {
            assert!(
                !is_file_write(tool),
                "{tool} must not count as a file write"
            );
        }
    }

    fn td(name: &str) -> crate::llm::ToolDefinition {
        crate::llm::ToolDefinition {
            r#type: "function".into(),
            function: crate::llm::FunctionDefinition {
                name: name.into(),
                description: String::new(),
                parameters: serde_json::json!({}),
            },
        }
    }

    #[test]
    fn visible_tool_defs_hides_writes_when_no_plan() {
        let all = vec![
            td("file"),
            td("plan"),
            td("edit_file"),
            td("write_file"),
            td("replace_range"),
            td("insert_at"),
            td("revert"),
        ];
        let visible = visible_tool_defs(&all, false);
        let names: Vec<&str> = visible.iter().map(|t| t.function.name.as_str()).collect();
        assert_eq!(names, ["file", "plan", "revert"]);
    }

    #[test]
    fn visible_tool_defs_returns_full_list_when_plan_exists() {
        let all = vec![td("file"), td("edit_file"), td("revert")];
        let visible = visible_tool_defs(&all, true);
        let names: Vec<&str> = visible.iter().map(|t| t.function.name.as_str()).collect();
        assert_eq!(names, ["file", "edit_file", "revert"]);
    }
}
