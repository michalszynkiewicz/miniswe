//! Injected hint messages and plan-checkpoint constants used across the
//! agent loop.

use crate::config::EditMode;

// ── Plan-checkpoint thresholds and nudges ────────────────────────────

pub const PLAN_CHECKPOINT_AFTER_EDITS: u32 = 5;
pub const PLAN_HARD_BLOCK_AFTER_EDITS: u32 = 8;

/// True if `content` is a `change_signature` validator-shaped failure that's
/// safe to drop from history. We rewind the assistant message + tool results
/// and replace with a user-role corrective so the model isn't primed by its
/// own bad-shape arguments. Only fires for *schema* failures (missing keys
/// or malformed `position`), not for downstream LSP / inner-rewrite errors.
pub fn is_prunable_change_signature_failure(content: &str, success: bool) -> bool {
    !success
        && content.starts_with("✗ change_signature(")
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
        "edit_file" | "write_file" | "change_signature" | "rename" | "replace_range" | "insert_at"
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
PLAN CHECKPOINT: You have made 5 edits since the last successful plan action. Before making many more edits, review the plan: use plan(action='check') for completed steps, plan(action='refine' or 'set') if direction changed, or plan(action='show') if no step is complete yet. Further edits may be blocked if you continue without any plan action.";

pub const PLAN_CHECKPOINT_BLOCK_MESSAGE: &str = "\
Plan checkpoint required before more edits. You have continued editing after the checkpoint warning. Use any successful plan action now: plan(action='check') for completed steps, plan(action='refine' or 'set') if direction changed, or plan(action='show') if no step is complete yet.";

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

/// Injected after the server rejects the model's tool call with "Failed
/// to parse tool call arguments as JSON" (see
/// `crate::llm::TRUNCATED_TOOL_CALL_MARKER`). The previous assistant
/// turn was streamed but never committed to history (the server dropped
/// it), so we push this hint instead of a tool_result and let the agent
/// try again with a smaller operation.
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
        assert!(is_prunable_change_signature_failure(
            "✗ change_signature(add_param): missing required parameter(s): name, default\nRequired: ...",
            false,
        ));
        // Malformed position.
        assert!(is_prunable_change_signature_failure(
            "✗ change_signature(add_param): the 'position' value you sent (\"bogus\") is malformed.",
            false,
        ));
        // Combined error.
        assert!(is_prunable_change_signature_failure(
            "✗ change_signature(add_param): missing required parameter(s): name\n\nAlso: the 'position' value you sent (\"x\") is malformed.",
            false,
        ));
    }

    #[test]
    fn prunable_failure_rejects_non_validator_failures() {
        // Tool succeeded → never prune.
        assert!(!is_prunable_change_signature_failure(
            "✓ add_param: signature + 2/2 callsite(s) rewritten.",
            true,
        ));
        // Different tool prefix.
        assert!(!is_prunable_change_signature_failure(
            "✗ rename: target symbol not found",
            false,
        ));
        // change_signature with a downstream LSP / inner-rewrite error
        // (not a schema-shape failure) — keep these so the agent sees the
        // real cause.
        assert!(!is_prunable_change_signature_failure(
            "change_signature error: apply signature rewrite to src/x.rs (model output didn't match)",
            false,
        ));
    }

    #[test]
    fn is_file_write_covers_smart_and_fast_edit_tools() {
        for tool in [
            "edit_file",
            "write_file",
            "change_signature",
            "rename",
            "replace_range",
            "insert_at",
        ] {
            assert!(is_file_write(tool), "{tool} should count as a file write");
        }
        // Excluded: read-only or undo tools.
        for tool in ["file", "code", "plan", "revert", "show_rev", "check"] {
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
