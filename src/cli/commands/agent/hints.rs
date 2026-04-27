//! Injected hint messages and plan-checkpoint constants used across the
//! agent loop.

use crate::config::EditMode;

// ── Plan-checkpoint thresholds and nudges ────────────────────────────

pub const PLAN_CHECKPOINT_AFTER_EDITS: u32 = 5;
pub const PLAN_HARD_BLOCK_AFTER_EDITS: u32 = 8;

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
            "ERROR: You are in a loop — this exact tool call has been repeated 3 times in a row. Stop retrying it in this turn. If you were repeating replace_range/insert_at with the same args, the edit already landed (or was rejected) — inspect the revision table with show_rev before trying again. If you were repeating revert to the same rev, pick a different live rev or move on. Use file(action='read') to re-check current state."
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
        assert!(!hint.contains("edit_file"));
    }
}
