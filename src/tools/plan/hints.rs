//! Architecture-review hints injected alongside plan snapshots. These
//! nudge the model to plan call-site updates, order definitions before
//! callers, and use `plan(action='refine')` when a step turns out to be
//! too coarse.

use crate::config::EditMode;

const ARCHITECTURE_REVIEW_HINT_SMART: &str = "\
Architecture check before editing:
- Check whether the plan changes the right abstraction level.
- If a change affects a function/API, plan the call-site and test updates too.
- Order steps so definitions are edited before callers (e.g. update a function signature before updating its call sites). This avoids intermediate compile errors that block edit_file's LSP validation.
- If a step must temporarily break compilation (e.g. updating a caller before the callee), pass lsp_validation='off' to edit_file and mention the expected error in the task description so the inner planner doesn't bail.
- If the plan edits many similar components, consider plan(action='refine') before editing.";

const ARCHITECTURE_REVIEW_HINT_FAST: &str = "\
Architecture check before editing:
- Check whether the plan changes the right abstraction level.
- If a change affects a function/API, plan the call-site and test updates too.
- Order steps so definitions are edited before callers (e.g. update a function signature before updating its call sites).
- Each step lands with replace_range / insert_at; if an edit regresses, revert that revision rather than layering more edits on top.
- Group related edits into one step only when they fit in a single replace_range; otherwise make them separate steps.";

pub fn architecture_review_hint(edit_mode: EditMode) -> &'static str {
    match edit_mode {
        EditMode::Smart => ARCHITECTURE_REVIEW_HINT_SMART,
        EditMode::Fast => ARCHITECTURE_REVIEW_HINT_FAST,
    }
}
