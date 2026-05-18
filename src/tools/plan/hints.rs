//! Architecture-review hints injected alongside plan snapshots. These
//! nudge the model to plan call-site updates, order definitions before
//! callers, and use `plan(action='refine')` when a step turns out to be
//! too coarse.

use crate::config::EditMode;

const ARCHITECTURE_REVIEW_HINT_SMART: &str = "\
Before editing: re-read 1-2 central files to verify each step targets real code. Edit definitions before callers. plan(action='refine') if anything's off.";

const ARCHITECTURE_REVIEW_HINT_FAST: &str = "\
Before editing: re-read 1-2 central files to verify each step targets real code. Edit definitions before callers. plan(action='refine') if anything's off.";

pub fn architecture_review_hint(edit_mode: EditMode) -> &'static str {
    match edit_mode {
        EditMode::Smart => ARCHITECTURE_REVIEW_HINT_SMART,
        EditMode::Fast => ARCHITECTURE_REVIEW_HINT_FAST,
    }
}
