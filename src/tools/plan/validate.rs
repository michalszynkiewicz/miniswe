//! Plan-level invariant checks.
//!
//! - The plan must have at least one step.
//! - The final step must be `compile: true` — the plan cannot end with a
//!   broken tree.
//! - Any `compile: false` step must carry a `reason` and must be
//!   followed by a later `compile: true` step that restores the
//!   invariant.

use super::step::Step;

/// Max total plan steps.
pub const MAX_PLAN_STEPS: usize = 30;

/// Validate plan invariants:
/// - Last step must be compile: true
/// - Any compile: false must have a reason
/// - Any compile: false must be followed by a compile: true before end
pub fn validate_steps(steps: &[Step]) -> Result<(), String> {
    if steps.is_empty() {
        return Err("Plan must have at least one step.".into());
    }

    // Last step must compile
    if !steps.last().unwrap().compile {
        return Err(
            "Last step must have compile: true. The plan cannot end with a broken tree.".into(),
        );
    }

    // Check compile: false steps
    for (i, step) in steps.iter().enumerate() {
        if !step.compile {
            // Must have reason
            if step.reason.as_ref().is_none_or(|r| r.trim().is_empty()) {
                return Err(format!(
                    "Step {} has compile: false but no reason. Explain why the tree will be broken.",
                    i + 1
                ));
            }
            // Must be followed by a compile: true before end
            let has_restore = steps[i + 1..].iter().any(|s| s.compile);
            if !has_restore {
                return Err(format!(
                    "Step {} has compile: false but no subsequent compile: true step to restore the invariant.",
                    i + 1
                ));
            }
        }
    }

    Ok(())
}
