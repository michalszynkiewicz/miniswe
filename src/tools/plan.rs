//! Structured planning tool — high-level plan with checkable steps.
//!
//! The model creates a plan at session start, then checks off steps
//! as it completes them. Each step declares whether the project should
//! compile at its end (`compile: true`). The `check` action enforces
//! this by running `cargo check` (or the appropriate checker). If a
//! step proves too coarse, the model can `refine` it into substeps.
//!
//! The plan persists in `.miniswe/plan.md` and is injected into context
//! each round.

mod actions;
mod hints;
mod step;
mod validate;

use std::fs;

use crate::config::Config;

pub use actions::execute;

/// Check if a plan has been created.
pub fn plan_exists(config: &Config) -> bool {
    let plan_path = config.miniswe_dir().join("plan.md");
    plan_path.exists()
        && fs::read_to_string(&plan_path)
            .map(|c| !c.trim().is_empty())
            .unwrap_or(false)
}

/// Load the current plan for context injection.
pub fn load_plan(config: &Config) -> Option<String> {
    let plan_path = config.miniswe_dir().join("plan.md");
    let content = fs::read_to_string(plan_path).ok()?;
    if content.trim().is_empty() {
        None
    } else {
        Some(content)
    }
}

/// Short summary rendered when the project fails to compile — reminds
/// the model of plan progress and suggests `plan(action='refine')` when
/// a step's scope has slipped.
pub fn failure_hint(config: &Config) -> Option<String> {
    let steps = step::parse_steps(&load_plan(config)?);
    if steps.is_empty() {
        return None;
    }

    let done = steps.iter().filter(|step| step.checked).count();
    let done_preview = steps
        .iter()
        .enumerate()
        .filter(|(_, step)| step.checked)
        .rev()
        .take(2)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .map(|(idx, step)| {
            format!(
                "{} {}",
                idx + 1,
                crate::truncate_chars(&step.description, 60)
            )
        })
        .collect::<Vec<_>>();
    let next_preview = steps
        .iter()
        .enumerate()
        .filter(|(_, step)| !step.checked)
        .take(3)
        .map(|(idx, step)| {
            format!(
                "{} {}",
                idx + 1,
                crate::truncate_chars(&step.description, 60)
            )
        })
        .collect::<Vec<_>>();

    let mut parts = vec![format!("Plan: {done}/{} done.", steps.len())];
    if !done_preview.is_empty() {
        parts.push(format!("Done: {}.", done_preview.join("; ")));
    }
    if next_preview.is_empty() {
        parts.push("Next: all checked.".to_string());
    } else {
        parts.push(format!("Next: {}.", next_preview.join("; ")));
    }
    parts.push(
        "Before fixing, check if this error means the plan changed; if so use plan(action='refine')."
            .to_string(),
    );

    Some(parts.join(" "))
}
