//! Repair-context formatting: renders prior progress and failures into
//! planner-facing blocks for the retry attempts.

use super::*;

pub(super) fn build_retry_feedback(last_error: &str, _signature_grounding: Option<&str>) -> String {
    let mut feedback = last_error.to_string();
    if is_signature_mismatch_error(last_error) {
        feedback.push_str(
            "\nDo not repeat the same patch shape on the same lines. Re-check the current code before retrying and narrow the edit to only the lines that clearly need to change.",
        );
    }
    feedback
}

pub(super) fn is_signature_mismatch_error(error: &str) -> bool {
    error.contains("expected ")
        && (error.contains("arguments, found") || error.contains("mismatched types"))
}

/// Render a `RepairContext` into a planner-facing block. Used at the top
/// of both the windowed pre-plan prompts and the finalize prompt so the
/// model can see exactly what the previous iteration tried and how it
/// landed. The shape differs for cleanly-applied vs failed attempts —
/// see `format_prior_applied` and `format_prior_failed` below.
pub(super) fn format_repair_context(ctx: &RepairContext) -> String {
    if ctx.cleanly_applied {
        format_prior_applied(ctx)
    } else {
        format_prior_failed(ctx)
    }
}

/// Render a cleanly-applied previous iteration so the model knows
/// exactly which steps it just ran and can decide the next verdict.
/// No "failed" language, no recovery framing — the file view already
/// reflects the applied steps, and the model is being asked to judge
/// whether the task is done now (COMPLETE), needs more work (edit-plan
/// blocks), or hit a wall (FAILED).
pub(super) fn format_prior_applied(ctx: &RepairContext) -> String {
    let applied = if ctx.completed_steps.is_empty() {
        "(no steps — the previous round was a no-op)\n".to_string()
    } else {
        format_completed_steps_compact(&ctx.completed_steps)
    };
    format!(
        "The previous iteration applied the following steps cleanly (they are ALREADY reflected in the file content below):\n{applied}\n\
         Decide the next verdict against the current file state. If those steps already accomplished the task, return `COMPLETE`. \
         If more edits are still needed to finish the task, emit `LITERAL_REPLACE`/`SMART_EDIT` blocks for the remaining work. \
         If the task cannot be completed even with more edits (e.g. it needs cross-file changes), return `FAILED: <reason>`.\n\n",
    )
}

/// Render a failed previous iteration (step blew up, LSP regressed,
/// parse error, or empty response) so the planner can reason about the
/// obstacle and pick a recovery strategy — or declare FAILED if the
/// obstacle is unrecoverable.
pub(super) fn format_prior_failed(ctx: &RepairContext) -> String {
    let previous_plan = if ctx.previous_plan.is_empty() {
        "(empty)\n".to_string()
    } else {
        format_edit_plan_steps(&ctx.previous_plan)
    };

    let completed = if ctx.completed_steps.is_empty() {
        "(none — the first step failed, file is unchanged from the initial state)\n".to_string()
    } else {
        format_completed_steps_compact(&ctx.completed_steps)
    };

    let failed = match &ctx.failed_step {
        Some(step) => format_edit_plan_steps(std::slice::from_ref(step)),
        None => "(no individual step failed; the plan executed in full but post-validation rejected the result — see failure reason below)\n".to_string(),
    };

    let failure_block = match &ctx.lsp_regression {
        Some(reg) => format_lsp_regression_for_planner(reg),
        None => ctx.failure_reason.clone(),
    };

    format!(
        "The previous iteration failed. Use the structured information below to decide the next verdict.\n\n\
         Previous edit plan (as tried):\n{previous_plan}\n\
         Steps that succeeded and have ALREADY been applied to the file shown below:\n{completed}\n\
         Step that FAILED:\n{failed}\n\
         Failure reason:\n{failure_block}\n\n\
         Decide against the current file state: plan recovery edits if you can, return `COMPLETE` if the task is already done despite the failure, or return `FAILED: <reason>` if the obstacle is unrecoverable in this file.\n\n",
    )
}

/// Format repair step outcomes that overlap a given line range, for
/// injection into the windowed observation prompt. Shows which steps
/// in this slice succeeded (✓), which failed (✗), and which are still
/// pending, so the model can reason about what remains to be done
/// without re-proposing edits for already-handled locations.
pub(super) fn format_repair_steps_for_window(
    ctx: &RepairContext,
    win_start: usize,
    win_end: usize,
) -> String {
    let overlaps = |step: &EditPlanStep| -> bool {
        step.start_line() <= win_end && step.end_line() >= win_start
    };
    let mut out = String::new();
    let mut any = false;

    for step in &ctx.completed_steps {
        if overlaps(step) {
            if !any {
                out.push_str("Previous edit attempt — steps in this slice:\n");
                any = true;
            }
            out.push_str(&format_completed_steps_compact(std::slice::from_ref(step)));
        }
    }

    if let Some(ref step) = ctx.failed_step
        && overlaps(step)
    {
        if !any {
            out.push_str("Previous edit attempt — steps in this slice:\n");
            any = true;
        }
        let reason_preview = if ctx.failure_reason.len() > 120 {
            format!("{}…", &ctx.failure_reason[..117])
        } else {
            ctx.failure_reason.clone()
        };
        out.push_str(&format!(
            "  ✗ L{}-L{}: FAILED ({})\n",
            step.start_line(),
            step.end_line(),
            reason_preview,
        ));
    }

    if any {
        out.push('\n');
    }
    out
}

/// Render an `LspRegression` with per-error post-edit source snippets so
/// the replanner sees *which lines of its patch* produced each new
/// error, not just the raw file:line:col blob.
///
/// Layout:
///
/// ```text
/// LSP diagnostics worsened: B -> T error(s)
/// [1] L<line>:<col>: <message>
///     post-edit context:
///     <snippet>
/// [2] ...
/// ```
///
/// The snippet uses the *candidate* file content captured at validation
/// time so the replanner sees the broken state it produced, not a stale
/// pre-edit view.
pub(super) fn format_lsp_regression_for_planner(reg: &LspRegression) -> String {
    const CONTEXT_RADIUS: usize = 5;
    let total = reg.errors.len() + reg.extra_error_count;
    let mut out = format!(
        "LSP diagnostics worsened: {} -> {} error(s)\n\
         The following errors were introduced by the previous attempt's patch. \
         Each entry shows the post-edit source around the error so you can see exactly what your patch produced.\n",
        reg.baseline_count, total
    );

    let lines: Vec<&str> = reg.candidate_content.lines().collect();
    for (idx, err) in reg.errors.iter().enumerate() {
        out.push_str(&format!(
            "\n[{n}] L{line}:{col}: {msg}\n",
            n = idx + 1,
            line = err.line,
            col = err.column,
            msg = err.message,
        ));
        if err.line == 0 || err.line > lines.len() {
            out.push_str("    (line out of range for post-edit content)\n");
            continue;
        }
        let zero_based = err.line - 1;
        let start = zero_based.saturating_sub(CONTEXT_RADIUS);
        let end = (zero_based + CONTEXT_RADIUS + 1).min(lines.len());
        out.push_str("    post-edit context:\n");
        for (i, line) in lines[start..end].iter().enumerate() {
            let line_no = start + i + 1;
            let marker = if line_no == err.line { ">>" } else { "  " };
            out.push_str(&format!("    {marker} {line_no:>5} │ {line}\n"));
        }
    }

    if reg.extra_error_count > 0 {
        out.push_str(&format!(
            "\n... and {} more error(s) not shown\n",
            reg.extra_error_count
        ));
    }
    out
}
