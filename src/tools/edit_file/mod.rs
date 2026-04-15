//! edit_file tool — LLM plans and applies bounded edits atomically.
//!
//! The model describes the task, miniswe asks the inner LLM for a structured
//! edit plan, and then executes the planned steps against an in-memory working
//! copy. If some steps succeed and later ones fail, the successful progress is
//! preserved in memory and the tool asks for a repaired plan against the updated
//! working copy. Only the final validated result is written to disk.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Result, bail};
use lsp_types::{Diagnostic, DiagnosticSeverity};
use serde_json::Value;

use super::ToolResult;
use super::permissions::PermissionManager;
use crate::config::{Config, ModelRole};
use crate::llm::{ChatRequest, Message, ModelRouter};
use crate::logging::SessionLog;
use crate::lsp::LspClient;

mod apply;
mod parse;

pub use apply::{apply_literal_replace_in_scope, apply_patch_dry_run};
pub use parse::{EditPlanStep, EditRegion, PatchOp, parse_edit_plan, parse_patch};

use apply::{RelocateOutcome, apply_patch_dry_run_in_region, try_relocate_and_replace};
use parse::{
    MAX_FAILED_REASON_CHARS, format_completed_steps_compact, format_edit_plan_steps,
    format_preplan_log, looks_like_complete, parse_failed, parse_needs_clarification,
    partition_overlapping_steps, truncate_multiline,
};

/// Max lines per window for reliable LLM recall.
const WINDOW_SIZE: usize = 800;
/// Overlap between windows to catch edits at boundaries.
const PREPLAN_READ_OVERLAP: usize = 60;
/// Files up to this many lines skip the windowed observation pass and feed
/// their full content directly into the finalize prompt. Below this size
/// the whole file already fits comfortably in a single slice, so the
/// observation round is pure latency overhead.
const SMALL_FILE_THRESHOLD: usize = 200;
const MAX_PLAN_ATTEMPTS: usize = 4;
/// Additional plan attempts granted when the failing trajectory is
/// strictly improving (new LSP error count < best-so-far). Bounds the
/// "promising-fix-loop" retry credit introduced to stop cutting off
/// slow converging runs at `MAX_PLAN_ATTEMPTS`.
const MAX_EXTRA_ATTEMPTS: usize = 2;
pub(super) const MAX_PREPLAN_STEPS: usize = 100;
pub(super) const MAX_PREPLAN_LOG_CHARS: usize = 20000;
const LARGE_TRUNCATION_MIN_LINES: usize = 50;

pub(super) fn ensure_not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
        bail!("edit_file interrupted by user");
    }
    Ok(())
}

pub(super) fn log_stage(log: Option<&SessionLog>, path_str: &str, stage: &str) {
    if let Some(log) = log {
        log.tool_stage("edit_file", &format!("{path_str} {stage}"));
    }
}

pub(super) fn log_debug(log: Option<&SessionLog>, path_str: &str, detail: &str) {
    if let Some(log) = log {
        log.tool_debug("edit_file", &format!("{path_str} {detail}"));
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn execute(
    args: &Value,
    config: &Config,
    router: &ModelRouter,
    lsp: Option<&LspClient>,
    cancelled: Option<&AtomicBool>,
    log: Option<&SessionLog>,
    baseline_lsp_errors: Option<usize>,
    perms: Option<&PermissionManager>,
) -> Result<ToolResult> {
    let path_str = args["path"].as_str().unwrap_or("");
    let task = args["task"].as_str().unwrap_or("");
    let lsp_validation = match LspValidationMode::from_args(args) {
        Ok(mode) => mode,
        Err(e) => return Ok(ToolResult::err(e.to_string())),
    };

    if path_str.is_empty() {
        return Ok(ToolResult::err("Missing required parameter: path".into()));
    }
    if task.is_empty() {
        return Ok(ToolResult::err("Missing required parameter: task".into()));
    }

    let path = config.project_root.join(path_str);
    if !path.exists() {
        return Ok(ToolResult::err(format!("File not found: {path_str}")));
    }
    ensure_not_cancelled(cancelled)?;

    let original = std::fs::read_to_string(&path)
        .map_err(|e| anyhow::anyhow!("Failed to read {path_str}: {e}"))?;

    match execute_preplanned_steps(
        path_str,
        task,
        &path,
        &original,
        router,
        config,
        lsp,
        lsp_validation,
        cancelled,
        log,
        baseline_lsp_errors,
        perms,
    )
    .await
    {
        Ok(PreplanResult::Applied(content)) => {
            std::fs::write(&path, &content)?;
            Ok(ToolResult::ok(format!("✓ edit_file({path_str}): done")))
        }
        Ok(PreplanResult::NothingToDo) => Ok(ToolResult::ok(format!(
            "✓ edit_file({path_str}): already satisfied"
        ))),
        Ok(PreplanResult::Failed(reason)) => Ok(ToolResult::err(format!(
            "✗ edit_file({path_str}): {reason}"
        ))),
        Ok(PreplanResult::NeedsClarification(question)) => {
            let question = if question.is_empty() {
                "no question provided".to_string()
            } else {
                question
            };
            Ok(ToolResult::err(format!(
                "edit_file needs clarification before it can apply edits.\n\n\
                 Original task: {task}\n\
                 Question: {question}\n\n\
                 Re-run edit_file with a task that addresses the question, \
                 or split the work into more specific steps. The file was not modified."
            )))
        }
        Err(e) => Ok(ToolResult::err(format!("✗ edit_file({path_str}): {}", e))),
    }
}

struct SplitResult {
    content: String,
    message: String,
}

/// A step that was rejected before execution because it overlapped an
/// earlier step. The kept step is applied normally; the dropped one is
/// reported as a per-step failure in the result so the agent sees both
/// the success and the failure side-by-side.
pub(super) struct DroppedStep {
    pub(super) step: EditPlanStep,
    pub(super) reason: String,
}

/// What one planning attempt produced. The model can return a concrete
/// edit plan, or at any phase ask for clarification via
/// `NEEDS_CLARIFICATION: <question>` when the task is too vague or
/// contradicts the file to execute without guessing. Clarification
/// requests short-circuit the entire repair retry loop — the whole point
/// is to stop burning attempts on guesses.
///
/// The pre-plan model's verdict for one iteration against the current
/// file state. Every round, the inner model sees the task + current file
/// + prior-attempt history and must pick one of these outputs.
///
/// `Continue` carries `dropped` so the executor can report overlapping
/// steps as failed steps in the per-step output instead of as opaque
/// warnings.
enum PreplanOutcome {
    /// More edits needed. The model emitted edit-plan blocks that should
    /// be executed against the current file state.
    Continue {
        steps: Vec<EditPlanStep>,
        dropped: Vec<DroppedStep>,
    },
    /// The model says the task is satisfied by the current file state.
    /// Either nothing needed to change (first iteration), or prior
    /// iterations already completed the task. Terminal verdict — the
    /// retry loop stops and reports success.
    Complete,
    /// The model decided the task CANNOT be completed — e.g. it requires
    /// cross-file changes, it contradicts file invariants, or prior
    /// attempts hit an obstacle the model can't plan around. Carries a
    /// short reason from the model itself. Terminal verdict — the retry
    /// loop stops and reports the failure.
    Failed(String),
    /// The pre-plan model decided the task is too ambiguous, under-specified,
    /// or contradictory to act on. Carries the model's question back to the
    /// caller so the outer agent can rephrase or split the task.
    NeedsClarification(String),
    /// The model returned empty or whitespace-only text. Treated as a
    /// transient pathology (stalled inference, template misfire) and
    /// retried with explicit feedback via `RepairContext`.
    EmptyResponse,
    /// The model emitted something that `parse_edit_plan` rejected
    /// (unknown token, missing `OLD:` after we removed the no-OLD
    /// shortcut, malformed SCOPE, etc). Carry the parser's error
    /// verbatim so the repair prompt can tell the model what to fix.
    ParseError(String),
}

/// What the whole pre-plan retry loop produced. The outer caller wires
/// each variant to a one-line agent-facing message, no per-step trail.
enum PreplanResult {
    /// Task is done; file content should be written to disk.
    Applied(String),
    /// Task is already satisfied; file was not modified.
    NothingToDo,
    /// Task could not be completed; carries a short reason (from the
    /// inner model if available, or from the retry loop on exhaustion).
    Failed(String),
    /// Task is too vague to act on; the agent must rephrase.
    NeedsClarification(String),
}

struct PlannedExecutionFailure {
    current_content: String,
    message: String,
    error: String,
    /// Steps from the plan that already applied successfully to
    /// `current_content`. Recorded in execution order (descending source
    /// line). Empty when the very first step blew up.
    completed_steps: Vec<EditPlanStep>,
    /// The step that failed, if any. `None` means the plan executed
    /// fully but then post-validation (e.g. LSP) rejected the result, so
    /// every step in `completed_steps` succeeded individually.
    failed_step: Option<EditPlanStep>,
    /// When the failure is an LSP regression, structured information
    /// about each error location plus the post-edit file snapshot so
    /// the replanner can see *which lines* of its patch produced the
    /// new errors instead of reading a truncated error blob.
    lsp_regression: Option<LspRegression>,
}

/// One diagnostic entry carried inside `LspRegression`. Lines and columns
/// are 1-based, matching the `file:line:col` rendering used throughout
/// the tool output.
#[derive(Clone, Debug)]
struct LspErrorLocation {
    line: usize,
    column: usize,
    message: String,
}

/// Structured LSP regression captured when post-edit validation fails.
/// Unlike the opaque error string, this carries the broken candidate
/// content so the repair prompt can show the planner the *exact lines*
/// its patch produced around every new error.
#[derive(Clone, Debug)]
struct LspRegression {
    baseline_count: usize,
    errors: Vec<LspErrorLocation>,
    /// Extra suffix "... and N more error(s)" — tracked separately so
    /// the rendered block can repeat the truncation note instead of
    /// losing it.
    extra_error_count: usize,
    /// The candidate file content at the moment validation ran. Used
    /// by `format_lsp_regression_for_planner` to extract ±5-line
    /// snippets around each error location.
    candidate_content: String,
}

/// Validation outcome when `validate_candidate_for_write` (or its LSP
/// sub-step) rejects a candidate. Separating `LspRegression` from
/// `Other` lets the retry loop capture structured diagnostic info to
/// feed back into the planner, while still allowing non-LSP failures
/// (truncation gate, IO errors, …) to surface as opaque errors.
enum ValidationError {
    LspRegression(LspRegression),
    Other(anyhow::Error),
}

impl ValidationError {
    fn summary(&self) -> String {
        match self {
            Self::LspRegression(reg) => {
                let mut out = format!(
                    "LSP diagnostics worsened: {} -> {} error(s)",
                    reg.baseline_count,
                    reg.errors.len() + reg.extra_error_count
                );
                for err in reg.errors.iter().take(5) {
                    out.push_str(&format!(
                        "\nL{}:{}: error: {}",
                        err.line, err.column, err.message
                    ));
                }
                if reg.extra_error_count > 0 {
                    out.push_str(&format!(
                        "\n... and {} more error(s)",
                        reg.extra_error_count
                    ));
                }
                out
            }
            Self::Other(e) => e.to_string(),
        }
    }
}

impl From<anyhow::Error> for ValidationError {
    fn from(e: anyhow::Error) -> Self {
        Self::Other(e)
    }
}

impl From<std::io::Error> for ValidationError {
    fn from(e: std::io::Error) -> Self {
        Self::Other(e.into())
    }
}

/// Structured information passed to `request_preplan_steps` describing
/// the most recent iteration's attempt. The planner sees it at the top
/// of every verdict prompt so every round is explicitly aware of what
/// was just tried and what happened.
///
/// Used both for failure context (step blew up, LSP regression, parse
/// error) and for success context (attempt applied cleanly, time to
/// decide if the task is now done). `cleanly_applied` flips the
/// rendering between "failure — plan the recovery" and "success —
/// decide COMPLETE or CONTINUE".
struct RepairContext {
    previous_plan: Vec<EditPlanStep>,
    completed_steps: Vec<EditPlanStep>,
    failed_step: Option<EditPlanStep>,
    failure_reason: String,
    /// When the failure was an LSP regression, carries the structured
    /// diagnostic info so `format_repair_context` can render post-edit
    /// snippets around each error instead of a truncated blob.
    lsp_regression: Option<LspRegression>,
    /// True when the previous iteration applied cleanly (no step failed,
    /// no validation regression). In that case the verdict prompt should
    /// frame the prior attempt as progress toward the task, not as a
    /// failure — the model is being asked to decide whether the task is
    /// now done (COMPLETE) or still needs more work (CONTINUE).
    cleanly_applied: bool,
}

fn max_literal_replace_lines(context_window: usize) -> usize {
    match context_window {
        0..=32_000 => 8,
        32_001..=64_000 => 12,
        64_001..=128_000 => 20,
        128_001..=256_000 => 32,
        _ => 48,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LspValidationMode {
    Auto,
    Require,
    Off,
}

impl LspValidationMode {
    fn from_args(args: &Value) -> Result<Self> {
        match args["lsp_validation"].as_str().unwrap_or("auto") {
            "auto" => Ok(Self::Auto),
            "require" => Ok(Self::Require),
            "off" => Ok(Self::Off),
            other => bail!("Invalid lsp_validation: {other}. Expected one of: auto, require, off"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Require => "require",
            Self::Off => "off",
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn execute_preplanned_steps(
    path_str: &str,
    task: &str,
    path: &std::path::Path,
    original: &str,
    router: &ModelRouter,
    config: &Config,
    lsp: Option<&LspClient>,
    lsp_validation: LspValidationMode,
    cancelled: Option<&AtomicBool>,
    log: Option<&SessionLog>,
    baseline_lsp_errors: Option<usize>,
    perms: Option<&PermissionManager>,
) -> Result<PreplanResult> {
    let max_literal_lines = max_literal_replace_lines(config.model.context_window);
    let mut current = original.to_string();
    let mut repair_context: Option<RepairContext> = None;
    // Track the best (lowest) candidate LSP error count observed across
    // attempts. When a new failing attempt improves on the best-so-far,
    // the loop grants an extra attempt credit on the theory that the
    // planner is converging and deserves another try. Cap extras at
    // `MAX_EXTRA_ATTEMPTS` so a slowly-wobbling trajectory still
    // terminates.
    let mut best_error_count: Option<usize> = None;
    let mut extra_attempts_granted: usize = 0;
    let mut attempt_budget = MAX_PLAN_ATTEMPTS;
    let mut attempt: usize = 0;

    while attempt < attempt_budget {
        attempt += 1;
        if repair_context.is_none() {
            log_stage(log, path_str, "preplan:start");
        }

        let outcome = request_preplan_steps(
            path_str,
            task,
            &current,
            router,
            repair_context.as_ref(),
            max_literal_lines,
            cancelled,
            log,
        )
        .await?;

        let (steps, dropped) = match outcome {
            PreplanOutcome::Continue { steps, dropped } => (steps, dropped),
            PreplanOutcome::Complete => {
                // Terminal verdict from the model: the task is satisfied
                // by the current file state. If we applied edits along
                // the way, the file is dirty and needs writing; if not,
                // it's a clean no-op.
                log_debug(
                    log,
                    path_str,
                    &format!("preplan:verdict=complete attempt={attempt}"),
                );
                if current == original {
                    return Ok(PreplanResult::NothingToDo);
                }
                // Run the normal post-write validation (LSP gate, size
                // guard, etc). If it passes, commit. If LSP regressed,
                // feed the regression back and loop — the model's
                // "complete" verdict was over-optimistic and it gets a
                // chance to repair before we give up.
                let validation = validate_candidate_for_write(
                    path_str,
                    path,
                    original,
                    &current,
                    config,
                    lsp,
                    lsp_validation,
                    cancelled,
                    log,
                    baseline_lsp_errors,
                    perms,
                )
                .await;
                match validation {
                    Ok(_note) => return Ok(PreplanResult::Applied(current)),
                    Err(ValidationError::Other(e)) => return Err(e),
                    Err(ValidationError::LspRegression(regression)) => {
                        let summary = ValidationError::LspRegression(regression.clone()).summary();
                        log_debug(
                            log,
                            path_str,
                            &format!(
                                "preplan:complete_validation_failed:{attempt} {}",
                                truncate_multiline(&summary, 2000)
                            ),
                        );

                        let new_count = regression.errors.len() + regression.extra_error_count;
                        let improved = match best_error_count {
                            Some(best) => new_count < best,
                            None => false,
                        };
                        best_error_count = Some(match best_error_count {
                            Some(best) => best.min(new_count),
                            None => new_count,
                        });
                        if improved && extra_attempts_granted < MAX_EXTRA_ATTEMPTS {
                            attempt_budget += 1;
                            extra_attempts_granted += 1;
                        }

                        repair_context = Some(RepairContext {
                            previous_plan: Vec::new(),
                            completed_steps: Vec::new(),
                            failed_step: None,
                            failure_reason: summary,
                            lsp_regression: Some(regression),
                            cleanly_applied: false,
                        });
                        continue;
                    }
                }
            }
            PreplanOutcome::Failed(reason) => {
                // Terminal verdict from the model: the task cannot be
                // completed in this file. Short-circuit the retry loop
                // and surface the model's own reason to the outer agent.
                log_debug(
                    log,
                    path_str,
                    &format!("preplan:verdict=failed attempt={attempt} reason={reason}"),
                );
                return Ok(PreplanResult::Failed(reason));
            }
            PreplanOutcome::NeedsClarification(question) => {
                // The pre-plan model decided the task is too vague or
                // contradictory to act on. Short-circuit the entire retry
                // loop and surface the question — repairing a guess won't
                // recover a task whose intent we don't know.
                log_debug(
                    log,
                    path_str,
                    &format!("preplan:needs_clarification attempt={attempt} question={question}"),
                );
                return Ok(PreplanResult::NeedsClarification(question));
            }
            PreplanOutcome::EmptyResponse => {
                // Transient pathology (stalled inference, template misfire,
                // empty streaming completion). Feed explicit guidance back
                // via `RepairContext` and let the outer loop re-prompt.
                log_debug(
                    log,
                    path_str,
                    &format!("preplan:empty_response attempt={attempt}; will retry"),
                );
                repair_context = Some(RepairContext {
                    previous_plan: Vec::new(),
                    completed_steps: Vec::new(),
                    failed_step: None,
                    failure_reason: String::from(
                        "The previous attempt returned an empty response, which is not a valid output. \
                         Emit one of: `LITERAL_REPLACE`/`SMART_EDIT` blocks, `COMPLETE`, `FAILED: <reason>`, or `NEEDS_CLARIFICATION: <question>`. \
                         Do not return an empty response.",
                    ),
                    lsp_regression: None,
                    cleanly_applied: false,
                });
                continue;
            }
            PreplanOutcome::ParseError(reason) => {
                // The model emitted something the plan parser couldn't
                // accept. Feed the parser's exact error back through the
                // repair prompt so the next attempt can correct it.
                log_debug(
                    log,
                    path_str,
                    &format!(
                        "preplan:parse_error attempt={attempt}; will retry reason={}",
                        truncate_multiline(&reason, 500)
                    ),
                );
                repair_context = Some(RepairContext {
                    previous_plan: Vec::new(),
                    completed_steps: Vec::new(),
                    failed_step: None,
                    failure_reason: format!(
                        "The previous attempt emitted an edit plan that failed to parse:\n{reason}\n\n\
                         Re-emit a valid plan. Every LITERAL_REPLACE block must include an OLD: section \
                         whose contents are copied verbatim from the file view above. If you cannot echo \
                         OLD verbatim, use SMART_EDIT for that region instead."
                    ),
                    lsp_regression: None,
                    cleanly_applied: false,
                });
                continue;
            }
        };

        // `Continue` with an empty plan is a degenerate case — the
        // model said "more edits" but listed none. Treat it as an empty
        // response and retry.
        if steps.is_empty() && dropped.is_empty() {
            log_debug(
                log,
                path_str,
                &format!("preplan:continue_empty_plan attempt={attempt}; will retry"),
            );
            repair_context = Some(RepairContext {
                previous_plan: Vec::new(),
                completed_steps: Vec::new(),
                failed_step: None,
                failure_reason: String::from(
                    "The previous attempt used the plan syntax but produced zero steps. \
                     Either emit concrete LITERAL_REPLACE/SMART_EDIT blocks, or emit exactly `COMPLETE` on its own line if the task is already satisfied.",
                ),
                lsp_regression: None,
                cleanly_applied: false,
            });
            continue;
        }

        let kept_count = steps.len();
        let dropped_count = dropped.len();
        let total_planned = kept_count + dropped_count;
        log_debug(
            log,
            path_str,
            &format_preplan_log(&format!("Pre-plan attempt {attempt}"), &steps),
        );
        let attempt_message = String::new();

        match execute_planned_steps(
            path_str,
            path,
            original,
            &current,
            router,
            config,
            lsp,
            lsp_validation,
            cancelled,
            log,
            steps.clone(),
            dropped,
            total_planned,
            attempt_message,
            "via pre-plan",
            baseline_lsp_errors,
            perms,
        )
        .await
        {
            Ok(result) => {
                // The steps executed cleanly. Do NOT declare success
                // here — the task-aware verdict is still the model's
                // call. Loop back with a cleanly-applied repair context
                // so the next iteration can emit COMPLETE (or FAILED,
                // or CONTINUE if more work is still needed).
                current = result.content;
                log_debug(
                    log,
                    path_str,
                    &format!(
                        "preplan:apply_ok:{attempt} kept={kept_count} dropped={dropped_count}"
                    ),
                );
                repair_context = Some(RepairContext {
                    previous_plan: steps.clone(),
                    completed_steps: steps,
                    failed_step: None,
                    failure_reason: String::new(),
                    lsp_regression: None,
                    cleanly_applied: true,
                });
            }
            Err(e) => {
                log_debug(
                    log,
                    path_str,
                    &format!(
                        "preplan:apply_failed:{attempt} {}",
                        truncate_multiline(&e.error, 2000)
                    ),
                );

                // Promising-fix-loop retry credit. When the failure is
                // an LSP regression and the new error count is strictly
                // better than our best-so-far, extend the budget so we
                // don't cut off a converging trajectory.
                if let Some(reg) = &e.lsp_regression {
                    let new_count = reg.errors.len() + reg.extra_error_count;
                    let improved = match best_error_count {
                        Some(best) => new_count < best,
                        None => false,
                    };
                    best_error_count = Some(match best_error_count {
                        Some(best) => best.min(new_count),
                        None => new_count,
                    });
                    if improved && extra_attempts_granted < MAX_EXTRA_ATTEMPTS {
                        attempt_budget += 1;
                        extra_attempts_granted += 1;
                    }
                }

                repair_context = Some(RepairContext {
                    previous_plan: steps,
                    completed_steps: e.completed_steps,
                    failed_step: e.failed_step,
                    failure_reason: e.error,
                    lsp_regression: e.lsp_regression,
                    cleanly_applied: false,
                });
                current = e.current_content;
            }
        }
    }

    // Budget exhausted without the model ever emitting a terminal
    // verdict. Surface whatever the last obstacle was so the outer
    // agent can decide how to proceed.
    let last_reason = repair_context
        .map(|ctx| truncate_multiline(&ctx.failure_reason, MAX_FAILED_REASON_CHARS))
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "planner did not converge".to_string());
    Ok(PreplanResult::Failed(format!(
        "could not complete after {attempt_budget} attempts. Last obstacle: {last_reason}"
    )))
}

#[allow(clippy::too_many_arguments)]
async fn execute_planned_steps(
    path_str: &str,
    path: &std::path::Path,
    file_original: &str,
    current_base: &str,
    router: &ModelRouter,
    config: &Config,
    lsp: Option<&LspClient>,
    lsp_validation: LspValidationMode,
    cancelled: Option<&AtomicBool>,
    log: Option<&SessionLog>,
    steps: Vec<EditPlanStep>,
    dropped: Vec<DroppedStep>,
    planned_count: usize,
    mut message: String,
    success_label: &str,
    baseline_lsp_errors: Option<usize>,
    perms: Option<&PermissionManager>,
) -> std::result::Result<SplitResult, PlannedExecutionFailure> {
    let mut current = current_base.to_string();
    let mut completed_count = 0usize;
    let dropped_count = dropped.len();
    // Records of steps that have been successfully applied to `current`,
    // captured in execution order so the repair planner can see exactly
    // what shifted and what's left.
    let mut completed_records: Vec<EditPlanStep> = Vec::new();
    // Only pad with a newline if the caller already supplied prelude
    // text that didn't end with one. With the current callers passing
    // empty, this stays empty so the final summary lands on line 1.
    if !message.is_empty() && !message.ends_with('\n') {
        message.push('\n');
    }

    let mut steps_desc = steps;
    steps_desc.sort_by(|a, b| b.start_line().cmp(&a.start_line()));

    for (idx, step) in steps_desc.iter().enumerate() {
        match step {
            EditPlanStep::LiteralReplace {
                scope_start,
                scope_end,
                all,
                old,
                new,
            } => {
                match apply_literal_replace_in_scope(
                    &current,
                    *scope_start,
                    *scope_end,
                    old,
                    new,
                    *all,
                ) {
                    Ok((candidate, _count)) => {
                        current = candidate;
                        completed_count += 1;
                        completed_records.push(step.clone());
                        // Successful literal replace is the boring happy
                        // path; do not chatter about it. The final summary
                        // line at the bottom captures the totals.
                    }
                    Err(literal_error) => {
                        // Try to rescue a misplaced OLD block by searching
                        // the whole file for a candidate (byte-exact first,
                        // then whitespace-tolerant), picking the best match
                        // with a locality bias toward the declared scope,
                        // and asking the planner to confirm the corrected
                        // line range via a single YES/NO round-trip. If the
                        // rescue cannot find a candidate or the planner
                        // rejects it, bubble straight up to plan-level
                        // repair — we no longer burn a smart-edit call on
                        // a LITERAL_REPLACE the planner got wrong.
                        let outcome = try_relocate_and_replace(
                            path_str,
                            &current,
                            *scope_start,
                            *scope_end,
                            old,
                            new,
                            *all,
                            router,
                            cancelled,
                            log,
                        )
                        .await;

                        match outcome {
                            RelocateOutcome::Applied {
                                new_content,
                                located_at: (new_start, new_end),
                            } => {
                                current = new_content;
                                completed_count += 1;
                                completed_records.push(step.clone());
                                message.push_str(&format!(
                                    "literal-replace L{scope_start}-L{scope_end} relocated to L{new_start}-L{new_end} after planner confirmation\n"
                                ));
                            }
                            RelocateOutcome::Rejected => {
                                return Err(PlannedExecutionFailure {
                                    current_content: current.clone(),
                                    message: format!(
                                        "{message}Pre-plan step {} literal L{}-L{} failed after {} completed step(s): {literal_error}; relocated candidate rejected by planner\n",
                                        idx + 1,
                                        scope_start,
                                        scope_end,
                                        completed_count
                                    ),
                                    error: format!(
                                        "step {} literal replace failed: {literal_error}; relocated candidate rejected by planner",
                                        idx + 1
                                    ),
                                    completed_steps: completed_records.clone(),
                                    failed_step: Some(step.clone()),
                                    lsp_regression: None,
                                });
                            }
                            RelocateOutcome::NoCandidate => {
                                return Err(PlannedExecutionFailure {
                                    current_content: current.clone(),
                                    message: format!(
                                        "{message}Pre-plan step {} literal L{}-L{} failed after {} completed step(s): {literal_error}; no relocation candidate found in file\n",
                                        idx + 1,
                                        scope_start,
                                        scope_end,
                                        completed_count
                                    ),
                                    error: format!(
                                        "step {} literal replace failed: {literal_error}; no relocation candidate found in file",
                                        idx + 1
                                    ),
                                    completed_steps: completed_records.clone(),
                                    failed_step: Some(step.clone()),
                                    lsp_regression: None,
                                });
                            }
                        }
                    }
                }
            }
            EditPlanStep::SmartEdit(region) => {
                let region_label =
                    format!("step {} smart L{}-L{}", idx + 1, region.start, region.end);
                let (candidate, count) = execute_smart_step(
                    path_str,
                    &region.task,
                    &current,
                    router,
                    lsp_validation,
                    region,
                    true,
                    cancelled,
                    log,
                )
                .await
                .map_err(|e| PlannedExecutionFailure {
                    current_content: current.clone(),
                    message: format!(
                        "{message}Pre-plan {region_label} failed after {completed_count} completed step(s): {e}\n",
                    ),
                    error: format!("{region_label} failed: {e}"),
                    completed_steps: completed_records.clone(),
                    failed_step: Some(step.clone()),
                    lsp_regression: None,
                })?;

                if count == 0 {
                    // count == 0 IS interesting — the model returned no
                    // changes for a region we expected to edit. Surface
                    // it so the agent knows the step ran but did nothing.
                    message.push_str(&format!(
                        "smart-edit L{}-L{}: no changes\n",
                        region.start, region.end
                    ));
                    completed_count += 1;
                    completed_records.push(step.clone());
                } else {
                    current = candidate;
                    completed_count += 1;
                    completed_records.push(step.clone());
                    // Successful smart edit is the boring happy path.
                }
            }
        }
    }

    // Surface overlap-rejected steps so the agent sees them alongside
    // the successes. The kept steps are already applied; the dropped
    // steps appear here purely as feedback.
    for d in &dropped {
        message.push_str(&format!(
            "dropped step L{}-L{} (overlap): {}\n",
            d.step.start_line(),
            d.step.end_line(),
            d.reason
        ));
    }

    let validation_note = validate_candidate_for_write(
        path_str,
        path,
        file_original,
        &current,
        config,
        lsp,
        lsp_validation,
        cancelled,
        log,
        baseline_lsp_errors,
        perms,
    )
    .await
    .map_err(|e| {
        let error_summary = e.summary();
        let lsp_regression = match e {
            ValidationError::LspRegression(reg) => Some(reg),
            ValidationError::Other(_) => None,
        };
        PlannedExecutionFailure {
            current_content: current.clone(),
            message: format!(
                "{message}Pre-plan validation failed after {completed_count}/{planned_count} completed step(s): {error_summary}\n"
            ),
            error: error_summary,
            completed_steps: completed_records.clone(),
            failed_step: None,
            lsp_regression,
        }
    })?;
    if let Some(note) = validation_note {
        message.push_str(&note);
        message.push('\n');
    }
    // Final summary is intentionally one line. Counts only show
    // partial-completion if some steps were dropped or didn't apply
    // cleanly; otherwise we just say "applied N step(s)".
    let summary = if completed_count == planned_count && dropped_count == 0 {
        format!(
            "✓ {success_label}: applied {completed_count} step(s) to {path_str} ({} lines)\n",
            current.lines().count()
        )
    } else {
        format!(
            "✓ {success_label}: applied {completed_count}/{planned_count} step(s) to {path_str} ({} lines)\n",
            current.lines().count()
        )
    };
    message = format!("{summary}{message}");

    Ok(SplitResult {
        content: current,
        message,
    })
}

async fn execute_smart_step(
    path_str: &str,
    task: &str,
    current: &str,
    router: &ModelRouter,
    lsp_validation: LspValidationMode,
    region: &EditRegion,
    allow_no_changes: bool,
    cancelled: Option<&AtomicBool>,
    log: Option<&SessionLog>,
) -> Result<(String, usize)> {
    // Single-shot: any failure here bubbles up to the plan-level retry
    // loop (`MAX_PLAN_ATTEMPTS` in `execute_preplanned_steps`), which
    // re-prompts the planner with full repair context. Bench logs across
    // many sessions show the inner retry never recovered a region — when
    // a smart-edit attempt failed, retrying with the same prompt + a
    // generic feedback string just produced the same failure. Letting
    // the planner re-plan is strictly better.
    let (ops, _) = request_patch_for_region(
        path_str,
        task,
        current,
        router,
        region,
        lsp_validation,
        cancelled,
        log,
    )
    .await?;

    if ops.is_empty() {
        if allow_no_changes {
            return Ok((current.to_string(), 0));
        }
        bail!("smart edit returned NO_CHANGES");
    }

    let candidate = apply_patch_dry_run_in_region(current, &ops, region.start, region.end)?;
    validate_candidate(current, &candidate)?;
    Ok((candidate, ops.len()))
}

#[allow(clippy::too_many_arguments)]
async fn validate_candidate_for_write(
    path_str: &str,
    path: &std::path::Path,
    original: &str,
    candidate: &str,
    config: &Config,
    lsp: Option<&LspClient>,
    lsp_validation: LspValidationMode,
    cancelled: Option<&AtomicBool>,
    log: Option<&SessionLog>,
    baseline_lsp_errors: Option<usize>,
    perms: Option<&PermissionManager>,
) -> std::result::Result<Option<String>, ValidationError> {
    ensure_not_cancelled(cancelled).map_err(ValidationError::Other)?;
    validate_candidate(original, candidate).map_err(ValidationError::Other)?;
    gate_truncation(path_str, original, candidate, perms).map_err(ValidationError::Other)?;
    log_stage(log, path_str, "validate:lsp");
    validate_candidate_with_lsp(
        path_str,
        path,
        original,
        candidate,
        config,
        lsp,
        lsp_validation,
        baseline_lsp_errors,
    )
    .await
}

async fn validate_candidate_with_lsp(
    path_str: &str,
    path: &std::path::Path,
    original: &str,
    candidate: &str,
    config: &Config,
    lsp: Option<&LspClient>,
    lsp_validation: LspValidationMode,
    baseline_lsp_errors: Option<usize>,
) -> std::result::Result<Option<String>, ValidationError> {
    if lsp_validation == LspValidationMode::Off {
        return Ok(Some("[lsp] skipped (off)".into()));
    }

    let Some(lsp) = lsp else {
        if lsp_validation == LspValidationMode::Require {
            return Err(ValidationError::Other(anyhow::anyhow!(
                "LSP validation required but no LSP client is available"
            )));
        }
        return Ok(None);
    };

    if !lsp.is_ready() || lsp.has_crashed() {
        if lsp_validation == LspValidationMode::Require {
            return Err(ValidationError::Other(anyhow::anyhow!(
                "LSP validation required but LSP is not ready"
            )));
        }
        return Ok(None);
    }

    let timeout = Duration::from_millis(config.lsp.diagnostic_timeout_ms);

    // Prefer the baseline captured by the outer tool dispatcher
    // (`capture_edit_baseline` in tools::mod) when it's available. That
    // baseline is taken once *before* this edit_file call begins, so it
    // stays consistent across pre-plan retries and matches what the outer
    // `auto_check` will compare against. Falling back to a local query is
    // only for legacy / direct callers that don't supply one.
    let baseline_count = match baseline_lsp_errors {
        Some(n) => n,
        None => match diagnostics_for_current_file(lsp, path, timeout).await {
            Ok(diags) => error_diagnostics(&diags).len(),
            Err(e) => {
                if lsp_validation == LspValidationMode::Require {
                    return Err(ValidationError::Other(anyhow::anyhow!(
                        "LSP baseline diagnostics failed: {e}"
                    )));
                }
                return Ok(None);
            }
        },
    };

    std::fs::write(path, candidate)?;

    let candidate_diags = match diagnostics_for_current_file(lsp, path, timeout).await {
        Ok(diags) => diags,
        Err(e) => {
            let _ = std::fs::write(path, original);
            let _ = diagnostics_for_current_file(lsp, path, timeout).await;
            if lsp_validation == LspValidationMode::Require {
                return Err(ValidationError::Other(anyhow::anyhow!(
                    "LSP candidate diagnostics failed: {e}"
                )));
            }
            return Ok(None);
        }
    };

    let candidate_errors = error_diagnostics(&candidate_diags);
    if candidate_errors.len() > baseline_count {
        let regression = build_lsp_regression(baseline_count, &candidate_errors, candidate);
        let _ = std::fs::write(path, original);
        let _ = diagnostics_for_current_file(lsp, path, timeout).await;
        let _ = path_str; // path_str retained for callers formatting the one-line summary
        return Err(ValidationError::LspRegression(regression));
    }

    Ok(Some(format!(
        "[lsp] OK ({} -> {} error(s), mode={})",
        baseline_count,
        candidate_errors.len(),
        lsp_validation.as_str()
    )))
}

/// Build the structured `LspRegression` carried back to the planner.
/// We keep the first 5 errors inline (matching the one-line summary
/// cap) and record the remaining count so the summary still says
/// "and N more".
fn build_lsp_regression(
    baseline_count: usize,
    candidate_errors: &[Diagnostic],
    candidate_content: &str,
) -> LspRegression {
    const MAX_ERRORS: usize = 5;
    let total = candidate_errors.len();
    let kept: Vec<LspErrorLocation> = candidate_errors
        .iter()
        .take(MAX_ERRORS)
        .map(|d| LspErrorLocation {
            line: (d.range.start.line + 1) as usize,
            column: (d.range.start.character + 1) as usize,
            message: d.message.clone(),
        })
        .collect();
    let extra_error_count = total.saturating_sub(kept.len());
    LspRegression {
        baseline_count,
        errors: kept,
        extra_error_count,
        candidate_content: candidate_content.to_string(),
    }
}

async fn diagnostics_for_current_file(
    lsp: &LspClient,
    path: &std::path::Path,
    timeout: Duration,
) -> Result<Vec<Diagnostic>> {
    lsp.notify_file_changed(path)?;
    Ok(lsp.get_diagnostics(path, timeout).await)
}

fn error_diagnostics(diagnostics: &[Diagnostic]) -> Vec<Diagnostic> {
    diagnostics
        .iter()
        .filter(|d| d.severity == Some(DiagnosticSeverity::ERROR))
        .cloned()
        .collect()
}

struct PatchResponse {
    ops: Vec<PatchOp>,
    output: String,
    raw_text: String,
}

const MAX_PREPLAN_SEARCHES: usize = 20;
const MAX_PREPLAN_READS_INITIAL: usize = 6;
const MAX_PREPLAN_READS_REPAIR: usize = 10;

fn build_retry_feedback(last_error: &str, _signature_grounding: Option<&str>) -> String {
    let mut feedback = last_error.to_string();
    if is_signature_mismatch_error(last_error) {
        feedback.push_str(
            "\nDo not repeat the same patch shape on the same lines. Re-check the current code before retrying and narrow the edit to only the lines that clearly need to change.",
        );
    }
    feedback
}

fn is_signature_mismatch_error(error: &str) -> bool {
    error.contains("expected ")
        && (error.contains("arguments, found") || error.contains("mismatched types"))
}

#[derive(Debug)]
struct PreplanWindowResponse {
    notes: Vec<String>,
    commands: Vec<InspectionCommand>,
    /// Set to `Some(question)` if any line in the response is a
    /// `NEEDS_CLARIFICATION:` sentinel. Short-circuits the windowed pass:
    /// the model already knows the task is ambiguous and there's no point
    /// scanning more slices.
    clarification: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum InspectionCommand {
    Search(String),
    Read { start: usize, end: usize },
}

/// Parse the response from a single windowed pre-plan turn. Each window
/// emits NOTE lines for structural landmarks and optional SEARCH/READ
/// commands requesting extra context before finalize. Commands are
/// *collected* across all windows and batch-executed once the full pass
/// is complete; the model doesn't see results until the finalize prompt.
///
/// If the model decides the task itself is too vague or contradictory to
/// act on, it can emit `NEEDS_CLARIFICATION: <question>` on any line. The
/// first such line short-circuits the windowed pass via the `clarification`
/// field so the caller can surface the question immediately instead of
/// burning the rest of the scan on a task with unknown intent.
///
/// The parser is liberal: anything that isn't a recognizable NOTE,
/// SEARCH/READ, or NEEDS_CLARIFICATION line is silently ignored. An empty
/// response is fine — the model may have nothing useful to say about a
/// particular slice.
fn parse_preplan_window_response(text: &str) -> PreplanWindowResponse {
    let mut notes = Vec::new();
    let mut commands = Vec::new();
    let mut clarification: Option<String> = None;

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        if clarification.is_none() {
            if let Some(question) = parse_needs_clarification(line) {
                clarification = Some(question);
                continue;
            }
        }
        if let Some(rest) = strip_case_insensitive_prefix(line, "NOTE ") {
            let note = rest.trim();
            if !note.is_empty() {
                notes.push(note.to_string());
            }
            continue;
        }
        if let Some(rest) = strip_case_insensitive_prefix(line, "NOTE:") {
            let note = rest.trim();
            if !note.is_empty() {
                notes.push(note.to_string());
            }
            continue;
        }
        if let Some(rest) = strip_case_insensitive_prefix(line, "SEARCH:") {
            let query = rest.trim();
            if !query.is_empty() {
                commands.push(InspectionCommand::Search(query.to_string()));
            }
            continue;
        }
        if let Some(rest) = strip_case_insensitive_prefix(line, "READ:") {
            let rest = rest.trim();
            if let Some((start_s, end_s)) = rest.split_once('-') {
                if let (Ok(start), Ok(end)) = (
                    start_s.trim().parse::<usize>(),
                    end_s.trim().parse::<usize>(),
                ) {
                    if start > 0 && end >= start {
                        commands.push(InspectionCommand::Read { start, end });
                    }
                }
            }
            continue;
        }
        // Anything else is silently ignored. Stray control words,
        // half-formed steps, or leftover DONE terminators all get dropped
        // without erroring out.
    }

    PreplanWindowResponse {
        notes,
        commands,
        clarification,
    }
}

/// Extract `(start, end)` line ranges from a note string.
///
/// Recognizes patterns like `L283-290`, `283-290`, `L41`, `line 137`.
/// Single-line references are expanded to a 5-line window around the
/// noted line so the finalize phase sees enough surrounding context.
fn extract_line_ranges_from_note(note: &str) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    // Work on chars to avoid multibyte boundary issues.
    let chars: Vec<char> = note.chars().collect();
    let len = chars.len();
    let mut i = 0;
    while i < len {
        let prefix_start = i;
        // Optional 'L'/'l' prefix or "line "/"Line " prefix
        if chars[i] == 'L' || chars[i] == 'l' {
            if i + 4 < len
                && (chars[i + 1] == 'i' || chars[i + 1] == 'I')
                && (chars[i + 2] == 'n' || chars[i + 2] == 'N')
                && (chars[i + 3] == 'e' || chars[i + 3] == 'E')
                && chars[i + 4] == ' '
            {
                i += 5; // "line "
            } else {
                i += 1; // bare "L"
            }
        }
        // Must be at a digit now
        if i >= len || !chars[i].is_ascii_digit() {
            i = prefix_start + 1;
            continue;
        }
        // Reject mid-word: char before prefix must not be alphanumeric
        if prefix_start > 0
            && (chars[prefix_start - 1].is_alphanumeric() || chars[prefix_start - 1] == '_')
        {
            i = prefix_start + 1;
            continue;
        }
        // Parse start number
        let num_start = i;
        while i < len && chars[i].is_ascii_digit() {
            i += 1;
        }
        let start_str: String = chars[num_start..i].iter().collect();
        let start: usize = match start_str.parse() {
            Ok(n) if n > 0 => n,
            _ => continue,
        };
        // Check for range separator (optional spaces around '-')
        let saved = i;
        if i < len && chars[i] == ' ' {
            i += 1;
        }
        if i < len && chars[i] == '-' {
            i += 1;
            if i < len && chars[i] == ' ' {
                i += 1;
            }
            let end_start = i;
            while i < len && chars[i].is_ascii_digit() {
                i += 1;
            }
            if i > end_start {
                let end_str: String = chars[end_start..i].iter().collect();
                if let Ok(end) = end_str.parse::<usize>() {
                    if end >= start {
                        ranges.push((start, end));
                        continue;
                    }
                }
            }
            // Dash but no valid end number — fall through to single-line
            i = saved;
        }
        // Single line reference — expand to ±2 line window
        ranges.push((start.saturating_sub(2).max(1), start + 2));
    }
    ranges
}

fn has_case_insensitive_prefix(text: &str, prefix: &str) -> bool {
    text.get(..prefix.len())
        .map(|head| head.eq_ignore_ascii_case(prefix))
        .unwrap_or(false)
}

fn strip_case_insensitive_prefix<'a>(text: &'a str, prefix: &str) -> Option<&'a str> {
    has_case_insensitive_prefix(text, prefix).then(|| &text[prefix.len()..])
}

fn search_in_file(content: &str, query: &str) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let matches: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter_map(|(idx, line)| line.contains(query).then_some(idx + 1))
        .collect();
    if matches.is_empty() {
        return format!("SEARCH RESULT for `{query}`: 0 hits");
    }

    let mut out = format!("SEARCH RESULT for `{query}`: {} hit(s)\n", matches.len());
    for line_no in matches.iter().take(8) {
        out.push_str(&format!("{:>4}│{}\n", line_no, lines[*line_no - 1]));
    }
    if matches.len() > 8 {
        out.push_str(&format!("... {} more hit(s)\n", matches.len() - 8));
    }
    out.trim_end().to_string()
}

fn read_in_file(content: &str, start: usize, end: usize) -> Result<String> {
    let lines: Vec<&str> = content.lines().collect();
    if end > lines.len() {
        bail!(
            "READ range L{start}-L{end} outside file with {} lines",
            lines.len()
        );
    }
    let mut out = format!("READ RESULT L{start}-L{end}:\n");
    for line_no in start..=end {
        out.push_str(&format!("{:>4}│{}\n", line_no, lines[line_no - 1]));
    }
    Ok(out.trim_end().to_string())
}

fn render_numbered_slice(lines: &[&str], start: usize, end: usize) -> String {
    lines[start..end.min(lines.len())]
        .iter()
        .enumerate()
        .map(|(offset, line)| format!("{:>4}│{}", start + offset + 1, line))
        .collect::<Vec<_>>()
        .join("\n")
}

fn extend_unique_notes(existing: &mut Vec<String>, new_notes: Vec<String>) {
    for note in new_notes {
        if !existing.contains(&note) {
            existing.push(note);
        }
    }
}

fn append_inspection_result(extra_context: &mut String, label: &str, result: &str) {
    extra_context.push_str(label);
    extra_context.push('\n');
    extra_context.push_str(result);
    extra_context.push_str("\n\n");
}

/// Per-edit totals for SEARCH and READ so that the batch inspection pass
/// can enforce the same caps whether the model requested one command or
/// many across several windows.
struct InspectionCounters {
    search_count: usize,
    read_count: usize,
    max_reads: usize,
}

/// Execute the SEARCH/READ commands collected across the windowed
/// observation pass, appending their formatted results to `extra_context`
/// for the finalize prompt. Commands that exceed the per-edit caps are
/// dropped with an inline note rather than erroring out.
fn execute_inspection_commands(
    content: &str,
    commands: &[InspectionCommand],
    counters: &mut InspectionCounters,
    extra_context: &mut String,
    path_str: &str,
    log: Option<&SessionLog>,
) -> Result<()> {
    for command in commands {
        match command {
            InspectionCommand::Search(query) => {
                if counters.search_count >= MAX_PREPLAN_SEARCHES {
                    extra_context.push_str(&format!(
                        "(SEARCH `{query}` skipped: per-edit limit of {MAX_PREPLAN_SEARCHES} searches reached)\n\n",
                    ));
                    continue;
                }
                counters.search_count += 1;
                let result = search_in_file(content, query);
                log_debug(
                    log,
                    path_str,
                    &format!(
                        "preplan:search:{} {}",
                        counters.search_count,
                        truncate_multiline(&result, 4000)
                    ),
                );
                append_inspection_result(
                    extra_context,
                    &format!("SEARCH_RESULT query=`{query}`"),
                    &result,
                );
            }
            InspectionCommand::Read { start, end } => {
                if counters.read_count >= counters.max_reads {
                    extra_context.push_str(&format!(
                        "(READ {start}-{end} skipped: per-edit limit of {} reads reached)\n\n",
                        counters.max_reads,
                    ));
                    continue;
                }
                counters.read_count += 1;
                let result = match read_in_file(content, *start, *end) {
                    Ok(r) => r,
                    Err(e) => {
                        extra_context.push_str(&format!("(READ {start}-{end} failed: {e})\n\n",));
                        continue;
                    }
                };
                log_debug(
                    log,
                    path_str,
                    &format!(
                        "preplan:read:{} {}",
                        counters.read_count,
                        truncate_multiline(&result, 4000)
                    ),
                );
                append_inspection_result(
                    extra_context,
                    &format!("READ_RESULT range=L{start}-L{end}"),
                    &result,
                );
            }
        }
    }
    Ok(())
}

async fn request_patch(
    path_str: &str,
    task: &str,
    content: &str,
    router: &ModelRouter,
    repair_feedback: Option<&str>,
    signature_grounding: Option<&str>,
    lsp_validation: LspValidationMode,
    cancelled: Option<&AtomicBool>,
    log: Option<&SessionLog>,
) -> Result<PatchResponse> {
    let lines: Vec<&str> = content.lines().collect();
    let total_lines = lines.len();
    let windows = build_windows(total_lines, WINDOW_SIZE, 0);
    let mut all_ops = Vec::new();
    let mut output = String::new();
    let mut raw_text_parts = Vec::new();

    for (win_idx, (start, end)) in windows.iter().enumerate() {
        ensure_not_cancelled(cancelled)?;
        let window_content = lines[*start..*end]
            .iter()
            .enumerate()
            .map(|(i, l)| format!("{:>4}│{}", start + i + 1, l))
            .collect::<Vec<_>>()
            .join("\n");

        let window_info = if windows.len() > 1 {
            format!(
                "(window {}/{}, lines {}-{} of {})",
                win_idx + 1,
                windows.len(),
                start + 1,
                end,
                total_lines
            )
        } else {
            format!("({total_lines} lines)")
        };

        let repair = repair_feedback
            .map(|f| {
                format!(
                    "\nPrevious patch was not applied.\nFailure: {f}\nReturn a corrected patch against the original file. If the failure mentions overlapping spans, use the smallest enclosing REPLACE_AT block that covers the overlap, or split the patch into separate non-overlapping regions. Do not rewrite a much larger block just to avoid overlap.\n"
                )
            })
            .unwrap_or_default();
        let signature_block = signature_grounding
            .map(|note| format!("\n{note}\n"))
            .unwrap_or_default();
        let prompt = format!(
            "You are editing one file: {path_str} {window_info}.\n\
             Task: {task}\n\
             {signature_block}\
             {repair}\n\
             Return a complete patch for all changes needed in this file/window.\n\
             LSP validation mode: {}. Your patch may be rejected if file diagnostics get worse.\n\
             Use this patch DSL exactly:\n\n\
             INSERT_BEFORE <line>\n\
             CONTENT:\n\
             <lines to insert>\n\
             END\n\n\
             INSERT_AFTER <line>\n\
             CONTENT:\n\
             <lines to insert>\n\
             END\n\n\
             REPLACE_AT <start_line>\n\
             OLD:\n\
             <exact original lines>\n\
             END_OLD\n\
             NEW:\n\
             <replacement lines>\n\
             END_NEW\n\n\
             DELETE_AT <start_line>\n\
             OLD:\n\
             <exact original lines>\n\
             END_OLD\n\n\
             Rules:\n\
             - Output ONLY patch DSL blocks, no markdown or explanations.\n\
             - If no changes are needed, output exactly NO_CHANGES.\n\
             - Line numbers refer to the original file shown below, before any operations apply.\n\
             - For REPLACE_AT/DELETE_AT, OLD determines how many lines are changed.\n\
             - Prefer small, non-overlapping operations. Do not output overlapping REPLACE_AT/DELETE_AT operations.\n\
             - If two edits overlap, use the smallest enclosing REPLACE_AT block that covers the overlap; do not rewrite a much larger block.\n\
             - Preserve indentation and blank lines exactly inside CONTENT/OLD/NEW.\n\
             - Validation is atomic: if any operation fails, no changes are applied.\n\n\
             File content:\n{window_content}",
            lsp_validation.as_str()
        );

        let request = ChatRequest {
            messages: vec![
                Message::system(
                    "You output only strict patch DSL blocks. No explanations, no markdown.",
                ),
                Message::user(&prompt),
            ],
            tools: None,
            tool_choice: None,
        };

        log_stage(
            log,
            path_str,
            &format!("patch:window:{}-{}", start + 1, end),
        );
        let response = router
            .chat_with_cancel(ModelRole::Fast, &request, cancelled)
            .await?;
        let text = response
            .choices
            .first()
            .and_then(|c| c.message.content.as_deref())
            .unwrap_or("");
        raw_text_parts.push(text.trim().to_string());
        log_debug(
            log,
            path_str,
            &format!(
                "patch:window:{}-{} raw_response:\n{}",
                start + 1,
                end,
                truncate_multiline(text, 12000)
            ),
        );

        let ops = match parse_patch(text) {
            Ok(ops) => ops,
            Err(e) => {
                log_debug(
                    log,
                    path_str,
                    &format!(
                        "patch:window:{}-{} parse_failed {}",
                        start + 1,
                        end,
                        truncate_multiline(&e.to_string(), 2000)
                    ),
                );
                return Err(e);
            }
        };
        if !ops.is_empty() {
            output.push_str(&format!(
                "Window {}: {} operation(s) found\n",
                win_idx + 1,
                ops.len()
            ));
        }
        all_ops.extend(ops);
    }

    Ok(PatchResponse {
        ops: all_ops,
        output,
        raw_text: raw_text_parts.join("\n---\n"),
    })
}

async fn request_patch_for_region(
    path_str: &str,
    task: &str,
    content: &str,
    router: &ModelRouter,
    region: &EditRegion,
    lsp_validation: LspValidationMode,
    cancelled: Option<&AtomicBool>,
    log: Option<&SessionLog>,
) -> Result<(Vec<PatchOp>, String)> {
    ensure_not_cancelled(cancelled)?;
    let lines: Vec<&str> = content.lines().collect();
    if region.start == 0 || region.end < region.start || region.end > lines.len() {
        bail!(
            "invalid edit region L{}-L{} for {} line file",
            region.start,
            region.end,
            lines.len()
        );
    }

    let region_content = lines[region.start - 1..region.end]
        .iter()
        .enumerate()
        .map(|(i, line)| format!("{:>4}│{}", region.start + i, line))
        .collect::<Vec<_>>()
        .join("\n");

    let prompt = format!(
        "You are editing one line region in {path_str}: lines {}-{}.\n\
         Task: {task}\n\
         You may edit ONLY lines {}-{}. Do not target lines outside this region.\n\
         LSP validation mode: {}. Your patch may be rejected if file diagnostics get worse.\n\
         Return a complete patch for this region using the patch DSL exactly:\n\n\
         INSERT_BEFORE <line>\n\
         CONTENT:\n\
         <lines to insert>\n\
         END\n\n\
         INSERT_AFTER <line>\n\
         CONTENT:\n\
         <lines to insert>\n\
         END\n\n\
         REPLACE_AT <start_line>\n\
         OLD:\n\
         <exact original lines>\n\
         END_OLD\n\
         NEW:\n\
         <replacement lines>\n\
         END_NEW\n\n\
         DELETE_AT <start_line>\n\
         OLD:\n\
         <exact original lines>\n\
         END_OLD\n\n\
         Rules:\n\
         - Output ONLY patch DSL blocks, no markdown or explanations.\n\
         - If no changes are needed, output exactly NO_CHANGES.\n\
         - Preserve indentation and blank lines exactly inside CONTENT/OLD/NEW.\n\
         - Keep edits small and inside the allowed line region.\n\n\
         Region content:\n{region_content}",
        region.start,
        region.end,
        region.start,
        region.end,
        lsp_validation.as_str()
    );

    let request = ChatRequest {
        messages: vec![
            Message::system(
                "You output only strict patch DSL blocks. No explanations, no markdown.",
            ),
            Message::user(&prompt),
        ],
        tools: None,
        tool_choice: None,
    };

    log_stage(
        log,
        path_str,
        &format!("patch:region:{}-{}", region.start, region.end),
    );
    let response = router
        .chat_with_cancel(ModelRole::Fast, &request, cancelled)
        .await?;
    let text = response
        .choices
        .first()
        .and_then(|c| c.message.content.as_deref())
        .unwrap_or("");
    log_debug(
        log,
        path_str,
        &format!(
            "patch:region:{}-{} raw_response:\n{}",
            region.start,
            region.end,
            truncate_multiline(text, 12000)
        ),
    );
    let ops = match parse_patch(text) {
        Ok(ops) => ops,
        Err(e) => {
            log_debug(
                log,
                path_str,
                &format!(
                    "patch:region:{}-{} parse_failed {}",
                    region.start,
                    region.end,
                    truncate_multiline(&e.to_string(), 2000)
                ),
            );
            return Err(e);
        }
    };
    let output = if ops.is_empty() {
        String::new()
    } else {
        format!(
            "Region L{}-L{}: {} operation(s) found\n",
            region.start,
            region.end,
            ops.len()
        )
    };
    Ok((ops, output))
}

async fn request_preplan_steps(
    path_str: &str,
    task: &str,
    content: &str,
    router: &ModelRouter,
    repair: Option<&RepairContext>,
    max_literal_lines: usize,
    cancelled: Option<&AtomicBool>,
    log: Option<&SessionLog>,
) -> Result<PreplanOutcome> {
    ensure_not_cancelled(cancelled)?;
    let lines: Vec<&str> = content.lines().collect();
    let total_lines = lines.len();
    let max_reads = if repair.is_some() {
        MAX_PREPLAN_READS_REPAIR
    } else {
        MAX_PREPLAN_READS_INITIAL
    };
    let feedback_block = repair.map(format_repair_context).unwrap_or_default();
    let mut notes = Vec::<String>::new();
    let mut collected_commands = Vec::<InspectionCommand>::new();

    // Small files (including empty ones) skip the windowed observation
    // pass entirely: the whole file already fits in the finalize prompt,
    // so the only value the windows would add is an extra LLM round-trip.
    let small_file = total_lines <= SMALL_FILE_THRESHOLD;

    // ── Phase 1: windowed observation + inspection collection ───────────
    // Walk the file slice-by-slice. Each window may emit NOTE lines
    // describing structural landmarks AND SEARCH/READ commands the
    // planner will want answered before finalize. Commands are *collected*,
    // not executed — they're batch-executed once the full pass completes
    // so the model sees everything at finalize time.
    //
    let windows = if small_file {
        Vec::new()
    } else {
        build_windows(total_lines, WINDOW_SIZE, PREPLAN_READ_OVERLAP)
    };

    for (idx, (start, end)) in windows.iter().copied().enumerate() {
        ensure_not_cancelled(cancelled)?;
        let slice = render_numbered_slice(&lines, start, end);
        let notes_block = if notes.is_empty() {
            String::new()
        } else {
            format!(
                "Notes gathered so far:\n{}\n\n",
                notes
                    .iter()
                    .map(|note| format!("- {note}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        };
        let pending_block = if collected_commands.is_empty() {
            String::new()
        } else {
            format!(
                "Inspection commands already queued (don't repeat):\n{}\n\n",
                collected_commands
                    .iter()
                    .map(|c| match c {
                        InspectionCommand::Search(q) => format!("- SEARCH: {q}"),
                        InspectionCommand::Read { start, end } => {
                            format!("- READ: {start}-{end}")
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
            )
        };
        let repair_steps_block = repair
            .map(|ctx| format_repair_steps_for_window(ctx, start + 1, end))
            .unwrap_or_default();
        let prompt = format!(
            "File: {path_str}\n\
             Task: {task}\n\n\
             {feedback_block}\
             {repair_steps_block}\
             {notes_block}\
             {pending_block}\
             Slice {current_slice}/{total_slices}, lines {start_line}-{end_line} of {total_lines}.\n\n\
             IMPORTANT: The planning phase will NOT see the full file — only your NOTEs and inspection results. Your notes are the planner's only window into file content.\n\n\
             Output zero or more of these lines about landmarks the planner will need:\n\
             NOTE <fact with line number> — function/struct spans, signatures the task touches, the line where a relevant block starts.\n\
             Include enough verbatim detail (e.g. exact function signatures, parameter lists) that the planner can write LITERAL_REPLACE patches without seeing the file. Line ranges in NOTEs (e.g. L283-290) are automatically READ for the planner.\n\n\
             You may also request extra context before finalize with:\n\
             SEARCH: <exact text to find>\n\
             READ: <start>-<end>\n\n\
             SEARCH/READ commands are collected across ALL slices and batch-executed before the finalize phase — their results will reach the planner. Line ranges in your NOTEs also auto-trigger READs, so you only need explicit READ for ranges NOT mentioned in a NOTE. Don't repeat commands.\n\n\
             Reference line numbers from the slice. Skip vague observations and anything the planner can derive itself.\n\n\
             If the task is genuinely too vague, underspecified, or contradictory to execute — e.g. it names no target, or it requires information only the outer agent has — output exactly one line:\n\
             NEEDS_CLARIFICATION: <one specific question>\n\
             This is RARE. Do NOT use it for implementation decisions you can make yourself (parameter order, types, naming). Do NOT ask about information visible in the file content above — read it. Do NOT use it for tasks where the target doesn't exist yet — that's what the edit is for; plan the edit that creates it. If the task tells you WHAT to do but not HOW, just choose a reasonable approach.\n\n\
             Slice:\n{slice}",
            current_slice = idx + 1,
            total_slices = windows.len(),
            start_line = start + 1,
            end_line = end,
        );

        let request = ChatRequest {
            messages: vec![
                Message::system(
                    "Observation phase. Output only NOTE lines, SEARCH:/READ: commands, or a NEEDS_CLARIFICATION: <question> line. No edits, no markdown.",
                ),
                Message::user(&prompt),
            ],
            tools: None,
            tool_choice: None,
        };

        log_stage(
            log,
            path_str,
            &format!(
                "preplan:window:{}/{}:{}-{}",
                idx + 1,
                windows.len(),
                start + 1,
                end
            ),
        );
        let response = router
            .chat_with_cancel(ModelRole::Fast, &request, cancelled)
            .await?;
        let text = response
            .choices
            .first()
            .and_then(|c| c.message.content.as_deref())
            .unwrap_or("");
        log_debug(
            log,
            path_str,
            &format!(
                "preplan:window:{}/{}:{}-{} raw_response:\n{}",
                idx + 1,
                windows.len(),
                start + 1,
                end,
                truncate_multiline(text, 12000)
            ),
        );

        let parsed = parse_preplan_window_response(text);
        if let Some(question) = parsed.clarification {
            log_debug(
                log,
                path_str,
                &format!(
                    "preplan:window:{}/{}:{}-{} needs_clarification question={question}",
                    idx + 1,
                    windows.len(),
                    start + 1,
                    end,
                ),
            );
            return Ok(PreplanOutcome::NeedsClarification(question));
        }
        extend_unique_notes(&mut notes, parsed.notes);
        for command in parsed.commands {
            if !collected_commands.contains(&command) {
                collected_commands.push(command);
            }
        }
    }

    // ── Phase 1b: auto-READ line ranges mentioned in notes ─────────────
    // The observation phase produces notes like "NOTE 283-290 — assemble
    // fn signature" but the finalize phase for large files only sees the
    // note text, not the actual file lines. Without the code, the planner
    // can't write LITERAL_REPLACE (needs verbatim OLD) and often falls
    // back to NEEDS_CLARIFICATION asking "what is the exact signature?"
    //
    // Fix: extract line-range patterns from notes and inject READ commands
    // so the finalize phase automatically sees the code at noted locations.
    if !small_file {
        for note in &notes {
            for range in extract_line_ranges_from_note(note) {
                let cmd = InspectionCommand::Read {
                    start: range.0,
                    end: range.1,
                };
                if !collected_commands.contains(&cmd) {
                    collected_commands.push(cmd);
                }
            }
        }
    }

    // ── Phase 2: batch-execute the collected inspection commands ───────
    // Small files never reach this with anything in collected_commands
    // because the window loop was skipped; large files may have any
    // number of queued commands that we now run against the real file
    // content before finalize.
    let mut extra_context = String::new();
    if !collected_commands.is_empty() {
        let mut counters = InspectionCounters {
            search_count: 0,
            read_count: 0,
            max_reads,
        };
        execute_inspection_commands(
            content,
            &collected_commands,
            &mut counters,
            &mut extra_context,
            path_str,
            log,
        )?;
    }

    // ── Phase 3: planning ──────────────────────────────────────────────
    // For small files, the full file content goes directly into the
    // prompt. For large files the planner only sees the windowed notes
    // and the batch-executed inspection results — the file itself is too
    // big to inline.
    let file_view_block = if small_file {
        if total_lines == 0 {
            String::from("File content: (empty)\n\n")
        } else {
            format!(
                "File content ({total_lines} lines):\n{}\n\n",
                render_numbered_slice(&lines, 0, total_lines),
            )
        }
    } else {
        let notes_block = if notes.is_empty() {
            String::from("Notes: (none)\n\n")
        } else {
            format!(
                "Notes:\n{}\n\n",
                notes
                    .iter()
                    .map(|note| format!("- {note}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        };
        let results_block = if extra_context.is_empty() {
            String::from("Inspection results: (none)\n\n")
        } else {
            format!("Inspection results:\n{extra_context}")
        };
        format!("{notes_block}{results_block}")
    };
    let view_note = if small_file {
        "You see the full file above — line numbers in any plan must match it exactly."
    } else {
        "You only see the notes and inspection results above — line numbers in any plan must match the real file."
    };
    let finalize_prompt = format!(
        "File: {path_str} ({total_lines} lines)\n\
         Task: {task}\n\n\
         {feedback_block}\
         {file_view_block}\
         {view_note}\n\n\
         Decide the verdict for this iteration against the CURRENT file state. Pick exactly one:\n\n\
         (A) TASK ALREADY SATISFIED by the current file state. Output exactly one line:\n\
         COMPLETE\n\
         Use this when the task is done — whether prior iterations already completed it or the file was already in the desired state. This is a terminal verdict; no further edits will run.\n\n\
         (B) MORE EDITS NEEDED. Begin the response directly with `LITERAL_REPLACE` or `SMART_EDIT` — no header word, no preamble, no code fences. Up to {MAX_PREPLAN_STEPS} non-overlapping steps, each covering at most 5 edit sites. Steps must not share any line, including endpoints — L10-L20 and L20-L30 overlap.\n\
         Every step must change something. Do not emit a LITERAL_REPLACE whose NEW is identical to its OLD, and do not emit a SMART_EDIT whose task is to verify a region is unchanged or to keep it as-is — just leave those regions out of the plan.\n\
         Use LITERAL_REPLACE when you have the OLD text verbatim from an inspection result and OLD/NEW each span ≤ {max_literal_lines} lines. Otherwise use SMART_EDIT — its execution phase will see the region content.\n\
         Never use LITERAL_REPLACE for whole functions, impl blocks, modules, or test cases.\n\n\
         LITERAL_REPLACE\n\
         SCOPE <start> <end>\n\
         ALL true\n\
         OLD:\n\
         <exact text copied verbatim from L<start>-L<end>>\n\
         END_OLD\n\
         NEW:\n\
         <replacement text>\n\
         END_NEW\n\n\
         OLD: is required. Copy it from the file content you see above — do not type it from memory. If the region is too large to echo verbatim, use SMART_EDIT instead.\n\n\
         SMART_EDIT\n\
         REGION <start> <end>\n\
         TASK: <specific edit for this region>\n\n\
         (C) TASK CANNOT BE COMPLETED in this file. Output exactly one line:\n\
         FAILED: <one-line reason, under {MAX_FAILED_REASON_CHARS} chars>\n\
         Use this when the task contradicts file invariants, or prior attempts kept regressing and you have no better idea. Be concrete: name the obstacle. Not \"LSP errors\" but \"changing parameter N of function F breaks 3 call sites in tests/\".\n\
         If the task description says certain compilation errors are expected (e.g. \"arity error expected — callee updated in next step\"), proceed with the edit rather than emitting FAILED. This is a terminal verdict; no further edits will run.\n\n\
         (D) TASK TOO VAGUE to act on. Output exactly one line:\n\
         NEEDS_CLARIFICATION: <one specific question>\n\
         This is RARE — use only when the task truly names no target or contradicts the file. Do NOT use it for implementation decisions you can make yourself (parameter placement, types, naming, defaults). Do NOT ask about file content already shown above — read it. If the task tells you WHAT to do, just choose HOW and proceed with (B).\n\n\
         Output only one of (A)/(B)/(C)/(D). No markdown, no explanation, no empty responses."
    );

    let request = ChatRequest {
        messages: vec![
            Message::system(
                "Verdict phase. Output exactly one of: `COMPLETE` (task done), one or more `LITERAL_REPLACE`/`SMART_EDIT` blocks (more edits), `FAILED: <reason>` (task impossible), `NEEDS_CLARIFICATION: <question>` (task too vague). Start the response with the keyword itself — no header word, no markdown. Empty responses are not valid.",
            ),
            Message::user(&finalize_prompt),
        ],
        tools: None,
        tool_choice: None,
    };

    log_stage(
        log,
        path_str,
        &format!("preplan:finalize:file:1-{total_lines}"),
    );
    log_debug(
        log,
        path_str,
        &format!(
            "preplan:finalize:prompt_len={} notes={} inspection_results={} small_file={small_file}",
            finalize_prompt.len(),
            notes.len(),
            if extra_context.is_empty() {
                0
            } else {
                extra_context.lines().count()
            },
        ),
    );
    let response = router
        .chat_with_cancel(ModelRole::Fast, &request, cancelled)
        .await?;
    let text = response
        .choices
        .first()
        .and_then(|c| c.message.content.as_deref())
        .unwrap_or("");
    log_debug(
        log,
        path_str,
        &format!(
            "preplan:finalize:file:1-{total_lines} raw_response:\n{}",
            truncate_multiline(text, 12000)
        ),
    );

    // Empty response is now a pathology signal, not a "nothing to do"
    // signal. Historically we collapsed it to an empty plan and exited,
    // but bench logs show it's usually a stalled or misfired inference
    // and a retry with explicit feedback recovers cleanly. Route it to
    // the outer retry loop via `EmptyResponse`.
    if text.trim().is_empty() {
        log_debug(
            log,
            path_str,
            &format!("preplan:finalize:file:1-{total_lines} empty_response"),
        );
        return Ok(PreplanOutcome::EmptyResponse);
    }

    // Explicit COMPLETE (or legacy NO_CHANGES) sentinel — the model's
    // verdict that the task is satisfied by the current file state.
    // Terminal; the retry loop stops and reports success.
    if looks_like_complete(text) {
        log_debug(
            log,
            path_str,
            &format!("preplan:finalize:file:1-{total_lines} verdict=complete"),
        );
        return Ok(PreplanOutcome::Complete);
    }

    // Explicit FAILED: <reason> sentinel — the model's verdict that the
    // task cannot be completed. Terminal; the retry loop stops and
    // reports the failure with the model's own reason.
    if let Some(reason) = parse_failed(text) {
        log_debug(
            log,
            path_str,
            &format!("preplan:finalize:file:1-{total_lines} verdict=failed reason={reason}"),
        );
        return Ok(PreplanOutcome::Failed(reason));
    }

    if let Some(question) = parse_needs_clarification(text) {
        log_debug(
            log,
            path_str,
            &format!(
                "preplan:finalize:file:1-{total_lines} needs_clarification question={question}"
            ),
        );
        return Ok(PreplanOutcome::NeedsClarification(question));
    }

    let parsed = match parse_edit_plan(text) {
        Ok(steps) => steps,
        Err(e) => {
            log_debug(
                log,
                path_str,
                &format!(
                    "preplan:finalize:file:1-{total_lines} parse_failed {}",
                    truncate_multiline(&e.to_string(), 2000)
                ),
            );
            // Parse errors used to kill the whole edit_file call. Now we
            // surface them to the outer retry loop so the repair prompt
            // can tell the model exactly what was malformed and ask for
            // a corrected plan on the next attempt.
            return Ok(PreplanOutcome::ParseError(e.to_string()));
        }
    };

    // Recover from overlapping steps: keep the first occurrence (in source
    // order), partition the rest as dropped-with-reason. The executor will
    // apply the kept steps and report the dropped ones as failed steps in
    // the per-step output, so the agent sees both successes and failures
    // in the same shape.
    let (mut steps, dropped) = partition_overlapping_steps(parsed);
    if !dropped.is_empty() {
        log_debug(
            log,
            path_str,
            &format!(
                "preplan:finalize:file:1-{total_lines} dropped_overlapping_steps={}",
                dropped.len()
            ),
        );
    }

    log_debug(
        log,
        path_str,
        &format!(
            "preplan:finalize:file:1-{total_lines} parsed_steps={}\n{}",
            steps.len(),
            truncate_multiline(&format_edit_plan_steps(&steps), 12000)
        ),
    );
    if steps.len() > MAX_PREPLAN_STEPS {
        steps.truncate(MAX_PREPLAN_STEPS);
    }

    if let Err(e) = validate_steps_in_file(&steps, total_lines) {
        log_debug(
            log,
            path_str,
            &format!(
                "preplan:file_validation_failed {}",
                truncate_multiline(&e.to_string(), 2000)
            ),
        );
        return Err(e);
    }
    Ok(PreplanOutcome::Continue { steps, dropped })
}

fn validate_steps_in_file(steps: &[EditPlanStep], total_lines: usize) -> Result<()> {
    for step in steps {
        if step.end_line() > total_lines {
            bail!(
                "planned step L{}-L{} falls outside file with {total_lines} lines",
                step.start_line(),
                step.end_line()
            );
        }
    }
    // Overlap is handled upstream by `partition_overlapping_steps`, which
    // keeps the first occurrence in source order and reports the rest as
    // dropped steps in the per-step output instead of bailing on the plan.
    Ok(())
}

/// Render a `RepairContext` into a planner-facing block. Used at the top
/// of both the windowed pre-plan prompts and the finalize prompt so the
/// model can see exactly what the previous iteration tried and how it
/// landed. The shape differs for cleanly-applied vs failed attempts —
/// see `format_prior_applied` and `format_prior_failed` below.
fn format_repair_context(ctx: &RepairContext) -> String {
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
fn format_prior_applied(ctx: &RepairContext) -> String {
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
fn format_prior_failed(ctx: &RepairContext) -> String {
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
fn format_repair_steps_for_window(ctx: &RepairContext, win_start: usize, win_end: usize) -> String {
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

    if let Some(ref step) = ctx.failed_step {
        if overlaps(step) {
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
fn format_lsp_regression_for_planner(reg: &LspRegression) -> String {
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

fn validate_candidate(original: &str, candidate: &str) -> Result<()> {
    if !original.is_empty() && candidate.is_empty() {
        bail!("candidate output is empty for a non-empty file");
    }

    Ok(())
}

/// Truncation guard — runs at write time only (not in the inner smart-edit
/// retry loop). Catches the common failure mode where the LLM emits only a
/// diff fragment and loses the rest of the file. In interactive mode the user
/// can confirm an intentional large deletion; in headless / test / auto-approve
/// mode we reject as before.
fn gate_truncation(
    path_str: &str,
    original: &str,
    candidate: &str,
    perms: Option<&PermissionManager>,
) -> Result<()> {
    let old_lines = original.lines().count();
    let new_lines = candidate.lines().count();
    if old_lines <= LARGE_TRUNCATION_MIN_LINES || new_lines >= old_lines / 2 {
        return Ok(());
    }

    let rejection = format!("candidate truncates {path_str} from {old_lines} to {new_lines} lines");
    if let Some(perms) = perms
        && perms.confirm(&format!(
            "\x1b[1;33medit_file wants to shrink {path_str} from {old_lines} to {new_lines} lines.\x1b[0m\n  Is this intentional? [y]es / [n]o: "
        ))
    {
        return Ok(());
    }
    bail!(rejection);
}

/// Build window ranges for a file.
pub fn build_windows(
    total_lines: usize,
    window_size: usize,
    overlap: usize,
) -> Vec<(usize, usize)> {
    if total_lines <= window_size {
        return vec![(0, total_lines)];
    }

    let mut windows = Vec::new();
    let mut start = 0;
    while start < total_lines {
        let end = (start + window_size).min(total_lines);
        windows.push((start, end));
        if end >= total_lines {
            break;
        }
        start = end - overlap;
    }
    windows
}

#[cfg(test)]
mod tests {
    use super::apply::{
        find_all_exact_line_matches, find_all_fuzzy_line_matches,
        find_all_ws_tolerant_line_matches, pick_best_candidate,
    };
    use super::parse::{
        MAX_FAILED_REASON_CHARS, looks_like_complete, parse_edit_plan, parse_failed,
        parse_needs_clarification, parse_patch, partition_overlapping_steps, strip_code_fences,
    };
    use super::*;
    #[test]
    fn retry_feedback_adds_signature_guidance_for_lsp_arity_errors() {
        let feedback = build_retry_feedback(
            "LSP diagnostics worsened for src/main.rs: 0 -> 3 error(s)\nsrc/main.rs:31:79: error: expected 4 arguments, found 5",
            None,
        );

        assert!(feedback.contains("Do not repeat the same patch shape"));
        assert!(!feedback.contains("Current known callee signatures"));
    }

    fn smart_step(start: usize, end: usize) -> EditPlanStep {
        EditPlanStep::SmartEdit(EditRegion {
            start,
            end,
            task: format!("dummy task L{start}-L{end}"),
        })
    }

    #[test]
    fn partition_overlapping_steps_keeps_first_in_source_order() {
        // Three steps in emission order: middle one overlaps the first.
        // The first (by source line) wins; the overlapper is reported as
        // dropped, the third non-overlapping step is kept.
        let steps = vec![
            smart_step(5, 15),
            smart_step(10, 20), // overlaps step at L5-L15
            smart_step(25, 35),
        ];
        let (kept, dropped) = partition_overlapping_steps(steps);
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0].start_line(), 5);
        assert_eq!(kept[1].start_line(), 25);
        assert_eq!(dropped.len(), 1);
        assert_eq!(dropped[0].step.start_line(), 10);
        assert!(
            dropped[0].reason.contains("L5-L15"),
            "reason should reference the conflicting kept step, got {:?}",
            dropped[0].reason
        );
    }

    #[test]
    fn partition_overlapping_steps_treats_shared_endpoint_as_overlap() {
        // L10-L20 and L20-L30 share line 20 — that counts as overlap so
        // the planner can't smuggle two edits through a single shared
        // endpoint.
        let steps = vec![smart_step(10, 20), smart_step(20, 30)];
        let (kept, dropped) = partition_overlapping_steps(steps);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].start_line(), 10);
        assert_eq!(dropped.len(), 1);
        assert_eq!(dropped[0].step.start_line(), 20);
    }

    #[test]
    fn partition_overlapping_steps_keeps_disjoint_steps() {
        let steps = vec![smart_step(1, 5), smart_step(10, 15), smart_step(20, 25)];
        let (kept, dropped) = partition_overlapping_steps(steps);
        assert_eq!(kept.len(), 3);
        assert!(dropped.is_empty());
    }

    #[test]
    fn gate_truncation_allows_small_files() {
        // Below LARGE_TRUNCATION_MIN_LINES, the guard never fires — even a
        // "delete everything except one line" edit is allowed (the empty-file
        // check in validate_candidate still catches full deletion).
        let original: String = (0..20).map(|i| format!("line {i}\n")).collect();
        let candidate = "line 0\n";
        assert!(gate_truncation("small.rs", &original, candidate, None).is_ok());
    }

    #[test]
    fn gate_truncation_allows_partial_shrink_above_half() {
        // Shrinking from 100 to 60 lines keeps us above the half threshold,
        // so the guard stays quiet.
        let original: String = (0..100).map(|i| format!("line {i}\n")).collect();
        let candidate: String = (0..60).map(|i| format!("line {i}\n")).collect();
        assert!(gate_truncation("big.rs", &original, &candidate, None).is_ok());
    }

    #[test]
    fn gate_truncation_rejects_large_cut_without_perms() {
        // 100 -> 40 lines on a >50 line file is below half; with no perms
        // available, the guard rejects (preserving the pre-existing behavior
        // for tests and headless runs).
        let original: String = (0..100).map(|i| format!("line {i}\n")).collect();
        let candidate: String = (0..40).map(|i| format!("line {i}\n")).collect();
        let err = gate_truncation("big.rs", &original, &candidate, None).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("truncates"), "unexpected error: {msg}");
        assert!(msg.contains("100"), "should name old line count: {msg}");
        assert!(msg.contains("40"), "should name new line count: {msg}");
    }

    #[test]
    fn gate_truncation_rejects_large_cut_in_headless_mode() {
        // With an auto-approve PermissionManager (headless), the guard
        // rejects without prompting — same as passing None.
        let config = Config::default();
        let perms = PermissionManager::headless(&config);
        let original: String = (0..100).map(|i| format!("line {i}\n")).collect();
        let candidate: String = (0..10).map(|i| format!("line {i}\n")).collect();
        let err = gate_truncation("big.rs", &original, &candidate, Some(&perms)).unwrap_err();
        assert!(err.to_string().contains("truncates"));
    }

    fn s(lines: &[&str]) -> Vec<String> {
        lines.iter().map(|l| l.to_string()).collect()
    }

    #[test]
    fn find_all_exact_line_matches_finds_single_hit() {
        let content = "alpha\nbeta\ngamma\n";
        let hits = find_all_exact_line_matches(content, &s(&["beta", "gamma"]));
        assert_eq!(hits, vec![(2, 3)]);
    }

    #[test]
    fn find_all_exact_line_matches_finds_multiple_hits() {
        let content = "foo\nfoo\nfoo\n";
        let hits = find_all_exact_line_matches(content, &s(&["foo"]));
        assert_eq!(hits, vec![(1, 1), (2, 2), (3, 3)]);
    }

    #[test]
    fn find_all_exact_line_matches_returns_empty_when_needle_longer_than_haystack() {
        let content = "one\n";
        let hits = find_all_exact_line_matches(content, &s(&["one", "two"]));
        assert!(hits.is_empty());
    }

    #[test]
    fn find_all_exact_line_matches_empty_needle_returns_empty() {
        let content = "one\ntwo\n";
        let hits = find_all_exact_line_matches(content, &[]);
        assert!(hits.is_empty());
    }

    #[test]
    fn find_all_ws_tolerant_ignores_indentation_drift() {
        // Planner used spaces, file uses tabs.
        let content = "\tif x {\n\t\treturn 1;\n\t}\n";
        let hits =
            find_all_ws_tolerant_line_matches(content, &s(&["if x {", "    return 1;", "}"]), &[]);
        assert_eq!(hits, vec![(1, 3)]);
    }

    #[test]
    fn find_all_ws_tolerant_skips_excluded_ranges() {
        // Byte-exact hit already present — whitespace search should not
        // re-add the same range.
        let content = "foo\nbar\n";
        let exclude = vec![(1, 1)];
        let hits = find_all_ws_tolerant_line_matches(content, &s(&["foo"]), &exclude);
        assert!(hits.is_empty());
    }

    #[test]
    fn find_all_ws_tolerant_rejects_all_blank_old() {
        // An OLD block that squashes to nothing would otherwise match
        // every stretch of blank lines.
        let content = "\n\n\nfoo\n\n\n";
        let hits = find_all_ws_tolerant_line_matches(content, &s(&["   ", "\t"]), &[]);
        assert!(hits.is_empty());
    }

    #[test]
    fn pick_best_candidate_prefers_overlap() {
        // Two candidates: (5, 10) overlaps declared (8, 12); (20, 25)
        // does not. Overlap wins even if it's further from the endpoints.
        let candidates = vec![(5, 10), (20, 25)];
        let picked = pick_best_candidate(&candidates, 8, 12).unwrap();
        assert_eq!(picked, (5, 10));
    }

    #[test]
    fn pick_best_candidate_picks_nearest_when_no_overlap() {
        let candidates = vec![(1, 5), (30, 35), (100, 105)];
        let picked = pick_best_candidate(&candidates, 28, 34).unwrap();
        assert_eq!(picked, (30, 35));
    }

    #[test]
    fn pick_best_candidate_tie_break_prefers_earlier_insertion() {
        // Two candidates equidistant from the scope — the first one
        // inserted wins. This lets the caller front-load byte-exact
        // hits ahead of whitespace-tolerant ones.
        let candidates = vec![(10, 14), (20, 24)];
        let picked = pick_best_candidate(&candidates, 15, 19).unwrap();
        assert_eq!(picked, (10, 14));
    }

    #[test]
    fn pick_best_candidate_returns_none_for_empty_list() {
        assert_eq!(pick_best_candidate(&[], 1, 5), None);
    }

    #[test]
    fn extract_line_ranges_parses_range_with_l_prefix() {
        let ranges = extract_line_ranges_from_note("L283-290 — assemble fn signature");
        assert_eq!(ranges, vec![(283, 290)]);
    }

    #[test]
    fn extract_line_ranges_parses_bare_range() {
        let ranges = extract_line_ranges_from_note("132-138 — context::assemble call");
        assert_eq!(ranges, vec![(132, 138)]);
    }

    #[test]
    fn extract_line_ranges_parses_single_line_to_window() {
        // Single line reference expands to ±2 window
        let ranges = extract_line_ranges_from_note("L41 — run function signature");
        assert_eq!(ranges, vec![(39, 43)]);
    }

    #[test]
    fn extract_line_ranges_parses_multiple_ranges() {
        let ranges = extract_line_ranges_from_note("L10-20 foo, L30-40 bar");
        assert_eq!(ranges, vec![(10, 20), (30, 40)]);
    }

    #[test]
    fn extract_line_ranges_ignores_mid_word_numbers() {
        // "v2" or "utf8" shouldn't parse as line references
        let ranges = extract_line_ranges_from_note("uses utf8 encoding and v2 protocol");
        assert!(ranges.is_empty());
    }

    #[test]
    fn extract_line_ranges_single_line_near_start_clamps() {
        // Line 1 expanded ±2 should clamp start to 1
        let ranges = extract_line_ranges_from_note("L1 — first line");
        assert_eq!(ranges, vec![(1, 3)]);
    }

    #[test]
    fn find_all_fuzzy_rescues_single_char_typo() {
        // Typo'd identifier: `callback` vs `calback`. Byte-exact and
        // whitespace-normalized both reject; fuzzy accepts.
        let content = "fn handler(callback: F) {\n    callback();\n}\n";
        let hits = find_all_fuzzy_line_matches(content, &s(&["    calback();"]), &[]);
        assert_eq!(hits, vec![(2, 2)]);
    }

    #[test]
    fn find_all_fuzzy_rescues_multiline_near_match() {
        // Two lines where one has a small typo. Both lines clear the
        // per-line floor and the block average clears the block floor.
        let content =
            "let mut config = Config::default();\nconfig.timeout = 30;\nconfig.retries = 5;\n";
        let hits = find_all_fuzzy_line_matches(
            content,
            &s(&["let mut config = Config::defalt();", "config.timeout = 30;"]),
            &[],
        );
        assert_eq!(hits, vec![(1, 2)]);
    }

    #[test]
    fn find_all_fuzzy_rejects_block_below_threshold() {
        // Block average similarity is too low — the second line is
        // entirely different. Fuzzy should decline rather than surface a
        // noisy match.
        let content = "fn main() {\n    println!(\"hello\");\n}\n";
        let hits = find_all_fuzzy_line_matches(
            content,
            &s(&["fn main() {", "    assert_eq!(foo, bar_baz);"]),
            &[],
        );
        assert!(hits.is_empty());
    }

    #[test]
    fn find_all_fuzzy_rejects_trivially_short_old() {
        // Single-line OLDs shorter than 5 non-whitespace chars are too
        // noisy to fuzzy-match reliably; the function bails early.
        let content = "x = 1;\ny = 2;\nz = 3;\n";
        let hits = find_all_fuzzy_line_matches(content, &s(&["x=1"]), &[]);
        assert!(hits.is_empty());
    }

    #[test]
    fn find_all_fuzzy_skips_excluded_ranges() {
        let content = "callback(foo);\ncallback(foo);\n";
        // Pretend the first line was already matched exactly elsewhere;
        // fuzzy should still return the second.
        let hits = find_all_fuzzy_line_matches(content, &s(&["calback(foo);"]), &[(1, 1)]);
        assert_eq!(hits, vec![(2, 2)]);
    }

    #[test]
    fn parse_preplan_window_response_collects_note_lines() {
        let response = parse_preplan_window_response(
            "NOTE provider loop spans L10-L20\nNOTE: assemble fn signature at L283\n",
        );
        assert_eq!(
            response.notes,
            vec![
                "provider loop spans L10-L20".to_string(),
                "assemble fn signature at L283".to_string(),
            ],
        );
        assert!(response.commands.is_empty());
    }

    #[test]
    fn parse_preplan_window_response_collects_search_and_read_commands() {
        // The windowed pre-plan pass no longer has a separate recon phase —
        // the model queues SEARCH/READ requests right alongside the notes
        // it emits, and they get batch-executed before finalize.
        let response = parse_preplan_window_response(
            "NOTE callers live at L10-L20\n\
             SEARCH: assemble(\n\
             READ: 280-300\n",
        );
        assert_eq!(response.notes, vec!["callers live at L10-L20".to_string()]);
        assert_eq!(
            response.commands,
            vec![
                InspectionCommand::Search("assemble(".into()),
                InspectionCommand::Read {
                    start: 280,
                    end: 300
                },
            ],
        );
    }

    #[test]
    fn parse_preplan_window_response_drops_invalid_read_ranges() {
        // Malformed READ ranges (start=0, end<start, non-numeric) are
        // silently dropped rather than erroring out so the model doesn't
        // crash the pipeline with a typo.
        let response =
            parse_preplan_window_response("READ: 0-5\nREAD: 10-3\nREAD: bad\nNOTE still here\n");
        assert!(response.commands.is_empty());
        assert_eq!(response.notes, vec!["still here".to_string()]);
    }

    #[test]
    fn parse_preplan_window_response_silently_ignores_unknown_lines() {
        // Stray control words, half-formed edit-plan blocks, and leftover
        // DONE terminators from the old recon phase all get dropped
        // without erroring out.
        let response = parse_preplan_window_response(
            "NOTE looks good\n\
             SMART_EDIT\n\
             REGION 1 5\n\
             TASK: do thing\n\
             END\n\
             NO_CHANGES\n\
             DONE\n",
        );
        assert_eq!(response.notes, vec!["looks good".to_string()]);
        assert!(response.commands.is_empty());
    }

    #[test]
    fn parse_preplan_window_response_accepts_empty_response() {
        // The model may have nothing to add for a slice — that's fine.
        let response = parse_preplan_window_response("");
        assert!(response.notes.is_empty());
        assert!(response.commands.is_empty());
        assert!(response.clarification.is_none());
    }

    #[test]
    fn parse_preplan_window_response_captures_clarification() {
        // A bare NEEDS_CLARIFICATION line sets the clarification field and
        // does not pollute notes/commands.
        let response = parse_preplan_window_response(
            "NEEDS_CLARIFICATION: which run() function should accept the override?\n",
        );
        assert_eq!(
            response.clarification,
            Some("which run() function should accept the override?".to_string())
        );
        assert!(response.notes.is_empty());
        assert!(response.commands.is_empty());
    }

    #[test]
    fn parse_preplan_window_response_clarification_coexists_with_notes() {
        // A confused model might emit a NOTE and also hedge with a
        // clarification — we keep both, and the caller short-circuits on
        // the clarification regardless.
        let response = parse_preplan_window_response(
            "NOTE run() at L12\nNEEDS_CLARIFICATION: which module owns this?\n",
        );
        assert_eq!(response.notes, vec!["run() at L12".to_string()]);
        assert_eq!(
            response.clarification,
            Some("which module owns this?".to_string())
        );
    }

    #[test]
    fn execute_inspection_commands_executes_search_and_read() {
        let content = "fn one() {}\nfn two() {}\nfn three() {}\n";
        let commands = vec![
            InspectionCommand::Search("two".into()),
            InspectionCommand::Read { start: 1, end: 2 },
        ];
        let mut counters = InspectionCounters {
            search_count: 0,
            read_count: 0,
            max_reads: 6,
        };
        let mut extra = String::new();
        execute_inspection_commands(
            content,
            &commands,
            &mut counters,
            &mut extra,
            "test.rs",
            None,
        )
        .unwrap();
        assert_eq!(counters.search_count, 1);
        assert_eq!(counters.read_count, 1);
        assert!(extra.contains("SEARCH_RESULT query=`two`"));
        assert!(extra.contains("READ_RESULT range=L1-L2"));
        assert!(extra.contains("fn one()"));
    }

    #[test]
    fn execute_inspection_commands_respects_read_cap() {
        // The per-edit READ cap is enforced across all queued commands in
        // the single batch pass.
        let content = "x\n";
        let mut counters = InspectionCounters {
            search_count: 0,
            read_count: 0,
            max_reads: 1,
        };
        let mut extra = String::new();
        execute_inspection_commands(
            content,
            &[
                InspectionCommand::Read { start: 1, end: 1 },
                InspectionCommand::Read { start: 1, end: 1 },
            ],
            &mut counters,
            &mut extra,
            "test.rs",
            None,
        )
        .unwrap();
        assert_eq!(counters.read_count, 1, "second read should be capped");
        assert!(extra.contains("READ 1-1 skipped: per-edit limit"));
    }

    #[test]
    fn execute_inspection_commands_drops_overflow_with_inline_note() {
        let content = "x\n";
        let mut commands = Vec::new();
        for _ in 0..(MAX_PREPLAN_SEARCHES + 2) {
            commands.push(InspectionCommand::Search("x".into()));
        }
        let mut counters = InspectionCounters {
            search_count: 0,
            read_count: 0,
            max_reads: 6,
        };
        let mut extra = String::new();
        execute_inspection_commands(
            content,
            &commands,
            &mut counters,
            &mut extra,
            "test.rs",
            None,
        )
        .unwrap();
        assert_eq!(counters.search_count, MAX_PREPLAN_SEARCHES);
        assert!(extra.contains("SEARCH `x` skipped: per-edit limit"));
    }

    #[test]
    fn inspection_results_are_labeled() {
        let mut extra = String::new();
        append_inspection_result(
            &mut extra,
            "SEARCH_RESULT query=`foo`",
            "SEARCH RESULT for `foo`: 1 hit",
        );
        append_inspection_result(
            &mut extra,
            "READ_RESULT range=L10-L12",
            "READ RESULT L10-L12:\n  10│x",
        );

        assert!(extra.contains("SEARCH_RESULT query=`foo`"));
        assert!(extra.contains("READ_RESULT range=L10-L12"));
        assert!(!extra.contains("Inspection result:"));
    }

    #[test]
    fn parse_needs_clarification_accepts_keyword_with_question() {
        let r = parse_needs_clarification(
            "NEEDS_CLARIFICATION: which of the two run() functions should accept the override?\n",
        );
        assert_eq!(
            r,
            Some("which of the two run() functions should accept the override?".to_string())
        );
    }

    #[test]
    fn parse_needs_clarification_accepts_keyword_with_whitespace_question() {
        let r = parse_needs_clarification(
            "NEEDS_CLARIFICATION   what should the new parameter default to?\n",
        );
        assert_eq!(
            r,
            Some("what should the new parameter default to?".to_string())
        );
    }

    #[test]
    fn parse_needs_clarification_accepts_keyword_alone() {
        let r = parse_needs_clarification("NEEDS_CLARIFICATION");
        assert_eq!(r, Some(String::new()));
    }

    #[test]
    fn parse_needs_clarification_tolerates_leading_whitespace() {
        let r = parse_needs_clarification("  \n  NEEDS_CLARIFICATION: which field?\n");
        assert_eq!(r, Some("which field?".to_string()));
    }

    #[test]
    fn parse_needs_clarification_rejects_lookalikes() {
        // Don't false-match on a word that just starts with NEEDS_CLARIFICATION.
        assert_eq!(parse_needs_clarification("NEEDS_CLARIFICATIONS"), None);
        assert_eq!(
            parse_needs_clarification("NEEDS_CLARIFICATIONAL: foo"),
            None
        );
    }

    #[test]
    fn parse_needs_clarification_rejects_unrelated_text() {
        assert_eq!(parse_needs_clarification("NO_CHANGES"), None);
        assert_eq!(
            parse_needs_clarification("LITERAL_REPLACE\nSCOPE 1 1\n"),
            None
        );
        assert_eq!(parse_needs_clarification(""), None);
    }

    #[test]
    fn parse_needs_clarification_keeps_only_first_line() {
        // The sentinel is single-line by contract — any trailing lines
        // (e.g. stray tokens from a confused model) are dropped.
        let r = parse_needs_clarification("NEEDS_CLARIFICATION: which file?\nNOTE stray");
        assert_eq!(r, Some("which file?".to_string()));
    }

    #[test]
    fn looks_like_complete_accepts_bare_sentinel() {
        assert!(looks_like_complete("COMPLETE"));
        // Legacy spelling preserved for back-compat during rollout.
        assert!(looks_like_complete("NO_CHANGES"));
    }

    #[test]
    fn looks_like_complete_tolerates_surrounding_whitespace() {
        assert!(looks_like_complete("  COMPLETE  "));
        assert!(looks_like_complete("\n\nCOMPLETE\n"));
        assert!(looks_like_complete("\tCOMPLETE\n\n"));
    }

    #[test]
    fn looks_like_complete_tolerates_code_fences() {
        // Smaller models sometimes wrap everything in a fence, and
        // `strip_code_fences` already handles that shape — we just need
        // to make sure the sentinel recognizer composes with it.
        assert!(looks_like_complete("```\nCOMPLETE\n```"));
        assert!(looks_like_complete("```text\nCOMPLETE\n```"));
    }

    #[test]
    fn looks_like_complete_rejects_empty() {
        // Empty response is the pathology case, not a verdict — it
        // must NOT be treated as COMPLETE.
        assert!(!looks_like_complete(""));
        assert!(!looks_like_complete("   "));
        assert!(!looks_like_complete("\n\n"));
    }

    #[test]
    fn looks_like_complete_rejects_content_alongside_sentinel() {
        assert!(!looks_like_complete("COMPLETE\nLITERAL_REPLACE\nSCOPE 1 1"));
        assert!(!looks_like_complete("LITERAL_REPLACE\nSCOPE 1 1\nCOMPLETE"));
        assert!(!looks_like_complete("NEEDS_CLARIFICATION: which field?"));
    }

    #[test]
    fn looks_like_complete_rejects_lookalikes() {
        assert!(!looks_like_complete("COMPLETED"));
        assert!(!looks_like_complete("COMPLETELY_DONE"));
        assert!(!looks_like_complete("complete")); // case-sensitive on purpose
    }

    #[test]
    fn parse_failed_accepts_reason_with_colon() {
        let r = parse_failed("FAILED: signature change breaks 3 callers in tests/");
        assert_eq!(
            r,
            Some("signature change breaks 3 callers in tests/".to_string())
        );
    }

    #[test]
    fn parse_failed_accepts_reason_with_whitespace() {
        let r = parse_failed("FAILED  region too large to replace verbatim");
        assert_eq!(r, Some("region too large to replace verbatim".to_string()));
    }

    #[test]
    fn parse_failed_accepts_bare_keyword() {
        // Reasonless failure still parses, caller substitutes placeholder.
        assert_eq!(parse_failed("FAILED"), Some(String::new()));
    }

    #[test]
    fn parse_failed_rejects_lookalikes() {
        assert!(parse_failed("FAILURE: whatever").is_none());
        assert!(parse_failed("failed: lower").is_none());
    }

    #[test]
    fn parse_failed_keeps_only_first_line() {
        let r = parse_failed("FAILED: top line\nsecond line");
        assert_eq!(r, Some("top line".to_string()));
    }

    #[test]
    fn parse_failed_caps_reason_length() {
        let long: String = "x".repeat(MAX_FAILED_REASON_CHARS + 50);
        let input = format!("FAILED: {long}");
        let r = parse_failed(&input).expect("should parse");
        // The returned reason is capped + ellipsis, so <= max+1 chars.
        assert!(r.chars().count() <= MAX_FAILED_REASON_CHARS + 1);
        assert!(r.ends_with('…'));
    }

    #[test]
    fn format_repair_context_first_step_failed_renders_empty_completed() {
        let ctx = RepairContext {
            previous_plan: vec![EditPlanStep::SmartEdit(EditRegion {
                start: 10,
                end: 20,
                task: "rewrite header".into(),
            })],
            completed_steps: Vec::new(),
            failed_step: Some(EditPlanStep::SmartEdit(EditRegion {
                start: 10,
                end: 20,
                task: "rewrite header".into(),
            })),
            failure_reason: "region missing anchor".into(),
            lsp_regression: None,
            cleanly_applied: false,
        };

        let block = format_repair_context(&ctx);
        assert!(block.contains("The previous iteration failed."));
        assert!(block.contains("Previous edit plan (as tried):"));
        assert!(block.contains("rewrite header"));
        assert!(block.contains(
            "Steps that succeeded and have ALREADY been applied to the file shown below:\n(none — the first step failed, file is unchanged from the initial state)"
        ));
        assert!(block.contains("Step that FAILED:"));
        assert!(block.contains("Failure reason:\nregion missing anchor"));
        assert!(block.contains("`FAILED: <reason>`"));
    }

    #[test]
    fn format_repair_context_validation_failure_has_no_failed_step() {
        let step_a = EditPlanStep::SmartEdit(EditRegion {
            start: 10,
            end: 20,
            task: "first step".into(),
        });
        let step_b = EditPlanStep::SmartEdit(EditRegion {
            start: 30,
            end: 40,
            task: "second step".into(),
        });
        let ctx = RepairContext {
            previous_plan: vec![step_a.clone(), step_b.clone()],
            completed_steps: vec![step_a, step_b],
            failed_step: None,
            failure_reason: "LSP diagnostics worsened: 0 -> 4 errors".into(),
            lsp_regression: None,
            cleanly_applied: false,
        };

        let block = format_repair_context(&ctx);
        // Both steps appear in completed.
        assert!(block.contains("first step"));
        assert!(block.contains("second step"));
        assert!(block.contains(
            "Step that FAILED:\n(no individual step failed; the plan executed in full but post-validation rejected the result"
        ));
        assert!(block.contains("LSP diagnostics worsened"));
    }

    #[test]
    fn format_repair_context_partial_success_lists_completed_and_failed() {
        let completed_step = EditPlanStep::LiteralReplace {
            scope_start: 5,
            scope_end: 10,
            all: false,
            old: vec!["foo".into()],
            new: vec!["bar".into()],
        };
        let failed_step = EditPlanStep::SmartEdit(EditRegion {
            start: 30,
            end: 40,
            task: "rewrite second region".into(),
        });
        let ctx = RepairContext {
            previous_plan: vec![completed_step.clone(), failed_step.clone()],
            completed_steps: vec![completed_step],
            failed_step: Some(failed_step),
            failure_reason: "smart edit returned no diff".into(),
            lsp_regression: None,
            cleanly_applied: false,
        };

        let block = format_repair_context(&ctx);
        // Completed section names the literal replace.
        assert!(block.contains("LITERAL_REPLACE"));
        assert!(block.contains("OLD:\nfoo"));
        // Failed section names the smart edit.
        assert!(block.contains("rewrite second region"));
        assert!(block.contains("smart edit returned no diff"));
        // The "first step failed" stub must NOT appear when partial success exists.
        assert!(!block.contains("(none — the first step failed"));
    }

    #[test]
    fn format_repair_context_cleanly_applied_uses_soft_language() {
        let step = EditPlanStep::LiteralReplace {
            scope_start: 5,
            scope_end: 10,
            all: false,
            old: vec!["old text".into()],
            new: vec!["new text".into()],
        };
        let ctx = RepairContext {
            previous_plan: vec![step.clone()],
            completed_steps: vec![step],
            failed_step: None,
            failure_reason: String::new(),
            lsp_regression: None,
            cleanly_applied: true,
        };

        let block = format_repair_context(&ctx);
        // Success framing — no "failed" language.
        assert!(!block.contains("failed"));
        assert!(!block.contains("FAILED") || block.contains("`FAILED:"));
        assert!(block.contains("applied the following steps cleanly"));
        assert!(block.contains("`COMPLETE`"));
        assert!(block.contains("LITERAL_REPLACE"));
    }

    #[test]
    fn format_repair_steps_for_window_shows_overlapping_steps() {
        let completed = EditPlanStep::LiteralReplace {
            scope_start: 17,
            scope_end: 17,
            all: true,
            old: vec!["    let x = assemble(&config, None);".into()],
            new: vec!["    let x = assemble(&config, None, None);".into()],
        };
        let failed = EditPlanStep::LiteralReplace {
            scope_start: 54,
            scope_end: 54,
            all: true,
            old: vec!["    let y = assemble(&config, None);".into()],
            new: vec!["    let y = assemble(&config, None, None);".into()],
        };
        let ctx = RepairContext {
            previous_plan: vec![completed.clone(), failed.clone()],
            completed_steps: vec![completed],
            failed_step: Some(failed),
            failure_reason: "literal OLD block was not found".into(),
            lsp_regression: None,
            cleanly_applied: false,
        };

        // Window 1-30: only the completed step overlaps
        let block = format_repair_steps_for_window(&ctx, 1, 30);
        assert!(block.contains("✓ L17"));
        assert!(block.contains("LITERAL_REPLACE applied"));
        assert!(
            !block.contains("✗"),
            "failed step at L54 should not appear in window 1-30"
        );

        // Window 40-70: only the failed step overlaps
        let block = format_repair_steps_for_window(&ctx, 40, 70);
        assert!(block.contains("✗ L54"));
        assert!(block.contains("FAILED"));
        assert!(
            !block.contains("✓"),
            "completed step at L17 should not appear in window 40-70"
        );

        // Window 1-100: both steps overlap
        let block = format_repair_steps_for_window(&ctx, 1, 100);
        assert!(block.contains("✓ L17"));
        assert!(block.contains("✗ L54"));

        // Window 200-300: neither step overlaps
        let block = format_repair_steps_for_window(&ctx, 200, 300);
        assert!(block.is_empty());
    }

    #[test]
    fn strip_code_fences_removes_full_wrap_with_language_tag() {
        let input = "```rust\nLITERAL_REPLACE\nSCOPE 1 1\nALL true\nOLD:\nfoo\nEND_OLD\nNEW:\nbar\nEND_NEW\nEND\n```";
        let stripped = strip_code_fences(input);
        assert!(!stripped.contains("```"));
        assert!(stripped.starts_with("LITERAL_REPLACE"));
        assert!(stripped.contains("END_NEW"));
    }

    #[test]
    fn strip_code_fences_removes_full_wrap_without_language_tag() {
        let input = "```\nREPLACE_AT 5\nOLD:\nx\nEND_OLD\nNEW:\ny\nEND_NEW\n```";
        let stripped = strip_code_fences(input);
        assert!(!stripped.contains("```"));
        assert!(stripped.starts_with("REPLACE_AT 5"));
    }

    #[test]
    fn strip_code_fences_tolerates_leading_whitespace() {
        let input = "\n  \n```rust\nSMART_EDIT\nREGION 1 5\nTASK: x\nEND\n```\n";
        let stripped = strip_code_fences(input);
        assert!(!stripped.contains("```"));
        assert!(stripped.starts_with("SMART_EDIT"));
    }

    #[test]
    fn strip_code_fences_keeps_text_with_no_fence() {
        let input =
            "LITERAL_REPLACE\nSCOPE 1 1\nALL true\nOLD:\nfoo\nEND_OLD\nNEW:\nbar\nEND_NEW\nEND\n";
        let stripped = strip_code_fences(input);
        // Identical pass-through.
        assert_eq!(stripped, input);
    }

    #[test]
    fn strip_code_fences_handles_truncated_unclosed_fence() {
        // Model started a ```rust block but ran out of tokens before
        // closing it. We should still strip the opener and keep the
        // body so the parser has a chance.
        let input = "```rust\nLITERAL_REPLACE\nSCOPE 1 1\nALL true\nOLD:\nfoo\nEND_OLD\nNEW:\nbar\nEND_NEW\nEND\n";
        let stripped = strip_code_fences(input);
        assert!(!stripped.starts_with("```"));
        assert!(stripped.starts_with("LITERAL_REPLACE"));
        assert!(stripped.contains("END_NEW"));
    }

    #[test]
    fn strip_code_fences_does_not_touch_inline_backticks() {
        // Backticks inside an otherwise plain block must not be
        // misinterpreted as a closing fence by some greedy regex
        // somewhere — we only consider an outer leading ``` fence.
        let input = "REPLACE_AT 1\nOLD:\nlet `s` = 1;\nEND_OLD\nNEW:\nlet `t` = 2;\nEND_NEW\n";
        let stripped = strip_code_fences(input);
        assert_eq!(stripped, input);
    }

    #[test]
    fn parse_edit_plan_accepts_markdown_fenced_response() {
        // Real failure mode from past benches: model wraps the entire
        // structured response in a ```rust ... ``` fence even though the
        // prompt forbids markdown. Parsing must succeed and return the
        // step inside.
        let input = "```rust\nLITERAL_REPLACE\nSCOPE 1 1\nALL true\nOLD:\nfoo\nEND_OLD\nNEW:\nbar\nEND_NEW\nEND\n```";
        let steps = parse_edit_plan(input).expect("fenced plan should parse");
        assert_eq!(steps.len(), 1);
        match &steps[0] {
            EditPlanStep::LiteralReplace {
                scope_start,
                scope_end,
                all,
                old,
                new,
            } => {
                assert_eq!(*scope_start, 1);
                assert_eq!(*scope_end, 1);
                assert!(*all);
                assert_eq!(old, &vec!["foo".to_string()]);
                assert_eq!(new, &vec!["bar".to_string()]);
            }
            other => panic!("unexpected step variant: {other:?}"),
        }
    }

    #[test]
    fn parse_edit_plan_rejects_literal_replace_without_old() {
        // The OLD-less shortcut is gone — the parser must reject it
        // with a message that tells the planner to either provide OLD
        // verbatim or switch to SMART_EDIT. This is the primary guard
        // against hallucinated replacements.
        let input = "\
LITERAL_REPLACE
SCOPE 10 12
ALL true
NEW:
new line one
new line two
END_NEW
END
";
        let err = parse_edit_plan(input).expect_err("no-OLD form must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("OLD:"),
            "error should instruct the planner to include OLD:, got: {msg}"
        );
        assert!(
            msg.contains("SMART_EDIT"),
            "error should mention SMART_EDIT as the fallback, got: {msg}"
        );
    }

    #[test]
    fn parse_edit_plan_rejects_no_old_form_even_when_all_false() {
        // Same rejection applies regardless of the ALL flag — OLD is
        // required unconditionally.
        let input = "\
LITERAL_REPLACE
SCOPE 1 3
ALL false
NEW:
hello
END_NEW
END
";
        let err = parse_edit_plan(input).expect_err("no-OLD form must be rejected");
        assert!(err.to_string().contains("OLD:"));
    }

    #[test]
    fn parse_edit_plan_classic_literal_replace_still_parses() {
        // Don't regress the existing OLD-bearing form when adding the
        // peek-ahead branch.
        let input = "\
LITERAL_REPLACE
SCOPE 5 7
ALL false
OLD:
foo()
END_OLD
NEW:
bar()
END_NEW
END
";
        let steps = parse_edit_plan(input).expect("classic literal replace should still parse");
        assert_eq!(steps.len(), 1);
        match &steps[0] {
            EditPlanStep::LiteralReplace {
                scope_start,
                scope_end,
                all,
                old,
                new,
            } => {
                assert_eq!(*scope_start, 5);
                assert_eq!(*scope_end, 7);
                assert!(!*all);
                assert_eq!(old, &vec!["foo()".to_string()]);
                assert_eq!(new, &vec!["bar()".to_string()]);
            }
            other => panic!("expected LiteralReplace, got {other:?}"),
        }
    }

    #[test]
    fn parse_patch_accepts_markdown_fenced_response() {
        let input = "```\nREPLACE_AT 5\nOLD:\nold line\nEND_OLD\nNEW:\nnew line\nEND_NEW\n```";
        let ops = parse_patch(input).expect("fenced patch should parse");
        assert_eq!(ops.len(), 1);
        match &ops[0] {
            PatchOp::ReplaceAt { start, old, new } => {
                assert_eq!(*start, 5);
                assert_eq!(old, &vec!["old line".to_string()]);
                assert_eq!(new, &vec!["new line".to_string()]);
            }
            other => panic!("unexpected op variant: {other:?}"),
        }
    }

    #[test]
    fn search_and_read_helpers_report_current_file_content() {
        let content = "\
fn one() {\n\
    context::assemble(&config, \"a\", &[], false, None);\n\
}\n\
\n\
fn two() {\n\
    context::assemble(&config, \"b\", &[], false, None);\n\
}\n";
        let search = search_in_file(content, "context::assemble");
        assert!(search.contains("2 hit(s)"));
        assert!(search.contains("context::assemble(&config, \"a\""));
        assert!(search.contains("context::assemble(&config, \"b\""));

        let read = read_in_file(content, 2, 3).unwrap();
        assert!(read.contains("READ RESULT L2-L3"));
        assert!(read.contains("context::assemble(&config, \"a\""));
    }
}
