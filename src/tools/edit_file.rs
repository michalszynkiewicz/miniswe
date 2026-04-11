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
const MAX_PREPLAN_STEPS: usize = 100;
const MAX_PREPLAN_LOG_CHARS: usize = 20000;
const LARGE_TRUNCATION_MIN_LINES: usize = 50;

fn ensure_not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
        bail!("edit_file interrupted by user");
    }
    Ok(())
}

fn log_stage(log: Option<&SessionLog>, path_str: &str, stage: &str) {
    if let Some(log) = log {
        log.tool_stage("edit_file", &format!("{path_str} {stage}"));
    }
}

fn log_debug(log: Option<&SessionLog>, path_str: &str, detail: &str) {
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
        Ok(PreplanResult::Applied(candidate_result)) => {
            std::fs::write(&path, &candidate_result.content)?;
            Ok(ToolResult::ok(candidate_result.message))
        }
        Ok(PreplanResult::NoChanges) => Ok(ToolResult::ok(format!(
            "No changes needed in {path_str} for task: {task}"
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
        Err(e) => Ok(ToolResult::err(format!(
            "edit_file failed: patch was not applied.\nReason: {}\n",
            e
        ))),
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
struct DroppedStep {
    step: EditPlanStep,
    reason: String,
}

/// What one planning attempt produced. The model can return a concrete
/// edit plan, or at any phase ask for clarification via
/// `NEEDS_CLARIFICATION: <question>` when the task is too vague or
/// contradicts the file to execute without guessing. Clarification
/// requests short-circuit the entire repair retry loop — the whole point
/// is to stop burning attempts on guesses.
///
/// `Steps` carries `dropped` so the executor can report overlapping
/// steps as failed steps in the per-step output instead of as opaque
/// warnings.
enum PreplanOutcome {
    Steps {
        steps: Vec<EditPlanStep>,
        dropped: Vec<DroppedStep>,
    },
    /// The pre-plan model decided the task is too ambiguous, under-specified,
    /// or contradictory to act on. Carries the model's question back to the
    /// caller so the outer agent can rephrase or split the task.
    NeedsClarification(String),
    /// The model explicitly emitted the `NO_CHANGES` sentinel, meaning the
    /// task is already satisfied in the current file state. This is the
    /// legitimate "nothing to do" signal, distinct from an empty response
    /// (which we now treat as a transient failure worth retrying).
    NothingToDo,
    /// The model returned empty or whitespace-only text. This used to
    /// collapse to `Steps { steps: [], dropped: [] }` and exit the retry
    /// loop, but bench logs show it's usually a transient pathology
    /// (stalled inference, template misfire) that a retry with feedback
    /// can recover from. Treat it as a failure and let the outer loop
    /// re-prompt via `RepairContext`.
    EmptyResponse,
    /// The model emitted something that `parse_edit_plan` rejected
    /// (unknown token, missing `OLD:` after we removed the no-OLD
    /// shortcut, malformed SCOPE, etc). Carry the parser's error
    /// verbatim so the repair prompt can tell the model what to fix.
    ParseError(String),
}

/// What the whole pre-plan retry loop produced. Distinguishes "applied
/// edits", "no edits needed", and "model needs the task clarified before
/// it can act" so the caller can render an appropriate tool result for
/// each.
enum PreplanResult {
    Applied(SplitResult),
    NoChanges,
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

/// Structured information passed to `request_preplan_steps` when running
/// a repair attempt. Carries enough context for the planner to reason
/// about *why* the previous plan failed and what state the file is in
/// now, instead of seeing only an opaque error string.
struct RepairContext {
    previous_plan: Vec<EditPlanStep>,
    completed_steps: Vec<EditPlanStep>,
    failed_step: Option<EditPlanStep>,
    failure_reason: String,
    /// When the failure was an LSP regression, carries the structured
    /// diagnostic info so `format_repair_context` can render post-edit
    /// snippets around each error instead of a truncated blob.
    lsp_regression: Option<LspRegression>,
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
    // `progress_log` is the agent-facing trail prepended to a successful
    // result when the pre-plan needed more than one attempt. We keep it
    // intentionally terse — one line per failed attempt — because the
    // verbose per-step trail and the raw plan dump are not useful to the
    // outer agent (they never see the inner plan otherwise). The full
    // failure detail still feeds the *inner* model on retry through
    // `RepairContext`, which is built independently below.
    let mut progress_log = String::new();
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
        // The label is only used for internal logging (`log_stage` /
        // `log_debug`) and for the inner model's repair feedback prompt.
        // It is intentionally NOT placed in the agent-facing message.
        let label = if repair_context.is_some() {
            format!("Pre-plan repair attempt {attempt}")
        } else {
            log_stage(log, path_str, "preplan:start");
            format!("Pre-plan attempt {attempt}")
        };

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
            PreplanOutcome::Steps { steps, dropped } => (steps, dropped),
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
            PreplanOutcome::NothingToDo => {
                // Explicit `NO_CHANGES` sentinel — the legitimate "task is
                // already satisfied" signal. Fall through to the existing
                // empty-steps convergence path below (which handles both
                // "nothing changed yet → NoChanges" and "we already applied
                // edits in an earlier attempt → converged" cases).
                log_debug(
                    log,
                    path_str,
                    &format!("preplan:nothing_to_do attempt={attempt}"),
                );
                (Vec::new(), Vec::new())
            }
            PreplanOutcome::EmptyResponse => {
                // Transient pathology (stalled inference, template misfire,
                // empty streaming completion). Historically this collapsed
                // to an empty plan and exited the retry loop, wasting the
                // whole edit_file invocation on one bad response. Now we
                // treat it as a failure and feed the model explicit
                // guidance via `RepairContext`, letting the outer
                // MAX_PLAN_ATTEMPTS loop re-prompt.
                log_debug(
                    log,
                    path_str,
                    &format!("preplan:empty_response attempt={attempt}; will retry"),
                );
                progress_log.push_str(&format!(
                    "{label}; model returned empty response, retrying\n"
                ));
                repair_context = Some(RepairContext {
                    previous_plan: Vec::new(),
                    completed_steps: Vec::new(),
                    failed_step: None,
                    failure_reason: String::from(
                        "The previous attempt returned an empty response, which is not a valid output. \
                         Either emit valid edit-plan blocks for the changes needed, \
                         or output exactly `NO_CHANGES` on its own line if the task is already satisfied, \
                         or output `NEEDS_CLARIFICATION: <question>` if the task is too vague to act on. \
                         Do not return an empty response.",
                    ),
                    lsp_regression: None,
                });
                continue;
            }
            PreplanOutcome::ParseError(reason) => {
                // The model emitted something the plan parser couldn't
                // accept — most commonly the now-removed OLD-less
                // LITERAL_REPLACE shortcut. Feed the parser's exact
                // error back through the repair prompt so the next
                // attempt can correct it.
                log_debug(
                    log,
                    path_str,
                    &format!(
                        "preplan:parse_error attempt={attempt}; will retry reason={}",
                        truncate_multiline(&reason, 500)
                    ),
                );
                progress_log.push_str(&format!(
                    "{label}; plan failed to parse, retrying\n"
                ));
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
                });
                continue;
            }
        };

        if steps.is_empty() && dropped.is_empty() {
            log_debug(log, path_str, "preplan:return_no_steps");
            if current == original {
                return Ok(PreplanResult::NoChanges);
            }

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
                Ok(validation_note) => {
                    let mut message = String::new();
                    if !progress_log.is_empty() {
                        message.push_str(&progress_log);
                    }
                    message.push_str(&format!(
                        "✓ via pre-plan: converged after {attempt} planning attempt(s), applied edits to {path_str} ({} lines)\n",
                        current.lines().count()
                    ));
                    if let Some(note) = validation_note {
                        message.push_str(&note);
                        message.push('\n');
                    }
                    return Ok(PreplanResult::Applied(SplitResult {
                        content: current,
                        message,
                    }));
                }
                Err(ValidationError::Other(e)) => return Err(e),
                Err(ValidationError::LspRegression(regression)) => {
                    // The planner said "nothing more to do" but the
                    // accumulated state is still broken. Instead of
                    // bailing, loop back with a repair context that
                    // carries the structured LSP errors so the next
                    // attempt can patch the regression.
                    let summary = ValidationError::LspRegression(regression.clone()).summary();
                    log_debug(
                        log,
                        path_str,
                        &format!(
                            "preplan:converged_validation_failed:{attempt} {}",
                            truncate_multiline(&summary, 2000)
                        ),
                    );
                    progress_log.push_str(&format!(
                        "Pre-plan attempt {attempt} converged but validation failed: {}\n",
                        truncate_multiline(&summary, 400)
                    ));

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
                        progress_log.push_str(&format!(
                            "Pre-plan attempt {attempt} improved errors to {new_count}; granted extra retry ({}/{})\n",
                            extra_attempts_granted, MAX_EXTRA_ATTEMPTS
                        ));
                    }

                    repair_context = Some(RepairContext {
                        previous_plan: Vec::new(),
                        completed_steps: Vec::new(),
                        failed_step: None,
                        failure_reason: summary,
                        lsp_regression: Some(regression),
                    });
                    continue;
                }
            }
        }

        let kept_count = steps.len();
        let dropped_count = dropped.len();
        let total_planned = kept_count + dropped_count;
        // Log the parsed plan so it shows up in the session log for
        // debugging, but do NOT inject it into the agent-facing message.
        log_debug(log, path_str, &format_preplan_log(&label, &steps));
        // The agent-facing message starts empty; per-attempt failures and
        // the final summary are appended by `execute_planned_steps`.
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
                if !progress_log.is_empty() {
                    let mut message = progress_log;
                    message.push_str(&result.message);
                    return Ok(PreplanResult::Applied(SplitResult {
                        content: result.content,
                        message,
                    }));
                }
                return Ok(PreplanResult::Applied(result));
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
                // Terse, one-line trail. Full per-step detail still goes
                // to the session log via `e.message` shouldn't bloat the
                // agent's view; the inner model gets the structured
                // failure via `RepairContext` below.
                progress_log.push_str(&format!(
                    "Pre-plan attempt {attempt} failed: {}\n",
                    truncate_multiline(&e.error, 400)
                ));

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
                        progress_log.push_str(&format!(
                            "Pre-plan attempt {attempt} improved errors to {new_count}; granted extra retry ({}/{})\n",
                            extra_attempts_granted, MAX_EXTRA_ATTEMPTS
                        ));
                    }
                }

                repair_context = Some(RepairContext {
                    previous_plan: steps,
                    completed_steps: e.completed_steps,
                    failed_step: e.failed_step,
                    failure_reason: e.error,
                    lsp_regression: e.lsp_regression,
                });
                current = e.current_content;
            }
        }
    }

    let mut message = progress_log;
    if let Some(ctx) = repair_context {
        message.push_str(&format!(
            "Pre-plan exhausted after {attempt_budget} attempt(s); last failure: {}\n",
            ctx.failure_reason
        ));
        bail!(message);
    }
    Ok(PreplanResult::NoChanges)
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
                        // Before falling back to smart-edit, try to rescue
                        // the common pathology where the planner hallucinated
                        // trivial whitespace in the OLD block (extra space,
                        // tab vs spaces, stray newline). The whitespace-
                        // tolerant fallback locates a candidate in scope and
                        // asks the planner via a single round-trip whether
                        // that candidate is the intended target.
                        let outcome = try_whitespace_tolerant_replace(
                            path_str,
                            &current,
                            *scope_start,
                            *scope_end,
                            old,
                            new,
                            router,
                            cancelled,
                            log,
                        )
                        .await;

                        match outcome {
                            WhitespaceOutcome::Applied(new_content) => {
                                current = new_content;
                                completed_count += 1;
                                completed_records.push(step.clone());
                                message.push_str(&format!(
                                    "literal-replace L{scope_start}-L{scope_end} matched after whitespace normalization\n"
                                ));
                                continue;
                            }
                            WhitespaceOutcome::Rejected => {
                                // Planner confirmed the candidate is wrong.
                                // Don't burn time on smart-edit (which won't
                                // have any new information); bubble straight
                                // up to plan-level repair so the planner
                                // re-emits a correct OLD with full context.
                                return Err(PlannedExecutionFailure {
                                    current_content: current.clone(),
                                    message: format!(
                                        "{message}Pre-plan step {} literal L{}-L{} failed after {} completed step(s): {literal_error}; whitespace-tolerant candidate rejected by planner\n",
                                        idx + 1,
                                        scope_start,
                                        scope_end,
                                        completed_count
                                    ),
                                    error: format!(
                                        "step {} literal replace failed: {literal_error}; whitespace-tolerant candidate rejected by planner",
                                        idx + 1
                                    ),
                                    completed_steps: completed_records.clone(),
                                    failed_step: Some(step.clone()),
                                    lsp_regression: None,
                                });
                            }
                            WhitespaceOutcome::NoCandidate => {
                                // Fall through to existing smart-edit fallback.
                            }
                        }

                        let fallback_task = format!(
                            "The planned exact literal replacement failed: {literal_error}\n\
                             Apply the same intended change manually within this region only.\n\n\
                             Intended OLD:\n{}\n\n\
                             Intended NEW:\n{}",
                            old.join("\n"),
                            new.join("\n")
                        );
                        let region = EditRegion {
                            start: *scope_start,
                            end: *scope_end,
                            task: fallback_task,
                        };
                        let (candidate, count) = execute_smart_step(
                            path_str,
                            &region.task,
                            &current,
                            router,
                            lsp_validation,
                            &region,
                            false,
                            cancelled,
                            log,
                        )
                        .await
                        .map_err(|e| PlannedExecutionFailure {
                            current_content: current.clone(),
                            message: format!(
                                "{message}Pre-plan step {} literal L{}-L{} failed after {} completed step(s): {literal_error}; smart fallback failed: {e}\n",
                                idx + 1,
                                scope_start,
                                scope_end,
                                completed_count
                            ),
                            error: format!(
                                "step {} literal replace failed: {literal_error}; smart fallback failed: {e}",
                                idx + 1
                            ),
                            completed_steps: completed_records.clone(),
                            failed_step: Some(step.clone()),
                            lsp_regression: None,
                        })?;
                        current = candidate;
                        completed_count += 1;
                        completed_records.push(step.clone());
                        // Smart fallback IS interesting — the planner's
                        // exact OLD didn't match and we had to ask the
                        // smart executor to redo the region. Surface it.
                        message.push_str(&format!(
                            "literal-replace L{scope_start}-L{scope_end} fell back to smart edit ({count} op(s))\n"
                        ));
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
        bail!("smart fallback returned NO_CHANGES");
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
                if let (Ok(start), Ok(end)) =
                    (start_s.trim().parse::<usize>(), end_s.trim().parse::<usize>())
                {
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

    PreplanWindowResponse { notes, commands, clarification }
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
        bail!("READ range L{start}-L{end} outside file with {} lines", lines.len());
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
                        extra_context.push_str(&format!(
                            "(READ {start}-{end} failed: {e})\n\n",
                        ));
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

        log_stage(log, path_str, &format!("patch:window:{}-{}", start + 1, end));
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
    let feedback_block = repair
        .map(format_repair_context)
        .unwrap_or_default();
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
                notes.iter().map(|note| format!("- {note}")).collect::<Vec<_>>().join("\n")
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
        let prompt = format!(
            "File: {path_str}\n\
             Task: {task}\n\n\
             {feedback_block}\
             {notes_block}\
             {pending_block}\
             Slice {current_slice}/{total_slices}, lines {start_line}-{end_line} of {total_lines}.\n\n\
             Output zero or more of these lines about landmarks the planner will need:\n\
             NOTE <fact with line number> — function/struct spans, signatures the task touches, the line where a relevant block starts.\n\n\
             You may also request extra context before finalize with:\n\
             SEARCH: <exact text to find>\n\
             READ: <start>-<end>\n\n\
             SEARCH/READ commands are collected across ALL slices and batch-executed before the finalize phase — their results will reach the planner. Don't repeat commands; the finalize phase already sees every result.\n\n\
             Reference line numbers from the slice. Skip vague observations and anything the planner can derive itself.\n\n\
             If the task is genuinely too vague, underspecified, or contradictory to execute — e.g. it names no target, it asks for an outcome the file can't support, or it requires information only the outer agent has — output exactly one line:\n\
             NEEDS_CLARIFICATION: <one specific question>\n\
             Use this only when you would otherwise have to guess. Prefer specific questions over vague ones. Do NOT use it for tasks where the target doesn't exist yet — that's what the edit is for; plan the edit that creates it.\n\n\
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
            &format!("preplan:window:{}/{}:{}-{}", idx + 1, windows.len(), start + 1, end),
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
                notes.iter().map(|note| format!("- {note}")).collect::<Vec<_>>().join("\n")
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
        "Plan the edit. You see the full file above — line numbers in your plan must match it exactly."
    } else {
        "Plan the edit. You only see the notes and inspection results above — line numbers in your plan must match the real file."
    };
    let finalize_prompt = format!(
        "File: {path_str} ({total_lines} lines)\n\
         Task: {task}\n\n\
         {feedback_block}\
         {file_view_block}\
         {view_note}\n\n\
         Up to {MAX_PREPLAN_STEPS} non-overlapping steps, each covering at most 5 edit sites. Steps must not share any line, including endpoints — L10-L20 and L20-L30 overlap.\n\
         Every step must change something. Do not emit a LITERAL_REPLACE whose NEW is identical to its OLD, and do not emit a SMART_EDIT whose task is to verify a region is unchanged or to keep it as-is — just leave those regions out of the plan.\n\
         Use LITERAL_REPLACE when you have the OLD text verbatim from an inspection result and OLD/NEW each span ≤ {max_literal_lines} lines. Otherwise use SMART_EDIT — its execution phase will see the region content.\n\
         Never use LITERAL_REPLACE for whole functions, impl blocks, modules, or test cases.\n\n\
         If the task is already satisfied in the current file, output exactly `NO_CHANGES` on its own line and nothing else. Do NOT return an empty response — empty responses are treated as a failure and retried.\n\n\
         Output only these blocks, nothing else:\n\n\
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
         If the task is genuinely too vague, underspecified, or contradictory to plan without guessing — e.g. it names no concrete target, it asks for an outcome the file can't support, or it requires information only the outer agent has — output exactly one line:\n\
         NEEDS_CLARIFICATION: <one specific question>\n\
         Use this only when you would otherwise have to guess. Do NOT use it just because a target the task wants to add doesn't exist yet — that's what the edit is for; plan the edit that creates it. Prefer specific questions over vague ones."
    );

    let request = ChatRequest {
        messages: vec![
            Message::system(
                "Final planning phase. Output only edit-plan blocks, or `NO_CHANGES` if the task is already satisfied, or `NEEDS_CLARIFICATION: <question>` if the task is too vague to act on. No markdown. Empty responses are not valid and will be retried.",
            ),
            Message::user(&finalize_prompt),
        ],
        tools: None,
        tool_choice: None,
    };

    log_stage(log, path_str, &format!("preplan:finalize:file:1-{total_lines}"));
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

    // Explicit NO_CHANGES sentinel — the legitimate "task is already
    // satisfied" signal. Route to `NothingToDo` so the caller can
    // converge cleanly without retrying.
    if looks_like_no_changes(text) {
        log_debug(
            log,
            path_str,
            &format!("preplan:finalize:file:1-{total_lines} nothing_to_do"),
        );
        return Ok(PreplanOutcome::NothingToDo);
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
    Ok(PreplanOutcome::Steps { steps, dropped })
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatchOp {
    InsertBefore {
        line: usize,
        content: Vec<String>,
    },
    InsertAfter {
        line: usize,
        content: Vec<String>,
    },
    ReplaceAt {
        start: usize,
        old: Vec<String>,
        new: Vec<String>,
    },
    DeleteAt {
        start: usize,
        old: Vec<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditRegion {
    pub start: usize,
    pub end: usize,
    pub task: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditPlanStep {
    SmartEdit(EditRegion),
    LiteralReplace {
        scope_start: usize,
        scope_end: usize,
        all: bool,
        old: Vec<String>,
        new: Vec<String>,
    },
}

impl EditPlanStep {
    fn start_line(&self) -> usize {
        match self {
            Self::SmartEdit(region) => region.start,
            Self::LiteralReplace { scope_start, .. } => *scope_start,
        }
    }

    fn end_line(&self) -> usize {
        match self {
            Self::SmartEdit(region) => region.end,
            Self::LiteralReplace { scope_end, .. } => *scope_end,
        }
    }
}

fn format_edit_plan_steps(steps: &[EditPlanStep]) -> String {
    let mut out = String::new();
    for step in steps {
        match step {
            EditPlanStep::SmartEdit(region) => {
                out.push_str("SMART_EDIT\n");
                out.push_str(&format!("REGION {} {}\n", region.start, region.end));
                out.push_str(&format!("TASK: {}\n\n", region.task));
            }
            EditPlanStep::LiteralReplace {
                scope_start,
                scope_end,
                all,
                old,
                new,
            } => {
                out.push_str("LITERAL_REPLACE\n");
                out.push_str(&format!("SCOPE {scope_start} {scope_end}\n"));
                out.push_str(&format!("ALL {all}\n"));
                out.push_str("OLD:\n");
                out.push_str(&old.join("\n"));
                out.push_str("\nEND_OLD\nNEW:\n");
                out.push_str(&new.join("\n"));
                out.push_str("\nEND_NEW\n\n");
            }
        }
    }
    out
}

fn format_preplan_log(label: &str, steps: &[EditPlanStep]) -> String {
    let plan = format_edit_plan_steps(steps);
    let plan = truncate_multiline(&plan, MAX_PREPLAN_LOG_CHARS);
    format!("Raw {label} ({} step(s), parsed):\n{plan}\n", steps.len())
}

/// Detect a finalize-phase `NO_CHANGES` sentinel — the explicit signal
/// from the planner that the task is already satisfied. We tolerate
/// surrounding whitespace and code fences so a model that wraps its
/// answer in ```` ``` ```` still works.
fn looks_like_no_changes(text: &str) -> bool {
    let unfenced = strip_code_fences(text);
    unfenced.trim() == "NO_CHANGES"
}

/// Detect a `NEEDS_CLARIFICATION: <question>` sentinel. Returns
/// `Some(question)` if the response opens with `NEEDS_CLARIFICATION`
/// (optionally followed by `:` and a question), and `None` otherwise. The
/// question is trimmed and may be empty if the model omits one — the
/// caller substitutes a placeholder at render time.
///
/// Guards against lookalikes like `NEEDS_CLARIFICATIONS` or
/// `NEEDS_CLARIFICATIONAL` by requiring end-of-string, whitespace, or `:`
/// immediately after the keyword.
fn parse_needs_clarification(text: &str) -> Option<String> {
    let trimmed = text.trim_start();
    let rest = trimmed.strip_prefix("NEEDS_CLARIFICATION")?;
    let after = match rest.chars().next() {
        None => "",
        Some(':') => &rest[1..],
        Some(c) if c.is_whitespace() => rest,
        Some(_) => return None,
    };
    // If there's a trailing newline after the question, keep only the
    // first line — the sentinel is single-line by contract.
    let first_line = after.lines().next().unwrap_or("");
    Some(first_line.trim().to_string())
}

/// Render a `RepairContext` into a planner-facing block. Used at the top
/// of both the windowed pre-plan prompts and the finalize prompt so the
/// model can see exactly what the previous attempt did, what survived,
/// what failed, and the failure reason.
fn format_repair_context(ctx: &RepairContext) -> String {
    let previous_plan = if ctx.previous_plan.is_empty() {
        "(empty)\n".to_string()
    } else {
        format_edit_plan_steps(&ctx.previous_plan)
    };

    let completed = if ctx.completed_steps.is_empty() {
        "(none — the first step failed, file is unchanged from the initial state)\n".to_string()
    } else {
        format_edit_plan_steps(&ctx.completed_steps)
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
        "A previous edit plan was attempted and failed. Use the structured information below to plan a better recovery.\n\n\
         Previous edit plan (as tried):\n{previous_plan}\n\
         Steps that succeeded and have ALREADY been applied to the file shown below:\n{completed}\n\
         Step that FAILED:\n{failed}\n\
         Failure reason:\n{failure_block}\n\n\
         Plan the remaining work needed to complete the original task against the current file content. \
         If the task already appears complete, return NO_CHANGES.\n\n",
    )
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

fn truncate_multiline(text: &str, max_chars: usize) -> String {
    let char_count = text.chars().count();
    if char_count <= max_chars {
        return text.to_string();
    }

    let truncated: String = text.chars().take(max_chars).collect();
    format!("{truncated}\n...({char_count} chars total, truncated)\n")
}

/// Strip a single outer ```...``` markdown fence from `text`, if present.
///
/// The finalize prompt explicitly says "no markdown", but smaller models
/// sometimes wrap their entire structured response in a ``` fence anyway
/// (often with a language tag like ```rust). When that happens the parser
/// fails on the leading backticks even though the body is otherwise
/// well-formed. This helper detects that one common shape and returns the
/// inner body so parsing can proceed.
///
/// Behaviour:
/// - If the leading non-whitespace characters are not ``` we return the
///   input unchanged (no fence to strip).
/// - If a leading ``` is present we drop everything from the start through
///   the first newline (i.e. the fence line, including any language tag).
/// - If there is *also* a trailing ``` (anywhere later in the text) we drop
///   it and everything after it. If the response was truncated mid-output
///   and never closed the fence, we still strip the opener and keep the
///   rest — that gives us the best shot at parsing a partial reply.
fn strip_code_fences(text: &str) -> String {
    let trimmed = text.trim_start();
    let Some(rest) = trimmed.strip_prefix("```") else {
        return text.to_string();
    };
    let Some(newline_pos) = rest.find('\n') else {
        return text.to_string();
    };
    let body = &rest[newline_pos + 1..];
    let inner = match body.rfind("```") {
        Some(pos) => &body[..pos],
        None => body,
    };
    inner.to_string()
}

pub fn parse_edit_plan(text: &str) -> Result<Vec<EditPlanStep>> {
    // Empty input parses to zero steps. The finalize caller now detects
    // empty responses and `NO_CHANGES` up-front and routes them into
    // dedicated `PreplanOutcome` variants, so this code path is only
    // reached when some caller passes a body that turned out to be empty
    // (e.g. after fence stripping) or on a stray NO_CHANGES token mixed
    // in with real steps — we tolerate those below rather than failing
    // the parse.
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }

    let unfenced = strip_code_fences(text);
    let text = unfenced.as_str();
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }

    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;
    let mut steps = Vec::new();

    while i < lines.len() {
        let line = lines[i];
        if line.trim().is_empty() {
            i += 1;
            continue;
        }

        // Tolerate a stray NO_CHANGES token from older prompts or
        // confused models — treat it as a no-op separator.
        if line.trim() == "NO_CHANGES" {
            i += 1;
            continue;
        }

        // Tolerate a stray END terminator. The current DSL no longer
        // uses END as a step terminator (END_OLD/END_NEW are enough),
        // but older models trained on the prior format may still emit
        // it. Skip it instead of failing the parse.
        if line.trim() == "END" {
            i += 1;
            continue;
        }

        if line == "SMART_EDIT" {
            i += 1;
            let (region, next) = parse_region_at(&lines, i)?;
            i = next;
            steps.push(EditPlanStep::SmartEdit(region));
            continue;
        }

        if line.starts_with("REGION ") {
            let (region, next) = parse_region_at(&lines, i)?;
            i = next;
            steps.push(EditPlanStep::SmartEdit(region));
            continue;
        }

        if line == "LITERAL_REPLACE" {
            i += 1;
            let scope_line = lines
                .get(i)
                .ok_or_else(|| anyhow::anyhow!("missing SCOPE line for literal replace"))?;
            let Some(rest) = scope_line.strip_prefix("SCOPE ") else {
                bail!("expected SCOPE line but found '{scope_line}'");
            };
            let (scope_start, scope_end) = parse_two_line_numbers(rest, scope_line, "scope")?;
            i += 1;

            let all_line = lines
                .get(i)
                .ok_or_else(|| anyhow::anyhow!("missing ALL line for literal replace"))?;
            let all = match all_line.strip_prefix("ALL ") {
                Some("true") => true,
                Some("false") => false,
                Some(other) => bail!("invalid ALL value '{other}' in line '{all_line}'"),
                None => bail!("expected ALL line but found '{all_line}'"),
            };
            i += 1;

            // Reject the old "NEW: without OLD:" shortcut. It let the
            // model regenerate the replacement text without cross-checking
            // the scope it was editing, which was the root cause of most
            // hallucination failures (inventing `&self` on a free fn,
            // dropping `pub struct Foo {` off the start of a scope, etc).
            // OLD: is now unconditional so we can verify at parse/apply
            // time that the model knows what it's replacing.
            let mut peek = i;
            while peek < lines.len() && lines[peek].trim().is_empty() {
                peek += 1;
            }
            let head = lines.get(peek).copied().unwrap_or("");
            if head == "NEW:" {
                bail!(
                    "LITERAL_REPLACE now requires an OLD: block — copy the exact current text of L{scope_start}-L{scope_end} into OLD:, then put the replacement in NEW:. If the region is too large to echo verbatim, use SMART_EDIT instead (its execution phase will see the region content)."
                );
            }

            expect_line(&lines, i, "OLD:")?;
            i += 1;
            let (old, next) = collect_until(&lines, i, "END_OLD")?;
            if old.is_empty() {
                bail!("literal OLD block must not be empty");
            }
            if old.iter().all(|line| line.trim().is_empty()) {
                bail!("literal OLD block must contain non-whitespace text");
            }
            i = next + 1;

            expect_line(&lines, i, "NEW:")?;
            i += 1;
            let (new, next) = collect_until(&lines, i, "END_NEW")?;
            i = next + 1;

            steps.push(EditPlanStep::LiteralReplace {
                scope_start,
                scope_end,
                all,
                old,
                new,
            });
            continue;
        }

        bail!("unexpected text in edit plan: {line}");
    }

    if steps.len() > MAX_PREPLAN_STEPS {
        bail!(
            "edit plan returned {} steps, maximum is {MAX_PREPLAN_STEPS}",
            steps.len()
        );
    }
    // Overlap is *not* a parse error: the planner caller resolves overlaps
    // by keeping the first step in source order and reporting the rest as
    // dropped steps in the per-step output. See `partition_overlapping_steps`.
    Ok(steps)
}

fn parse_region_at(lines: &[&str], idx: usize) -> Result<(EditRegion, usize)> {
    let line = lines
        .get(idx)
        .ok_or_else(|| anyhow::anyhow!("missing REGION header"))?;
    let Some(rest) = line.strip_prefix("REGION ") else {
        bail!("unexpected text in region plan: {line}");
    };
    let rest = rest.trim();
    let (start, end) = if let Some((start, end)) = rest.split_once('-') {
        let start = start.trim().parse::<usize>().map_err(|e| {
            anyhow::anyhow!("invalid region start '{start}' in header '{line}': {e}")
        })?;
        let end = end
            .trim()
            .parse::<usize>()
            .map_err(|e| anyhow::anyhow!("invalid region end '{end}' in header '{line}': {e}"))?;
        (start, end)
    } else {
        let parts: Vec<_> = rest.split_whitespace().collect();
        match parts.as_slice() {
            [single] => {
                let value = single.parse::<usize>().map_err(|e| {
                    anyhow::anyhow!("invalid region line '{single}' in header '{line}': {e}")
                })?;
                (value, value)
            }
            [_, _] => parse_two_line_numbers(rest, line, "region")?,
            _ => bail!("invalid REGION header: {line}"),
        }
    };

    let task_line = lines
        .get(idx + 1)
        .ok_or_else(|| anyhow::anyhow!("missing TASK line for region"))?;
    let task = task_line
        .strip_prefix("TASK:")
        .ok_or_else(|| anyhow::anyhow!("expected TASK line but found '{task_line}'"))?
        .trim()
        .to_string();
    if task.is_empty() {
        bail!("region task must not be empty");
    }

    Ok((EditRegion { start, end, task }, idx + 2))
}

fn parse_two_line_numbers(rest: &str, header: &str, label: &str) -> Result<(usize, usize)> {
    let mut parts = rest.split_whitespace();
    let start_token = parts
        .next()
        .ok_or_else(|| anyhow::anyhow!("missing {label} start in header: {header}"))?;
    let start = start_token.parse::<usize>().map_err(|e| {
        anyhow::anyhow!("invalid {label} start '{start_token}' in header '{header}': {e}")
    })?;
    let end_token = parts
        .next()
        .ok_or_else(|| anyhow::anyhow!("missing {label} end in header: {header}"))?;
    let end = end_token.parse::<usize>().map_err(|e| {
        anyhow::anyhow!("invalid {label} end '{end_token}' in header '{header}': {e}")
    })?;
    if parts.next().is_some() {
        bail!("too many fields in {label} header: {header}");
    }
    if start == 0 || end < start {
        bail!("invalid {label} L{start}-L{end}");
    }
    Ok((start, end))
}

fn reject_overlapping_regions(regions: &[EditRegion]) -> Result<()> {
    let mut sorted: Vec<&EditRegion> = regions.iter().collect();
    sorted.sort_unstable_by(|a, b| a.start.cmp(&b.start).then_with(|| a.end.cmp(&b.end)));

    for pair in sorted.windows(2) {
        let prev = pair[0];
        let next = pair[1];
        if next.start <= prev.end {
            bail!(
                "region plan has overlapping regions: L{}-L{} overlaps L{}-L{}",
                prev.start,
                prev.end,
                next.start,
                next.end
            );
        }
    }
    Ok(())
}

/// Partition planned steps into "kept" (first occurrence wins, by source
/// order) and "dropped" (overlaps an earlier step). The kept set keeps
/// original emission order so downstream logging stays intuitive. The
/// dropped set carries the original step plus a human-readable reason
/// pointing at the kept step that caused the conflict, so the executor
/// can report each one as a failed step in the per-step output.
fn partition_overlapping_steps(steps: Vec<EditPlanStep>) -> (Vec<EditPlanStep>, Vec<DroppedStep>) {
    // Sort by source position so "first wins" is deterministic and matches
    // file order rather than emission order.
    let mut indexed: Vec<(usize, EditPlanStep)> = steps.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| {
        a.1.start_line()
            .cmp(&b.1.start_line())
            .then_with(|| a.1.end_line().cmp(&b.1.end_line()))
            .then_with(|| a.0.cmp(&b.0))
    });

    let mut kept: Vec<(usize, EditPlanStep)> = Vec::with_capacity(indexed.len());
    let mut dropped: Vec<(usize, DroppedStep)> = Vec::new();
    for (orig_idx, step) in indexed {
        let conflict = kept.iter().find(|(_, prev)| {
            // [a,b] overlaps [c,d] iff a <= d && c <= b
            step.start_line() <= prev.end_line() && prev.start_line() <= step.end_line()
        });
        if let Some((_, prev)) = conflict {
            let reason = format!(
                "overlaps earlier step L{}-L{}",
                prev.start_line(),
                prev.end_line()
            );
            dropped.push((orig_idx, DroppedStep { step, reason }));
        } else {
            kept.push((orig_idx, step));
        }
    }

    // Restore original emission order so downstream "in order" logging stays
    // intuitive for the agent.
    kept.sort_by_key(|(idx, _)| *idx);
    dropped.sort_by_key(|(idx, _)| *idx);
    (
        kept.into_iter().map(|(_, s)| s).collect(),
        dropped.into_iter().map(|(_, d)| d).collect(),
    )
}

/// Parse strict patch DSL blocks.
pub fn parse_patch(text: &str) -> Result<Vec<PatchOp>> {
    let unfenced = strip_code_fences(text);
    let text = unfenced.as_str();
    if text.trim() == "NO_CHANGES" {
        return Ok(Vec::new());
    }
    if text.trim().is_empty() {
        bail!("empty patch");
    }

    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;
    let mut ops = Vec::new();

    while i < lines.len() {
        let line = lines[i];
        if line.trim().is_empty() {
            i += 1;
            continue;
        }

        if let Some(rest) = line.strip_prefix("INSERT_BEFORE ") {
            let line_num = parse_line_number(rest)?;
            i += 1;
            expect_line(&lines, i, "CONTENT:")?;
            i += 1;
            let (content, next) = collect_until(&lines, i, "END")?;
            i = next + 1;
            ops.push(PatchOp::InsertBefore {
                line: line_num,
                content,
            });
        } else if let Some(rest) = line.strip_prefix("INSERT_AFTER ") {
            let line_num = parse_line_number(rest)?;
            i += 1;
            expect_line(&lines, i, "CONTENT:")?;
            i += 1;
            let (content, next) = collect_until(&lines, i, "END")?;
            i = next + 1;
            ops.push(PatchOp::InsertAfter {
                line: line_num,
                content,
            });
        } else if let Some(rest) = line.strip_prefix("REPLACE_AT ") {
            let start = parse_line_number(rest)?;
            i += 1;
            expect_line(&lines, i, "OLD:")?;
            i += 1;
            let (old, next) = collect_until(&lines, i, "END_OLD")?;
            i = next + 1;
            expect_line(&lines, i, "NEW:")?;
            i += 1;
            let (new, next) = collect_until(&lines, i, "END_NEW")?;
            i = next + 1;
            ops.push(PatchOp::ReplaceAt { start, old, new });
        } else if let Some(rest) = line.strip_prefix("DELETE_AT ") {
            let start = parse_line_number(rest)?;
            i += 1;
            expect_line(&lines, i, "OLD:")?;
            i += 1;
            let (old, next) = collect_until(&lines, i, "END_OLD")?;
            i = next + 1;
            ops.push(PatchOp::DeleteAt { start, old });
        } else {
            bail!("unexpected text in patch: {line}");
        }
    }

    Ok(ops)
}

fn parse_line_number(text: &str) -> Result<usize> {
    let raw = text.trim();
    let line = raw
        .parse::<usize>()
        .map_err(|e| anyhow::anyhow!("invalid line number '{raw}': {e}"))?;
    if line == 0 {
        bail!("line numbers are 1-based");
    }
    Ok(line)
}

fn expect_line(lines: &[&str], idx: usize, expected: &str) -> Result<()> {
    match lines.get(idx) {
        Some(line) if *line == expected => Ok(()),
        Some(line) => bail!("expected '{expected}' but found '{line}'"),
        None => bail!("expected '{expected}' but reached end of patch"),
    }
}

fn collect_until(lines: &[&str], start: usize, sentinel: &str) -> Result<(Vec<String>, usize)> {
    let mut collected = Vec::new();
    for (idx, line) in lines.iter().enumerate().skip(start) {
        if *line == sentinel {
            return Ok((collected, idx));
        }
        collected.push((*line).to_string());
    }
    bail!("missing sentinel '{sentinel}'");
}

/// Apply all operations to memory only. If any operation fails, returns an
/// error and the original file on disk remains untouched.
pub fn apply_patch_dry_run(content: &str, ops: &[PatchOp]) -> Result<String> {
    let lines: Vec<String> = content.lines().map(str::to_string).collect();
    let resolved = resolve_ops(&lines, ops)?;
    apply_resolved_patch(content, resolved)
}

fn apply_patch_dry_run_in_region(
    content: &str,
    ops: &[PatchOp],
    start_line: usize,
    end_line: usize,
) -> Result<String> {
    let lines: Vec<String> = content.lines().map(str::to_string).collect();
    if start_line == 0 || end_line < start_line || end_line > lines.len() {
        bail!(
            "invalid edit region L{start_line}-L{end_line} for {} line file",
            lines.len()
        );
    }

    let resolved = resolve_ops(&lines, ops)?;
    let allowed_start = start_line - 1;
    let allowed_end = end_line;

    for op in &resolved {
        if op.start < allowed_start || op.end > allowed_end {
            bail!(
                "{} resolves to {}, outside allowed region L{}-L{}",
                op.label,
                display_span(op.start, op.end),
                start_line,
                end_line
            );
        }
    }

    apply_resolved_patch(content, resolved)
}

pub fn apply_literal_replace_in_scope(
    content: &str,
    scope_start: usize,
    scope_end: usize,
    old: &[String],
    new: &[String],
    all: bool,
) -> Result<(String, usize)> {
    if scope_start == 0 || scope_end < scope_start {
        bail!("invalid literal scope L{scope_start}-L{scope_end}");
    }
    if old.is_empty() {
        bail!("literal OLD block must not be empty");
    }

    let line_count = content.lines().count();
    if scope_end > line_count {
        bail!("literal scope L{scope_start}-L{scope_end} outside {line_count} line file");
    }

    let parts: Vec<&str> = content.split_inclusive('\n').collect();
    let start_byte: usize = parts[..scope_start - 1].iter().map(|part| part.len()).sum();
    let end_byte: usize = parts[..scope_end].iter().map(|part| part.len()).sum();

    let old_text = old.join("\n");
    let new_text = new.join("\n");
    let scoped = &content[start_byte..end_byte];
    let count = scoped.matches(&old_text).count();

    if all {
        if count == 0 {
            bail!(
                "literal OLD block was not found in scope L{scope_start}-L{scope_end}\nOLD block:\n{}",
                preview_block(old, None)
            );
        }
    } else if count != 1 {
        bail!(
            "literal OLD block matched {count} occurrence(s) in scope L{scope_start}-L{scope_end}; expected exactly 1"
        );
    }

    let replaced_scope = if all {
        scoped.replace(&old_text, &new_text)
    } else {
        scoped.replacen(&old_text, &new_text, 1)
    };

    let mut out = String::with_capacity(content.len() + replaced_scope.len());
    out.push_str(&content[..start_byte]);
    out.push_str(&replaced_scope);
    out.push_str(&content[end_byte..]);
    Ok((out, count))
}

/// Outcome of a whitespace-tolerant literal-replace fallback attempt.
///
/// The planner sometimes hallucinates trivial whitespace inside an OLD
/// block (extra space, tab vs spaces, stray newline). When the byte-exact
/// matcher rejects such a block we strip all whitespace from both the
/// OLD text and the scoped slice, look for the OLD as a substring, and —
/// if found — ask the planner via a single round-trip whether the
/// candidate is the intended target.
enum WhitespaceOutcome {
    /// Candidate found, planner confirmed, replacement applied. Carries
    /// the new file content ready to swap into `current`.
    Applied(String),
    /// Candidate found but the planner rejected it. The caller should
    /// bubble straight up to plan-level repair instead of running the
    /// smart-edit fallback (which has no new information to work with).
    Rejected,
    /// No whitespace-tolerant candidate exists in the scope, or the
    /// confirmation request itself failed. The caller should fall through
    /// to the existing smart-edit fallback.
    NoCandidate,
}

/// Walk `scoped` looking for the first whitespace-tolerant occurrence of
/// `old_text`. Returns the byte range in `scoped` (relative to the slice
/// start, not the whole file) of that match, or `None` if no match
/// exists.
///
/// "Whitespace-tolerant" means: strip every whitespace character from
/// both `old_text` and `scoped`, find `old_text` as a substring, then
/// map the match position back to the original byte range in `scoped`.
/// This rescues hallucinations like `</div >` (extra space inside an
/// HTML tag) and tab/space drift in indented code.
///
/// Multiple candidates: returns the first occurrence only. We could
/// disambiguate by asking the model which one it meant, but in practice
/// the first match is almost always right and the planner round-trip
/// already filters out wrong picks.
fn find_whitespace_tolerant_match(scoped: &str, old_text: &str) -> Option<(usize, usize)> {
    let old_squashed: String = old_text.chars().filter(|c| !c.is_whitespace()).collect();
    if old_squashed.is_empty() {
        return None;
    }

    // Build a parallel byte-index map so a match position in the squashed
    // string can be translated back to the byte range it occupied in the
    // original `scoped` slice. `starts[i]` and `ends[i]` are the start and
    // end byte offsets in `scoped` of the source character whose first
    // (squashed) byte is at squashed-index `i`. Multiple squashed bytes
    // share the same source range when the source character is multibyte.
    let mut squashed = String::with_capacity(scoped.len());
    let mut starts: Vec<usize> = Vec::with_capacity(scoped.len());
    let mut ends: Vec<usize> = Vec::with_capacity(scoped.len());
    for (byte_idx, c) in scoped.char_indices() {
        if c.is_whitespace() {
            continue;
        }
        let char_end = byte_idx + c.len_utf8();
        for _ in 0..c.len_utf8() {
            starts.push(byte_idx);
            ends.push(char_end);
        }
        squashed.push(c);
    }

    let match_start = squashed.find(&old_squashed)?;
    let match_end = match_start + old_squashed.len();
    if match_end == 0 || match_end > starts.len() {
        return None;
    }

    let scoped_start = starts[match_start];
    let scoped_end = ends[match_end - 1];
    Some((scoped_start, scoped_end))
}

/// Try to rescue a failed literal replace by locating a whitespace-
/// tolerant candidate in scope and asking the planner to confirm it.
/// See [`WhitespaceOutcome`] for the three possible results.
#[allow(clippy::too_many_arguments)]
async fn try_whitespace_tolerant_replace(
    path_str: &str,
    content: &str,
    scope_start: usize,
    scope_end: usize,
    old: &[String],
    new: &[String],
    router: &ModelRouter,
    cancelled: Option<&AtomicBool>,
    log: Option<&SessionLog>,
) -> WhitespaceOutcome {
    let parts: Vec<&str> = content.split_inclusive('\n').collect();
    if scope_start == 0 || scope_end < scope_start || scope_end > parts.len() {
        return WhitespaceOutcome::NoCandidate;
    }
    let scope_byte_start: usize = parts[..scope_start - 1].iter().map(|p| p.len()).sum();
    let scope_byte_end: usize = parts[..scope_end].iter().map(|p| p.len()).sum();
    let scoped = &content[scope_byte_start..scope_byte_end];
    let old_text = old.join("\n");

    let Some((rel_start, rel_end)) = find_whitespace_tolerant_match(scoped, &old_text) else {
        return WhitespaceOutcome::NoCandidate;
    };
    let actual_match = scoped[rel_start..rel_end].to_string();

    log_debug(
        log,
        path_str,
        &format!(
            "literal:whitespace_drift L{scope_start}-L{scope_end} candidate found, asking planner to confirm\n  planner OLD: {}\n  actual:      {}",
            truncate_multiline(&old_text, 400),
            truncate_multiline(&actual_match, 400)
        ),
    );

    let confirmed = match request_whitespace_drift_confirmation(
        path_str,
        scope_start,
        scope_end,
        &old_text,
        &actual_match,
        router,
        cancelled,
        log,
    )
    .await
    {
        Ok(yes) => yes,
        Err(e) => {
            // Confirmation request itself failed (network, timeout,
            // cancellation). Don't treat that as a "rejection" — fall
            // through to the existing smart-edit fallback so a transient
            // transport hiccup doesn't bypass that recovery path.
            log_debug(
                log,
                path_str,
                &format!(
                    "literal:whitespace_confirm L{scope_start}-L{scope_end} request failed: {e}"
                ),
            );
            return WhitespaceOutcome::NoCandidate;
        }
    };

    if !confirmed {
        log_debug(
            log,
            path_str,
            &format!(
                "literal:whitespace_confirm L{scope_start}-L{scope_end} planner rejected candidate"
            ),
        );
        return WhitespaceOutcome::Rejected;
    }

    let new_text = new.join("\n");
    let abs_start = scope_byte_start + rel_start;
    let abs_end = scope_byte_start + rel_end;
    let mut spliced = String::with_capacity(content.len() + new_text.len());
    spliced.push_str(&content[..abs_start]);
    spliced.push_str(&new_text);
    spliced.push_str(&content[abs_end..]);
    log_debug(
        log,
        path_str,
        &format!("literal:whitespace_confirm L{scope_start}-L{scope_end} planner confirmed, applied"),
    );
    WhitespaceOutcome::Applied(spliced)
}

/// Ask the planner model whether a whitespace-normalized candidate is
/// the same edit target it intended to write. Single round-trip, single
/// word reply expected (`YES` / `NO`).
#[allow(clippy::too_many_arguments)]
async fn request_whitespace_drift_confirmation(
    path_str: &str,
    scope_start: usize,
    scope_end: usize,
    planner_old: &str,
    actual_match: &str,
    router: &ModelRouter,
    cancelled: Option<&AtomicBool>,
    log: Option<&SessionLog>,
) -> Result<bool> {
    ensure_not_cancelled(cancelled)?;
    let prompt = format!(
        "An edit-plan step for {path_str} (lines {scope_start}-{scope_end}) emitted an OLD \
         block that does not byte-exactly match the file, but after ignoring whitespace \
         differences a near-match was found in the same scope. Confirm whether this near-match \
         is the same edit target you intended.\n\n\
         OLD as written by the planner:\n\
         ----\n{planner_old}\n----\n\n\
         Actual matching text from the file:\n\
         ----\n{actual_match}\n----\n\n\
         Reply with exactly `YES` if this is the intended target, or `NO` if it isn't. \
         Output only the single word, no other text."
    );

    let request = ChatRequest {
        messages: vec![
            Message::system(
                "You answer with exactly `YES` or `NO`. No other output, no markdown.",
            ),
            Message::user(&prompt),
        ],
        tools: None,
        tool_choice: None,
    };

    log_stage(
        log,
        path_str,
        &format!("literal:whitespace_confirm:L{scope_start}-L{scope_end}"),
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
            "literal:whitespace_confirm:L{scope_start}-L{scope_end} reply: {}",
            truncate_multiline(text, 400)
        ),
    );

    // Tolerate trailing punctuation, mixed case, and a leading word like
    // "Yes." or "yes,". Reject anything that doesn't have "yes" as its
    // first alphabetic token.
    let first_word: String = text
        .trim()
        .chars()
        .take_while(|c| c.is_alphabetic())
        .flat_map(|c| c.to_lowercase())
        .collect();
    Ok(first_word == "yes")
}

fn apply_resolved_patch(content: &str, mut resolved: Vec<ResolvedOp>) -> Result<String> {
    let had_trailing_newline = content.ends_with('\n');
    let mut lines: Vec<String> = content.lines().map(str::to_string).collect();

    resolved.sort_by(|a, b| b.start.cmp(&a.start).then_with(|| b.end.cmp(&a.end)));

    for op in &resolved {
        match &op.kind {
            ResolvedKind::Insert { content } => {
                lines.splice(op.start..op.start, content.clone());
            }
            ResolvedKind::Replace { content } => {
                lines.splice(op.start..op.end, content.clone());
            }
            ResolvedKind::Delete => {
                lines.splice(op.start..op.end, Vec::<String>::new());
            }
        }
    }

    let mut out = lines.join("\n");
    if had_trailing_newline && !out.ends_with('\n') {
        out.push('\n');
    }
    Ok(out)
}

#[derive(Debug, Clone)]
struct ResolvedOp {
    label: String,
    start: usize,
    end: usize,
    kind: ResolvedKind,
}

#[derive(Debug, Clone)]
enum ResolvedKind {
    Insert { content: Vec<String> },
    Replace { content: Vec<String> },
    Delete,
}

fn resolve_ops(original: &[String], ops: &[PatchOp]) -> Result<Vec<ResolvedOp>> {
    let mut resolved = Vec::new();

    for (idx, op) in ops.iter().enumerate() {
        let label = op_label(idx + 1, op);
        match op {
            PatchOp::InsertBefore { line, content } => {
                validate_insert_line(*line, original.len())?;
                resolved.push(ResolvedOp {
                    label,
                    start: *line - 1,
                    end: *line - 1,
                    kind: ResolvedKind::Insert {
                        content: content.clone(),
                    },
                });
            }
            PatchOp::InsertAfter { line, content } => {
                validate_insert_line(*line, original.len())?;
                resolved.push(ResolvedOp {
                    label,
                    start: *line,
                    end: *line,
                    kind: ResolvedKind::Insert {
                        content: content.clone(),
                    },
                });
            }
            PatchOp::ReplaceAt { start, old, new } => {
                let start_idx = resolve_old_anchor(original, *start, old, "REPLACE_AT")?;
                resolved.push(ResolvedOp {
                    label,
                    start: start_idx,
                    end: start_idx + old.len(),
                    kind: ResolvedKind::Replace {
                        content: new.clone(),
                    },
                });
            }
            PatchOp::DeleteAt { start, old } => {
                let start_idx = resolve_old_anchor(original, *start, old, "DELETE_AT")?;
                resolved.push(ResolvedOp {
                    label,
                    start: start_idx,
                    end: start_idx + old.len(),
                    kind: ResolvedKind::Delete,
                });
            }
        }
    }

    reject_overlapping_spans(&resolved)?;
    Ok(resolved)
}

fn op_label(ordinal: usize, op: &PatchOp) -> String {
    match op {
        PatchOp::InsertBefore { line, .. } => format!("op {ordinal} INSERT_BEFORE {line}"),
        PatchOp::InsertAfter { line, .. } => format!("op {ordinal} INSERT_AFTER {line}"),
        PatchOp::ReplaceAt { start, old, .. } => {
            format!(
                "op {ordinal} REPLACE_AT {start} ({} OLD line(s))",
                old.len()
            )
        }
        PatchOp::DeleteAt { start, old } => {
            format!("op {ordinal} DELETE_AT {start} ({} OLD line(s))", old.len())
        }
    }
}

fn resolve_old_anchor(
    original: &[String],
    start_line: usize,
    old: &[String],
    op_name: &str,
) -> Result<usize> {
    if old.is_empty() {
        bail!("{op_name} OLD block must not be empty");
    }
    if start_line == 0 {
        bail!("line numbers are 1-based");
    }

    let hinted = start_line - 1;
    if hinted + old.len() <= original.len() && original[hinted..hinted + old.len()] == *old {
        return Ok(hinted);
    }

    let matches = find_exact_block_matches(original, old);
    match matches.as_slice() {
        [idx] => Ok(*idx),
        [] => {
            let mut msg = format!(
                "OLD mismatch for {op_name} {start_line}: OLD block was not found at the anchor or elsewhere"
            );
            msg.push_str(&format!(
                "\nOLD block ({} line(s)):\n{}",
                old.len(),
                preview_block(old, None)
            ));
            msg.push_str(&format!(
                "\nActual text at anchor:\n{}",
                preview_anchor(original, hinted, old.len())
            ));
            let trimmed_matches = find_trimmed_block_matches(original, old);
            match trimmed_matches.as_slice() {
                [] => {}
                [idx] => msg.push_str(&format!(
                    "\nWhitespace-trimmed OLD would match at L{}; preserve exact indentation/spacing in OLD.",
                    idx + 1
                )),
                many => msg.push_str(&format!(
                    "\nWhitespace-trimmed OLD would match {} locations: {}. Use a more specific OLD block.",
                    many.len(),
                    format_line_list(many)
                )),
            }
            bail!("{msg}");
        }
        _ => bail!(
            "OLD mismatch for {op_name} {start_line}: OLD block matched {} locations: {}. Use a more specific OLD block.\nOLD block:\n{}",
            matches.len(),
            format_line_list(&matches),
            preview_block(old, None)
        ),
    }
}

fn find_exact_block_matches(haystack: &[String], needle: &[String]) -> Vec<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return Vec::new();
    }

    let mut matches = Vec::new();
    for start in 0..=haystack.len() - needle.len() {
        if haystack[start..start + needle.len()] == *needle {
            matches.push(start);
        }
    }
    matches
}

fn find_trimmed_block_matches(haystack: &[String], needle: &[String]) -> Vec<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return Vec::new();
    }

    let mut matches = Vec::new();
    for start in 0..=haystack.len() - needle.len() {
        if haystack[start..start + needle.len()]
            .iter()
            .zip(needle)
            .all(|(left, right)| left.trim() == right.trim())
        {
            matches.push(start);
        }
    }
    matches
}

fn preview_anchor(original: &[String], start_idx: usize, desired_len: usize) -> String {
    if start_idx >= original.len() {
        return format!(
            "anchor L{} is beyond end of file ({} line(s))",
            start_idx + 1,
            original.len()
        );
    }

    let len = desired_len.max(1);
    let end = (start_idx + len).min(original.len());
    preview_block(&original[start_idx..end], Some(start_idx + 1))
}

fn preview_block(lines: &[String], first_line: Option<usize>) -> String {
    const MAX_PREVIEW_LINES: usize = 6;
    let mut out = String::new();
    for (idx, line) in lines.iter().take(MAX_PREVIEW_LINES).enumerate() {
        if !out.is_empty() {
            out.push('\n');
        }
        match first_line {
            Some(first) => out.push_str(&format!("L{}: {:?}", first + idx, line)),
            None => out.push_str(&format!("OLD{}: {:?}", idx + 1, line)),
        }
    }
    if lines.len() > MAX_PREVIEW_LINES {
        out.push_str(&format!(
            "\n... {} more line(s)",
            lines.len() - MAX_PREVIEW_LINES
        ));
    }
    out
}

fn format_line_list(indices: &[usize]) -> String {
    const MAX_LINES: usize = 8;
    let mut parts: Vec<String> = indices
        .iter()
        .take(MAX_LINES)
        .map(|idx| format!("L{}", idx + 1))
        .collect();
    if indices.len() > MAX_LINES {
        parts.push(format!("...{} more", indices.len() - MAX_LINES));
    }
    parts.join(", ")
}

fn reject_overlapping_spans(ops: &[ResolvedOp]) -> Result<()> {
    let mut spans: Vec<&ResolvedOp> = ops.iter().filter(|op| op.start != op.end).collect();
    spans.sort_unstable_by(|a, b| a.start.cmp(&b.start).then_with(|| a.end.cmp(&b.end)));

    for pair in spans.windows(2) {
        let prev = pair[0];
        let next = pair[1];
        if next.start < prev.end {
            bail!(
                "patch operations have overlapping replacement/delete spans: {} covers {}, overlaps {} covers {}. Use the smallest enclosing REPLACE_AT block for the overlap, split the patch into non-overlapping regions, or retry with a narrower edit_file task for one region/function.",
                prev.label,
                display_span(prev.start, prev.end),
                next.label,
                display_span(next.start, next.end),
            );
        }
    }

    Ok(())
}

fn display_span(start: usize, end: usize) -> String {
    if end <= start + 1 {
        format!("L{}", start + 1)
    } else {
        format!("L{}-L{}", start + 1, end)
    }
}

fn validate_insert_line(line: usize, total_lines: usize) -> Result<()> {
    if line == 0 || line > total_lines {
        bail!("insert line {line} out of range for {total_lines} line file");
    }
    Ok(())
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

    let rejection =
        format!("candidate truncates {path_str} from {old_lines} to {new_lines} lines");
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

    #[test]
    fn find_whitespace_tolerant_match_finds_exact_substring() {
        // Sanity check: a byte-exact substring is still found.
        let scoped = "let foo = bar();\n";
        let (start, end) = find_whitespace_tolerant_match(scoped, "foo = bar").unwrap();
        assert_eq!(&scoped[start..end], "foo = bar");
    }

    #[test]
    fn find_whitespace_tolerant_match_rescues_extra_whitespace_in_old() {
        // The Gemma `</div >` pathology: the model emitted an extra
        // space inside an HTML tag. The actual file has `</div>` and
        // we want the matcher to map back to those bytes.
        let scoped = "<div>hello</div>\n";
        let (start, end) = find_whitespace_tolerant_match(scoped, "<div >hello</div >").unwrap();
        assert_eq!(&scoped[start..end], "<div>hello</div>");
    }

    #[test]
    fn find_whitespace_tolerant_match_rescues_tab_vs_space_drift() {
        // Indented code where the planner used spaces but the file uses
        // a tab (or vice-versa).
        let scoped = "\tif x {\n\t\treturn 1;\n\t}\n";
        let (start, end) = find_whitespace_tolerant_match(scoped, "if x { return 1; }").unwrap();
        assert_eq!(&scoped[start..end], "if x {\n\t\treturn 1;\n\t}");
    }

    #[test]
    fn find_whitespace_tolerant_match_rescues_newline_drift() {
        // Multi-line OLD where the planner ran two lines together.
        let scoped = "first\nsecond\nthird\n";
        let (start, end) = find_whitespace_tolerant_match(scoped, "firstsecond").unwrap();
        assert_eq!(&scoped[start..end], "first\nsecond");
    }

    #[test]
    fn find_whitespace_tolerant_match_returns_none_when_no_candidate() {
        let scoped = "let foo = bar();\n";
        assert_eq!(find_whitespace_tolerant_match(scoped, "qux"), None);
    }

    #[test]
    fn find_whitespace_tolerant_match_rejects_empty_old_text() {
        // An OLD that's all whitespace squashes to nothing — we can't
        // anchor anywhere, so return None instead of pretending to match.
        assert_eq!(find_whitespace_tolerant_match("foo bar\n", ""), None);
        assert_eq!(find_whitespace_tolerant_match("foo bar\n", "  \n\t"), None);
    }

    #[test]
    fn find_whitespace_tolerant_match_returns_first_when_multiple_candidates() {
        // Two non-overlapping candidates → first wins. The user accepted
        // this simplification: "show first only for now."
        let scoped = "x = 1;\nx = 1;\n";
        let (start, end) = find_whitespace_tolerant_match(scoped, "x=1").unwrap();
        assert_eq!(&scoped[start..end], "x = 1");
        // The match is at the very beginning, not the second occurrence.
        assert_eq!(start, 0);
    }

    #[test]
    fn find_whitespace_tolerant_match_handles_multibyte_characters() {
        // Make sure char_indices accounting doesn't panic on multibyte chars.
        let scoped = "let s = \"héllo\";\n";
        let (start, end) = find_whitespace_tolerant_match(scoped, "\"héllo\"").unwrap();
        assert_eq!(&scoped[start..end], "\"héllo\"");
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
                InspectionCommand::Read { start: 280, end: 300 },
            ],
        );
    }

    #[test]
    fn parse_preplan_window_response_drops_invalid_read_ranges() {
        // Malformed READ ranges (start=0, end<start, non-numeric) are
        // silently dropped rather than erroring out so the model doesn't
        // crash the pipeline with a typo.
        let response = parse_preplan_window_response(
            "READ: 0-5\nREAD: 10-3\nREAD: bad\nNOTE still here\n",
        );
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
        append_inspection_result(&mut extra, "SEARCH_RESULT query=`foo`", "SEARCH RESULT for `foo`: 1 hit");
        append_inspection_result(&mut extra, "READ_RESULT range=L10-L12", "READ RESULT L10-L12:\n  10│x");

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
        assert_eq!(r, Some("what should the new parameter default to?".to_string()));
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
        assert_eq!(parse_needs_clarification("NEEDS_CLARIFICATIONAL: foo"), None);
    }

    #[test]
    fn parse_needs_clarification_rejects_unrelated_text() {
        assert_eq!(parse_needs_clarification("NO_CHANGES"), None);
        assert_eq!(parse_needs_clarification("LITERAL_REPLACE\nSCOPE 1 1\n"), None);
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
    fn looks_like_no_changes_accepts_bare_sentinel() {
        assert!(looks_like_no_changes("NO_CHANGES"));
    }

    #[test]
    fn looks_like_no_changes_tolerates_surrounding_whitespace() {
        assert!(looks_like_no_changes("  NO_CHANGES  "));
        assert!(looks_like_no_changes("\n\nNO_CHANGES\n"));
        assert!(looks_like_no_changes("\tNO_CHANGES\n\n"));
    }

    #[test]
    fn looks_like_no_changes_tolerates_code_fences() {
        // Smaller models sometimes wrap everything in a fence, and
        // `strip_code_fences` already handles that shape — we just need
        // to make sure the NO_CHANGES recognizer composes with it.
        assert!(looks_like_no_changes("```\nNO_CHANGES\n```"));
        assert!(looks_like_no_changes("```text\nNO_CHANGES\n```"));
    }

    #[test]
    fn looks_like_no_changes_rejects_empty() {
        // Empty response is the pathology case, not the legitimate
        // "nothing to do" case — it must NOT be treated as NO_CHANGES.
        assert!(!looks_like_no_changes(""));
        assert!(!looks_like_no_changes("   "));
        assert!(!looks_like_no_changes("\n\n"));
    }

    #[test]
    fn looks_like_no_changes_rejects_content_alongside_sentinel() {
        // A plan with steps is NOT NO_CHANGES, even if the model
        // sprinkled a stray NO_CHANGES somewhere. (The fall-through
        // parser tolerates stray sentinels mid-plan; this helper only
        // recognizes the pure, unambiguous signal.)
        assert!(!looks_like_no_changes("NO_CHANGES\nLITERAL_REPLACE\nSCOPE 1 1"));
        assert!(!looks_like_no_changes("LITERAL_REPLACE\nSCOPE 1 1\nNO_CHANGES"));
        assert!(!looks_like_no_changes("NEEDS_CLARIFICATION: which field?"));
    }

    #[test]
    fn looks_like_no_changes_rejects_lookalikes() {
        assert!(!looks_like_no_changes("NO_CHANGE"));
        assert!(!looks_like_no_changes("NO_CHANGES_NEEDED"));
        assert!(!looks_like_no_changes("no_changes")); // case-sensitive on purpose
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
        };

        let block = format_repair_context(&ctx);
        assert!(block.contains("A previous edit plan was attempted and failed."));
        assert!(block.contains("Previous edit plan (as tried):"));
        assert!(block.contains("rewrite header"));
        assert!(block.contains(
            "Steps that succeeded and have ALREADY been applied to the file shown below:\n(none — the first step failed, file is unchanged from the initial state)"
        ));
        assert!(block.contains("Step that FAILED:"));
        assert!(block.contains("Failure reason:\nregion missing anchor"));
        assert!(block.contains("If the task already appears complete, return NO_CHANGES."));
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
        let input = "LITERAL_REPLACE\nSCOPE 1 1\nALL true\nOLD:\nfoo\nEND_OLD\nNEW:\nbar\nEND_NEW\nEND\n";
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
            EditPlanStep::LiteralReplace { scope_start, scope_end, all, old, new } => {
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
