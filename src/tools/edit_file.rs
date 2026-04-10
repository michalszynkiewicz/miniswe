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
use crate::config::{Config, ModelRole};
use crate::llm::{ChatRequest, Message, ModelRouter};
use crate::logging::SessionLog;
use crate::lsp::LspClient;

/// Max lines per window for reliable LLM recall.
const WINDOW_SIZE: usize = 800;
/// Overlap between windows to catch edits at boundaries.
const PREPLAN_READ_OVERLAP: usize = 60;
/// Max iterations of the reconnaissance loop. Each iteration is one LLM call
/// where the model decides what to SEARCH/READ next, or emits DONE.
const MAX_RECON_ROUNDS: usize = 3;
const MAX_PATCH_ATTEMPTS: usize = 3;
const MAX_PLAN_ATTEMPTS: usize = 4;
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

pub async fn execute(
    args: &Value,
    config: &Config,
    router: &ModelRouter,
    lsp: Option<&LspClient>,
    cancelled: Option<&AtomicBool>,
    log: Option<&SessionLog>,
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
        Ok(PreplanResult::InvalidTask(reason)) => {
            let reason = if reason.is_empty() {
                "no reason provided".to_string()
            } else {
                reason
            };
            Ok(ToolResult::err(format!(
                "edit_file rejected task as invalid: {reason}\n\
                 The pre-plan model determined this task is incoherent, malformed, \
                 or impossible to satisfy from {path_str} alone. The file was not modified."
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

/// What one planning attempt produced. The model can return either a
/// concrete edit plan or, at the finalize phase, reject the task as
/// invalid via `INVALID_TASK: <reason>`. The escape hatch is meant for
/// genuinely incoherent or impossible tasks — not "this is hard" — and
/// short-circuits the entire repair retry loop.
enum PreplanOutcome {
    Steps(Vec<EditPlanStep>),
    InvalidTask(String),
}

/// What the whole pre-plan retry loop produced. Distinguishes "applied
/// edits", "no edits needed", and "model rejected the task as invalid"
/// so the caller can render an appropriate tool result for each.
enum PreplanResult {
    Applied(SplitResult),
    NoChanges,
    InvalidTask(String),
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
) -> Result<PreplanResult> {
    let max_literal_lines = max_literal_replace_lines(config.model.context_window);
    let mut current = original.to_string();
    let mut repair_context: Option<RepairContext> = None;
    let mut progress_log = String::new();

    for attempt in 1..=MAX_PLAN_ATTEMPTS {
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

        let steps = match outcome {
            PreplanOutcome::Steps(s) => s,
            PreplanOutcome::InvalidTask(reason) => {
                // The model rejected the task as fundamentally invalid.
                // Short-circuit the entire retry loop and surface the
                // rejection — no point repairing a malformed task.
                log_debug(
                    log,
                    path_str,
                    &format!("preplan:invalid_task attempt={attempt} reason={reason}"),
                );
                return Ok(PreplanResult::InvalidTask(reason));
            }
        };

        if steps.is_empty() {
            log_debug(log, path_str, "preplan:return_no_steps");
            if current == original {
                return Ok(PreplanResult::NoChanges);
            }

            let mut message = String::new();
            if !progress_log.is_empty() {
                message.push_str(&progress_log);
            }
            message.push_str(&format!(
                "✓ via pre-plan: converged after {attempt} planning attempt(s), applied edits to {path_str} ({} lines)\n",
                current.lines().count()
            ));
            let validation_note = validate_candidate_for_write(
                path_str,
                path,
                original,
                &current,
                config,
                lsp,
                lsp_validation,
                cancelled,
                log,
            )
            .await?;
            if let Some(note) = validation_note {
                message.push_str(&note);
                message.push('\n');
            }
            return Ok(PreplanResult::Applied(SplitResult {
                content: current,
                message,
            }));
        }

        let step_count = steps.len();
        let mut attempt_message = if let Some(ctx) = &repair_context {
            format!("{label}; previous plan failed: {}", ctx.failure_reason)
        } else {
            label.clone()
        };
        attempt_message.push('\n');
        attempt_message.push_str(&format_preplan_log(&label, &steps));

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
            step_count,
            attempt_message,
            "via pre-plan",
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
                progress_log.push_str(&e.message);
                repair_context = Some(RepairContext {
                    previous_plan: steps,
                    completed_steps: e.completed_steps,
                    failed_step: e.failed_step,
                    failure_reason: e.error,
                });
                current = e.current_content;
            }
        }
    }

    let mut message = progress_log;
    if let Some(ctx) = repair_context {
        message.push_str(&format!(
            "Pre-plan exhausted after {MAX_PLAN_ATTEMPTS} attempt(s); last failure: {}\n",
            ctx.failure_reason
        ));
        bail!(message);
    }
    Ok(PreplanResult::NoChanges)
}

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
    planned_count: usize,
    mut message: String,
    success_label: &str,
) -> std::result::Result<SplitResult, PlannedExecutionFailure> {
    let mut current = current_base.to_string();
    let mut total_ops = 0usize;
    let mut completed_count = 0usize;
    // Records of steps that have been successfully applied to `current`,
    // captured in execution order so the repair planner can see exactly
    // what shifted and what's left.
    let mut completed_records: Vec<EditPlanStep> = Vec::new();
    if !message.ends_with('\n') {
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
                    Ok((candidate, count)) => {
                        current = candidate;
                        total_ops += count;
                        completed_count += 1;
                        completed_records.push(step.clone());
                        message.push_str(&format!(
                            "Pre-plan step {} literal L{}-L{}: replaced {count} occurrence(s)\n",
                            idx + 1,
                            scope_start,
                            scope_end
                        ));
                    }
                    Err(literal_error) => {
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
                            MAX_PATCH_ATTEMPTS,
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
                        })?;
                        current = candidate;
                        total_ops += count;
                        completed_count += 1;
                        completed_records.push(step.clone());
                        message.push_str(&format!(
                            "Pre-plan step {} literal fallback L{}-L{}: applied {count} operation(s)\n",
                            idx + 1,
                            scope_start,
                            scope_end
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
                    MAX_PATCH_ATTEMPTS,
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
                })?;

                if count == 0 {
                    message.push_str(&format!("Pre-plan {region_label}: no changes\n"));
                    completed_count += 1;
                    completed_records.push(step.clone());
                } else {
                    total_ops += count;
                    current = candidate;
                    completed_count += 1;
                    completed_records.push(step.clone());
                    message.push_str(&format!(
                        "Pre-plan {region_label}: applied {count} operation(s)\n"
                    ));
                }
            }
        }
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
    )
    .await
    .map_err(|e| PlannedExecutionFailure {
        current_content: current.clone(),
        message: format!(
            "{message}Pre-plan validation failed after {completed_count}/{planned_count} completed step(s): {e}\n"
        ),
        error: e.to_string(),
        completed_steps: completed_records.clone(),
        failed_step: None,
    })?;
    if let Some(note) = validation_note {
        message.push_str(&note);
        message.push('\n');
    }
    let summary = format!(
        "✓ {success_label}: {completed_count}/{planned_count} step(s) completed, {total_ops} operation(s) applied to {path_str} ({} lines)\n",
        current.lines().count()
    );
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
    max_attempts: usize,
    allow_no_changes: bool,
    cancelled: Option<&AtomicBool>,
    log: Option<&SessionLog>,
) -> Result<(String, usize)> {
    let mut feedback: Option<String> = None;
    let mut last_error = String::new();

    for attempt in 1..=max_attempts {
        let (ops, _) = match request_patch_for_region(
            path_str,
            task,
            current,
            router,
            region,
            feedback.as_deref(),
            lsp_validation,
            cancelled,
            log,
        )
        .await
        {
            Ok(result) => result,
            Err(e) => {
                last_error = e.to_string();
                if attempt < max_attempts {
                    feedback = Some(last_error.clone());
                    continue;
                }
                break;
            }
        };

        if ops.is_empty() {
            if allow_no_changes {
                return Ok((current.to_string(), 0));
            }
            last_error = "smart fallback returned NO_CHANGES".into();
            if attempt < max_attempts {
                feedback = Some(last_error.clone());
                continue;
            }
            break;
        }

        let candidate = match apply_patch_dry_run_in_region(current, &ops, region.start, region.end)
        {
            Ok(candidate) => candidate,
            Err(e) => {
                last_error = e.to_string();
                if attempt < max_attempts {
                    feedback = Some(last_error.clone());
                    continue;
                }
                break;
            }
        };

        if let Err(e) = validate_candidate(path_str, current, &candidate) {
            last_error = e.to_string();
            if attempt < max_attempts {
                feedback = Some(last_error.clone());
                continue;
            }
            break;
        }

        return Ok((candidate, ops.len()));
    }

    bail!("{last_error}")
}

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
) -> Result<Option<String>> {
    ensure_not_cancelled(cancelled)?;
    validate_candidate(path_str, original, candidate)?;
    log_stage(log, path_str, "validate:lsp");
    validate_candidate_with_lsp(
        path_str,
        path,
        original,
        candidate,
        config,
        lsp,
        lsp_validation,
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
) -> Result<Option<String>> {
    if lsp_validation == LspValidationMode::Off {
        return Ok(Some("[lsp] skipped (off)".into()));
    }

    let Some(lsp) = lsp else {
        if lsp_validation == LspValidationMode::Require {
            bail!("LSP validation required but no LSP client is available");
        }
        return Ok(None);
    };

    if !lsp.is_ready() || lsp.has_crashed() {
        if lsp_validation == LspValidationMode::Require {
            bail!("LSP validation required but LSP is not ready");
        }
        return Ok(None);
    }

    let timeout = Duration::from_millis(config.lsp.diagnostic_timeout_ms);
    let baseline_errors = match diagnostics_for_current_file(lsp, path, timeout).await {
        Ok(diags) => error_diagnostics(&diags),
        Err(e) => {
            if lsp_validation == LspValidationMode::Require {
                bail!("LSP baseline diagnostics failed: {e}");
            }
            return Ok(None);
        }
    };

    std::fs::write(path, candidate)?;

    let candidate_diags = match diagnostics_for_current_file(lsp, path, timeout).await {
        Ok(diags) => diags,
        Err(e) => {
            let _ = std::fs::write(path, original);
            let _ = diagnostics_for_current_file(lsp, path, timeout).await;
            if lsp_validation == LspValidationMode::Require {
                bail!("LSP candidate diagnostics failed: {e}");
            }
            return Ok(None);
        }
    };

    let candidate_errors = error_diagnostics(&candidate_diags);
    if candidate_errors.len() > baseline_errors.len() {
        let summary =
            format_lsp_error_regression(path_str, baseline_errors.len(), &candidate_errors);
        let _ = std::fs::write(path, original);
        let _ = diagnostics_for_current_file(lsp, path, timeout).await;
        bail!("{summary}");
    }

    Ok(Some(format!(
        "[lsp] OK ({} -> {} error(s), mode={})",
        baseline_errors.len(),
        candidate_errors.len(),
        lsp_validation.as_str()
    )))
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

fn format_lsp_error_regression(
    path_str: &str,
    baseline_error_count: usize,
    candidate_errors: &[Diagnostic],
) -> String {
    let mut out = format!(
        "LSP diagnostics worsened for {path_str}: {baseline_error_count} -> {} error(s)",
        candidate_errors.len()
    );
    for diag in candidate_errors.iter().take(5) {
        out.push_str(&format!(
            "\n{}:{}:{}: error: {}",
            path_str,
            diag.range.start.line + 1,
            diag.range.start.character + 1,
            diag.message
        ));
    }
    if candidate_errors.len() > 5 {
        out.push_str(&format!(
            "\n... and {} more error(s)",
            candidate_errors.len() - 5
        ));
    }
    out
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
}

#[derive(Debug)]
struct PreplanReconResponse {
    commands: Vec<InspectionCommand>,
    /// True if the model emitted DONE (or NO_CHANGES, or any other "stop"
    /// signal). The recon loop also stops when commands are empty even
    /// without an explicit DONE.
    done: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum InspectionCommand {
    Search(String),
    Read { start: usize, end: usize },
}

/// Parse the response from a single windowed pre-plan turn. Each window
/// emits NOTE lines describing structural landmarks for later phases. The
/// parser is liberal: anything that isn't a recognizable NOTE is silently
/// ignored. Returning empty notes is fine — the model may have nothing
/// useful to say about a particular slice.
fn parse_preplan_window_response(text: &str) -> PreplanWindowResponse {
    let mut notes = Vec::new();

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
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
        // Anything else is silently ignored. The window phase is observation
        // only — stray control words, half-formed steps, or hallucinated
        // SEARCH/READ commands all get dropped without erroring out.
    }

    PreplanWindowResponse { notes }
}

/// Parse the response from one reconnaissance round. Accepts SEARCH and
/// READ commands and a DONE terminator. Liberal: unknown lines are ignored,
/// and an empty/unparseable response is treated as DONE so that the loop
/// terminates rather than hanging on a confused model.
fn parse_preplan_recon_response(text: &str) -> PreplanReconResponse {
    let mut commands = Vec::new();
    let mut done = false;

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        if line.eq_ignore_ascii_case("DONE")
            || line.eq_ignore_ascii_case("NONE")
            || line.eq_ignore_ascii_case("NO_CHANGES")
            || line.eq_ignore_ascii_case("FINALIZE")
            || line.eq_ignore_ascii_case("PLAN")
        {
            done = true;
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
        // Ignore everything else.
    }

    if commands.is_empty() {
        // An empty response or one with only unknown lines should terminate
        // the loop, not silently hang waiting for the next round.
        done = true;
    }

    PreplanReconResponse { commands, done }
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

/// Tracks running totals across all reconnaissance rounds so that the
/// per-edit limits on SEARCH and READ apply globally, not per round.
struct ReconCounters {
    search_count: usize,
    read_count: usize,
    max_reads: usize,
}

/// Execute the SEARCH/READ commands emitted in one recon round, appending
/// their formatted results to `extra_context`. Commands that exceed the
/// per-edit caps are dropped with an inline note rather than erroring out.
fn execute_recon_commands(
    content: &str,
    commands: &[InspectionCommand],
    counters: &mut ReconCounters,
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
    repair_feedback: Option<&str>,
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

    let repair = repair_feedback
        .map(|f| {
            format!(
                "\nPrevious region patch was not applied.\nFailure: {f}\nReturn a corrected patch for this same line region only.\n"
            )
        })
        .unwrap_or_default();
    let prompt = format!(
        "You are editing one line region in {path_str}: lines {}-{}.\n\
         Task: {task}\n\
         {repair}\n\
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

    // ── Phase 1: windowed observation ────────────────────────────────────
    // Walk the file slice-by-slice. The model emits NOTE lines describing
    // structural landmarks. No tentative steps, no SEARCH/READ here — those
    // come in the recon phase.
    //
    // Empty files have nothing to observe and nothing to inspect; both
    // Phase 1 and Phase 2 are skipped (zero windows / zero recon rounds)
    // and we go straight to planning, which already knows how to handle
    // a "0 lines" file from the task alone.
    let windows = if total_lines == 0 {
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
        let prompt = format!(
            "You are observing one file slice-by-slice to gather context for an edit plan.\n\n\
             File: {path_str}\n\
             Task: {task}\n\n\
             {feedback_block}\
             {notes_block}\
             This is slice {current_slice} of {total_slices}, covering lines {start_line}-{end_line} of {total_lines}.\n\
             Adjacent slices overlap by {PREPLAN_READ_OVERLAP} lines.\n\n\
             Your only job in this phase is OBSERVATION. Output zero or more NOTE lines that future planning phases will rely on. You will not see this slice again — capture what matters now.\n\n\
             Good notes are concrete and reusable: function/struct spans with line numbers, signatures verbatim (especially for anything the task touches), the exact line where a relevant block starts, places where the file structure differs from expectation. Reference line numbers from the slice content below.\n\n\
             Bad notes are vague or restate the obvious: \"this file has 800 lines\", \"there is a function\", commentary, opinions, proposed changes.\n\n\
             Output format (one per line, no other content):\n\
             NOTE <concise factual observation referencing line numbers when relevant>\n\n\
             Do NOT propose edits, do NOT request additional reads, do NOT plan. Other phases handle those.\n\n\
             Slice content:\n{slice}",
            current_slice = idx + 1,
            total_slices = windows.len(),
            start_line = start + 1,
            end_line = end,
        );

        let request = ChatRequest {
            messages: vec![
                Message::system(
                    "You are in the observation phase of edit planning. Walk through one file slice and emit NOTE lines for the planner to use later. No edits, no questions, no SEARCH/READ — only NOTE lines. No explanations, no markdown.",
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
        extend_unique_notes(&mut notes, parsed.notes);
    }

    // ── Phase 2: iterative reconnaissance ────────────────────────────────
    // The model now decides what to SEARCH/READ before planning. It sees
    // the accumulated notes and the file metadata, but NOT the file
    // content. Each round may emit more commands or DONE; we run at most
    // MAX_RECON_ROUNDS rounds, applying the per-edit caps on SEARCH/READ.
    let mut extra_context = String::new();
    let mut counters = ReconCounters {
        search_count: 0,
        read_count: 0,
        max_reads,
    };

    let recon_rounds = if total_lines == 0 { 0 } else { MAX_RECON_ROUNDS };
    for round in 0..recon_rounds {
        ensure_not_cancelled(cancelled)?;
        let rounds_remaining = MAX_RECON_ROUNDS - round;
        let notes_block = if notes.is_empty() {
            String::from("Notes from observation phase: (none)\n\n")
        } else {
            format!(
                "Notes from observation phase:\n{}\n\n",
                notes.iter().map(|note| format!("- {note}")).collect::<Vec<_>>().join("\n")
            )
        };
        let results_block = if extra_context.is_empty() {
            String::from("Inspection results so far: (none)\n\n")
        } else {
            format!("Inspection results so far:\n{extra_context}")
        };
        let recon_prompt = format!(
            "You are deciding what to inspect before planning an edit. You do NOT see the file content here — only the notes and the inspection results below.\n\n\
             File: {path_str} ({total_lines} lines)\n\
             Task: {task}\n\n\
             {feedback_block}\
             {notes_block}\
             {results_block}\
             You may issue SEARCH and READ commands to look at parts of the file. The next phase will plan from notes + inspection results, with no other view of the file. Make sure you have enough to plan.\n\n\
             Round {current_round} of {MAX_RECON_ROUNDS} ({rounds_remaining} round(s) remaining including this one). Per-edit caps so far: {used_searches}/{MAX_PREPLAN_SEARCHES} SEARCH, {used_reads}/{max_reads} READ.\n\n\
             Output one of:\n\
             - One or more SEARCH/READ lines (results will be delivered next round):\n\
             SEARCH: <exact text to find in the file>\n\
             READ: <start>-<end>\n\
             - Or, if you have enough information to plan now, exactly:\n\
             DONE\n\n\
             Prefer DONE if the notes already cover what you need. Do not request the same thing twice.",
            current_round = round + 1,
            used_searches = counters.search_count,
            used_reads = counters.read_count,
        );

        let request = ChatRequest {
            messages: vec![
                Message::system(
                    "You are in the reconnaissance phase of edit planning. Decide what to SEARCH/READ in the file, or emit DONE if you have enough. Output only SEARCH:/READ: lines or the single word DONE. No explanations, no markdown.",
                ),
                Message::user(&recon_prompt),
            ],
            tools: None,
            tool_choice: None,
        };

        log_stage(
            log,
            path_str,
            &format!("preplan:recon:{}/{MAX_RECON_ROUNDS}", round + 1),
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
                "preplan:recon:{}/{MAX_RECON_ROUNDS} raw_response:\n{}",
                round + 1,
                truncate_multiline(text, 12000)
            ),
        );

        let parsed = parse_preplan_recon_response(text);
        if !parsed.commands.is_empty() {
            execute_recon_commands(
                content,
                &parsed.commands,
                &mut counters,
                &mut extra_context,
                path_str,
                log,
            )?;
        }
        if parsed.done {
            break;
        }
    }

    // ── Phase 3: planning ────────────────────────────────────────────────
    // Plan the edit using only the notes and the inspection results. The
    // full file content is intentionally NOT in this prompt; large files
    // wouldn't fit, and we want a single uniform path for all sizes.
    let notes_block = if notes.is_empty() {
        String::from("Planning notes: (none collected)\n\n")
    } else {
        format!(
            "Planning notes:\n{}\n\n",
            notes.iter().map(|note| format!("- {note}")).collect::<Vec<_>>().join("\n")
        )
    };
    let results_block = if extra_context.is_empty() {
        String::from("Inspection results: (none)\n\n")
    } else {
        format!("Inspection results:\n{extra_context}")
    };
    let finalize_prompt = format!(
        "Plan the edit for one file. You do NOT see the file content directly — plan from the notes and inspection results below. Line numbers in your plan must match the actual file.\n\n\
         File: {path_str} ({total_lines} lines)\n\
         Task: {task}\n\n\
         {feedback_block}\
         {notes_block}\
         {results_block}\
         Return up to {MAX_PREPLAN_STEPS} non-overlapping steps. Each step should cover at most 5 edit sites.\n\
         Use LITERAL_REPLACE for short, exact text replacements where you have the OLD text verbatim from a SEARCH or READ result.\n\
         Use SMART_EDIT for structural or larger changes; the smart-edit phase will see the file content for that region.\n\
         Do not use LITERAL_REPLACE when OLD or NEW spans more than {max_literal_lines} lines, or for whole functions, impl blocks, modules, or test cases.\n\
         For code, prefer functions/classes/import blocks. For config/text files, prefer logical sections.\n\n\
         Output only these step formats, no explanations, no markdown:\n\n\
         LITERAL_REPLACE\n\
         SCOPE <start> <end>\n\
         ALL true\n\
         OLD:\n\
         <exact text to replace>\n\
         END_OLD\n\
         NEW:\n\
         <replacement text>\n\
         END_NEW\n\
         END\n\n\
         SMART_EDIT\n\
         REGION <start> <end>\n\
         TASK: <specific edit for this region>\n\
         END\n\n\
         Or exactly:\n\
         NO_CHANGES\n\n\
         Escape hatch: if (and only if) the task itself is incoherent, malformed, contradicts the file, or is impossible to satisfy from this file alone — not merely hard or large — reject it with a single line:\n\
         INVALID_TASK: <one short sentence explaining why>\n\
         Do not use INVALID_TASK to dodge difficulty. Use it for genuinely broken tasks."
    );

    let request = ChatRequest {
        messages: vec![
            Message::system(
                "You are in the final planning phase for one file edit. Output only strict edit-plan blocks, exactly NO_CHANGES, or a single INVALID_TASK: <reason> line for genuinely broken tasks. No explanations, no markdown.",
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

    if let Some(reason) = parse_invalid_task(text) {
        log_debug(
            log,
            path_str,
            &format!(
                "preplan:finalize:file:1-{total_lines} invalid_task reason={reason}"
            ),
        );
        return Ok(PreplanOutcome::InvalidTask(reason));
    }

    let mut steps = match parse_edit_plan(text) {
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
            return Err(e);
        }
    };
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
    Ok(PreplanOutcome::Steps(steps))
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
    reject_overlapping_steps(steps)
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
                out.push_str(&format!("TASK: {}\n", region.task));
                out.push_str("END\n\n");
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
                out.push_str("\nEND_NEW\nEND\n\n");
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

/// Detect a finalize-phase `INVALID_TASK` rejection. Returns `Some(reason)`
/// if the response opens with `INVALID_TASK` (optionally followed by `:`
/// and a reason), and `None` otherwise. The reason is trimmed and may be
/// empty if the model omits one.
fn parse_invalid_task(text: &str) -> Option<String> {
    let trimmed = text.trim_start();
    let rest = trimmed.strip_prefix("INVALID_TASK")?;
    // Require either end-of-string, whitespace, or ':' immediately after
    // the keyword so we don't false-match on something like
    // `INVALID_TASKLIST` that a model might invent.
    let after = match rest.chars().next() {
        None => "",
        Some(':') => &rest[1..],
        Some(c) if c.is_whitespace() => rest,
        Some(_) => return None,
    };
    Some(after.trim().to_string())
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

    format!(
        "A previous edit plan was attempted and failed. Use the structured information below to plan a better recovery.\n\n\
         Previous edit plan (as tried):\n{previous_plan}\n\
         Steps that succeeded and have ALREADY been applied to the file shown below:\n{completed}\n\
         Step that FAILED:\n{failed}\n\
         Failure reason:\n{reason}\n\n\
         Plan the remaining work needed to complete the original task against the current file content. \
         If the task already appears complete, return NO_CHANGES.\n\n",
        reason = ctx.failure_reason,
    )
}

fn truncate_multiline(text: &str, max_chars: usize) -> String {
    let char_count = text.chars().count();
    if char_count <= max_chars {
        return text.to_string();
    }

    let truncated: String = text.chars().take(max_chars).collect();
    format!("{truncated}\n...({char_count} chars total, truncated)\n")
}

pub fn parse_edit_plan(text: &str) -> Result<Vec<EditPlanStep>> {
    if text.trim() == "NO_CHANGES" {
        return Ok(Vec::new());
    }
    if text.trim().is_empty() {
        bail!("empty edit plan");
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

            expect_line(&lines, i, "END")?;
            i += 1;

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
    reject_overlapping_steps(&steps)?;
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

    expect_line(lines, idx + 2, "END")?;
    Ok((EditRegion { start, end, task }, idx + 3))
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

fn reject_overlapping_steps(steps: &[EditPlanStep]) -> Result<()> {
    let mut sorted: Vec<&EditPlanStep> = steps.iter().collect();
    sorted.sort_unstable_by(|a, b| {
        a.start_line()
            .cmp(&b.start_line())
            .then_with(|| a.end_line().cmp(&b.end_line()))
    });

    for pair in sorted.windows(2) {
        let prev = pair[0];
        let next = pair[1];
        if next.start_line() <= prev.end_line() {
            bail!(
                "edit plan has overlapping steps: L{}-L{} overlaps L{}-L{}",
                prev.start_line(),
                prev.end_line(),
                next.start_line(),
                next.end_line()
            );
        }
    }
    Ok(())
}

/// Parse strict patch DSL blocks.
pub fn parse_patch(text: &str) -> Result<Vec<PatchOp>> {
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

fn validate_candidate(path_str: &str, original: &str, candidate: &str) -> Result<()> {
    if !original.is_empty() && candidate.is_empty() {
        bail!("candidate output is empty for a non-empty file");
    }

    let old_lines = original.lines().count();
    let new_lines = candidate.lines().count();
    if old_lines > LARGE_TRUNCATION_MIN_LINES && new_lines < old_lines / 2 {
        bail!("candidate truncates file from {old_lines} to {new_lines} lines");
    }

    if is_brace_file(path_str) {
        let old_balance = delimiter_imbalance(original);
        let new_balance = delimiter_imbalance(candidate);
        if new_balance > old_balance {
            bail!("candidate worsens delimiter balance from {old_balance} to {new_balance}");
        }
    }

    Ok(())
}

fn is_brace_file(path: &str) -> bool {
    let brace_exts = [
        "rs", "js", "ts", "tsx", "jsx", "go", "java", "c", "cpp", "h", "hpp", "cs", "kt", "swift",
        "scala", "zig",
    ];
    brace_exts
        .iter()
        .any(|ext| path.ends_with(&format!(".{ext}")))
}

fn delimiter_imbalance(text: &str) -> i64 {
    let parens = text.matches('(').count() as i64 - text.matches(')').count() as i64;
    let braces = text.matches('{').count() as i64 - text.matches('}').count() as i64;
    let brackets = text.matches('[').count() as i64 - text.matches(']').count() as i64;
    parens.abs() + braces.abs() + brackets.abs()
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
    }

    #[test]
    fn parse_preplan_window_response_silently_ignores_non_notes() {
        // Anything that isn't a NOTE line is dropped without erroring. That
        // includes stray control words, hallucinated SEARCH/READ commands,
        // and partial step blocks. The window phase is observation-only.
        let response = parse_preplan_window_response(
            "NOTE looks good\n\
             SEARCH: foo\n\
             READ: 10-20\n\
             SMART_EDIT\n\
             REGION 1 5\n\
             TASK: do thing\n\
             END\n\
             NO_CHANGES\n\
             DONE\n",
        );
        assert_eq!(response.notes, vec!["looks good".to_string()]);
    }

    #[test]
    fn parse_preplan_window_response_accepts_empty_response() {
        // The model may have nothing to add for a slice — that's fine.
        let response = parse_preplan_window_response("");
        assert!(response.notes.is_empty());
    }

    #[test]
    fn parse_preplan_recon_response_accepts_search_and_read() {
        let response = parse_preplan_recon_response("SEARCH: foo\nREAD: 10-20\n");
        assert_eq!(
            response.commands,
            vec![
                InspectionCommand::Search("foo".into()),
                InspectionCommand::Read { start: 10, end: 20 },
            ],
        );
        assert!(!response.done, "explicit DONE not present");
    }

    #[test]
    fn parse_preplan_recon_response_recognizes_done_keyword() {
        let response = parse_preplan_recon_response("DONE\n");
        assert!(response.commands.is_empty());
        assert!(response.done);
    }

    #[test]
    fn parse_preplan_recon_response_treats_empty_as_done() {
        let response = parse_preplan_recon_response("");
        assert!(response.commands.is_empty());
        assert!(response.done, "empty response should terminate the loop");
    }

    #[test]
    fn parse_preplan_recon_response_done_alongside_commands_executes_then_stops() {
        let response = parse_preplan_recon_response("SEARCH: bar\nDONE\n");
        assert_eq!(
            response.commands,
            vec![InspectionCommand::Search("bar".into())],
        );
        assert!(response.done);
    }

    #[test]
    fn parse_preplan_recon_response_drops_invalid_read_ranges() {
        let response = parse_preplan_recon_response("READ: 0-5\nREAD: 10-3\nREAD: bad\n");
        assert!(response.commands.is_empty());
        assert!(response.done);
    }

    #[test]
    fn parse_preplan_recon_response_alternative_terminators() {
        for terminator in ["DONE", "NONE", "NO_CHANGES", "FINALIZE", "PLAN"] {
            let response = parse_preplan_recon_response(terminator);
            assert!(
                response.done,
                "terminator `{terminator}` should set done"
            );
        }
    }

    #[test]
    fn execute_recon_commands_executes_search_and_read() {
        let content = "fn one() {}\nfn two() {}\nfn three() {}\n";
        let commands = vec![
            InspectionCommand::Search("two".into()),
            InspectionCommand::Read { start: 1, end: 2 },
        ];
        let mut counters = ReconCounters {
            search_count: 0,
            read_count: 0,
            max_reads: 6,
        };
        let mut extra = String::new();
        execute_recon_commands(content, &commands, &mut counters, &mut extra, "test.rs", None)
            .unwrap();
        assert_eq!(counters.search_count, 1);
        assert_eq!(counters.read_count, 1);
        assert!(extra.contains("SEARCH_RESULT query=`two`"));
        assert!(extra.contains("READ_RESULT range=L1-L2"));
        assert!(extra.contains("fn one()"));
    }

    #[test]
    fn execute_recon_commands_persists_counters_across_calls() {
        // Counters track totals across multiple recon rounds, not per round.
        let content = "x\n";
        let mut counters = ReconCounters {
            search_count: 0,
            read_count: 0,
            max_reads: 1,
        };
        let mut extra = String::new();
        execute_recon_commands(
            content,
            &[InspectionCommand::Read { start: 1, end: 1 }],
            &mut counters,
            &mut extra,
            "test.rs",
            None,
        )
        .unwrap();
        execute_recon_commands(
            content,
            &[InspectionCommand::Read { start: 1, end: 1 }],
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
    fn execute_recon_commands_drops_overflow_with_inline_note() {
        let content = "x\n";
        let mut commands = Vec::new();
        for _ in 0..(MAX_PREPLAN_SEARCHES + 2) {
            commands.push(InspectionCommand::Search("x".into()));
        }
        let mut counters = ReconCounters {
            search_count: 0,
            read_count: 0,
            max_reads: 6,
        };
        let mut extra = String::new();
        execute_recon_commands(content, &commands, &mut counters, &mut extra, "test.rs", None)
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
    fn parse_invalid_task_accepts_keyword_with_reason() {
        let r = parse_invalid_task("INVALID_TASK: file does not contain any auth code\n");
        assert_eq!(r, Some("file does not contain any auth code".to_string()));
    }

    #[test]
    fn parse_invalid_task_accepts_keyword_with_whitespace_reason() {
        let r = parse_invalid_task("INVALID_TASK   the request is contradictory\n");
        assert_eq!(r, Some("the request is contradictory".to_string()));
    }

    #[test]
    fn parse_invalid_task_accepts_keyword_alone() {
        let r = parse_invalid_task("INVALID_TASK");
        assert_eq!(r, Some(String::new()));
    }

    #[test]
    fn parse_invalid_task_tolerates_leading_whitespace() {
        let r = parse_invalid_task("  \n  INVALID_TASK: oops\n");
        assert_eq!(r, Some("oops".to_string()));
    }

    #[test]
    fn parse_invalid_task_rejects_lookalikes() {
        // Don't false-match on a word that just starts with INVALID_TASK.
        assert_eq!(parse_invalid_task("INVALID_TASKLIST"), None);
        assert_eq!(parse_invalid_task("INVALID_TASKING: foo"), None);
    }

    #[test]
    fn parse_invalid_task_rejects_unrelated_text() {
        assert_eq!(parse_invalid_task("NO_CHANGES"), None);
        assert_eq!(parse_invalid_task("LITERAL_REPLACE\nSCOPE 1 1\n"), None);
        assert_eq!(parse_invalid_task(""), None);
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
