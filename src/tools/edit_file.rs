//! edit_file tool — LLM generates a strict patch, miniswe applies it atomically.
//!
//! The model describes the task, miniswe sends file content to the LLM, and the
//! inner LLM returns a small patch DSL. Patches are dry-run validated before any
//! write. If the broad patch path fails, edit_file falls back to smaller,
//! non-overlapping line regions and validates the combined result before writing.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use std::{collections::BTreeSet, path::Path};

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
const WINDOW_OVERLAP: usize = 100;
const MAX_PATCH_ATTEMPTS: usize = 3;
const MAX_PLAN_ATTEMPTS: usize = 2;
const MAX_LITERAL_FALLBACK_ATTEMPTS: usize = 2;
const MAX_PLANNED_REGIONS: usize = 100;
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

    let content = std::fs::read_to_string(&path)
        .map_err(|e| anyhow::anyhow!("Failed to read {path_str}: {e}"))?;

    if should_preplan(task) {
        log_stage(log, path_str, "preplan:start");
        match execute_preplanned_steps(
            path_str,
            task,
            &path,
            &content,
            router,
            config,
            lsp,
            lsp_validation,
            cancelled,
            log,
        )
        .await
        {
            Ok(Some(candidate_result)) => {
                std::fs::write(&path, &candidate_result.content)?;
                return Ok(ToolResult::ok(candidate_result.message));
            }
            Ok(None) => {}
            Err(e) => {
                log_debug(
                    log,
                    path_str,
                    &format!("preplan:failed {}", truncate_multiline(&e.to_string(), 2000)),
                );
                // Treat pre-planning as an optimization only. If the mixed plan
                // is malformed or a scoped step fails, fall back to the
                // established broad patch path and its repair loop.
            }
        }
    }

    let mut feedback: Option<String> = None;
    let mut last_error = String::new();
    let signature_grounding =
        build_signature_grounding_note(path_str, task, &content, &config.project_root);
    let mut last_raw_patch: Option<String> = None;

    for attempt in 1..=MAX_PATCH_ATTEMPTS {
        log_stage(log, path_str, &format!("broad_patch:attempt:{attempt}"));
        let patch = match request_patch(
            path_str,
            task,
            &content,
            router,
            feedback.as_deref(),
            signature_grounding.as_deref(),
            lsp_validation,
            cancelled,
            log,
        )
        .await
        {
            Ok(r) => r,
            Err(e) => {
                last_error = e.to_string();
                log_debug(
                    log,
                    path_str,
                    &format!(
                        "broad_patch:request_failed:{attempt} {}",
                        truncate_multiline(&last_error, 2000)
                    ),
                );
                if attempt < MAX_PATCH_ATTEMPTS {
                    feedback = Some(build_retry_feedback(
                        &last_error,
                        signature_grounding.as_deref(),
                    ));
                    continue;
                }
                break;
            }
        };
        let PatchResponse {
            ops,
            mut output,
            raw_text,
        } = patch;

        if last_raw_patch.as_deref() == Some(raw_text.trim())
            && is_signature_mismatch_error(&last_error)
        {
            last_error = "model repeated the same invalid patch after LSP rejection".into();
            log_debug(
                log,
                path_str,
                &format!("broad_patch:duplicate_attempt:{attempt} {last_error}"),
            );
            break;
        }
        last_raw_patch = Some(raw_text.trim().to_string());

        if ops.is_empty() {
            return Ok(ToolResult::ok(format!(
                "No changes needed in {path_str} for task: {task}"
            )));
        }

        let candidate = match apply_patch_dry_run(&content, &ops) {
            Ok(candidate) => candidate,
            Err(e) => {
                last_error = e.to_string();
                log_debug(
                    log,
                    path_str,
                    &format!(
                        "broad_patch:apply_failed:{attempt} {}",
                        truncate_multiline(&last_error, 2000)
                    ),
                );
                if attempt < MAX_PATCH_ATTEMPTS {
                    feedback = Some(build_retry_feedback(
                        &last_error,
                        signature_grounding.as_deref(),
                    ));
                    continue;
                }
                break;
            }
        };

        let validation_note = match validate_candidate_for_write(
            path_str,
            &path,
            &content,
            &candidate,
            config,
            lsp,
            lsp_validation,
            cancelled,
            log,
        )
        .await
        {
            Ok(note) => note,
            Err(e) => {
                last_error = e.to_string();
                log_debug(
                    log,
                    path_str,
                    &format!(
                        "broad_patch:validate_failed:{attempt} {}",
                        truncate_multiline(&last_error, 2000)
                    ),
                );
                if attempt < MAX_PATCH_ATTEMPTS {
                    feedback = Some(build_retry_feedback(
                        &last_error,
                        signature_grounding.as_deref(),
                    ));
                    continue;
                }
                break;
            }
        };

        std::fs::write(&path, &candidate)?;
        if let Some(note) = validation_note {
            output.push_str(&note);
            output.push('\n');
        }
        output.push_str(&format!(
            "✓ Applied {} operation(s) to {path_str} ({} lines)\n",
            ops.len(),
            candidate.lines().count()
        ));
        return Ok(ToolResult::ok(output));
    }

    match execute_split_fallback(
        path_str,
        task,
        &path,
        &content,
        router,
        &last_error,
        config,
        lsp,
        lsp_validation,
        cancelled,
        log,
    )
    .await
    {
        Ok(Some(candidate_result)) => {
            std::fs::write(&path, &candidate_result.content)?;
            return Ok(ToolResult::ok(candidate_result.message));
        }
        Ok(None) => {}
        Err(split_error) => {
            log_debug(
                log,
                path_str,
                &format!(
                    "split_fallback:failed {}",
                    truncate_multiline(&split_error.to_string(), 2000)
                ),
            );
            return Ok(ToolResult::err(format!(
                "edit_file failed: patch was not applied.\nReason: {last_error}\nSplit fallback failed: {split_error}\n"
            )));
        }
    }

    Ok(ToolResult::err(format!(
        "edit_file failed: patch was not applied.\nReason: {last_error}\n"
    )))
}

struct SplitResult {
    content: String,
    message: String,
}

fn should_preplan(task: &str) -> bool {
    let lower = task.to_ascii_lowercase();
    [
        "all ",
        "every ",
        "call site",
        "call sites",
        "throughout",
        "all calls",
        "every call",
        "all occurrences",
        "every occurrence",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
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
) -> Result<Option<SplitResult>> {
    let max_literal_lines = max_literal_replace_lines(config.model.context_window);
    let signature_grounding =
        build_signature_grounding_note(path_str, task, original, &config.project_root);
    let mut steps = request_preplan_steps(
        path_str,
        task,
        original,
        router,
        None,
        None,
        max_literal_lines,
        signature_grounding.as_deref(),
        cancelled,
        log,
    )
    .await?;
    if steps.is_empty() {
        log_debug(log, path_str, "preplan:return_no_steps");
        return Ok(None);
    }

    let mut last_error: Option<String> = None;
    for attempt in 1..=MAX_PLAN_ATTEMPTS {
        let step_count = steps.len();
        let label = if last_error.is_some() {
            format!("Pre-plan repair attempt {attempt}")
        } else {
            format!("Pre-plan attempt {attempt}")
        };
        let mut message = if let Some(error) = &last_error {
            format!("{label}; previous plan failed: {error}")
        } else {
            label.clone()
        };
        message.push('\n');
        message.push_str(&format_preplan_log(&label, &steps));

        match execute_planned_steps(
            path_str,
            path,
            original,
            router,
            config,
            lsp,
            lsp_validation,
            cancelled,
            log,
            steps.clone(),
            step_count,
            message,
            "via pre-plan",
        )
        .await
        {
            Ok(result) => return Ok(Some(result)),
            Err(e) if attempt < MAX_PLAN_ATTEMPTS => {
                let error = e.to_string();
                log_debug(
                    log,
                    path_str,
                    &format!(
                        "preplan:apply_failed:{attempt} {}",
                        truncate_multiline(&error, 2000)
                    ),
                );
                let previous_plan = format_edit_plan_steps(&steps);
                steps = request_preplan_steps(
                    path_str,
                    task,
                    original,
                    router,
                    Some(&error),
                    Some(&previous_plan),
                    max_literal_lines,
                    signature_grounding.as_deref(),
                    cancelled,
                    log,
                )
                .await?;
                if steps.is_empty() {
                    log_debug(
                        log,
                        path_str,
                        &format!(
                            "preplan:repair_returned_no_steps:{}",
                            truncate_multiline(&error, 2000)
                        ),
                    );
                    return Ok(None);
                }
                last_error = Some(error);
            }
            Err(e) => {
                log_debug(
                    log,
                    path_str,
                    &format!(
                        "preplan:terminal_failure {}",
                        truncate_multiline(&e.to_string(), 2000)
                    ),
                );
                return Err(e);
            }
        }
    }

    unreachable!("plan attempts loop must return")
}

async fn execute_split_fallback(
    path_str: &str,
    task: &str,
    path: &std::path::Path,
    original: &str,
    router: &ModelRouter,
    broad_error: &str,
    config: &Config,
    lsp: Option<&LspClient>,
    lsp_validation: LspValidationMode,
    cancelled: Option<&AtomicBool>,
    log: Option<&SessionLog>,
) -> Result<Option<SplitResult>> {
    let regions =
        request_region_plan(path_str, task, original, router, broad_error, cancelled, log).await?;
    if regions.is_empty() {
        log_debug(
            log,
            path_str,
            &format!(
                "split_plan:return_no_regions broad_error={}",
                truncate_multiline(broad_error, 2000)
            ),
        );
        return Ok(None);
    }

    let region_count = regions.len();
    execute_planned_regions(
        path_str,
        path,
        original,
        router,
        config,
        lsp,
        lsp_validation,
        cancelled,
        log,
        regions,
        region_count,
        format!(
            "Broad patch failed: {broad_error}\nSplit fallback: {region_count} region(s) planned"
        ),
        "via split fallback",
    )
        .await
        .map(Some)
}

async fn execute_planned_regions(
    path_str: &str,
    path: &std::path::Path,
    original: &str,
    router: &ModelRouter,
    config: &Config,
    lsp: Option<&LspClient>,
    lsp_validation: LspValidationMode,
    cancelled: Option<&AtomicBool>,
    log: Option<&SessionLog>,
    regions: Vec<EditRegion>,
    planned_count: usize,
    mut message: String,
    success_label: &str,
) -> Result<SplitResult> {
    let mut current = original.to_string();
    let mut total_ops = 0usize;
    let mut completed_steps = 0usize;
    if !message.ends_with('\n') {
        message.push('\n');
    }

    let mut regions_desc = regions;
    regions_desc.sort_by(|a, b| b.start.cmp(&a.start).then_with(|| b.end.cmp(&a.end)));

    for (idx, region) in regions_desc.iter().enumerate() {
        let mut feedback: Option<String> = None;
        let mut last_region_error = String::new();
        let region_label = format!("region {} L{}-L{}", idx + 1, region.start, region.end);

        for attempt in 1..=MAX_PATCH_ATTEMPTS {
            let (ops, _) = match request_patch_for_region(
                path_str,
                &region.task,
                &current,
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
                    last_region_error = e.to_string();
                    if attempt < MAX_PATCH_ATTEMPTS {
                        feedback = Some(last_region_error.clone());
                        continue;
                    }
                    break;
                }
            };

            if ops.is_empty() {
                message.push_str(&format!("Split {region_label}: no changes\n"));
                completed_steps += 1;
                last_region_error.clear();
                break;
            }

            let candidate =
                match apply_patch_dry_run_in_region(&current, &ops, region.start, region.end) {
                    Ok(candidate) => candidate,
                    Err(e) => {
                        last_region_error = e.to_string();
                        if attempt < MAX_PATCH_ATTEMPTS {
                            feedback = Some(last_region_error.clone());
                            continue;
                        }
                        break;
                    }
                };

            if let Err(e) = validate_candidate(path_str, &current, &candidate) {
                last_region_error = e.to_string();
                if attempt < MAX_PATCH_ATTEMPTS {
                    feedback = Some(last_region_error.clone());
                    continue;
                }
                break;
            }

            total_ops += ops.len();
            completed_steps += 1;
            current = candidate;
            message.push_str(&format!(
                "Split {region_label}: applied {} operation(s)\n",
                ops.len()
            ));
            last_region_error.clear();
            break;
        }

        if !last_region_error.is_empty() {
            bail!("{region_label} failed: {last_region_error}");
        }
    }

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
    let summary = format!(
        "✓ {success_label}: {completed_steps}/{planned_count} step(s) completed, {total_ops} operation(s) applied to {path_str} ({} lines)\n",
        current.lines().count()
    );
    message = format!("{summary}{message}");

    Ok(SplitResult {
        content: current,
        message,
    })
}

async fn execute_planned_steps(
    path_str: &str,
    path: &std::path::Path,
    original: &str,
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
) -> Result<SplitResult> {
    let mut current = original.to_string();
    let mut total_ops = 0usize;
    let mut completed_steps = 0usize;
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
                        completed_steps += 1;
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
                            MAX_LITERAL_FALLBACK_ATTEMPTS,
                            false,
                            cancelled,
                            log,
                        )
                        .await
                        .map_err(|e| {
                            anyhow::anyhow!(
                                "step {} literal replace failed: {literal_error}; smart fallback failed: {e}",
                                idx + 1
                            )
                        })?;
                        current = candidate;
                        total_ops += count;
                        completed_steps += 1;
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
                .map_err(|e| anyhow::anyhow!("{region_label} failed: {e}"))?;

                if count == 0 {
                    message.push_str(&format!("Pre-plan {region_label}: no changes\n"));
                    completed_steps += 1;
                } else {
                    total_ops += count;
                    current = candidate;
                    completed_steps += 1;
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
    let summary = format!(
        "✓ {success_label}: {completed_steps}/{planned_count} step(s) completed, {total_ops} operation(s) applied to {path_str} ({} lines)\n",
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

const MAX_PREPLAN_SEARCHES: usize = 10;
const MAX_PREPLAN_READS_INITIAL: usize = 3;
const MAX_PREPLAN_READS_REPAIR: usize = 5;

fn build_retry_feedback(last_error: &str, signature_grounding: Option<&str>) -> String {
    let mut feedback = last_error.to_string();
    if is_signature_mismatch_error(last_error) {
        feedback.push_str(
            "\nDo not repeat the same patch shape on the same lines. Re-check the current callee signatures and only edit call sites whose current argument count and types match the intended change.",
        );
    }
    if let Some(grounding) = signature_grounding {
        if !grounding.is_empty() {
            feedback.push_str("\n\n");
            feedback.push_str(grounding);
        }
    }
    feedback
}

fn is_signature_mismatch_error(error: &str) -> bool {
    error.contains("expected ")
        && (error.contains("arguments, found") || error.contains("mismatched types"))
}

fn build_signature_grounding_note(
    path_str: &str,
    task: &str,
    content: &str,
    project_root: &Path,
) -> Option<String> {
    if !path_str.ends_with(".rs") {
        return None;
    }
    let lower = task.to_ascii_lowercase();
    if !(lower.contains("call")
        || lower.contains("argument")
        || lower.contains("parameter")
        || lower.contains("pass ")
        || lower.contains("as_deref"))
    {
        return None;
    }

    let mut seen = BTreeSet::new();
    let mut signatures = Vec::new();
    for line in content.lines() {
        for (module_path, fn_name) in extract_qualified_rust_calls(line) {
            let key = format!("{module_path}::{fn_name}");
            if !seen.insert(key) {
                continue;
            }
            if let Some(signature) =
                resolve_rust_signature_hint(project_root, &module_path, &fn_name)
            {
                signatures.push(signature);
            }
            if signatures.len() >= 6 {
                break;
            }
        }
        if signatures.len() >= 6 {
            break;
        }
    }

    if signatures.is_empty() {
        return None;
    }

    let mut note =
        String::from("Current known callee signatures from the repo; use these before editing call sites:\n");
    for signature in signatures {
        note.push_str("- ");
        note.push_str(&signature);
        note.push('\n');
    }
    Some(note.trim_end().to_string())
}

enum PreplanAssistantResponse {
    Plan(String),
    Search(String),
    Read { start: usize, end: usize },
}

fn parse_preplan_assistant_response(text: &str) -> Result<PreplanAssistantResponse> {
    let trimmed = text.trim();
    if let Some(rest) = trimmed.strip_prefix("SEARCH:") {
        let query = rest.trim();
        if query.is_empty() {
            bail!("empty SEARCH query");
        }
        return Ok(PreplanAssistantResponse::Search(query.to_string()));
    }
    if let Some(rest) = trimmed.strip_prefix("READ:") {
        let rest = rest.trim();
        let (start, end) = rest
            .split_once('-')
            .ok_or_else(|| anyhow::anyhow!("READ must be in the form READ: <start>-<end>"))?;
        let start = start.trim().parse::<usize>()?;
        let end = end.trim().parse::<usize>()?;
        if start == 0 || end < start {
            bail!("invalid READ range {start}-{end}");
        }
        return Ok(PreplanAssistantResponse::Read { start, end });
    }
    Ok(PreplanAssistantResponse::Plan(trimmed.to_string()))
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

fn extract_qualified_rust_calls(line: &str) -> Vec<(String, String)> {
    let mut calls = Vec::new();
    let bytes = line.as_bytes();
    for (idx, ch) in line.char_indices() {
        if ch != '(' {
            continue;
        }
        let mut start = idx;
        while start > 0 {
            let prev = bytes[start - 1] as char;
            if prev.is_ascii_alphanumeric() || prev == '_' || prev == ':' {
                start -= 1;
            } else {
                break;
            }
        }
        let token = line[start..idx].trim();
        if !token.contains("::") {
            continue;
        }
        let Some((module_path, fn_name)) = token.rsplit_once("::") else {
            continue;
        };
        if module_path.is_empty() || fn_name.is_empty() {
            continue;
        }
        calls.push((module_path.to_string(), fn_name.to_string()));
    }
    calls
}

fn resolve_rust_signature_hint(
    project_root: &Path,
    module_path: &str,
    fn_name: &str,
) -> Option<String> {
    let module_path = module_path.strip_prefix("crate::").unwrap_or(module_path);
    let module_path = module_path.strip_prefix("miniswe::").unwrap_or(module_path);
    let segments: Vec<&str> = module_path.split("::").filter(|s| !s.is_empty()).collect();
    if segments.is_empty() {
        return None;
    }

    let mut candidates = Vec::new();
    candidates.push(project_root.join(format!("{}.rs", segments.join("/"))));
    candidates.push(project_root.join(segments.join("/")).join("mod.rs"));
    candidates.push(project_root.join("src").join(format!("{}.rs", segments.join("/"))));
    candidates.push(project_root.join("src").join(segments.join("/")).join("mod.rs"));

    for path in candidates {
        if !path.exists() {
            continue;
        }
        let Ok(content) = std::fs::read_to_string(&path) else {
            continue;
        };
        for (idx, line) in content.lines().enumerate() {
            if line.contains(&format!("fn {fn_name}(")) {
                let rel = path.strip_prefix(project_root).ok().unwrap_or(&path);
                return Some(format!(
                    "{}:{} `{}`",
                    rel.display(),
                    idx + 1,
                    line.trim()
                ));
            }
        }
    }
    None
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
    feedback: Option<&str>,
    previous_plan: Option<&str>,
    max_literal_lines: usize,
    signature_grounding: Option<&str>,
    cancelled: Option<&AtomicBool>,
    log: Option<&SessionLog>,
) -> Result<Vec<EditPlanStep>> {
    ensure_not_cancelled(cancelled)?;
    let lines: Vec<&str> = content.lines().collect();
    let total_lines = lines.len();
    let windows = build_windows(total_lines, WINDOW_SIZE, WINDOW_OVERLAP);
    let mut steps = Vec::new();
    let mut search_count = 0usize;
    let mut read_count = 0usize;
    let max_reads = if feedback.is_some() {
        MAX_PREPLAN_READS_REPAIR
    } else {
        MAX_PREPLAN_READS_INITIAL
    };

    for (win_idx, (start, end)) in windows.iter().enumerate() {
        ensure_not_cancelled(cancelled)?;
        if steps.len() >= MAX_PREPLAN_STEPS {
            break;
        }

        let window_content = lines[*start..*end]
            .iter()
            .enumerate()
            .map(|(i, line)| format!("{:>4}│{}", start + i + 1, line))
            .collect::<Vec<_>>()
            .join("\n");
        let remaining = MAX_PREPLAN_STEPS - steps.len();

        let feedback_block = feedback
            .map(|failure| {
                let previous = previous_plan
                    .map(|plan| format!("Previous edit plan:\n{plan}\n\n"))
                    .unwrap_or_default();
                format!(
                    "Previous edit plan was not applied.\nFailure: {failure}\n{previous}Return a corrected complete edit plan against the original file.\n\n"
                )
            })
            .unwrap_or_default();
        let signature_block = signature_grounding.map(|note| format!("{note}\n\n")).unwrap_or_default();
        let mut extra_context = String::new();
        let text = loop {
            let prompt = format!(
                "Plan small edit steps for one file.\n\n\
                 File: {path_str}\n\
                 Task: {task}\n\n\
                 {signature_block}\
                 {feedback_block}\
                 {extra_context}\
                 You may first inspect the current file using these bounded commands:\n\
                 - SEARCH: <exact text>   (current file only, max {MAX_PREPLAN_SEARCHES} total)\n\
                 - READ: <start>-<end>    (current file only, max {max_reads} total in this planning phase)\n\
                 After any inspection, return the final plan or NO_REGIONS.\n\
                 Return up to {remaining} non-overlapping steps within this window.\n\
                 Use LITERAL_REPLACE for obvious exact text replacements.\n\
                 Use SMART_EDIT for ambiguous or structural edits.\n\
                 Do not use LITERAL_REPLACE when OLD or NEW spans more than {max_literal_lines} lines.\n\
                 Do not use LITERAL_REPLACE for whole functions, impl blocks, modules, test cases, or other large code blocks even if the text matches exactly.\n\
                 If the edit would require a larger span, split it into smaller LITERAL_REPLACE steps or use SMART_EDIT.\n\
                 Each step should cover at most 5 edit sites.\n\
                 For repeated call-site updates, group nearby exact calls with LITERAL_REPLACE when safe.\n\
                 For code, prefer functions/classes/import blocks.\n\
                 For config/text files, prefer logical sections.\n\
                 Line numbers refer to the full file shown below.\n\n\
                 Output only one of these formats:\n\n\
                 SEARCH: <exact text>\n\n\
                 READ: <start>-<end>\n\n\
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
                 NO_REGIONS\n\n\
                 Window {}/{} lines {}-{} of {}:\n{window_content}",
                win_idx + 1,
                windows.len(),
                start + 1,
                end,
                total_lines
            );

            let request = ChatRequest {
                messages: vec![
                    Message::system(
                        "You output only strict edit-plan blocks or one SEARCH/READ command. No explanations, no markdown.",
                    ),
                    Message::user(&prompt),
                ],
                tools: None,
                tool_choice: None,
            };

            log_stage(log, path_str, &format!("preplan:window:{}-{}", start + 1, end));
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
                    "preplan:window:{}-{} raw_response:\n{}",
                    start + 1,
                    end,
                    truncate_multiline(text, 12000)
                ),
            );

            match parse_preplan_assistant_response(text)? {
                PreplanAssistantResponse::Plan(plan) => break plan,
                PreplanAssistantResponse::Search(query) => {
                    if search_count >= MAX_PREPLAN_SEARCHES {
                        bail!("preplan exceeded SEARCH limit of {MAX_PREPLAN_SEARCHES}");
                    }
                    search_count += 1;
                    let result = search_in_file(content, &query);
                    log_debug(
                        log,
                        path_str,
                        &format!("preplan:search:{} {}", search_count, truncate_multiline(&result, 4000)),
                    );
                    extra_context.push_str("\nInspection result:\n");
                    extra_context.push_str(&result);
                    extra_context.push_str("\n\n");
                }
                PreplanAssistantResponse::Read { start: read_start, end: read_end } => {
                    if read_count >= max_reads {
                        bail!("preplan exceeded READ limit of {max_reads}");
                    }
                    read_count += 1;
                    let result = read_in_file(content, read_start, read_end)?;
                    log_debug(
                        log,
                        path_str,
                        &format!("preplan:read:{} {}", read_count, truncate_multiline(&result, 4000)),
                    );
                    extra_context.push_str("\nInspection result:\n");
                    extra_context.push_str(&result);
                    extra_context.push_str("\n\n");
                }
            }
        };

        let mut planned = match parse_edit_plan(&text) {
            Ok(steps) => steps,
            Err(e) => {
                log_debug(
                    log,
                    path_str,
                    &format!(
                        "preplan:window:{}-{} parse_failed {}",
                        start + 1,
                        end,
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
                "preplan:window:{}-{} parsed_steps={}\n{}",
                start + 1,
                end,
                planned.len(),
                truncate_multiline(&format_edit_plan_steps(&planned), 12000)
            ),
        );
        for step in &planned {
            if step.start_line() < start + 1 || step.end_line() > *end {
                log_debug(
                    log,
                    path_str,
                    &format!(
                        "preplan:window:{}-{} validation_failed planned step L{}-L{} outside window",
                        start + 1,
                        end,
                        step.start_line(),
                        step.end_line()
                    ),
                );
                bail!(
                    "planned step L{}-L{} falls outside window L{}-L{}",
                    step.start_line(),
                    step.end_line(),
                    start + 1,
                    end
                );
            }
        }
        steps.append(&mut planned);
        if steps.len() > MAX_PREPLAN_STEPS {
            steps.truncate(MAX_PREPLAN_STEPS);
        }
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
    Ok(steps)
}

async fn request_region_plan(
    path_str: &str,
    task: &str,
    content: &str,
    router: &ModelRouter,
    broad_error: &str,
    cancelled: Option<&AtomicBool>,
    log: Option<&SessionLog>,
) -> Result<Vec<EditRegion>> {
    ensure_not_cancelled(cancelled)?;
    let lines: Vec<&str> = content.lines().collect();
    let total_lines = lines.len();
    let windows = build_windows(total_lines, WINDOW_SIZE, WINDOW_OVERLAP);
    let mut regions = Vec::new();
    let mut search_count = 0usize;
    let mut read_count = 0usize;

    for (win_idx, (start, end)) in windows.iter().enumerate() {
        ensure_not_cancelled(cancelled)?;
        if regions.len() >= MAX_PLANNED_REGIONS {
            break;
        }

        let window_content = lines[*start..*end]
            .iter()
            .enumerate()
            .map(|(i, line)| format!("{:>4}│{}", start + i + 1, line))
            .collect::<Vec<_>>()
            .join("\n");
        let remaining = MAX_PLANNED_REGIONS - regions.len();

        let mut extra_context = String::new();
        let text = loop {
            let prompt = format!(
                "A broad patch for {path_str} failed.\n\
                 Failure: {broad_error}\n\
                 Original task: {task}\n\n\
                 You may first inspect the current file using these bounded commands:\n\
                 - SEARCH: <exact text>   (current file only, max {MAX_PREPLAN_SEARCHES} total)\n\
                 - READ: <start>-<end>    (current file only, max {MAX_PREPLAN_READS_REPAIR} total)\n\
                 After any inspection, return the final region plan or NO_REGIONS.\n\
                 Break the task into up to {remaining} small, non-overlapping line regions within this window.\n\
                 Each region must be the smallest contiguous block that can be edited independently.\n\
                 For code, prefer functions/classes/import blocks. For YAML/TOML/JSON/Markdown/config files, prefer logical sections or key blocks.\n\
                 If the task needs only one region in this window, return one region. If no region is needed in this window, output exactly NO_REGIONS.\n\n\
                 Output only one of these formats:\n\
                 SEARCH: <exact text>\n\n\
                 READ: <start>-<end>\n\n\
                 REGION <line>\n\
                 or\n\
                 REGION <start_line>-<end_line>\n\
                 TASK: <specific subtask for this region>\n\
                 END\n\n\
                 Window {}/{} lines {}-{} of {}:\n{window_content}\n\n\
                 {extra_context}",
                win_idx + 1,
                windows.len(),
                start + 1,
                end,
                total_lines
            );

            let request = ChatRequest {
                messages: vec![
                    Message::system(
                        "You output only strict REGION blocks or one SEARCH/READ command. No explanations, no markdown.",
                    ),
                    Message::user(&prompt),
                ],
                tools: None,
                tool_choice: None,
            };

            log_stage(
                log,
                path_str,
                &format!("split_plan:window:{}-{}", start + 1, end),
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
                    "split_plan:window:{}-{} raw_response:\n{}",
                    start + 1,
                    end,
                    truncate_multiline(text, 12000)
                ),
            );

            match parse_preplan_assistant_response(text)? {
                PreplanAssistantResponse::Plan(plan) => break plan,
                PreplanAssistantResponse::Search(query) => {
                    if search_count >= MAX_PREPLAN_SEARCHES {
                        bail!("split fallback exceeded SEARCH limit of {MAX_PREPLAN_SEARCHES}");
                    }
                    search_count += 1;
                    let result = search_in_file(content, &query);
                    log_debug(
                        log,
                        path_str,
                        &format!(
                            "split_plan:search:{} {}",
                            search_count,
                            truncate_multiline(&result, 4000)
                        ),
                    );
                    extra_context.push_str("Inspection result:\n");
                    extra_context.push_str(&result);
                    extra_context.push_str("\n\n");
                }
                PreplanAssistantResponse::Read { start: read_start, end: read_end } => {
                    if read_count >= MAX_PREPLAN_READS_REPAIR {
                        bail!(
                            "split fallback exceeded READ limit of {}",
                            MAX_PREPLAN_READS_REPAIR
                        );
                    }
                    read_count += 1;
                    let result = read_in_file(content, read_start, read_end)?;
                    log_debug(
                        log,
                        path_str,
                        &format!(
                            "split_plan:read:{} {}",
                            read_count,
                            truncate_multiline(&result, 4000)
                        ),
                    );
                    extra_context.push_str("Inspection result:\n");
                    extra_context.push_str(&result);
                    extra_context.push_str("\n\n");
                }
            }
        };

        let mut planned = match parse_region_plan(&text) {
            Ok(regions) => regions,
            Err(e) => {
                log_debug(
                    log,
                    path_str,
                    &format!(
                        "split_plan:window:{}-{} parse_failed {}",
                        start + 1,
                        end,
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
                "split_plan:window:{}-{} parsed_regions={}\n{}",
                start + 1,
                end,
                planned.len(),
                truncate_multiline(&format_regions_for_log(&planned), 12000)
            ),
        );
        for region in &planned {
            if region.start < start + 1 || region.end > *end {
                log_debug(
                    log,
                    path_str,
                    &format!(
                        "split_plan:window:{}-{} validation_failed planned region L{}-L{} outside window",
                        start + 1,
                        end,
                        region.start,
                        region.end
                    ),
                );
                bail!(
                    "planned region L{}-L{} falls outside window L{}-L{}",
                    region.start,
                    region.end,
                    start + 1,
                    end
                );
            }
        }
        regions.append(&mut planned);
        if regions.len() > MAX_PLANNED_REGIONS {
            regions.truncate(MAX_PLANNED_REGIONS);
        }
    }

    if let Err(e) = reject_overlapping_regions(&regions) {
        log_debug(
            log,
            path_str,
            &format!(
                "split_plan:file_validation_failed {}",
                truncate_multiline(&e.to_string(), 2000)
            ),
        );
        return Err(e);
    }
    Ok(regions)
}

fn validate_regions_in_file(regions: &[EditRegion], total_lines: usize) -> Result<()> {
    for region in regions {
        if region.end > total_lines {
            bail!(
                "planned region L{}-L{} falls outside file with {total_lines} lines",
                region.start,
                region.end
            );
        }
    }
    reject_overlapping_regions(regions)
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

fn format_regions_for_log(regions: &[EditRegion]) -> String {
    if regions.is_empty() {
        return "NO_REGIONS".to_string();
    }
    let mut out = String::new();
    for region in regions {
        if region.start == region.end {
            out.push_str(&format!("REGION {}\nTASK: {}\nEND\n\n", region.start, region.task));
        } else {
            out.push_str(&format!(
                "REGION {}-{}\nTASK: {}\nEND\n\n",
                region.start, region.end, region.task
            ));
        }
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

pub fn parse_edit_plan(text: &str) -> Result<Vec<EditPlanStep>> {
    if text.trim() == "NO_REGIONS" {
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

pub fn parse_region_plan(text: &str) -> Result<Vec<EditRegion>> {
    if text.trim() == "NO_REGIONS" {
        return Ok(Vec::new());
    }
    if text.trim().is_empty() {
        bail!("empty region plan");
    }

    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;
    let mut regions = Vec::new();

    while i < lines.len() {
        let line = lines[i];
        if line.trim().is_empty() {
            i += 1;
            continue;
        }

        let (region, next) = parse_region_at(&lines, i)?;
        i = next;
        regions.push(region);
    }

    if regions.len() > MAX_PLANNED_REGIONS {
        bail!(
            "region plan returned {} regions, maximum is {MAX_PLANNED_REGIONS}",
            regions.len()
        );
    }
    reject_overlapping_regions(&regions)?;
    Ok(regions)
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
    use tempfile::tempdir;

    #[test]
    fn signature_grounding_extracts_known_rust_callees() {
        let tmp = tempdir().unwrap();
        let project_root = tmp.path();
        std::fs::create_dir_all(project_root.join("src/cli/commands")).unwrap();
        std::fs::write(
            project_root.join("src/cli/commands/run.rs"),
            "pub async fn run(config: Config, message: &str, plan_only: bool, headless: bool) -> Result<()> {\n}\n",
        )
        .unwrap();
        std::fs::write(
            project_root.join("src/cli/commands/repl.rs"),
            "pub async fn run(config: Config, headless: bool) -> Result<()> {\n}\n",
        )
        .unwrap();

        let content = "\
miniswe::cli::commands::run::run(config, &message, true, cli.yes).await?;\n\
miniswe::cli::commands::repl::run(config, cli.yes).await?;\n";
        let note = build_signature_grounding_note(
            "src/main.rs",
            "Update main.rs to pass system_prompt_override to run and repl commands",
            content,
            project_root,
        )
        .unwrap();

        assert!(note.contains("src/cli/commands/run.rs:1"));
        assert!(note.contains("pub async fn run(config: Config, message: &str, plan_only: bool, headless: bool) -> Result<()>"));
        assert!(note.contains("src/cli/commands/repl.rs:1"));
    }

    #[test]
    fn retry_feedback_adds_signature_guidance_for_lsp_arity_errors() {
        let feedback = build_retry_feedback(
            "LSP diagnostics worsened for src/main.rs: 0 -> 3 error(s)\nsrc/main.rs:31:79: error: expected 4 arguments, found 5",
            Some("Current known callee signatures from the repo:\n- src/cli/commands/run.rs:1 `pub async fn run(...)`"),
        );

        assert!(feedback.contains("Do not repeat the same patch shape"));
        assert!(feedback.contains("Current known callee signatures"));
    }

    #[test]
    fn should_preplan_requires_broad_scope_language() {
        assert!(should_preplan(
            "Update all context::assemble() calls to include the new parameter"
        ));
        assert!(!should_preplan(
            "Update main.rs to pass system_prompt_override to run and repl commands"
        ));
    }

    #[test]
    fn parse_preplan_assistant_response_supports_search_and_read() {
        match parse_preplan_assistant_response("SEARCH: context::assemble").unwrap() {
            PreplanAssistantResponse::Search(query) => assert_eq!(query, "context::assemble"),
            _ => panic!("expected SEARCH"),
        }
        match parse_preplan_assistant_response("READ: 10-20").unwrap() {
            PreplanAssistantResponse::Read { start, end } => {
                assert_eq!(start, 10);
                assert_eq!(end, 20);
            }
            _ => panic!("expected READ"),
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
