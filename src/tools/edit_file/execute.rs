//! Entry point and step execution: applies the planned steps against the
//! in-memory working copy and drives the retry loop helpers.

use super::*;

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
    let path_str = match crate::tools::args::require_str(args, "path") {
        Ok(p) => p,
        Err(e) => return Ok(ToolResult::err(e)),
    };
    let task = match crate::tools::args::require_str(args, "task") {
        Ok(t) => t,
        Err(e) => return Ok(ToolResult::err(e)),
    };
    let lsp_validation = match LspValidationMode::from_args(args) {
        Ok(mode) => mode,
        Err(e) => return Ok(ToolResult::err(e.to_string())),
    };

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

pub(super) fn max_literal_replace_lines(context_window: usize) -> usize {
    match context_window {
        0..=32_000 => 8,
        32_001..=64_000 => 12,
        64_001..=128_000 => 20,
        128_001..=256_000 => 32,
        _ => 48,
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn execute_planned_steps(
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
) -> std::result::Result<SplitResult, Box<PlannedExecutionFailure>> {
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
    steps_desc.sort_by_key(|s| std::cmp::Reverse(s.start_line()));

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
                                return Err(Box::new(PlannedExecutionFailure {
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
                                }));
                            }
                            RelocateOutcome::NoCandidate => {
                                return Err(Box::new(PlannedExecutionFailure {
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
                                }));
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
                .map_err(|e| {
                    Box::new(PlannedExecutionFailure {
                        current_content: current.clone(),
                        message: format!(
                            "{message}Pre-plan {region_label} failed after {completed_count} completed step(s): {e}\n",
                        ),
                        error: format!("{region_label} failed: {e}"),
                        completed_steps: completed_records.clone(),
                        failed_step: Some(step.clone()),
                        lsp_regression: None,
                    })
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
        Box::new(PlannedExecutionFailure {
            current_content: current.clone(),
            message: format!(
                "{message}Pre-plan validation failed after {completed_count}/{planned_count} completed step(s): {error_summary}\n"
            ),
            error: error_summary,
            completed_steps: completed_records.clone(),
            failed_step: None,
            lsp_regression,
        })
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

pub(super) async fn execute_smart_step(
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

pub(super) async fn request_patch(
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
            max_tokens_override: None,
            chat_template_kwargs: Some(serde_json::json!({"enable_thinking": false})),
            temperature_override: None,
            cache_prompt: None,
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

pub(super) async fn request_patch_for_region(
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
        max_tokens_override: None,
        chat_template_kwargs: Some(serde_json::json!({"enable_thinking": false})),
        temperature_override: None,
        cache_prompt: None,
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

pub(super) fn validate_steps_in_file(steps: &[EditPlanStep], total_lines: usize) -> Result<()> {
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

pub(super) fn validate_candidate(original: &str, candidate: &str) -> Result<()> {
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
pub(super) fn gate_truncation(
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
