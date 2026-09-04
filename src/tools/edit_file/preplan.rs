//! Pre-plan phase: windowed observation, batch inspection, and the
//! finalize verdict that produces the structured edit plan.

use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) async fn execute_preplanned_steps(
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
                let e = *e;
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

pub(super) const MAX_PREPLAN_SEARCHES: usize = 20;
pub(super) const MAX_PREPLAN_READS_INITIAL: usize = 6;
pub(super) const MAX_PREPLAN_READS_REPAIR: usize = 10;

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
pub(super) fn parse_preplan_window_response(text: &str) -> PreplanWindowResponse {
    let mut notes = Vec::new();
    let mut commands = Vec::new();
    let mut clarification: Option<String> = None;

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        if clarification.is_none()
            && let Some(question) = parse_needs_clarification(line)
        {
            clarification = Some(question);
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
        if let Some(rest) = strip_case_insensitive_prefix(line, "SEARCH:") {
            let query = rest.trim();
            if !query.is_empty() {
                commands.push(InspectionCommand::Search(query.to_string()));
            }
            continue;
        }
        if let Some(rest) = strip_case_insensitive_prefix(line, "READ:") {
            let rest = rest.trim();
            if let Some((start_s, end_s)) = rest.split_once('-')
                && let (Ok(start), Ok(end)) = (
                    start_s.trim().parse::<usize>(),
                    end_s.trim().parse::<usize>(),
                )
                && start > 0
                && end >= start
            {
                commands.push(InspectionCommand::Read { start, end });
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
pub(super) fn extract_line_ranges_from_note(note: &str) -> Vec<(usize, usize)> {
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
                if let Ok(end) = end_str.parse::<usize>()
                    && end >= start
                {
                    ranges.push((start, end));
                    continue;
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

pub(super) fn has_case_insensitive_prefix(text: &str, prefix: &str) -> bool {
    text.get(..prefix.len())
        .map(|head| head.eq_ignore_ascii_case(prefix))
        .unwrap_or(false)
}

pub(super) fn strip_case_insensitive_prefix<'a>(text: &'a str, prefix: &str) -> Option<&'a str> {
    has_case_insensitive_prefix(text, prefix).then(|| &text[prefix.len()..])
}

pub(super) fn search_in_file(content: &str, query: &str) -> String {
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

pub(super) fn read_in_file(content: &str, start: usize, end: usize) -> Result<String> {
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

pub(super) fn render_numbered_slice(lines: &[&str], start: usize, end: usize) -> String {
    lines[start..end.min(lines.len())]
        .iter()
        .enumerate()
        .map(|(offset, line)| format!("{:>4}│{}", start + offset + 1, line))
        .collect::<Vec<_>>()
        .join("\n")
}

pub(super) fn extend_unique_notes(existing: &mut Vec<String>, new_notes: Vec<String>) {
    for note in new_notes {
        if !existing.contains(&note) {
            existing.push(note);
        }
    }
}

pub(super) fn append_inspection_result(extra_context: &mut String, label: &str, result: &str) {
    extra_context.push_str(label);
    extra_context.push('\n');
    extra_context.push_str(result);
    extra_context.push_str("\n\n");
}

/// Execute the SEARCH/READ commands collected across the windowed
/// observation pass, appending their formatted results to `extra_context`
/// for the finalize prompt. Commands that exceed the per-edit caps are
/// dropped with an inline note rather than erroring out.
pub(super) fn execute_inspection_commands(
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

pub(super) async fn request_preplan_steps(
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
            max_tokens_override: None,
            chat_template_kwargs: Some(serde_json::json!({"enable_thinking": false})),
            temperature_override: None,
            cache_prompt: None,
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
        max_tokens_override: None,
        chat_template_kwargs: Some(serde_json::json!({"enable_thinking": false})),
        temperature_override: None,
        cache_prompt: None,
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
