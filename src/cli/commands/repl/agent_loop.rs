//! The REPL agent loop: LLM call → tool execution → repeat, plus the
//! live plan panel and responsive compression.

use super::*;

/// Run the agent loop (LLM call → tool execution → repeat).
/// Refresh the live plan panel from `plan.md` (the single source of truth).
/// No-op when no task is active (Q&A turns). Called both at the top of each
/// round and immediately after the plan tool runs, so a checked-off step shows
/// the instant `plan(check)` returns rather than lagging to the next round.
pub(super) fn refresh_plan_panel(app: &mut App, config: &Config, round: usize) {
    if app.plan_task.is_none() {
        return;
    }
    app.plan_steps = tools::plan::parsed_steps(config)
        .into_iter()
        .map(|(checked, checked_round, text)| PlanStepView {
            checked,
            checked_round,
            text,
        })
        .collect();
    app.round = round;
}

/// `compressor::force_compress` driven through the same select!-with-redraw
/// pattern the round loop uses for `maybe_compress`, so a long LLM-based
/// summarization during reactive context-exhaustion recovery keeps the TUI
/// responsive. Returns force_compress's "did anything shrink" result.
#[allow(clippy::too_many_arguments)]
pub(super) async fn force_compress_responsive(
    app: &mut App,
    rx: &mut mpsc::UnboundedReceiver<AppEvent>,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    messages: &mut Vec<Message>,
    config: &Config,
    router: &ModelRouter,
    llm_worker: &LlmWorkerHandle,
    tool_def_tokens: usize,
) -> bool {
    let fut =
        context::compressor::force_compress(messages, config, router, llm_worker, tool_def_tokens);
    let mut fut = std::pin::pin!(fut);
    loop {
        tokio::select! {
            biased;
            freed = &mut fut => break freed,
            evt = rx.recv() => {
                if matches!(evt, Some(AppEvent::Tick)) {
                    let _ = terminal.draw(|frame| ui::draw(frame, app));
                }
            }
        }
    }
}

/// This runs inline in the main loop, processing events between rounds.
#[allow(clippy::too_many_arguments)]
pub(super) async fn run_agent_loop(
    app: &mut App,
    rx: &mut mpsc::UnboundedReceiver<AppEvent>,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    router: &Arc<ModelRouter>,
    llm_worker: &LlmWorkerHandle,
    tool_pool: &ToolWorkerPool,
    tool_defs: &[crate::llm::ToolDefinition],
    config: &Config,
    // Explore mode: hard-block mutating tool calls at runtime (read-only shell ok).
    read_only: bool,
    perms: &Arc<PermissionManager>,
    mcp_registry: &Option<Arc<Mutex<McpRegistry>>>,
    cancelled: &Arc<AtomicBool>,
    messages: &mut Vec<Message>,
    conversation_history: &mut Vec<Message>,
    max_rounds: usize,
    log: Arc<SessionLog>,
    lsp: &Option<Arc<LspClient>>,
    fast_revisions: &Option<Arc<tools::RevisionStore>>,
    fast_baseline_errors: usize,
    snapshots: &Option<Arc<Mutex<tools::snapshots::SnapshotManager>>>,
    tool_def_tokens: usize,
    mcp_summary: Option<&str>,
    // The original user message for this turn — the recovery goal used by the
    // done-gate re-anchor, the debugger sub-agent, and whole-tree SCRAP/reset
    // re-assembly.
    goal: &str,
    // Session-scoped background-job registry (file shell background=true).
    job_registry: &Arc<tools::jobs::JobRegistry>,
) {
    // Ceremony=Strict re-enables the legacy plan-first machinery (plan gate,
    // plan/no-plan nudges, hide-edit-tools-until-plan). Derived from the
    // per-turn config, which already has ceremony forced Off for explore turns.
    let strict = config.tools.ceremony == CeremonyMode::Strict;
    let pause_at = config.context.pause_after_rounds;

    let mut round = 0;
    let mut had_error = false;
    let mut user_continued = false;
    // Track consecutive identical tool calls to detect loops
    let mut last_call_key: Option<String> = None;
    let mut same_call_streak = 0u32;
    // Short rolling history of call keys for period-2 cycle detection
    // (edit↔revert oscillation — invisible to the consecutive detector).
    let mut recent_call_keys: Vec<String> = Vec::new();
    // Number of distinct loops the model has been pulled out of in this
    // turn. We give one recovery; a second loop ends the turn for real.
    let mut loop_recoveries = 0u32;
    // Read-loop escalation ladder (see run.rs for the full rationale): first
    // detection gets REPEATED_READ_NUDGE; a re-detection forces a context
    // compaction before the next request — breaking the cache-hot prefix is
    // what actually ends the loop. Resets after each escalation.
    let mut read_nudges = 0u32;
    let mut force_compact_next_round = false;
    let mut calls_since_last_edit = 0u32;
    let mut successful_edits_since_plan_update = 0u32;
    let mut plan_update_requested = false;
    let mut nudged_premature_exit = false;
    let mut nudged_no_plan = false;
    // Consecutive reactive-compaction retries (context exhaustion signaled
    // by the server — see compressor::force_compress). Reset whenever a
    // response is successfully consumed; bounds futile retries of one
    // failing request, not total compactions over a long turn.
    let mut context_compact_retries: usize = 0;
    // Consecutive LLM requests that died on a tool-call argument problem
    // (server-side parse error or our streaming size cap) with no completed
    // response in between. See run.rs for the escalation ladder.
    let mut truncated_call_errors_in_a_row: usize = 0;
    // How many times the behavioral done-gate has blocked completion this turn.
    let mut validation_blocks: usize = 0;
    // The model's stated rationale each time the gate blocked it (bounded, auditable).
    let mut validation_disputes: Vec<String> = Vec::new();
    // Reactive-debugger / restart / replan bookkeeping (each fires at most once
    // per turn; debugger_multifire walks the failure chain up to MAX fires).
    let mut replan_fired = false;
    let mut restart_fired = false;
    let mut debugger_fires = 0usize;
    let mut last_debugged_failure: Option<String> = None;
    // `plan_gate_debugger`: consecutive plan(check) failures on the SAME step.
    let mut same_plan_step_failures: u32 = 0;
    let mut last_failed_plan_step: Option<u64> = None;
    // Gate-triggered context resets fired this turn (bounded — don't loop).
    let mut gate_resets: usize = 0;
    // Spiral-reset: per-file revert counts + how many resets fired this turn.
    let mut revert_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    let mut spiral_resets: usize = 0;
    // revert-to-green state (opt-in `tools.revert_to_green`): the last round
    // whose start-of-round snapshot was green (project errors ≤ baseline) and
    // how many consecutive rounds the project has stayed broken.
    const REVERT_TO_GREEN_BLOCKS: usize = 6;
    let mut last_green_round: usize = 0;
    let mut red_streak: usize = 0;
    // Ceremony-gate latch — see run.rs for the rationale. Reset on every
    // SCRAP, which reassembles the context and restarts the ceremony.
    let mut plan_ever_set = false;

    'round: loop {
        if had_error {
            break;
        }
        // Check cancellation at the top of every round
        if consume_interrupt(cancelled) {
            app.push_output("(interrupted)", LineStyle::Status);
            break;
        }

        round += 1;
        log.round_start(round);

        // Snapshot at the start of each round for revert support (SCRAP /
        // revert-to-green rely on these per-round commits in the shadow repo).
        if let Some(snap) = snapshots {
            let mut guard = snap.lock();
            let _ = guard.begin_round(round);
        }

        // revert-to-green: if the project has been broken above baseline for
        // REVERT_TO_GREEN_BLOCKS rounds, the agent is digging deeper, not
        // recovering — reset the whole tree to the last green snapshot.
        if config.tools.revert_to_green
            && config.tools.edit_mode == EditMode::Fast
            && let Some(snap) = snapshots
        {
            let errs = tools::fast::project_error_count(lsp.as_deref()).await;
            if errs <= fast_baseline_errors {
                last_green_round = round;
                red_streak = 0;
            } else {
                red_streak += 1;
                if red_streak >= REVERT_TO_GREEN_BLOCKS {
                    let result = {
                        let guard = snap.lock();
                        guard.revert_to_round(last_green_round)
                    };
                    match result {
                        Ok(m) => {
                            app.push_output(
                                &format!("[revert-to-green] stuck {red_streak} rounds; {m}"),
                                LineStyle::Status,
                            );
                            messages.push(Message::user(&format!(
                                "[auto-revert-to-green] The project has had compile errors for \
                                 {red_streak} rounds straight and you are not converging — you are \
                                 digging deeper, not recovering. I reverted the ENTIRE working tree \
                                 to round {last_green_round}, the last state that compiled cleanly. \
                                 Your edits since then are GONE; do not replay them. Start over from \
                                 this clean base: re-read the relevant code, make ONE small complete \
                                 change, and run a check before continuing."
                            )));
                            red_streak = 0;
                        }
                        Err(e) => {
                            app.push_output(
                                &format!("[revert-to-green] revert failed: {e}"),
                                LineStyle::Error,
                            );
                        }
                    }
                }
            }
        }

        if round > max_rounds {
            app.push_output("Maximum tool rounds reached.", LineStyle::Error);
            break;
        }

        // Ask the user whether to continue after pause_after_rounds rounds.
        if round == pause_at && !user_continued {
            app.pending_permission = Some(format!(
                "{pause_at} tool rounds used. Continue? [y]es / [n]o:"
            ));
            app.input.clear();
            app.cursor = 0;
            let response = wait_for_modal_input(app, rx, terminal, &['y', 'n']).await;
            app.pending_permission = None;
            match response.as_str() {
                "y" | "yes" | "" => user_continued = true,
                _ => messages.push(Message::user("[Stop now. Summarize what you've done.]")),
            }
        }

        // Warn the LLM when approaching the hard limit.
        if round == max_rounds.saturating_sub(5) {
            messages.push(Message::user(
                "[Approaching tool limit. Wrap up and summarize.]",
            ));
        }

        // Refresh the live plan panel from plan.md (single source of truth).
        refresh_plan_panel(app, config, round);

        // Unified context compression — handles both tool results and
        // conversation, every round (matching run.rs). Driven through a
        // select! so a long LLM-based summarization keeps the TUI responsive.
        {
            // See `agent::prune_reads` — surgically drop the middle of a deep
            // run of identical reads, which compaction structurally cannot
            // reach (it summarizes the oldest end; the loop is in the newest).
            let pruned = prune_repeated_reads(messages);
            if !pruned.is_empty() {
                log.reads_pruned(
                    pruned.removed / 2,
                    pruned.keys,
                    pruned.deepest.as_deref().unwrap_or("?"),
                );
                app.push_output(
                    &format!(
                        "  ⋯ pruned {} repeated calls from context",
                        pruned.removed / 2
                    ),
                    LineStyle::Status,
                );
            }
            let pre = messages.len();
            // Read-loop escalation (see REPEATED_READ_ESCALATION): the loop
            // is sustained by the cache-hot prompt prefix, so break it
            // deliberately even though no budget pressure asks for it. Runs
            // before maybe_compress so refresh_current_state still lands on
            // the tail.
            if force_compact_next_round {
                force_compact_next_round = false;
                app.push_output(
                    "  ⚠ Read loop persisted — forcing context compaction",
                    LineStyle::Status,
                );
                force_compress_responsive(
                    app,
                    rx,
                    terminal,
                    messages,
                    config,
                    router,
                    llm_worker,
                    tool_def_tokens,
                )
                .await;
            }
            {
                let compress_fut = context::compressor::maybe_compress(
                    messages,
                    config,
                    router,
                    llm_worker,
                    tool_def_tokens,
                    &mut plan_update_requested,
                );
                let mut compress_fut = std::pin::pin!(compress_fut);
                let mut done = false;
                while !done {
                    tokio::select! {
                        biased;
                        () = &mut compress_fut, if !done => { done = true; }
                        evt = rx.recv() => {
                            if matches!(evt, Some(AppEvent::Tick)) {
                                let _ = terminal.draw(|frame| ui::draw(frame, app));
                            }
                        }
                    }
                }
            }
            log.masking_applied(pre.saturating_sub(messages.len()), pre);
        }

        // Sanitize messages
        context::sanitize_messages(messages);

        // Hide edit tools until a plan exists; see visible_tool_defs.
        let plan_set = tools::plan::plan_exists(config);
        plan_ever_set |= plan_set;
        // Off: never hide edit tools (pass plan_exists=true). Strict: legacy
        // hide-until-plan behavior, latched so a plan that goes away mid-segment
        // cannot retract tools the model has already been shown.
        let visible = visible_tool_defs(tool_defs, plan_ever_set || !strict);
        // Build request. See run.rs for the per-model reasoning_effort and
        // thinking-mode logic.
        let (chat_template_kwargs, temperature_override) =
            if config.model.is_mistral_small_4_family() {
                let effort = if plan_set { "none" } else { "high" };
                (serde_json::json!({"reasoning_effort": effort}), None)
            } else if config.model.thinking {
                (
                    serde_json::json!({"enable_thinking": true}),
                    Some(config.model.thinking_temperature),
                )
            } else {
                (serde_json::json!({"enable_thinking": false}), None)
            };
        // Bump output budget for Mistral 4 — see run.rs for rationale
        // (probe data: 8K truncates with empty content, 16K emits clean
        // correct output at ~6K tokens used).
        let max_tokens_override = if config.model.is_mistral_small_4_family() {
            Some(16384)
        } else {
            None
        };
        let request = ChatRequest {
            messages: messages.clone(),
            tools: Some(visible),
            tool_choice: None,
            max_tokens_override,
            chat_template_kwargs: Some(chat_template_kwargs),
            temperature_override,
            cache_prompt: None,
        };
        log.llm_request(&request);

        cancelled.store(false, Ordering::Relaxed);
        app.is_thinking = true;
        app.set_active_job("llm");

        // Render before LLM call so spinner is visible immediately
        let _ = terminal.draw(|frame| ui::draw(frame, app));

        // Call LLM with streaming — render on each token
        let mut rendered_assistant_text = String::new();
        // Set when the server rejected the model's tool call as
        // truncated JSON. In that case we inject a synthetic user-role
        // hint and continue the outer loop so the agent can recover
        // with a smaller operation, instead of aborting the session.
        let mut truncated_tool_call_hint_pushed = false;
        // Set when the failure is really CONTEXT EXHAUSTION (the server
        // rejected an over-size request, or clipped a tool call because the
        // prompt sits near the window). Handled after the select loop —
        // force_compress is a long await that must not run inside it.
        let mut context_ceiling_hit = false;
        let response = {
            let mut token_count = 0u32;
            let mut llm_events =
                llm_worker.submit(ModelRole::Default, request.clone(), cancelled.clone());
            loop {
                tokio::select! {
                    evt = llm_events.recv() => {
                        match evt {
                            Some(LlmWorkerEvent::Token(token)) => {
                                app.push_token(&token);
                                rendered_assistant_text.push_str(&token);
                                token_count += 1;
                                if token_count.is_multiple_of(3) {
                                    let _ = terminal.draw(|frame| ui::draw(frame, app));
                                }
                            }
                            Some(LlmWorkerEvent::Completed(Ok(r))) => break Some(r),
                            Some(LlmWorkerEvent::Completed(Err(err_str))) => {
                                if err_str.contains("Interrupted") {
                                    cancelled.store(false, Ordering::Relaxed);
                                    app.push_output("Generation interrupted.", LineStyle::Status);
                                } else if is_context_exceeded_error(&err_str)
                                    && context_compact_retries
                                        < context::compressor::FORCE_COMPRESS_MAX_RETRIES
                                {
                                    // Prompt alone exceeds the context window
                                    // — recoverable by compacting + resending
                                    // (primary path for compaction="lazy",
                                    // safety net for every other strategy).
                                    context_ceiling_hit = true;
                                } else if is_tool_call_args_cap_error(&err_str) {
                                    // Our streaming assembler aborted the
                                    // generation: an anchor-only tool's
                                    // arguments outgrew the cap. Nothing was
                                    // persisted; hint and retry, or give up
                                    // when the model keeps doing it.
                                    truncated_call_errors_in_a_row += 1;
                                    if truncated_call_errors_in_a_row
                                        >= TRUNCATED_CALL_ABORT_AFTER
                                    {
                                        log.llm_error(&format!(
                                            "{truncated_call_errors_in_a_row} consecutive oversized tool calls — aborting turn"
                                        ));
                                        // Falls through to `break None` below:
                                        // no hint flag set, so the turn ends.
                                        app.push_output(
                                            "The model keeps emitting oversized tool-call arguments — giving up on this turn.",
                                            LineStyle::Error,
                                        );
                                    } else {
                                    log.llm_error(&format!(
                                        "tool call aborted by the argument size cap: {err_str}"
                                    ));
                                    app.push_output(
                                        "Tool call arguments exceeded the size cap — retrying with guidance.",
                                        LineStyle::Status,
                                    );
                                    let hint = Message::user(&format!(
                                        "{err_str}. Anchor-style tools take identifiers and short expressions only — \
                                         never paste code bodies into their arguments. {}",
                                        truncated_tool_call_hint(config.tools.edit_mode)
                                    ));
                                    messages.push(hint.clone());
                                    conversation_history.push(hint);
                                    truncated_tool_call_hint_pushed = true;
                                    }
                                } else if is_truncated_tool_call_error(&err_str) {
                                    // The server's chat template could not
                                    // parse some assistant tool call's
                                    // arguments as JSON: either this
                                    // response was cut off mid-call
                                    // (nothing persisted), or a previously
                                    // persisted call is broken and every
                                    // request will fail until it is gone.
                                    // Handle the second first — it is a
                                    // zero-progress spin otherwise.
                                    truncated_call_errors_in_a_row += 1;
                                    let scrubbed = if truncated_call_errors_in_a_row >= 2 {
                                        scrub_unparseable_tool_calls(messages)
                                            + scrub_unparseable_tool_calls(conversation_history)
                                    } else {
                                        0
                                    };
                                    if scrubbed > 0 {
                                        log.llm_error(&format!(
                                            "scrubbed {scrubbed} unparseable tool call(s) from history after repeated parse failures — retrying"
                                        ));
                                        app.push_output(
                                            "Repaired a truncated tool call left in history — retrying.",
                                            LineStyle::Status,
                                        );
                                        truncated_tool_call_hint_pushed = true;
                                    } else if truncated_call_errors_in_a_row
                                        >= TRUNCATED_CALL_ABORT_AFTER
                                    {
                                        log.llm_error(&format!(
                                            "{truncated_call_errors_in_a_row} consecutive tool-call parse failures with nothing left to repair — aborting turn"
                                        ));
                                        // Falls through to `break None`: turn ends.
                                        app.push_output(
                                            "The server keeps rejecting tool-call arguments — giving up on this turn.",
                                            LineStyle::Error,
                                        );
                                    } else if context::compressor::estimated_context_tokens(
                                        messages,
                                        tool_def_tokens,
                                    ) > config.model.context_window * 3 / 4
                                        && context_compact_retries
                                            < context::compressor::FORCE_COMPRESS_MAX_RETRIES
                                    {
                                        context_ceiling_hit = true;
                                    } else {
                                        // Clear the partial UI text (don't
                                        // persist the half-streamed output)
                                        // and push a user-role hint so the
                                        // agent retries with a smaller
                                        // operation.
                                        log.llm_error(
                                            "tool call JSON truncated (max_tokens) — \
                                             injecting hint and continuing",
                                        );
                                        app.push_output(
                                            "Previous tool call truncated — retrying with guidance.",
                                            LineStyle::Status,
                                        );
                                        let hint = Message::user(truncated_tool_call_hint(
                                            config.tools.edit_mode,
                                        ));
                                        messages.push(hint.clone());
                                        conversation_history.push(hint);
                                        truncated_tool_call_hint_pushed = true;
                                    }
                                } else {
                                    let clean = if err_str.contains('<') {
                                        err_str
                                            .split('<')
                                            .next()
                                            .unwrap_or(&err_str)
                                            .trim()
                                            .to_string()
                                    } else {
                                        err_str
                                    };
                                    log.llm_error(&clean);
                                    app.push_output(&format!("LLM error: {clean}"), LineStyle::Error);
                                }
                                app.clear_active_job();
                                break None;
                            }
                            None => {
                                app.push_output("LLM worker stopped unexpectedly.", LineStyle::Error);
                                app.clear_active_job();
                                break None;
                            }
                        }
                    }
                    app_evt = rx.recv() => {
                        match app_evt {
                            Some(AppEvent::Tick) => {
                                let _ = terminal.draw(|frame| ui::draw(frame, app));
                            }
                            Some(AppEvent::Key(key)) if handle_background_key(app, &key) => {
                                let _ = terminal.draw(|frame| ui::draw(frame, app));
                            }
                            Some(AppEvent::Key(key)) if event::is_ctrl_c(&key) => {
                                cancelled.store(true, Ordering::Relaxed);
                                app.push_output("(interrupted)", LineStyle::Status);
                                let _ = terminal.draw(|frame| ui::draw(frame, app));
                            }
                            Some(AppEvent::Mouse(_)) => {}
                            Some(AppEvent::PermissionRequest(prompt, response_tx)) => {
                                let response =
                                    fulfill_permission_request(app, rx, terminal, prompt).await;
                                let _ = response_tx.send(response);
                            }
                            Some(_) => {}
                            None => {
                                app.push_output("Event stream closed.", LineStyle::Error);
                                app.clear_active_job();
                                break None;
                            }
                        }
                    }
                }
            }
        };

        app.clear_active_job();

        // Re-render after LLM response
        let _ = terminal.draw(|frame| ui::draw(frame, app));

        let Some(response) = response else {
            if context_ceiling_hit {
                context_compact_retries += 1;
                app.push_output(
                    "Context window exceeded — compacting and retrying.",
                    LineStyle::Status,
                );
                if force_compress_responsive(
                    app,
                    rx,
                    terminal,
                    messages,
                    config,
                    router,
                    llm_worker,
                    tool_def_tokens,
                )
                .await
                {
                    log.llm_error("context window exceeded — compacted history, retrying");
                    continue;
                }
                // Nothing could be freed — retrying would fail identically.
                app.push_output(
                    "Compaction could not free any context — stopping this turn.",
                    LineStyle::Error,
                );
                break;
            }
            if truncated_tool_call_hint_pushed {
                // Hint was injected into `messages`; loop back and let
                // the agent try again with smaller operations.
                continue;
            }
            break;
        };

        // A 200 response can still be a context-exhaustion casualty:
        // finish_reason="length" with the completion well under the
        // requested cap means the server clipped generation at n_ctx (see
        // run.rs / is_context_truncated_response). Discard the partial
        // output, compact, regenerate.
        let effective_max_tokens =
            max_tokens_override.unwrap_or(config.model.max_output_tokens as u64) as usize;
        if is_context_truncated_response(&response, effective_max_tokens)
            && context::compressor::estimated_context_tokens(messages, tool_def_tokens)
                > config.model.context_window * 3 / 4
            && context_compact_retries < context::compressor::FORCE_COMPRESS_MAX_RETRIES
        {
            context_compact_retries += 1;
            app.push_output(
                "Generation truncated by context ceiling — compacting and regenerating.",
                LineStyle::Status,
            );
            if force_compress_responsive(
                app,
                rx,
                terminal,
                messages,
                config,
                router,
                llm_worker,
                tool_def_tokens,
            )
            .await
            {
                log.llm_error(
                    "generation truncated by context ceiling — compacted history, regenerating",
                );
                continue;
            }
        }

        let choice = match response.choices.first() {
            Some(c) => c,
            None => break,
        };
        // A response made it through whole — any prior reactive-compaction
        // retries resolved this request; reset the budget for the next one.
        context_compact_retries = 0;
        truncated_call_errors_in_a_row = 0;

        // Never persist an unparseable tool call (see run.rs): stub the
        // cut-off arguments; the tool loop answers the stub with guidance.
        let mut assistant_msg = choice.message.clone();
        let truncated_calls = sanitize_truncated_tool_calls(&mut assistant_msg);
        if truncated_calls > 0 {
            log.llm_error(&format!(
                "{truncated_calls} tool call(s) arrived with unparseable arguments (cut off by the output limit) — stubbed before persisting"
            ));
            app.push_output(
                "A tool call was cut off by the output limit — it will not be executed.",
                LineStyle::Status,
            );
        }
        let assistant_msg = &assistant_msg;

        // Flush any remaining tokens
        app.flush_tokens();

        if let Some(content) = &assistant_msg.content {
            log.llm_response(content);
            if let Some(missing) =
                reconcile_streamed_assistant_content(&rendered_assistant_text, content)
            {
                app.push_token(&missing);
                app.flush_tokens();
                let _ = terminal.draw(|frame| ui::draw(frame, app));
            }
        }
        if assistant_msg.is_meaningful() {
            conversation_history.push(assistant_msg.clone());
        }

        let tool_calls = match &assistant_msg.tool_calls {
            Some(tc) if !tc.is_empty() => tc.clone(),
            _ => {
                // See run.rs for rationale — nudge on both "mid-plan exit"
                // and "no-plan exit" (the latter caught Mistral Small 4
                // bailing during exploration before any meaningful work).
                // Strict/legacy only.
                if strict && !nudged_premature_exit && config.tools.plan {
                    let has_unchecked = tools::plan::has_unchecked_steps(config);
                    let plan_exists = tools::plan::plan_exists(config);
                    if has_unchecked || !plan_exists {
                        nudged_premature_exit = true;
                        let nudge_text = if plan_exists {
                            PREMATURE_EXIT_NUDGE.to_string()
                        } else {
                            "[You returned no tool call before setting a plan. \
                             Don't exit yet — call plan(action='set') with your \
                             step-by-step approach (or file/code if you need more \
                             exploration). The task isn't done.]"
                                .to_string()
                        };
                        let nudge = Message::user(&nudge_text);
                        messages.push(nudge.clone());
                        conversation_history.push(nudge);
                        continue;
                    }
                }

                // Behavioral done-gate: before accepting completion, verify the
                // change actually works at runtime. Default config has no
                // command → no-op. See docs/success-validation-design.md.
                // Skipped for read-only (explore) turns — a Q&A turn makes no
                // edits, so there is nothing to behaviorally verify and blocking
                // the answer would be nonsensical.
                if !read_only
                    && validation_blocks < config.validation.max_retries
                    && config.validation.command().is_some()
                {
                    match validation::run_behavioral_check(config).await {
                        validation::CheckOutcome::Fail(output) => {
                            validation_blocks += 1;
                            // Record the model's completion rationale (its
                            // no-tool-call exit content) — a bounded, auditable
                            // voice, not a silent free pass.
                            if let Some(rationale) = assistant_msg
                                .content
                                .as_deref()
                                .map(str::trim)
                                .filter(|c| !c.is_empty())
                            {
                                tracing::warn!(
                                    "[validation] blocked completion (attempt {validation_blocks}); model rationale: {}",
                                    crate::truncate_chars(rationale, 300)
                                );
                                validation_disputes.push(rationale.to_string());
                            }
                            app.push_output(
                                "Behavioral check failed — not done yet.",
                                LineStyle::Status,
                            );

                            // Full restart (opt-in `gate_restart`): on the FIRST
                            // gate block, abandon the possibly-poisoned attempt —
                            // revert the WHOLE tree to the clean baseline AND
                            // reset the context. Fires once per turn.
                            if config.tools.gate_restart && !restart_fired {
                                restart_fired = true;
                                plan_ever_set = false;
                                *messages =
                                    scrap_restart(app, config, goal, mcp_summary, snapshots, false);
                                conversation_history.clear();
                                validation_blocks = 0;
                                same_plan_step_failures = 0;
                                last_failed_plan_step = None;
                                continue;
                            }

                            // Goal re-anchor (opt-in `gate_replan`): re-anchor on
                            // the ORIGINAL goal and force a fresh plan — but skip
                            // when the block is a COMPILE failure (re-anchoring on
                            // a broken tree just digs deeper).
                            let is_compile_fail = output.contains("DOES NOT COMPILE")
                                || output.contains("could not compile")
                                || output.contains("error[E");
                            if config.tools.gate_replan && !replan_fired && !is_compile_fail {
                                replan_fired = true;
                                app.push_output(
                                    "Re-anchoring on the original goal — re-plan from the task…",
                                    LineStyle::Status,
                                );
                                let msg = Message::user(&format!(
                                    "[A check that exercises the change end-to-end FAILED — it \
                                     COMPILES but does not yet BEHAVE as required. After fixing \
                                     errors it is easy to lose the original goal and stop at \"it \
                                     compiles\". Re-anchor on the task: \"{goal}\". Use \
                                     plan(action='set') to re-derive the FULL plan from that goal — \
                                     list every step the feature needs end-to-end, INCLUDING the \
                                     code that actually USES the new input to change behavior (not \
                                     just declaring or plumbing it). For each step, confirm it is \
                                     DONE in the code, not merely compiling — then implement \
                                     whatever is missing before finishing.\nCheck output:\n{output}]"
                                ));
                                messages.push(msg.clone());
                                conversation_history.push(msg);
                                continue;
                            }

                            // Reactive debugger (opt-in): hand the SPECIFIC
                            // failure to a fresh-context sub-agent once the
                            // primary agent has failed the gate a couple times.
                            let fkey = crate::cli::commands::run::failure_key(&output);
                            let may_fire = if config.tools.debugger_multifire {
                                debugger_fires < debugger::MAX_DEBUGGER_FIRES
                                    && last_debugged_failure.as_deref() != Some(fkey.as_str())
                            } else {
                                debugger_fires == 0
                            };
                            if (config.tools.reactive_debugger || config.tools.debugger_judge)
                                && may_fire
                                && validation_blocks >= debugger::DEBUGGER_TRIGGER_BLOCKS
                            {
                                debugger_fires += 1;
                                last_debugged_failure = Some(fkey);
                                app.push_output(
                                    "Still failing — spinning up a fresh-context debugger sub-agent…",
                                    LineStyle::Status,
                                );
                                let verdict = debugger::run_debugger(
                                    &output,
                                    goal,
                                    config,
                                    llm_worker,
                                    tool_pool,
                                    tool_defs,
                                    perms,
                                    mcp_registry,
                                    lsp,
                                    fast_revisions,
                                    fast_baseline_errors,
                                    cancelled,
                                )
                                .await;

                                let msg = match verdict {
                                    debugger::DebuggerVerdict::Scrap if !restart_fired => {
                                        restart_fired = true;
                                        plan_ever_set = false;
                                        *messages = scrap_restart(
                                            app,
                                            config,
                                            goal,
                                            mcp_summary,
                                            snapshots,
                                            true,
                                        );
                                        conversation_history.clear();
                                        validation_blocks = 0;
                                        same_plan_step_failures = 0;
                                        last_failed_plan_step = None;
                                        continue;
                                    }
                                    debugger::DebuggerVerdict::Scrap => Message::user(
                                        "[A fresh-context review voted to reset again, but the \
                                         tree was already reset once this turn. Keep going: read \
                                         the current failure carefully and fix it directly.]",
                                    ),
                                    debugger::DebuggerVerdict::Rewind(candidate) => {
                                        rewind_message_repl(
                                            app,
                                            &candidate,
                                            config,
                                            perms,
                                            lsp,
                                            fast_revisions,
                                            fast_baseline_errors,
                                            &output,
                                        )
                                        .await
                                    }
                                    debugger::DebuggerVerdict::Report(body) => {
                                        let output_note =
                                            crate::cli::commands::run::write_gate_failure_output(
                                                config, &output,
                                            )
                                            .map(|path| {
                                                format!(
                                                    "\nFull raw check output: read(\"{path}\")."
                                                )
                                            })
                                            .unwrap_or_default();
                                        Message::user(&format!(
                                            "[A read-only debugger with fresh eyes investigated the failing \
                                         check and produced this DIAGNOSIS. It did not edit anything — \
                                         YOU must apply the fix and finish the plan it lays out:\n{body}\n\
                                         Make the change(s), then finish; the verification will re-run.{output_note}]"
                                        ))
                                    }
                                };
                                messages.push(msg.clone());
                                conversation_history.push(msg);
                                continue;
                            }

                            // Gate context-reset (opt-in): drop the polluted
                            // history and re-assemble a clean context (files
                            // persist on disk). Bounded per turn.
                            if config.tools.gate_context_reset
                                && gate_resets < spiral::MAX_GATE_RESETS
                                && validation_blocks >= spiral::GATE_RESET_AFTER_BLOCKS
                            {
                                gate_resets += 1;
                                validation_blocks = 0;
                                let fresh = spiral::build_gate_reset_prompt(goal, &output);
                                let assembled =
                                    context::assemble(config, &fresh, &[], false, mcp_summary);
                                *messages = assembled.messages;
                                app.push_output(
                                    "Gate context-reset — fresh start (history cleared, files kept).",
                                    LineStyle::Status,
                                );
                                log.tool_debug(
                                    "agent",
                                    "gate context-reset: re-assembled clean context after repeated gate blocks",
                                );
                                continue;
                            }

                            let msg = Message::user(&format!(
                                "[Verification failed — do NOT finish yet. A check that exercises \
                                 the change end-to-end exited non-zero; the output below shows what \
                                 is actually wrong. Read it carefully and fix the SPECIFIC problem \
                                 it reports (it may be a compile error, not a logic error), then \
                                 continue. (If you are certain the check itself is wrong, finish \
                                 anyway and state the specific reason — it will be recorded.)\n\
                                 Check output:\n{output}]"
                            ));
                            messages.push(msg.clone());
                            conversation_history.push(msg);
                            continue;
                        }
                        validation::CheckOutcome::Pass | validation::CheckOutcome::Skipped => {}
                    }
                }
                // Exiting now. Surface any recorded gate rationale(s) for audit.
                if !validation_disputes.is_empty() {
                    app.push_output(
                        &format!(
                            "Completed after {} blocked verification(s); model's reasons recorded in the log.",
                            validation_disputes.len()
                        ),
                        LineStyle::Status,
                    );
                    tracing::warn!(
                        "[validation] turn completed over {} blocked check(s); model rationale(s): {}",
                        validation_disputes.len(),
                        validation_disputes.join(" | ")
                    );
                }
                break;
            }
        };

        messages.push(assistant_msg.clone());

        // See run.rs for the rationale — both buffers' last entry is the
        // assistant_msg we just pushed, so truncate one before to also
        // drop it. If every tool call in this assistant message turns out
        // to be a prunable validator failure, we rewind here and replace
        // with a single user-role corrective.
        let messages_pre = messages.len() - 1;
        let history_pre = if conversation_history
            .last()
            .is_some_and(|m| m.role == "assistant")
        {
            conversation_history.len() - 1
        } else {
            conversation_history.len()
        };
        let mut all_prunable_failures = !tool_calls.is_empty();
        let mut prunable_errors: Vec<String> = Vec::new();

        // Execute tool calls
        for tc in &tool_calls {
            // Check cancellation between tool calls
            if consume_interrupt(cancelled) {
                app.push_output("(interrupted)", LineStyle::Status);
                break 'round;
            }
            let args: serde_json::Value = match serde_json::from_str(&tc.function.arguments) {
                Ok(v) => v,
                Err(e) => {
                    let result_msg = Message::tool_result(
                        &tc.id,
                        &format!("Invalid JSON in tool arguments: {e}"),
                    );
                    messages.push(result_msg.clone());
                    conversation_history.push(result_msg);
                    app.push_output(
                        &format!("  ✗ {}: invalid JSON args", tc.function.name),
                        LineStyle::ToolErr,
                    );
                    continue;
                }
            };
            if let Some(info) = truncated_args_info(&args) {
                // Stubbed by sanitize_truncated_tool_calls: nothing to run.
                let result_msg = Message::tool_result(
                    &tc.id,
                    &format!(
                        "{}\n\n{}",
                        truncated_args_tool_result(&tc.function.name, &info),
                        truncated_tool_call_hint(config.tools.edit_mode)
                    ),
                );
                messages.push(result_msg.clone());
                conversation_history.push(result_msg);
                app.push_output(
                    &format!(
                        "  ✗ {}: arguments cut off by the output limit after {} chars — not executed",
                        tc.function.name, info.original_chars
                    ),
                    LineStyle::ToolErr,
                );
                continue;
            }

            let args_summary = summarize_args(&tc.function.name, &args);

            // Detect tool call loops: identical calls repeated consecutively
            // (period-1), or the SAME two calls alternating (period-2 — the
            // edit↔revert oscillation the streak counter is blind to).
            let call_key = loop_call_key(&tc.function.name, &args);
            if last_call_key.as_ref() == Some(&call_key) {
                same_call_streak += 1;
            } else {
                last_call_key = Some(call_key.clone());
                same_call_streak = 1;
            }
            recent_call_keys.push(call_key.clone());
            if recent_call_keys.len() > 12 {
                recent_call_keys.remove(0);
            }
            let cycle = cycle_period(&recent_call_keys);
            if same_call_streak >= 3 || cycle.is_some() {
                // Cycle-only detection (not also a plain streak).
                let cycle_only = cycle.filter(|_| same_call_streak < 3);
                // A cycle is harmful if ANY member mutates.
                let mutating = if let Some(period) = cycle_only {
                    let tail = &recent_call_keys[recent_call_keys.len().saturating_sub(period)..];
                    tail.iter().any(|k| key_is_mutating(k))
                } else {
                    is_mutating_call(&tc.function.name, &args)
                };
                log.loop_detected(&tc.function.name, &args_summary, same_call_streak as usize);

                // Read-only repetition: harmless per call, just wasted tokens.
                // First detection: polite nudge, let processing continue.
                // Re-detection: escalate — the nudge can't reach a
                // cache-numerics rut, so force a compaction next round.
                if !mutating {
                    read_nudges += 1;
                    let escalate = read_nudges >= 2;
                    let text = if escalate {
                        read_nudges = 0;
                        force_compact_next_round = true;
                        REPEATED_READ_ESCALATION
                    } else {
                        REPEATED_READ_NUDGE
                    };
                    let result_msg = Message::tool_result(&tc.id, text);
                    messages.push(result_msg.clone());
                    conversation_history.push(result_msg);
                    app.push_output(
                        &format!(
                            "  ⓘ Repeated read: {}({}) — {}, continuing",
                            tc.function.name,
                            args_summary,
                            if escalate {
                                "nudge failed, forcing compaction next round"
                            } else {
                                "nudge sent"
                            }
                        ),
                        LineStyle::Status,
                    );
                    last_call_key = None;
                    same_call_streak = 0;
                    recent_call_keys.clear();
                    continue;
                }

                let hint = if let Some(period) = cycle_only {
                    cycle_loop_hint(period)
                } else {
                    loop_detected_hint(config.tools.edit_mode).to_string()
                };
                let result_msg = Message::tool_result(&tc.id, &hint);
                messages.push(result_msg.clone());
                conversation_history.push(result_msg);

                // First mutating loop in this turn: surface the hint, reset
                // the streak, and let the model try a different approach.
                if loop_recoveries == 0 {
                    loop_recoveries += 1;
                    last_call_key = None;
                    same_call_streak = 0;
                    recent_call_keys.clear();
                    app.push_output(
                        &format!(
                            "  ⚠ Loop detected: {}({}) {} — surfacing a hint, giving the model one more round",
                            tc.function.name,
                            args_summary,
                            if let Some(period) = cycle_only {
                                format!("cycling through the same {period} calls (period-{period} cycle)")
                            } else {
                                "repeated 3 times".to_string()
                            }
                        ),
                        LineStyle::Status,
                    );
                    break;
                }

                // Second mutating loop after the recovery hint. With a
                // behavioral done-gate configured this is NOT a dead end — it
                // is the same "stuck but the task isn't done" state as a
                // premature exit, so route it through the gate ladder instead
                // of dying with the whole recovery stack idle.
                if !read_only
                    && config.validation.command().is_some()
                    && validation_blocks < config.validation.max_retries
                {
                    app.push_output(
                        &format!(
                            "  Loop detected again ({}({})) — routing through the done-gate instead of stopping",
                            tc.function.name, args_summary
                        ),
                        LineStyle::Status,
                    );
                    if let validation::CheckOutcome::Fail(output) =
                        validation::run_behavioral_check(config).await
                    {
                        validation_blocks += 1;
                        // Fresh recovery budget for the rounds the gate grants.
                        loop_recoveries = 0;
                        last_call_key = None;
                        same_call_streak = 0;
                        recent_call_keys.clear();

                        let fkey = crate::cli::commands::run::failure_key(&output);
                        let may_fire = if config.tools.debugger_multifire {
                            debugger_fires < debugger::MAX_DEBUGGER_FIRES
                                && last_debugged_failure.as_deref() != Some(fkey.as_str())
                        } else {
                            debugger_fires == 0
                        };
                        if (config.tools.reactive_debugger || config.tools.debugger_judge)
                            && may_fire
                            && validation_blocks >= debugger::DEBUGGER_TRIGGER_BLOCKS
                        {
                            debugger_fires += 1;
                            last_debugged_failure = Some(fkey);
                            app.push_output(
                                "Looping + failing gate — spinning up a fresh-context debugger sub-agent…",
                                LineStyle::Status,
                            );
                            let verdict = debugger::run_debugger(
                                &output,
                                goal,
                                config,
                                llm_worker,
                                tool_pool,
                                tool_defs,
                                perms,
                                mcp_registry,
                                lsp,
                                fast_revisions,
                                fast_baseline_errors,
                                cancelled,
                            )
                            .await;

                            let msg = match verdict {
                                debugger::DebuggerVerdict::Scrap if !restart_fired => {
                                    restart_fired = true;
                                    plan_ever_set = false;
                                    *messages = scrap_restart(
                                        app,
                                        config,
                                        goal,
                                        mcp_summary,
                                        snapshots,
                                        true,
                                    );
                                    conversation_history.clear();
                                    validation_blocks = 0;
                                    same_plan_step_failures = 0;
                                    last_failed_plan_step = None;
                                    continue 'round;
                                }
                                debugger::DebuggerVerdict::Scrap => Message::user(
                                    "[A fresh-context review voted to reset again, but the tree \
                                     was already reset once this turn. Keep going: read the \
                                     current failure carefully and fix it directly.]",
                                ),
                                debugger::DebuggerVerdict::Rewind(candidate) => {
                                    rewind_message_repl(
                                        app,
                                        &candidate,
                                        config,
                                        perms,
                                        lsp,
                                        fast_revisions,
                                        fast_baseline_errors,
                                        &output,
                                    )
                                    .await
                                }
                                debugger::DebuggerVerdict::Report(body) => {
                                    let output_note =
                                        crate::cli::commands::run::write_gate_failure_output(
                                            config, &output,
                                        )
                                        .map(|path| {
                                            format!("\nFull raw check output: read(\"{path}\").")
                                        })
                                        .unwrap_or_default();
                                    Message::user(&format!(
                                        "[A read-only debugger with fresh eyes investigated the failing \
                                     check and produced this DIAGNOSIS. It did not edit anything — \
                                     YOU must apply the fix and finish the plan it lays out:\n{body}\n\
                                     Make the change(s), then finish; the verification will re-run.{output_note}]"
                                    ))
                                }
                            };
                            messages.push(msg.clone());
                            conversation_history.push(msg);
                            continue 'round;
                        }

                        let msg = Message::user(&format!(
                            "[Your repeated failing tool call was aborted, and the task is NOT \
                             done — the verification check failed:\n{output}\nRead the check \
                             output and your tool errors carefully, fix the SPECIFIC problem, \
                             and use a correctly-formed call (include every required parameter) \
                             or a different tool.]"
                        ));
                        messages.push(msg.clone());
                        conversation_history.push(msg);
                        continue 'round;
                    }
                    // Gate passed (or skipped): the loop was on something the
                    // check doesn't care about — fall through to the stop.
                }
                app.push_output(
                    &format!(
                        "  ✗ Loop detected again ({}({})) after the recovery hint — stopping this turn",
                        tc.function.name, args_summary
                    ),
                    LineStyle::Error,
                );
                had_error = true;
                break;
            }

            log.tool_call_detail(&tc.function.name, &args);
            app.push_output(
                &format!("  → {}({})", tc.function.name, args_summary),
                LineStyle::ToolCall,
            );

            // Re-render to show tool call
            let _ = terminal.draw(|frame| ui::draw(frame, app));

            // Read-only investigation (explore) mode: hard-block any mutating
            // tool call at runtime, BEFORE any permission prompt. The def filter
            // and the prompt are advisory — shell can still mutate and the model
            // can emit tools that aren't in the list. Read-only shell is allowed.
            if read_only {
                let file_action = args["action"].as_str().unwrap_or("");
                if let Some(reason) = explore_block_reason(&tc.function.name, file_action, &args) {
                    let msg = Message::tool_result(
                        &tc.id,
                        &format!(
                            "[blocked: read-only investigation mode] {reason}. To change code or \
                             run mutating shell commands, switch to coding (e.g. say \"actually, \
                             change it\")."
                        ),
                    );
                    messages.push(msg.clone());
                    conversation_history.push(msg);
                    app.push_output(
                        &format!("  ⛔ {}: blocked — read-only mode", tc.function.name),
                        LineStyle::ToolErr,
                    );
                    continue;
                }
            }

            // Determine if this tool call needs a permission prompt
            let perm_action = permission_action(&tc.function.name, &args);

            // Check permission via TUI prompt (not raw stderr)
            let mut perm_denied = false;
            if let Some(ref action) = perm_action
                && matches!(action, Action::Shell(_) | Action::McpUse(_, _))
            {
                match perms.check_needs_prompt(action) {
                    Err(e) => {
                        // Blocklisted — skip this tool call
                        let result_msg = Message::tool_result(&tc.id, &e);
                        messages.push(result_msg.clone());
                        conversation_history.push(result_msg);
                        app.push_output(
                            &format!("  ✗ {}: {e}", tc.function.name),
                            LineStyle::ToolErr,
                        );
                        continue;
                    }
                    Ok(Some(prompt)) => {
                        // Needs user approval — show prompt in TUI
                        app.pending_permission = Some(prompt);
                        app.input.clear();
                        app.cursor = 0;
                        let _ = terminal.draw(|frame| ui::draw(frame, app));

                        // Wait for user input (y/n/a)
                        let response = wait_for_permission_input(app, rx, terminal).await;
                        app.pending_permission = None;

                        match response.as_str() {
                            "y" | "yes" => {
                                perms.approve(action, false);
                                app.push_output(
                                    "  · Permission granted, running tool...",
                                    LineStyle::Status,
                                );
                            }
                            "a" | "always" => {
                                perms.approve(action, true);
                                app.push_output(
                                    "  · Permission granted and saved, running tool...",
                                    LineStyle::Status,
                                );
                            }
                            _ => {
                                perm_denied = true;
                                app.push_output("  · Permission denied.", LineStyle::Status);
                            }
                        }

                        let _ = terminal.draw(|frame| ui::draw(frame, app));
                    }
                    Ok(None) => {} // No prompt needed
                }
            }

            if perm_denied {
                let result_msg =
                    Message::tool_result(&tc.id, &format!("{} denied by user", tc.function.name));
                messages.push(result_msg.clone());
                conversation_history.push(result_msg);
                app.push_output(
                    &format!("  ✗ {}: denied", tc.function.name),
                    LineStyle::ToolErr,
                );
                continue;
            }

            let file_action = args["action"].as_str().unwrap_or("");

            // Write gating: require a plan before write tools (strict only).
            let is_write_action = is_file_write(tc.function.name.as_str());
            if strict && config.tools.plan && !tools::plan::plan_exists(config) && is_write_action {
                let result_msg = Message::tool_result(
                    &tc.id,
                    "Create a plan first: use plan(action='set') with your step-by-step approach before making changes.",
                );
                messages.push(result_msg.clone());
                conversation_history.push(result_msg);
                app.push_output(
                    &format!("  ✗ {}: blocked — no plan", tc.function.name),
                    LineStyle::ToolErr,
                );
                continue;
            }
            // (Plan-checkpoint used to hard-block writes after N edits without
            //  a plan action; that interacted poorly with the compile-gate on
            //  `plan(check)` — if the project didn't compile, the model
            //  couldn't escape the block, couldn't fix the project, deadlock.
            //  Now we just warn at the threshold via PLAN_CHECKPOINT_WARNING
            //  appended to the tool result; the model decides what to do.)

            // Execute tool (permissions already checked above for shell/web/mcp)
            let mut result = if matches!(
                tc.function.name.as_str(),
                "replace_range" | "insert_at" | "revert" | "show_rev" | "check"
            ) && config.tools.edit_mode == EditMode::Fast
            {
                let tool_name = tc.function.name.clone();
                let args = args.clone();
                let config = config.clone();
                let perms = perms.clone();
                let lsp = lsp.clone();
                let revisions = fast_revisions.clone();
                let baseline = fast_baseline_errors;
                let mut result_rx = tool_pool.submit(move || {
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .map_err(|e| e.to_string())?;
                    let Some(revisions) = revisions else {
                        return Ok(crate::tools::ToolResult::err(
                            "fast mode: revision store unavailable".into(),
                        ));
                    };
                    runtime
                        .block_on(async move {
                            tools::execute_fast_tool(
                                &tool_name,
                                &args,
                                &config,
                                perms.as_ref(),
                                lsp.as_deref(),
                                revisions.as_ref(),
                                baseline,
                            )
                            .await
                        })
                        .map_err(|e| format!("fast tool error: {e}"))
                });
                await_tool_job_ui(
                    rx,
                    terminal,
                    app,
                    &tc.function.name,
                    &mut result_rx,
                    cancelled,
                )
                .await
            } else if tc.function.name == "mcp_use" {
                let server = args["server"].as_str().unwrap_or("").to_string();
                let tool = args["tool"].as_str().unwrap_or("").to_string();
                let tool_args = args.get("arguments").cloned().unwrap_or_default();
                if server.is_empty() || tool.is_empty() {
                    crate::tools::ToolResult::err(
                        "mcp_use requires top-level 'server' and 'tool' string fields. \
                         Example: {\"server\": \"my-server\", \"tool\": \"my-tool\", \"arguments\": {}}".into(),
                    )
                } else {
                    let registry = mcp_registry.clone();
                    let mut result_rx = tool_pool.submit(move || match registry {
                        Some(registry) => {
                            let mut guard = registry.lock();
                            guard
                                .call_tool(&server, &tool, tool_args)
                                .map(crate::tools::ToolResult::ok)
                                .map_err(|e| format!("MCP error: {e}"))
                        }
                        None => Ok(crate::tools::ToolResult::err(
                            "No MCP servers connected".into(),
                        )),
                    });
                    await_tool_job_ui(rx, terminal, app, "mcp_use", &mut result_rx, cancelled).await
                }
            } else if tc.function.name == "spawn_agents" {
                let tasks = crate::cli::commands::agent::subagent::parse_tasks(&args);
                if tasks.is_empty() {
                    crate::tools::ToolResult::err(
                        "spawn_agents: 'agents' must be a non-empty array of {label, prompt}"
                            .into(),
                    )
                } else {
                    let (out_tx, mut out_rx) =
                        tokio::sync::mpsc::unbounded_channel::<(String, LineStyle)>();
                    let subagents_fut = crate::cli::commands::agent::subagent::run_subagents(
                        tasks,
                        config,
                        llm_worker,
                        tool_pool,
                        tool_defs,
                        perms,
                        mcp_registry,
                        lsp,
                        fast_revisions,
                        fast_baseline_errors,
                        cancelled,
                        Some(out_tx),
                    );
                    let mut subagents_fut = std::pin::pin!(subagents_fut);
                    let mut outputs = None;
                    while outputs.is_none() {
                        tokio::select! {
                            biased;
                            r = &mut subagents_fut, if outputs.is_none() => { outputs = Some(r); }
                            line = out_rx.recv() => {
                                if let Some((text, style)) = line {
                                    app.push_output(&text, style);
                                    let _ = terminal.draw(|frame| ui::draw(frame, app));
                                }
                            }
                            evt = rx.recv() => {
                                if matches!(evt, Some(AppEvent::Tick)) {
                                    let _ = terminal.draw(|frame| ui::draw(frame, app));
                                }
                            }
                        }
                    }
                    let combined =
                        crate::cli::commands::agent::subagent::format_outputs(outputs.unwrap());
                    crate::tools::ToolResult::ok(combined)
                }
            } else if tc.function.name == "edit_file" {
                let args = args.clone();
                let config = config.clone();
                let perms = perms.clone();
                let router = router.clone();
                let lsp = lsp.clone();
                let cancelled_for_job = cancelled.clone();
                let log_for_job = log.clone();
                let mut result_rx = tool_pool.submit(move || {
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .map_err(|e| e.to_string())?;
                    runtime
                        .block_on(async move {
                            crate::tools::execute_edit_file_tool(
                                &args,
                                &config,
                                perms.as_ref(),
                                router.as_ref(),
                                lsp.as_deref(),
                                Some(cancelled_for_job.as_ref()),
                                Some(log_for_job.as_ref()),
                            )
                            .await
                        })
                        .map_err(|e| format!("edit_file error: {e}"))
                });
                await_tool_job_ui(rx, terminal, app, "edit_file", &mut result_rx, cancelled).await
            } else if tc.function.name == "plan" {
                let args = args.clone();
                let config_for_job = config.clone();
                let mut result_rx = tool_pool.submit(move || {
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .map_err(|e| e.to_string())?;
                    runtime
                        .block_on(async move {
                            tools::plan::execute(&args, &config_for_job, round).await
                        })
                        .map_err(|e| format!("plan error: {e}"))
                });
                let r =
                    await_tool_job_ui(rx, terminal, app, "plan", &mut result_rx, cancelled).await;
                // The plan tool just mutated plan.md mid-round — refresh the
                // panel and redraw now so a checked/added/refined step appears
                // immediately instead of lagging to the next round's refresh.
                refresh_plan_panel(app, config, round);
                let _ = terminal.draw(|frame| ui::draw(frame, app));
                r
            } else if tc.function.name == "refactor" {
                let args = args.clone();
                let config = config.clone();
                let router = router.clone();
                let lsp = lsp.clone();
                let log_for_job = log.clone();
                let revisions_for_job = fast_revisions.clone();
                let cancelled_for_job = cancelled.clone();
                let mut result_rx = tool_pool.submit(move || {
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .map_err(|e| e.to_string())?;
                    runtime
                        .block_on(async move {
                            crate::tools::execute_refactor_tool(
                                &args,
                                &config,
                                router.as_ref(),
                                lsp.as_deref(),
                                Some(log_for_job.as_ref()),
                                revisions_for_job.as_deref(),
                                Some(cancelled_for_job.as_ref()),
                            )
                            .await
                        })
                        .map_err(|e| format!("refactor error: {e}"))
                });
                await_tool_job_ui(rx, terminal, app, "refactor", &mut result_rx, cancelled).await
            } else if (tc.function.name == "shell" && args["action"].as_str() == Some("run"))
                || (tc.function.name == "file" && file_action == "shell")
            {
                if args["background"].as_bool() == Some(true) {
                    // Explicit background start (cheap, non-blocking) —
                    // registered in the session job registry, managed via
                    // the jobs tool in this or any later turn.
                    tools::jobs::start_background(&args, config, job_registry.as_ref())
                } else {
                    await_shell_job_repl(
                        tool_pool.submit_shell(args.clone(), config.clone(), cancelled.clone()),
                        app,
                        rx,
                        terminal,
                        cancelled,
                    )
                    .await
                }
            } else if tc.function.name == "shell" {
                // Runs on the pool (own runtime) so jobs(wait) keeps the TUI
                // responsive via await_tool_job_ui, like other pooled tools.
                let args_for_job = args.clone();
                let config_for_job = config.clone();
                let perms_for_job = perms.clone();
                let registry_for_job = job_registry.clone();
                let cancelled_for_job = cancelled.clone();
                let mut result_rx = tool_pool.submit(move || {
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .map_err(|e| e.to_string())?;
                    Ok(runtime.block_on(async {
                        tools::jobs::execute(
                            &args_for_job,
                            &config_for_job,
                            perms_for_job.as_ref(),
                            registry_for_job.as_ref(),
                            Some(cancelled_for_job.as_ref()),
                        )
                        .await
                    }))
                });
                await_tool_job_ui(rx, terminal, app, "jobs", &mut result_rx, cancelled).await
            } else {
                let tool_name = tc.function.name.clone();
                let args = args.clone();
                let config = config.clone();
                let perms = perms.clone();
                let lsp = lsp.clone();
                let mut result_rx = tool_pool.submit(move || {
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .map_err(|e| e.to_string())?;
                    runtime
                        .block_on(async move {
                            tools::execute_tool(
                                &tool_name,
                                &args,
                                &config,
                                perms.as_ref(),
                                lsp.as_deref(),
                            )
                            .await
                        })
                        .map_err(|e| format!("Tool error: {e}"))
                });
                await_tool_job_ui(
                    rx,
                    terminal,
                    app,
                    &tc.function.name,
                    &mut result_rx,
                    cancelled,
                )
                .await
            };

            if !result.success
                && let Some(hint) = tools::plan::failure_hint(config)
            {
                result.content.push('\n');
                result.content.push_str(&hint);
            }

            // Append round number to every tool result.
            result
                .content
                .push_str(&format!("\n[round {round}/{max_rounds}]"));

            let first_line = result.content.lines().next().unwrap_or("(empty)");
            log.tool_call(&tc.function.name, &args_summary, result.success, first_line);
            log.tool_result_detail(&tc.function.name, result.success, &result.content);
            let style = if result.success {
                LineStyle::ToolOk
            } else {
                LineStyle::ToolErr
            };
            let icon = if result.success { "✓" } else { "✗" };
            app.push_output(
                &format!("  {icon} {}: {first_line}", tc.function.name),
                style,
            );
            app.store_tool_result(&tc.function.name, &result.content);

            if result.success && tc.function.name == "plan" {
                successful_edits_since_plan_update = 0;
            }

            // Successful file write = code changed, reset loop/stall trackers.
            if result.success && is_file_write(tc.function.name.as_str()) {
                last_call_key = None;
                same_call_streak = 0;
                calls_since_last_edit = 0;
                if strict && config.tools.plan {
                    if tools::plan::plan_exists(config) {
                        result.content.push('\n');
                        result.content.push_str(PLAN_PROGRESS_NUDGE);
                    }
                    successful_edits_since_plan_update += 1;
                    if successful_edits_since_plan_update == PLAN_CHECKPOINT_AFTER_EDITS {
                        result.content.push('\n');
                        result.content.push_str(PLAN_CHECKPOINT_WARNING);
                    }
                }
            } else {
                calls_since_last_edit += 1;
            }

            if !is_prunable_refactor_failure(&result.content, result.success) {
                all_prunable_failures = false;
            } else {
                prunable_errors.push(result.content.clone());
            }

            let result_msg = Message::tool_result(&tc.id, &result.content);
            messages.push(result_msg.clone());
            conversation_history.push(result_msg);

            // `plan_gate_debugger`: the plan tool's OWN compile gate repeatedly
            // blocking the SAME step is a distinct stall signature from the
            // behavioral done-gate (`validation_blocks`) — the primary agent is
            // re-litigating one step rather than making forward progress.
            if tc.function.name == "plan"
                && args.get("action").and_then(|a| a.as_str()) == Some("check")
            {
                if result.success {
                    same_plan_step_failures = 0;
                    last_failed_plan_step = None;
                } else if let Some(step) = args.get("step").and_then(|s| s.as_u64()) {
                    crate::cli::commands::run::track_plan_step_failure(
                        &mut last_failed_plan_step,
                        &mut same_plan_step_failures,
                        step,
                    );

                    let fkey = crate::cli::commands::run::failure_key(&result.content);
                    let may_fire = if config.tools.debugger_multifire {
                        debugger_fires < debugger::MAX_DEBUGGER_FIRES
                            && last_debugged_failure.as_deref() != Some(fkey.as_str())
                    } else {
                        debugger_fires == 0
                    };
                    if config.tools.plan_gate_debugger
                        && may_fire
                        && same_plan_step_failures as usize >= debugger::DEBUGGER_TRIGGER_BLOCKS
                    {
                        debugger_fires += 1;
                        last_debugged_failure = Some(fkey);
                        app.push_output(
                            "Plan-check gate failing repeatedly on the same step — spinning up a fresh-context debugger sub-agent…",
                            LineStyle::Status,
                        );
                        let verdict = debugger::run_debugger(
                            &result.content,
                            goal,
                            config,
                            llm_worker,
                            tool_pool,
                            tool_defs,
                            perms,
                            mcp_registry,
                            lsp,
                            fast_revisions,
                            fast_baseline_errors,
                            cancelled,
                        )
                        .await;

                        let extra_msg = match verdict {
                            debugger::DebuggerVerdict::Scrap if !restart_fired => {
                                restart_fired = true;
                                plan_ever_set = false;
                                *messages =
                                    scrap_restart(app, config, goal, mcp_summary, snapshots, true);
                                conversation_history.clear();
                                validation_blocks = 0;
                                same_plan_step_failures = 0;
                                last_failed_plan_step = None;
                                continue 'round;
                            }
                            debugger::DebuggerVerdict::Scrap => Message::user(
                                "[A fresh-context review voted to reset again, but the tree \
                                 was already reset once this turn. Keep going: read the \
                                 current failure carefully and fix it directly.]",
                            ),
                            debugger::DebuggerVerdict::Rewind(candidate) => {
                                rewind_message_repl(
                                    app,
                                    &candidate,
                                    config,
                                    perms,
                                    lsp,
                                    fast_revisions,
                                    fast_baseline_errors,
                                    &result.content,
                                )
                                .await
                            }
                            debugger::DebuggerVerdict::Report(body) => {
                                let output_note =
                                    crate::cli::commands::run::write_gate_failure_output(
                                        config,
                                        &result.content,
                                    )
                                    .map(|path| {
                                        format!("\nFull raw check output: read(\"{path}\").")
                                    })
                                    .unwrap_or_default();
                                Message::user(&format!(
                                    "[A read-only debugger with fresh eyes investigated the failing \
                                 plan-check step and produced this DIAGNOSIS. It did not edit \
                                 anything — YOU must apply the fix and finish the step it lays \
                                 out:\n{body}\nMake the change(s), then re-check the step.{output_note}]"
                                ))
                            }
                        };
                        messages.push(extra_msg.clone());
                        conversation_history.push(extra_msg);
                    }
                }
            }

            // Spiral-reset: a revert-loop (same file reverted repeatedly) means
            // the agent is cycling on the same failing edits. Inject a cognitive
            // reset (names what failed + forces a replan + concrete redirection).
            if config.tools.spiral_reset
                && result.success
                && tc.function.name == "revert"
                && config.tools.edit_mode == EditMode::Fast
                && spiral_resets < spiral::MAX_RESETS_PER_TURN
                && let Some(path) = args.get("path").and_then(|p| p.as_str())
            {
                let count = revert_counts.entry(path.to_string()).or_insert(0);
                *count += 1;
                if *count >= spiral::SPIRAL_REVERT_THRESHOLD {
                    let n = *count;
                    *count = 0;
                    spiral_resets += 1;
                    let tried = fast_revisions
                        .as_deref()
                        .map(|r| spiral::tried_edit_labels(r, path, 4))
                        .unwrap_or_default();
                    let reset = Message::user(&spiral::build_reset_message(path, n, &tried));
                    messages.push(reset.clone());
                    conversation_history.push(reset);
                    app.push_output(
                        "Spiral detected (revert-loop) — reset + replan injected.",
                        LineStyle::Status,
                    );
                    log.tool_debug(
                        "agent",
                        &format!("spiral-reset fired for {path} after {n} reverts"),
                    );
                }
            }

            // Re-render after tool result
            let _ = terminal.draw(|frame| ui::draw(frame, app));
        }

        // History pruning — see run.rs for rationale.
        if all_prunable_failures && !prunable_errors.is_empty() {
            messages.truncate(messages_pre);
            conversation_history.truncate(history_pre);
            let hint = Message::user(&format!(
                "Your previous refactor call(s) were rejected:\n\n{}\n\n\
                 Retry with all required parameters and a clean position value \
                 (one of 'start' or 'after:<single_param_name>').",
                prunable_errors.join("\n\n---\n\n")
            ));
            messages.push(hint.clone());
            conversation_history.push(hint);
            log.tool_debug(
                "agent",
                &format!(
                    "history pruned: dropped {} tool_result(s) after refactor validator failure",
                    prunable_errors.len()
                ),
            );
        }

        // Early no-plan nudge (strict only): edit tools are hidden until
        // plan(action='set'). Nudge around round 12 so a model that ignores the
        // system prompt gets a course correction before it's deeply stuck.
        if strict && round >= 12 && !nudged_no_plan && !tools::plan::plan_exists(config) {
            let unlock_tools = "refactor, replace_range, insert_at, write_file";
            messages.push(Message::user(&format!(
                "[Reminder: you've explored for several rounds without a plan. \
                 Call plan(action='set') with your step-by-step approach now — \
                 the edit tools ({unlock_tools}) are hidden until you do, and \
                 you'll need them to make changes.]"
            )));
            nudged_no_plan = true;
        }

        // Stall detection: too many tool calls without any edits. Content is
        // plan-state aware — without a plan the edit tools are hidden, so
        // re-fire the plan nudge instead of pointing at hidden tools.
        if calls_since_last_edit >= 20 && calls_since_last_edit.is_multiple_of(20) {
            let body = if strict && !tools::plan::plan_exists(config) {
                "Still no plan set after 20+ exploration calls. \
                 Edit tools cannot appear in your tool list until plan(action='set') is called. \
                 Stop exploring and set a plan now — even an imperfect plan can be refined later. \
                 If something is blocking you from planning, say so."
                    .to_string()
            } else {
                let edit_hint = match config.tools.edit_mode {
                    EditMode::Smart => "Use edit_file for semantic file edits.",
                    EditMode::Fast => "Use replace_range or insert_at to land targeted edits.",
                };
                format!(
                    "You have used 20+ tool calls without making any edits. \
                     You likely have enough information. Start making changes now. \
                     {edit_hint} \
                     If you're stuck, explain what's blocking you."
                )
            };
            messages.push(Message::user(&format!("[WARNING: {body}]")));
        }
    }

    log.session_end(round, had_error);
}
