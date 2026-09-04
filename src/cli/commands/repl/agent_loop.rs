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

    let mut round = 0;
    let mut had_error = false;
    // Per-turn agent-loop state shared with the headless loop (see the field
    // docs on `turn_state::TurnState` and its sub-structs).
    let mut state = turn_state::TurnState::default();
    // Unused in the REPL (`opts.skill_steps` is false, which short-circuits every
    // read of it); the shared phases take it by reference and Stage 4's driver
    // will own both state structs.
    let mut skill_state = turn_state::SkillTurnState::default();
    let mut ui = TuiUi::new(app, rx, terminal);

    // Explicit behavior deltas for the shared turn phases — see
    // `turn::TurnOptions` field docs.
    let opts = turn::TurnOptions {
        skill_steps: false,
        clear_cancel_on_interrupt: true,
        worker_stopped_ends_turn: true,
        compaction: turn::CompactionUx::Interactive,
        read_only,
        live_jobs_gate: false,
        failure_tracking: false,
        stuck_tracking: false,
    };

    'round: loop {
        if had_error {
            break;
        }
        // Check cancellation at the top of every round
        if consume_interrupt(cancelled) {
            ui.notify_interrupted();
            break;
        }

        round += 1;
        log.round_start(round);

        let ctx = turn::TurnCtx {
            config,
            router,
            llm_worker,
            lsp,
            snapshots,
            log: &log,
            tool_defs,
            cancelled,
            model_role: ModelRole::Default,
            fast_baseline_errors,
            tool_def_tokens,
            max_rounds,
            perms,
            tool_pool,
            mcp_registry,
            fast_revisions,
            job_registry,
            task: goal,
            mcp_summary,
            plan_only: false,
        };
        match turn::preamble::begin_round(ctx, &mut state, &mut ui, messages, round).await {
            turn::RoundFlow::Continue => {}
            turn::RoundFlow::EndTurn { .. } => break,
        }

        let assistant_msg = match turn::llm_call::generate(
            ctx,
            opts,
            &mut state,
            &mut ui,
            messages,
            conversation_history,
        )
        .await
        {
            turn::LlmFlow::Ready(msg) => msg,
            turn::LlmFlow::Retry => continue,
            turn::LlmFlow::EndTurn { .. } => break,
        };
        let assistant_msg = &assistant_msg;

        let tool_calls = match &assistant_msg.tool_calls {
            Some(tc) if !tc.is_empty() => tc.clone(),
            _ => match turn::done_gate::check(
                ctx,
                opts,
                &mut state,
                &mut skill_state,
                &mut ui,
                messages,
                conversation_history,
                assistant_msg,
            )
            .await
            {
                turn::RoundFlow::Continue => continue,
                turn::RoundFlow::EndTurn { .. } => break,
            },
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
                ui.notify_interrupted();
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
                    ui.app.push_output(
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
                ui.app.push_output(
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
            if state.loops.last_call_key.as_ref() == Some(&call_key) {
                state.loops.same_call_streak += 1;
            } else {
                state.loops.last_call_key = Some(call_key.clone());
                state.loops.same_call_streak = 1;
            }
            state.loops.recent_call_keys.push(call_key.clone());
            if state.loops.recent_call_keys.len() > 12 {
                state.loops.recent_call_keys.remove(0);
            }
            let cycle = cycle_period(&state.loops.recent_call_keys);
            if state.loops.same_call_streak >= 3 || cycle.is_some() {
                // Cycle-only detection (not also a plain streak).
                let cycle_only = cycle.filter(|_| state.loops.same_call_streak < 3);
                // A cycle is harmful if ANY member mutates.
                let mutating = if let Some(period) = cycle_only {
                    let tail = &state.loops.recent_call_keys
                        [state.loops.recent_call_keys.len().saturating_sub(period)..];
                    tail.iter().any(|k| key_is_mutating(k))
                } else {
                    is_mutating_call(&tc.function.name, &args)
                };
                log.loop_detected(
                    &tc.function.name,
                    &args_summary,
                    state.loops.same_call_streak as usize,
                );

                // Read-only repetition: harmless per call, just wasted tokens.
                // First detection: polite nudge, let processing continue.
                // Re-detection: escalate — the nudge can't reach a
                // cache-numerics rut, so force a compaction next round.
                if !mutating {
                    state.loops.read_nudges += 1;
                    let escalate = state.loops.read_nudges >= 2;
                    let text = if escalate {
                        state.loops.read_nudges = 0;
                        state.force_compact_next_round = true;
                        REPEATED_READ_ESCALATION
                    } else {
                        REPEATED_READ_NUDGE
                    };
                    let result_msg = Message::tool_result(&tc.id, text);
                    messages.push(result_msg.clone());
                    conversation_history.push(result_msg);
                    ui.status(&format!(
                        "  ⓘ Repeated read: {}({}) — {}, continuing",
                        tc.function.name,
                        args_summary,
                        if escalate {
                            "nudge failed, forcing compaction next round"
                        } else {
                            "nudge sent"
                        }
                    ));
                    state.loops.last_call_key = None;
                    state.loops.same_call_streak = 0;
                    state.loops.recent_call_keys.clear();
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
                if state.loops.recoveries == 0 {
                    state.loops.recoveries += 1;
                    state.loops.last_call_key = None;
                    state.loops.same_call_streak = 0;
                    state.loops.recent_call_keys.clear();
                    ui.status(&format!(
                        "  ⚠ Loop detected: {}({}) {} — surfacing a hint, giving the model one more round",
                        tc.function.name,
                        args_summary,
                        if let Some(period) = cycle_only {
                            format!("cycling through the same {period} calls (period-{period} cycle)")
                        } else {
                            "repeated 3 times".to_string()
                        }
                    ));
                    break;
                }

                // Second mutating loop after the recovery hint. With a
                // behavioral done-gate configured this is NOT a dead end — it
                // is the same "stuck but the task isn't done" state as a
                // premature exit, so route it through the gate ladder instead
                // of dying with the whole recovery stack idle.
                if !read_only
                    && config.validation.command().is_some()
                    && state.gate.validation_blocks < config.validation.max_retries
                {
                    ui.status(&format!(
                        "  Loop detected again ({}({})) — routing through the done-gate instead of stopping",
                        tc.function.name, args_summary
                    ));
                    if let validation::CheckOutcome::Fail(output) =
                        validation::run_behavioral_check(config).await
                    {
                        state.gate.validation_blocks += 1;
                        // Fresh recovery budget for the rounds the gate grants.
                        state.loops.recoveries = 0;
                        state.loops.last_call_key = None;
                        state.loops.same_call_streak = 0;
                        state.loops.recent_call_keys.clear();

                        let fkey = debugger::failure_key(&output);
                        let may_fire = if config.tools.debugger_multifire {
                            state.debugger.fires < debugger::MAX_DEBUGGER_FIRES
                                && state.debugger.last_failure.as_deref() != Some(fkey.as_str())
                        } else {
                            state.debugger.fires == 0
                        };
                        if (config.tools.reactive_debugger || config.tools.debugger_judge)
                            && may_fire
                            && state.gate.validation_blocks >= debugger::DEBUGGER_TRIGGER_BLOCKS
                        {
                            state.debugger.fires += 1;
                            state.debugger.last_failure = Some(fkey);
                            ui.status(
                                "Looping + failing gate — spinning up a fresh-context debugger sub-agent…",
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
                                debugger::DebuggerVerdict::Scrap if !state.gate.restart_fired => {
                                    state.gate.restart_fired = true;
                                    state.plan_ever_set = false;
                                    *messages = turn::restart::scrap_restart(
                                        &mut ui,
                                        config,
                                        goal,
                                        mcp_summary,
                                        snapshots,
                                        false,
                                        true,
                                    );
                                    conversation_history.clear();
                                    state.gate.validation_blocks = 0;
                                    state.gate.plan_step_failures.reset();
                                    continue 'round;
                                }
                                debugger::DebuggerVerdict::Scrap => {
                                    Message::user(debugger::SCRAP_ALREADY_RESET_MSG)
                                }
                                debugger::DebuggerVerdict::Rewind(candidate) => {
                                    turn::restart::rewind_message(
                                        &mut ui,
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
                                        validation::gate_failure_note(config, &output);
                                    Message::user(&debugger::build_gate_report_message(
                                        &body,
                                        &output_note,
                                    ))
                                }
                            };
                            messages.push(msg.clone());
                            conversation_history.push(msg);
                            continue 'round;
                        }

                        let msg = Message::user(&validation::build_loop_abort_message(&output));
                        messages.push(msg.clone());
                        conversation_history.push(msg);
                        continue 'round;
                    }
                    // Gate passed (or skipped): the loop was on something the
                    // check doesn't care about — fall through to the stop.
                }
                ui.error(&format!(
                    "  ✗ Loop detected again ({}({})) after the recovery hint — stopping this turn",
                    tc.function.name, args_summary
                ));
                had_error = true;
                break;
            }

            log.tool_call_detail(&tc.function.name, &args);
            ui.tool_call_started(&tc.function.name, &args_summary);

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
                    ui.app.push_output(
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
                        ui.app.push_output(
                            &format!("  ✗ {}: {e}", tc.function.name),
                            LineStyle::ToolErr,
                        );
                        continue;
                    }
                    Ok(Some(prompt)) => {
                        // Needs user approval — show prompt in TUI
                        ui.app.pending_permission = Some(prompt);
                        ui.app.input.clear();
                        ui.app.cursor = 0;
                        let _ = ui.terminal.draw(|frame| ui::draw(frame, ui.app));

                        // Wait for user input (y/n/a)
                        let response = wait_for_permission_input(ui.app, ui.rx, ui.terminal).await;
                        ui.app.pending_permission = None;

                        match response.as_str() {
                            "y" | "yes" => {
                                perms.approve(action, false);
                                ui.status("  · Permission granted, running tool...");
                            }
                            "a" | "always" => {
                                perms.approve(action, true);
                                ui.status("  · Permission granted and saved, running tool...");
                            }
                            _ => {
                                perm_denied = true;
                                ui.status("  · Permission denied.");
                            }
                        }

                        let _ = ui.terminal.draw(|frame| ui::draw(frame, ui.app));
                    }
                    Ok(None) => {} // No prompt needed
                }
            }

            if perm_denied {
                let result_msg =
                    Message::tool_result(&tc.id, &format!("{} denied by user", tc.function.name));
                messages.push(result_msg.clone());
                conversation_history.push(result_msg);
                ui.app.push_output(
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
                ui.app.push_output(
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
                let result_rx = tool_pool.submit(move || {
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
                ui.await_tool_job(result_rx, &tc.function.name, cancelled)
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
                    let result_rx = tool_pool.submit(move || match registry {
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
                    ui.await_tool_job(result_rx, "mcp_use", cancelled).await
                }
            } else if tc.function.name == "spawn_agents" {
                let tasks = crate::cli::commands::agent::subagent::parse_tasks(&args);
                if tasks.is_empty() {
                    crate::tools::ToolResult::err(
                        "spawn_agents: 'agents' must be a non-empty array of {label, prompt}"
                            .into(),
                    )
                } else {
                    let outputs = ui
                        .drive_subagents(
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
                        )
                        .await;
                    let combined = crate::cli::commands::agent::subagent::format_outputs(outputs);
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
                let result_rx = tool_pool.submit(move || {
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
                ui.await_tool_job(result_rx, "edit_file", cancelled).await
            } else if tc.function.name == "plan" {
                let args = args.clone();
                let config_for_job = config.clone();
                let result_rx = tool_pool.submit(move || {
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
                let r = ui.await_tool_job(result_rx, "plan", cancelled).await;
                // The plan tool just mutated plan.md mid-round — refresh the
                // panel and redraw now so a checked/added/refined step appears
                // immediately instead of lagging to the next round's refresh.
                ui.refresh_plan(config, round);
                let _ = ui.terminal.draw(|frame| ui::draw(frame, ui.app));
                r
            } else if tc.function.name == "refactor" {
                let args = args.clone();
                let config = config.clone();
                let router = router.clone();
                let lsp = lsp.clone();
                let log_for_job = log.clone();
                let revisions_for_job = fast_revisions.clone();
                let cancelled_for_job = cancelled.clone();
                let result_rx = tool_pool.submit(move || {
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
                ui.await_tool_job(result_rx, "refactor", cancelled).await
            } else if (tc.function.name == "shell" && args["action"].as_str() == Some("run"))
                || (tc.function.name == "file" && file_action == "shell")
            {
                if args["background"].as_bool() == Some(true) {
                    // Explicit background start (cheap, non-blocking) —
                    // registered in the session job registry, managed via
                    // the jobs tool in this or any later turn.
                    tools::jobs::start_background(&args, config, job_registry.as_ref())
                } else {
                    ui.await_shell_job(
                        tool_pool.submit_shell(args.clone(), config.clone(), cancelled.clone()),
                        cancelled,
                        None,
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
                let result_rx = tool_pool.submit(move || {
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
                ui.await_tool_job(result_rx, "jobs", cancelled).await
            } else {
                let tool_name = tc.function.name.clone();
                let args = args.clone();
                let config = config.clone();
                let perms = perms.clone();
                let lsp = lsp.clone();
                let result_rx = tool_pool.submit(move || {
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
                ui.await_tool_job(result_rx, &tc.function.name, cancelled)
                    .await
            };

            match turn::dispatch::finish_call(
                ctx,
                opts,
                &mut state,
                &mut skill_state,
                &mut ui,
                messages,
                conversation_history,
                round,
                tc,
                &args,
                &args_summary,
                &call_key,
                &mut result,
                &mut all_prunable_failures,
                &mut prunable_errors,
            )
            .await
            {
                turn::CallFlow::NextCall => {}
                turn::CallFlow::RestartRound => continue 'round,
            }

            // Re-render after tool result
            let _ = ui.terminal.draw(|frame| ui::draw(frame, ui.app));
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
        if strict && round >= 12 && !state.nudged_no_plan && !tools::plan::plan_exists(config) {
            let unlock_tools = "refactor, replace_range, insert_at, write_file";
            messages.push(Message::user(&format!(
                "[Reminder: you've explored for several rounds without a plan. \
                 Call plan(action='set') with your step-by-step approach now — \
                 the edit tools ({unlock_tools}) are hidden until you do, and \
                 you'll need them to make changes.]"
            )));
            state.nudged_no_plan = true;
        }

        // Stall detection: too many tool calls without any edits. Content is
        // plan-state aware — without a plan the edit tools are hidden, so
        // re-fire the plan nudge instead of pointing at hidden tools.
        if state.calls_since_last_edit >= 20 && state.calls_since_last_edit.is_multiple_of(20) {
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
