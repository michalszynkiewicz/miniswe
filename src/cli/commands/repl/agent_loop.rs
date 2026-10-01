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

    // Fed into `TurnCtx` but never read on this side — `opts.stuck_tracking`
    // is always false in the REPL, so the headless-only `tools.stuck_check`
    // fire in `postamble::finish_round` never evaluates it.
    let session_start = std::time::Instant::now();
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
        snapshot_revert_arm: false,
        flat_refactor_aliases: false,
        inline_mcp_permission_check: false,
        register_shell_jobs: false,
        jobs_on_pool: true,
        plan_job_direct_await: false,
        window_repeat_detector: false,
        jobs_poll_redirect: false,
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
        // Headless-only stuck tracker (`opts.stuck_tracking` is false here);
        // kept at the same position as the headless loop for Stage 4's driver.
        if opts.stuck_tracking {
            state
                .stuck_tracker
                .on_round(round, session_start.elapsed().as_secs_f64());
        }

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
            session_start,
        };
        // Skill step-cursor maintenance: no-op here (`opts.skill_steps` is
        // false in the REPL — it has no cursor), kept at the same position
        // as the headless loop for Stage 4's driver.
        turn::skill_round::prepare(
            ctx,
            opts,
            &mut skill_state,
            &mut ui,
            messages,
            conversation_history,
        )
        .await;

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
            let admitted = match turn::call_gate::admit(
                ctx,
                opts,
                &mut state,
                &mut ui,
                messages,
                conversation_history,
                tc,
                None,
            )
            .await
            {
                turn::AdmitFlow::Run(a) => a,
                turn::AdmitFlow::NextCall => continue,
                turn::AdmitFlow::StopCalls { error } => {
                    had_error |= error;
                    break;
                }
                turn::AdmitFlow::RestartRound => continue 'round,
            };

            // Execute tool (permissions already checked above for shell/web/mcp)
            let mut result = turn::dispatch::execute(
                ctx,
                opts,
                &mut skill_state,
                &mut ui,
                round,
                tc,
                &admitted.args,
                &admitted.file_action,
                router,
                &log,
            )
            .await;

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
                &admitted.args,
                &admitted.args_summary,
                &admitted.call_key,
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

        turn::postamble::finish_round(
            ctx,
            opts,
            &mut state,
            &mut ui,
            messages,
            conversation_history,
            round,
            strict,
            turn::Prunable {
                messages_pre,
                history_pre,
                all_failures: all_prunable_failures,
                errors: prunable_errors,
            },
        )
        .await;
    }

    log.session_end(round, had_error);
}
