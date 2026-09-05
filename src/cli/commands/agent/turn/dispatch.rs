//! Post-dispatch tail of one tool call in the round's batch: result
//! post-processing (round-number footer, logging, loop-key bookkeeping,
//! plan-checkpoint nudges), the `plan_gate_debugger` fresh-context
//! escalation, and spiral-reset. Shared verbatim by both loops except where
//! [`TurnOptions`] names a delta.

use std::sync::Arc;

use crate::cli::commands::agent::debugger;
use crate::cli::commands::agent::hints::{
    PLAN_CHECKPOINT_AFTER_EDITS, PLAN_CHECKPOINT_WARNING, PLAN_PROGRESS_NUDGE, is_file_write,
    is_prunable_refactor_failure,
};
use crate::cli::commands::agent::job_banners::note_job_banners;
use crate::cli::commands::agent::skill_cursor;
use crate::cli::commands::agent::skill_step::{
    descend_into_skill, report_cursor_gaps, resolve_handoff,
};
use crate::cli::commands::agent::spiral;
use crate::cli::commands::agent::stuck_check;
use crate::cli::commands::agent::turn_state::{SkillTurnState, TurnState};
use crate::cli::commands::agent::ui::AgentUi;
use crate::cli::commands::agent::validation;
use crate::config::EditMode;
use crate::llm::{Message, ModelRouter};
use crate::logging::SessionLog;
use crate::tools;
use crate::tools::permissions::Action;

use super::restart;
use super::{CallFlow, TurnCtx, TurnOptions};

/// Everything the round does after one tool call in the batch finishes
/// executing — shared verbatim by both loops except where [`TurnOptions`]
/// names a delta.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn finish_call(
    ctx: TurnCtx<'_>,
    opts: TurnOptions,
    state: &mut TurnState,
    skill_state: &mut SkillTurnState,
    ui: &mut impl AgentUi,
    messages: &mut Vec<Message>,
    conversation_history: &mut Vec<Message>,
    round: usize,
    tc: &crate::llm::ToolCall,
    args: &serde_json::Value,
    args_summary: &str,
    call_key: &str,
    result: &mut crate::tools::ToolResult,
    all_prunable_failures: &mut bool,
    prunable_errors: &mut Vec<String>,
) -> CallFlow {
    let strict = ctx.config.tools.ceremony == crate::config::CeremonyMode::Strict;
    // Bound as `max_rounds` so the round-footer format string below stays
    // byte-identical to the original — it's a captured-identifier
    // interpolation and this string is grepped by the bench tooling.
    let max_rounds = ctx.max_rounds;

    if !result.success
        && let Some(hint) = tools::plan::failure_hint(ctx.config)
    {
        result.content.push('\n');
        result.content.push_str(&hint);
    }

    // Append round number to every tool result
    result
        .content
        .push_str(&format!("\n[round {round}/{max_rounds}]"));

    let first_line = result.content.lines().next().unwrap_or("(empty)");
    ctx.log
        .tool_call(&tc.function.name, args_summary, result.success, first_line);
    ctx.log
        .tool_result_detail(&tc.function.name, result.success, &result.content);
    ui.tool_result(&tc.function.name, result.success, first_line);
    ui.store_tool_result(&tc.function.name, &result.content);

    // Headless only (`TurnOptions::failure_tracking`): both blocks feed the
    // loop-recovery ladder's `recover_output` chain.
    if opts.failure_tracking {
        // Remember the last failing tool call keyed by its loop key, so a
        // loop on a *failing* command can hand the real error to the
        // debugger (see the loop-recovery ladder). Shell exit≠0 sets
        // success=false, so this catches the `pack package create` case.
        // Keyed by the SAME tagged `call_key` the ladder compares against
        // — an untagged key here never matches during skill runs.
        if !result.success {
            state.last_tool_failure = Some((
                call_key.to_string(),
                crate::truncate_chars(result.content.trim(), 2000),
            ));
        }
        // Background-job bookkeeping is per BANNER, not per result: a
        // FAILED deploy surfaces later in a wait/status result (possibly
        // aggregated with other jobs' banners), so the wrapper's ok/err
        // can't attribute verdicts to commands.
        note_job_banners(&result.content, &mut state.failed_job_commands);
    }

    if result.success && tc.function.name == "plan" {
        state.successful_edits_since_plan_update = 0;
    }

    // A successful file write means code changed — reset trackers.
    if result.success && is_file_write(tc.function.name.as_str()) {
        state.loops.last_call_key = None;
        state.loops.same_call_streak = 0;
        state.calls_since_last_edit = 0;
        if strict && ctx.config.tools.plan {
            if tools::plan::plan_exists(ctx.config) {
                result.content.push('\n');
                result.content.push_str(PLAN_PROGRESS_NUDGE);
            }
            state.successful_edits_since_plan_update += 1;
            if state.successful_edits_since_plan_update == PLAN_CHECKPOINT_AFTER_EDITS {
                result.content.push('\n');
                result.content.push_str(PLAN_CHECKPOINT_WARNING);
            }
        }
    } else {
        state.calls_since_last_edit += 1;
    }

    if !is_prunable_refactor_failure(&result.content, result.success) {
        *all_prunable_failures = false;
    } else {
        prunable_errors.push(result.content.clone());
    }

    // Headless only (`TurnOptions::stuck_tracking`): feeds `tools.stuck_check`.
    if opts.stuck_tracking {
        state
            .stuck_tracker
            .on_tool(&tc.function.name, args, result.success, &result.content);
    }
    // Headless only (`TurnOptions::skill_steps`): the freshness clock for
    // `skill_state.last_judge_block` read by the skill judge-veto check.
    if opts.skill_steps
        && result.success
        && stuck_check::is_mutating_edit(
            &tc.function.name,
            args.get("action").and_then(|a| a.as_str()).unwrap_or(""),
        )
    {
        skill_state.edits_total += 1;
    }

    let result_msg = Message::tool_result(&tc.id, &result.content);
    messages.push(result_msg.clone());
    conversation_history.push(result_msg);

    // `tools.plan_gate_debugger`: the plan tool's OWN compile gate
    // repeatedly blocking the SAME step is a distinct stall signature
    // from the behavioral done-gate (`state.gate.validation_blocks` above) — the
    // primary agent is re-litigating one step in its own accumulated
    // context rather than making forward progress. See the field doc
    // in config/mod.rs for the forensic evidence motivating this.
    if tc.function.name == "plan" && args.get("action").and_then(|a| a.as_str()) == Some("check") {
        if result.success {
            state.gate.plan_step_failures.reset();
        } else if let Some(step) = args.get("step").and_then(|s| s.as_u64()) {
            state.gate.plan_step_failures.note(step);

            let fkey = debugger::failure_key(&result.content);
            let may_fire = if ctx.config.tools.debugger_multifire {
                state.debugger.fires < debugger::MAX_DEBUGGER_FIRES
                    && state.debugger.last_failure.as_deref() != Some(fkey.as_str())
            } else {
                state.debugger.fires == 0
            };
            if ctx.config.tools.plan_gate_debugger
                && may_fire
                && state.gate.plan_step_failures.streak() as usize
                    >= debugger::DEBUGGER_TRIGGER_BLOCKS
            {
                state.debugger.fires += 1;
                state.debugger.last_failure = Some(fkey);
                ui.status(
                    "Plan-check gate failing repeatedly on the same step — spinning up a fresh-context debugger sub-agent…",
                );
                let verdict = debugger::run_debugger(
                    &result.content,
                    ctx.task,
                    ctx.config,
                    ctx.llm_worker,
                    ctx.tool_pool,
                    ctx.tool_defs,
                    ctx.perms,
                    ctx.mcp_registry,
                    ctx.lsp,
                    ctx.fast_revisions,
                    ctx.fast_baseline_errors,
                    ctx.cancelled,
                )
                .await;

                let extra_msg = match verdict {
                    debugger::DebuggerVerdict::Scrap if !state.gate.restart_fired => {
                        state.gate.restart_fired = true;
                        state.plan_ever_set = false;
                        *messages = restart::scrap_restart(
                            ui,
                            ctx.config,
                            ctx.task,
                            ctx.mcp_summary,
                            ctx.snapshots,
                            ctx.plan_only,
                            true,
                        );
                        conversation_history.clear();
                        state.gate.validation_blocks = 0;
                        state.gate.plan_step_failures.reset();
                        return CallFlow::RestartRound;
                    }
                    debugger::DebuggerVerdict::Scrap => {
                        Message::user(debugger::SCRAP_ALREADY_RESET_MSG)
                    }
                    debugger::DebuggerVerdict::Rewind(candidate) => {
                        restart::rewind_message(
                            ui,
                            &candidate,
                            ctx.config,
                            ctx.perms,
                            ctx.lsp,
                            ctx.fast_revisions,
                            ctx.fast_baseline_errors,
                            &result.content,
                        )
                        .await
                    }
                    debugger::DebuggerVerdict::Report(body) => {
                        let output_note =
                            validation::gate_failure_note(ctx.config, &result.content);
                        Message::user(&debugger::build_plan_step_report_message(
                            &body,
                            &output_note,
                        ))
                    }
                };
                messages.push(extra_msg.clone());
                conversation_history.push(extra_msg);
            }
        }
    }

    // Spiral-reset: a revert-loop (same file reverted repeatedly) means
    // the agent is cycling on the same failing edits. A bare revert
    // won't break it — its context keeps dragging it back. Inject a
    // cognitive reset (names what failed + forces a replan + concrete
    // redirection). API-probe-validated framing; see agent::spiral.
    if ctx.config.tools.spiral_reset
        && result.success
        && tc.function.name == "revert"
        && ctx.config.tools.edit_mode == EditMode::Fast
        && state.spiral.resets < spiral::MAX_RESETS_PER_TURN
        && let Some(path) = args.get("path").and_then(|p| p.as_str())
    {
        let count = state
            .spiral
            .revert_counts
            .entry(path.to_string())
            .or_insert(0);
        *count += 1;
        if *count >= spiral::SPIRAL_REVERT_THRESHOLD {
            let n = *count;
            *count = 0;
            state.spiral.resets += 1;
            let tried = ctx
                .fast_revisions
                .as_deref()
                .map(|r| spiral::tried_edit_labels(r, path, 4))
                .unwrap_or_default();
            let reset = Message::user(&spiral::build_reset_message(path, n, &tried));
            messages.push(reset.clone());
            conversation_history.push(reset);
            ui.status("Spiral detected (revert-loop) — reset + replan injected.");
            ctx.log.tool_debug(
                "agent",
                &format!("spiral-reset fired for {path} after {n} reverts"),
            );
        }
    }

    CallFlow::NextCall
}

/// Dispatch one tool call to its executor and return the raw
/// [`crate::tools::ToolResult`] — shared verbatim by both loops except
/// where [`TurnOptions`] names a delta. `router` and `log` are separate
/// from [`TurnCtx`] because its `router`/`log` fields are bare references
/// (unlike every other Arc-wrapped service field there): the `edit_file`
/// and `refactor` arms need an owned, `'static`-safe `Arc` to move into a
/// `tool_pool.submit(move || ..)` closure.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn execute(
    ctx: TurnCtx<'_>,
    opts: TurnOptions,
    skill_state: &mut SkillTurnState,
    ui: &mut impl AgentUi,
    round: usize,
    tc: &crate::llm::ToolCall,
    args: &serde_json::Value,
    file_action: &str,
    router: &Arc<ModelRouter>,
    log: &Arc<SessionLog>,
) -> crate::tools::ToolResult {
    if opts.snapshot_revert_arm && tc.function.name == "file" && file_action == "revert" {
        let snapshots = ctx.snapshots.clone();
        let args = args.clone();
        match ctx
            .tool_pool
            .submit(move || {
                let to_round = args["to_round"].as_u64().unwrap_or(0) as usize;
                let path = args["path"].as_str().unwrap_or("").to_string();
                match snapshots {
                    Some(snap) => {
                        let guard = snap.lock();
                        let res = if !path.is_empty() {
                            guard.revert_file(&path, to_round)
                        } else {
                            guard.revert_to_round(to_round)
                        };
                        res.map(crate::tools::ToolResult::ok)
                            .map_err(|e| format!("Revert failed: {e}"))
                    }
                    None => Ok(crate::tools::ToolResult::err(
                        "Snapshot system not available (git not found?)".into(),
                    )),
                }
            })
            .await
        {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => crate::tools::ToolResult::err(e),
            Err(_) => crate::tools::ToolResult::err("Tool worker dropped revert job".into()),
        }
    } else if tc.function.name == "plan" {
        let plan_args = args.clone();
        let config = ctx.config.clone();
        let result_rx = ctx.tool_pool.submit(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| e.to_string())?;
            runtime
                .block_on(async move { tools::plan::execute(&plan_args, &config, round).await })
                .map_err(|e| format!("plan error: {e}"))
        });
        if opts.plan_job_direct_await {
            match result_rx.await {
                Ok(Ok(r)) => r,
                Ok(Err(e)) => crate::tools::ToolResult::err(e),
                Err(_) => crate::tools::ToolResult::err("Tool worker dropped plan job".into()),
            }
        } else {
            let r = ui.await_tool_job(result_rx, "plan", ctx.cancelled).await;
            // The plan tool just mutated plan.md mid-round — refresh the
            // panel and redraw now so a checked/added/refined step appears
            // immediately instead of lagging to the next round's refresh.
            ui.after_plan_tool(ctx.config, round);
            r
        }
    } else if tc.function.name == "edit_file" {
        let args = args.clone();
        let config = ctx.config.clone();
        let perms = ctx.perms.clone();
        let router = router.clone();
        let lsp = ctx.lsp.clone();
        let cancelled_for_job = ctx.cancelled.clone();
        let log_for_job = log.clone();
        ui.await_tool_job(
            ctx.tool_pool.submit(move || {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|e| e.to_string())?;
                runtime
                    .block_on(async move {
                        tools::execute_edit_file_tool(
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
            }),
            "edit_file",
            ctx.cancelled,
        )
        .await
    } else if tc.function.name == "refactor"
        || (opts.flat_refactor_aliases
            && matches!(
                tc.function.name.as_str(),
                "add_function_param" | "drop_function_param" | "rename_symbol"
            ))
    {
        // Flat refactor tools normalize into the grouped
        // `refactor` args shape; same executor.
        let args = tools::definitions::flat_to_refactor_args(&tc.function.name, args)
            .unwrap_or_else(|| args.clone());
        let config = ctx.config.clone();
        let router = router.clone();
        let lsp = ctx.lsp.clone();
        let log_for_job = log.clone();
        let revisions_for_job = ctx.fast_revisions.clone();
        let cancelled_for_job = ctx.cancelled.clone();
        ui.await_tool_job(
            ctx.tool_pool.submit(move || {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|e| e.to_string())?;
                runtime
                    .block_on(async move {
                        tools::execute_refactor_tool(
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
            }),
            "refactor",
            ctx.cancelled,
        )
        .await
    } else if (tc.function.name == "shell" && args["action"].as_str() == Some("run"))
        || (tc.function.name == "file" && file_action == "shell")
    {
        if args["background"].as_bool() == Some(true) {
            // Explicit background start: the sanctioned form of the
            // model's "cmd & echo $! > .pid" instinct — registered,
            // output-captured, managed via the jobs tool.
            tools::jobs::start_background(args, ctx.config, ctx.job_registry.as_ref())
        } else {
            ui.await_shell_job(
                ctx.tool_pool
                    .submit_shell(args.clone(), ctx.config.clone(), ctx.cancelled.clone()),
                ctx.cancelled,
                opts.register_shell_jobs
                    .then_some(ctx.job_registry.as_ref()),
            )
            .await
        }
    } else if tc.function.name == "shell" {
        if opts.jobs_on_pool {
            // Runs on the pool (own runtime) so jobs(wait) keeps the TUI
            // responsive via await_tool_job_ui, like other pooled tools.
            let args_for_job = args.clone();
            let config_for_job = ctx.config.clone();
            let perms_for_job = ctx.perms.clone();
            let registry_for_job = ctx.job_registry.clone();
            let cancelled_for_job = ctx.cancelled.clone();
            let result_rx = ctx.tool_pool.submit(move || {
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
            ui.await_tool_job(result_rx, "jobs", ctx.cancelled).await
        } else {
            tools::jobs::execute(
                args,
                ctx.config,
                ctx.perms.as_ref(),
                ctx.job_registry.as_ref(),
                Some(ctx.cancelled.as_ref()),
            )
            .await
        }
    } else if matches!(
        tc.function.name.as_str(),
        "replace_range" | "insert_at" | "revert" | "show_rev" | "check"
    ) && ctx.config.tools.edit_mode == EditMode::Fast
    {
        let tool_name = tc.function.name.clone();
        let args = args.clone();
        let config = ctx.config.clone();
        let perms = ctx.perms.clone();
        let lsp = ctx.lsp.clone();
        let revisions = ctx.fast_revisions.clone();
        let baseline = ctx.fast_baseline_errors;
        ui.await_tool_job(
            ctx.tool_pool.submit(move || {
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
            }),
            &tc.function.name,
            ctx.cancelled,
        )
        .await
    } else if tc.function.name == "mcp_use" {
        let server = args["server"].as_str().unwrap_or("").to_string();
        let tool = args["tool"].as_str().unwrap_or("").to_string();
        let tool_args = args.get("arguments").cloned().unwrap_or_default();
        if server.is_empty() || tool.is_empty() {
            crate::tools::ToolResult::err(
                "mcp_use requires top-level 'server' and 'tool' string fields. \
                 Example: {\"server\": \"my-server\", \"tool\": \"my-tool\", \"arguments\": {}}"
                    .into(),
            )
        } else if opts.inline_mcp_permission_check {
            match ctx
                .perms
                .check(&Action::McpUse(server.clone(), tool.clone()))
            {
                Err(e) => crate::tools::ToolResult::err(e),
                Ok(()) => {
                    let registry = ctx.mcp_registry.clone();
                    match ctx
                        .tool_pool
                        .submit(move || match registry {
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
                        })
                        .await
                    {
                        Ok(Ok(r)) => r,
                        Ok(Err(e)) => crate::tools::ToolResult::err(e),
                        Err(_) => {
                            crate::tools::ToolResult::err("Tool worker dropped mcp job".into())
                        }
                    }
                }
            }
        } else {
            let registry = ctx.mcp_registry.clone();
            let result_rx = ctx.tool_pool.submit(move || match registry {
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
            ui.await_tool_job(result_rx, "mcp_use", ctx.cancelled).await
        }
    } else if tc.function.name == "spawn_agents" {
        let tasks = crate::cli::commands::agent::subagent::parse_tasks(args);
        if tasks.is_empty() {
            crate::tools::ToolResult::err(
                "spawn_agents: 'agents' must be a non-empty array of {label, prompt}".into(),
            )
        } else {
            ui.spawning_subagents(tasks.len());
            let outputs = ui
                .drive_subagents(
                    tasks,
                    ctx.config,
                    ctx.llm_worker,
                    ctx.tool_pool,
                    ctx.tool_defs,
                    ctx.perms,
                    ctx.mcp_registry,
                    ctx.lsp,
                    ctx.fast_revisions,
                    ctx.fast_baseline_errors,
                    ctx.cancelled,
                )
                .await;
            let combined = crate::cli::commands::agent::subagent::format_outputs(outputs);
            crate::tools::ToolResult::ok(combined)
        }
    } else if opts.skill_steps && tc.function.name == "skill" {
        // Harness-owned step cursor: the model signals it finished the
        // current [SKILL STEP]; advance and announce the next one (its
        // instructions arrive via [SKILL STEP] on the next round,
        // distilled in round maintenance).
        let action = args["action"].as_str().unwrap_or("done");
        if action == "done" {
            let mut cursor = skill_cursor::load(ctx.config);
            match cursor
                .current()
                .map(|(sk, st)| (sk.to_string(), st.name.clone()))
            {
                Some((skill, finished)) => {
                    // Gate on the step's completion check with a
                    // one-retry override: the FIRST skill(done) runs the
                    // check and, on failure, is refused with the check
                    // output; a SECOND consecutive skill(done) advances
                    // anyway (the model overrules a possibly-wrong check —
                    // the probe measured ~6% of checks over-specify and
                    // false-fail, so the model needs an escape hatch).
                    let check = cursor.current_check().map(str::to_string);
                    let unchecked = check.is_none();
                    let attempt = cursor.note_done_attempt();
                    let blocked = match check.filter(|_| attempt < 2) {
                        Some(cmd) => match validation::run_check_command(ctx.config, &cmd).await {
                            validation::CheckOutcome::Fail(out) => Some(out),
                            _ => None,
                        },
                        None => None,
                    };
                    if let Some(out) = blocked {
                        // Persist the incremented attempt; do NOT advance.
                        skill_cursor::save(ctx.config, &cursor);
                        crate::tools::ToolResult::err(format!(
                            "Step '{finished}' does not meet its DONE WHEN yet — the \
                             completion check failed:\n{out}\nFix it and call \
                             skill(action='done') again. If you are certain the step is \
                             actually complete and the check is wrong, call it once more \
                             to override."
                        ))
                    } else {
                        // LOG-ONLY judge-veto observation (2026-09-01
                        // e2e: 4 unchecked steps advanced right over a
                        // standing not-done verdict). An UNCHECKED
                        // step leans on the judge alone, so when one
                        // gets a done with a standing verdict and no
                        // mutating edit since (the verdict can't be
                        // stale), record what an enforced veto would
                        // have refused. Enforcement waits on measured
                        // live judge quality — verdicts have been
                        // observed zero times in the field.
                        if unchecked
                            && let Some((key, reason, at_edits)) = &skill_state.last_judge_block
                            && *key == format!("{skill}::{finished}")
                            && *at_edits == skill_state.edits_total
                        {
                            ui.status(&format!(
                                "[skills] judge-veto (log-only) on '{finished}': {reason}"
                            ));
                        }
                        skill_state.last_judge_block = None;
                        // prepare_step guarantees the parked step is
                        // distilled, so a `done` here is always a
                        // verdict on something the model was actually
                        // shown. If that ever stops holding the step
                        // was invisible this round and the verdict is
                        // meaningless — say so rather than let it pass
                        // silently, as it did before the loop in
                        // prepare_step closed that window.
                        if cursor.cached().is_none() {
                            ui.status(&format!(
                                "[skills] warning: done on '{finished}' while undistilled \
                                 — the step was never shown"
                            ));
                        }
                        // A skill's LAST step often exists only to hand
                        // off (build → integrate). Resolve that BEFORE
                        // mark_done pops the frame: once popped there is
                        // no cursor left to descend from, and the run
                        // ends the build → integrate → validate lifecycle
                        // a phase early while reporting success. Live
                        // e2e: `done` on the build skill's
                        // EnterIntegrationPhase step silently dropped the
                        // Package CR, networking, Postgres, IDP,
                        // monitoring and validation steps.
                        let installed: Vec<String> =
                            crate::skills::discover(&ctx.config.project_root)
                                .into_iter()
                                .map(|e| e.name)
                                .collect();
                        let handed_off = match resolve_handoff(
                            &mut cursor,
                            &installed,
                            ctx.llm_worker,
                            ctx.cancelled,
                        )
                        .await
                        {
                            Some(next) => {
                                descend_into_skill(
                                    &mut cursor,
                                    &next,
                                    ctx.config,
                                    ctx.llm_worker,
                                    ctx.cancelled,
                                )
                                .await
                            }
                            None => false,
                        };
                        // descend() already consumed the invoking step —
                        // marking done as well would skip the sub-skill's
                        // first step.
                        if !handed_off {
                            cursor.mark_done();
                        }
                        report_cursor_gaps(&cursor);
                        skill_cursor::save(ctx.config, &cursor);
                        let msg = match cursor.current() {
                            Some((_, next)) => format!(
                                "Step '{finished}' marked done. Next step: '{}'. Its full \
                                 instructions will appear under [SKILL STEP] — follow them \
                                 exactly.",
                                next.name
                            ),
                            // The frame popped. That only means every
                            // step is complete when none were abandoned
                            // on the way — say which are outstanding
                            // rather than inviting a finish over them.
                            None if !cursor.dropped_unfinished().is_empty() => format!(
                                "Step '{finished}' marked done, but the {skill} skill ends \
                                 with unfinished steps: {}. Those were never completed — \
                                 go back and finish them before you stop.",
                                cursor.dropped_unfinished().join(", ")
                            ),
                            None => format!(
                                "Step '{finished}' marked done. All {skill} skill steps are \
                                 complete — finish the task."
                            ),
                        };
                        crate::tools::ToolResult::ok(msg)
                    }
                }
                None => crate::tools::ToolResult::err("No active skill step to complete.".into()),
            }
        } else {
            crate::tools::ToolResult::err(format!(
                "Unknown skill action '{action}'. Use action='done'."
            ))
        }
    } else {
        let tool_name = tc.function.name.clone();
        let args = args.clone();
        let config = ctx.config.clone();
        let perms = ctx.perms.clone();
        let lsp = ctx.lsp.clone();
        ui.await_tool_job(
            ctx.tool_pool.submit(move || {
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
            }),
            &tc.function.name,
            ctx.cancelled,
        )
        .await
    }
}
