//! Post-dispatch tail of one tool call in the round's batch: result
//! post-processing (round-number footer, logging, loop-key bookkeeping,
//! plan-checkpoint nudges), the `plan_gate_debugger` fresh-context
//! escalation, and spiral-reset. Shared verbatim by both loops except where
//! [`TurnOptions`] names a delta.

use crate::cli::commands::agent::debugger;
use crate::cli::commands::agent::hints::{
    PLAN_CHECKPOINT_AFTER_EDITS, PLAN_CHECKPOINT_WARNING, PLAN_PROGRESS_NUDGE, is_file_write,
    is_prunable_refactor_failure,
};
use crate::cli::commands::agent::job_banners::note_job_banners;
use crate::cli::commands::agent::spiral;
use crate::cli::commands::agent::stuck_check;
use crate::cli::commands::agent::turn_state::{SkillTurnState, TurnState};
use crate::cli::commands::agent::ui::AgentUi;
use crate::cli::commands::agent::validation;
use crate::config::EditMode;
use crate::llm::Message;
use crate::tools;

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
