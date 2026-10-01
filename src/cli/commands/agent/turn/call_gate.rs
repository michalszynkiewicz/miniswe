//! Pre-dispatch gate for one tool call in the round's batch: JSON-arg
//! parsing, loop-call-key bookkeeping, the full loop-detection/recovery
//! ladder (including the headless-only window-repeat and jobs-poll-redirect
//! checks, and the debugger SCRAP/Rewind/Report escalation), the REPL-only
//! explore-mode block and permission preflight, and the plan-only /
//! write-gate blocks. Shared verbatim by both loops except where
//! [`TurnOptions`] names a delta. Everything that clears every gate comes
//! back as [`Admitted`], ready for `dispatch::execute`.

use crate::cli::commands::agent::debugger;
use crate::cli::commands::agent::display::summarize_args;
use crate::cli::commands::agent::explore_gate::explore_block_reason;
use crate::cli::commands::agent::hints::{
    REPEATED_READ_ESCALATION, REPEATED_READ_NUDGE, cycle_loop_hint, is_file_write,
    loop_detected_hint, truncated_tool_call_hint,
};
use crate::cli::commands::agent::job_banners::failing_job_output;
use crate::cli::commands::agent::loop_detector::{
    cycle_period, is_mutating_call, key_is_file_edit, key_is_mutating, loop_call_key_tagged,
};
use crate::cli::commands::agent::permissions::permission_action;
use crate::cli::commands::agent::skill_cursor;
use crate::cli::commands::agent::turn_state::TurnState;
use crate::cli::commands::agent::ui::{AgentUi, PreflightPermission, UiEvent};
use crate::cli::commands::agent::validation;
use crate::llm::{Message, truncated_args_info, truncated_args_tool_result};
use crate::tools;
use crate::tools::permissions::Action;

use super::restart;
use super::{AdmitFlow, Admitted, TurnCtx, TurnOptions};

/// Everything the round does to one tool call BEFORE dispatch — shared
/// verbatim by both loops except where [`TurnOptions`] names a delta.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn admit(
    ctx: TurnCtx<'_>,
    opts: TurnOptions,
    state: &mut TurnState,
    ui: &mut impl AgentUi,
    messages: &mut Vec<Message>,
    conversation_history: &mut Vec<Message>,
    tc: &crate::llm::ToolCall,
    active_step_tag: Option<&str>,
) -> AdmitFlow {
    let strict = ctx.config.tools.ceremony == crate::config::CeremonyMode::Strict;
    // Bound as `message` so the recovery-ladder format strings below stay
    // byte-identical to the headless original — these literals are grepped
    // by the bench tooling.
    let message = ctx.task;

    let args: serde_json::Value = match serde_json::from_str(&tc.function.arguments) {
        Ok(v) => v,
        Err(e) => {
            // Unreachable after sanitize_truncated_tool_calls above, kept as
            // a belt-and-braces path. Never echo the raw arguments back:
            // that is the flood we just refused to persist.
            let result_msg =
                Message::tool_result(&tc.id, &format!("Invalid JSON in tool arguments: {e}"));
            messages.push(result_msg.clone());
            conversation_history.push(result_msg);
            ui.tool_result(&tc.function.name, false, "invalid JSON args");
            return AdmitFlow::NextCall;
        }
    };
    if let Some(info) = truncated_args_info(&args) {
        // A call stubbed by sanitize_truncated_tool_calls: the arguments
        // were cut off by the output limit, so there is nothing to execute.
        // Answer with guidance, not a run.
        let result_msg = Message::tool_result(
            &tc.id,
            &format!(
                "{}\n\n{}",
                truncated_args_tool_result(&tc.function.name, &info),
                truncated_tool_call_hint(ctx.config.tools.edit_mode)
            ),
        );
        messages.push(result_msg.clone());
        conversation_history.push(result_msg);
        ctx.log.tool_debug(
            "agent",
            &format!(
                "{} call skipped: arguments truncated after {} chars",
                tc.function.name, info.original_chars
            ),
        );
        ui.event(UiEvent::TruncatedArgs {
            name: tc.function.name.clone(),
            original_chars: info.original_chars,
        });
        return AdmitFlow::NextCall;
    }

    let args_summary = summarize_args(&tc.function.name, &args);

    // Detect tool call loops: identical calls repeated consecutively
    // (period-1), or the SAME two calls alternating (period-2 — the
    // edit↔revert oscillation that the streak counter is blind to because
    // every alternation resets it).
    let call_key = loop_call_key_tagged(&tc.function.name, &args, active_step_tag);
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

    if opts.window_repeat_detector {
        // Soft loop-breaker for the "wandering grind": a call that recurs
        // FREQUENTLY in the window even when INTERSPERSED (so it never
        // trips the 3-consecutive detector below) — e.g. the model
        // re-`ls -R`ing / re-`helm show`ing the chart between other calls.
        // Clear the window on fire so it must re-accumulate — bounds the
        // escalation below to at most one fire per ~N repeats.
        // 4 (not 5) in a 12-window: a period-3 cycle (A,B,C,A,B,C…) puts
        // each element at exactly 12/3=4, so 4 catches period-2 AND
        // period-3 wandering; 5 would miss period-3.
        const WINDOW_REPEAT_FREQ: usize = 4;
        if state
            .loops
            .recent_call_keys
            .iter()
            .filter(|k| **k == call_key)
            .count()
            >= WINDOW_REPEAT_FREQ
        {
            state.loops.recent_call_keys.clear();
            // Escalate on a RECURRING file edit: one recurrence can be a
            // legitimate retry, a second is a rut, so break the cache-hot
            // prefix for real. Reads/checks/tests are exempt — repeating
            // those between different edits is a normal rhythm.
            let escalate = key_is_file_edit(&call_key) && {
                state.loops.window_edit_fires += 1;
                state.loops.window_edit_fires >= 2
            };
            if escalate {
                state.force_compact_next_round = true;
                state.loops.window_edit_fires = 0;
            }
            ui.status(&format!(
                "[loop] '{args_summary}' recurred {WINDOW_REPEAT_FREQ}x in the window{}",
                if escalate {
                    " — forcing context compaction next round"
                } else {
                    ""
                }
            ));
        }
    }

    let cycle = cycle_period(&state.loops.recent_call_keys);
    if state.loops.same_call_streak >= 3 || cycle.is_some() {
        // Cycle-only detection (not also a plain streak). Captured before
        // any state resets below so messaging stays accurate.
        let cycle_only = cycle.filter(|_| state.loops.same_call_streak < 3);
        // A cycle is harmful if ANY member mutates (the classic case is
        // edit↔revert — both mutate; edit↔read still re-applies the same
        // broken edit).
        let mutating = if let Some(period) = cycle_only {
            let tail = &state.loops.recent_call_keys
                [state.loops.recent_call_keys.len().saturating_sub(period)..];
            tail.iter().any(|k| key_is_mutating(k))
        } else {
            is_mutating_call(&tc.function.name, &args)
        };
        ctx.log.loop_detected(
            &tc.function.name,
            &args_summary,
            state.loops.same_call_streak as usize,
        );

        // Polling a status command while a background job runs is the
        // unpaced form of monitoring — redirect to the paced one, naming
        // the polled command as the check probe. Fires BEFORE the mutating
        // classification: shell commands classify as mutating, which
        // routed the jobs e2e's status-poll loop into the turn-stopping
        // path (2026-07-14, stuck scenario died 11s into a monitoring
        // task). Capped so a genuine runaway still escalates normally.
        if opts.jobs_poll_redirect
            && ((tc.function.name == "shell" && args["action"].as_str() == Some("run"))
                || (tc.function.name == "file" && args["action"].as_str() == Some("shell")))
            && !ctx.job_registry.is_empty()
            && state.loops.jobs_poll_redirects < 2
        {
            state.loops.jobs_poll_redirects += 1;
            let polled = args["command"].as_str().unwrap_or("<status command>");
            let result_msg = Message::tool_result(
                &tc.id,
                &format!(
                    "You are polling `{polled}` in a loop while a background job runs. \
                     Use shell(action='wait', secs=60, check='{polled}') instead — it \
                     waits, THEN runs the probe, one paced cycle per call."
                ),
            );
            messages.push(result_msg.clone());
            conversation_history.push(result_msg);
            ui.status(&format!(
                "Job-poll loop: {}({}) — redirected to jobs(wait), continuing",
                tc.function.name, args_summary
            ));
            state.loops.last_call_key = None;
            state.loops.same_call_streak = 0;
            state.loops.recent_call_keys.clear();
            return AdmitFlow::NextCall;
        }

        // Read-only repetition: harmless per call, just wasted tokens.
        // First detection: polite nudge inline, let the batch continue.
        // Re-detection: escalate — the nudge can't reach a cache-numerics
        // rut, so force a compaction next round.
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
            ui.event(UiEvent::RepeatedRead {
                name: tc.function.name.clone(),
                args_summary: args_summary.clone(),
                escalate,
            });
            state.loops.last_call_key = None;
            state.loops.same_call_streak = 0;
            state.loops.recent_call_keys.clear();
            return AdmitFlow::NextCall;
        }

        let hint = if let Some(period) = cycle_only {
            cycle_loop_hint(period)
        } else {
            loop_detected_hint(ctx.config.tools.edit_mode).to_string()
        };
        let result_msg = Message::tool_result(&tc.id, &hint);
        messages.push(result_msg.clone());
        conversation_history.push(result_msg);

        // First mutating loop in this turn: surface the hint, reset the
        // streak, and let the model try a different approach. Subsequent
        // loops mean the recovery itself spiraled — abort for real.
        if state.loops.recoveries == 0 {
            state.loops.recoveries += 1;
            state.loops.last_call_key = None;
            state.loops.same_call_streak = 0;
            state.loops.recent_call_keys.clear();
            ui.event(UiEvent::LoopDetected {
                name: tc.function.name.clone(),
                args_summary: args_summary.clone(),
                cycle_period: cycle_only,
            });
            return AdmitFlow::StopCalls { error: false };
        }
        // Second mutating loop after the recovery hint. With a behavioral
        // done-gate configured this is NOT a dead end — it is the same
        // "stuck but the task isn't done" state as a premature exit, so
        // route it through the gate ladder (block → debugger/judge at 2
        // blocks) instead of dying with the whole recovery stack idle.
        // (Real case: a run died at 70s looping on a malformed
        // replace_range while gate + judge never ran.) Without a gate:
        // original behavior — stop the turn. An active skill step's
        // completion check counts as the gate here too.
        //
        // Budget checked BEFORE running the check command (the one
        // sanctioned reorder in this unification, matching the REPL's
        // existing order) — on an exhausted budget, skip the check-command
        // run entirely instead of paying for it and discarding the result.
        if !opts.read_only && state.gate.validation_blocks < ctx.config.validation.max_retries {
            let effective_check = if opts.skill_steps {
                skill_cursor::current_check_command(ctx.config)
            } else {
                None
            }
            .or_else(|| ctx.config.validation.command().map(str::to_string));
            // Failure text to route through the recovery ladder, if we
            // should recover at all, in priority order:
            //  - the LOOPING COMMAND ITSELF keeps FAILING (its real error) —
            //    the missing trigger: a read-only per-step check can PASS
            //    while `pack package create` returns a lint error, so the
            //    command's own failure never surfaced. A fresh-context
            //    debugger fixes exactly this (probe: 10/10 on the flavor
            //    bug);
            //  - else a check that FAILS → its output;
            //  - else a skill step IS active (Fix 2) → synthesize the
            //    stuck state so a non-checkable (or check-passing but
            //    still looping) step reaches the debugger instead of the
            //    hard stop below, which assumes no cursor.
            let recover_output: Option<String> = if let Some((k, out)) = &state.last_tool_failure
                && *k == call_key
            {
                Some(format!(
                    "The agent is stuck repeating a tool call that keeps FAILING: \
                     {}({}). Its latest error output:\n{out}\n\nDiagnose the root cause and \
                     give the single concrete fix (exact command or edit) that makes it \
                     succeed.",
                    tc.function.name, args_summary
                ))
            } else if let Some((cmd, out)) =
                failing_job_output(&tc.function.name, &args, &state.failed_job_commands)
            {
                // The looping command launches a BACKGROUND job (e.g. the
                // detached `pkg run dev` deploy) that keeps FAILING — its
                // failure lands in a status result, not the launch, so it
                // never tripped the foreground trigger above.
                Some(format!(
                    "The agent keeps re-running a command whose background job FAILS: \
                     `{cmd}`. Its latest error output:\n{out}\n\nDiagnose the root cause and \
                     give the single concrete fix (exact command or edit) that makes it \
                     succeed."
                ))
            } else {
                let check_fail = if let Some(cmd) = effective_check.as_deref() {
                    match validation::run_check_command(ctx.config, cmd).await {
                        validation::CheckOutcome::Fail(o) => Some(o),
                        _ => None,
                    }
                } else {
                    None
                };
                check_fail.or_else(|| {
                    // A passing (often read-only proxy) check does NOT
                    // mean the step is unstuck — the model is still
                    // looping. Fall through to the synthesized stuck
                    // state rather than the cursor-less hard stop.
                    // Skill-cursor state is headless-only; the REPL has
                    // no cursor to read and never had this fallback.
                    if !opts.skill_steps {
                        return None;
                    }
                    let cursor = skill_cursor::load(ctx.config);
                    cursor.current().map(|(_, s)| {
                        let def = cursor.cached().unwrap_or("");
                        format!(
                            "The agent is stuck in a repeating loop while executing the '{}' step \
                             of a skill and is making no progress. Repeated tool call: {}({}). \
                             The current step:\n{def}\n\nDiagnose why it is stuck and the single \
                             concrete action that would unblock it — or, if the current state is \
                             a dead end, whether to scrap and restart from a clean base.",
                            s.name, tc.function.name, args_summary
                        )
                    })
                })
            };
            if let Some(output) = recover_output {
                ui.event(UiEvent::LoopRecovering {
                    name: tc.function.name.clone(),
                    args_summary: args_summary.clone(),
                });
                state.gate.validation_blocks += 1;
                // Fresh recovery budget for the rounds the ladder grants.
                state.loops.recoveries = 0;
                state.loops.last_call_key = None;
                state.loops.same_call_streak = 0;
                state.loops.recent_call_keys.clear();

                let fkey = debugger::failure_key(&output);
                let may_fire = if ctx.config.tools.debugger_multifire {
                    state.debugger.fires < debugger::MAX_DEBUGGER_FIRES
                        && state.debugger.last_failure.as_deref() != Some(fkey.as_str())
                } else {
                    state.debugger.fires == 0
                };
                if (ctx.config.tools.reactive_debugger || ctx.config.tools.debugger_judge)
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
                        message,
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

                    let msg = match verdict {
                        debugger::DebuggerVerdict::Scrap if !state.gate.restart_fired => {
                            state.gate.restart_fired = true;
                            state.plan_ever_set = false;
                            *messages = restart::scrap_restart(
                                ui,
                                ctx.config,
                                message,
                                ctx.mcp_summary,
                                ctx.snapshots,
                                ctx.plan_only,
                                true,
                            );
                            conversation_history.clear();
                            state.gate.validation_blocks = 0;
                            state.gate.plan_step_failures.reset();
                            return AdmitFlow::RestartRound;
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
                                &output,
                            )
                            .await
                        }
                        debugger::DebuggerVerdict::Report(body) => {
                            let output_note = validation::gate_failure_note(ctx.config, &output);
                            Message::user(&debugger::build_gate_report_message(&body, &output_note))
                        }
                    };
                    messages.push(msg.clone());
                    conversation_history.push(msg);
                    return AdmitFlow::RestartRound;
                }

                let msg = Message::user(&validation::build_loop_abort_message(&output));
                messages.push(msg.clone());
                conversation_history.push(msg);
                return AdmitFlow::RestartRound;
            }
        }
        // No check failed and no skill cursor is active — the loop is on
        // something the recovery ladder can't act on; stop the turn.
        ui.event(UiEvent::LoopStopping {
            name: tc.function.name.clone(),
            args_summary: args_summary.clone(),
        });
        return AdmitFlow::StopCalls { error: true };
    }

    ctx.log.tool_call_detail(&tc.function.name, &args);
    ui.tool_call_started(&tc.function.name, &args_summary);

    // Owned (not borrowed) so it outlives the `args` move into `Admitted`
    // below.
    let file_action = args["action"].as_str().unwrap_or("").to_string();

    // Read-only investigation (explore) mode: hard-block any mutating tool
    // call at runtime, BEFORE any permission prompt. The def filter and the
    // prompt are advisory — shell can still mutate and the model can emit
    // tools that aren't in the list. Read-only shell is allowed. Never
    // fires headless, which has no explore mode (`opts.read_only` is always
    // `false` there).
    if opts.read_only
        && let Some(reason) = explore_block_reason(&tc.function.name, &file_action, &args)
    {
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
        ui.event(UiEvent::ExploreBlocked {
            name: tc.function.name.clone(),
        });
        return AdmitFlow::NextCall;
    }

    // Permission preflight for shell/MCP tool calls. The REPL shows a
    // blocking TUI modal here; headless's impl is a no-op passthrough
    // (permission prompts, if any, are handled lazily on stdin inside tool
    // execution instead).
    let perm_action = permission_action(&tc.function.name, &args);
    if let Some(ref action) = perm_action
        && matches!(action, Action::Shell(_) | Action::McpUse(_, _))
    {
        match ui.preflight_permission(ctx.perms, action).await {
            PreflightPermission::Allowed => {}
            PreflightPermission::Denied => {
                let result_msg =
                    Message::tool_result(&tc.id, &format!("{} denied by user", tc.function.name));
                messages.push(result_msg.clone());
                conversation_history.push(result_msg);
                ui.tool_result(&tc.function.name, false, "denied");
                return AdmitFlow::NextCall;
            }
            PreflightPermission::Blocked(e) => {
                let result_msg = Message::tool_result(&tc.id, &e);
                messages.push(result_msg.clone());
                conversation_history.push(result_msg);
                ui.tool_result(&tc.function.name, false, &e);
                return AdmitFlow::NextCall;
            }
        }
    }

    // Block write tools in plan-only mode (headless only — the REPL always
    // passes `ctx.plan_only = false`).
    if ctx.plan_only
        && ((tc.function.name == "file" && file_action == "shell")
            || matches!(
                tc.function.name.as_str(),
                "edit_file" | "write_file" | "refactor"
            ))
    {
        let result_msg = Message::tool_result(
            &tc.id,
            "Blocked: plan mode is read-only. No edits or shell commands allowed.",
        );
        messages.push(result_msg.clone());
        conversation_history.push(result_msg);
        ui.tool_result(&tc.function.name, false, "blocked in plan mode");
        return AdmitFlow::NextCall;
    }

    // Write gating: require a plan before write tools (strict only).
    let is_write_action = is_file_write(tc.function.name.as_str());
    if strict && ctx.config.tools.plan && !tools::plan::plan_exists(ctx.config) && is_write_action {
        let result_msg = Message::tool_result(
            &tc.id,
            "Create a plan first: use plan(action='set') with your step-by-step approach before making changes.",
        );
        messages.push(result_msg.clone());
        conversation_history.push(result_msg);
        ui.event(UiEvent::WriteBlockedNoPlan {
            name: tc.function.name.clone(),
        });
        return AdmitFlow::NextCall;
    }
    // (Plan-checkpoint used to hard-block writes after N edits without a
    //  plan action; that interacted poorly with the compile-gate on
    //  `plan(check)` — if the project didn't compile, the model couldn't
    //  escape the block, couldn't fix the project, deadlock. Now we just
    //  warn at the threshold via PLAN_CHECKPOINT_WARNING appended to the
    //  tool result; the model decides what to do.)

    AdmitFlow::Run(Admitted {
        args,
        args_summary,
        call_key,
        file_action,
    })
}
