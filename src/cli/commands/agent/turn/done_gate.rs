//! No-tool-call done-gate: everything the loop does when the model returns
//! no tool calls this round — premature-exit nudges, the skill step-cursor
//! finish-gate, the live-jobs finish-gate, and the behavioral done-gate
//! (with its restart/replan/debugger/context-reset escalation ladder).
//! Shared verbatim by both loops except where [`TurnOptions`] names a delta.

use crate::cli::commands::agent::debugger;
use crate::cli::commands::agent::hints::PREMATURE_EXIT_NUDGE;
use crate::cli::commands::agent::skill_cursor;
use crate::cli::commands::agent::skill_step::{report_cursor_gaps, step_judge_escalation};
use crate::cli::commands::agent::spiral;
use crate::cli::commands::agent::turn_state::{SkillTurnState, TurnState};
use crate::cli::commands::agent::ui::AgentUi;
use crate::cli::commands::agent::validation;
use crate::context;
use crate::llm::Message;
use crate::tools;

use super::restart;
use super::{RoundFlow, TurnCtx, TurnOptions};

/// What the round does when the model returns no tool calls this round —
/// shared verbatim by both loops except where [`TurnOptions`] names a delta.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn check(
    ctx: TurnCtx<'_>,
    opts: TurnOptions,
    state: &mut TurnState,
    skill_state: &mut SkillTurnState,
    ui: &mut impl AgentUi,
    messages: &mut Vec<Message>,
    conversation_history: &mut Vec<Message>,
    assistant_msg: &Message,
) -> RoundFlow {
    let strict = ctx.config.tools.ceremony == crate::config::CeremonyMode::Strict;
    // Bound as `message` so the gate-replan format string below stays
    // byte-identical to the headless original — these literals are grepped by
    // the bench tooling.
    let message = ctx.task;

    // Two distinct "model returned nothing" situations:
    //  (1) plan exists, steps remain → standard mid-task exit
    //  (2) no plan set yet → model stopped during exploration
    //      before doing meaningful work. Mistral Small 4 with
    //      reasoning_effort=high triggered this — read a few
    //      files, reasoned heavily, then returned empty.
    // Both deserve one nudge to recover.
    if strict && !state.nudged_premature_exit && ctx.config.tools.plan {
        let has_unchecked = tools::plan::has_unchecked_steps(ctx.config);
        let plan_exists = tools::plan::plan_exists(ctx.config);
        if has_unchecked || !plan_exists {
            state.nudged_premature_exit = true;
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
            return RoundFlow::Continue;
        }
    }

    // Skill step-cursor finish-gate (PERSISTENT): a cursor with
    // steps remaining means the whole task (build → integrate →
    // deploy) is NOT done, so the model must not be allowed to
    // finish here — it satisfices at the first artifact (e2e
    // 2026-07-17: stopped at 65/3000 rounds with the cursor at
    // build 9/18). On every stop attempt we block the finish and
    // re-nudge; there is no per-step round budget anymore
    // (rationale at STEP_JUDGE_PROMPT), so a step ends only via
    // skill(done), a judge advance, or an abandon.
    // Anti-spin: if it insists (stops SKILL_EXIT_MAX_STOPS times on
    // the SAME step), take that as "done with this step" and advance
    // the cursor rather than spinning forever on the nudge.
    if opts.skill_steps {
        let mut cursor = skill_cursor::load(ctx.config);
        if let Some((skill, step)) = cursor
            .current()
            .map(|(sk, st)| (sk.to_string(), st.name.clone()))
        {
            const SKILL_EXIT_MAX_STOPS: usize = 3;
            let key = format!("{skill}::{step}");
            if skill_state.exit_step.as_deref() != Some(&key) {
                skill_state.exit_step = Some(key);
                skill_state.exit_stops = 0;
                skill_state.stop_judge_fires = 0;
            }
            skill_state.exit_stops += 1;
            // As with the round cap, a frame's last step is never
            // retired out-of-band: the model stopping on it is not
            // evidence the phase is over. Falling through leaves
            // the nudge below to push it back to work.
            if skill_state.exit_stops >= SKILL_EXIT_MAX_STOPS && cursor.may_auto_advance() {
                cursor.mark_abandoned();
                skill_cursor::save(ctx.config, &cursor);
                skill_state.exit_step = None;
                skill_state.exit_stops = 0;
                skill_state.stop_judge_fires = 0;
                let msg = match cursor.current() {
                    Some((_, next)) => format!(
                        "[You kept trying to finish while on the '{step}' step — \
                         moving you off it. It is NOT complete. Next step: '{}' (see \
                         [SKILL STEP]). Do NOT stop: the task is not complete until \
                         the package is built, integrated, AND deployed.]",
                        next.name
                    ),
                    None => format!(
                        "[You kept trying to finish the '{step}' step — moving you off \
                         it. That was the last {skill} step; verify the task \
                         (build → integrate → deploy) is truly finished before stopping.]"
                    ),
                };
                let m = Message::user(&msg);
                messages.push(m.clone());
                conversation_history.push(m);
                ui.status(&format!(
                    "[skills] model kept stopping on '{step}' — abandoning it (NOT done)"
                ));
                report_cursor_gaps(&cursor);
                return RoundFlow::Continue;
            }
            // A step the anti-spin valve can't retire (a frame's
            // LAST step — may_auto_advance()=false) used to mean
            // nudge-forever: the 2026-09-01 e2e logged 70 blocked
            // finishes on DeployReviewWorkspace with no
            // escalation. Repeated insistence that the task is
            // done is the cleanest can't-finish signal we have,
            // so every SKILL_EXIT_MAX_STOPS-th blocked stop asks
            // the step judge (capped per step); judge failure
            // falls through to the plain nudge.
            const STOP_JUDGE_MAX_FIRES: usize = 3;
            if skill_state.exit_stops >= SKILL_EXIT_MAX_STOPS
                && skill_state.exit_stops.is_multiple_of(SKILL_EXIT_MAX_STOPS)
                && skill_state.stop_judge_fires < STOP_JUDGE_MAX_FIRES
            {
                skill_state.stop_judge_fires += 1;
                let trigger = format!(
                    "It has tried to declare the whole task finished \
                     {} times while on this step; the harness \
                     refused each time because steps remain.",
                    skill_state.exit_stops
                );
                if let Some(note) = step_judge_escalation(
                    message,
                    &trigger,
                    ctx.config,
                    ctx.llm_worker,
                    ctx.tool_defs,
                    ctx.perms,
                    ctx.lsp,
                    ctx.fast_revisions,
                    ctx.fast_baseline_errors,
                    ctx.cancelled,
                    &mut state.force_compact_next_round,
                )
                .await
                {
                    let m = Message::user(&note);
                    messages.push(m.clone());
                    conversation_history.push(m);
                    return RoundFlow::Continue;
                }
            }
            let nudge = Message::user(&format!(
                "[Don't stop — the task is NOT done. You're on the '{step}' step of \
                 the {skill} skill (build → integrate → deploy lifecycle); see \
                 [SKILL STEP]. If this step is complete call skill(action='done'), \
                 otherwise keep working it. Do not finish until the package is built, \
                 integrated, AND deployed.]"
            ));
            messages.push(nudge.clone());
            conversation_history.push(nudge);
            ui.status(&format!(
                "[skills] blocked premature finish on '{step}' (stop #{})",
                skill_state.exit_stops
            ));
            return RoundFlow::Continue;
        }
    }

    // Live-jobs finish-gate: finishing while background jobs run
    // abandons them (session end kills them — a deploy started
    // with background=true dies half-way). One nudge to wait or
    // kill deliberately; jobs e2e (2026-07-14) showed the model
    // fire-and-forgetting a background deploy otherwise.
    if opts.live_jobs_gate && !ctx.job_registry.is_empty() && !state.nudged_live_jobs {
        state.nudged_live_jobs = true;
        let nudge = Message::user(
            "[Background job(s) still running — the task is not done. \
             Wait for them with shell(action='wait', secs=60, check='<status command>') \
             and verify the result, or shell(action='kill') them deliberately. \
             Finishing now would abandon and kill them.]",
        );
        messages.push(nudge.clone());
        conversation_history.push(nudge);
        return RoundFlow::Continue;
    }

    // Behavioral done-gate: before accepting completion, verify the
    // change actually works at runtime. A configured check that
    // exits non-zero blocks the exit and feeds its output back so
    // the model can fix a plumbed-but-not-consumed change (the
    // change compiles + tests pass but the feature doesn't work).
    // Default config has no command → this is a no-op UNLESS a
    // skill step is active with a generated completion check,
    // which becomes the effective command (lighting up this gate
    // + the debugger per-step). See docs/success-validation-design.md.
    let effective_check = if opts.skill_steps {
        skill_cursor::current_check_command(ctx.config)
    } else {
        None
    }
    .or_else(|| ctx.config.validation.command().map(str::to_string));
    if !opts.read_only
        && state.gate.validation_blocks < ctx.config.validation.max_retries
        && let Some(check_cmd) = effective_check.as_deref()
    {
        match validation::run_check_command(ctx.config, check_cmd).await {
            validation::CheckOutcome::Fail(output) => {
                state.gate.validation_blocks += 1;
                // Record the model's completion rationale (its
                // no-tool-call exit content). If it believes the
                // check is wrong, this is its bounded, auditable
                // voice — it counts as a block, not a free pass.
                if let Some(rationale) = assistant_msg
                    .content
                    .as_deref()
                    .map(str::trim)
                    .filter(|c| !c.is_empty())
                {
                    tracing::warn!(
                        "[validation] blocked completion (attempt {}); model rationale: {}",
                        state.gate.validation_blocks,
                        crate::truncate_chars(rationale, 300)
                    );
                    state.gate.validation_disputes.push(rationale.to_string());
                }
                ui.status("Behavioral check failed — not done yet.");

                // Full restart (opt-in `tools.gate_restart`): on the
                // FIRST gate block, ABANDON the (possibly poisoned)
                // attempt — revert the WHOLE tree to the clean baseline
                // (round 0) AND reset the context to a fresh from-scratch
                // attempt at the task, clearing the degraded plan. Tests
                // detect-and-restart: a stuck/off-path state is worse than
                // a clean start (run2), so scrap it. Fires once per turn.
                if ctx.config.tools.gate_restart && !state.gate.restart_fired {
                    state.gate.restart_fired = true;
                    state.plan_ever_set = false;
                    *messages = restart::scrap_restart(
                        ui,
                        ctx.config,
                        message,
                        ctx.mcp_summary,
                        ctx.snapshots,
                        ctx.plan_only,
                        false,
                    );
                    conversation_history.clear();
                    state.gate.validation_blocks = 0;
                    state.gate.plan_step_failures.reset();
                    return RoundFlow::Continue;
                }

                // Goal re-anchor (opt-in `tools.gate_replan`): the first
                // time the gate blocks on BEHAVIOR (the tree compiles but
                // the feature doesn't work), the agent may be running a
                // degraded compile-repair plan that dropped the feature
                // objective (run2: it fixes the compile and stops at
                // "compiles", never writing the consumption its original
                // plan called for). Re-anchor on the ORIGINAL goal and
                // force a fresh plan. Fires once per turn. CRUCIAL: skip
                // when the block is a COMPILE failure — re-anchoring the
                // agent to "add the behavior" on a broken tree just makes
                // it dig deeper; only fire once the compile is green.
                let is_compile_fail = output.contains("DOES NOT COMPILE")
                    || output.contains("could not compile")
                    || output.contains("error[E");
                if ctx.config.tools.gate_replan && !state.gate.replan_fired && !is_compile_fail {
                    state.gate.replan_fired = true;
                    ui.status("Re-anchoring on the original goal — re-plan from the task…");
                    let msg = Message::user(&format!(
                        "[A check that exercises the change end-to-end FAILED — it \
                         COMPILES but does not yet BEHAVE as required. After fixing \
                         errors it is easy to lose the original goal and stop at \"it \
                         compiles\". Re-anchor on the task: \"{message}\". Use \
                         plan(action='set') to re-derive the FULL plan from that goal — \
                         list every step the feature needs end-to-end, INCLUDING the \
                         code that actually USES the new input to change behavior (not \
                         just declaring or plumbing it). For each step, confirm it is \
                         DONE in the code, not merely compiling — then implement \
                         whatever is missing before finishing.\nCheck output:\n{output}]"
                    ));
                    messages.push(msg.clone());
                    conversation_history.push(msg);
                    return RoundFlow::Continue;
                }

                // Reactive debugger (opt-in): once the primary
                // agent has failed the gate a couple times on its
                // own, hand the SPECIFIC failure to a fresh-context
                // sub-agent. Its fix lands in the shared revision
                // store, so the next gate re-check (continue below)
                // validates it. Single-fire by default; with
                // `debugger_multifire` it re-fires only on a CHANGED
                // failure signature (walk compile→smoke).
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
                    ui.status("Still failing — spinning up a fresh-context debugger sub-agent…");
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

                    // Debugger-as-judge: SCRAP → the LOOP executes the
                    // whole-tree restart (the stuck agent never decides);
                    // Rewind → the loop reverts JUST the one flagged file;
                    // Report → inject the diagnosis for the main agent.
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
                            return RoundFlow::Continue;
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
                    return RoundFlow::Continue;
                }

                // Gate context-reset (opt-in): instead of grinding
                // in-context after repeated gate blocks, drop the
                // polluted history and re-assemble a clean context —
                // the in-session equivalent of a best-of-3 fresh
                // attempt (files persist on disk). Bounded per turn.
                if ctx.config.tools.gate_context_reset
                    && state.gate.context_resets < spiral::MAX_GATE_RESETS
                    && state.gate.validation_blocks >= spiral::GATE_RESET_AFTER_BLOCKS
                {
                    state.gate.context_resets += 1;
                    state.gate.validation_blocks = 0; // fresh gate budget for the clean restart
                    let fresh = spiral::build_gate_reset_prompt(message, &output);
                    let assembled =
                        context::assemble(ctx.config, &fresh, &[], ctx.plan_only, ctx.mcp_summary);
                    *messages = assembled.messages;
                    ui.status("Gate context-reset — fresh start (history cleared, files kept).");
                    ctx.log.tool_debug(
                        "agent",
                        "gate context-reset: re-assembled clean context after repeated gate blocks",
                    );
                    return RoundFlow::Continue;
                }

                let msg = Message::user(&validation::build_verification_failed_message(&output));
                messages.push(msg.clone());
                conversation_history.push(msg);
                return RoundFlow::Continue;
            }
            validation::CheckOutcome::Pass | validation::CheckOutcome::Skipped => {}
        }
    }
    // Exiting now. If the gate blocked the model along the way,
    // surface its recorded rationale(s) for audit — whether it
    // ultimately fixed the change or exhausted the retry budget.
    if !state.gate.validation_disputes.is_empty() {
        ui.status(&format!(
            "Completed after {} blocked verification(s); model's reasons recorded in the log.",
            state.gate.validation_disputes.len()
        ));
        tracing::warn!(
            "[validation] turn completed over {} blocked check(s); model rationale(s): {}",
            state.gate.validation_disputes.len(),
            state.gate.validation_disputes.join(" | ")
        );
    }
    RoundFlow::EndTurn { error: false }
}
