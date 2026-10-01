//! Headless-only skill step-cursor maintenance at the top of a round:
//! handoff descent, step preparation, and the periodic completion judge.
//! Moved out of `run/main_loop.rs` verbatim — the REPL has no cursor, so
//! [`prepare`] is a no-op there.

use crate::cli::commands::agent::skill_cursor;
use crate::cli::commands::agent::skill_router;
use crate::cli::commands::agent::skill_step::{
    descend_into_skill, prepare_step, recent_activity, report_cursor_gaps,
};
use crate::cli::commands::agent::turn_state::SkillTurnState;
use crate::cli::commands::agent::ui::AgentUi;
use crate::cli::commands::agent::validation;
use crate::llm::Message;

use super::{TurnCtx, TurnOptions};

/// Skill step-cursor maintenance at the top of a round (handoff descent,
/// prepare_step, periodic completion judge). Headless only
/// (`TurnOptions::skill_steps`); the REPL has no cursor and returns at once.
pub(crate) async fn prepare(
    ctx: TurnCtx<'_>,
    opts: TurnOptions,
    skill_state: &mut SkillTurnState,
    ui: &mut impl AgentUi,
    messages: &mut Vec<Message>,
    conversation_history: &mut Vec<Message>,
) {
    if !opts.skill_steps {
        return;
    }

    // Skill step-cursor maintenance (harness-owned; runs before the LLM
    // call so the [SKILL STEP] re-injection is current):
    //  1. Handoff: if the current step delegates to an installed skill
    //     not yet on the stack (e.g. build -> integrate), DESCEND into it
    //     (extract its steps + push a frame). The model never resolves
    //     the handoff name itself (a lookup a probe showed it won't do).
    //  2. prepare_step: distil the current step, resolve a prose-level
    //     handoff, generate its DONE WHEN check. Runs once in the normal
    //     position AND again after every advance in this block — see the
    //     note on prepare_step for why an unprepared step is dangerous.
    // There is deliberately NO per-step round budget here — a fixed cap
    // is a refuted rounds-only trigger and fires unavoidably on an
    // unsatisfiable step (rationale at STEP_JUDGE_PROMPT). A stuck step
    // escalates to the step judge through two triggers instead: the
    // stuck_check Red fire (frozen signals), and every K-th blocked
    // premature finish at the stop-valve (the model insisting the task
    // is done while steps remain).
    let mut cursor = skill_cursor::load(ctx.config);
    if cursor.is_active() {
        let installed: Vec<String> = crate::skills::discover(&ctx.config.project_root)
            .into_iter()
            .map(|e| e.name)
            .collect();
        if let Some(next) = cursor.handoff_target(&installed)
            && !descend_into_skill(
                &mut cursor,
                &next,
                ctx.config,
                ctx.llm_worker,
                ctx.cancelled,
            )
            .await
        {
            // Name-based handoff (umbrella "Invoke X skill" step) but
            // extraction failed. Step extraction is an LLM call, so a
            // failure can be transient — retry on later rounds and only
            // consume the invoke step (skipping the whole sub-skill)
            // after several consecutive failures.
            const MAX_HANDOFF_FAILURES: usize = 3;
            let n = cursor.note_handoff_failure();
            if n >= MAX_HANDOFF_FAILURES {
                // Abandoned, not done: skipping an invoke step skips
                // the entire sub-skill behind it. Not guarded by
                // may_auto_advance — the blocker here is an extraction
                // that will not resolve, so holding on the step would
                // retry it forever instead of grinding on real work.
                cursor.mark_abandoned();
                ui.status(&format!(
                    "[skills] handoff '{next}' yielded no steps {n}x; skipping (sub-skill NOT run)"
                ));
                report_cursor_gaps(&cursor);
            } else {
                ui.status(&format!(
                    "[skills] handoff '{next}' yielded no steps (attempt {n}); will retry"
                ));
            }
        }
        if cursor.is_active() {
            let n = cursor.note_round();
            if !prepare_step(
                &mut cursor,
                &installed,
                ctx.config,
                ctx.llm_worker,
                ctx.cancelled,
            )
            .await
            {
                // Periodic completion judge — the out-of-band advance
                // driver, for models that don't call skill(done) on
                // their own (an early e2e model managed 2 calls in 4
                // attempts). Do NOT assume it is the primary path: the
                // 2026-08-28 Laguna run advanced 14 steps by the
                // model's own done-tool call against 2 from this judge,
                // so changes to the done-tool handler sit on the hot
                // path. Every JUDGE_EVERY rounds we ask out-of-band
                // whether the step is done. On DONE the
                // step's check acts as a VETO (not an auto-advancer, which
                // would skip a creates-then-modifies step the instant its
                // file exists): check fails → hold + tell it; check passes
                // or is absent → advance. Probe: judge 40/40 honest (8/8
                // NOT DONE on stubs), so DONE is earned, not rubber-stamped.
                const JUDGE_EVERY: usize = 3;
                if n.is_multiple_of(JUDGE_EVERY)
                    && let Some(def) = cursor.cached().map(str::to_string)
                {
                    let (skill_name, step_name) = cursor
                        .current()
                        .map(|(sk, st)| (sk.to_string(), st.name.clone()))
                        .unwrap_or_default();
                    let check = cursor.current_check().map(str::to_string);
                    let recent = recent_activity(messages, 8, 2600);
                    let (judged_done, reason) = skill_router::judge_step_done(
                        ctx.llm_worker,
                        &step_name,
                        &def,
                        &recent,
                        ctx.cancelled,
                    )
                    .await;
                    if judged_done {
                        let verdict = match &check {
                            Some(cmd) => validation::run_check_command(ctx.config, cmd).await,
                            None => validation::CheckOutcome::Skipped,
                        };
                        if let validation::CheckOutcome::Fail(out) = verdict {
                            ui.status(&format!(
                                "[skills] '{step_name}' judged done but its check failed — holding"
                            ));
                            let msg = Message::user(&format!(
                                "[Status check on the '{step_name}' step: you consider it \
                             done, but its completion check failed:\n{out}\nThe step is \
                             NOT complete — fix this before moving on.]"
                            ));
                            messages.push(msg.clone());
                            conversation_history.push(msg);
                        } else {
                            cursor.mark_done();
                            skill_state.last_judge_block = None;
                            match cursor.current() {
                                Some((_, next)) => ui.status(&format!(
                                    "[skills] '{step_name}' judged done → advancing to '{}'",
                                    next.name
                                )),
                                None => ui.status(&format!(
                                    "[skills] '{step_name}' judged done → {skill_name} skill complete"
                                )),
                            }
                            report_cursor_gaps(&cursor);
                            // The judge runs LAST in this block, so the
                            // step it advances onto has missed this
                            // round's preparation. Prepare it now — an
                            // unprepared step goes out with no body, no
                            // DONE WHEN check to veto a premature done,
                            // and no handoff resolution.
                            prepare_step(
                                &mut cursor,
                                &installed,
                                ctx.config,
                                ctx.llm_worker,
                                ctx.cancelled,
                            )
                            .await;
                        }
                    } else {
                        // Surface the judge's reason to the model — it's
                        // often the correct diagnosis the silent gate was
                        // discarding (e2e: it flagged the tmp_repo build).
                        // Dedup so an identical reason isn't re-nudged every
                        // cycle (repeats feed loops).
                        ui.status(&format!("[skills] '{step_name}' judged not done: {reason}"));
                        // Record the standing verdict for the log-only
                        // judge-veto check in the skill(done) handler.
                        skill_state.last_judge_block = Some((
                            format!("{skill_name}::{step_name}"),
                            reason.clone(),
                            skill_state.edits_total,
                        ));
                        if !reason.is_empty()
                            && skill_state.last_judge_nudge.as_deref() != Some(reason.as_str())
                        {
                            skill_state.last_judge_nudge = Some(reason.clone());
                            let msg = Message::user(&format!(
                                "[Status check on the '{step_name}' step — it is NOT done \
                             yet: {reason}\nAddress this specifically before continuing.]"
                            ));
                            messages.push(msg.clone());
                            conversation_history.push(msg);
                        }
                    }
                }
            }
        }
        skill_cursor::save(ctx.config, &cursor);
    }
}
