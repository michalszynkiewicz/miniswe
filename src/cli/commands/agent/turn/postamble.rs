//! Round tail, run after the `for tc in &tool_calls { ... }` batch loop:
//! history pruning, the headless-only `tools.stuck_check` fire, the early
//! no-plan nudge, and stall detection. Shared verbatim by both loops except
//! where [`TurnOptions`] names a delta (here, just the stuck-check gate).
//! No control flow leaves this tail in either loop — the function returns
//! `()`.

use crate::cli::commands::agent::skill_step::step_judge_escalation;
use crate::cli::commands::agent::stuck_check;
use crate::cli::commands::agent::turn_state::TurnState;
use crate::cli::commands::agent::ui::AgentUi;
use crate::config::EditMode;
use crate::llm::Message;
use crate::tools;

use super::{Prunable, TurnCtx, TurnOptions};

/// Everything the round does after the tool-call batch finishes — shared
/// verbatim by both loops except where [`TurnOptions`] names a delta.
pub(crate) async fn finish_round<U: AgentUi>(
    ctx: TurnCtx<'_>,
    opts: TurnOptions,
    state: &mut TurnState,
    ui: &mut U,
    messages: &mut Vec<Message>,
    conversation_history: &mut Vec<Message>,
    round: usize,
    strict: bool,
    prunable: Prunable,
) {
    // History pruning: if every tool call in this assistant message was
    // a prunable validator failure, drop the assistant message + its
    // tool_results and replace with a user-role corrective. The
    // assistant's bad-shape arguments are what prime the model to
    // repeat them; removing them breaks the loop. Verified empirically
    // (probe D3): clean history → clean output.
    if prunable.all_failures && !prunable.errors.is_empty() {
        messages.truncate(prunable.messages_pre);
        conversation_history.truncate(prunable.history_pre);
        let hint = Message::user(&format!(
            "Your previous refactor call(s) were rejected:\n\n{}\n\n\
             Retry with all required parameters and a clean position value \
             (one of 'start' or 'after:<single_param_name>').",
            prunable.errors.join("\n\n---\n\n")
        ));
        messages.push(hint.clone());
        conversation_history.push(hint);
        ctx.log.tool_debug(
            "agent",
            &format!(
                "history pruned: dropped {} tool_result(s) after refactor validator failure",
                prunable.errors.len()
            ),
        );
    }

    // `tools.stuck_check`: T2c frozen-signature fire → append the stuck/
    // done note to the round's last tool result (the placement the
    // warm-replay probes validated; a trailing user message was not what
    // was tested). Runs AFTER history pruning so the note can't land on
    // a tool result that was just truncated away. Headless only — the REPL
    // never sets `opts.stuck_tracking`.
    if opts.stuck_tracking
        && ctx.config.tools.stuck_check
        && let Some(kind) = state
            .stuck_tracker
            .check_fire(round, ctx.session_start.elapsed().as_secs_f64())
    {
        // Red fire while a skill step is active → the fresh-context step
        // judge decides instead of the generic note (the removed
        // MAX_ROUNDS_PER_STEP valve's replacement — rationale at
        // STEP_JUDGE_PROMPT). The helper performs each verdict's side
        // effects and returns one injected note; judge failure leaves
        // `escalation` unset so the plain note goes out below. Bounded
        // by the tracker's MAX_FIRES and the run's max_rounds.
        let escalation: Option<String> = if kind == stuck_check::StuckKind::Red {
            let trigger = format!(
                "Its observable state (compiler/test/check signals) has been frozen for the \
                 last {} rounds.",
                state.stuck_tracker.frozen_rounds()
            );
            step_judge_escalation(
                ctx.task,
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
        } else {
            None
        };
        let plan_done =
            tools::plan::plan_exists(ctx.config) && !tools::plan::has_unchecked_steps(ctx.config);
        let note = if let Some(esc) = escalation {
            esc
        } else if kind == stuck_check::StuckKind::Green && plan_done {
            stuck_check::done_note()
        } else {
            let first_unchecked = tools::plan::parsed_steps(ctx.config)
                .iter()
                .find(|(checked, _, _)| !checked)
                .and_then(|(_, n, _)| *n);
            stuck_check::stuck_note(
                state.stuck_tracker.frozen_rounds(),
                state.stuck_tracker.frozen_minutes(),
                state.stuck_tracker.looping_read_path(),
                first_unchecked,
            )
        };
        let mut appended = false;
        for msgs in [&mut *messages, &mut *conversation_history] {
            if let Some(m) = msgs.iter_mut().rev().find(|m| m.role == "tool")
                && let Some(c) = m.content.as_mut()
            {
                c.push('\n');
                c.push_str(&note);
                appended = true;
            }
        }
        if !appended {
            // No tool result this round (pruned, or a pure-text reply
            // survived the gates) — fall back to a user message.
            let msg = Message::user(&note);
            messages.push(msg.clone());
            conversation_history.push(msg);
        }
        ui.status(&format!(
            "[stuck-check] fired ({}) after {} frozen rounds",
            if kind == stuck_check::StuckKind::Green {
                "green"
            } else {
                "red"
            },
            state.stuck_tracker.frozen_rounds(),
        ));
        ctx.log.tool_debug(
            "agent",
            &format!(
                "stuck-check fired: kind={kind:?} plan_done={plan_done} frozen_rounds={} note={}",
                state.stuck_tracker.frozen_rounds(),
                crate::truncate_chars(&note, 120),
            ),
        );
    }

    // Early no-plan nudge: edit tools are hidden until plan(action='set').
    // The system prompt explains this but some models (GPT-OSS in particular)
    // ignore it and explore until the stall warning fires at round 20+ —
    // wasting most of an attempt. Nudge around round 12 so the model gets a
    // course correction before it's deeply stuck, but late enough that real
    // multi-file exploration has had room to breathe (a few file reads, a
    // search, a goto_definition or two).
    if strict && round >= 12 && !state.nudged_no_plan && !tools::plan::plan_exists(ctx.config) {
        // Must match the now-uniform post-unlock surface (refactor
        // for all, edit_file hidden). Mismatch here is exactly the
        // schema-runtime confusion we work to avoid.
        let unlock_tools = "refactor, replace_range, insert_at, write_file";
        messages.push(Message::user(&format!(
            "[Reminder: you've explored for several rounds without a plan. \
             Call plan(action='set') with your step-by-step approach now — \
             the edit tools ({unlock_tools}) are hidden until you do, and \
             you'll need them to make changes.]"
        )));
        state.nudged_no_plan = true;
    }

    // Stall detection: too many tool calls without any edits.
    // Content is plan-state aware: without a plan the edit tools are
    // hidden, so pointing the model at them is a schema-runtime
    // mismatch. Re-fire the plan nudge instead (with a more urgent
    // tone than the round-12 first nudge).
    if state.calls_since_last_edit >= 20 && state.calls_since_last_edit.is_multiple_of(20) {
        let body = if strict && !tools::plan::plan_exists(ctx.config) {
            "Still no plan set after 20+ exploration calls. \
             Edit tools cannot appear in your tool list until plan(action='set') is called. \
             Stop exploring and set a plan now — even an imperfect plan can be refined later. \
             If something is blocking you from planning, say so."
                .to_string()
        } else {
            let edit_hint = match ctx.config.tools.edit_mode {
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
