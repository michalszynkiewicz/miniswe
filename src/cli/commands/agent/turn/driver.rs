//! The single agent round loop: `'round` / `for tc` live ONLY here. Both
//! frontends (`run/main_loop.rs`'s headless `run()` and `repl/agent_turn.rs`)
//! build a [`TurnCtx`] and a [`TurnOptions`] once above the call and hand
//! them to [`run_turn`], which owns every `break`/`continue` for the turn.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::cli::commands::agent::skill_cursor;
use crate::cli::commands::agent::turn_state::{SkillTurnState, TurnState};
use crate::cli::commands::agent::ui::AgentUi;
use crate::config::CeremonyMode;
use crate::llm::{Message, ModelRouter};
use crate::logging::SessionLog;

use super::{
    AdmitFlow, CallFlow, LlmFlow, Prunable, RoundFlow, TurnCtx, TurnOptions, call_gate, dispatch,
    done_gate, llm_call, postamble, preamble, skill_round,
};

/// What the turn ended with — the two loops' post-loop needs.
pub(crate) struct TurnResult {
    /// Rounds run (the value `log.session_end` was given).
    pub rounds: usize,
    /// The headless `had_error` exit flag (REPL callers ignore it).
    pub had_error: bool,
}

/// Consume the interrupt flag: `true` and resets it if it was armed, `false`
/// otherwise. REPL-only (`opts.interrupt_checkpoints`) — the headless loop
/// never reads the cancel flag here, so an armed flag stays armed for the
/// LLM/tool paths to read themselves.
pub(crate) fn consume_interrupt(cancelled: &AtomicBool) -> bool {
    cancelled.swap(false, Ordering::Relaxed)
}

/// The one agent round loop: `'round` / `for tc` live ONLY here. `router`
/// and `log` are separate from `ctx` for the same reason `dispatch::execute`
/// takes them separately — `ctx`'s `router`/`log` fields are bare
/// references, but the `edit_file`/`refactor` dispatch arms need an owned,
/// `'static`-safe `Arc` to move into a `tool_pool.submit(move || ..)`
/// closure.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_turn<U: AgentUi>(
    ctx: TurnCtx<'_>,
    opts: TurnOptions,
    state: &mut TurnState,
    skill_state: &mut SkillTurnState,
    ui: &mut U,
    messages: &mut Vec<Message>,
    conversation_history: &mut Vec<Message>,
    router: &Arc<ModelRouter>,
    log: &Arc<SessionLog>,
) -> TurnResult {
    // Ceremony=Strict re-enables the legacy plan-first machinery (plan gate,
    // plan/no-plan nudges, hide-edit-tools-until-plan). Derived from exactly
    // the config `ctx.config` points at — headless keeps its own `strict`
    // local too, for the pre-loop plan nudge.
    let strict = ctx.config.tools.ceremony == CeremonyMode::Strict;

    let mut round = 0;
    let mut had_error = false;

    'round: loop {
        if had_error {
            break;
        }
        // Check cancellation at the top of every round. REPL only — the
        // headless loop has never consumed the cancel flag here.
        if opts.interrupt_checkpoints && consume_interrupt(ctx.cancelled) {
            ui.notify_interrupted();
            break;
        }

        round += 1;
        log.round_start(round);

        // Session-wide input-token budget guard. Opt-in (0 = disabled,
        // the default) — a hosted provider's per-token billing makes an
        // unbounded context window a real cost risk in a way a local
        // server's own wall-clock slowness already self-limited. Checked
        // at each round boundary against the router's running usage
        // totals, so it trips before the next request goes out rather
        // than mid-round.
        let budget = ctx.config.runtime.max_session_input_tokens;
        if budget > 0 {
            let used = router.usage_totals().prompt_tokens;
            if used >= budget {
                log.budget_exceeded(used, budget);
                ui.error(&format!(
                    "Session input-token budget exceeded ({used} >= {budget} tokens) — stopping."
                ));
                had_error = true;
                break;
            }
        }
        // `tools.stuck_check`'s frozen-signature tracker: feed it the round
        // number + elapsed wall time BEFORE the skill-cursor block below (its
        // periodic judge is an LLM call, so the timestamp must be taken
        // first). Headless only (`opts.stuck_tracking`); the REPL skips it.
        if opts.stuck_tracking {
            state
                .stuck_tracker
                .on_round(round, ctx.session_start.elapsed().as_secs_f64());
        }

        // Skill step-cursor maintenance (harness-owned; runs before the LLM
        // call so the [SKILL STEP] re-injection is current) — handoff
        // descent, prepare_step, periodic completion judge. Headless only;
        // no-op on the REPL side (`opts.skill_steps` is false there).
        skill_round::prepare(ctx, opts, skill_state, ui, messages, conversation_history).await;

        match preamble::begin_round(ctx, state, ui, messages, round).await {
            RoundFlow::Continue => {}
            RoundFlow::EndTurn { error } => {
                if opts.fatal_marks_error {
                    had_error |= error;
                }
                break;
            }
        }

        let assistant_msg =
            match llm_call::generate(ctx, opts, state, ui, messages, conversation_history).await {
                LlmFlow::Ready(msg) => msg,
                LlmFlow::Retry => continue,
                LlmFlow::EndTurn { error } => {
                    if opts.fatal_marks_error {
                        had_error |= error;
                    }
                    break;
                }
            };
        let assistant_msg = &assistant_msg;

        // Check for tool calls
        let tool_calls = match &assistant_msg.tool_calls {
            Some(tc) if !tc.is_empty() => tc.clone(),
            _ => match done_gate::check(
                ctx,
                opts,
                state,
                skill_state,
                ui,
                messages,
                conversation_history,
                assistant_msg,
            )
            .await
            {
                RoundFlow::Continue => continue,
                RoundFlow::EndTurn { .. } => break,
            },
        };

        // Add assistant's tool call message to messages
        messages.push(assistant_msg.clone());

        // Snapshot lengths so we can rewind both buffers if every tool call
        // in this assistant message turned out to be a prunable validator
        // failure. The assistant message and its tool_results then get
        // replaced with a single user-role corrective — this kills the
        // priming chain that keeps the model copying the same bad shape.
        // Both buffers' last entry IS the assistant_msg we just pushed, so
        // truncate one before to also drop it.
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

        // Active skill step, folded into each loop key below so the SAME tool
        // call on different steps (notably skill(done), byte-identical on every
        // step) yields distinct keys — legitimately advancing through steps is
        // not misread as a repeat, while a within-step rut (constant tag) is.
        let active_step_tag = if opts.skill_steps {
            skill_cursor::current_step_tag(ctx.config)
        } else {
            None
        };

        for tc in &tool_calls {
            // Check cancellation between tool calls. REPL only.
            if opts.interrupt_checkpoints && consume_interrupt(ctx.cancelled) {
                ui.notify_interrupted();
                break 'round;
            }
            let admitted = match call_gate::admit(
                ctx,
                opts,
                state,
                ui,
                messages,
                conversation_history,
                tc,
                active_step_tag.as_deref(),
            )
            .await
            {
                AdmitFlow::Run(a) => a,
                AdmitFlow::NextCall => continue,
                AdmitFlow::StopCalls { error } => {
                    had_error |= error;
                    break;
                }
                AdmitFlow::RestartRound => continue 'round,
            };

            // Handle tool dispatch
            let mut result = dispatch::execute(
                ctx,
                opts,
                skill_state,
                ui,
                round,
                tc,
                &admitted.args,
                &admitted.file_action,
                router,
                log,
            )
            .await;

            match dispatch::finish_call(
                ctx,
                opts,
                state,
                skill_state,
                ui,
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
                CallFlow::NextCall => {}
                CallFlow::RestartRound => continue 'round,
            }

            // Re-render after a tool result landed — the TUI redraws its
            // frame so the result shows before the next call starts; no-op
            // headless.
            ui.after_tool_call();
        }

        postamble::finish_round(
            ctx,
            opts,
            state,
            ui,
            messages,
            conversation_history,
            round,
            strict,
            Prunable {
                messages_pre,
                history_pre,
                all_failures: all_prunable_failures,
                errors: prunable_errors,
            },
        )
        .await;
    }

    log.usage_total(&router.usage_totals());
    log.session_end(round, had_error);

    TurnResult {
        rounds: round,
        had_error,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;

    use super::*;

    #[test]
    fn consume_interrupt_clears_flag_after_first_read() {
        let cancelled = AtomicBool::new(true);
        assert!(consume_interrupt(&cancelled));
        assert!(!consume_interrupt(&cancelled));
        assert!(!cancelled.load(Ordering::Relaxed));
    }
}
