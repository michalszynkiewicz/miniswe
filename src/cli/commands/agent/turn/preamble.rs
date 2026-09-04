//! Round preamble: snapshot, revert-to-green, max-rounds stop,
//! pause-after-N, approaching-limit warning, plan-panel refresh,
//! repeated-read pruning, forced/budgeted compaction, sanitize.

use crate::cli::commands::agent::prune_reads::prune_repeated_reads;
use crate::cli::commands::agent::spiral;
use crate::cli::commands::agent::turn_state::TurnState;
use crate::cli::commands::agent::ui::{AgentUi, PauseDecision, UiEvent};
use crate::config::EditMode;
use crate::context;
use crate::llm::Message;
use crate::tools;

use super::{RoundFlow, TurnCtx};

/// Everything between `log.round_start` and LLM-request assembly, shared
/// verbatim by both loops. Loop-specific round setup (REPL interrupt
/// checkpoint, headless stuck tracker + skill cursor) stays in the
/// skeletons above the call.
pub(crate) async fn begin_round(
    ctx: TurnCtx<'_>,
    state: &mut TurnState,
    ui: &mut impl AgentUi,
    messages: &mut Vec<Message>,
    round: usize,
) -> RoundFlow {
    // Snapshot at the start of each round for revert support (SCRAP /
    // revert-to-green rely on these per-round commits in the shadow repo).
    if let Some(snap) = ctx.snapshots {
        let mut guard = snap.lock();
        let _ = guard.begin_round(round);
    }

    // revert-to-green: this round STARTS from the state the previous round
    // left (just snapshotted above). If the project has been broken above
    // baseline for spiral::REVERT_TO_GREEN_BLOCKS rounds, the agent is
    // digging deeper, not recovering — reset the whole tree to the last
    // green snapshot and tell it to start over from a clean base.
    if ctx.config.tools.revert_to_green
        && ctx.config.tools.edit_mode == EditMode::Fast
        && let Some(snap) = ctx.snapshots
    {
        let errs = tools::fast::project_error_count(ctx.lsp.as_deref()).await;
        if errs <= ctx.fast_baseline_errors {
            state.green.last_green_round = round;
            state.green.red_streak = 0;
        } else {
            state.green.red_streak += 1;
            if state.green.red_streak >= spiral::REVERT_TO_GREEN_BLOCKS {
                let result = {
                    let guard = snap.lock();
                    guard.revert_to_round(state.green.last_green_round)
                };
                match result {
                    Ok(m) => {
                        ui.status(&format!(
                            "[revert-to-green] stuck {} rounds; {m}",
                            state.green.red_streak
                        ));
                        messages.push(Message::user(&spiral::build_revert_to_green_message(
                            state.green.red_streak,
                            state.green.last_green_round,
                        )));
                        state.green.red_streak = 0;
                    }
                    Err(e) => {
                        ui.event(UiEvent::RevertToGreenFailed {
                            error: e.to_string(),
                        });
                    }
                }
            }
        }
    }

    if round > ctx.max_rounds {
        ui.event(UiEvent::MaxRoundsReached);
        return RoundFlow::EndTurn { error: false };
    }

    // Ask whether to continue after pause_after_rounds rounds (headless
    // auto-continues inside HeadlessUi — blocking on stdin hung real e2e
    // runs); max_rounds stays the hard stop.
    let pause_at = ctx.config.context.pause_after_rounds;
    if round == pause_at && !state.user_continued {
        match ui.confirm_continue(pause_at).await {
            PauseDecision::Continue => state.user_continued = true,
            PauseDecision::WrapUp => {
                messages.push(Message::user("[Stop now. Summarize what you've done.]"))
            }
        }
    }

    // Warn the LLM when approaching the hard limit.
    if round == ctx.max_rounds.saturating_sub(5) {
        messages.push(Message::user(
            "[Approaching tool limit. Wrap up and summarize.]",
        ));
    }

    // Refresh the live plan panel from plan.md (no-op headless).
    ui.refresh_plan(ctx.config, round);

    // See `agent::prune_reads` — drop the middle of a deep run of identical
    // reads BEFORE compaction, which structurally cannot reach it (it
    // summarizes the oldest end; the loop lives in the newest messages).
    let pruned = prune_repeated_reads(messages);
    if !pruned.is_empty() {
        ctx.log.reads_pruned(
            pruned.removed / 2,
            pruned.keys,
            pruned.deepest.as_deref().unwrap_or("?"),
        );
        ui.event(UiEvent::ReadsPruned {
            pairs: pruned.removed / 2,
            keys: pruned.keys,
            deepest: pruned.deepest,
        });
    }

    // Unified context compression — handles both tool results and conversation
    let pre = messages.len();
    // Read-loop escalation (see REPEATED_READ_ESCALATION): the loop is
    // sustained by the cache-hot prompt prefix, so break it deliberately
    // even though no budget pressure asks for it. Runs before
    // maybe_compress so refresh_current_state still lands on the tail.
    if state.force_compact_next_round {
        state.force_compact_next_round = false;
        ui.event(UiEvent::ForcingCompaction);
        ui.pump(context::compressor::force_compress(
            messages,
            ctx.config,
            ctx.router,
            ctx.llm_worker,
            ctx.tool_def_tokens,
        ))
        .await;
    }
    ui.pump(context::compressor::maybe_compress(
        messages,
        ctx.config,
        ctx.router,
        ctx.llm_worker,
        ctx.tool_def_tokens,
        &mut state.plan_update_requested,
    ))
    .await;
    ctx.log
        .masking_applied(pre.saturating_sub(messages.len()), pre);

    // Sanitize message roles before sending (strict chat template compat)
    context::sanitize_messages(messages);

    RoundFlow::Continue
}
