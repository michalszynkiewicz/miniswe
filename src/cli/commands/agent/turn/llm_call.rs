//! LLM-call phase: request assembly (plan latch, per-model reasoning
//! kwargs), the streamed call, the error-recovery ladder (compact-retry,
//! truncated/oversized tool calls), post-200 ceiling regeneration, and
//! assistant-message sanitize + history push.

use std::sync::atomic::Ordering;

use crate::cli::commands::agent::hints::{truncated_tool_call_hint, visible_tool_defs};
use crate::cli::commands::agent::skill_cursor;
use crate::cli::commands::agent::turn_state::TurnState;
use crate::cli::commands::agent::ui::{AgentUi, LlmOutcome, UiEvent};
use crate::config::CeremonyMode;
use crate::context;
use crate::llm::{
    ChatRequest, Message, TRUNCATED_CALL_ABORT_AFTER, is_context_exceeded_error,
    is_context_truncated_response, is_tool_call_args_cap_error, is_truncated_tool_call_error,
    sanitize_truncated_tool_calls, scrub_unparseable_tool_calls,
};
use crate::tools;

use super::{CompactionUx, LlmFlow, TurnCtx, TurnOptions};

/// One LLM round trip, shared verbatim by both loops except where
/// [`TurnOptions`] names a delta.
pub(crate) async fn generate(
    ctx: TurnCtx<'_>,
    opts: TurnOptions,
    state: &mut TurnState,
    ui: &mut impl AgentUi,
    messages: &mut Vec<Message>,
    conversation_history: &mut Vec<Message>,
) -> LlmFlow {
    // Call LLM with streaming.
    //
    // Disable thinking mode: Gemma's chat template defaults to a
    // long internal-reasoning pass that lands in `reasoning_content`,
    // which we do NOT persist to history. The reasoning is
    // write-only. Worse, on tight token budgets the reasoning eats
    // the whole response and `content`/`tool_calls` come back empty
    // (probe in /tmp/gemma-thinking-probe.py — 0 chars content,
    // finish_reason=length, after burning 2K tokens reasoning to a
    // simple question). The kwarg is a no-op for models whose chat
    // template doesn't honor it (e.g. Devstral). For strategic
    // reasoning the agent has plan/scratchpad — that's persistent
    // and visible to subsequent turns.
    // Hide edit tools from the model until a plan exists. See
    // visible_tool_defs for rationale.
    let strict = ctx.config.tools.ceremony == CeremonyMode::Strict;
    let plan_set = tools::plan::plan_exists(ctx.config);
    state.plan_ever_set |= plan_set;
    // Off: never hide edit tools (pass plan_exists=true). Strict:
    // legacy hide-until-plan behavior, latched so a plan that goes away
    // mid-segment cannot retract tools the model has already been shown.
    let mut visible = visible_tool_defs(ctx.tool_defs, state.plan_ever_set || !strict);
    // Expose the skill(done) advance control only while a step-cursor is
    // active — inert otherwise, so it never clutters non-skill turns.
    if opts.skill_steps && skill_cursor::load(ctx.config).is_active() {
        visible.push(tools::definitions::skill_tool_definition());
    }
    // Mistral Small 4 honors `reasoning_effort` ("none"/"high"); other
    // models (Gemma, GPT-OSS, Devstral) use `enable_thinking` or
    // ignore the kwarg entirely. For Mistral 4 we want deep reasoning
    // during the planning phase (decomposing the task — exactly where
    // it goes wrong, picking the wrong file family) and fast execution
    // once a plan is set. Per-model gating keeps the cost localized.
    // `model.thinking` opts the main loop into thinking-mode reasoning at
    // `thinking_temperature` (reasoning degenerates at code-task temps —
    // see ModelConfig::thinking_temperature).
    let (chat_template_kwargs, temperature_override) =
        if ctx.config.model.is_mistral_small_4_family() {
            let effort = if plan_set { "none" } else { "high" };
            (serde_json::json!({"reasoning_effort": effort}), None)
        } else if ctx.config.model.thinking {
            (
                serde_json::json!({"enable_thinking": true}),
                Some(ctx.config.model.thinking_temperature),
            )
        } else {
            (serde_json::json!({"enable_thinking": false}), None)
        };
    // Mistral Small 4 with reasoning_effort=high needs significant
    // output budget. Probe data: at 8192 max_tokens the model hits
    // finish_reason=length after ~32K chars of reasoning_content with
    // ZERO chars of content emitted. At 16384 it reasons for ~24K
    // chars and emits a clean ~2K-char correct plan (finish_reason=stop,
    // ~6K tokens used). Per llama.cpp #20668 and vLLM #37081 — known
    // Mistral 4 budget-hungry reasoning behavior.
    let max_tokens_override = if ctx.config.model.is_mistral_small_4_family() {
        Some(16384)
    } else {
        None
    };
    let request = ChatRequest {
        messages: messages.clone(),
        tools: Some(visible),
        tool_choice: None,
        max_tokens_override,
        chat_template_kwargs: Some(chat_template_kwargs),
        temperature_override,
        // Never forced from the agent loop; see `state.loops.window_edit_fires`.
        cache_prompt: None,
    };
    ctx.log.llm_request(&request);

    ui.separator();

    // Reset cancel flag for this round
    ctx.cancelled.store(false, Ordering::Relaxed);

    let response = match ui
        .stream_llm(ctx.llm_worker, ctx.model_role, request, ctx.cancelled)
        .await
    {
        LlmOutcome::Response(r) => r,
        // stream_llm already surfaced the error line for both.
        LlmOutcome::WorkerStopped | LlmOutcome::UiClosed if opts.worker_stopped_ends_turn => {
            return LlmFlow::EndTurn { error: false };
        }
        outcome => {
            let err_str = match outcome {
                LlmOutcome::Error(e) => e,
                // WorkerStopped (UiClosed never occurs headless) routes
                // through the generic error ladder below.
                _ => "LLM worker stopped unexpectedly".to_string(),
            };
            if err_str.contains("Interrupted") {
                if opts.clear_cancel_on_interrupt {
                    ctx.cancelled.store(false, Ordering::Relaxed);
                }
                ui.status("Generation interrupted.");
                return LlmFlow::EndTurn { error: false };
            }
            // The server rejected the request outright: prompt alone
            // exceeds the context window. Compact and resend — this is
            // the primary recovery path for compaction="lazy" (which
            // never compacts proactively), and a safety net for every
            // other strategy.
            if is_context_exceeded_error(&err_str)
                && state.context_compact_retries < context::compressor::FORCE_COMPRESS_MAX_RETRIES
            {
                state.context_compact_retries += 1;
                if opts.compaction == CompactionUx::Interactive {
                    ui.status("Context window exceeded — compacting and retrying.");
                }
                if ui
                    .pump(context::compressor::force_compress(
                        messages,
                        ctx.config,
                        ctx.router,
                        ctx.llm_worker,
                        ctx.tool_def_tokens,
                    ))
                    .await
                {
                    ctx.log
                        .llm_error("context window exceeded — compacted history, retrying");
                    if opts.compaction == CompactionUx::Batch {
                        ui.status("Context window exceeded — compacting and retrying.");
                    }
                    return LlmFlow::Retry;
                }
                // Nothing could be freed — retrying would fail identically.
                if opts.compaction == CompactionUx::Interactive {
                    ui.error("Compaction could not free any context — stopping this turn.");
                    return LlmFlow::EndTurn { error: false };
                }
                // Batch: fall through to the normal error handling.
            }
            if is_tool_call_args_cap_error(&err_str) {
                // Our streaming assembler aborted the generation because
                // an anchor-only tool's arguments outgrew the cap (see
                // llm::tool_call_args_cap). Nothing was persisted; tell
                // the model what it did and let it re-issue.
                state.truncated_call_errors_in_a_row += 1;
                if state.truncated_call_errors_in_a_row >= TRUNCATED_CALL_ABORT_AFTER {
                    ctx.log.llm_error(&format!(
                        "{} consecutive oversized tool calls — aborting turn",
                        state.truncated_call_errors_in_a_row
                    ));
                    ui.error(
                        "The model keeps emitting oversized tool-call arguments — giving up on this turn.",
                    );
                    return LlmFlow::EndTurn { error: true };
                }
                ctx.log.llm_error(&format!(
                    "tool call aborted by the argument size cap: {err_str}"
                ));
                ui.status("Tool call arguments exceeded the size cap — retrying with guidance.");
                let hint = Message::user(&format!(
                    "{err_str}. Anchor-style tools take identifiers and short expressions only — \
                     never paste code bodies into their arguments. {}",
                    truncated_tool_call_hint(ctx.config.tools.edit_mode)
                ));
                messages.push(hint.clone());
                conversation_history.push(hint);
                return LlmFlow::Retry;
            }
            if is_truncated_tool_call_error(&err_str) {
                // The server's chat template could not parse some
                // assistant tool call's arguments as JSON. Two sources:
                // the model hit the output/context ceiling mid-call on
                // THIS request (non-streaming path, nothing persisted),
                // or a previously persisted call is broken and every
                // request will keep failing until it is gone. Handle
                // the second before doing anything else: it is the
                // 436-round spin.
                state.truncated_call_errors_in_a_row += 1;
                if state.truncated_call_errors_in_a_row >= 2 {
                    let scrubbed = scrub_unparseable_tool_calls(messages)
                        + scrub_unparseable_tool_calls(conversation_history);
                    if scrubbed > 0 {
                        ctx.log.llm_error(&format!(
                            "scrubbed {scrubbed} unparseable tool call(s) from history after repeated parse failures — retrying"
                        ));
                        ui.status("Repaired a truncated tool call left in history — retrying.");
                        return LlmFlow::Retry;
                    }
                }
                if state.truncated_call_errors_in_a_row >= TRUNCATED_CALL_ABORT_AFTER {
                    ctx.log.llm_error(&format!(
                        "{} consecutive tool-call parse failures with nothing left to repair — aborting turn",
                        state.truncated_call_errors_in_a_row
                    ));
                    ui.error(
                        "The server keeps rejecting tool-call arguments — giving up on this turn.",
                    );
                    return LlmFlow::EndTurn { error: true };
                }
                // When the prompt is sitting near the context
                // window, the truncation is really context exhaustion
                // (the server clamps generation to the remaining room):
                // a hint can't fix that, compaction can.
                if context::compressor::estimated_context_tokens(messages, ctx.tool_def_tokens)
                    > ctx.config.model.context_window * 3 / 4
                    && state.context_compact_retries
                        < context::compressor::FORCE_COMPRESS_MAX_RETRIES
                {
                    state.context_compact_retries += 1;
                    // The two frontends kept different wording here —
                    // preserved per CompactionUx, never silently unified.
                    match opts.compaction {
                        CompactionUx::Interactive => {
                            ui.status("Context window exceeded — compacting and retrying.");
                            if ui
                                .pump(context::compressor::force_compress(
                                    messages,
                                    ctx.config,
                                    ctx.router,
                                    ctx.llm_worker,
                                    ctx.tool_def_tokens,
                                ))
                                .await
                            {
                                ctx.log.llm_error(
                                    "context window exceeded — compacted history, retrying",
                                );
                                return LlmFlow::Retry;
                            }
                            ui.error("Compaction could not free any context — stopping this turn.");
                            return LlmFlow::EndTurn { error: false };
                        }
                        CompactionUx::Batch => {
                            if ui
                                .pump(context::compressor::force_compress(
                                    messages,
                                    ctx.config,
                                    ctx.router,
                                    ctx.llm_worker,
                                    ctx.tool_def_tokens,
                                ))
                                .await
                            {
                                ctx.log.llm_error(
                                    "tool call truncated near context ceiling — compacted history, retrying",
                                );
                                ui.status(
                                    "Tool call truncated near context ceiling — compacting and retrying.",
                                );
                                return LlmFlow::Retry;
                            }
                            // Stuck — fall through to the hint path.
                        }
                    }
                }
                // Push a user-role hint and let the agent retry with a
                // smaller operation.
                ctx.log.llm_error(
                    "tool call JSON truncated (max_tokens) — injecting hint and continuing",
                );
                ui.status("Previous tool call truncated — retrying with guidance.");
                let hint = Message::user(truncated_tool_call_hint(ctx.config.tools.edit_mode));
                messages.push(hint.clone());
                conversation_history.push(hint);
                return LlmFlow::Retry;
            }
            let clean = if err_str.contains('<') {
                err_str
                    .split('<')
                    .next()
                    .unwrap_or(&err_str)
                    .trim()
                    .to_string()
            } else {
                err_str
            };
            ctx.log.llm_error(&clean);
            ui.error(&format!("LLM error: {clean}"));
            ui.event(UiEvent::LlmErrorEndpointHint {
                endpoint: ctx.config.model.endpoint.clone(),
            });
            return LlmFlow::EndTurn { error: true };
        }
    };

    // A 200 response can still be a context-exhaustion casualty: llama.cpp
    // silently stops generation the instant prompt+completion reaches
    // n_ctx (finish_reason="length", completion well under the requested
    // cap — see is_context_truncated_response). The partial output is
    // unusable (often a half-finished thought or clipped tool call), so
    // discard it, compact, and regenerate. Gated on the prompt actually
    // sitting near the window, which the "legitimately hit max_tokens"
    // case can never satisfy.
    let effective_max_tokens =
        max_tokens_override.unwrap_or(ctx.config.model.max_output_tokens as u64) as usize;
    if is_context_truncated_response(&response, effective_max_tokens)
        && context::compressor::estimated_context_tokens(messages, ctx.tool_def_tokens)
            > ctx.config.model.context_window * 3 / 4
        && state.context_compact_retries < context::compressor::FORCE_COMPRESS_MAX_RETRIES
    {
        state.context_compact_retries += 1;
        if opts.compaction == CompactionUx::Interactive {
            ui.status("Generation truncated by context ceiling — compacting and regenerating.");
        }
        if ui
            .pump(context::compressor::force_compress(
                messages,
                ctx.config,
                ctx.router,
                ctx.llm_worker,
                ctx.tool_def_tokens,
            ))
            .await
        {
            ctx.log.llm_error(
                "generation truncated by context ceiling — compacted history, regenerating",
            );
            if opts.compaction == CompactionUx::Batch {
                ui.status("Generation truncated by context ceiling — compacting and regenerating.");
            }
            return LlmFlow::Retry;
        }
    }

    // Get the assistant's response
    let choice = match response.choices.first() {
        Some(c) => c,
        None => {
            ui.event(UiEvent::EmptyLlmResponse);
            return LlmFlow::EndTurn { error: false };
        }
    };
    // A response made it through whole — any prior reactive-compaction
    // retries resolved this request; reset the budget for the next one.
    state.context_compact_retries = 0;
    state.truncated_call_errors_in_a_row = 0;

    // Never let an unparseable tool call into history: the server's
    // chat template re-parses every persisted call on every later
    // request and fails the whole request when one is broken. Replace
    // the cut-off arguments with a small valid stub; the tool loop
    // below answers the stub with a "not executed, re-issue smaller"
    // tool_result so the call/result pairing stays intact.
    let mut assistant_msg = choice.message.clone();
    let truncated_calls = sanitize_truncated_tool_calls(&mut assistant_msg);
    if truncated_calls > 0 {
        ctx.log.llm_error(&format!(
            "{truncated_calls} tool call(s) arrived with unparseable arguments (cut off by the output limit) — stubbed before persisting"
        ));
        ui.status("A tool call was cut off by the output limit — it will not be executed.");
    }

    ui.finish_assistant_text(assistant_msg.content.as_deref());

    // Log and add assistant message to history
    if let Some(content) = &assistant_msg.content {
        ctx.log.llm_response(content);
    }
    if assistant_msg.is_meaningful() {
        conversation_history.push(assistant_msg.clone());
    }

    LlmFlow::Ready(assistant_msg)
}
