//! Agent execution — the main LLM loop.
//!
//! Implements the core agent loop:
//! 1. Assemble context for the turn
//! 2. Call the LLM with tools
//! 3. Parse tool calls from the response
//! 4. Execute tools
//! 5. Apply observation masking (compress old tool results)
//! 6. Feed results back and repeat

use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::Mutex;

use anyhow::Result;

use crate::cli::commands::agent::debugger;
use crate::cli::commands::agent::display::summarize_args;
use crate::cli::commands::agent::hints::{
    PERIOD2_LOOP_HINT, PLAN_CHECKPOINT_AFTER_EDITS, PLAN_CHECKPOINT_WARNING, PLAN_PROGRESS_NUDGE,
    PREMATURE_EXIT_NUDGE, REPEATED_READ_NUDGE, is_file_write, is_prunable_refactor_failure,
    loop_detected_hint, truncated_tool_call_hint, visible_tool_defs,
};
use crate::cli::commands::agent::loop_detector::{
    is_mutating_call, is_period2_cycle, key_is_mutating, loop_call_key,
};
use crate::cli::commands::agent::spiral;
use crate::cli::commands::agent::validation;
use crate::config::{Config, EditMode, ModelRole};
use crate::context;
use crate::llm::{
    ChatRequest, Message, ModelRouter, is_context_exceeded_error, is_context_truncated_response,
    is_truncated_tool_call_error,
};
use crate::logging::SessionLog;
use crate::lsp::LspClient;
use crate::mcp::{McpConfig, McpRegistry};
use crate::runtime::{
    LlmWorkerEvent, LlmWorkerHandle, ShellControl, ShellWorkerEvent, ToolWorkerPool,
};
use crate::tools;
use crate::tools::permissions::{Action, PermissionManager};
use crate::tui;

/// Project-relative paths touched by a unified-diff patch file (parsed from
/// its `+++ b/<path>` headers). Used by replay mode to notify the LSP of files
/// changed out-of-band by `--replay-apply`. Best-effort: returns `[]` on read error.
fn changed_paths_in_patch(patch: &std::path::Path) -> Vec<String> {
    let Ok(text) = std::fs::read_to_string(patch) else {
        return Vec::new();
    };
    text.lines()
        .filter_map(|l| l.strip_prefix("+++ b/"))
        .map(|p| p.trim().to_string())
        .filter(|p| p != "/dev/null")
        .collect()
}

/// Stable signature of a gate failure, used by `debugger_multifire` to decide
/// whether the failure CHANGED since the last diagnosis (so the debugger walks
/// compile→smoke rather than re-diagnosing the same failure). Using only the
/// first line breaks this for commands that wrap every failure in a constant
/// banner (e.g. `echo "DOES NOT COMPILE:"; echo "$out" | tail -20`) — every
/// compile error then hashes to the identical string, so multifire silently
/// refuses to refire when the underlying error actually changed. Join enough
/// of the (non-empty) output to reach past such banners into the real detail.
pub(crate) fn failure_key(output: &str) -> String {
    let joined = output
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    joined.to_ascii_lowercase().chars().take(400).collect()
}

#[cfg(test)]
mod failure_key_tests {
    use super::failure_key;

    #[test]
    fn distinguishes_different_errors_behind_the_same_banner() {
        let a = "DOES NOT COMPILE:\nerror[E0599]: no method named `call_chat`\n --> src/cli/commands/run.rs:222:41";
        let b = "DOES NOT COMPILE:\nerror[E0382]: borrow of partially moved value: `system_prompt_override`\n --> src/context/mod.rs:306:8";
        assert_ne!(failure_key(a), failure_key(b));
    }

    #[test]
    fn treats_the_same_error_as_the_same_key() {
        let a = "DOES NOT COMPILE:\nerror[E0382]: borrow of partially moved value: `x`\n --> src/context/mod.rs:306:8";
        let b = "DOES NOT COMPILE:\nerror[E0382]: borrow of partially moved value: `x`\n --> src/context/mod.rs:306:8";
        assert_eq!(failure_key(a), failure_key(b));
    }

    #[test]
    fn ignores_blank_lines_and_case() {
        let a = "\n\nDOES NOT COMPILE:\n\nError[E0308]\n";
        let b = "DOES NOT COMPILE:\nerror[e0308]";
        assert_eq!(failure_key(a), failure_key(b));
    }
}

/// Track consecutive `plan(action='check')` failures on the SAME step
/// (`tools.plan_gate_debugger`'s trigger). A failure on a step other than the
/// last-failed one (or the first ever) resets the streak to 1 — only
/// *repeated* blocking on one step signals a stall worth escalating.
pub(crate) fn track_plan_step_failure(
    last_failed_step: &mut Option<u64>,
    streak: &mut u32,
    step: u64,
) {
    if *last_failed_step == Some(step) {
        *streak += 1;
    } else {
        *last_failed_step = Some(step);
        *streak = 1;
    }
}

#[cfg(test)]
mod track_plan_step_failure_tests {
    use super::track_plan_step_failure;

    #[test]
    fn same_step_repeated_increments_streak() {
        let mut last = None;
        let mut streak = 0;
        track_plan_step_failure(&mut last, &mut streak, 1);
        assert_eq!(streak, 1);
        track_plan_step_failure(&mut last, &mut streak, 1);
        assert_eq!(streak, 2);
        track_plan_step_failure(&mut last, &mut streak, 1);
        assert_eq!(streak, 3);
        assert_eq!(last, Some(1));
    }

    #[test]
    fn different_step_resets_streak_to_one() {
        let mut last = None;
        let mut streak = 0;
        track_plan_step_failure(&mut last, &mut streak, 1);
        track_plan_step_failure(&mut last, &mut streak, 1);
        assert_eq!(streak, 2);
        track_plan_step_failure(&mut last, &mut streak, 2);
        assert_eq!(streak, 1);
        assert_eq!(last, Some(2));
    }

    #[test]
    fn first_failure_ever_sets_streak_to_one() {
        let mut last = None;
        let mut streak = 0;
        track_plan_step_failure(&mut last, &mut streak, 7);
        assert_eq!(streak, 1);
        assert_eq!(last, Some(7));
    }
}

/// Execute the debugger's proposed single-file rewind (`tools.
/// Persist the behavioral gate's raw failure output (stdout+stderr from the
/// configured validation command) to a file the model can `read` directly.
/// The reactive debugger's Report/Rewind verdicts summarize this output in
/// their own words before handing it to the primary agent — a paraphrase
/// that can lose precision the raw text had (e.g. the exact "GOT: <value>"
/// line pointing at which callsite is actually broken). Writing the raw
/// text alongside the summary, rather than instead of it, lets the model
/// fall back to ground truth when the summary steers it wrong. Best-effort:
/// a write failure just means no pointer gets appended, never a hard error.
pub(crate) fn write_gate_failure_output(config: &Config, output: &str) -> Option<String> {
    let rel = "last_gate_failure.txt";
    std::fs::write(config.miniswe_path(rel), output).ok()?;
    Some(format!(".miniswe/{rel}"))
}

/// debugger_judge_rewind`) and build the message to inject afterward.
/// Best-effort, matching the codebase's other auto-recovery guards: on
/// failure the tree is left as-is and the model just sees the original
/// verification failure, no different from a plain Report verdict.
#[allow(clippy::too_many_arguments)]
async fn rewind_message(
    candidate: &tools::RewindCandidate,
    config: &Config,
    perms: &Arc<PermissionManager>,
    lsp_client: &Option<Arc<LspClient>>,
    fast_revisions: &Option<Arc<tools::RevisionStore>>,
    fast_baseline_errors: usize,
    output: &str,
) -> Message {
    let Some(revisions) = fast_revisions.as_deref() else {
        return Message::user(&format!(
            "[Verification failed — do NOT finish yet. Check output:\n{output}]"
        ));
    };
    let args = serde_json::json!({"path": candidate.path, "rev": candidate.rev});
    let ok = tools::execute_fast_tool(
        "revert",
        &args,
        config,
        perms.as_ref(),
        lsp_client.as_deref(),
        revisions,
        fast_baseline_errors,
    )
    .await
    .is_ok_and(|r| r.success);

    if ok {
        tui::print_status(&format!(
            "[debugger-judge] REWIND — reverted {} to rev_{} (file_errors {} → {})",
            candidate.path, candidate.rev, candidate.file_errors_now, candidate.file_errors_then
        ));
        // REWIND fixes the ONE regressed file the debugger flagged — it doesn't
        // mean the gate's original failure is fully resolved (the behavioral
        // check may still be failing for an unrelated reason elsewhere). Point
        // at the raw check output so the model isn't left guessing what
        // "the remaining problem" actually is from the rewind summary alone.
        let output_note = write_gate_failure_output(config, output)
            .map(|path| format!(" Full check output that triggered this: read(\"{path}\")."))
            .unwrap_or_default();
        Message::user(&format!(
            "[A read-only debugger with fresh eyes found that {} had regressed from a much \
             cleaner earlier revision. The loop has ALREADY reverted it to rev_{} for you \
             (file_errors {} → {}) — do NOT redo the discarded edits the same way. Re-read the \
             file to see its current (reverted) content, then continue the plan, fixing the \
             remaining problem differently. Everything outside this file is untouched.{output_note}]",
            candidate.path, candidate.rev, candidate.file_errors_now, candidate.file_errors_then
        ))
    } else {
        tui::print_status(&format!(
            "[debugger-judge] REWIND — revert of {} to rev_{} failed; continuing without it",
            candidate.path, candidate.rev
        ));
        Message::user(&format!(
            "[Verification failed — do NOT finish yet. Check output:\n{output}]"
        ))
    }
}

/// Run the agent for a single message.
pub async fn run(
    mut config: Config,
    message: &str,
    plan_only: bool,
    headless: bool,
    continue_session: bool,
    replay_context: Option<PathBuf>,
    replay_apply: Option<PathBuf>,
) -> Result<()> {
    let log = Arc::new(SessionLog::new(&config));
    log.user_message(message);

    let router = Arc::new(ModelRouter::new(&config));
    // Probe the server for the actual model identity before building the
    // tool list — model-family checks need the server-reported name, not
    // the user's config alias. Probe failure leaves probed_model = None
    // and we fall back to the config string.
    config.model.probed_model = router.probe_default_model().await.ok();
    let llm_worker = LlmWorkerHandle::new(router.clone(), config.runtime.llm_concurrency);
    let perms = Arc::new(if headless {
        PermissionManager::headless(&config)
    } else {
        PermissionManager::new(&config)
    });
    let tool_pool = ToolWorkerPool::new(config.runtime.tool_worker_pool_size);
    let mut tool_defs = tools::tool_definitions(config.tools.edit_mode);
    // Filter tools based on config
    {
        let mut disabled = Vec::new();
        if !config.tools.web_tools {
            disabled.push("web");
        }
        if !config.tools.plan {
            disabled.push("plan");
        }
        // Uniform across all models: refactor available, edit_file hidden.
        // The old Devstral carve-out (hide refactor, keep edit_file) was
        // protecting against `position`-arg mangling from the *old*
        // `change_signature` tool; the rename to `refactor` fixed the
        // formatting (replay: clean args when called), and the gate's
        // real effect was just suppressing adoption. edit_file stays
        // hidden because it monopolizes tool choice (Gemma Apr 30 fast
        // 6/6 at 291s vs May 11 6/6 at 2195s with edit_file visible).
        // Adoption is driven by the phase-aware system prompt (see
        // context::build_system_prompt's plan_set branch), not the gate.
        disabled.push("edit_file");
        // In Fast mode we ALSO expose `edit_file` alongside the
        // primitives. Body edits with tricky brace nesting (e.g.
        // wrapping an existing block in `if let Some(x) = ... {} else
        // {}`) are an attention-quality problem for small models —
        // probe in /tmp/gemma-edit-probe.py shows Gemma writes them
        // first-try with focused context but takes 10+ revisions in the
        // full agent context. `edit_file` runs an inner focused LLM
        // call which avoids the dilution. Primitives stay available
        // for surgical line-precise edits.
        // tools.flat: swap grouped `refactor{action,position,...}` for
        // flat single-purpose refactor tools (no DSL footgun).
        if config.tools.flat {
            disabled.push("refactor");
        }
        tool_defs.retain(|t| !disabled.contains(&t.function.name.as_str()));
        if config.tools.edit_mode == EditMode::Fast {
            tool_defs.extend(tools::fast_mode_tool_definitions());
        }
        if config.tools.flat {
            tool_defs.extend(tools::definitions::flat_refactor_tool_definitions());
        }
        // spawn_agents only buys anything when the worker pool can run LLM
        // calls concurrently. At llm_concurrency=1 (the default) it's an inert
        // per-round schema tax — and the most complex tool shape — that the
        // model never benefits from. Hide it unless parallelism is available.
        if config.runtime.llm_concurrency > 1 {
            tool_defs.push(tools::definitions::spawn_agents_tool_definition());
        }
    }

    // Clear stale scratchpad/plan from previous sessions — unless this
    // is a `--continue` invocation, in which case the model is meant to
    // pick up where the previous session left off.
    if !continue_session {
        let _ = std::fs::remove_file(config.miniswe_path("scratchpad.md"));
        let _ = std::fs::remove_file(config.miniswe_path("plan.md"));
    }

    tui::print_header(if plan_only {
        "Plan Mode (read-only)"
    } else {
        "miniswe"
    });

    // Ask the server what's actually running, so startup reflects reality
    // rather than what config.toml claims. Done once per session.
    for line in router.startup_summary().await {
        tui::print_status(&line);
    }

    // Select model role: plan mode uses the plan model, normal mode uses default
    let model_role = if plan_only {
        ModelRole::Plan
    } else {
        ModelRole::Default
    };

    // Spawn LSP client (non-blocking — initializes in background)
    let lsp_client: Option<Arc<LspClient>> = if config.lsp.enabled {
        match LspClient::spawn(config.project_root.clone()).await {
            Ok(client) => {
                tui::print_status("LSP: starting...");
                Some(Arc::new(client))
            }
            Err(e) => {
                tui::print_status(&format!("LSP: not available ({e})"));
                None
            }
        }
    } else {
        None
    };

    // Initialize MCP servers
    let mcp_config = McpConfig::load(&config.project_root)?;
    let mcp_registry = if mcp_config.has_servers() {
        let cache_dir = config.miniswe_path("mcp");
        match McpRegistry::connect(&mcp_config, &cache_dir) {
            Ok(registry) => {
                if registry.has_servers() {
                    tui::print_status(&format!(
                        "MCP: {} servers, {} tools",
                        registry.servers.len(),
                        registry.tool_count()
                    ));
                    // Add mcp_use tool definition
                    tool_defs.push(tools::definitions::mcp_tool_definition());
                }
                Some(Arc::new(Mutex::new(registry)))
            }
            Err(e) => {
                tui::print_status(&format!("MCP: failed to connect ({e})"));
                None
            }
        }
    } else {
        None
    };

    let mcp_summary = mcp_registry
        .as_ref()
        .and_then(|r| r.lock().context_summary());

    // Estimate tool definition overhead for context budgeting
    let tool_def_tokens =
        context::estimate_tokens(&serde_json::to_string(&tool_defs).unwrap_or_default());

    let max_rounds = config.context.max_rounds;
    let pause_at = config.context.pause_after_rounds;
    // Ceremony=Off (default, evidence-distilled): no plan gate, no
    // plan/no-plan nudges, all edit tools always visible, no phase
    // rebuild. `strict` re-enables the legacy plan-first machinery.
    // See docs/tiered-agent-design.md.
    let strict = config.tools.ceremony == crate::config::CeremonyMode::Strict;

    let mut conversation_history: Vec<Message> = Vec::new();
    let mut round = 0;
    let mut had_error = false;
    let mut user_continued = false;

    // Track consecutive identical tool calls for loop detection
    let mut last_call_key: Option<String> = None;
    let mut same_call_streak = 0u32;
    // Short rolling history of call keys for period-2 cycle detection
    // (edit↔revert oscillation — invisible to the consecutive detector,
    // which resets its streak on every alternation).
    let mut recent_call_keys: Vec<String> = Vec::new();
    // Number of distinct loops the model has been pulled out of in this
    // turn. We give one recovery; a second loop ends the turn for real.
    let mut loop_recoveries = 0u32;
    let mut calls_since_last_edit = 0u32;
    let mut successful_edits_since_plan_update = 0u32;
    let mut plan_update_requested = false;
    let mut nudged_premature_exit = false;
    let mut nudged_no_plan = false;
    // Consecutive reactive-compaction retries (context exhaustion signaled
    // by the server — see compressor::force_compress). Reset whenever a
    // response is successfully consumed, so this bounds futile retries of
    // one failing request, not total compactions over a long run.
    let mut context_compact_retries: usize = 0;
    // How many times the behavioral done-gate has blocked completion this turn.
    let mut validation_blocks: usize = 0;
    // The model's stated rationale each time the gate blocked it — so a model
    // that believes the check is wrong has an auditable voice (bounded by
    // max_retries; never a silent free pass).
    let mut validation_disputes: Vec<String> = Vec::new();
    // The reactive debugger sub-agent fires at most once per turn.
    let mut replan_fired = false;
    let mut restart_fired = false;
    let mut debugger_fires = 0usize;
    // Signature of the last failure the debugger was handed. With
    // `debugger_multifire`, the debugger re-fires only when this CHANGES — so it
    // walks compile→smoke one diagnosis per distinct failure, never re-diagnosing
    // the same one (the blunt fire-≤N× variant regressed by doing exactly that).
    let mut last_debugged_failure: Option<String> = None;
    // `tools.plan_gate_debugger`: consecutive plan(check) failures on the SAME
    // step. Distinct from validation_blocks (the behavioral done-gate) — this
    // is the plan tool's OWN compile gate repeatedly blocking one step.
    let mut same_plan_step_failures: u32 = 0;
    let mut last_failed_plan_step: Option<u64> = None;
    // Gate-triggered context resets fired this turn (bounded — don't loop).
    let mut gate_resets: usize = 0;
    // Spiral-reset: per-file revert counts this turn + how many resets fired,
    // to detect a revert-loop (agent cycling on the same failing edits).
    let mut revert_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    let mut spiral_resets: usize = 0;

    // Ctrl+C cancellation flag. The handler fires once and exits — no
    // loop, because `ctrl_c().await` resolves immediately after the
    // first signal and would otherwise busy-spin at 100% CPU.
    let cancelled = Arc::new(AtomicBool::new(false));
    let cancelled_for_handler = cancelled.clone();
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        cancelled_for_handler.store(true, Ordering::Relaxed);
        eprintln!("\n\x1b[33m(interrupted — finishing current step)\x1b[0m");
    });

    // Initialize snapshot manager for revert support
    let snapshots = tools::snapshots::SnapshotManager::init(&config.project_root)
        .ok()
        .map(|s| Arc::new(Mutex::new(s)));

    // Fast-mode state: per-file revision store (in-memory, session-scoped)
    // + the project-wide LSP error count captured at session start. The
    // baseline lets each edit's feedback line report `(+N from baseline)`
    // so regressions jump out.
    let fast_revisions: Option<Arc<tools::RevisionStore>> =
        if config.tools.edit_mode == EditMode::Fast {
            let miniswe_dir = config.miniswe_path("revisions");
            match tools::RevisionStore::new(&miniswe_dir) {
                Ok(s) => Some(Arc::new(s)),
                Err(e) => {
                    tui::print_status(&format!("fast mode: revision store init failed ({e})"));
                    None
                }
            }
        } else {
            None
        };
    let fast_baseline_errors: usize = if config.tools.edit_mode == EditMode::Fast {
        tools::fast::project_error_count(lsp_client.as_deref()).await
    } else {
        0
    };

    // Initial context assembly
    let assembled = context::assemble(
        &config,
        message,
        &conversation_history,
        plan_only,
        mcp_summary.as_deref(),
    );
    log.context_assembled(assembled.token_estimate, assembled.messages.len());
    tui::print_status(&format!(
        "Context: ~{} tokens assembled",
        assembled.token_estimate
    ));

    let mut messages = assembled.messages;
    // Replay mode: replace the freshly-assembled context with a captured one
    // (faithful "context we had then"). The fixture's messages already end with
    // the gate rejection the agent must respond to, so the loop's first LLM call
    // resumes exactly where the original run was before its first fix.
    // See docs/replay-mode-design.md. (Run with gate_context_reset=false so a
    // mid-loop reset can't clobber the seeded context.)
    let replay_mode = replay_context.is_some();
    if let Some(ref path) = replay_context {
        let raw = std::fs::read_to_string(path)?;
        let v: serde_json::Value = serde_json::from_str(&raw)?;
        let captured: Vec<Message> = serde_json::from_value(v["messages"].clone())?;
        if captured.is_empty() {
            anyhow::bail!("replay context {} has no messages", path.display());
        }
        tui::print_status(&format!(
            "Replay: seeded {} captured messages from {}",
            captured.len(),
            path.display()
        ));
        messages = captured;
    }
    // Replay helper: apply the captured prior-edits patch to the working tree
    // now — AFTER snapshot init (so round 0 stays the clean baseline) and after
    // baseline-error capture (= 0 on the clean tree), so `revert_to_green` can
    // restore the clean state the resumed agent never reached on its own. The
    // agent still resumes ON the broken tree.
    if let Some(ref patch) = replay_apply {
        let abs = std::fs::canonicalize(patch).unwrap_or_else(|_| patch.clone());
        let status = std::process::Command::new("git")
            .arg("apply")
            .arg(&abs)
            .current_dir(&config.project_root)
            .status();
        match status {
            Ok(s) if s.success() => {
                tui::print_status(&format!(
                    "Replay: applied working-tree patch {}",
                    patch.display()
                ));
                // Tell the LSP the patched files changed so the revert-to-green
                // green-check reads the resumed broken tree, not a stale clean
                // snapshot. Best-effort; rust-analyzer file-watching also catches it.
                if let Some(ref lsp) = lsp_client {
                    for rel in changed_paths_in_patch(&abs) {
                        let _ = lsp.notify_file_changed(&config.project_root.join(rel));
                    }
                }
            }
            _ => anyhow::bail!("replay: failed to apply patch {}", patch.display()),
        }
    }
    // The system prompt is phase-aware (pre-plan "explore→plan" vs
    // post-plan "you are EDITING" + routing), but no longer carries plan or
    // scratchpad content — that's now attached to the tail of the message
    // list every round, inside `context::compressor::maybe_compress`'s
    // `refresh_current_state` (see its doc comment for why it lives there).

    // Nudge the model to plan before editing (strict/legacy only). Skipped in
    // replay mode — the captured context already reflects whatever planning the
    // original run did; injecting a fresh nudge would corrupt the resume.
    if strict && config.tools.plan && !replay_mode {
        messages.push(Message::user(
            "[Before making changes, explore the codebase and use the plan tool to outline your approach. \
             Each step has compile: true (default) — the compiler must pass to check it off. \
             Set compile: false with a reason only if a step intentionally breaks the tree (e.g. renaming a function before updating callers). \
             If a step proves too complex, use action='refine' to split it into substeps. \
             Check off steps as you complete them.]"
        ));
    }

    // revert-to-green state (opt-in `tools.revert_to_green`): the last round
    // whose start-of-round snapshot was green (project errors ≤ baseline) and
    // how many consecutive rounds the project has stayed broken. A stuck agent
    // that keeps the tree red for REVERT_TO_GREEN_BLOCKS rounds gets the whole
    // tree reset to that last green snapshot (see below).
    const REVERT_TO_GREEN_BLOCKS: usize = 6;
    let mut last_green_round: usize = 0;
    let mut red_streak: usize = 0;

    'round: loop {
        if had_error {
            break;
        }
        round += 1;
        log.round_start(round);

        // Snapshot at start of each round for revert support
        if let Some(ref snap) = snapshots {
            let mut guard = snap.lock();
            let _ = guard.begin_round(round);
        }

        // revert-to-green: this round STARTS from the state the previous round
        // left (just snapshotted above). If the project has been broken above
        // baseline for REVERT_TO_GREEN_BLOCKS rounds, the agent is digging
        // deeper, not recovering — reset the whole tree to the last green
        // snapshot and tell it to start over from a clean base.
        if config.tools.revert_to_green
            && config.tools.edit_mode == EditMode::Fast
            && let Some(ref snap) = snapshots
        {
            {
                let errs = tools::fast::project_error_count(lsp_client.as_deref()).await;
                if errs <= fast_baseline_errors {
                    last_green_round = round;
                    red_streak = 0;
                } else {
                    red_streak += 1;
                    if red_streak >= REVERT_TO_GREEN_BLOCKS {
                        let result = {
                            let guard = snap.lock();
                            guard.revert_to_round(last_green_round)
                        };
                        match result {
                            Ok(m) => {
                                tui::print_status(&format!(
                                    "[revert-to-green] stuck {red_streak} rounds; {m}"
                                ));
                                messages.push(Message::user(&format!(
                                    "[auto-revert-to-green] The project has had compile errors for \
                                     {red_streak} rounds straight and you are not converging — you are \
                                     digging deeper, not recovering. I reverted the ENTIRE working tree \
                                     to round {last_green_round}, the last state that compiled cleanly. \
                                     Your edits since then are GONE; do not replay them. Start over from \
                                     this clean base: re-read the relevant code, make ONE small complete \
                                     change, and run a check before continuing."
                                )));
                                red_streak = 0;
                            }
                            Err(e) => {
                                tui::print_status(&format!("[revert-to-green] revert failed: {e}"));
                            }
                        }
                    }
                }
            }
        }
        if round > max_rounds {
            tui::print_error("Maximum tool rounds reached. Stopping.");
            break;
        }

        // Ask user if they want to continue after pause_after_rounds.
        // Headless runs have nobody to answer — blocking on stdin hung real
        // e2e harness runs for their full timeout (uds-mcp 2026-07-13: two
        // of three attempts died waiting at exactly this prompt). Continue
        // with a logged notice instead; max_rounds stays the hard stop.
        if round == pause_at && !user_continued {
            if headless {
                tui::print_status(&format!(
                    "{pause_at} tool rounds used — headless, continuing without prompt."
                ));
                user_continued = true;
            } else {
                tui::print_status(&format!("{pause_at} tool rounds used."));
                let response = tui::read_input("Continue? [y]es / [n]o:");
                match response.as_deref() {
                    Some("y") | Some("yes") | Some("") => {
                        user_continued = true;
                    }
                    _ => {
                        // Tell the LLM to wrap up
                        messages.push(Message::user("[Stop now. Summarize what you've done.]"));
                    }
                }
            }
        }

        // Warn the LLM when approaching the hard limit
        if round == max_rounds.saturating_sub(5) {
            messages.push(Message::user(
                "[Approaching tool limit. Wrap up and summarize.]",
            ));
        }

        // Unified context compression — handles both tool results and conversation
        let pre_mask = messages.len();
        context::compressor::maybe_compress(
            &mut messages,
            &config,
            &router,
            &llm_worker,
            tool_def_tokens,
            &mut plan_update_requested,
        )
        .await;
        log.masking_applied(pre_mask.saturating_sub(messages.len()), pre_mask);

        // Sanitize message roles before sending (strict chat template compat)
        context::sanitize_messages(&mut messages);

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
        let plan_set = tools::plan::plan_exists(&config);
        // Off: never hide edit tools (pass plan_exists=true). Strict:
        // legacy hide-until-plan behavior.
        let visible = visible_tool_defs(&tool_defs, plan_set || !strict);
        // Mistral Small 4 honors `reasoning_effort` ("none"/"high"); other
        // models (Gemma, GPT-OSS, Devstral) use `enable_thinking` or
        // ignore the kwarg entirely. For Mistral 4 we want deep reasoning
        // during the planning phase (decomposing the task — exactly where
        // it goes wrong, picking the wrong file family) and fast execution
        // once a plan is set. Per-model gating keeps the cost localized.
        let chat_template_kwargs = if config.model.is_mistral_small_4_family() {
            let effort = if plan_set { "none" } else { "high" };
            serde_json::json!({"reasoning_effort": effort})
        } else {
            serde_json::json!({"enable_thinking": false})
        };
        // Mistral Small 4 with reasoning_effort=high needs significant
        // output budget. Probe data: at 8192 max_tokens the model hits
        // finish_reason=length after ~32K chars of reasoning_content with
        // ZERO chars of content emitted. At 16384 it reasons for ~24K
        // chars and emits a clean ~2K-char correct plan (finish_reason=stop,
        // ~6K tokens used). Per llama.cpp #20668 and vLLM #37081 — known
        // Mistral 4 budget-hungry reasoning behavior.
        let max_tokens_override = if config.model.is_mistral_small_4_family() {
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
        };
        log.llm_request(&request);

        tui::print_separator();

        // Reset cancel flag for this round
        cancelled.store(false, Ordering::Relaxed);

        eprint!("\x1b[2m⠋ thinking...\x1b[0m");
        std::io::stderr().flush().ok();
        let thinking = Arc::new(AtomicBool::new(true));

        let mut llm_events = llm_worker.submit(model_role, request.clone(), cancelled.clone());
        let response = match loop {
            match llm_events.recv().await {
                Some(LlmWorkerEvent::Token(token)) => {
                    if thinking.load(Ordering::Relaxed) {
                        thinking.store(false, Ordering::Relaxed);
                        eprint!("\r\x1b[2K");
                        std::io::stderr().flush().ok();
                    }
                    tui::print_token(&token);
                }
                Some(LlmWorkerEvent::Completed(Ok(r))) => break Ok(r),
                Some(LlmWorkerEvent::Completed(Err(e))) => break Err(e),
                None => break Err("LLM worker stopped unexpectedly".to_string()),
            }
        } {
            Ok(r) => {
                if thinking.load(Ordering::Relaxed) {
                    eprint!("\r\x1b[2K");
                    std::io::stderr().flush().ok();
                }
                r
            }
            Err(err_str) => {
                eprint!("\r\x1b[2K");
                std::io::stderr().flush().ok();
                if err_str.contains("Interrupted") {
                    tui::print_status("Generation interrupted.");
                    break;
                }
                // The server rejected the request outright: prompt alone
                // exceeds the context window. Compact and resend — this is
                // the primary recovery path for compaction="lazy" (which
                // never compacts proactively), and a safety net for every
                // other strategy.
                if is_context_exceeded_error(&err_str)
                    && context_compact_retries < context::compressor::FORCE_COMPRESS_MAX_RETRIES
                {
                    context_compact_retries += 1;
                    if context::compressor::force_compress(
                        &mut messages,
                        &config,
                        &router,
                        &llm_worker,
                        tool_def_tokens,
                    )
                    .await
                    {
                        log.llm_error("context window exceeded — compacted history, retrying");
                        tui::print_status("Context window exceeded — compacting and retrying.");
                        continue;
                    }
                    // Nothing could be freed — fall through to the normal
                    // error handling; retrying would fail identically.
                }
                if is_truncated_tool_call_error(&err_str) {
                    // Model hit max_tokens mid tool-call — the server
                    // dropped the assistant turn, no tool_call_id was
                    // issued. When the prompt is sitting near the context
                    // window, the truncation is really context exhaustion
                    // (the server clamps generation to the remaining room):
                    // a hint can't fix that, compaction can.
                    if context::compressor::estimated_context_tokens(&messages, tool_def_tokens)
                        > config.model.context_window * 3 / 4
                        && context_compact_retries < context::compressor::FORCE_COMPRESS_MAX_RETRIES
                    {
                        context_compact_retries += 1;
                        if context::compressor::force_compress(
                            &mut messages,
                            &config,
                            &router,
                            &llm_worker,
                            tool_def_tokens,
                        )
                        .await
                        {
                            log.llm_error(
                                "tool call truncated near context ceiling — compacted history, retrying",
                            );
                            tui::print_status(
                                "Tool call truncated near context ceiling — compacting and retrying.",
                            );
                            continue;
                        }
                    }
                    // Push a user-role hint and let the agent retry with a
                    // smaller operation.
                    log.llm_error(
                        "tool call JSON truncated (max_tokens) — injecting hint and continuing",
                    );
                    tui::print_status("Previous tool call truncated — retrying with guidance.");
                    let hint = Message::user(truncated_tool_call_hint(config.tools.edit_mode));
                    messages.push(hint.clone());
                    conversation_history.push(hint);
                    continue;
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
                log.llm_error(&clean);
                tui::print_error(&format!("LLM error: {clean}"));
                tui::print_status(&format!(
                    "Check that your LLM server is running at {}",
                    config.model.endpoint
                ));
                had_error = true;
                break;
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
            max_tokens_override.unwrap_or(config.model.max_output_tokens as u64) as usize;
        if is_context_truncated_response(&response, effective_max_tokens)
            && context::compressor::estimated_context_tokens(&messages, tool_def_tokens)
                > config.model.context_window * 3 / 4
            && context_compact_retries < context::compressor::FORCE_COMPRESS_MAX_RETRIES
        {
            context_compact_retries += 1;
            if context::compressor::force_compress(
                &mut messages,
                &config,
                &router,
                &llm_worker,
                tool_def_tokens,
            )
            .await
            {
                log.llm_error(
                    "generation truncated by context ceiling — compacted history, regenerating",
                );
                tui::print_status(
                    "Generation truncated by context ceiling — compacting and regenerating.",
                );
                continue;
            }
        }

        // Get the assistant's response
        let choice = match response.choices.first() {
            Some(c) => c,
            None => {
                tui::print_error("Empty response from LLM");
                break;
            }
        };
        // A response made it through whole — any prior reactive-compaction
        // retries resolved this request; reset the budget for the next one.
        context_compact_retries = 0;

        let assistant_msg = &choice.message;

        // Print newline after streaming content
        if assistant_msg.content.is_some() {
            println!();
        }

        // Log and add assistant message to history
        if let Some(content) = &assistant_msg.content {
            log.llm_response(content);
        }
        if assistant_msg.is_meaningful() {
            conversation_history.push(assistant_msg.clone());
        }

        // Check for tool calls
        let tool_calls = match &assistant_msg.tool_calls {
            Some(tc) if !tc.is_empty() => tc.clone(),
            _ => {
                // Two distinct "model returned nothing" situations:
                //  (1) plan exists, steps remain → standard mid-task exit
                //  (2) no plan set yet → model stopped during exploration
                //      before doing meaningful work. Mistral Small 4 with
                //      reasoning_effort=high triggered this — read a few
                //      files, reasoned heavily, then returned empty.
                // Both deserve one nudge to recover.
                if strict && !nudged_premature_exit && config.tools.plan {
                    let has_unchecked = tools::plan::has_unchecked_steps(&config);
                    let plan_exists = tools::plan::plan_exists(&config);
                    if has_unchecked || !plan_exists {
                        nudged_premature_exit = true;
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
                        continue;
                    }
                }
                // Behavioral done-gate: before accepting completion, verify the
                // change actually works at runtime. A configured check that
                // exits non-zero blocks the exit and feeds its output back so
                // the model can fix a plumbed-but-not-consumed change (the
                // change compiles + tests pass but the feature doesn't work).
                // Default config has no command → this is a no-op.
                // See docs/success-validation-design.md.
                if validation_blocks < config.validation.max_retries
                    && config.validation.command().is_some()
                {
                    match validation::run_behavioral_check(&config).await {
                        validation::CheckOutcome::Fail(output) => {
                            validation_blocks += 1;
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
                                    "[validation] blocked completion (attempt {validation_blocks}); model rationale: {}",
                                    crate::truncate_chars(rationale, 300)
                                );
                                validation_disputes.push(rationale.to_string());
                            }
                            tui::print_status("Behavioral check failed — not done yet.");

                            // Full restart (opt-in `tools.gate_restart`): on the
                            // FIRST gate block, ABANDON the (possibly poisoned)
                            // attempt — revert the WHOLE tree to the clean baseline
                            // (round 0) AND reset the context to a fresh from-scratch
                            // attempt at the task, clearing the degraded plan. Tests
                            // detect-and-restart: a stuck/off-path state is worse than
                            // a clean start (run2), so scrap it. Fires once per turn.
                            if config.tools.gate_restart && !restart_fired {
                                restart_fired = true;
                                if let Some(ref snap) = snapshots {
                                    let guard = snap.lock();
                                    match guard.revert_to_round(0) {
                                        Ok(m) => tui::print_status(&format!("[gate-restart] {m}")),
                                        Err(e) => tui::print_status(&format!(
                                            "[gate-restart] tree revert failed: {e}"
                                        )),
                                    }
                                }
                                // Whole-tree revert changed many files outside the
                                // per-edit reindex path — resync the symbol index /
                                // repo-map to the clean baseline so the fresh agent
                                // doesn't see the reverted-away structure.
                                tools::reindex_project_incremental(&config);
                                let _ = std::fs::remove_file(config.miniswe_path("plan.md"));
                                let _ = std::fs::remove_file(config.miniswe_path("scratchpad.md"));
                                let assembled = context::assemble(
                                    &config,
                                    message,
                                    &[],
                                    plan_only,
                                    mcp_summary.as_deref(),
                                );
                                messages = assembled.messages;
                                conversation_history.clear();
                                validation_blocks = 0;
                                same_plan_step_failures = 0;
                                last_failed_plan_step = None;
                                tui::print_status(
                                    "[gate-restart] scrapped the stuck state — tree at clean baseline + fresh context; restarting from scratch.",
                                );
                                continue;
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
                            if config.tools.gate_replan && !replan_fired && !is_compile_fail {
                                replan_fired = true;
                                tui::print_status(
                                    "Re-anchoring on the original goal — re-plan from the task…",
                                );
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
                                continue;
                            }

                            // Reactive debugger (opt-in): once the primary
                            // agent has failed the gate a couple times on its
                            // own, hand the SPECIFIC failure to a fresh-context
                            // sub-agent. Its fix lands in the shared revision
                            // store, so the next gate re-check (continue below)
                            // validates it. Single-fire by default; with
                            // `debugger_multifire` it re-fires only on a CHANGED
                            // failure signature (walk compile→smoke).
                            let fkey = failure_key(&output);
                            let may_fire = if config.tools.debugger_multifire {
                                debugger_fires < debugger::MAX_DEBUGGER_FIRES
                                    && last_debugged_failure.as_deref() != Some(fkey.as_str())
                            } else {
                                debugger_fires == 0
                            };
                            if (config.tools.reactive_debugger || config.tools.debugger_judge)
                                && may_fire
                                && validation_blocks >= debugger::DEBUGGER_TRIGGER_BLOCKS
                            {
                                debugger_fires += 1;
                                last_debugged_failure = Some(fkey);
                                tui::print_status(
                                    "Still failing — spinning up a fresh-context debugger sub-agent…",
                                );
                                let verdict = debugger::run_debugger(
                                    &output,
                                    message,
                                    &config,
                                    &llm_worker,
                                    &tool_pool,
                                    &tool_defs,
                                    &perms,
                                    &mcp_registry,
                                    &lsp_client,
                                    &fast_revisions,
                                    fast_baseline_errors,
                                    &cancelled,
                                )
                                .await;

                                // Debugger-as-judge: SCRAP → the LOOP executes the
                                // whole-tree restart (the stuck agent never decides);
                                // Rewind → the loop reverts JUST the one flagged file;
                                // Report → inject the diagnosis for the main agent.
                                let msg = match verdict {
                                    debugger::DebuggerVerdict::Scrap if !restart_fired => {
                                        restart_fired = true;
                                        if let Some(ref snap) = snapshots {
                                            let guard = snap.lock();
                                            match guard.revert_to_round(0) {
                                                Ok(m) => tui::print_status(&format!(
                                                    "[debugger-judge] SCRAP — {m}"
                                                )),
                                                Err(e) => tui::print_status(&format!(
                                                    "[debugger-judge] SCRAP — tree revert failed: {e}"
                                                )),
                                            }
                                        }
                                        // Resync the symbol index / repo-map to the clean
                                        // baseline after the whole-tree revert (the
                                        // per-edit reindex path doesn't cover it).
                                        tools::reindex_project_incremental(&config);
                                        let _ =
                                            std::fs::remove_file(config.miniswe_path("plan.md"));
                                        let _ = std::fs::remove_file(
                                            config.miniswe_path("scratchpad.md"),
                                        );
                                        let assembled = context::assemble(
                                            &config,
                                            message,
                                            &[],
                                            plan_only,
                                            mcp_summary.as_deref(),
                                        );
                                        messages = assembled.messages;
                                        conversation_history.clear();
                                        validation_blocks = 0;
                                        same_plan_step_failures = 0;
                                        last_failed_plan_step = None;
                                        tui::print_status(
                                            "[debugger-judge] scrapped the stuck state — clean baseline + fresh context; restarting from scratch.",
                                        );
                                        continue;
                                    }
                                    debugger::DebuggerVerdict::Scrap => Message::user(
                                        "[A fresh-context review voted to reset again, but the \
                                         tree was already reset once this turn. Keep going: read \
                                         the current failure carefully and fix it directly.]",
                                    ),
                                    debugger::DebuggerVerdict::Rewind(candidate) => {
                                        rewind_message(
                                            &candidate,
                                            &config,
                                            &perms,
                                            &lsp_client,
                                            &fast_revisions,
                                            fast_baseline_errors,
                                            &output,
                                        )
                                        .await
                                    }
                                    debugger::DebuggerVerdict::Report(body) => {
                                        // The debugger's diagnosis is a paraphrase of the raw
                                        // check output — a paraphrase can lose precision the
                                        // raw text had (e.g. the exact "GOT: <value>" line
                                        // pointing at which callsite is actually broken).
                                        // Point at the raw output alongside the diagnosis so
                                        // the model can fall back to ground truth if the
                                        // summary steers it wrong.
                                        let output_note =
                                            write_gate_failure_output(&config, &output)
                                                .map(|path| {
                                                    format!(
                                                        "\nFull raw check output: read(\"{path}\")."
                                                    )
                                                })
                                                .unwrap_or_default();
                                        Message::user(&format!(
                                            "[A read-only debugger with fresh eyes investigated the failing \
                                         check and produced this DIAGNOSIS. It did not edit anything — \
                                         YOU must apply the fix and finish the plan it lays out:\n{body}\n\
                                         Make the change(s), then finish; the verification will re-run.{output_note}]"
                                        ))
                                    }
                                };
                                messages.push(msg.clone());
                                conversation_history.push(msg);
                                continue;
                            }

                            // Gate context-reset (opt-in): instead of grinding
                            // in-context after repeated gate blocks, drop the
                            // polluted history and re-assemble a clean context —
                            // the in-session equivalent of a best-of-3 fresh
                            // attempt (files persist on disk). Bounded per turn.
                            if config.tools.gate_context_reset
                                && gate_resets < spiral::MAX_GATE_RESETS
                                && validation_blocks >= spiral::GATE_RESET_AFTER_BLOCKS
                            {
                                gate_resets += 1;
                                validation_blocks = 0; // fresh gate budget for the clean restart
                                let fresh = spiral::build_gate_reset_prompt(message, &output);
                                let assembled = context::assemble(
                                    &config,
                                    &fresh,
                                    &[],
                                    plan_only,
                                    mcp_summary.as_deref(),
                                );
                                messages = assembled.messages;
                                tui::print_status(
                                    "Gate context-reset — fresh start (history cleared, files kept).",
                                );
                                log.tool_debug(
                                    "agent",
                                    "gate context-reset: re-assembled clean context after repeated gate blocks",
                                );
                                continue;
                            }

                            let msg = Message::user(&format!(
                                "[Verification failed — do NOT finish yet. A check that exercises \
                                 the change end-to-end exited non-zero; the output below shows what \
                                 is actually wrong. Read it carefully and fix the SPECIFIC problem \
                                 it reports (it may be a compile error, not a logic error), then \
                                 continue. (If you are certain the check itself is wrong, finish \
                                 anyway and state the specific reason — it will be recorded.)\n\
                                 Check output:\n{output}]"
                            ));
                            messages.push(msg.clone());
                            conversation_history.push(msg);
                            continue;
                        }
                        validation::CheckOutcome::Pass | validation::CheckOutcome::Skipped => {}
                    }
                }
                // Exiting now. If the gate blocked the model along the way,
                // surface its recorded rationale(s) for audit — whether it
                // ultimately fixed the change or exhausted the retry budget.
                if !validation_disputes.is_empty() {
                    tui::print_status(&format!(
                        "Completed after {} blocked verification(s); model's reasons recorded in the log.",
                        validation_disputes.len()
                    ));
                    tracing::warn!(
                        "[validation] turn completed over {} blocked check(s); model rationale(s): {}",
                        validation_disputes.len(),
                        validation_disputes.join(" | ")
                    );
                }
                break;
            }
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

        for tc in &tool_calls {
            let args: serde_json::Value = match serde_json::from_str(&tc.function.arguments) {
                Ok(v) => v,
                Err(e) => {
                    let result_msg = Message::tool_result(
                        &tc.id,
                        &format!(
                            "Invalid JSON in tool arguments: {e}\nRaw: {}",
                            tc.function.arguments
                        ),
                    );
                    messages.push(result_msg.clone());
                    conversation_history.push(result_msg);
                    tui::print_tool_result(&tc.function.name, false, "invalid JSON args");
                    continue;
                }
            };

            let args_summary = summarize_args(&tc.function.name, &args);

            // Detect tool call loops: identical calls repeated consecutively
            // (period-1), or the SAME two calls alternating (period-2 — the
            // edit↔revert oscillation that the streak counter is blind to
            // because every alternation resets it).
            let call_key = loop_call_key(&tc.function.name, &args);
            if last_call_key.as_ref() == Some(&call_key) {
                same_call_streak += 1;
            } else {
                last_call_key = Some(call_key.clone());
                same_call_streak = 1;
            }
            recent_call_keys.push(call_key.clone());
            if recent_call_keys.len() > 12 {
                recent_call_keys.remove(0);
            }
            let period2 = is_period2_cycle(&recent_call_keys);
            if same_call_streak >= 3 || period2 {
                // Period-2-only detection (not also a plain streak). Captured
                // before any state resets below so messaging stays accurate.
                let p2_only = period2 && same_call_streak < 3;
                // A period-2 cycle is harmful if EITHER member mutates (the
                // classic case is edit↔revert — both mutate; edit↔read still
                // re-applies the same broken edit).
                let mutating = if p2_only {
                    let tail = &recent_call_keys[recent_call_keys.len().saturating_sub(2)..];
                    tail.iter().any(|k| key_is_mutating(k))
                } else {
                    is_mutating_call(&tc.function.name, &args)
                };
                log.loop_detected(&tc.function.name, &args_summary, same_call_streak as usize);

                // Read-only repetition: harmless, just wasted tokens.
                // Surface a polite nudge inline and let the for-loop
                // continue so the model can react in subsequent rounds.
                if !mutating {
                    let result_msg = Message::tool_result(&tc.id, REPEATED_READ_NUDGE);
                    messages.push(result_msg.clone());
                    conversation_history.push(result_msg);
                    tui::print_status(&format!(
                        "Repeated read: {}({}) — nudge sent, continuing",
                        tc.function.name, args_summary
                    ));
                    last_call_key = None;
                    same_call_streak = 0;
                    recent_call_keys.clear();
                    continue;
                }

                let hint = if p2_only {
                    PERIOD2_LOOP_HINT
                } else {
                    loop_detected_hint(config.tools.edit_mode)
                };
                let result_msg = Message::tool_result(&tc.id, hint);
                messages.push(result_msg.clone());
                conversation_history.push(result_msg);

                // First mutating loop in this turn: surface the hint, reset
                // the streak, and let the model try a different approach.
                // Subsequent loops mean the recovery itself spiraled —
                // abort for real.
                if loop_recoveries == 0 {
                    loop_recoveries += 1;
                    last_call_key = None;
                    same_call_streak = 0;
                    recent_call_keys.clear();
                    tui::print_error(&format!(
                        "Loop detected: {}({}) {} — surfacing a hint, giving the model one more round",
                        tc.function.name,
                        args_summary,
                        if p2_only {
                            "alternating with the same partner call (period-2 cycle)"
                        } else {
                            "repeated 3 times"
                        }
                    ));
                    break;
                }
                // Second mutating loop after the recovery hint. With a
                // behavioral done-gate configured this is NOT a dead end — it
                // is the same "stuck but the task isn't done" state as a
                // premature exit, so route it through the gate ladder (block →
                // debugger/judge at 2 blocks) instead of dying with the whole
                // recovery stack idle. (Real case: a run died at 70s looping
                // on a malformed replace_range while gate + judge never ran.)
                // Without a gate: original behavior — stop the turn.
                if config.validation.command().is_some()
                    && validation_blocks < config.validation.max_retries
                {
                    tui::print_error(&format!(
                        "Loop detected again ({}({})) — routing through the done-gate instead of stopping",
                        tc.function.name, args_summary
                    ));
                    if let validation::CheckOutcome::Fail(output) =
                        validation::run_behavioral_check(&config).await
                    {
                        validation_blocks += 1;
                        // Fresh recovery budget for the rounds the gate grants.
                        loop_recoveries = 0;
                        last_call_key = None;
                        same_call_streak = 0;
                        recent_call_keys.clear();

                        let fkey = failure_key(&output);
                        let may_fire = if config.tools.debugger_multifire {
                            debugger_fires < debugger::MAX_DEBUGGER_FIRES
                                && last_debugged_failure.as_deref() != Some(fkey.as_str())
                        } else {
                            debugger_fires == 0
                        };
                        if (config.tools.reactive_debugger || config.tools.debugger_judge)
                            && may_fire
                            && validation_blocks >= debugger::DEBUGGER_TRIGGER_BLOCKS
                        {
                            debugger_fires += 1;
                            last_debugged_failure = Some(fkey);
                            tui::print_status(
                                "Looping + failing gate — spinning up a fresh-context debugger sub-agent…",
                            );
                            let verdict = debugger::run_debugger(
                                &output,
                                message,
                                &config,
                                &llm_worker,
                                &tool_pool,
                                &tool_defs,
                                &perms,
                                &mcp_registry,
                                &lsp_client,
                                &fast_revisions,
                                fast_baseline_errors,
                                &cancelled,
                            )
                            .await;

                            let msg = match verdict {
                                debugger::DebuggerVerdict::Scrap if !restart_fired => {
                                    restart_fired = true;
                                    if let Some(ref snap) = snapshots {
                                        let guard = snap.lock();
                                        match guard.revert_to_round(0) {
                                            Ok(m) => tui::print_status(&format!(
                                                "[debugger-judge] SCRAP — {m}"
                                            )),
                                            Err(e) => tui::print_status(&format!(
                                                "[debugger-judge] SCRAP — tree revert failed: {e}"
                                            )),
                                        }
                                    }
                                    tools::reindex_project_incremental(&config);
                                    let _ = std::fs::remove_file(config.miniswe_path("plan.md"));
                                    let _ =
                                        std::fs::remove_file(config.miniswe_path("scratchpad.md"));
                                    let assembled = context::assemble(
                                        &config,
                                        message,
                                        &[],
                                        plan_only,
                                        mcp_summary.as_deref(),
                                    );
                                    messages = assembled.messages;
                                    conversation_history.clear();
                                    validation_blocks = 0;
                                    same_plan_step_failures = 0;
                                    last_failed_plan_step = None;
                                    tui::print_status(
                                        "[debugger-judge] scrapped the stuck state — clean baseline + fresh context; restarting from scratch.",
                                    );
                                    continue 'round;
                                }
                                debugger::DebuggerVerdict::Scrap => Message::user(
                                    "[A fresh-context review voted to reset again, but the tree \
                                     was already reset once this turn. Keep going: read the \
                                     current failure carefully and fix it directly.]",
                                ),
                                debugger::DebuggerVerdict::Rewind(candidate) => {
                                    rewind_message(
                                        &candidate,
                                        &config,
                                        &perms,
                                        &lsp_client,
                                        &fast_revisions,
                                        fast_baseline_errors,
                                        &output,
                                    )
                                    .await
                                }
                                debugger::DebuggerVerdict::Report(body) => {
                                    let output_note = write_gate_failure_output(&config, &output)
                                        .map(|path| {
                                            format!("\nFull raw check output: read(\"{path}\").")
                                        })
                                        .unwrap_or_default();
                                    Message::user(&format!(
                                        "[A read-only debugger with fresh eyes investigated the failing \
                                     check and produced this DIAGNOSIS. It did not edit anything — \
                                     YOU must apply the fix and finish the plan it lays out:\n{body}\n\
                                     Make the change(s), then finish; the verification will re-run.{output_note}]"
                                    ))
                                }
                            };
                            messages.push(msg.clone());
                            conversation_history.push(msg);
                            continue 'round;
                        }

                        let msg = Message::user(&format!(
                            "[Your repeated failing tool call was aborted, and the task is NOT \
                             done — the verification check failed:\n{output}\nRead the check \
                             output and your tool errors carefully, fix the SPECIFIC problem, \
                             and use a correctly-formed call (include every required parameter) \
                             or a different tool.]"
                        ));
                        messages.push(msg.clone());
                        conversation_history.push(msg);
                        continue 'round;
                    }
                    // Gate passed (or skipped): the loop was on something the
                    // check doesn't care about — fall through to the stop.
                }
                tui::print_error(&format!(
                    "Loop detected again ({}({})) after the recovery hint — stopping this turn",
                    tc.function.name, args_summary
                ));
                had_error = true;
                break;
            }

            log.tool_call_detail(&tc.function.name, &args);
            tui::print_tool_call(&tc.function.name, &args_summary);

            // Block write tools in plan-only mode
            let file_action = args["action"].as_str().unwrap_or("");
            if plan_only
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
                tui::print_tool_result(&tc.function.name, false, "blocked in plan mode");
                continue;
            }

            // Write gating: require plan before write tools (strict only)
            let is_write_action = is_file_write(tc.function.name.as_str());
            if strict && config.tools.plan && !tools::plan::plan_exists(&config) && is_write_action
            {
                let result_msg = Message::tool_result(
                    &tc.id,
                    "Create a plan first: use plan(action='set') with your step-by-step approach before making changes.",
                );
                messages.push(result_msg.clone());
                conversation_history.push(result_msg);
                tui::print_tool_result(&tc.function.name, false, "blocked: no plan");
                continue;
            }
            // (Plan-checkpoint used to hard-block writes after N edits without
            //  a plan action; that interacted poorly with the compile-gate on
            //  `plan(check)` — if the project didn't compile, the model
            //  couldn't escape the block, couldn't fix the project, deadlock.
            //  Now we just warn at the threshold via PLAN_CHECKPOINT_WARNING
            //  appended to the tool result; the model decides what to do.)

            // Handle tool dispatch
            let mut result = if tc.function.name == "file" && file_action == "revert" {
                let snapshots = snapshots.clone();
                let args = args.clone();
                match tool_pool
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
                    Err(_) => {
                        crate::tools::ToolResult::err("Tool worker dropped revert job".into())
                    }
                }
            } else if tc.function.name == "plan" {
                let args = args.clone();
                let config = config.clone();
                match tool_pool
                    .submit(move || {
                        let runtime = tokio::runtime::Builder::new_current_thread()
                            .enable_all()
                            .build()
                            .map_err(|e| e.to_string())?;
                        runtime
                            .block_on(
                                async move { tools::plan::execute(&args, &config, round).await },
                            )
                            .map_err(|e| format!("plan error: {e}"))
                    })
                    .await
                {
                    Ok(Ok(r)) => r,
                    Ok(Err(e)) => crate::tools::ToolResult::err(e),
                    Err(_) => crate::tools::ToolResult::err("Tool worker dropped plan job".into()),
                }
            } else if tc.function.name == "edit_file" {
                let args = args.clone();
                let config = config.clone();
                let perms = perms.clone();
                let router = router.clone();
                let lsp = lsp_client.clone();
                let cancelled = cancelled.clone();
                let log = log.clone();
                match tool_pool
                    .submit(move || {
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
                                    Some(cancelled.as_ref()),
                                    Some(log.as_ref()),
                                )
                                .await
                            })
                            .map_err(|e| format!("edit_file error: {e}"))
                    })
                    .await
                {
                    Ok(Ok(r)) => r,
                    Ok(Err(e)) => crate::tools::ToolResult::err(e),
                    Err(_) => {
                        crate::tools::ToolResult::err("Tool worker dropped edit_file job".into())
                    }
                }
            } else if tc.function.name == "refactor"
                || matches!(
                    tc.function.name.as_str(),
                    "add_function_param" | "drop_function_param" | "rename_symbol"
                )
            {
                // Flat refactor tools normalize into the grouped
                // `refactor` args shape; same executor.
                let args = tools::definitions::flat_to_refactor_args(&tc.function.name, &args)
                    .unwrap_or_else(|| args.clone());
                let config = config.clone();
                let router = router.clone();
                let lsp = lsp_client.clone();
                let log_for_job = log.clone();
                let revisions_for_job = fast_revisions.clone();
                let cancelled = cancelled.clone();
                match tool_pool
                    .submit(move || {
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
                                    Some(cancelled.as_ref()),
                                )
                                .await
                            })
                            .map_err(|e| format!("refactor error: {e}"))
                    })
                    .await
                {
                    Ok(Ok(r)) => r,
                    Ok(Err(e)) => crate::tools::ToolResult::err(e),
                    Err(_) => {
                        crate::tools::ToolResult::err("Tool worker dropped refactor job".into())
                    }
                }
            } else if tc.function.name == "file" && file_action == "shell" {
                await_shell_job_run(
                    tool_pool.submit_shell(args.clone(), config.clone(), cancelled.clone()),
                    cancelled.as_ref(),
                )
                .await
            } else if matches!(
                tc.function.name.as_str(),
                "replace_range" | "insert_at" | "revert" | "show_rev" | "check"
            ) && config.tools.edit_mode == EditMode::Fast
            {
                let tool_name = tc.function.name.clone();
                let args = args.clone();
                let config = config.clone();
                let perms = perms.clone();
                let lsp = lsp_client.clone();
                let revisions = fast_revisions.clone();
                let baseline = fast_baseline_errors;
                match tool_pool
                    .submit(move || {
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
                    })
                    .await
                {
                    Ok(Ok(r)) => r,
                    Ok(Err(e)) => crate::tools::ToolResult::err(e),
                    Err(_) => {
                        crate::tools::ToolResult::err("Tool worker dropped fast tool job".into())
                    }
                }
            } else if tc.function.name == "mcp_use" {
                let server = args["server"].as_str().unwrap_or("").to_string();
                let tool = args["tool"].as_str().unwrap_or("").to_string();
                let tool_args = args.get("arguments").cloned().unwrap_or_default();
                if server.is_empty() || tool.is_empty() {
                    crate::tools::ToolResult::err(
                        "mcp_use requires top-level 'server' and 'tool' string fields. \
                         Example: {\"server\": \"my-server\", \"tool\": \"my-tool\", \"arguments\": {}}".into(),
                    )
                } else {
                    match perms.check(&Action::McpUse(server.clone(), tool.clone())) {
                        Err(e) => crate::tools::ToolResult::err(e),
                        Ok(()) => {
                            let registry = mcp_registry.clone();
                            match tool_pool
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
                                Err(_) => crate::tools::ToolResult::err(
                                    "Tool worker dropped mcp job".into(),
                                ),
                            }
                        }
                    }
                }
            } else if tc.function.name == "spawn_agents" {
                let tasks = crate::cli::commands::agent::subagent::parse_tasks(&args);
                if tasks.is_empty() {
                    crate::tools::ToolResult::err(
                        "spawn_agents: 'agents' must be a non-empty array of {label, prompt}"
                            .into(),
                    )
                } else {
                    tui::print_status(&format!("spawning {} subagents...", tasks.len()));
                    let outputs = crate::cli::commands::agent::subagent::run_subagents(
                        tasks,
                        &config,
                        &llm_worker,
                        &tool_pool,
                        &tool_defs,
                        &perms,
                        &mcp_registry,
                        &lsp_client,
                        &fast_revisions,
                        fast_baseline_errors,
                        &cancelled,
                        None,
                    )
                    .await;
                    let combined = crate::cli::commands::agent::subagent::format_outputs(outputs);
                    crate::tools::ToolResult::ok(combined)
                }
            } else {
                let tool_name = tc.function.name.clone();
                let args = args.clone();
                let config = config.clone();
                let perms = perms.clone();
                let lsp = lsp_client.clone();
                match tool_pool
                    .submit(move || {
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
                    })
                    .await
                {
                    Ok(Ok(r)) => r,
                    Ok(Err(e)) => crate::tools::ToolResult::err(e),
                    Err(_) => crate::tools::ToolResult::err("Tool worker dropped job".into()),
                }
            };

            if !result.success
                && let Some(hint) = tools::plan::failure_hint(&config)
            {
                result.content.push('\n');
                result.content.push_str(&hint);
            }

            // Append round number to every tool result
            result
                .content
                .push_str(&format!("\n[round {round}/{max_rounds}]"));

            let first_line = result.content.lines().next().unwrap_or("(empty)");
            log.tool_call(&tc.function.name, &args_summary, result.success, first_line);
            log.tool_result_detail(&tc.function.name, result.success, &result.content);
            tui::print_tool_result(&tc.function.name, result.success, first_line);

            if result.success && tc.function.name == "plan" {
                successful_edits_since_plan_update = 0;
            }

            // A successful file write means code changed — reset trackers.
            if result.success && is_file_write(tc.function.name.as_str()) {
                last_call_key = None;
                same_call_streak = 0;
                calls_since_last_edit = 0;
                if strict && config.tools.plan {
                    if tools::plan::plan_exists(&config) {
                        result.content.push('\n');
                        result.content.push_str(PLAN_PROGRESS_NUDGE);
                    }
                    successful_edits_since_plan_update += 1;
                    if successful_edits_since_plan_update == PLAN_CHECKPOINT_AFTER_EDITS {
                        result.content.push('\n');
                        result.content.push_str(PLAN_CHECKPOINT_WARNING);
                    }
                }
            } else {
                calls_since_last_edit += 1;
            }

            if !is_prunable_refactor_failure(&result.content, result.success) {
                all_prunable_failures = false;
            } else {
                prunable_errors.push(result.content.clone());
            }

            let result_msg = Message::tool_result(&tc.id, &result.content);
            messages.push(result_msg.clone());
            conversation_history.push(result_msg);

            // `tools.plan_gate_debugger`: the plan tool's OWN compile gate
            // repeatedly blocking the SAME step is a distinct stall signature
            // from the behavioral done-gate (`validation_blocks` above) — the
            // primary agent is re-litigating one step in its own accumulated
            // context rather than making forward progress. See the field doc
            // in config/mod.rs for the forensic evidence motivating this.
            if tc.function.name == "plan"
                && args.get("action").and_then(|a| a.as_str()) == Some("check")
            {
                if result.success {
                    same_plan_step_failures = 0;
                    last_failed_plan_step = None;
                } else if let Some(step) = args.get("step").and_then(|s| s.as_u64()) {
                    track_plan_step_failure(
                        &mut last_failed_plan_step,
                        &mut same_plan_step_failures,
                        step,
                    );

                    let fkey = failure_key(&result.content);
                    let may_fire = if config.tools.debugger_multifire {
                        debugger_fires < debugger::MAX_DEBUGGER_FIRES
                            && last_debugged_failure.as_deref() != Some(fkey.as_str())
                    } else {
                        debugger_fires == 0
                    };
                    if config.tools.plan_gate_debugger
                        && may_fire
                        && same_plan_step_failures as usize >= debugger::DEBUGGER_TRIGGER_BLOCKS
                    {
                        debugger_fires += 1;
                        last_debugged_failure = Some(fkey);
                        tui::print_status(
                            "Plan-check gate failing repeatedly on the same step — spinning up a fresh-context debugger sub-agent…",
                        );
                        let verdict = debugger::run_debugger(
                            &result.content,
                            message,
                            &config,
                            &llm_worker,
                            &tool_pool,
                            &tool_defs,
                            &perms,
                            &mcp_registry,
                            &lsp_client,
                            &fast_revisions,
                            fast_baseline_errors,
                            &cancelled,
                        )
                        .await;

                        let extra_msg = match verdict {
                            debugger::DebuggerVerdict::Scrap if !restart_fired => {
                                restart_fired = true;
                                if let Some(ref snap) = snapshots {
                                    let guard = snap.lock();
                                    match guard.revert_to_round(0) {
                                        Ok(m) => tui::print_status(&format!(
                                            "[debugger-judge] SCRAP — {m}"
                                        )),
                                        Err(e) => tui::print_status(&format!(
                                            "[debugger-judge] SCRAP — tree revert failed: {e}"
                                        )),
                                    }
                                }
                                tools::reindex_project_incremental(&config);
                                let _ = std::fs::remove_file(config.miniswe_path("plan.md"));
                                let _ = std::fs::remove_file(config.miniswe_path("scratchpad.md"));
                                let assembled = context::assemble(
                                    &config,
                                    message,
                                    &[],
                                    plan_only,
                                    mcp_summary.as_deref(),
                                );
                                messages = assembled.messages;
                                conversation_history.clear();
                                validation_blocks = 0;
                                same_plan_step_failures = 0;
                                last_failed_plan_step = None;
                                tui::print_status(
                                    "[debugger-judge] scrapped the stuck state — clean baseline + fresh context; restarting from scratch.",
                                );
                                continue 'round;
                            }
                            debugger::DebuggerVerdict::Scrap => Message::user(
                                "[A fresh-context review voted to reset again, but the tree \
                                 was already reset once this turn. Keep going: read the \
                                 current failure carefully and fix it directly.]",
                            ),
                            debugger::DebuggerVerdict::Rewind(candidate) => {
                                rewind_message(
                                    &candidate,
                                    &config,
                                    &perms,
                                    &lsp_client,
                                    &fast_revisions,
                                    fast_baseline_errors,
                                    &result.content,
                                )
                                .await
                            }
                            debugger::DebuggerVerdict::Report(body) => {
                                let output_note =
                                    write_gate_failure_output(&config, &result.content)
                                        .map(|path| {
                                            format!("\nFull raw check output: read(\"{path}\").")
                                        })
                                        .unwrap_or_default();
                                Message::user(&format!(
                                    "[A read-only debugger with fresh eyes investigated the failing \
                                 plan-check step and produced this DIAGNOSIS. It did not edit \
                                 anything — YOU must apply the fix and finish the step it lays \
                                 out:\n{body}\nMake the change(s), then re-check the step.{output_note}]"
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
            if config.tools.spiral_reset
                && result.success
                && tc.function.name == "revert"
                && config.tools.edit_mode == EditMode::Fast
                && spiral_resets < spiral::MAX_RESETS_PER_TURN
                && let Some(path) = args.get("path").and_then(|p| p.as_str())
            {
                let count = revert_counts.entry(path.to_string()).or_insert(0);
                *count += 1;
                if *count >= spiral::SPIRAL_REVERT_THRESHOLD {
                    let n = *count;
                    *count = 0;
                    spiral_resets += 1;
                    let tried = fast_revisions
                        .as_deref()
                        .map(|r| spiral::tried_edit_labels(r, path, 4))
                        .unwrap_or_default();
                    let reset = Message::user(&spiral::build_reset_message(path, n, &tried));
                    messages.push(reset.clone());
                    conversation_history.push(reset);
                    tui::print_status("Spiral detected (revert-loop) — reset + replan injected.");
                    log.tool_debug(
                        "agent",
                        &format!("spiral-reset fired for {path} after {n} reverts"),
                    );
                }
            }
        }

        // History pruning: if every tool call in this assistant message was
        // a prunable validator failure, drop the assistant message + its
        // tool_results and replace with a user-role corrective. The
        // assistant's bad-shape arguments are what prime the model to
        // repeat them; removing them breaks the loop. Verified empirically
        // (probe D3): clean history → clean output.
        if all_prunable_failures && !prunable_errors.is_empty() {
            messages.truncate(messages_pre);
            conversation_history.truncate(history_pre);
            let hint = Message::user(&format!(
                "Your previous refactor call(s) were rejected:\n\n{}\n\n\
                 Retry with all required parameters and a clean position value \
                 (one of 'start' or 'after:<single_param_name>').",
                prunable_errors.join("\n\n---\n\n")
            ));
            messages.push(hint.clone());
            conversation_history.push(hint);
            log.tool_debug(
                "agent",
                &format!(
                    "history pruned: dropped {} tool_result(s) after refactor validator failure",
                    prunable_errors.len()
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
        if strict && round >= 12 && !nudged_no_plan && !tools::plan::plan_exists(&config) {
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
            nudged_no_plan = true;
        }

        // Stall detection: too many tool calls without any edits.
        // Content is plan-state aware: without a plan the edit tools are
        // hidden, so pointing the model at them is a schema-runtime
        // mismatch. Re-fire the plan nudge instead (with a more urgent
        // tone than the round-12 first nudge).
        if calls_since_last_edit >= 20 && calls_since_last_edit.is_multiple_of(20) {
            let body = if strict && !tools::plan::plan_exists(&config) {
                "Still no plan set after 20+ exploration calls. \
                 Edit tools cannot appear in your tool list until plan(action='set') is called. \
                 Stop exploring and set a plan now — even an imperfect plan can be refined later. \
                 If something is blocking you from planning, say so."
                    .to_string()
            } else {
                let edit_hint = match config.tools.edit_mode {
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

    log.session_end(round, had_error);

    // Shut down LSP
    if let Some(lsp) = lsp_client
        && let Ok(lsp) = Arc::try_unwrap(lsp)
    {
        lsp.shutdown().await;
    }

    tui::print_separator();
    if !had_error {
        tui::print_complete("Done");
    }

    Ok(())
}

async fn await_shell_job_run(
    mut shell_job: crate::runtime::ShellJobHandle,
    cancelled: &AtomicBool,
) -> crate::tools::ToolResult {
    while let Some(event) = shell_job.events_rx.recv().await {
        match event {
            ShellWorkerEvent::TimedOut {
                command,
                timeout_secs,
            } => {
                let prompt = format!(
                    "Shell command still running after {timeout_secs}s.\n  $ {command}\n[c]ontinue waiting / [k]ill: "
                );
                let response = crate::tui::read_input(&prompt)
                    .unwrap_or_else(|| "k".into())
                    .trim()
                    .to_lowercase();
                let control = if response == "c" || response == "continue" {
                    crate::tui::print_status("continuing to wait for shell command...");
                    ShellControl::Continue
                } else {
                    ShellControl::Kill
                };
                if shell_job.send_control(control).is_err() {
                    return crate::tools::ToolResult::err(
                        "Shell worker dropped control channel".into(),
                    );
                }
            }
            ShellWorkerEvent::Completed(result) => {
                if cancelled.load(Ordering::Relaxed) {
                    cancelled.store(false, Ordering::Relaxed);
                }
                return match result {
                    Ok(tool_result) => tool_result,
                    Err(err) => crate::tools::ToolResult::err(err),
                };
            }
        }
    }
    if cancelled.load(Ordering::Relaxed) {
        cancelled.store(false, Ordering::Relaxed);
    }
    crate::tools::ToolResult::err("Shell worker dropped before reporting a result".into())
}
