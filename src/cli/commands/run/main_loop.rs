//! The main agent loop: `run()` drives rounds of LLM calls and tool
//! execution until the task completes or is stopped.

use super::*;

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
    // This surface registers the `skill` tool, so the [SKILL STEP] block's
    // instruction to call it is actionable here (unlike the repl).
    config.skill_step_injection = true;
    let log = Arc::new(SessionLog::new(&config));
    log.user_message(message);

    let router = Arc::new(ModelRouter::new(&config));
    // Fail fast on a missing hosted-provider API key, before anything else
    // starts (session dir, worker pool) only to die on the first LLM call.
    router.check_credentials()?;
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
        // Background-jobs surface: explicit background=true works in every
        // mode; auto-promotion of long foreground commands stays headless-
        // only (interactive keeps the human continue/kill prompt).
        tool_defs.push(tools::definitions::shell_tool_definition());
    }

    // Registry for background jobs (explicit background=true + promoted).
    let job_registry = Arc::new(tools::jobs::JobRegistry::default());

    // Session working state (plan.md, scratchpad.md) lives in a private
    // per-session directory, so there is nothing stale to clear and no
    // shared path a concurrent or nested run could wipe out from under us.
    // `--continue` adopts the previous session's directory rather than
    // opening a fresh one.
    let sessions_dir = config.sessions_dir();
    if continue_session && let Some(previous) = crate::config::session::last_id(&sessions_dir) {
        config.session_id = previous;
    }
    let _ = config.ensure_session_dir();
    crate::config::session::record_last(&sessions_dir, &config.session_id);
    crate::config::session::prune(
        &sessions_dir,
        crate::config::session::RETENTION,
        &config.session_id,
    );

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
                // `{e:#}` not `{e}`: the whole cause chain matters here. The
                // outer context alone reads as an unexplained "failed to get
                // rust-analyzer binary" and hides whether it was an HTTP
                // status, a bad gzip, or a binary that would not exec.
                tui::print_status(&format!("LSP: not available ({e:#})"));
                // Stable, greppable marker. Losing the LSP silently removes
                // the refactor tools, and the session still runs to completion
                // and produces a plausible score — that is how a benchmark 4/6
                // caused by a failed download got read as a model regression.
                tui::print_status(
                    "LSP: DEGRADED — refactor tools (add_param/drop_param/rename) unavailable",
                );
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

    // `MINISWE_MAX_ROUNDS` overrides the configured round cap without editing
    // config.toml — used by the e2e harness to give long multi-phase skills
    // (build → integrate → deploy) enough budget. Invalid/absent → config.
    let max_rounds = std::env::var("MINISWE_MAX_ROUNDS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(config.context.max_rounds);
    // Ceremony=Off (default, evidence-distilled): no plan gate, no
    // plan/no-plan nudges, all edit tools always visible, no phase
    // rebuild. `strict` re-enables the legacy plan-first machinery.
    // See docs/tiered-agent-design.md.
    let strict = config.tools.ceremony == crate::config::CeremonyMode::Strict;

    let mut conversation_history: Vec<Message> = Vec::new();
    // Per-turn agent-loop state shared with the REPL loop (see the field docs
    // on `turn_state::TurnState` and its sub-structs).
    let mut state = turn_state::TurnState::default();
    // Skill-cursor finish-gate state (headless-only — see the field docs on
    // `turn_state::SkillTurnState`).
    let mut skill_state = turn_state::SkillTurnState::default();
    // `tools.stuck_check`: T2c frozen-signature detector (see the module doc
    // in agent/stuck_check.rs and the config field doc). Fed unconditionally
    // (cheap string scans); fires only when the flag is on.
    let session_start = std::time::Instant::now();

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

    // Pre-turn skill router (fail-safe): a dedicated no-tools classifier
    // maps the task onto one installed skill, then the task is rewritten to
    // an imperative read-and-follow. Probe 2026-07-15: classifier 30/30,
    // first-action skill adoption 0/8 -> 8/8. NONE/parse-failure/LLM-error
    // all fall through to the original task unchanged.
    use crate::cli::commands::agent::{skill_cursor, skill_router};
    // Stale skill cursor from a previous session (unless --continue).
    if !continue_session {
        skill_cursor::clear(&config);
    }
    // On a --continue retry with a still-active cursor, DON'T re-route: the
    // continuation message (e.g. "fix these validation errors") drags the
    // router onto a narrow sub-skill and a fresh push would clobber the
    // in-progress lifecycle cursor (pkg e2e 2026-07-16: the umbrella
    // build/integrate cursor got overwritten by pkg-package-validate on
    // retry). Keep the existing cursor; the continuation prompt flows to the
    // model with the current [SKILL STEP] still injected.
    let routed_message: String = if continue_session && skill_cursor::load(&config).is_active() {
        tui::print_status("[skills] continuing in-progress step-cursor; skipping re-route");
        message.to_string()
    } else {
        match skill_router::route_task_to_skill(
            &llm_worker,
            &config.project_root,
            message,
            &cancelled,
        )
        .await
        {
            Some(skill) => {
                tui::print_status(&format!("[skills] task routed to skill '{skill}'"));
                log.user_message(&format!("[skill-router] using '{skill}'"));
                // Decoupled step-cursor execution: extract the skill's steps into
                // a harness-owned cursor (separate from the model's plan.md, so
                // they can't fight). Each round the CURRENT step's instructions
                // are distilled just-in-time and re-injected under [SKILL STEP];
                // the model signals skill(action='done') to advance. Falls back
                // to plain read-and-follow if extraction yields nothing.
                let mut used_cursor = false;
                if let Some(entry) = crate::skills::discover(&config.project_root)
                    .into_iter()
                    .find(|e| e.name == skill)
                    && let Ok(loaded) = crate::skills::load(&entry.path)
                {
                    let steps =
                        skill_router::extract_skill_steps(&llm_worker, &loaded.body, &cancelled)
                            .await;
                    if !steps.is_empty() {
                        let dir = entry
                            .path
                            .parent()
                            .unwrap_or(&config.project_root)
                            .to_path_buf();
                        let mut cursor = skill_cursor::SkillCursor::default();
                        cursor.push_skill(&skill, &dir, steps);
                        skill_cursor::save(&config, &cursor);
                        used_cursor = true;
                        tui::print_status(&format!(
                            "[skills] seeded step-cursor from '{skill}'; will distill + re-inject each step"
                        ));
                    }
                }
                if used_cursor {
                    format!(
                        "Handle this request by following the guided skill steps. The current step's \
                     instructions appear under [SKILL STEP] in the current state — do exactly that \
                     step, then call skill(action='done') to advance. Do not skip ahead or \
                     improvise. Request: {message}"
                    )
                } else {
                    skill_router::rewrite_task_for_skill(&skill, message)
                }
            }
            None => message.to_string(),
        }
    };
    let message: &str = &routed_message;

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

    let mut ui = HeadlessUi { headless };

    // Explicit behavior deltas for the shared turn phases — see
    // `turn::TurnOptions` field docs.
    let opts = turn::TurnOptions {
        skill_steps: true,
        clear_cancel_on_interrupt: false,
        worker_stopped_ends_turn: false,
        compaction: turn::CompactionUx::Batch,
        read_only: false,
        live_jobs_gate: true,
        failure_tracking: true,
        stuck_tracking: true,
        snapshot_revert_arm: true,
        flat_refactor_aliases: true,
        inline_mcp_permission_check: true,
        register_shell_jobs: headless,
        jobs_on_pool: false,
        plan_job_direct_await: true,
        window_repeat_detector: true,
        jobs_poll_redirect: true,
        interrupt_checkpoints: false,
        fatal_marks_error: true,
    };

    let ctx = turn::TurnCtx {
        config: &config,
        router: &router,
        llm_worker: &llm_worker,
        lsp: &lsp_client,
        snapshots: &snapshots,
        log: &log,
        tool_defs: &tool_defs,
        cancelled: &cancelled,
        model_role,
        fast_baseline_errors,
        tool_def_tokens,
        max_rounds,
        perms: &perms,
        tool_pool: &tool_pool,
        mcp_registry: &mcp_registry,
        fast_revisions: &fast_revisions,
        job_registry: &job_registry,
        task: message,
        mcp_summary: mcp_summary.as_deref(),
        plan_only,
        session_start,
    };

    let turn::driver::TurnResult { had_error, .. } = turn::driver::run_turn(
        ctx,
        opts,
        &mut state,
        &mut skill_state,
        &mut ui,
        &mut messages,
        &mut conversation_history,
        &router,
        &log,
    )
    .await;

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
