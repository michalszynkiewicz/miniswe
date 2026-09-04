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
    let pause_at = config.context.pause_after_rounds;
    // Ceremony=Off (default, evidence-distilled): no plan gate, no
    // plan/no-plan nudges, all edit tools always visible, no phase
    // rebuild. `strict` re-enables the legacy plan-first machinery.
    // See docs/tiered-agent-design.md.
    let strict = config.tools.ceremony == crate::config::CeremonyMode::Strict;

    let mut conversation_history: Vec<Message> = Vec::new();
    let mut round = 0;
    let mut had_error = false;
    // Per-turn agent-loop state shared with the REPL loop (see the field docs
    // on `turn_state::TurnState` and its sub-structs).
    let mut state = turn_state::TurnState::default();
    // (call_key, failure output) of the most recent FAILED tool call. When the
    // model loops on a call that keeps failing (e.g. `pack package create`
    // returning a lint error 10× — e2e 2026-07-17), this is the real error to
    // hand the debugger. The per-step check can't surface it: it's a read-only
    // existence proxy that PASSES while the command fails, so nothing else
    // triggers recovery on a failing command's own exit code.
    let mut last_tool_failure: Option<(String, String)> = None;
    // Background-job failures keyed by their COMMAND. A detached deploy
    // (`pkg run dev`) fails in a later status/wait result, not on the launch,
    // and each re-launch is a new job id — so the identical-call loop detector
    // never sees the failing DEPLOY. Keyed by command, a repeated failing
    // deploy still routes to the debugger via the recovery ladder (e2e
    // 2026-07-17: the run never deployed because a bad chart git-ref made
    // `pkg run dev` fail, and nothing surfaced it to the debugger).
    let mut failed_job_commands: std::collections::HashMap<String, String> = Default::default();
    let mut nudged_live_jobs = false;
    // Skill-cursor finish-gate state (headless-only — see the field docs on
    // `turn_state::SkillTurnState`).
    let mut skill_state = turn_state::SkillTurnState::default();
    // `tools.stuck_check`: T2c frozen-signature detector (see the module doc
    // in agent/stuck_check.rs and the config field doc). Fed unconditionally
    // (cheap string scans); fires only when the flag is on.
    let session_start = std::time::Instant::now();
    let mut stuck_tracker = stuck_check::StuckTracker::new();

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

    'round: loop {
        if had_error {
            break;
        }
        round += 1;
        log.round_start(round);
        stuck_tracker.on_round(round, session_start.elapsed().as_secs_f64());

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
        {
            let mut cursor = skill_cursor::load(&config);
            if cursor.is_active() {
                let installed: Vec<String> = crate::skills::discover(&config.project_root)
                    .into_iter()
                    .map(|e| e.name)
                    .collect();
                if let Some(next) = cursor.handoff_target(&installed)
                    && !descend_into_skill(&mut cursor, &next, &config, &llm_worker, &cancelled)
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
                    if !prepare_step(&mut cursor, &installed, &config, &llm_worker, &cancelled)
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
                            let recent = recent_activity(&messages, 8, 2600);
                            let (judged_done, reason) = skill_router::judge_step_done(
                                &llm_worker,
                                &step_name,
                                &def,
                                &recent,
                                &cancelled,
                            )
                            .await;
                            if judged_done {
                                let verdict = match &check {
                                    Some(cmd) => validation::run_check_command(&config, cmd).await,
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
                                        &config,
                                        &llm_worker,
                                        &cancelled,
                                    )
                                    .await;
                                }
                            } else {
                                // Surface the judge's reason to the model — it's
                                // often the correct diagnosis the silent gate was
                                // discarding (e2e: it flagged the tmp_repo build).
                                // Dedup so an identical reason isn't re-nudged every
                                // cycle (repeats feed loops).
                                ui.status(&format!(
                                    "[skills] '{step_name}' judged not done: {reason}"
                                ));
                                // Record the standing verdict for the log-only
                                // judge-veto check in the skill(done) handler.
                                skill_state.last_judge_block = Some((
                                    format!("{skill_name}::{step_name}"),
                                    reason.clone(),
                                    skill_state.edits_total,
                                ));
                                if !reason.is_empty()
                                    && skill_state.last_judge_nudge.as_deref()
                                        != Some(reason.as_str())
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
                skill_cursor::save(&config, &cursor);
            }
        }

        // Snapshot at start of each round for revert support
        if let Some(ref snap) = snapshots {
            let mut guard = snap.lock();
            let _ = guard.begin_round(round);
        }

        // revert-to-green: this round STARTS from the state the previous round
        // left (just snapshotted above). If the project has been broken above
        // baseline for spiral::REVERT_TO_GREEN_BLOCKS rounds, the agent is digging
        // deeper, not recovering — reset the whole tree to the last green
        // snapshot and tell it to start over from a clean base.
        if config.tools.revert_to_green
            && config.tools.edit_mode == EditMode::Fast
            && let Some(ref snap) = snapshots
        {
            {
                let errs = tools::fast::project_error_count(lsp_client.as_deref()).await;
                if errs <= fast_baseline_errors {
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
                                messages.push(Message::user(
                                    &spiral::build_revert_to_green_message(
                                        state.green.red_streak,
                                        state.green.last_green_round,
                                    ),
                                ));
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
        }
        if round > max_rounds {
            ui.event(UiEvent::MaxRoundsReached);
            break;
        }

        // Ask user if they want to continue after pause_after_rounds.
        // The headless auto-continue notice lives in HeadlessUi (blocking on
        // stdin hung real e2e harness runs for their full timeout — pkg-mcp
        // 2026-07-13: two of three attempts died waiting at exactly this
        // prompt); max_rounds stays the hard stop.
        if round == pause_at && !state.user_continued {
            match ui.confirm_continue(pause_at).await {
                PauseDecision::Continue => {
                    state.user_continued = true;
                }
                PauseDecision::WrapUp => {
                    // Tell the LLM to wrap up
                    messages.push(Message::user("[Stop now. Summarize what you've done.]"));
                }
            }
        }

        // Warn the LLM when approaching the hard limit
        if round == max_rounds.saturating_sub(5) {
            messages.push(Message::user(
                "[Approaching tool limit. Wrap up and summarize.]",
            ));
        }

        // Drop the middle of any deep run of identical read/inspection pairs
        // BEFORE compaction: compaction only ever summarizes the oldest end,
        // and a read loop lives in the newest messages, so every forced
        // compaction used to leave the repeats untouched and raise their
        // share of the prompt. See `agent::prune_reads`.
        let pruned = prune_repeated_reads(&mut messages);
        if !pruned.is_empty() {
            log.reads_pruned(
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
        let pre_mask = messages.len();
        // Read-loop escalation (see REPEATED_READ_ESCALATION): the loop is
        // sustained by the cache-hot prompt prefix, so break it deliberately
        // even though no budget pressure asks for it. Runs before
        // maybe_compress so refresh_current_state still lands on the tail.
        if state.force_compact_next_round {
            state.force_compact_next_round = false;
            ui.event(UiEvent::ForcingCompaction);
            ui.pump(context::compressor::force_compress(
                &mut messages,
                &config,
                &router,
                &llm_worker,
                tool_def_tokens,
            ))
            .await;
        }
        ui.pump(context::compressor::maybe_compress(
            &mut messages,
            &config,
            &router,
            &llm_worker,
            tool_def_tokens,
            &mut state.plan_update_requested,
        ))
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
        state.plan_ever_set |= plan_set;
        // Off: never hide edit tools (pass plan_exists=true). Strict:
        // legacy hide-until-plan behavior, latched so a plan that goes away
        // mid-segment cannot retract tools the model has already been shown.
        let mut visible = visible_tool_defs(&tool_defs, state.plan_ever_set || !strict);
        // Expose the skill(done) advance control only while a step-cursor is
        // active — inert otherwise, so it never clutters non-skill turns.
        if skill_cursor::load(&config).is_active() {
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
            if config.model.is_mistral_small_4_family() {
                let effort = if plan_set { "none" } else { "high" };
                (serde_json::json!({"reasoning_effort": effort}), None)
            } else if config.model.thinking {
                (
                    serde_json::json!({"enable_thinking": true}),
                    Some(config.model.thinking_temperature),
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
            temperature_override,
            // Never forced from the agent loop; see `state.loops.window_edit_fires`.
            cache_prompt: None,
        };
        log.llm_request(&request);

        ui.separator();

        // Reset cancel flag for this round
        cancelled.store(false, Ordering::Relaxed);

        let response = match ui
            .stream_llm(&llm_worker, model_role, request, &cancelled)
            .await
        {
            LlmOutcome::Response(r) => r,
            outcome => {
                let err_str = match outcome {
                    LlmOutcome::Error(e) => e,
                    // WorkerStopped (UiClosed never occurs headless) routes
                    // through the generic error ladder below.
                    _ => "LLM worker stopped unexpectedly".to_string(),
                };
                if err_str.contains("Interrupted") {
                    ui.status("Generation interrupted.");
                    break;
                }
                // The server rejected the request outright: prompt alone
                // exceeds the context window. Compact and resend — this is
                // the primary recovery path for compaction="lazy" (which
                // never compacts proactively), and a safety net for every
                // other strategy.
                if is_context_exceeded_error(&err_str)
                    && state.context_compact_retries
                        < context::compressor::FORCE_COMPRESS_MAX_RETRIES
                {
                    state.context_compact_retries += 1;
                    if ui
                        .pump(context::compressor::force_compress(
                            &mut messages,
                            &config,
                            &router,
                            &llm_worker,
                            tool_def_tokens,
                        ))
                        .await
                    {
                        log.llm_error("context window exceeded — compacted history, retrying");
                        ui.status("Context window exceeded — compacting and retrying.");
                        continue;
                    }
                    // Nothing could be freed — fall through to the normal
                    // error handling; retrying would fail identically.
                }
                if is_tool_call_args_cap_error(&err_str) {
                    // Our streaming assembler aborted the generation because
                    // an anchor-only tool's arguments outgrew the cap (see
                    // llm::tool_call_args_cap). Nothing was persisted; tell
                    // the model what it did and let it re-issue.
                    state.truncated_call_errors_in_a_row += 1;
                    if state.truncated_call_errors_in_a_row >= TRUNCATED_CALL_ABORT_AFTER {
                        log.llm_error(&format!(
                            "{} consecutive oversized tool calls — aborting turn",
                            state.truncated_call_errors_in_a_row
                        ));
                        ui.error(
                            "The model keeps emitting oversized tool-call arguments — giving up on this turn.",
                        );
                        had_error = true;
                        break;
                    }
                    log.llm_error(&format!(
                        "tool call aborted by the argument size cap: {err_str}"
                    ));
                    ui.status(
                        "Tool call arguments exceeded the size cap — retrying with guidance.",
                    );
                    let hint = Message::user(&format!(
                        "{err_str}. Anchor-style tools take identifiers and short expressions only — \
                         never paste code bodies into their arguments. {}",
                        truncated_tool_call_hint(config.tools.edit_mode)
                    ));
                    messages.push(hint.clone());
                    conversation_history.push(hint);
                    continue;
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
                        let scrubbed = scrub_unparseable_tool_calls(&mut messages)
                            + scrub_unparseable_tool_calls(&mut conversation_history);
                        if scrubbed > 0 {
                            log.llm_error(&format!(
                                "scrubbed {scrubbed} unparseable tool call(s) from history after repeated parse failures — retrying"
                            ));
                            ui.status("Repaired a truncated tool call left in history — retrying.");
                            continue;
                        }
                    }
                    if state.truncated_call_errors_in_a_row >= TRUNCATED_CALL_ABORT_AFTER {
                        log.llm_error(&format!(
                            "{} consecutive tool-call parse failures with nothing left to repair — aborting turn",
                            state.truncated_call_errors_in_a_row
                        ));
                        ui.error(
                            "The server keeps rejecting tool-call arguments — giving up on this turn.",
                        );
                        had_error = true;
                        break;
                    }
                    // When the prompt is sitting near the context
                    // window, the truncation is really context exhaustion
                    // (the server clamps generation to the remaining room):
                    // a hint can't fix that, compaction can.
                    if context::compressor::estimated_context_tokens(&messages, tool_def_tokens)
                        > config.model.context_window * 3 / 4
                        && state.context_compact_retries
                            < context::compressor::FORCE_COMPRESS_MAX_RETRIES
                    {
                        state.context_compact_retries += 1;
                        if ui
                            .pump(context::compressor::force_compress(
                                &mut messages,
                                &config,
                                &router,
                                &llm_worker,
                                tool_def_tokens,
                            ))
                            .await
                        {
                            log.llm_error(
                                "tool call truncated near context ceiling — compacted history, retrying",
                            );
                            ui.status(
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
                    ui.status("Previous tool call truncated — retrying with guidance.");
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
                ui.error(&format!("LLM error: {clean}"));
                ui.status(&format!(
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
            && state.context_compact_retries < context::compressor::FORCE_COMPRESS_MAX_RETRIES
        {
            state.context_compact_retries += 1;
            if ui
                .pump(context::compressor::force_compress(
                    &mut messages,
                    &config,
                    &router,
                    &llm_worker,
                    tool_def_tokens,
                ))
                .await
            {
                log.llm_error(
                    "generation truncated by context ceiling — compacted history, regenerating",
                );
                ui.status("Generation truncated by context ceiling — compacting and regenerating.");
                continue;
            }
        }

        // Get the assistant's response
        let choice = match response.choices.first() {
            Some(c) => c,
            None => {
                ui.event(UiEvent::EmptyLlmResponse);
                break;
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
            log.llm_error(&format!(
                "{truncated_calls} tool call(s) arrived with unparseable arguments (cut off by the output limit) — stubbed before persisting"
            ));
            ui.status("A tool call was cut off by the output limit — it will not be executed.");
        }
        let assistant_msg = &assistant_msg;

        ui.finish_assistant_text(assistant_msg.content.as_deref());

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
                if strict && !state.nudged_premature_exit && config.tools.plan {
                    let has_unchecked = tools::plan::has_unchecked_steps(&config);
                    let plan_exists = tools::plan::plan_exists(&config);
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
                        continue;
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
                {
                    let mut cursor = skill_cursor::load(&config);
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
                        if skill_state.exit_stops >= SKILL_EXIT_MAX_STOPS
                            && cursor.may_auto_advance()
                        {
                            cursor.mark_abandoned();
                            skill_cursor::save(&config, &cursor);
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
                            continue;
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
                                &config,
                                &llm_worker,
                                &tool_defs,
                                &perms,
                                &lsp_client,
                                &fast_revisions,
                                fast_baseline_errors,
                                &cancelled,
                                &mut state.force_compact_next_round,
                            )
                            .await
                            {
                                let m = Message::user(&note);
                                messages.push(m.clone());
                                conversation_history.push(m);
                                continue;
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
                        continue;
                    }
                }
                // Live-jobs finish-gate: finishing while background jobs run
                // abandons them (session end kills them — a deploy started
                // with background=true dies half-way). One nudge to wait or
                // kill deliberately; jobs e2e (2026-07-14) showed the model
                // fire-and-forgetting a background deploy otherwise.
                if !job_registry.is_empty() && !nudged_live_jobs {
                    nudged_live_jobs = true;
                    let nudge = Message::user(
                        "[Background job(s) still running — the task is not done. \
                         Wait for them with shell(action='wait', secs=60, check='<status command>') \
                         and verify the result, or shell(action='kill') them deliberately. \
                         Finishing now would abandon and kill them.]",
                    );
                    messages.push(nudge.clone());
                    conversation_history.push(nudge);
                    continue;
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
                let effective_check = skill_cursor::current_check_command(&config)
                    .or_else(|| config.validation.command().map(str::to_string));
                if state.gate.validation_blocks < config.validation.max_retries
                    && let Some(check_cmd) = effective_check.as_deref()
                {
                    match validation::run_check_command(&config, check_cmd).await {
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
                            if config.tools.gate_restart && !state.gate.restart_fired {
                                state.gate.restart_fired = true;
                                if let Some(ref snap) = snapshots {
                                    let guard = snap.lock();
                                    match guard.revert_to_round(0) {
                                        Ok(m) => ui.status(&format!("[gate-restart] {m}")),
                                        Err(e) => ui.status(&format!(
                                            "[gate-restart] tree revert failed: {e}"
                                        )),
                                    }
                                }
                                // Whole-tree revert changed many files outside the
                                // per-edit reindex path — resync the symbol index /
                                // repo-map to the clean baseline so the fresh agent
                                // doesn't see the reverted-away structure.
                                tools::reindex_project_incremental(&config);
                                let _ = std::fs::remove_file(config.session_path("plan.md"));
                                state.plan_ever_set = false;
                                let _ = std::fs::remove_file(config.session_path("scratchpad.md"));
                                let assembled = context::assemble(
                                    &config,
                                    message,
                                    &[],
                                    plan_only,
                                    mcp_summary.as_deref(),
                                );
                                messages = assembled.messages;
                                conversation_history.clear();
                                state.gate.validation_blocks = 0;
                                state.gate.plan_step_failures.reset();
                                ui.status(
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
                            if config.tools.gate_replan
                                && !state.gate.replan_fired
                                && !is_compile_fail
                            {
                                state.gate.replan_fired = true;
                                ui.status(
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
                            let fkey = debugger::failure_key(&output);
                            let may_fire = if config.tools.debugger_multifire {
                                state.debugger.fires < debugger::MAX_DEBUGGER_FIRES
                                    && state.debugger.last_failure.as_deref() != Some(fkey.as_str())
                            } else {
                                state.debugger.fires == 0
                            };
                            if (config.tools.reactive_debugger || config.tools.debugger_judge)
                                && may_fire
                                && state.gate.validation_blocks >= debugger::DEBUGGER_TRIGGER_BLOCKS
                            {
                                state.debugger.fires += 1;
                                state.debugger.last_failure = Some(fkey);
                                ui.status(
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
                                    debugger::DebuggerVerdict::Scrap
                                        if !state.gate.restart_fired =>
                                    {
                                        state.gate.restart_fired = true;
                                        if let Some(ref snap) = snapshots {
                                            let guard = snap.lock();
                                            match guard.revert_to_round(0) {
                                                Ok(m) => ui.status(&format!(
                                                    "[debugger-judge] SCRAP — {m}"
                                                )),
                                                Err(e) => ui.status(&format!(
                                                    "[debugger-judge] SCRAP — tree revert failed: {e}"
                                                )),
                                            }
                                        }
                                        // Resync the symbol index / repo-map to the clean
                                        // baseline after the whole-tree revert (the
                                        // per-edit reindex path doesn't cover it).
                                        tools::reindex_project_incremental(&config);
                                        let _ =
                                            std::fs::remove_file(config.session_path("plan.md"));
                                        let _ = std::fs::remove_file(
                                            config.session_path("scratchpad.md"),
                                        );
                                        state.plan_ever_set = false;
                                        let assembled = context::assemble(
                                            &config,
                                            message,
                                            &[],
                                            plan_only,
                                            mcp_summary.as_deref(),
                                        );
                                        messages = assembled.messages;
                                        conversation_history.clear();
                                        state.gate.validation_blocks = 0;
                                        state.gate.plan_step_failures.reset();
                                        ui.status(
                                            "[debugger-judge] scrapped the stuck state — clean baseline + fresh context; restarting from scratch.",
                                        );
                                        continue;
                                    }
                                    debugger::DebuggerVerdict::Scrap => {
                                        Message::user(debugger::SCRAP_ALREADY_RESET_MSG)
                                    }
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
                                        let output_note =
                                            validation::gate_failure_note(&config, &output);
                                        Message::user(&debugger::build_gate_report_message(
                                            &body,
                                            &output_note,
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
                                && state.gate.context_resets < spiral::MAX_GATE_RESETS
                                && state.gate.validation_blocks >= spiral::GATE_RESET_AFTER_BLOCKS
                            {
                                state.gate.context_resets += 1;
                                state.gate.validation_blocks = 0; // fresh gate budget for the clean restart
                                let fresh = spiral::build_gate_reset_prompt(message, &output);
                                let assembled = context::assemble(
                                    &config,
                                    &fresh,
                                    &[],
                                    plan_only,
                                    mcp_summary.as_deref(),
                                );
                                messages = assembled.messages;
                                ui.status(
                                    "Gate context-reset — fresh start (history cleared, files kept).",
                                );
                                log.tool_debug(
                                    "agent",
                                    "gate context-reset: re-assembled clean context after repeated gate blocks",
                                );
                                continue;
                            }

                            let msg = Message::user(
                                &validation::build_verification_failed_message(&output),
                            );
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

        // Active skill step, folded into each loop key below so the SAME tool
        // call on different steps (notably skill(done), byte-identical on every
        // step) yields distinct keys — legitimately advancing through steps is
        // not misread as a repeat, while a within-step rut (constant tag) is.
        let active_step_tag = skill_cursor::current_step_tag(&config);

        for tc in &tool_calls {
            let args: serde_json::Value = match serde_json::from_str(&tc.function.arguments) {
                Ok(v) => v,
                Err(e) => {
                    // Unreachable after sanitize_truncated_tool_calls above,
                    // kept as a belt-and-braces path. Never echo the raw
                    // arguments back: that is the flood we just refused to
                    // persist.
                    let result_msg = Message::tool_result(
                        &tc.id,
                        &format!("Invalid JSON in tool arguments: {e}"),
                    );
                    messages.push(result_msg.clone());
                    conversation_history.push(result_msg);
                    ui.tool_result(&tc.function.name, false, "invalid JSON args");
                    continue;
                }
            };
            if let Some(info) = truncated_args_info(&args) {
                // A call stubbed by sanitize_truncated_tool_calls: the
                // arguments were cut off by the output limit, so there is
                // nothing to execute. Answer with guidance, not a run.
                let result_msg = Message::tool_result(
                    &tc.id,
                    &format!(
                        "{}\n\n{}",
                        truncated_args_tool_result(&tc.function.name, &info),
                        truncated_tool_call_hint(config.tools.edit_mode)
                    ),
                );
                messages.push(result_msg.clone());
                conversation_history.push(result_msg);
                log.tool_debug(
                    "agent",
                    &format!(
                        "{} call skipped: arguments truncated after {} chars",
                        tc.function.name, info.original_chars
                    ),
                );
                ui.tool_result(
                    &tc.function.name,
                    false,
                    "arguments cut off by the output limit — not executed",
                );
                continue;
            }

            let args_summary = summarize_args(&tc.function.name, &args);

            // Detect tool call loops: identical calls repeated consecutively
            // (period-1), or the SAME two calls alternating (period-2 — the
            // edit↔revert oscillation that the streak counter is blind to
            // because every alternation resets it).
            let call_key =
                loop_call_key_tagged(&tc.function.name, &args, active_step_tag.as_deref());
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
            let cycle = cycle_period(&state.loops.recent_call_keys);
            if state.loops.same_call_streak >= 3 || cycle.is_some() {
                // Cycle-only detection (not also a plain streak). Captured
                // before any state resets below so messaging stays accurate.
                let cycle_only = cycle.filter(|_| state.loops.same_call_streak < 3);
                // A cycle is harmful if ANY member mutates (the classic case
                // is edit↔revert — both mutate; edit↔read still re-applies
                // the same broken edit).
                let mutating = if let Some(period) = cycle_only {
                    let tail = &state.loops.recent_call_keys
                        [state.loops.recent_call_keys.len().saturating_sub(period)..];
                    tail.iter().any(|k| key_is_mutating(k))
                } else {
                    is_mutating_call(&tc.function.name, &args)
                };
                log.loop_detected(
                    &tc.function.name,
                    &args_summary,
                    state.loops.same_call_streak as usize,
                );

                // Polling a status command while a background job runs is
                // the unpaced form of monitoring — redirect to the paced one,
                // naming the polled command as the check probe. Fires BEFORE
                // the mutating classification: shell commands classify as
                // mutating, which routed the jobs e2e's status-poll loop into
                // the turn-stopping path (2026-07-14, stuck scenario died 11s
                // into a monitoring task). Capped so a genuine runaway still
                // escalates normally.
                if ((tc.function.name == "shell" && args["action"].as_str() == Some("run"))
                    || (tc.function.name == "file" && args["action"].as_str() == Some("shell")))
                    && !job_registry.is_empty()
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
                    continue;
                }

                // Read-only repetition: harmless per call, just wasted tokens.
                // First detection: polite nudge inline, let the for-loop
                // continue. Re-detection: escalate — the nudge can't reach a
                // cache-numerics rut, so force a compaction next round.
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
                    ui.status(&format!(
                        "Repeated read: {}({}) — {}, continuing",
                        tc.function.name,
                        args_summary,
                        if escalate {
                            "nudge failed, forcing compaction next round"
                        } else {
                            "nudge sent"
                        }
                    ));
                    state.loops.last_call_key = None;
                    state.loops.same_call_streak = 0;
                    state.loops.recent_call_keys.clear();
                    continue;
                }

                let hint = if let Some(period) = cycle_only {
                    cycle_loop_hint(period)
                } else {
                    loop_detected_hint(config.tools.edit_mode).to_string()
                };
                let result_msg = Message::tool_result(&tc.id, &hint);
                messages.push(result_msg.clone());
                conversation_history.push(result_msg);

                // First mutating loop in this turn: surface the hint, reset
                // the streak, and let the model try a different approach.
                // Subsequent loops mean the recovery itself spiraled —
                // abort for real.
                if state.loops.recoveries == 0 {
                    state.loops.recoveries += 1;
                    state.loops.last_call_key = None;
                    state.loops.same_call_streak = 0;
                    state.loops.recent_call_keys.clear();
                    ui.error(&format!(
                        "Loop detected: {}({}) {} — surfacing a hint, giving the model one more round",
                        tc.function.name,
                        args_summary,
                        if let Some(period) = cycle_only {
                            format!(
                                "cycling through the same {period} calls (period-{period} cycle)"
                            )
                        } else {
                            "repeated 3 times".to_string()
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
                // Without a gate: original behavior — stop the turn. An active
                // skill step's completion check counts as the gate here too.
                let effective_check = skill_cursor::current_check_command(&config)
                    .or_else(|| config.validation.command().map(str::to_string));
                // Failure text to route through the recovery ladder, if we
                // should recover at all, in priority order:
                //  - the LOOPING COMMAND ITSELF keeps FAILING (its real error) —
                //    the missing trigger: a read-only per-step check can PASS
                //    while `pack package create` returns a lint error, so the
                //    command's own failure never surfaced. A fresh-context
                //    debugger fixes exactly this (probe: 10/10 on the flavor bug);
                //  - else a check that FAILS → its output;
                //  - else a skill step IS active (Fix 2) → synthesize the
                //    stuck state so a non-checkable (or check-passing but
                //    still looping) step reaches the debugger instead of the
                //    hard stop below, which assumes no cursor.
                let recover_output: Option<String> = if let Some((k, out)) = &last_tool_failure
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
                    failing_job_output(&tc.function.name, &args, &failed_job_commands)
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
                        match validation::run_check_command(&config, cmd).await {
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
                        let cursor = skill_cursor::load(&config);
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
                if let Some(output) = recover_output
                    && state.gate.validation_blocks < config.validation.max_retries
                {
                    ui.error(&format!(
                        "Loop detected again ({}({})) — routing through the recovery ladder instead of stopping",
                        tc.function.name, args_summary
                    ));
                    {
                        state.gate.validation_blocks += 1;
                        // Fresh recovery budget for the rounds the ladder grants.
                        state.loops.recoveries = 0;
                        state.loops.last_call_key = None;
                        state.loops.same_call_streak = 0;
                        state.loops.recent_call_keys.clear();

                        let fkey = debugger::failure_key(&output);
                        let may_fire = if config.tools.debugger_multifire {
                            state.debugger.fires < debugger::MAX_DEBUGGER_FIRES
                                && state.debugger.last_failure.as_deref() != Some(fkey.as_str())
                        } else {
                            state.debugger.fires == 0
                        };
                        if (config.tools.reactive_debugger || config.tools.debugger_judge)
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
                                debugger::DebuggerVerdict::Scrap if !state.gate.restart_fired => {
                                    state.gate.restart_fired = true;
                                    if let Some(ref snap) = snapshots {
                                        let guard = snap.lock();
                                        match guard.revert_to_round(0) {
                                            Ok(m) => {
                                                ui.status(&format!("[debugger-judge] SCRAP — {m}"))
                                            }
                                            Err(e) => ui.status(&format!(
                                                "[debugger-judge] SCRAP — tree revert failed: {e}"
                                            )),
                                        }
                                    }
                                    tools::reindex_project_incremental(&config);
                                    let _ = std::fs::remove_file(config.session_path("plan.md"));
                                    state.plan_ever_set = false;
                                    let _ =
                                        std::fs::remove_file(config.session_path("scratchpad.md"));
                                    let assembled = context::assemble(
                                        &config,
                                        message,
                                        &[],
                                        plan_only,
                                        mcp_summary.as_deref(),
                                    );
                                    messages = assembled.messages;
                                    conversation_history.clear();
                                    state.gate.validation_blocks = 0;
                                    state.gate.plan_step_failures.reset();
                                    ui.status(
                                        "[debugger-judge] scrapped the stuck state — clean baseline + fresh context; restarting from scratch.",
                                    );
                                    continue 'round;
                                }
                                debugger::DebuggerVerdict::Scrap => {
                                    Message::user(debugger::SCRAP_ALREADY_RESET_MSG)
                                }
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
                                    let output_note =
                                        validation::gate_failure_note(&config, &output);
                                    Message::user(&debugger::build_gate_report_message(
                                        &body,
                                        &output_note,
                                    ))
                                }
                            };
                            messages.push(msg.clone());
                            conversation_history.push(msg);
                            continue 'round;
                        }

                        let msg = Message::user(&validation::build_loop_abort_message(&output));
                        messages.push(msg.clone());
                        conversation_history.push(msg);
                        continue 'round;
                    }
                }
                // No check failed and no skill cursor is active — the loop is
                // on something the recovery ladder can't act on; stop the turn.
                ui.error(&format!(
                    "Loop detected again ({}({})) after the recovery hint — stopping this turn",
                    tc.function.name, args_summary
                ));
                had_error = true;
                break;
            }

            log.tool_call_detail(&tc.function.name, &args);
            ui.tool_call_started(&tc.function.name, &args_summary);

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
                ui.tool_result(&tc.function.name, false, "blocked in plan mode");
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
                ui.tool_result(&tc.function.name, false, "blocked: no plan");
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
                let cancelled_for_job = cancelled.clone();
                let log = log.clone();
                ui.await_tool_job(
                    tool_pool.submit(move || {
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
                                    Some(cancelled_for_job.as_ref()),
                                    Some(log.as_ref()),
                                )
                                .await
                            })
                            .map_err(|e| format!("edit_file error: {e}"))
                    }),
                    "edit_file",
                    &cancelled,
                )
                .await
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
                let cancelled_for_job = cancelled.clone();
                ui.await_tool_job(
                    tool_pool.submit(move || {
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
                                    Some(cancelled_for_job.as_ref()),
                                )
                                .await
                            })
                            .map_err(|e| format!("refactor error: {e}"))
                    }),
                    "refactor",
                    &cancelled,
                )
                .await
            } else if (tc.function.name == "shell" && args["action"].as_str() == Some("run"))
                || (tc.function.name == "file" && file_action == "shell")
            {
                if args["background"].as_bool() == Some(true) {
                    // Explicit background start: the sanctioned form of the
                    // model's "cmd & echo $! > .pid" instinct — registered,
                    // output-captured, managed via the jobs tool.
                    tools::jobs::start_background(&args, &config, job_registry.as_ref())
                } else {
                    ui.await_shell_job(
                        tool_pool.submit_shell(args.clone(), config.clone(), cancelled.clone()),
                        &cancelled,
                        headless.then_some(job_registry.as_ref()),
                    )
                    .await
                }
            } else if tc.function.name == "shell" {
                tools::jobs::execute(
                    &args,
                    &config,
                    perms.as_ref(),
                    job_registry.as_ref(),
                    Some(cancelled.as_ref()),
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
                ui.await_tool_job(
                    tool_pool.submit(move || {
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
                    }),
                    &tc.function.name,
                    &cancelled,
                )
                .await
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
                    ui.status(&format!("spawning {} subagents...", tasks.len()));
                    let outputs = ui
                        .drive_subagents(
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
                        )
                        .await;
                    let combined = crate::cli::commands::agent::subagent::format_outputs(outputs);
                    crate::tools::ToolResult::ok(combined)
                }
            } else if tc.function.name == "skill" {
                // Harness-owned step cursor: the model signals it finished the
                // current [SKILL STEP]; advance and announce the next one (its
                // instructions arrive via [SKILL STEP] on the next round,
                // distilled in round maintenance).
                let action = args["action"].as_str().unwrap_or("done");
                if action == "done" {
                    let mut cursor = skill_cursor::load(&config);
                    match cursor
                        .current()
                        .map(|(sk, st)| (sk.to_string(), st.name.clone()))
                    {
                        Some((skill, finished)) => {
                            // Gate on the step's completion check with a
                            // one-retry override: the FIRST skill(done) runs the
                            // check and, on failure, is refused with the check
                            // output; a SECOND consecutive skill(done) advances
                            // anyway (the model overrules a possibly-wrong check —
                            // the probe measured ~6% of checks over-specify and
                            // false-fail, so the model needs an escape hatch).
                            let check = cursor.current_check().map(str::to_string);
                            let unchecked = check.is_none();
                            let attempt = cursor.note_done_attempt();
                            let blocked = match check.filter(|_| attempt < 2) {
                                Some(cmd) => {
                                    match validation::run_check_command(&config, &cmd).await {
                                        validation::CheckOutcome::Fail(out) => Some(out),
                                        _ => None,
                                    }
                                }
                                None => None,
                            };
                            if let Some(out) = blocked {
                                // Persist the incremented attempt; do NOT advance.
                                skill_cursor::save(&config, &cursor);
                                crate::tools::ToolResult::err(format!(
                                    "Step '{finished}' does not meet its DONE WHEN yet — the \
                                     completion check failed:\n{out}\nFix it and call \
                                     skill(action='done') again. If you are certain the step is \
                                     actually complete and the check is wrong, call it once more \
                                     to override."
                                ))
                            } else {
                                // LOG-ONLY judge-veto observation (2026-09-01
                                // e2e: 4 unchecked steps advanced right over a
                                // standing not-done verdict). An UNCHECKED
                                // step leans on the judge alone, so when one
                                // gets a done with a standing verdict and no
                                // mutating edit since (the verdict can't be
                                // stale), record what an enforced veto would
                                // have refused. Enforcement waits on measured
                                // live judge quality — verdicts have been
                                // observed zero times in the field.
                                if unchecked
                                    && let Some((key, reason, at_edits)) =
                                        &skill_state.last_judge_block
                                    && *key == format!("{skill}::{finished}")
                                    && *at_edits == skill_state.edits_total
                                {
                                    ui.status(&format!(
                                        "[skills] judge-veto (log-only) on '{finished}': {reason}"
                                    ));
                                }
                                skill_state.last_judge_block = None;
                                // prepare_step guarantees the parked step is
                                // distilled, so a `done` here is always a
                                // verdict on something the model was actually
                                // shown. If that ever stops holding the step
                                // was invisible this round and the verdict is
                                // meaningless — say so rather than let it pass
                                // silently, as it did before the loop in
                                // prepare_step closed that window.
                                if cursor.cached().is_none() {
                                    ui.status(&format!(
                                        "[skills] warning: done on '{finished}' while undistilled \
                                         — the step was never shown"
                                    ));
                                }
                                // A skill's LAST step often exists only to hand
                                // off (build → integrate). Resolve that BEFORE
                                // mark_done pops the frame: once popped there is
                                // no cursor left to descend from, and the run
                                // ends the build → integrate → validate lifecycle
                                // a phase early while reporting success. Live
                                // e2e: `done` on the build skill's
                                // EnterIntegrationPhase step silently dropped the
                                // Package CR, networking, Postgres, IDP,
                                // monitoring and validation steps.
                                let installed: Vec<String> =
                                    crate::skills::discover(&config.project_root)
                                        .into_iter()
                                        .map(|e| e.name)
                                        .collect();
                                let handed_off = match resolve_handoff(
                                    &mut cursor,
                                    &installed,
                                    &llm_worker,
                                    &cancelled,
                                )
                                .await
                                {
                                    Some(next) => {
                                        descend_into_skill(
                                            &mut cursor,
                                            &next,
                                            &config,
                                            &llm_worker,
                                            &cancelled,
                                        )
                                        .await
                                    }
                                    None => false,
                                };
                                // descend() already consumed the invoking step —
                                // marking done as well would skip the sub-skill's
                                // first step.
                                if !handed_off {
                                    cursor.mark_done();
                                }
                                report_cursor_gaps(&cursor);
                                skill_cursor::save(&config, &cursor);
                                let msg = match cursor.current() {
                                    Some((_, next)) => format!(
                                        "Step '{finished}' marked done. Next step: '{}'. Its full \
                                         instructions will appear under [SKILL STEP] — follow them \
                                         exactly.",
                                        next.name
                                    ),
                                    // The frame popped. That only means every
                                    // step is complete when none were abandoned
                                    // on the way — say which are outstanding
                                    // rather than inviting a finish over them.
                                    None if !cursor.dropped_unfinished().is_empty() => format!(
                                        "Step '{finished}' marked done, but the {skill} skill ends \
                                         with unfinished steps: {}. Those were never completed — \
                                         go back and finish them before you stop.",
                                        cursor.dropped_unfinished().join(", ")
                                    ),
                                    None => format!(
                                        "Step '{finished}' marked done. All {skill} skill steps are \
                                         complete — finish the task."
                                    ),
                                };
                                crate::tools::ToolResult::ok(msg)
                            }
                        }
                        None => crate::tools::ToolResult::err(
                            "No active skill step to complete.".into(),
                        ),
                    }
                } else {
                    crate::tools::ToolResult::err(format!(
                        "Unknown skill action '{action}'. Use action='done'."
                    ))
                }
            } else {
                let tool_name = tc.function.name.clone();
                let args = args.clone();
                let config = config.clone();
                let perms = perms.clone();
                let lsp = lsp_client.clone();
                ui.await_tool_job(
                    tool_pool.submit(move || {
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
                    }),
                    &tc.function.name,
                    &cancelled,
                )
                .await
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
            ui.tool_result(&tc.function.name, result.success, first_line);
            ui.store_tool_result(&tc.function.name, &result.content);

            // Remember the last failing tool call keyed by its loop key, so a
            // loop on a *failing* command can hand the real error to the
            // debugger (see the loop-recovery ladder). Shell exit≠0 sets
            // success=false, so this catches the `pack package create` case.
            // Keyed by the SAME tagged `call_key` the ladder compares against
            // — an untagged key here never matches during skill runs.
            if !result.success {
                last_tool_failure = Some((
                    call_key.clone(),
                    crate::truncate_chars(result.content.trim(), 2000),
                ));
            }
            // Background-job bookkeeping is per BANNER, not per result: a
            // FAILED deploy surfaces later in a wait/status result (possibly
            // aggregated with other jobs' banners), so the wrapper's ok/err
            // can't attribute verdicts to commands.
            note_job_banners(&result.content, &mut failed_job_commands);

            if result.success && tc.function.name == "plan" {
                state.successful_edits_since_plan_update = 0;
            }

            // A successful file write means code changed — reset trackers.
            if result.success && is_file_write(tc.function.name.as_str()) {
                state.loops.last_call_key = None;
                state.loops.same_call_streak = 0;
                state.calls_since_last_edit = 0;
                if strict && config.tools.plan {
                    if tools::plan::plan_exists(&config) {
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
                all_prunable_failures = false;
            } else {
                prunable_errors.push(result.content.clone());
            }

            stuck_tracker.on_tool(&tc.function.name, &args, result.success, &result.content);
            if result.success
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
            if tc.function.name == "plan"
                && args.get("action").and_then(|a| a.as_str()) == Some("check")
            {
                if result.success {
                    state.gate.plan_step_failures.reset();
                } else if let Some(step) = args.get("step").and_then(|s| s.as_u64()) {
                    state.gate.plan_step_failures.note(step);

                    let fkey = debugger::failure_key(&result.content);
                    let may_fire = if config.tools.debugger_multifire {
                        state.debugger.fires < debugger::MAX_DEBUGGER_FIRES
                            && state.debugger.last_failure.as_deref() != Some(fkey.as_str())
                    } else {
                        state.debugger.fires == 0
                    };
                    if config.tools.plan_gate_debugger
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
                            debugger::DebuggerVerdict::Scrap if !state.gate.restart_fired => {
                                state.gate.restart_fired = true;
                                if let Some(ref snap) = snapshots {
                                    let guard = snap.lock();
                                    match guard.revert_to_round(0) {
                                        Ok(m) => {
                                            ui.status(&format!("[debugger-judge] SCRAP — {m}"))
                                        }
                                        Err(e) => ui.status(&format!(
                                            "[debugger-judge] SCRAP — tree revert failed: {e}"
                                        )),
                                    }
                                }
                                tools::reindex_project_incremental(&config);
                                let _ = std::fs::remove_file(config.session_path("plan.md"));
                                state.plan_ever_set = false;
                                let _ = std::fs::remove_file(config.session_path("scratchpad.md"));
                                let assembled = context::assemble(
                                    &config,
                                    message,
                                    &[],
                                    plan_only,
                                    mcp_summary.as_deref(),
                                );
                                messages = assembled.messages;
                                conversation_history.clear();
                                state.gate.validation_blocks = 0;
                                state.gate.plan_step_failures.reset();
                                ui.status(
                                    "[debugger-judge] scrapped the stuck state — clean baseline + fresh context; restarting from scratch.",
                                );
                                continue 'round;
                            }
                            debugger::DebuggerVerdict::Scrap => {
                                Message::user(debugger::SCRAP_ALREADY_RESET_MSG)
                            }
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
                                    validation::gate_failure_note(&config, &result.content);
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
            if config.tools.spiral_reset
                && result.success
                && tc.function.name == "revert"
                && config.tools.edit_mode == EditMode::Fast
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
                    let tried = fast_revisions
                        .as_deref()
                        .map(|r| spiral::tried_edit_labels(r, path, 4))
                        .unwrap_or_default();
                    let reset = Message::user(&spiral::build_reset_message(path, n, &tried));
                    messages.push(reset.clone());
                    conversation_history.push(reset);
                    ui.status("Spiral detected (revert-loop) — reset + replan injected.");
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

        // `tools.stuck_check`: T2c frozen-signature fire → append the stuck/
        // done note to the round's last tool result (the placement the
        // warm-replay probes validated; a trailing user message was not what
        // was tested). Runs AFTER history pruning so the note can't land on
        // a tool result that was just truncated away.
        if config.tools.stuck_check
            && let Some(kind) =
                stuck_tracker.check_fire(round, session_start.elapsed().as_secs_f64())
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
                    stuck_tracker.frozen_rounds()
                );
                step_judge_escalation(
                    message,
                    &trigger,
                    &config,
                    &llm_worker,
                    &tool_defs,
                    &perms,
                    &lsp_client,
                    &fast_revisions,
                    fast_baseline_errors,
                    &cancelled,
                    &mut state.force_compact_next_round,
                )
                .await
            } else {
                None
            };
            let plan_done =
                tools::plan::plan_exists(&config) && !tools::plan::has_unchecked_steps(&config);
            let note = if let Some(esc) = escalation {
                esc
            } else if kind == stuck_check::StuckKind::Green && plan_done {
                stuck_check::done_note()
            } else {
                let first_unchecked = tools::plan::parsed_steps(&config)
                    .iter()
                    .find(|(checked, _, _)| !checked)
                    .and_then(|(_, n, _)| *n);
                stuck_check::stuck_note(
                    stuck_tracker.frozen_rounds(),
                    stuck_tracker.frozen_minutes(),
                    stuck_tracker.looping_read_path(),
                    first_unchecked,
                )
            };
            let mut appended = false;
            for msgs in [&mut messages, &mut conversation_history] {
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
                stuck_tracker.frozen_rounds(),
            ));
            log.tool_debug(
                "agent",
                &format!(
                    "stuck-check fired: kind={kind:?} plan_done={plan_done} frozen_rounds={} note={}",
                    stuck_tracker.frozen_rounds(),
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
        if strict && round >= 12 && !state.nudged_no_plan && !tools::plan::plan_exists(&config) {
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
