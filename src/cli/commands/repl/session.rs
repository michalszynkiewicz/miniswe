//! REPL session entry: terminal setup/teardown and the outer `run()`
//! that drives turns until the user exits.

use super::*;

/// True while a `ReplTerminalGuard` owns the terminal (gates the panic hook).
static TERMINAL_ACTIVE: AtomicBool = AtomicBool::new(false);

struct ReplTerminalGuard {
    /// Held for its lifetime; `drop` restores stderr explicitly (and
    /// idempotently) before leaving the alternate screen.
    _stderr: Option<StderrRedirect>,
}

impl ReplTerminalGuard {
    /// Enter the alternate screen and point fd 2 at `stderr_log`, so stray
    /// `eprintln!`s / subprocess stderr cannot corrupt the frame.
    fn enter(stderr_log: &std::path::Path) -> Result<Self> {
        terminal::enable_raw_mode()?;
        io::stdout().execute(EnterAlternateScreen)?;
        let stderr = StderrRedirect::to_file(stderr_log).ok();
        TERMINAL_ACTIVE.store(true, Ordering::SeqCst);
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if TERMINAL_ACTIVE.swap(false, Ordering::SeqCst) {
                restore_stderr();
                let _ = terminal::disable_raw_mode();
                let _ = io::stdout().execute(LeaveAlternateScreen);
            }
            previous(info);
        }));
        Ok(Self { _stderr: stderr })
    }
}

impl Drop for ReplTerminalGuard {
    fn drop(&mut self) {
        // Restore stderr BEFORE leaving the alternate screen so anything
        // printed after exit reaches the terminal.
        restore_stderr();
        if TERMINAL_ACTIVE.swap(false, Ordering::SeqCst) {
            let _ = terminal::disable_raw_mode();
            let _ = io::stdout().execute(LeaveAlternateScreen);
        }
    }
}

/// Where fd 2 goes while the TUI is up: the session log if there is one,
/// else `.buzztr/logs/stderr.log`.
fn stderr_log_path(config: &Config, log: &SessionLog) -> std::path::PathBuf {
    let p = log.path();
    if p.as_os_str().is_empty() {
        config.buzztr_path("logs/stderr.log")
    } else {
        p.clone()
    }
}

/// Run the interactive REPL with TUI.
pub async fn run(mut config: Config, headless: bool, continue_session: bool) -> Result<()> {
    let log = Arc::new(SessionLog::new(&config));

    let router = Arc::new(ModelRouter::new(&config));
    // Fail fast on a missing hosted-provider API key, before the terminal
    // enters raw mode.
    router.check_credentials()?;
    // Probe server for the actual model identity and context window (see
    // run.rs for rationale).
    let probe = router.probe_default().await.ok();
    config.model.probed_model = probe.as_ref().map(|p| p.model.clone());
    let model_label = probe
        .as_ref()
        .map(|p| crate::llm::router::display_model_id(&p.model).to_string());
    config.model.probed_context_window = probe.and_then(|p| p.context_window);
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
        // Devstral carve-out removed — see run.rs for the rationale.
        disabled.push("edit_file");
        // Fast mode keeps `edit_file` available alongside the
        // primitives — see run.rs for rationale.
        tool_defs.retain(|t| !disabled.contains(&t.function.name.as_str()));
        if config.tools.edit_mode == EditMode::Fast {
            tool_defs.extend(tools::fast_mode_tool_definitions());
        }
        // Only expose spawn_agents when concurrency makes it useful (see run.rs).
        if config.runtime.llm_concurrency > 1 {
            tool_defs.push(tools::definitions::spawn_agents_tool_definition());
        }
        // Background jobs: explicit file(shell background=true) start +
        // jobs(wait/status/kill) management. Session-scoped registry so a
        // server started in one turn is manageable in later turns.
        tool_defs.push(tools::definitions::shell_tool_definition());
    }
    let job_registry = Arc::new(tools::jobs::JobRegistry::default());

    // Spawn LSP client (non-blocking)
    let lsp_client: Option<Arc<LspClient>> = if config.lsp.enabled {
        match LspClient::spawn(config.project_root.clone()).await {
            Ok(client) => Some(Arc::new(client)),
            Err(_) => None,
        }
    } else {
        None
    };

    // Fast-mode state: per-file revisions + project-wide LSP baseline.
    // Same structure as the one in run.rs — see its comment for rationale.
    let fast_revisions: Option<Arc<tools::RevisionStore>> =
        if config.tools.edit_mode == EditMode::Fast {
            let buzztr_dir = config.buzztr_path("revisions");
            tools::RevisionStore::new(&buzztr_dir).ok().map(Arc::new)
        } else {
            None
        };
    let fast_baseline_errors: usize = if config.tools.edit_mode == EditMode::Fast {
        tools::fast::project_error_count(lsp_client.as_deref()).await
    } else {
        0
    };

    // Snapshot manager for whole-tree revert support (SCRAP restart,
    // revert-to-green). Unlike run.rs (where one session IS one task, so a
    // single session-scoped instance is correct), REPL is a persistent
    // multi-turn surface — (re-)initialized fresh at the start of every turn
    // below, right before run_agent_turn, so SCRAP's revert-to-round-0 only
    // ever reverts the CURRENT turn's changes, never prior turns' work.
    // `SnapshotManager::init` wipes and recreates the shadow-git repo each
    // call, so this is just a relocation, not new plumbing. No instance
    // exists yet before the first turn runs — always overwritten before use.
    #[allow(unused_assignments)]
    let mut snapshots: Option<Arc<Mutex<tools::snapshots::SnapshotManager>>> = None;

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

    // Initialize MCP
    let mcp_config = McpConfig::load(&config.project_root)?;
    let mcp_registry = if mcp_config.has_servers() {
        let cache_dir = config.buzztr_path("mcp");
        match McpRegistry::connect(&mcp_config, &cache_dir) {
            Ok(registry) => {
                if registry.has_servers() {
                    tool_defs.push(tools::definitions::mcp_tool_definition());
                }
                Some(Arc::new(Mutex::new(registry)))
            }
            Err(_) => None,
        }
    } else {
        None
    };

    let mcp_summary = mcp_registry
        .as_ref()
        .and_then(|r| r.lock().context_summary());

    // Token budget for compression decisions. Tool definitions are a fixed
    // overhead per request, so compute once.
    let tool_def_tokens =
        context::estimate_tokens(&serde_json::to_string(&tool_defs).unwrap_or_default());

    // Set up terminal
    let _terminal_guard = ReplTerminalGuard::enter(&stderr_log_path(&config, &log))?;
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;

    // Set up app state
    let mut app = App::new();
    app.model_label = model_label;
    let history_file = config.buzztr_path("sessions/repl_history.txt");
    app.load_history(&history_file);

    // Welcome message — probe what the server is actually serving rather
    // than parroting config.toml, which can disagree with reality when a
    // llama-swap/llama-cpp in front of the endpoint is loading a
    // different gguf than named in config.
    for line in router.startup_summary().await {
        app.push_output(&format!("buzztr — {line}"), LineStyle::Status);
    }
    if let Some(ref mcp) = mcp_registry {
        let guard = mcp.lock();
        if guard.has_servers() {
            app.push_output(
                &format!(
                    "MCP: {} servers, {} tools",
                    guard.servers.len(),
                    guard.tool_count()
                ),
                LineStyle::Status,
            );
        }
    }
    app.push_output(
        "Type your message. Ctrl+O: details, Ctrl+C: interrupt, Ctrl+D: quit",
        LineStyle::Status,
    );
    app.push_output(
        "────────────────────────────────────────────────",
        LineStyle::Separator,
    );

    // Event channel
    let (tx, mut rx) = mpsc::unbounded_channel::<AppEvent>();
    perms.set_prompt_event_tx(tx.clone());

    // Cancellation flag for LLM
    let cancelled = Arc::new(AtomicBool::new(false));

    // Spawn keyboard reader (passes cancel flag for direct Ctrl+C handling)
    event::spawn_key_reader(tx.clone(), cancelled.clone());

    let mut conversation_history: Vec<Message> = Vec::new();

    // Main event loop
    loop {
        // Render
        terminal
            .draw(|frame| ui::draw(frame, &app))
            .map_err(io::Error::other)?;

        // Wait for next event
        let evt = match rx.recv().await {
            Some(e) => e,
            None => break,
        };

        match evt {
            AppEvent::Tick => {
                // Just triggers a re-render for spinner animation
            }

            AppEvent::Key(key) => {
                match app.mode {
                    AppMode::Detail => {
                        // In detail view: Esc, Ctrl+O, or q closes it;
                        // arrows / PgUp / PgDn / Home / End scroll.
                        if key.code == KeyCode::Esc
                            || event::is_ctrl_o(&key)
                            || key.code == KeyCode::Char('q')
                        {
                            app.close_detail();
                        } else {
                            let size = terminal.size()?;
                            let area = ratatui::layout::Rect::new(0, 0, size.width, size.height);
                            let max = ui::detail_max_scroll(&app, area);
                            let page = size.height.saturating_sub(2).saturating_sub(1).max(1);
                            // Clamp first so End then Up moves at once.
                            let cur = app.detail_scroll.min(max);
                            app.detail_scroll = match key.code {
                                KeyCode::Up => cur.saturating_sub(1),
                                KeyCode::Down => cur.saturating_add(1),
                                KeyCode::PageUp => cur.saturating_sub(page),
                                KeyCode::PageDown => cur.saturating_add(page),
                                KeyCode::Home => 0,
                                KeyCode::End => max,
                                _ => cur,
                            }
                            .min(max);
                        }
                    }
                    AppMode::Normal => {
                        if event::is_ctrl_d(&key) {
                            break;
                        }

                        if event::is_ctrl_c(&key) {
                            if app.is_thinking {
                                cancelled.store(true, Ordering::Relaxed);
                                app.note_interrupt();
                                app.is_thinking = false;
                            }
                            continue;
                        }

                        if event::is_ctrl_o(&key) {
                            app.open_detail();
                            continue;
                        }

                        if app.is_thinking {
                            // Ignore input while LLM is generating
                            continue;
                        }

                        match key.code {
                            KeyCode::Enter | KeyCode::Char('\n') | KeyCode::Char('\r') => {
                                let input = app.submit_input();
                                if input.is_empty() {
                                    continue;
                                }

                                // Handle commands
                                if is_quit_command(&input) {
                                    // Exiting is not worth recalling with Up.
                                    app.forget_last_history();
                                    break;
                                }

                                // Built-in commands echo like a normal turn.
                                if is_builtin_command(&input) {
                                    app.push_output(&format!("you> {input}"), LineStyle::Normal);
                                }

                                if input == "/clear" || input == "/new" {
                                    conversation_history.clear();
                                    if input == "/new" {
                                        let _ = std::fs::remove_file(
                                            config.session_path("scratchpad.md"),
                                        );
                                        let _ =
                                            std::fs::remove_file(config.session_path("plan.md"));
                                        app.push_output(
                                            "Cleared history, scratchpad, and plan.",
                                            LineStyle::Status,
                                        );
                                    } else {
                                        app.push_output(
                                            "Cleared conversation history.",
                                            LineStyle::Status,
                                        );
                                    }
                                    continue;
                                }

                                if input == "/help" {
                                    for line in help_lines() {
                                        app.push_output(line, LineStyle::Status);
                                    }
                                    continue;
                                }

                                if input == "/skills" || input == "/skills list" {
                                    let entries = crate::skills::discover(&config.project_root);
                                    if entries.is_empty() {
                                        app.push_output(
                                            "No skills found in .ai/skills/",
                                            LineStyle::Status,
                                        );
                                    } else {
                                        for entry in &entries {
                                            if let Ok(skill) = crate::skills::load(&entry.path) {
                                                app.push_output(
                                                    &crate::skills::format_list_entry(&skill),
                                                    LineStyle::Status,
                                                );
                                            }
                                        }
                                    }
                                    continue;
                                }

                                if let Some(rest) = input.strip_prefix("/skills ") {
                                    let name = rest
                                        .trim_end_matches(" help")
                                        .trim_end_matches(" --help")
                                        .trim();
                                    if rest.ends_with(" help") || rest.ends_with(" --help") {
                                        match crate::skills::load_by_name(
                                            name,
                                            &config.project_root,
                                        ) {
                                            Some(skill) => {
                                                for line in crate::skills::format_help(
                                                    &skill,
                                                    &config.project_root,
                                                ) {
                                                    app.push_output(&line, LineStyle::Status);
                                                }
                                            }
                                            None => app.push_output(
                                                &format!("skill '{name}' not found"),
                                                LineStyle::Error,
                                            ),
                                        }
                                        continue;
                                    }
                                }

                                // Check for skill invocation: /skill-name [args]
                                let (user_message, active_skill_reminder) = if let Some(
                                    slash_rest,
                                ) =
                                    input.strip_prefix('/')
                                {
                                    let (name, args) =
                                        slash_rest.split_once(' ').unwrap_or((slash_rest, ""));
                                    if let Some(skill) =
                                        crate::skills::load_by_name(name, &config.project_root)
                                    {
                                        let skill_path = skill.path.clone();
                                        let perms_for_skill = perms.clone();
                                        let authorize =
                                            move || perms_for_skill.check_skill_shell(&skill_path);
                                        match crate::skills::render(&skill, args, authorize) {
                                            Ok(rendered) => {
                                                let display = if args.is_empty() {
                                                    format!("/{}", skill.name)
                                                } else {
                                                    format!("/{} {args}", skill.name)
                                                };
                                                app.push_output(
                                                    &format!("you> {display}"),
                                                    LineStyle::Normal,
                                                );
                                                let reminder = format!(
                                                    "Follow the instructions from {} (already provided as your task).",
                                                    skill.display_path(&config.project_root)
                                                );
                                                (rendered, Some(reminder))
                                            }
                                            Err(e) => {
                                                app.push_output(
                                                    &format!("skill error: {e}"),
                                                    LineStyle::Error,
                                                );
                                                continue;
                                            }
                                        }
                                    } else {
                                        app.push_output(
                                            &format!("you> {input}"),
                                            LineStyle::Normal,
                                        );
                                        (input.clone(), None)
                                    }
                                } else {
                                    app.push_output(&format!("you> {input}"), LineStyle::Normal);
                                    (input.clone(), None)
                                };

                                // The turn is real from here: make it visible
                                // NOW (spinner, "routing" job, log line) — the
                                // routers below are LLM calls that can take a
                                // while, so `redraw_while` keeps the spinner and
                                // timer moving until they return.
                                // Every path from here reaches the turn-end
                                // `finish_completed_turn`, which clears
                                // `is_thinking`.
                                app.begin_routing();
                                log.user_message(&input);
                                let _ = terminal.draw(|f| ui::draw(f, &app));

                                // Per-turn intent router (fail-safe to
                                // CODING). EXPLORE turns run a read-only,
                                // no-plan, Q&A-directed variant; the proven
                                // coding path is byte-unchanged otherwise.
                                // Pure model-driven: the artifact-mutation
                                // prompt routes build/change requests to CODING
                                // on its own (validated battery), so there is no
                                // keyword pre-route — on any parse failure the
                                // classifier still fails safe to CODING.
                                // Pre-turn skill router (fail-safe; skipped when the
                                // user already invoked a skill via /name). See
                                // agent::skill_router — probe: adoption 0/8 -> 8/8.
                                let user_message = if active_skill_reminder.is_none() {
                                    match redraw_while(
                                        crate::cli::commands::agent::skill_router::route_task_to_skill(
                                            &llm_worker,
                                            &config.project_root,
                                            &user_message,
                                            &cancelled,
                                        ),
                                        &mut terminal,
                                        &app,
                                    )
                                    .await
                                    {
                                        Some(skill) => {
                                            log.route_skill(&skill);
                                            app.push_output(
                                                &format!("  · task routed to skill '{skill}'"),
                                                LineStyle::Status,
                                            );
                                            crate::cli::commands::agent::skill_router::rewrite_task_for_skill(
                                                &skill,
                                                &user_message,
                                            )
                                        }
                                        None => user_message,
                                    }
                                } else {
                                    user_message
                                };

                                let is_explore = redraw_while(
                                    classify_is_explore(&llm_worker, &user_message, &cancelled),
                                    &mut terminal,
                                    &app,
                                )
                                .await;
                                app.clear_active_job();
                                log.route(if is_explore { "explore" } else { "coding" });
                                let turn_cfg = if is_explore {
                                    let mut c = config.clone();
                                    c.tools.plan = false;
                                    c.tools.ceremony = crate::config::CeremonyMode::Off;
                                    c
                                } else {
                                    config.clone()
                                };
                                let turn_tools = if is_explore {
                                    read_only_tool_defs(&tool_defs)
                                } else {
                                    tool_defs.clone()
                                };
                                if is_explore {
                                    app.push_output(
                                        "[explore] read-only investigation — no edits. \
                                         Say e.g. \"actually, change it\" to switch to coding.",
                                        LineStyle::Status,
                                    );
                                }

                                // Run the agent loop
                                let mcp_summary_clone = mcp_summary.clone();

                                // Assemble context
                                let assembled = context::assemble(
                                    &turn_cfg,
                                    &user_message,
                                    &conversation_history,
                                    false,
                                    mcp_summary_clone.as_deref(),
                                );
                                conversation_history.push(Message::user(&user_message));

                                // Inject skill reminder into system prompt so every LLM call
                                // in this turn is reminded which skill is being executed.
                                let mut messages = assembled.messages;
                                if let Some(ref reminder) = active_skill_reminder
                                    && let Some(sys_msg) = messages.first_mut()
                                    && let Some(ref mut content) = sys_msg.content
                                {
                                    content.push_str("\n[ACTIVE SKILL]\n");
                                    content.push_str(reminder);
                                }
                                if is_explore
                                    && let Some(sys_msg) = messages.first_mut()
                                    && let Some(ref mut content) = sys_msg.content
                                {
                                    content.push_str(
                                        "\n[INVESTIGATION MODE] Read-only. Investigate with the \
                                         read tools and answer the question precisely, citing \
                                         file:line. Do NOT modify code or files. If a code change \
                                         is actually wanted, state that and ask the user to \
                                         rephrase as an edit request.\n",
                                    );
                                }

                                // Live plan panel: show it for coding turns
                                // (it fills in as the model sets/checks steps).
                                // Suppressed for read-only Q&A (no plan there).
                                app.plan_task = if is_explore {
                                    None
                                } else {
                                    Some(input.clone())
                                };
                                app.plan_steps.clear();
                                // plan.md persists across turns: a finished
                                // plan from the previous turn must not show
                                // under this turn's task (display-only).
                                if is_explore {
                                    app.stale_plan = None;
                                } else {
                                    app.mark_stale_if_complete(parsed_plan_view(&config));
                                }

                                let max_rounds = config.context.max_rounds;
                                let perms_ref = &perms;
                                let mcp_ref = &mcp_registry;
                                let conv_ref = &mut conversation_history;

                                // Fresh snapshot baseline for THIS turn: SCRAP's
                                // revert-to-round-0 must only undo this turn's
                                // changes, never prior turns' work (see the
                                // `snapshots` declaration above for why this
                                // can't be session-scoped the way run.rs's is).
                                snapshots =
                                    tools::snapshots::SnapshotManager::init(&config.project_root)
                                        .ok()
                                        .map(|s| Arc::new(Mutex::new(s)));

                                // Run agent loop inline (not spawned — needs mutable refs).
                                // Context compaction now happens EVERY round inside
                                // the turn driver (matching run.rs) rather than once
                                // per turn out here.
                                run_agent_turn(
                                    &mut app,
                                    &mut rx,
                                    &mut terminal,
                                    &router,
                                    &llm_worker,
                                    &tool_pool,
                                    &turn_tools,
                                    &turn_cfg,
                                    is_explore,
                                    perms_ref,
                                    mcp_ref,
                                    &cancelled,
                                    &mut messages,
                                    conv_ref,
                                    max_rounds,
                                    log.clone(),
                                    &lsp_client,
                                    &fast_revisions,
                                    fast_baseline_errors,
                                    &snapshots,
                                    tool_def_tokens,
                                    mcp_summary.as_deref(),
                                    &user_message,
                                    &job_registry,
                                )
                                .await;

                                // The agent loop may exit via early-break
                                // paths (empty choices, errors) that skip
                                // flush_tokens. Flush here so stray tokens
                                // don't sit in the buffer.
                                app.flush_tokens();

                                // The turn and any post-turn work are fully
                                // done — flush tokens, draw the separator, flip
                                // `is_thinking=false`, redraw.
                                finish_completed_turn(&mut app, &mut terminal, None, None)?;

                                // Turns that never produced a plan (the model
                                // answered without planning) shouldn't leave an
                                // empty "(exploring…)" panel lingering. Keep the
                                // panel only when a real plan exists.
                                if app.plan_steps.is_empty() {
                                    app.clear_plan();
                                }

                                // Discard any keys / ticks that queued up
                                // while `is_thinking` was true. If we don't,
                                // a paste or impatient typing during the
                                // compressor await would replay against the
                                // freshly-idle input box and appear to
                                // submit the next prompt "on its own".
                                drain_stale_key_events(&mut rx);
                            }
                            KeyCode::Backspace => app.delete_char(),
                            KeyCode::Left => app.cursor_left(),
                            KeyCode::Right => app.cursor_right(),
                            // Shell convention: arrows are input history,
                            // always; the transcript scrolls on PgUp/PgDn
                            // and Ctrl+Home/End.
                            KeyCode::Up => app.history_up(),
                            KeyCode::Down => app.history_down(),
                            KeyCode::PageUp => app.scroll_up(10),
                            KeyCode::PageDown => app.scroll_down(10),
                            KeyCode::Home => {
                                if key.modifiers.contains(KeyModifiers::CONTROL) {
                                    app.scroll_offset = app.output.len().saturating_sub(1) as u16;
                                } else {
                                    app.cursor = 0;
                                }
                            }
                            KeyCode::End => {
                                if key.modifiers.contains(KeyModifiers::CONTROL) {
                                    app.scroll_offset = 0;
                                } else {
                                    app.cursor = app.input.len();
                                }
                            }
                            KeyCode::Char(c) => app.insert_char(c),
                            _ => {}
                        }
                    }
                }
            }

            AppEvent::Mouse(_) => {}
            AppEvent::PermissionRequest(prompt, response_tx) => {
                let response =
                    fulfill_permission_request(&mut app, &mut rx, &mut terminal, prompt).await;
                let _ = response_tx.send(response);
            }

            // Events from agent loop
            AppEvent::Token(token) => {
                app.push_token(&token);
            }
            AppEvent::ToolCall(name, summary) => {
                app.push_output(&format!("  → {name}({summary})"), LineStyle::ToolCall);
            }
            AppEvent::ToolResult(name, success, summary, full_content) => {
                let style = if success {
                    LineStyle::ToolOk
                } else {
                    LineStyle::ToolErr
                };
                let icon = if success { "✓" } else { "✗" };
                app.push_output(&format!("  {icon} {name}: {summary}"), style);
                app.store_tool_result(&name, &full_content);
            }
            AppEvent::Status(msg) => {
                app.push_output(&msg, LineStyle::Status);
            }
            AppEvent::LlmError(msg) => {
                app.push_output(&format!("error: {msg}"), LineStyle::Error);
                app.is_thinking = false;
            }
            AppEvent::LlmDone | AppEvent::AgentDone => {
                app.is_thinking = false;
                app.flush_tokens();
            }
        }
    }

    // Cleanup
    app.save_history(&history_file);

    // Shut down LSP
    if let Some(lsp) = lsp_client
        && let Ok(lsp) = Arc::try_unwrap(lsp)
    {
        lsp.shutdown().await;
    }

    Ok(())
}
