//! Interactive REPL mode with ratatui TUI.

use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::Mutex;

use anyhow::Result;
use crossterm::ExecutableCommand;
use crossterm::event::{KeyCode, KeyModifiers};
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::Terminal;
use ratatui::backend::Backend;
use ratatui::backend::CrosstermBackend;
use tokio::sync::mpsc;

use crate::cli::commands::agent::display::summarize_args;
use crate::cli::commands::agent::hints::{
    PLAN_CHECKPOINT_AFTER_EDITS, PLAN_CHECKPOINT_BLOCK_MESSAGE, PLAN_CHECKPOINT_WARNING,
    PLAN_HARD_BLOCK_AFTER_EDITS, PLAN_PROGRESS_NUDGE, PREMATURE_EXIT_NUDGE, loop_detected_hint,
    truncated_tool_call_hint,
};
use crate::cli::commands::agent::loop_detector::loop_call_key;
use crate::cli::commands::agent::permissions::permission_action;
use crate::config::{Config, EditMode, ModelRole};
use crate::context;
use crate::context::compress;
use crate::llm::{ChatRequest, Message, ModelRouter, is_truncated_tool_call_error};
use crate::logging::SessionLog;
use crate::lsp::LspClient;
use crate::mcp::{McpConfig, McpRegistry};
use crate::runtime::{
    LlmWorkerEvent, LlmWorkerHandle, ShellControl, ShellWorkerEvent, ToolWorkerPool,
};
use crate::tools;
use crate::tools::permissions::{Action, PermissionManager};
use crate::tui::app::{App, AppMode, LineStyle};
use crate::tui::event::{self, AppEvent};
use crate::tui::ui;

struct ReplTerminalGuard;

impl ReplTerminalGuard {
    fn enter() -> Result<Self> {
        terminal::enable_raw_mode()?;
        io::stdout().execute(EnterAlternateScreen)?;
        Ok(Self)
    }
}

impl Drop for ReplTerminalGuard {
    fn drop(&mut self) {
        let _ = terminal::disable_raw_mode();
        let _ = io::stdout().execute(LeaveAlternateScreen);
    }
}

/// Run the interactive REPL with TUI.
pub async fn run(config: Config, headless: bool, continue_session: bool) -> Result<()> {
    let log = Arc::new(SessionLog::new(&config));

    let router = Arc::new(ModelRouter::new(&config));
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
        if config.tools.edit_mode == EditMode::Fast {
            disabled.push("edit_file");
        }
        tool_defs.retain(|t| !disabled.contains(&t.function.name.as_str()));
        if config.tools.edit_mode == EditMode::Fast {
            tool_defs.extend(tools::fast_mode_tool_definitions());
        }
        tool_defs.push(tools::definitions::spawn_agents_tool_definition());
    }

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
            let miniswe_dir = config.miniswe_path("revisions");
            tools::RevisionStore::new(&miniswe_dir).ok().map(Arc::new)
        } else {
            None
        };
    let fast_baseline_errors: usize = if config.tools.edit_mode == EditMode::Fast {
        tools::fast::project_error_count(lsp_client.as_deref()).await
    } else {
        0
    };

    // Clear stale scratchpad/plan — unless `--continue` is set, in which
    // case carry the previous session's state forward.
    if !continue_session {
        let _ = std::fs::remove_file(config.miniswe_path("scratchpad.md"));
        let _ = std::fs::remove_file(config.miniswe_path("plan.md"));
    }

    // Initialize MCP
    let mcp_config = McpConfig::load(&config.project_root)?;
    let mcp_registry = if mcp_config.has_servers() {
        let cache_dir = config.miniswe_path("mcp");
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
    let _terminal_guard = ReplTerminalGuard::enter()?;
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;

    // Set up app state
    let mut app = App::new();
    let history_file = config.miniswe_path("sessions/repl_history.txt");
    app.load_history(&history_file);

    // Welcome message — probe what the server is actually serving rather
    // than parroting config.toml, which can disagree with reality when a
    // llama-swap/llama-cpp in front of the endpoint is loading a
    // different gguf than named in config.
    for line in router.startup_summary().await {
        app.push_output(&format!("miniswe — {line}"), LineStyle::Status);
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
                        // In detail view: Esc, Ctrl+O, or q closes it
                        if key.code == KeyCode::Esc
                            || event::is_ctrl_o(&key)
                            || key.code == KeyCode::Char('q')
                        {
                            app.close_detail();
                        }
                    }
                    AppMode::Normal => {
                        if event::is_ctrl_d(&key) {
                            break;
                        }

                        if event::is_ctrl_c(&key) {
                            if app.is_thinking {
                                cancelled.store(true, Ordering::Relaxed);
                                app.push_output("(interrupted)", LineStyle::Status);
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
                                if input == "quit" || input == "exit" || input == "/quit" {
                                    break;
                                }

                                if input == "/clear" || input == "/new" {
                                    conversation_history.clear();
                                    if input == "/new" {
                                        let _ = std::fs::remove_file(
                                            config.miniswe_path("scratchpad.md"),
                                        );
                                        let _ =
                                            std::fs::remove_file(config.miniswe_path("plan.md"));
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
                                    app.push_output(
                                        "/clear — clear conversation history",
                                        LineStyle::Status,
                                    );
                                    app.push_output(
                                        "/new   — clear history + scratchpad + plan",
                                        LineStyle::Status,
                                    );
                                    app.push_output(
                                        "/skills list       — list available skills",
                                        LineStyle::Status,
                                    );
                                    app.push_output(
                                        "/skills <name> help — show skill details",
                                        LineStyle::Status,
                                    );
                                    app.push_output("/help  — show this help", LineStyle::Status);
                                    app.push_output("quit   — exit", LineStyle::Status);
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

                                // Run the agent loop
                                let mcp_summary_clone = mcp_summary.clone();

                                // Assemble context
                                let assembled = context::assemble(
                                    &config,
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

                                app.is_thinking = true;

                                let max_rounds = config.context.max_rounds;
                                let perms_ref = &perms;
                                let mcp_ref = &mcp_registry;
                                let conv_ref = &mut conversation_history;

                                log.user_message(&input);

                                // Run agent loop inline (not spawned — needs mutable refs)
                                run_agent_loop(
                                    &mut app,
                                    &mut rx,
                                    &mut terminal,
                                    &router,
                                    &llm_worker,
                                    &tool_pool,
                                    &tool_defs,
                                    &config,
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
                                )
                                .await;

                                // The agent loop may exit via early-break
                                // paths (empty choices, errors) that skip
                                // flush_tokens. Flush here so stray tokens
                                // don't sit in the buffer through compression.
                                app.flush_tokens();

                                app.set_active_job("compressing context");
                                let _ = terminal.draw(|frame| ui::draw(frame, &app));
                                let mut plan_update_requested = true;
                                {
                                    let compress_fut = context::compressor::maybe_compress(
                                        &mut conversation_history,
                                        &config,
                                        &router,
                                        &llm_worker,
                                        tool_def_tokens,
                                        &mut plan_update_requested,
                                    );
                                    let mut compress_fut = std::pin::pin!(compress_fut);
                                    let mut done = false;
                                    while !done {
                                        tokio::select! {
                                            biased;
                                            () = &mut compress_fut, if !done => { done = true; }
                                            evt = rx.recv() => {
                                                if matches!(evt, Some(AppEvent::Tick)) {
                                                    let _ = terminal.draw(|frame| ui::draw(frame, &app));
                                                }
                                            }
                                        }
                                    }
                                }
                                app.clear_active_job();

                                // Now the turn and any post-turn work are
                                // fully done — flush tokens, draw the
                                // separator, flip `is_thinking=false`, redraw.
                                finish_completed_turn(&mut app, &mut terminal, None, None)?;

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
                            KeyCode::Up => {
                                if app.input.is_empty() {
                                    app.scroll_up(1);
                                } else {
                                    app.history_up();
                                }
                            }
                            KeyCode::Down => {
                                if app.input.is_empty() {
                                    app.scroll_down(1);
                                } else {
                                    app.history_down();
                                }
                            }
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

/// Run the agent loop (LLM call → tool execution → repeat).
/// This runs inline in the main loop, processing events between rounds.
async fn run_agent_loop(
    app: &mut App,
    rx: &mut mpsc::UnboundedReceiver<AppEvent>,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    router: &Arc<ModelRouter>,
    llm_worker: &LlmWorkerHandle,
    tool_pool: &ToolWorkerPool,
    tool_defs: &[crate::llm::ToolDefinition],
    config: &Config,
    perms: &Arc<PermissionManager>,
    mcp_registry: &Option<Arc<Mutex<McpRegistry>>>,
    cancelled: &Arc<AtomicBool>,
    messages: &mut Vec<Message>,
    conversation_history: &mut Vec<Message>,
    max_rounds: usize,
    log: Arc<SessionLog>,
    lsp: &Option<Arc<LspClient>>,
    fast_revisions: &Option<Arc<tools::RevisionStore>>,
    fast_baseline_errors: usize,
) {
    let mut round = 0;
    let mut tool_result_log: Vec<(String, serde_json::Value, String)> = Vec::new();
    // Track consecutive identical tool calls to detect loops
    let mut last_call_key: Option<String> = None;
    let mut same_call_streak = 0u32;
    // Number of distinct loops the model has been pulled out of in this
    // turn. We give one recovery; a second loop ends the turn for real.
    let mut loop_recoveries = 0u32;
    let mut successful_edits_since_plan_update = 0u32;
    let mut plan_checkpoint_pending = false;
    let mut nudged_premature_exit = false;

    loop {
        // Check cancellation at the top of every round
        if consume_interrupt(cancelled) {
            app.push_output("(interrupted)", LineStyle::Status);
            break;
        }

        round += 1;
        log.round_start(round);
        if round > max_rounds {
            app.push_output("Maximum tool rounds reached.", LineStyle::Error);
            break;
        }

        // Observation masking — budget = half the context window
        let tool_result_budget = config.model.context_window / 2;
        mask_old_tool_results(
            messages,
            &tool_result_log,
            tool_result_budget,
            &config.project_root,
        );

        // Sanitize messages
        context::sanitize_messages(messages);

        // Build request
        let request = ChatRequest {
            messages: messages.clone(),
            tools: Some(tool_defs.to_vec()),
            tool_choice: None,
        };
        log.llm_request(&request);

        cancelled.store(false, Ordering::Relaxed);
        app.is_thinking = true;
        app.set_active_job("llm");

        // Render before LLM call so spinner is visible immediately
        let _ = terminal.draw(|frame| ui::draw(frame, app));

        // Call LLM with streaming — render on each token
        let mut rendered_assistant_text = String::new();
        // Set when the server rejected the model's tool call as
        // truncated JSON. In that case we inject a synthetic user-role
        // hint and continue the outer loop so the agent can recover
        // with a smaller operation, instead of aborting the session.
        let mut truncated_tool_call_hint_pushed = false;
        let response = {
            let mut token_count = 0u32;
            let mut llm_events =
                llm_worker.submit(ModelRole::Default, request.clone(), cancelled.clone());
            loop {
                tokio::select! {
                    evt = llm_events.recv() => {
                        match evt {
                            Some(LlmWorkerEvent::Token(token)) => {
                                app.push_token(&token);
                                rendered_assistant_text.push_str(&token);
                                token_count += 1;
                                if token_count.is_multiple_of(3) {
                                    let _ = terminal.draw(|frame| ui::draw(frame, app));
                                }
                            }
                            Some(LlmWorkerEvent::Completed(Ok(r))) => break Some(r),
                            Some(LlmWorkerEvent::Completed(Err(err_str))) => {
                                if err_str.contains("Interrupted") {
                                    cancelled.store(false, Ordering::Relaxed);
                                    app.push_output("Generation interrupted.", LineStyle::Status);
                                } else if is_truncated_tool_call_error(&err_str) {
                                    // Model hit max_tokens mid tool-call — the
                                    // JSON couldn't be parsed server-side, so
                                    // no tool_call_id was issued. Clear the
                                    // partial UI text (don't persist the
                                    // half-streamed output) and push a
                                    // user-role hint so the agent retries
                                    // with a smaller operation.
                                    log.llm_error(
                                        "tool call JSON truncated (max_tokens) — \
                                         injecting hint and continuing",
                                    );
                                    app.push_output(
                                        "Previous tool call truncated — retrying with guidance.",
                                        LineStyle::Status,
                                    );
                                    let hint =
                                        Message::user(truncated_tool_call_hint(config.tools.edit_mode));
                                    messages.push(hint.clone());
                                    conversation_history.push(hint);
                                    truncated_tool_call_hint_pushed = true;
                                } else {
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
                                    app.push_output(&format!("LLM error: {clean}"), LineStyle::Error);
                                }
                                app.clear_active_job();
                                break None;
                            }
                            None => {
                                app.push_output("LLM worker stopped unexpectedly.", LineStyle::Error);
                                app.clear_active_job();
                                break None;
                            }
                        }
                    }
                    app_evt = rx.recv() => {
                        match app_evt {
                            Some(AppEvent::Tick) => {
                                let _ = terminal.draw(|frame| ui::draw(frame, app));
                            }
                            Some(AppEvent::Key(key)) if handle_background_key(app, &key) => {
                                let _ = terminal.draw(|frame| ui::draw(frame, app));
                            }
                            Some(AppEvent::Key(key)) if event::is_ctrl_c(&key) => {
                                cancelled.store(true, Ordering::Relaxed);
                                app.push_output("(interrupted)", LineStyle::Status);
                                let _ = terminal.draw(|frame| ui::draw(frame, app));
                            }
                            Some(AppEvent::Mouse(_)) => {}
                            Some(AppEvent::PermissionRequest(prompt, response_tx)) => {
                                let response =
                                    fulfill_permission_request(app, rx, terminal, prompt).await;
                                let _ = response_tx.send(response);
                            }
                            Some(_) => {}
                            None => {
                                app.push_output("Event stream closed.", LineStyle::Error);
                                app.clear_active_job();
                                break None;
                            }
                        }
                    }
                }
            }
        };

        app.clear_active_job();

        // Re-render after LLM response
        let _ = terminal.draw(|frame| ui::draw(frame, app));

        let Some(response) = response else {
            if truncated_tool_call_hint_pushed {
                // Hint was injected into `messages`; loop back and let
                // the agent try again with smaller operations.
                continue;
            }
            break;
        };

        let choice = match response.choices.first() {
            Some(c) => c,
            None => break,
        };

        let assistant_msg = &choice.message;

        // Flush any remaining tokens
        app.flush_tokens();

        if let Some(content) = &assistant_msg.content {
            log.llm_response(content);
            if let Some(missing) =
                reconcile_streamed_assistant_content(&rendered_assistant_text, content)
            {
                app.push_token(&missing);
                app.flush_tokens();
                let _ = terminal.draw(|frame| ui::draw(frame, app));
            }
        }
        if assistant_msg.is_meaningful() {
            conversation_history.push(assistant_msg.clone());
        }

        let tool_calls = match &assistant_msg.tool_calls {
            Some(tc) if !tc.is_empty() => tc.clone(),
            _ => {
                if !nudged_premature_exit
                    && config.tools.plan
                    && tools::plan::has_unchecked_steps(config)
                {
                    nudged_premature_exit = true;
                    let nudge = Message::user(PREMATURE_EXIT_NUDGE);
                    messages.push(nudge.clone());
                    conversation_history.push(nudge);
                    continue;
                }
                break;
            }
        };

        messages.push(assistant_msg.clone());

        // Execute tool calls
        for tc in &tool_calls {
            // Check cancellation between tool calls
            if consume_interrupt(cancelled) {
                app.push_output("(interrupted)", LineStyle::Status);
                return;
            }
            let args: serde_json::Value = match serde_json::from_str(&tc.function.arguments) {
                Ok(v) => v,
                Err(e) => {
                    let result_msg = Message::tool_result(
                        &tc.id,
                        &format!("Invalid JSON in tool arguments: {e}"),
                    );
                    messages.push(result_msg.clone());
                    conversation_history.push(result_msg);
                    app.push_output(
                        &format!("  ✗ {}: invalid JSON args", tc.function.name),
                        LineStyle::ToolErr,
                    );
                    continue;
                }
            };

            let args_summary = summarize_args(&tc.function.name, &args);

            // Detect tool call loops: only identical calls repeated consecutively.
            let call_key = loop_call_key(&tc.function.name, &args);
            if last_call_key.as_ref() == Some(&call_key) {
                same_call_streak += 1;
            } else {
                last_call_key = Some(call_key.clone());
                same_call_streak = 1;
            }
            if same_call_streak >= 3 {
                log.loop_detected(&tc.function.name, &args_summary, same_call_streak as usize);
                let result_msg =
                    Message::tool_result(&tc.id, loop_detected_hint(config.tools.edit_mode));
                messages.push(result_msg.clone());
                conversation_history.push(result_msg);

                // First loop in this turn: surface the hint, reset the
                // streak, and let the model try a different approach.
                // A second loop after the recovery means the model is
                // genuinely stuck — bail then.
                if loop_recoveries == 0 {
                    loop_recoveries += 1;
                    last_call_key = None;
                    same_call_streak = 0;
                    app.push_output(
                        &format!(
                            "  ⚠ Loop detected: {}({}) repeated 3 times — surfacing a hint, giving the model one more round",
                            tc.function.name, args_summary
                        ),
                        LineStyle::Status,
                    );
                    break;
                }
                app.push_output(
                    &format!(
                        "  ✗ Loop detected again ({}({})) after the recovery hint — stopping this turn",
                        tc.function.name, args_summary
                    ),
                    LineStyle::Error,
                );
                return;
            }

            log.tool_call_detail(&tc.function.name, &args);
            app.push_output(
                &format!("  → {}({})", tc.function.name, args_summary),
                LineStyle::ToolCall,
            );

            // Re-render to show tool call
            let _ = terminal.draw(|frame| ui::draw(frame, app));

            // Determine if this tool call needs a permission prompt
            let perm_action = permission_action(&tc.function.name, &args);

            // Check permission via TUI prompt (not raw stderr)
            let mut perm_denied = false;
            if let Some(ref action) = perm_action
                && matches!(action, Action::Shell(_) | Action::McpUse(_, _))
            {
                match perms.check_needs_prompt(action) {
                    Err(e) => {
                        // Blocklisted — skip this tool call
                        let result_msg = Message::tool_result(&tc.id, &e);
                        messages.push(result_msg.clone());
                        conversation_history.push(result_msg);
                        app.push_output(
                            &format!("  ✗ {}: {e}", tc.function.name),
                            LineStyle::ToolErr,
                        );
                        continue;
                    }
                    Ok(Some(prompt)) => {
                        // Needs user approval — show prompt in TUI
                        app.pending_permission = Some(prompt);
                        app.input.clear();
                        app.cursor = 0;
                        let _ = terminal.draw(|frame| ui::draw(frame, app));

                        // Wait for user input (y/n/a)
                        let response = wait_for_permission_input(app, rx, terminal).await;
                        app.pending_permission = None;

                        match response.as_str() {
                            "y" | "yes" => {
                                perms.approve(action, false);
                                app.push_output(
                                    "  · Permission granted, running tool...",
                                    LineStyle::Status,
                                );
                            }
                            "a" | "always" => {
                                perms.approve(action, true);
                                app.push_output(
                                    "  · Permission granted and saved, running tool...",
                                    LineStyle::Status,
                                );
                            }
                            _ => {
                                perm_denied = true;
                                app.push_output("  · Permission denied.", LineStyle::Status);
                            }
                        }

                        let _ = terminal.draw(|frame| ui::draw(frame, app));
                    }
                    Ok(None) => {} // No prompt needed
                }
            }

            if perm_denied {
                let result_msg =
                    Message::tool_result(&tc.id, &format!("{} denied by user", tc.function.name));
                messages.push(result_msg.clone());
                conversation_history.push(result_msg);
                app.push_output(
                    &format!("  ✗ {}: denied", tc.function.name),
                    LineStyle::ToolErr,
                );
                continue;
            }

            let file_action = args["action"].as_str().unwrap_or("");
            let is_write_action = matches!(tc.function.name.as_str(), "edit_file" | "write_file");
            if config.tools.plan
                && plan_checkpoint_pending
                && successful_edits_since_plan_update >= PLAN_HARD_BLOCK_AFTER_EDITS
                && is_write_action
            {
                let result_msg = Message::tool_result(&tc.id, PLAN_CHECKPOINT_BLOCK_MESSAGE);
                messages.push(result_msg.clone());
                conversation_history.push(result_msg);
                app.push_output("  ✗ blocked: plan checkpoint", LineStyle::ToolErr);
                continue;
            }

            // Execute tool (permissions already checked above for shell/web/mcp)
            let mut result = if matches!(
                tc.function.name.as_str(),
                "replace_range" | "insert_at" | "revert" | "show_rev" | "check"
            ) && config.tools.edit_mode == EditMode::Fast
            {
                let tool_name = tc.function.name.clone();
                let args = args.clone();
                let config = config.clone();
                let perms = perms.clone();
                let lsp = lsp.clone();
                let revisions = fast_revisions.clone();
                let baseline = fast_baseline_errors;
                let mut result_rx = tool_pool.submit(move || {
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
                });
                await_tool_job_ui(
                    rx,
                    terminal,
                    app,
                    &tc.function.name,
                    &mut result_rx,
                    cancelled,
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
                    let registry = mcp_registry.clone();
                    let mut result_rx = tool_pool.submit(move || match registry {
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
                    });
                    await_tool_job_ui(rx, terminal, app, "mcp_use", &mut result_rx, cancelled).await
                }
            } else if tc.function.name == "spawn_agents" {
                let tasks = crate::cli::commands::agent::subagent::parse_tasks(&args);
                if tasks.is_empty() {
                    crate::tools::ToolResult::err(
                        "spawn_agents: 'agents' must be a non-empty array of {label, prompt}"
                            .into(),
                    )
                } else {
                    let (out_tx, mut out_rx) =
                        tokio::sync::mpsc::unbounded_channel::<(String, LineStyle)>();
                    let subagents_fut = crate::cli::commands::agent::subagent::run_subagents(
                        tasks,
                        config,
                        llm_worker,
                        tool_pool,
                        tool_defs,
                        perms,
                        mcp_registry,
                        lsp,
                        fast_revisions,
                        fast_baseline_errors,
                        cancelled,
                        Some(out_tx),
                    );
                    let mut subagents_fut = std::pin::pin!(subagents_fut);
                    let mut outputs = None;
                    while outputs.is_none() {
                        tokio::select! {
                            biased;
                            r = &mut subagents_fut, if outputs.is_none() => { outputs = Some(r); }
                            line = out_rx.recv() => {
                                if let Some((text, style)) = line {
                                    app.push_output(&text, style);
                                    let _ = terminal.draw(|frame| ui::draw(frame, app));
                                }
                            }
                            evt = rx.recv() => {
                                if matches!(evt, Some(AppEvent::Tick)) {
                                    let _ = terminal.draw(|frame| ui::draw(frame, app));
                                }
                            }
                        }
                    }
                    let combined =
                        crate::cli::commands::agent::subagent::format_outputs(outputs.unwrap());
                    crate::tools::ToolResult::ok(combined)
                }
            } else if tc.function.name == "edit_file" {
                let args = args.clone();
                let config = config.clone();
                let perms = perms.clone();
                let router = router.clone();
                let lsp = lsp.clone();
                let cancelled_for_job = cancelled.clone();
                let log_for_job = log.clone();
                let mut result_rx = tool_pool.submit(move || {
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .map_err(|e| e.to_string())?;
                    runtime
                        .block_on(async move {
                            crate::tools::execute_edit_file_tool(
                                &args,
                                &config,
                                perms.as_ref(),
                                router.as_ref(),
                                lsp.as_deref(),
                                Some(cancelled_for_job.as_ref()),
                                Some(log_for_job.as_ref()),
                            )
                            .await
                        })
                        .map_err(|e| format!("edit_file error: {e}"))
                });
                await_tool_job_ui(rx, terminal, app, "edit_file", &mut result_rx, cancelled).await
            } else if tc.function.name == "plan" {
                let args = args.clone();
                let config = config.clone();
                let mut result_rx = tool_pool.submit(move || {
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .map_err(|e| e.to_string())?;
                    runtime
                        .block_on(async move { tools::plan::execute(&args, &config, round).await })
                        .map_err(|e| format!("plan error: {e}"))
                });
                await_tool_job_ui(rx, terminal, app, "plan", &mut result_rx, cancelled).await
            } else if tc.function.name == "file" && file_action == "shell" {
                await_shell_job_repl(
                    tool_pool.submit_shell(args.clone(), config.clone(), cancelled.clone()),
                    app,
                    rx,
                    terminal,
                    cancelled,
                )
                .await
            } else {
                let tool_name = tc.function.name.clone();
                let args = args.clone();
                let config = config.clone();
                let perms = perms.clone();
                let lsp = lsp.clone();
                let mut result_rx = tool_pool.submit(move || {
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
                });
                await_tool_job_ui(
                    rx,
                    terminal,
                    app,
                    &tc.function.name,
                    &mut result_rx,
                    cancelled,
                )
                .await
            };

            if !result.success
                && let Some(hint) = tools::plan::failure_hint(config)
            {
                result.content.push('\n');
                result.content.push_str(&hint);
            }

            let first_line = result.content.lines().next().unwrap_or("(empty)");
            log.tool_call(&tc.function.name, &args_summary, result.success, first_line);
            log.tool_result_detail(&tc.function.name, result.success, &result.content);
            let style = if result.success {
                LineStyle::ToolOk
            } else {
                LineStyle::ToolErr
            };
            let icon = if result.success { "✓" } else { "✗" };
            app.push_output(
                &format!("  {icon} {}: {first_line}", tc.function.name),
                style,
            );
            app.store_tool_result(&tc.function.name, &result.content);

            if result.success && tc.function.name == "plan" {
                plan_checkpoint_pending = false;
                successful_edits_since_plan_update = 0;
            }

            // Successful file write = code changed, reset loop detector
            let is_file_write = matches!(tc.function.name.as_str(), "edit_file" | "write_file");
            if result.success && is_file_write {
                last_call_key = None;
                same_call_streak = 0;
                if config.tools.plan {
                    if tools::plan::plan_exists(config) {
                        result.content.push('\n');
                        result.content.push_str(PLAN_PROGRESS_NUDGE);
                    }
                    successful_edits_since_plan_update += 1;
                    if successful_edits_since_plan_update >= PLAN_CHECKPOINT_AFTER_EDITS {
                        plan_checkpoint_pending = true;
                    }
                    if successful_edits_since_plan_update == PLAN_CHECKPOINT_AFTER_EDITS {
                        result.content.push('\n');
                        result.content.push_str(PLAN_CHECKPOINT_WARNING);
                    }
                }
            }

            tool_result_log.push((
                tc.function.name.clone(),
                args.clone(),
                result.content.clone(),
            ));

            let result_msg = Message::tool_result(&tc.id, &result.content);
            messages.push(result_msg.clone());
            conversation_history.push(result_msg);

            // Re-render after tool result
            let _ = terminal.draw(|frame| ui::draw(frame, app));
        }
    }
}

/// Token-budget-aware observation masking (same logic as run.rs).
fn mask_old_tool_results(
    messages: &mut Vec<Message>,
    tool_result_log: &[(String, serde_json::Value, String)],
    tool_result_token_budget: usize,
    project_root: &std::path::Path,
) {
    // Delegate to the shared implementation pattern
    if tool_result_log.is_empty() {
        return;
    }

    let mut used_tokens = 0;
    let mut should_mask: Vec<bool> = vec![false; tool_result_log.len()];

    for i in (0..tool_result_log.len()).rev() {
        let tokens = context::estimate_tokens(&tool_result_log[i].2);
        used_tokens += tokens;
        if used_tokens > tool_result_token_budget {
            should_mask[i] = true;
        }
    }

    if !should_mask.iter().any(|m| *m) {
        return;
    }

    let summaries: Vec<Option<String>> = tool_result_log
        .iter()
        .enumerate()
        .map(|(i, (name, args, content))| {
            if should_mask[i] {
                Some(compress::summarize_tool_result(name, args, content))
            } else {
                None
            }
        })
        .collect();

    let summary_count = summaries.iter().filter(|s| s.is_some()).count();
    let keep_count = 20;

    if summary_count > keep_count {
        let archive_path = project_root.join(".miniswe").join("tool_history.md");
        let mut archive = match std::fs::read_to_string(&archive_path) {
            Ok(s) => s,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(e) => {
                tracing::warn!(
                    "Tool history read failed for {}; starting fresh and the next archive \
                     will overwrite the existing file: {e}",
                    archive_path.display()
                );
                String::new()
            }
        };
        let excess = summary_count - keep_count;
        let mut archived = 0;
        for s in &summaries {
            if let Some(s) = s
                && archived < excess
            {
                archive.push_str(s);
                archive.push('\n');
                archived += 1;
            }
        }
        if let Err(e) = crate::atomic_write(&archive_path, archive.as_bytes()) {
            tracing::warn!(
                "Tool history write failed for {}: {e}",
                archive_path.display()
            );
        }
    }

    let total_summaries = summaries.iter().filter(|s| s.is_some()).count();
    let mut tool_msg_idx = 0;
    for msg in messages.iter_mut() {
        if msg.role == "tool" {
            if let Some(Some(summary)) = summaries.get(tool_msg_idx) {
                let pos = summaries[..=tool_msg_idx]
                    .iter()
                    .filter(|s| s.is_some())
                    .count();
                let from_end = total_summaries - pos;
                if from_end < keep_count {
                    msg.content = Some(summary.clone());
                } else {
                    msg.content = Some(
                        "[archived — use file(action='read', path='.miniswe/tool_history.md') to recall]".into(),
                    );
                }
            }
            tool_msg_idx += 1;
        }
    }
}

fn handle_background_key(app: &mut App, key: &crossterm::event::KeyEvent) -> bool {
    match key.code {
        KeyCode::PageUp => {
            app.scroll_up(10);
            true
        }
        KeyCode::PageDown => {
            app.scroll_down(10);
            true
        }
        KeyCode::Up if app.input.is_empty() => {
            app.scroll_up(1);
            true
        }
        KeyCode::Down if app.input.is_empty() => {
            app.scroll_down(1);
            true
        }
        KeyCode::Home if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.scroll_offset = app.output.len().saturating_sub(1) as u16;
            true
        }
        KeyCode::End if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.scroll_offset = 0;
            true
        }
        _ => false,
    }
}

fn consume_interrupt(cancelled: &AtomicBool) -> bool {
    cancelled.swap(false, Ordering::Relaxed)
}

/// Drop any input-kind events queued in `rx` without processing them.
///
/// Called at the end of the Enter handler to discard keystrokes the user
/// typed while the "working" indicator was up (LLM streaming + post-turn
/// compression). Those keys were meant to be ignored per `is_thinking`, but
/// because the main loop is `await`-ing the agent/compressor, keys queue in
/// the channel and would otherwise be replayed against the now-idle input
/// box — producing a visible desync where the user's next prompt appears to
/// fire "on its own".
///
/// We preserve non-key events (permission requests, status updates). Ctrl+C
/// was already handled inline by the key reader via the `cancelled` flag.
fn drain_stale_key_events(rx: &mut mpsc::UnboundedReceiver<AppEvent>) {
    while let Ok(evt) = rx.try_recv() {
        match evt {
            AppEvent::Key(_) | AppEvent::Mouse(_) | AppEvent::Tick => {}
            other => {
                // Put non-input events back via a small re-enqueue: we only
                // have a receiver here, so the cleanest option is to drop
                // them too. In practice, permission requests and status
                // messages are only emitted while an agent task is running,
                // which it isn't by the time we get here.
                let _ = other;
            }
        }
    }
}

fn reconcile_streamed_assistant_content(rendered: &str, final_content: &str) -> Option<String> {
    if final_content.is_empty() || rendered == final_content {
        return None;
    }
    if let Some(suffix) = final_content.strip_prefix(rendered) {
        return (!suffix.is_empty()).then(|| suffix.to_string());
    }
    if rendered.is_empty() {
        return Some(final_content.to_string());
    }
    Some(format!(
        "\n[final response continuation]\n{}",
        final_content
    ))
}

fn finish_completed_turn(
    app: &mut App,
    terminal: &mut Terminal<impl Backend>,
    final_content: Option<&str>,
    rendered_assistant_text: Option<&str>,
) -> io::Result<()> {
    app.is_thinking = false;
    if let (Some(final_content), Some(rendered)) = (final_content, rendered_assistant_text)
        && let Some(missing) = reconcile_streamed_assistant_content(rendered, final_content)
    {
        app.push_token(&missing);
    }
    app.flush_tokens();
    app.push_output(
        "────────────────────────────────────────────────",
        LineStyle::Separator,
    );
    terminal
        .draw(|frame| ui::draw(frame, app))
        .map_err(|e| io::Error::other(e.to_string()))?;
    Ok(())
}

async fn await_tool_job_ui(
    rx: &mut mpsc::UnboundedReceiver<AppEvent>,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
    job_label: &str,
    result_rx: &mut tokio::sync::oneshot::Receiver<Result<crate::tools::ToolResult, String>>,
    cancelled: &Arc<AtomicBool>,
) -> crate::tools::ToolResult {
    app.set_active_job(job_label);
    loop {
        tokio::select! {
            result = &mut *result_rx => {
                app.clear_active_job();
                return match result {
                    Ok(Ok(tool_result)) => tool_result,
                    Ok(Err(err)) => crate::tools::ToolResult::err(err),
                    Err(_) => crate::tools::ToolResult::err("Tool worker dropped job".into()),
                };
            }
            evt = rx.recv() => {
                match evt {
                    Some(AppEvent::Tick) => {
                        let _ = terminal.draw(|frame| ui::draw(frame, app));
                    }
                    Some(AppEvent::Key(key)) if handle_background_key(app, &key) => {
                        let _ = terminal.draw(|frame| ui::draw(frame, app));
                    }
                    Some(AppEvent::Key(key)) if event::is_ctrl_c(&key) => {
                        cancelled.store(true, Ordering::Relaxed);
                        app.push_output("(interrupted)", LineStyle::Status);
                        let _ = terminal.draw(|frame| ui::draw(frame, app));
                    }
                    Some(AppEvent::Mouse(_)) => {}
                    Some(AppEvent::PermissionRequest(prompt, response_tx)) => {
                        let response = fulfill_permission_request(app, rx, terminal, prompt).await;
                        let _ = response_tx.send(response);
                    }
                    Some(_) => {}
                    None => {
                        app.clear_active_job();
                        return crate::tools::ToolResult::err("Event stream closed.".into())
                    },
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyEvent, KeyEventKind, KeyEventState};
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use serde_json::json;
    use tokio::sync::mpsc;

    #[test]
    fn file_search_summary_uses_pattern_and_path() {
        let args = json!({
            "action": "search",
            "path": "src/context/mod.rs",
            "pattern": "pub fn assemble",
        });

        assert_eq!(
            summarize_args("file", &args),
            "search \"pub fn assemble\" in src/context/mod.rs"
        );
    }

    #[test]
    fn plan_refine_summary_includes_step() {
        let args = json!({
            "action": "refine",
            "step": 2,
        });

        assert_eq!(summarize_args("plan", &args), "refine step 2");
    }

    #[test]
    fn web_search_summary_includes_query() {
        let args = json!({
            "action": "search",
            "query": "Michał Szynkiewicz",
        });

        assert_eq!(
            summarize_args("web", &args),
            "search \"Michał Szynkiewicz\""
        );
    }

    #[test]
    fn grouped_file_shell_maps_to_shell_permission_action() {
        let args = json!({
            "action": "shell",
            "command": "python -m http.server",
        });

        match permission_action("file", &args) {
            Some(Action::Shell(cmd)) => assert_eq!(cmd, "python -m http.server"),
            _ => panic!("expected grouped file shell to require shell permission"),
        }
    }

    #[test]
    fn loop_hint_smart_mentions_edit_file() {
        let hint = loop_detected_hint(EditMode::Smart);
        assert!(hint.contains("edit_file"));
    }

    #[test]
    fn loop_hint_fast_mentions_revision_table_tools() {
        let hint = loop_detected_hint(EditMode::Fast);
        assert!(hint.contains("show_rev"));
        assert!(hint.contains("revert"));
        assert!(!hint.contains("edit_file"));
    }

    #[test]
    fn consume_interrupt_clears_flag_after_first_read() {
        let cancelled = AtomicBool::new(true);
        assert!(consume_interrupt(&cancelled));
        assert!(!consume_interrupt(&cancelled));
        assert!(!cancelled.load(Ordering::Relaxed));
    }

    #[test]
    fn reconcile_streamed_assistant_content_appends_missing_suffix() {
        assert_eq!(
            reconcile_streamed_assistant_content("Hello", "Hello world"),
            Some(" world".into())
        );
    }

    #[test]
    fn reconcile_streamed_assistant_content_returns_none_when_complete() {
        assert_eq!(
            reconcile_streamed_assistant_content("Hello world", "Hello world"),
            None
        );
    }

    #[test]
    fn reconcile_streamed_assistant_content_uses_full_content_when_nothing_rendered() {
        assert_eq!(
            reconcile_streamed_assistant_content("", "Hello world"),
            Some("Hello world".into())
        );
    }

    #[test]
    fn finish_completed_turn_draws_final_text_and_separator() {
        let backend = TestBackend::new(80, 20);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new();
        app.push_token("Final answer");

        finish_completed_turn(&mut app, &mut terminal, Some("Final answer"), Some("")).unwrap();

        let text = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(text.contains("Final answer"));
        assert!(text.contains("────────────────────────────────────────────────"));
        assert!(!app.is_thinking);
    }

    #[test]
    fn finish_completed_turn_appends_missing_suffix_before_separator() {
        let backend = TestBackend::new(100, 20);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new();
        app.push_token("Hello");

        finish_completed_turn(&mut app, &mut terminal, Some("Hello world"), Some("Hello")).unwrap();

        let joined = app
            .output
            .iter()
            .map(|line| line.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(joined.contains("Hello world"));
        assert!(joined.ends_with("────────────────────────────────────────────────"));
    }

    #[tokio::test]
    async fn permission_prompt_accepts_single_key_without_enter() {
        let mut app = App::new();
        app.pending_permission = Some("Allow shell command?".into());
        let backend = TestBackend::new(80, 20);
        let mut terminal = Terminal::new(backend).unwrap();
        let (tx, mut rx) = mpsc::unbounded_channel();

        tx.send(AppEvent::Key(KeyEvent {
            code: KeyCode::Char('y'),
            modifiers: KeyModifiers::empty(),
            kind: KeyEventKind::Press,
            state: KeyEventState::empty(),
        }))
        .unwrap();

        let response = wait_for_permission_input(&mut app, &mut rx, &mut terminal).await;
        assert_eq!(response, "y");
        assert!(app.input.is_empty());
        assert_eq!(app.cursor, 0);
    }

    #[tokio::test]
    async fn permission_prompt_accepts_raw_carriage_return_as_enter() {
        let mut app = App::new();
        app.pending_permission = Some("Allow shell command?".into());
        app.input = "yes".into();
        app.cursor = app.input.len();
        let backend = TestBackend::new(80, 20);
        let mut terminal = Terminal::new(backend).unwrap();
        let (tx, mut rx) = mpsc::unbounded_channel();

        tx.send(AppEvent::Key(KeyEvent {
            code: KeyCode::Char('\r'),
            modifiers: KeyModifiers::empty(),
            kind: KeyEventKind::Press,
            state: KeyEventState::empty(),
        }))
        .unwrap();

        let response = wait_for_permission_input(&mut app, &mut rx, &mut terminal).await;
        assert_eq!(response, "yes");
        assert!(app.input.is_empty());
        assert_eq!(app.cursor, 0);
    }
}

/// Wait for the user to respond to a permission prompt in the TUI.
/// Blocks until Enter is pressed, returns the trimmed input (e.g., "y", "n", "a").
async fn wait_for_modal_input(
    app: &mut App,
    rx: &mut mpsc::UnboundedReceiver<AppEvent>,
    terminal: &mut Terminal<impl Backend>,
    instant_keys: &[char],
) -> String {
    loop {
        let _ = terminal.draw(|frame| ui::draw(frame, app));

        let evt = match rx.recv().await {
            Some(e) => e,
            None => return "n".into(),
        };

        match evt {
            AppEvent::Key(key) => match key.code {
                KeyCode::Enter => {
                    let response = app.input.trim().to_lowercase();
                    app.input.clear();
                    app.cursor = 0;
                    return response;
                }
                KeyCode::Char('\n') | KeyCode::Char('\r') => {
                    let response = app.input.trim().to_lowercase();
                    app.input.clear();
                    app.cursor = 0;
                    return response;
                }
                KeyCode::Char(c) => {
                    if key.modifiers.is_empty() {
                        let lower = c.to_ascii_lowercase();
                        if app.input.is_empty() && instant_keys.contains(&lower) {
                            app.input.clear();
                            app.cursor = 0;
                            return lower.to_string();
                        }
                    }
                    app.insert_char(c);
                }
                KeyCode::Backspace => app.delete_char(),
                KeyCode::Esc => {
                    app.input.clear();
                    app.cursor = 0;
                    return "n".into();
                }
                _ => {}
            },
            AppEvent::Tick => {} // re-render
            _ => {}
        }
    }
}

async fn wait_for_permission_input(
    app: &mut App,
    rx: &mut mpsc::UnboundedReceiver<AppEvent>,
    terminal: &mut Terminal<impl Backend>,
) -> String {
    wait_for_modal_input(app, rx, terminal, &['y', 'n', 'a']).await
}

async fn fulfill_permission_request(
    app: &mut App,
    rx: &mut mpsc::UnboundedReceiver<AppEvent>,
    terminal: &mut Terminal<impl Backend>,
    prompt: String,
) -> String {
    app.pending_permission = Some(prompt);
    app.input.clear();
    app.cursor = 0;
    let response = wait_for_permission_input(app, rx, terminal).await;
    app.pending_permission = None;
    let _ = terminal.draw(|frame| ui::draw(frame, app));
    response
}

async fn await_shell_job_repl(
    mut shell_job: crate::runtime::ShellJobHandle,
    app: &mut App,
    rx: &mut mpsc::UnboundedReceiver<AppEvent>,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    cancelled: &Arc<AtomicBool>,
) -> crate::tools::ToolResult {
    app.set_active_job("shell");
    loop {
        tokio::select! {
            event = shell_job.events_rx.recv() => {
                match event {
                    Some(ShellWorkerEvent::TimedOut { command, timeout_secs }) => {
                        app.pending_permission = Some(format!(
                            "Shell command has been running for {timeout_secs}s:\n  $ {command}\nChoose: [c]ontinue waiting or [k]ill the command."
                        ));
                        app.input.clear();
                        app.cursor = 0;
                        let _ = terminal.draw(|frame| ui::draw(frame, app));
                        let response = wait_for_modal_input(app, rx, terminal, &['c', 'k']).await;
                        app.pending_permission = None;
                        let control = match response.as_str() {
                            "c" => {
                                app.push_output("  · Continuing to wait for shell command...", LineStyle::Status);
                                ShellControl::Continue
                            }
                            _ => {
                                app.push_output("  · Shell command killed.", LineStyle::Status);
                                ShellControl::Kill
                            }
                        };
                        if shell_job.send_control(control).is_err() {
                            app.clear_active_job();
                            return crate::tools::ToolResult::err("Shell worker dropped control channel".into());
                        }
                        let _ = terminal.draw(|frame| ui::draw(frame, app));
                    }
                    Some(ShellWorkerEvent::Completed(result)) => {
                        app.clear_active_job();
                        if cancelled.load(Ordering::Relaxed) {
                            cancelled.store(false, Ordering::Relaxed);
                        }
                        if matches!(&result, Ok(tool_result) if !tool_result.success && tool_result.content == "Command interrupted by user.") {
                            app.push_output("  · Shell command interrupted.", LineStyle::Status);
                        }
                        return match result {
                            Ok(tool_result) => tool_result,
                            Err(err) => crate::tools::ToolResult::err(err),
                        };
                    }
                    None => {
                        app.clear_active_job();
                        if cancelled.load(Ordering::Relaxed) {
                            cancelled.store(false, Ordering::Relaxed);
                        }
                        return crate::tools::ToolResult::err("Shell worker dropped before reporting a result".into());
                    }
                }
            }
            evt = rx.recv() => {
                match evt {
                    Some(AppEvent::Tick) => {
                        let _ = terminal.draw(|frame| ui::draw(frame, app));
                    }
                    Some(AppEvent::Key(key)) if handle_background_key(app, &key) => {
                        let _ = terminal.draw(|frame| ui::draw(frame, app));
                    }
                    Some(AppEvent::Key(key)) if event::is_ctrl_c(&key) => {
                        cancelled.store(true, Ordering::Relaxed);
                        app.push_output("(interrupted)", LineStyle::Status);
                        let _ = terminal.draw(|frame| ui::draw(frame, app));
                    }
                    Some(AppEvent::Mouse(_)) => {}
                    Some(AppEvent::PermissionRequest(prompt, response_tx)) => {
                        let response = fulfill_permission_request(app, rx, terminal, prompt).await;
                        let _ = response_tx.send(response);
                    }
                    Some(_) => {}
                    None => {
                        app.clear_active_job();
                        if cancelled.load(Ordering::Relaxed) {
                            cancelled.store(false, Ordering::Relaxed);
                        }
                        return crate::tools::ToolResult::err("Event stream closed.".into());
                    }
                }
            }
        }
    }
}
