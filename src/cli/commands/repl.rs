//! Interactive REPL mode with ratatui TUI.

use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use crossterm::event::{KeyCode, KeyModifiers};
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::ExecutableCommand;
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use tokio::sync::mpsc;

use crate::config::{Config, ModelRole};
use crate::context;
use crate::context::compress;
use crate::llm::{ChatRequest, Message, ModelRouter};
use crate::logging::SessionLog;
use crate::lsp::LspClient;
use crate::mcp::{McpConfig, McpRegistry};
use crate::tools;
use crate::tools::permissions::{Action, PermissionManager};
use crate::tui::app::{App, AppMode, LineStyle};
use crate::tui::event::{self, AppEvent};
use crate::tui::ui;

/// Per-tool-type masking thresholds: how many recent full results to keep.
fn mask_keep_count(tool_name: &str) -> usize {
    match tool_name {
        "read_file" | "read_symbol" => 3,
        "write_file" | "edit" => 2,
        "shell" | "diagnostics" => 2,
        "search" | "web_search" | "web_fetch" | "docs_lookup"
        | "goto_definition" | "find_references" => 1,
        _ => 2,
    }
}

/// Run the interactive REPL with TUI.
pub async fn run(config: Config, headless: bool) -> Result<()> {
    let log = SessionLog::new(&config);

    let router = ModelRouter::new(&config);
    let perms = if headless {
        PermissionManager::headless(&config)
    } else {
        PermissionManager::new(&config)
    };
    let mut tool_defs = tools::tool_definitions();

    // Spawn LSP client (non-blocking)
    let lsp_client: Option<Arc<LspClient>> = if config.lsp.enabled {
        match LspClient::spawn(config.project_root.clone()).await {
            Ok(client) => Some(Arc::new(client)),
            Err(_) => None,
        }
    } else {
        None
    };
    if lsp_client.is_some() {
        tool_defs.extend(tools::definitions::lsp_tool_definitions());
    }

    // Clear stale scratchpad/plan
    let _ = std::fs::remove_file(config.miniswe_path("scratchpad.md"));
    let _ = std::fs::remove_file(config.miniswe_path("plan.md"));

    // Initialize MCP
    let mcp_config = McpConfig::load(&config.project_root)?;
    let mut mcp_registry = if mcp_config.has_servers() {
        let cache_dir = config.miniswe_path("mcp");
        match McpRegistry::connect(&mcp_config, &cache_dir) {
            Ok(registry) => {
                if registry.has_servers() {
                    tool_defs.push(tools::definitions::mcp_tool_definition());
                }
                Some(registry)
            }
            Err(_) => None,
        }
    } else {
        None
    };

    let mcp_summary = mcp_registry.as_ref().and_then(|r| r.context_summary());

    // Set up terminal
    terminal::enable_raw_mode()?;
    io::stdout().execute(EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;

    // Set up app state
    let mut app = App::new();
    let history_file = config.miniswe_path("sessions/repl_history.txt");
    app.load_history(&history_file);

    // Welcome message
    for line in router.startup_summary() {
        app.push_output(&format!("miniswe — {line}"), LineStyle::Status);
    }
    if !router.is_multi_model() {
        app.push_output(
            "Tip: configure [models] in config.toml with llama-swap for multi-model routing",
            LineStyle::Status,
        );
    }
    if let Some(ref mcp) = mcp_registry {
        if mcp.has_servers() {
            app.push_output(
                &format!("MCP: {} servers, {} tools", mcp.servers.len(), mcp.tool_count()),
                LineStyle::Status,
            );
        }
    }
    app.push_output(
        "Type your message. Ctrl+O: details, Ctrl+C: interrupt, Ctrl+D: quit",
        LineStyle::Status,
    );
    app.push_output("────────────────────────────────────────────────", LineStyle::Separator);

    // Event channel
    let (tx, mut rx) = mpsc::unbounded_channel::<AppEvent>();

    // Cancellation flag for LLM
    let cancelled = Arc::new(AtomicBool::new(false));

    // Spawn keyboard reader (passes cancel flag for direct Ctrl+C handling)
    event::spawn_key_reader(tx.clone(), cancelled.clone());

    let mut conversation_history: Vec<Message> = Vec::new();

    // Main event loop
    loop {
        // Render
        terminal.draw(|frame| ui::draw(frame, &app))?;

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
                            KeyCode::Enter => {
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
                                        let _ = std::fs::remove_file(config.miniswe_path("scratchpad.md"));
                                        let _ = std::fs::remove_file(config.miniswe_path("plan.md"));
                                        app.push_output("Cleared history, scratchpad, and plan.", LineStyle::Status);
                                    } else {
                                        app.push_output("Cleared conversation history.", LineStyle::Status);
                                    }
                                    continue;
                                }

                                if input == "/help" {
                                    app.push_output("/clear — clear conversation history", LineStyle::Status);
                                    app.push_output("/new   — clear history + scratchpad + plan", LineStyle::Status);
                                    app.push_output("/help  — show this help", LineStyle::Status);
                                    app.push_output("quit   — exit", LineStyle::Status);
                                    continue;
                                }

                                // Show user message in output
                                app.push_output(&format!("you> {input}"), LineStyle::Normal);

                                // Run the agent loop
                                let mcp_summary_clone = mcp_summary.clone();

                                // Assemble context
                                let assembled = context::assemble(
                                    &config,
                                    &input,
                                    &conversation_history,
                                    false,
                                    mcp_summary_clone.as_deref(),
                                );
                                conversation_history.push(Message::user(&input));

                                app.is_thinking = true;

                                // Spawn agent loop as async task
                                let mut messages = assembled.messages;
                                let max_rounds = config.context.max_rounds;
                                let perms_ref = &perms;
                                let mcp_ref = &mut mcp_registry;
                                let conv_ref = &mut conversation_history;

                                log.user_message(&input);

                                // Run agent loop inline (not spawned — needs mutable refs)
                                run_agent_loop(
                                    &mut app,
                                    &mut rx,
                                    &mut terminal,
                                    &router,
                                    &tool_defs,
                                    &config,
                                    perms_ref,
                                    mcp_ref,
                                    &cancelled,
                                    &mut messages,
                                    conv_ref,
                                    max_rounds,
                                    &log,
                                    &lsp_client,
                                ).await;

                                app.is_thinking = false;
                                app.flush_tokens();
                                app.push_output("────────────────────────────────────────────────", LineStyle::Separator);

                                // Trim history
                                let max_history = config.context.history_turns * 6;
                                if conversation_history.len() > max_history {
                                    let drain_count = conversation_history.len() - max_history;
                                    conversation_history.drain(..drain_count);
                                }
                            }
                            KeyCode::Backspace => app.delete_char(),
                            KeyCode::Left => app.cursor_left(),
                            KeyCode::Right => app.cursor_right(),
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

            // Events from agent loop
            AppEvent::Token(token) => {
                app.push_token(&token);
            }
            AppEvent::ToolCall(name, summary) => {
                app.push_output(&format!("  → {name}({summary})"), LineStyle::ToolCall);
            }
            AppEvent::ToolResult(name, success, summary, full_content) => {
                let style = if success { LineStyle::ToolOk } else { LineStyle::ToolErr };
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
    terminal::disable_raw_mode()?;
    io::stdout().execute(LeaveAlternateScreen)?;

    Ok(())
}

/// Run the agent loop (LLM call → tool execution → repeat).
/// This runs inline in the main loop, processing events between rounds.
async fn run_agent_loop(
    app: &mut App,
    rx: &mut mpsc::UnboundedReceiver<AppEvent>,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    router: &ModelRouter,
    tool_defs: &[crate::llm::ToolDefinition],
    config: &Config,
    perms: &PermissionManager,
    mcp_registry: &mut Option<McpRegistry>,
    cancelled: &Arc<AtomicBool>,
    messages: &mut Vec<Message>,
    conversation_history: &mut Vec<Message>,
    max_rounds: usize,
    log: &SessionLog,
    lsp: &Option<Arc<LspClient>>,
) {
    let mut round = 0;
    let mut tool_result_log: Vec<(String, serde_json::Value, String)> = Vec::new();
    // Track recent tool calls to detect loops
    let mut recent_calls: Vec<String> = Vec::new();

    loop {
        // Check cancellation at the top of every round
        if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
            app.push_output("(interrupted)", LineStyle::Status);
            break;
        }

        round += 1;
        log.round_start(round);
        if round > max_rounds {
            app.push_output("Maximum tool rounds reached.", LineStyle::Error);
            break;
        }

        // Observation masking (per-type thresholds)
        mask_old_tool_results(messages, &tool_result_log);

        // Sanitize messages
        context::sanitize_messages(messages);

        // Build request
        let request = ChatRequest {
            messages: messages.clone(),
            tools: Some(tool_defs.to_vec()),
            tool_choice: None,
        };

        cancelled.store(false, Ordering::Relaxed);
        app.is_thinking = true;

        // Render before LLM call so spinner is visible immediately
        let _ = terminal.draw(|frame| ui::draw(frame, app));

        // Call LLM with streaming — render on each token
        let response = {
            let mut token_count = 0u32;
            match router.chat_stream(ModelRole::Default, &request, |token| {
                app.push_token(token);
                // Re-render every few tokens (not every token — too expensive)
                token_count += 1;
                if token_count % 3 == 0 {
                    let _ = terminal.draw(|frame| ui::draw(frame, app));
                }
            }, cancelled).await {
                Ok(r) => r,
                Err(e) => {
                    let err_str = e.to_string();
                    if err_str.contains("Interrupted") {
                        app.push_output("Generation interrupted.", LineStyle::Status);
                    } else {
                        let clean = if err_str.contains('<') {
                            err_str.split('<').next().unwrap_or(&err_str).trim().to_string()
                        } else {
                            err_str
                        };
                        log.llm_error(&clean);
                        app.push_output(&format!("LLM error: {clean}"), LineStyle::Error);
                    }
                    break;
                }
            }
        };

        // Re-render after LLM response
        let _ = terminal.draw(|frame| ui::draw(frame, app));

        let choice = match response.choices.first() {
            Some(c) => c,
            None => break,
        };

        let assistant_msg = &choice.message;

        // Flush any remaining tokens
        app.flush_tokens();

        if let Some(content) = &assistant_msg.content {
            log.llm_response(content);
        }
        conversation_history.push(assistant_msg.clone());

        let tool_calls = match &assistant_msg.tool_calls {
            Some(tc) if !tc.is_empty() => tc.clone(),
            _ => break,
        };

        messages.push(assistant_msg.clone());

        // Execute tool calls
        for tc in &tool_calls {
            // Check cancellation between tool calls
            if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
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

            // Detect tool call loops (same call repeated 3+ times in last 6 calls)
            let call_key = format!("{}:{}", tc.function.name, args_summary);
            recent_calls.push(call_key.clone());
            if recent_calls.len() > 6 {
                recent_calls.remove(0);
            }
            let repeat_count = recent_calls.iter().filter(|c| *c == &call_key).count();
            if repeat_count >= 3 {
                log.loop_detected(&tc.function.name, &args_summary, repeat_count);
                app.push_output(
                    &format!("  ✗ Loop detected: {}({}) called {} times — stopping", tc.function.name, args_summary, repeat_count),
                    LineStyle::Error,
                );
                let result_msg = Message::tool_result(
                    &tc.id,
                    "ERROR: You are in a loop — this exact tool call has been repeated 3 times with no edits in between. Try a different approach, or explain what's blocking you.",
                );
                messages.push(result_msg.clone());
                conversation_history.push(result_msg);
                // Don't break — let the model see the error and hopefully stop
                continue;
            }

            log.tool_call_detail(&tc.function.name, &args);
            app.push_output(
                &format!("  → {}({})", tc.function.name, args_summary),
                LineStyle::ToolCall,
            );

            // Re-render to show tool call
            let _ = terminal.draw(|frame| ui::draw(frame, app));

            // Determine if this tool call needs a permission prompt
            let perm_action = match tc.function.name.as_str() {
                "shell" => Some(Action::Shell(args["command"].as_str().unwrap_or("").into())),
                "web_search" => Some(Action::WebSearch(args["query"].as_str().unwrap_or("").into())),
                "web_fetch" => Some(Action::WebFetch(args["url"].as_str().unwrap_or("").into())),
                "mcp_use" => Some(Action::McpUse(
                    args["server"].as_str().unwrap_or("").into(),
                    args["tool"].as_str().unwrap_or("").into(),
                )),
                _ => None,
            };

            // Check permission via TUI prompt (not raw stderr)
            let mut perm_denied = false;
            if let Some(ref action) = perm_action {
                match perms.check_needs_prompt(action) {
                    Err(e) => {
                        // Blocklisted — skip this tool call
                        let result_msg = Message::tool_result(&tc.id, &e);
                        messages.push(result_msg.clone());
                        conversation_history.push(result_msg);
                        app.push_output(&format!("  ✗ {}: {e}", tc.function.name), LineStyle::ToolErr);
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
                            }
                            "a" | "always" => {
                                perms.approve(action, true);
                            }
                            _ => {
                                perm_denied = true;
                            }
                        }

                        let _ = terminal.draw(|frame| ui::draw(frame, app));
                    }
                    Ok(None) => {} // No prompt needed
                }
            }

            if perm_denied {
                let result_msg = Message::tool_result(
                    &tc.id,
                    &format!("{} denied by user", tc.function.name),
                );
                messages.push(result_msg.clone());
                conversation_history.push(result_msg);
                app.push_output(
                    &format!("  ✗ {}: denied", tc.function.name),
                    LineStyle::ToolErr,
                );
                continue;
            }

            // Execute tool (permissions already checked above for shell/web/mcp)
            let result = if tc.function.name == "mcp_use" {
                let server = args["server"].as_str().unwrap_or("");
                let tool = args["tool"].as_str().unwrap_or("");
                let tool_args = args.get("arguments").cloned().unwrap_or_default();
                match mcp_registry {
                    Some(registry) => match registry.call_tool(server, tool, tool_args) {
                        Ok(content) => crate::tools::ToolResult::ok(content),
                        Err(e) => crate::tools::ToolResult::err(format!("MCP error: {e}")),
                    },
                    None => crate::tools::ToolResult::err("No MCP servers connected".into()),
                }
            } else {
                match tools::execute_tool(&tc.function.name, &args, config, perms, lsp.as_deref()).await {
                    Ok(r) => r,
                    Err(e) => crate::tools::ToolResult::err(format!("Tool error: {e}")),
                }
            };

            let first_line = result.content.lines().next().unwrap_or("(empty)");
            log.tool_call(&tc.function.name, &args_summary, result.success, first_line);
            log.tool_result_detail(&tc.function.name, result.success, &result.content);
            let style = if result.success { LineStyle::ToolOk } else { LineStyle::ToolErr };
            let icon = if result.success { "✓" } else { "✗" };
            app.push_output(&format!("  {icon} {}: {first_line}", tc.function.name), style);
            app.store_tool_result(&tc.function.name, &result.content);

            // Successful edit/write = code changed, reset loop detector
            if result.success
                && (tc.function.name == "edit" || tc.function.name == "write_file")
            {
                recent_calls.clear();
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

/// Replace old tool results with compressed summaries using per-type thresholds.
fn mask_old_tool_results(
    messages: &mut Vec<Message>,
    tool_result_log: &[(String, serde_json::Value, String)],
) {
    if tool_result_log.is_empty() {
        return;
    }

    let mut type_counts: std::collections::HashMap<&str, usize> =
        std::collections::HashMap::new();
    let mut should_mask: Vec<bool> = vec![false; tool_result_log.len()];

    for i in (0..tool_result_log.len()).rev() {
        let tool_name = tool_result_log[i].0.as_str();
        let count = type_counts.entry(tool_name).or_insert(0);
        *count += 1;
        if *count > mask_keep_count(tool_name) {
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

    let mut tool_msg_idx = 0;
    for msg in messages.iter_mut() {
        if msg.role == "tool" {
            if let Some(Some(summary)) = summaries.get(tool_msg_idx) {
                msg.content = Some(summary.clone());
            }
            tool_msg_idx += 1;
        }
    }
}

/// Create a brief summary of tool arguments for display.
fn summarize_args(tool_name: &str, args: &serde_json::Value) -> String {
    match tool_name {
        "read_file" => {
            let path = args["path"].as_str().unwrap_or("?");
            let start = args["start_line"].as_u64();
            let end = args["end_line"].as_u64();
            match (start, end) {
                (Some(s), Some(e)) => format!("{path}:{s}-{e}"),
                (Some(s), None) => format!("{path}:{s}-"),
                _ => path.to_string(),
            }
        }
        "read_symbol" => args["name"].as_str().unwrap_or("?").to_string(),
        "search" => {
            let query = args["query"].as_str().unwrap_or("?");
            let scope = args["scope"].as_str().unwrap_or("project");
            format!("\"{query}\" in {scope}")
        }
        "edit" | "write_file" => args["path"].as_str().unwrap_or("?").to_string(),
        "shell" => {
            let cmd = args["command"].as_str().unwrap_or("?");
            if cmd.len() > 50 {
                format!("{}...", &cmd[..47])
            } else {
                cmd.to_string()
            }
        }
        "task_update" => "scratchpad".to_string(),
        "web_search" => args["query"].as_str().unwrap_or("?").to_string(),
        "web_fetch" => args["url"].as_str().unwrap_or("?").to_string(),
        "docs_lookup" => {
            let lib = args["library"].as_str().unwrap_or("?");
            let topic = args["topic"].as_str().unwrap_or("");
            if topic.is_empty() { lib.to_string() } else { format!("{lib}/{topic}") }
        }
        "mcp_use" => {
            let server = args["server"].as_str().unwrap_or("?");
            let tool = args["tool"].as_str().unwrap_or("?");
            format!("{server}/{tool}")
        }
        _ => format!("{args}"),
    }
}

/// Wait for the user to respond to a permission prompt in the TUI.
/// Blocks until Enter is pressed, returns the trimmed input (e.g., "y", "n", "a").
async fn wait_for_permission_input(
    app: &mut App,
    rx: &mut mpsc::UnboundedReceiver<AppEvent>,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) -> String {
    loop {
        let _ = terminal.draw(|frame| ui::draw(frame, app));

        let evt = match rx.recv().await {
            Some(e) => e,
            None => return "n".into(),
        };

        match evt {
            AppEvent::Key(key) => {
                match key.code {
                    KeyCode::Enter => {
                        let response = app.input.trim().to_lowercase();
                        app.input.clear();
                        app.cursor = 0;
                        return response;
                    }
                    KeyCode::Char(c) => app.insert_char(c),
                    KeyCode::Backspace => app.delete_char(),
                    KeyCode::Esc => {
                        app.input.clear();
                        app.cursor = 0;
                        return "n".into();
                    }
                    _ => {}
                }
            }
            AppEvent::Tick => {} // re-render
            _ => {}
        }
    }
}
