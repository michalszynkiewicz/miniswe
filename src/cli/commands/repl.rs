//! Interactive REPL mode.

use std::io::{self, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use reedline::{
    DefaultPrompt, DefaultPromptSegment, FileBackedHistory, Reedline, Signal,
};

use crate::config::Config;
use crate::context;
use crate::context::compress;
use crate::llm::{ChatRequest, LlmClient, Message};
use crate::mcp::{McpConfig, McpRegistry};
use crate::tools;
use crate::tools::permissions::{Action, PermissionManager};
use crate::tui;

/// Maximum tool rounds per user message.
/// After this many tool results, start masking old ones.
const MASK_AFTER_RESULTS: usize = 6;

/// Run the interactive REPL.
pub async fn run(config: Config, headless: bool) -> Result<()> {
    let client = LlmClient::new(config.model.clone());
    let perms = if headless {
        PermissionManager::headless(&config)
    } else {
        PermissionManager::new(&config)
    };
    let mut tool_defs = tools::tool_definitions();

    // Clear stale scratchpad/plan from previous sessions
    let _ = std::fs::remove_file(config.miniswe_path("scratchpad.md"));
    let _ = std::fs::remove_file(config.miniswe_path("plan.md"));

    tui::print_header("Interactive Mode");
    tui::print_status(&format!(
        "Model: {} @ {}",
        config.model.model, config.model.endpoint
    ));

    // Initialize MCP servers
    let mcp_config = McpConfig::load(&config.project_root)?;
    let mut mcp_registry = if mcp_config.has_servers() {
        let cache_dir = config.miniswe_path("mcp");
        match McpRegistry::connect(&mcp_config, &cache_dir) {
            Ok(registry) => {
                if registry.has_servers() {
                    tui::print_status(&format!(
                        "MCP: {} servers, {} tools",
                        registry.servers.len(),
                        registry.tool_count()
                    ));
                    tool_defs.push(tools::definitions::mcp_tool_definition());
                }
                Some(registry)
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
        .and_then(|r| r.context_summary());

    // Ctrl+C cancellation flag
    let cancelled = Arc::new(AtomicBool::new(false));
    let cancelled_for_handler = cancelled.clone();
    tokio::spawn(async move {
        loop {
            tokio::signal::ctrl_c().await.ok();
            cancelled_for_handler.store(true, Ordering::Relaxed);
            eprintln!("\n\x1b[33m(interrupted — finishing current step)\x1b[0m");
        }
    });

    tui::print_status("Type your message, or 'quit' to exit. Ctrl+R to search history.");
    tui::print_separator();

    // Set up reedline with file-backed history
    let history_file = config.miniswe_path("sessions/repl_history.txt");
    if let Some(parent) = history_file.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let history = Box::new(
        FileBackedHistory::with_file(200, history_file)
            .expect("Failed to create history file"),
    );
    let mut editor = Reedline::create()
        .with_history(history)
        .use_bracketed_paste(true)
        .with_quick_completions(false)
        .with_partial_completions(false);

    let mut conversation_history: Vec<Message> = Vec::new();
    let prompt = DefaultPrompt::new(
        DefaultPromptSegment::Basic("you".to_string()),
        DefaultPromptSegment::Empty,
    );

    loop {
        let input = match editor.read_line(&prompt) {
            Ok(Signal::Success(line)) => {
                let trimmed = line.trim().to_string();
                if trimmed.is_empty() {
                    continue;
                }
                trimmed
            }
            Ok(Signal::CtrlC) => continue,
            Ok(Signal::CtrlD) => break,
            Err(_) => break,
        };

        if input == "quit" || input == "exit" || input == "/quit" {
            break;
        }

        // REPL commands
        if input == "/clear" || input == "/new" {
            conversation_history.clear();
            // Clear scratchpad and plan for a fully fresh start
            if input == "/new" {
                let _ = std::fs::remove_file(config.miniswe_path("scratchpad.md"));
                let _ = std::fs::remove_file(config.miniswe_path("plan.md"));
                tui::print_status("Cleared history, scratchpad, and plan.");
            } else {
                tui::print_status("Cleared conversation history.");
            }
            tui::print_separator();
            continue;
        }

        if input == "/help" {
            tui::print_status("Commands:");
            tui::print_status("  /clear  — clear conversation history");
            tui::print_status("  /new    — clear history + scratchpad + plan (fresh start)");
            tui::print_status("  /help   — show this help");
            tui::print_status("  quit    — exit");
            tui::print_separator();
            continue;
        }

        // Assemble context using history from previous turns
        // (current user message is added by the assembler, not from history)
        let assembled = context::assemble(
            &config,
            &input,
            &conversation_history,
            false,
            mcp_summary.as_deref(),
        );

        // Now add user message to history for future turns
        conversation_history.push(Message::user(&input));

        let mut messages = assembled.messages;
        let max_rounds = config.context.max_rounds;
        let pause_at = config.context.pause_after_rounds;
        let mut round = 0;
        let mut user_continued = false;
        let mut tool_result_log: Vec<(String, serde_json::Value, String)> = Vec::new();

        loop {
            round += 1;
            if round > max_rounds {
                tui::print_error("Maximum tool rounds reached.");
                break;
            }

            // Ask user if they want to continue
            if round == pause_at && !user_continued {
                tui::print_status(&format!("{pause_at} tool rounds used."));
                let response = tui::read_input("Continue? [y]es / [n]o:");
                match response.as_deref() {
                    Some("y") | Some("yes") | Some("") => {
                        user_continued = true;
                    }
                    _ => {
                        messages.push(Message::user(
                            "[Stop now. Summarize what you've done.]"
                        ));
                    }
                }
            }

            // Warn the LLM when approaching the hard limit
            if round == max_rounds.saturating_sub(5) {
                messages.push(Message::user(
                    "[Approaching tool limit. Wrap up and summarize.]"
                ));
            }

            // Observation masking
            if tool_result_log.len() > MASK_AFTER_RESULTS {
                mask_old_tool_results(&mut messages, &tool_result_log);
            }

            // Sanitize message roles before sending (strict chat template compat)
            context::sanitize_messages(&mut messages);

            // Debug: dump role sequence, context size, and LLM memory
            if std::env::var("MINISWE_DEBUG").is_ok() {
                let total_chars: usize = messages.iter().map(|m| {
                    let content_len = m.content.as_deref().map(|c| c.len()).unwrap_or(0);
                    let tc_len: usize = m.tool_calls.as_ref()
                        .map(|tcs| tcs.iter().map(|tc| tc.function.arguments.len() + tc.function.name.len()).sum())
                        .unwrap_or(0);
                    content_len + tc_len
                }).sum();
                let est_tokens = total_chars / 4;
                let budget = config.model.context_window;

                // Try to fetch KV cache usage from llama.cpp /metrics
                let kv_info = fetch_kv_usage(&config.model.endpoint).await;

                eprintln!("\x1b[2m[DEBUG ~{}k/{}k tokens ({} msgs){} roles: {}]\x1b[0m",
                    est_tokens / 1000,
                    budget / 1000,
                    messages.len(),
                    kv_info.as_deref().unwrap_or(""),
                    messages.iter().map(|m| {
                        let r = m.role.as_str();
                        if r == "assistant" && m.tool_calls.is_some() { "a+tc" }
                        else { &r[..1] }
                    }).collect::<Vec<_>>().join("→"));
            }

            let request = ChatRequest {
                messages: messages.clone(),
                tools: Some(tool_defs.clone()),
                tool_choice: None,
            };

            // Reset cancel flag for this round
            cancelled.store(false, Ordering::Relaxed);

            // Show spinner while waiting for LLM (use atomic flag, no thread)
            let thinking = Arc::new(AtomicBool::new(true));
            eprint!("\x1b[2m⠋ thinking...\x1b[0m");
            io::stderr().flush().ok();

            let response = match client
                .chat_stream(&request, |token| {
                    if thinking.load(Ordering::Relaxed) {
                        thinking.store(false, Ordering::Relaxed);
                        eprint!("\r\x1b[2K"); // clear spinner
                        io::stderr().flush().ok();
                    }
                    tui::print_token(token);
                }, &cancelled)
                .await
            {
                Ok(r) => {
                    if thinking.load(Ordering::Relaxed) {
                        eprint!("\r\x1b[2K");
                        io::stderr().flush().ok();
                    }
                    r
                }
                Err(e) => {
                    eprint!("\r\x1b[2K");
                    io::stderr().flush().ok();
                    let err_str = e.to_string();
                    if err_str.contains("Interrupted") {
                        tui::print_status("Generation interrupted.");
                        break;
                    }
                    tui::print_error(&format!("LLM error: {e}"));
                    break;
                }
            };

            let choice = match response.choices.first() {
                Some(c) => c,
                None => break,
            };

            let assistant_msg = &choice.message;
            if assistant_msg.content.is_some() {
                println!();
            }

            conversation_history.push(assistant_msg.clone());

            let tool_calls = match &assistant_msg.tool_calls {
                Some(tc) if !tc.is_empty() => tc.clone(),
                _ => break,
            };

            messages.push(assistant_msg.clone());

            for tc in &tool_calls {
                let args: serde_json::Value = match serde_json::from_str(&tc.function.arguments) {
                    Ok(v) => v,
                    Err(e) => {
                        let result_msg = Message::tool_result(
                            &tc.id,
                            &format!("Invalid JSON in tool arguments: {e}\nRaw: {}", tc.function.arguments),
                        );
                        messages.push(result_msg.clone());
                        conversation_history.push(result_msg);
                        tui::print_tool_result(&tc.function.name, false, "invalid JSON args");
                        continue;
                    }
                };

                let args_summary = summarize_args(&tc.function.name, &args);
                tui::print_tool_call(&tc.function.name, &args_summary);

                // Handle MCP tool calls
                let result = if tc.function.name == "mcp_use" {
                    let server = args["server"].as_str().unwrap_or("");
                    let tool = args["tool"].as_str().unwrap_or("");
                    let tool_args = args.get("arguments").cloned().unwrap_or_default();
                    match perms.check(&Action::McpUse(server.into(), tool.into())) {
                        Err(e) => crate::tools::ToolResult::err(e),
                        Ok(()) => match &mut mcp_registry {
                            Some(registry) => match registry.call_tool(server, tool, tool_args) {
                                Ok(content) => crate::tools::ToolResult::ok(content),
                                Err(e) => crate::tools::ToolResult::err(format!("MCP error: {e}")),
                            },
                            None => crate::tools::ToolResult::err("No MCP servers connected".into()),
                        },
                    }
                } else {
                    match tools::execute_tool(&tc.function.name, &args, &config, &perms).await {
                        Ok(r) => r,
                        Err(e) => crate::tools::ToolResult::err(format!("Tool error: {e}")),
                    }
                };

                let first_line = result.content.lines().next().unwrap_or("(empty)");
                tui::print_tool_result(&tc.function.name, result.success, first_line);

                tool_result_log.push((
                    tc.function.name.clone(),
                    args.clone(),
                    result.content.clone(),
                ));

                let result_msg = Message::tool_result(&tc.id, &result.content);
                messages.push(result_msg.clone());
                conversation_history.push(result_msg);
            }
        }

        // Keep conversation history bounded (use drain instead of repeated remove(0))
        let max_history = config.context.history_turns * 6;
        if conversation_history.len() > max_history {
            let drain_count = conversation_history.len() - max_history;
            conversation_history.drain(..drain_count);
        }

        tui::print_separator();
    }

    tui::print_complete("Session ended");
    Ok(())
}

/// Replace old tool results with compressed summaries.
fn mask_old_tool_results(
    messages: &mut Vec<Message>,
    tool_result_log: &[(String, serde_json::Value, String)],
) {
    let mask_count = tool_result_log.len().saturating_sub(MASK_AFTER_RESULTS);
    if mask_count == 0 {
        return;
    }

    let old_results: Vec<String> = tool_result_log[..mask_count]
        .iter()
        .map(|(name, args, content)| compress::summarize_tool_result(name, args, content))
        .collect();

    let mut tool_msg_count = 0;
    for msg in messages.iter_mut() {
        if msg.role == "tool" {
            if tool_msg_count < mask_count {
                if let Some(summary) = old_results.get(tool_msg_count) {
                    msg.content = Some(summary.clone());
                }
            }
            tool_msg_count += 1;
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
            if topic.is_empty() {
                lib.to_string()
            } else {
                format!("{lib}/{topic}")
            }
        }
        "mcp_use" => {
            let server = args["server"].as_str().unwrap_or("?");
            let tool = args["tool"].as_str().unwrap_or("?");
            format!("{server}/{tool}")
        }
        _ => format!("{args}"),
    }
}

/// Fetch KV cache usage from llama.cpp's /metrics endpoint (best-effort).
/// Returns None silently if the endpoint is unavailable or not llama.cpp.
async fn fetch_kv_usage(endpoint: &str) -> Option<String> {
    let base = endpoint.trim_end_matches('/');
    let url = format!("{base}/metrics");

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_millis(500))
        .build()
        .ok()?;

    let text = client.get(&url).send().await.ok()?.text().await.ok()?;

    // Parse Prometheus format lines
    let mut kv_usage = None;
    let mut kv_tokens = None;
    for line in text.lines() {
        if line.starts_with("llamacpp:kv_cache_usage_ratio ") {
            kv_usage = line.split_whitespace().nth(1)
                .and_then(|v| v.parse::<f64>().ok());
        }
        if line.starts_with("llamacpp:kv_cache_tokens ") {
            kv_tokens = line.split_whitespace().nth(1)
                .and_then(|v| v.parse::<u64>().ok());
        }
    }

    match (kv_usage, kv_tokens) {
        (Some(usage), Some(tokens)) => {
            Some(format!(" kv:{:.0}%/{}tok", usage * 100.0, tokens))
        }
        (Some(usage), None) => {
            Some(format!(" kv:{:.0}%", usage * 100.0))
        }
        _ => None,
    }
}
