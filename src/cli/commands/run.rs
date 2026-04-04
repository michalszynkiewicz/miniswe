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
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;

use crate::config::{Config, ModelRole};
use crate::context;
use crate::llm::{ChatRequest, Message, ModelRouter};
use crate::logging::SessionLog;
use crate::lsp::LspClient;
use crate::mcp::{McpConfig, McpRegistry};
use crate::tools;
use crate::tools::permissions::{Action, PermissionManager};
use crate::tui;


/// Run the agent for a single message.
pub async fn run(config: Config, message: &str, plan_only: bool, headless: bool) -> Result<()> {
    let log = SessionLog::new(&config);
    log.user_message(message);

    let router = ModelRouter::new(&config);
    let perms = if headless {
        PermissionManager::headless(&config)
    } else {
        PermissionManager::new(&config)
    };
    let mut tool_defs = tools::tool_definitions();
    // Filter tools based on config
    let disabled_tools: Vec<&str> = {
        let mut d = Vec::new();
        if !config.tools.context_tools {
            d.extend_from_slice(&["get_repo_map", "get_project_info", "get_architecture_notes"]);
        }
        if !config.tools.transform {
            d.push("replace_all");
        }
        if !config.tools.web_tools {
            d.extend_from_slice(&["web_search", "web_fetch", "docs_lookup"]);
        }
        d
    };
    tool_defs.retain(|t| !disabled_tools.contains(&t.function.name.as_str()));

    // Conditional: context tools
    if config.tools.context_tools {
        tool_defs.extend(tools::definitions::context_tool_definitions());
    }

    // Clear stale scratchpad/plan from previous sessions
    let _ = std::fs::remove_file(config.miniswe_path("scratchpad.md"));
    let _ = std::fs::remove_file(config.miniswe_path("plan.md"));

    tui::print_header(if plan_only {
        "Plan Mode (read-only)"
    } else {
        "miniswe"
    });

    // Show model info
    for line in router.startup_summary() {
        tui::print_status(&line);
    }
    if !router.is_multi_model() {
        tui::print_status(
            "Tip: configure [models] in config.toml with llama-swap for multi-model routing (plan/code/fast)"
        );
    }

    // Select model role: plan mode uses the plan model, normal mode uses default
    let model_role = if plan_only { ModelRole::Plan } else { ModelRole::Default };

    // Spawn LSP client (non-blocking — initializes in background)
    let lsp_client: Option<Arc<LspClient>> = if config.lsp.enabled {
        match LspClient::spawn(config.project_root.clone()).await {
            Ok(client) => {
                tui::print_status("LSP: rust-analyzer starting...");
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

    // Add LSP tool definitions if available and enabled
    if lsp_client.is_some() && config.tools.lsp_tools {
        tool_defs.extend(tools::definitions::lsp_tool_definitions());
    }

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
                    // Add mcp_use tool definition
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

    // Estimate tool definition overhead for context budgeting
    let tool_def_tokens = context::estimate_tokens(
        &serde_json::to_string(&tool_defs).unwrap_or_default()
    );

    let max_rounds = config.context.max_rounds;
    let pause_at = config.context.pause_after_rounds;

    let mut conversation_history: Vec<Message> = Vec::new();
    let mut round = 0;
    let mut had_error = false;
    let mut user_continued = false;

    // Track tool calls for loop detection
    let mut recent_calls: Vec<String> = Vec::new();
    let mut consecutive_loops = 0u32;
    let mut calls_since_last_edit = 0u32;
    let mut edit_fail_count: std::collections::HashMap<String, u32> = std::collections::HashMap::new();


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

    // Initialize snapshot manager for revert support
    let mut snapshots = tools::snapshots::SnapshotManager::init(&config.project_root)
        .ok();

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

    loop {
        if had_error {
            break;
        }
        round += 1;
        log.round_start(round);

        // Snapshot at start of each round for revert support
        if let Some(ref mut snap) = snapshots {
            let _ = snap.begin_round(round);
        }
        if round > max_rounds {
            tui::print_error("Maximum tool rounds reached. Stopping.");
            break;
        }

        // Ask user if they want to continue after pause_after_rounds
        if round == pause_at && !user_continued {
            tui::print_status(&format!("{pause_at} tool rounds used."));
            let response = tui::read_input("Continue? [y]es / [n]o:");
            match response.as_deref() {
                Some("y") | Some("yes") | Some("") => {
                    user_continued = true;
                }
                _ => {
                    // Tell the LLM to wrap up
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

        // Unified context compression — handles both tool results and conversation
        let pre_mask = messages.len();
        context::compressor::maybe_compress(&mut messages, &config, &router, tool_def_tokens).await;
        log.masking_applied(
            pre_mask.saturating_sub(messages.len()),
            pre_mask,
        );

        // Sanitize message roles before sending (strict chat template compat)
        context::sanitize_messages(&mut messages);

        // Call LLM with streaming
        let request = ChatRequest {
            messages: messages.clone(),
            tools: Some(tool_defs.clone()),
            tool_choice: None,
        };

        tui::print_separator();

        // Reset cancel flag for this round
        cancelled.store(false, Ordering::Relaxed);

        eprint!("\x1b[2m⠋ thinking...\x1b[0m");
        std::io::stderr().flush().ok();
        let thinking = Arc::new(AtomicBool::new(true));

        let response = match router
            .chat_stream(model_role, &request, |token| {
                if thinking.load(Ordering::Relaxed) {
                    thinking.store(false, Ordering::Relaxed);
                    eprint!("\r\x1b[2K");
                    std::io::stderr().flush().ok();
                }
                tui::print_token(token);
            }, &cancelled)
            .await
        {
            Ok(r) => {
                if thinking.load(Ordering::Relaxed) {
                    eprint!("\r\x1b[2K");
                    std::io::stderr().flush().ok();
                }
                r
            }
            Err(e) => {
                eprint!("\r\x1b[2K");
                std::io::stderr().flush().ok();
                let err_str = e.to_string();
                if err_str.contains("Interrupted") {
                    tui::print_status("Generation interrupted.");
                    break;
                }
                let clean = if err_str.contains('<') {
                    err_str.split('<').next().unwrap_or(&err_str).trim().to_string()
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

        // Get the assistant's response
        let choice = match response.choices.first() {
            Some(c) => c,
            None => {
                tui::print_error("Empty response from LLM");
                break;
            }
        };

        let assistant_msg = &choice.message;

        // Print newline after streaming content
        if assistant_msg.content.is_some() {
            println!();
        }

        // Log and add assistant message to history
        if let Some(content) = &assistant_msg.content {
            log.llm_response(content);
        }
        conversation_history.push(assistant_msg.clone());

        // Check for tool calls
        let tool_calls = match &assistant_msg.tool_calls {
            Some(tc) if !tc.is_empty() => tc.clone(),
            _ => {
                if let Some(finish) = &choice.finish_reason {
                    if finish == "stop" {
                        break;
                    }
                }
                break;
            }
        };

        // Add assistant's tool call message to messages
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

            // Detect tool call loops
            let call_key = format!("{}:{}", tc.function.name, args_summary);
            recent_calls.push(call_key.clone());
            if recent_calls.len() > 6 {
                recent_calls.remove(0);
            }
            let repeat_count = recent_calls.iter().filter(|c| *c == &call_key).count();
            if repeat_count >= 3 {
                log.loop_detected(&tc.function.name, &args_summary, repeat_count);
                tui::print_error(&format!(
                    "Loop detected: {}({}) called {} times — stopping",
                    tc.function.name, args_summary, repeat_count
                ));
                consecutive_loops += 1;
                let result_msg = Message::tool_result(
                    &tc.id,
                    "ERROR: You are in a loop — this exact tool call has been repeated 3 times. Try a completely different approach. Remember: you have tools like get_repo_map(), search(), read_file(), write_file(), diagnostics() available.",
                );
                messages.push(result_msg.clone());
                conversation_history.push(result_msg);
                if consecutive_loops >= 3 {
                    tui::print_error("Too many consecutive loops — ending session");
                    had_error = true;
                    // Break inner tool loop; outer round loop checks had_error
                    break;
                }
                continue;
            } else {
                consecutive_loops = 0;
            }

            log.tool_call_detail(&tc.function.name, &args);
            tui::print_tool_call(&tc.function.name, &args_summary);

            // Block edit tools in plan-only mode
            if plan_only
                && (tc.function.name == "edit"
                    || tc.function.name == "write_file"
                    || tc.function.name == "shell")
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

            // Handle special tool calls that need extra context
            let mut result = if tc.function.name == "revert" {
                let to_round = args["to_round"].as_u64().unwrap_or(0) as usize;
                let path = args["path"].as_str().unwrap_or("");
                match &snapshots {
                    Some(snap) => {
                        let res = if !path.is_empty() {
                            snap.revert_file(path, to_round)
                        } else {
                            snap.revert_to_round(to_round)
                        };
                        match res {
                            Ok(msg) => crate::tools::ToolResult::ok(msg),
                            Err(e) => crate::tools::ToolResult::err(format!("Revert failed: {e}")),
                        }
                    }
                    None => crate::tools::ToolResult::err("Snapshot system not available (git not found?)".into()),
                }
            } else if tc.function.name == "fix_file" {
                match tools::fix_file::execute(&args, &config, &router).await {
                    Ok(r) => r,
                    Err(e) => crate::tools::ToolResult::err(format!("fix_file error: {e}")),
                }
            } else if tc.function.name == "replace_all" {
                match tools::transform::execute(&args, &config).await {
                    Ok(r) => r,
                    Err(e) => crate::tools::ToolResult::err(format!("replace_all error: {e}")),
                }
            } else if tc.function.name == "mcp_use" {
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
                match tools::execute_tool(&tc.function.name, &args, &config, &perms, lsp_client.as_deref()).await {
                    Ok(r) => r,
                    Err(e) => crate::tools::ToolResult::err(format!("Tool error: {e}")),
                }
            };

            let first_line = result.content.lines().next().unwrap_or("(empty)");
            log.tool_call(&tc.function.name, &args_summary, result.success, first_line);
            log.tool_result_detail(&tc.function.name, result.success, &result.content);
            tui::print_tool_result(&tc.function.name, result.success, first_line);

            // A successful edit/write means code changed — reset trackers.
            if result.success
                && (tc.function.name == "edit" || tc.function.name == "write_file")
            {
                recent_calls.clear();
                calls_since_last_edit = 0;
            } else {
                calls_since_last_edit += 1;
            }

            // Track edit failures per file — suggest write_file after 2 failures
            if tc.function.name == "edit" && !result.success {
                let path = args["path"].as_str().unwrap_or("").to_string();
                let count = edit_fail_count.entry(path.clone()).or_insert(0);
                *count += 1;
                if *count >= 2 {
                    result.content.push_str(&format!(
                        "\nERROR: edit has failed {} times on {path}. STOP using edit. Use write_file instead — read the file first, then write the complete new content.\n",
                        count
                    ));
                }
            }

            let result_msg = Message::tool_result(&tc.id, &result.content);
            messages.push(result_msg.clone());
            conversation_history.push(result_msg);
        }

        // Stall detection: too many tool calls without any edits
        if calls_since_last_edit >= 20 && calls_since_last_edit % 20 == 0 {
            messages.push(Message::user(
                "[WARNING: You have used 20+ tool calls without making any edits. \
                 You likely have enough information. Start making changes now. \
                 Use write_file for files under 200 lines. \
                 If you're stuck, explain what's blocking you.]"
            ));
        }
    }

    log.session_end(round, had_error);

    // Shut down LSP
    if let Some(lsp) = lsp_client {
        if let Ok(lsp) = Arc::try_unwrap(lsp) {
            lsp.shutdown().await;
        }
    }

    tui::print_separator();
    if !had_error {
        tui::print_complete("Done");
    }

    Ok(())
}

/// Replace old tool results with compressed summaries using per-type thresholds.
///
/// For each tool type, keeps the N most recent results in full (where N
/// depends on the tool type — reads are kept longer than writes/searches).
/// Older results are replaced with one-line summaries.

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
        "edit" | "write_file" => {
            let path = args["path"].as_str().unwrap_or("?");
            format!("{path}")
        }
        "shell" => {
            let cmd = args["command"].as_str().unwrap_or("?");
            crate::truncate_chars(cmd, 47)
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
