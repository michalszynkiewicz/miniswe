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
use crate::context::compress;
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
    // Context tools — pull-based access to repo map, profile, architecture notes
    tool_defs.extend(tools::definitions::context_tool_definitions());

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

    // Add LSP tool definitions if available
    if lsp_client.is_some() {
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

    let max_rounds = config.context.max_rounds;
    let pause_at = config.context.pause_after_rounds;

    let mut conversation_history: Vec<Message> = Vec::new();
    let mut round = 0;
    let mut had_error = false;
    let mut user_continued = false;

    // Track tool calls for loop detection
    let mut recent_calls: Vec<String> = Vec::new();
    let mut consecutive_loops = 0u32;

    // Track tool results for observation masking
    let mut tool_result_log: Vec<(String, serde_json::Value, String)> = Vec::new();

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

        // Apply observation masking: replace old tool results with summaries
        // Budget = half the context window for tool results
        let tool_result_budget = config.model.context_window / 2;
        let pre_mask = tool_result_log.len();
        mask_old_tool_results(
            &mut messages,
            &tool_result_log,
            tool_result_budget,
            &config.project_root,
            &router,
        ).await;
        log.masking_applied(
            messages.iter().filter(|m| m.role == "tool" && m.content.as_ref().is_some_and(|c| c.starts_with('['))).count(),
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
                match tools::execute_tool(&tc.function.name, &args, &config, &perms, lsp_client.as_deref()).await {
                    Ok(r) => r,
                    Err(e) => crate::tools::ToolResult::err(format!("Tool error: {e}")),
                }
            };

            let first_line = result.content.lines().next().unwrap_or("(empty)");
            log.tool_call(&tc.function.name, &args_summary, result.success, first_line);
            log.tool_result_detail(&tc.function.name, result.success, &result.content);
            tui::print_tool_result(&tc.function.name, result.success, first_line);

            // A successful edit/write means code changed — the next shell/test
            // call is on different code, so it's not a loop. Reset the tracker.
            if result.success
                && (tc.function.name == "edit" || tc.function.name == "write_file")
            {
                recent_calls.clear();
            }

            // Log for future observation masking
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
/// Token-budget-aware observation masking with LLM-driven summarization.
///
/// When tool results exceed the token budget, the oldest results are
/// batched and sent to the LLM for summarization. The LLM decides
/// what's important and returns concise summaries. Full content is
/// archived to `.miniswe/tool_history.md` for retrieval.
///
/// Falls back to heuristic summaries if the LLM call fails.
async fn mask_old_tool_results(
    messages: &mut Vec<Message>,
    tool_result_log: &[(String, serde_json::Value, String)],
    tool_result_token_budget: usize,
    project_root: &std::path::Path,
    router: &ModelRouter,
) {
    if tool_result_log.is_empty() {
        return;
    }

    // Walk backwards (newest first), accumulate tokens
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

    // Collect results that need summarization
    let to_summarize: Vec<(usize, &str, &serde_json::Value, &str)> = tool_result_log
        .iter()
        .enumerate()
        .filter(|(i, _)| should_mask[*i])
        .map(|(i, (name, args, content))| (i, name.as_str(), args, content.as_str()))
        .collect();

    // Try LLM-driven summarization
    let llm_summaries = llm_summarize_tool_results(&to_summarize, router).await;

    // Build final summaries — LLM if available, heuristic fallback
    let summaries: Vec<Option<String>> = tool_result_log
        .iter()
        .enumerate()
        .map(|(i, (name, args, content))| {
            if !should_mask[i] {
                return None;
            }
            // Find this index in llm_summaries
            if let Some(llm_summary) = llm_summaries.as_ref() {
                let pos = to_summarize.iter().position(|(idx, _, _, _)| *idx == i);
                if let Some(pos) = pos {
                    if let Some(summary) = llm_summary.get(pos) {
                        if !summary.is_empty() && summary != "[drop]" {
                            return Some(summary.clone());
                        }
                        if summary == "[drop]" {
                            return Some("[dropped by summarizer]".into());
                        }
                    }
                }
            }
            // Fallback to heuristic
            Some(compress::summarize_tool_result(name, args, content))
        })
        .collect();

    // Archive full content of masked results
    let archive_path = project_root.join(".miniswe").join("tool_history.md");
    let mut archive = std::fs::read_to_string(&archive_path).unwrap_or_default();
    for (i, (name, args, content)) in tool_result_log.iter().enumerate() {
        if should_mask[i] {
            let path = args["path"].as_str().or(args["query"].as_str()).unwrap_or("?");
            archive.push_str(&format!("## {}({})\n{}\n\n", name, path,
                if content.len() > 2000 { &content[..2000] } else { content }));
        }
    }
    let _ = std::fs::write(&archive_path, &archive);

    // Apply summaries to messages
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

/// Ask the LLM to summarize a batch of tool results.
/// Returns None if the call fails (caller falls back to heuristic).
async fn llm_summarize_tool_results(
    results: &[(usize, &str, &serde_json::Value, &str)],
    router: &ModelRouter,
) -> Option<Vec<String>> {
    if results.is_empty() {
        return Some(Vec::new());
    }

    // Build the summarization prompt
    let mut prompt = String::from(
        "Summarize these tool results from earlier in a coding session. \
         For each, write ONE concise line capturing what's important \
         (key functions, types, patterns found). \
         Write [drop] if the result has no future value.\n\
         Respond with exactly one line per result, numbered.\n\n"
    );

    for (idx, (_, name, args, content)) in results.iter().enumerate() {
        let path = args["path"].as_str()
            .or(args["query"].as_str())
            .or(args["command"].as_str())
            .unwrap_or("?");
        // Truncate content to keep prompt manageable
        let truncated = if content.len() > 1000 {
            &content[..1000]
        } else {
            content
        };
        prompt.push_str(&format!("{}. {}({}):\n{}\n\n", idx + 1, name, path, truncated));
    }

    let request = ChatRequest {
        messages: vec![
            Message::system("You are a concise summarizer. Respond with numbered lines only."),
            Message::user(&prompt),
        ],
        tools: None,
        tool_choice: None,
    };

    // Use Fast role if available, otherwise Default
    let cancelled = Arc::new(AtomicBool::new(false));
    let response = router
        .chat_stream(ModelRole::Fast, &request, |_| {}, &cancelled)
        .await
        .ok()?;

    let text = response.choices.first()?.message.content.as_deref()?;

    // Parse numbered lines
    let summaries: Vec<String> = text
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            // Strip leading number + period/parenthesis
            let content = trimmed
                .trim_start_matches(|c: char| c.is_ascii_digit())
                .trim_start_matches(['.', ')', ':', ' ']);
            if content.is_empty() {
                None
            } else {
                Some(content.trim().to_string())
            }
        })
        .collect();

    // Must match the number of results
    if summaries.len() == results.len() {
        Some(summaries)
    } else {
        None // Mismatch — fall back to heuristic
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
        "edit" | "write_file" => {
            let path = args["path"].as_str().unwrap_or("?");
            format!("{path}")
        }
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
