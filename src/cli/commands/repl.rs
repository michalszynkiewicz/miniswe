//! Interactive REPL mode.

use std::path::PathBuf;

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

    tui::print_status("Type your message, or 'quit' to exit. Ctrl+R to search history.");
    tui::print_separator();

    // Set up reedline with file-backed history
    let history_file = config.miniswe_path("sessions/repl_history.txt");
    if let Some(parent) = history_file.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let history = Box::new(
        FileBackedHistory::with_file(1000, history_file)
            .expect("Failed to create history file"),
    );
    let mut editor = Reedline::create().with_history(history);
    let prompt = DefaultPrompt::new(
        DefaultPromptSegment::Basic("you".to_string()),
        DefaultPromptSegment::Empty,
    );

    let mut conversation_history: Vec<Message> = Vec::new();

    loop {
        let input = match editor.read_line(&prompt) {
            Ok(Signal::Success(line)) => {
                let trimmed = line.trim().to_string();
                if trimmed.is_empty() {
                    continue;
                }
                trimmed
            }
            Ok(Signal::CtrlC) => continue,  // Ctrl+C: cancel current line
            Ok(Signal::CtrlD) => break,     // Ctrl+D: exit
            Err(_) => break,
        };

        if input == "quit" || input == "exit" || input == "/quit" {
            break;
        }

        // Assemble context (with compressed history)
        let assembled = context::assemble(
            &config,
            &input,
            &conversation_history,
            false,
            mcp_summary.as_deref(),
        );

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
                            "[SYSTEM: The user has asked you to stop. \
                             Wrap up immediately and summarize what you've done so far.]"
                        ));
                    }
                }
            }

            // Warn the LLM when approaching the hard limit
            if round == max_rounds.saturating_sub(5) {
                messages.push(Message::user(
                    "[SYSTEM: You are approaching the tool call limit. \
                     Wrap up your current task and summarize what you've done.]"
                ));
            }

            // Observation masking
            if tool_result_log.len() > MASK_AFTER_RESULTS {
                mask_old_tool_results(&mut messages, &tool_result_log);
            }

            // Sanitize message roles before sending (strict chat template compat)
            context::sanitize_messages(&mut messages);

            let request = ChatRequest {
                messages: messages.clone(),
                tools: Some(tool_defs.clone()),
                tool_choice: None,
            };

            let response = match client
                .chat_stream(&request, |token| {
                    tui::print_token(token);
                })
                .await
            {
                Ok(r) => r,
                Err(e) => {
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

        // Keep conversation history bounded
        let max_history = config.context.history_turns * 6;
        while conversation_history.len() > max_history {
            conversation_history.remove(0);
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
