//! Agent execution — the main LLM loop.
//!
//! Implements the core agent loop:
//! 1. Assemble context for the turn
//! 2. Call the LLM with tools
//! 3. Parse tool calls from the response
//! 4. Execute tools
//! 5. Feed results back and repeat

use anyhow::Result;

use crate::config::Config;
use crate::context;
use crate::llm::{LlmClient, ChatRequest, Message};
use crate::tools;
use crate::tui;

/// Maximum number of tool-call rounds before stopping.
const MAX_ROUNDS: usize = 25;

/// Run the agent for a single message.
pub async fn run(config: Config, message: &str, plan_only: bool) -> Result<()> {
    let client = LlmClient::new(config.model.clone());
    let tool_defs = tools::tool_definitions();

    tui::print_header(if plan_only {
        "Plan Mode (read-only)"
    } else {
        "minime"
    });

    let mut conversation_history: Vec<Message> = Vec::new();
    let mut round = 0;

    // Initial context assembly
    let assembled = context::assemble(&config, message, &conversation_history, plan_only);
    tui::print_status(&format!(
        "Context: ~{} tokens assembled",
        assembled.token_estimate
    ));

    let mut messages = assembled.messages;

    loop {
        round += 1;
        if round > MAX_ROUNDS {
            tui::print_error("Maximum tool rounds reached. Stopping.");
            break;
        }

        // Call LLM with streaming
        let request = ChatRequest {
            messages: messages.clone(),
            tools: Some(tool_defs.clone()),
            tool_choice: None,
        };

        tui::print_separator();

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

        // Add assistant message to history
        conversation_history.push(assistant_msg.clone());

        // Check for tool calls
        let tool_calls = match &assistant_msg.tool_calls {
            Some(tc) if !tc.is_empty() => tc.clone(),
            _ => {
                // No tool calls — the model is done or just responded with text
                if let Some(finish) = &choice.finish_reason {
                    if finish == "stop" {
                        break;
                    }
                }
                break;
            }
        };

        // Execute tool calls
        // Add assistant's tool call message to messages
        messages.push(assistant_msg.clone());

        for tc in &tool_calls {
            let args: serde_json::Value =
                serde_json::from_str(&tc.function.arguments).unwrap_or_default();

            let args_summary = summarize_args(&tc.function.name, &args);
            tui::print_tool_call(&tc.function.name, &args_summary);

            // Block edit tools in plan-only mode
            if plan_only && (tc.function.name == "edit" || tc.function.name == "shell") {
                let result_msg = Message::tool_result(
                    &tc.id,
                    "Blocked: plan mode is read-only. No edits or shell commands allowed.",
                );
                messages.push(result_msg.clone());
                conversation_history.push(result_msg);
                tui::print_tool_result(&tc.function.name, false, "blocked in plan mode");
                continue;
            }

            let result = match tools::execute_tool(&tc.function.name, &args, &config).await {
                Ok(r) => r,
                Err(e) => crate::tools::ToolResult::err(format!("Tool error: {e}")),
            };

            let first_line = result
                .content
                .lines()
                .next()
                .unwrap_or("(empty)");
            tui::print_tool_result(&tc.function.name, result.success, first_line);

            let result_msg = Message::tool_result(&tc.id, &result.content);
            messages.push(result_msg.clone());
            conversation_history.push(result_msg);
        }

        // Continue the loop — the LLM will process tool results
    }

    tui::print_separator();
    tui::print_complete("Done");

    Ok(())
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
        "read_symbol" => {
            args["name"].as_str().unwrap_or("?").to_string()
        }
        "search" => {
            let query = args["query"].as_str().unwrap_or("?");
            let scope = args["scope"].as_str().unwrap_or("project");
            format!("\"{query}\" in {scope}")
        }
        "edit" => {
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
        "web_search" => {
            args["query"].as_str().unwrap_or("?").to_string()
        }
        "web_fetch" => {
            args["url"].as_str().unwrap_or("?").to_string()
        }
        "docs_lookup" => {
            let lib = args["library"].as_str().unwrap_or("?");
            let topic = args["topic"].as_str().unwrap_or("");
            if topic.is_empty() {
                lib.to_string()
            } else {
                format!("{lib}/{topic}")
            }
        }
        _ => format!("{args}"),
    }
}
