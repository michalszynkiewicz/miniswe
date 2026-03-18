//! Interactive REPL mode.

use anyhow::Result;

use crate::config::Config;
use crate::context;
use crate::llm::{ChatRequest, LlmClient, Message};
use crate::tools;
use crate::tui;

/// Maximum tool rounds per user message.
const MAX_ROUNDS: usize = 25;

/// Run the interactive REPL.
pub async fn run(config: Config) -> Result<()> {
    let client = LlmClient::new(config.model.clone());
    let tool_defs = tools::tool_definitions();

    tui::print_header("Interactive Mode");
    tui::print_status(&format!(
        "Model: {} @ {}",
        config.model.model, config.model.endpoint
    ));
    tui::print_status("Type your message, or 'quit' to exit.");
    tui::print_separator();

    let mut conversation_history: Vec<Message> = Vec::new();

    loop {
        let input = match tui::read_input("you>") {
            Some(input) if !input.is_empty() => input,
            Some(_) => continue,
            None => break, // EOF
        };

        if input == "quit" || input == "exit" || input == "/quit" {
            break;
        }

        // Assemble context
        let assembled =
            context::assemble(&config, &input, &conversation_history, false);

        let mut messages = assembled.messages;
        let mut round = 0;

        loop {
            round += 1;
            if round > MAX_ROUNDS {
                tui::print_error("Maximum tool rounds reached.");
                break;
            }

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
                let args: serde_json::Value =
                    serde_json::from_str(&tc.function.arguments).unwrap_or_default();

                tui::print_tool_call(&tc.function.name, &format!("{args}"));

                let result =
                    match tools::execute_tool(&tc.function.name, &args, &config).await {
                        Ok(r) => r,
                        Err(e) => crate::tools::ToolResult::err(format!("Tool error: {e}")),
                    };

                let first_line = result.content.lines().next().unwrap_or("(empty)");
                tui::print_tool_result(&tc.function.name, result.success, first_line);

                let result_msg = Message::tool_result(&tc.id, &result.content);
                messages.push(result_msg.clone());
                conversation_history.push(result_msg);
            }
        }

        // Keep conversation history bounded
        while conversation_history.len() > config.context.history_turns * 4 {
            conversation_history.remove(0);
        }

        tui::print_separator();
    }

    tui::print_complete("Session ended");
    Ok(())
}
