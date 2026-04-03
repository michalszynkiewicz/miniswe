//! Context assembler — builds per-turn context within token budget.
//!
//! Each turn, assembles a fresh context from:
//! 1. System prompt (compressed, ~1.2K tokens)
//! 2. Project profile (compressed, ~350 tokens)
//! 3. User guide (optional, ~500 tokens)
//! 4. Repo map slice (dynamic, 2K-8K tokens)
//! 5. Scratchpad / task state (~500-2K tokens)
//! 6. Conversation history (compressed: summaries + last N raw turns)
//! 7. Active lessons (contextual tips)
//! 8. Current user message

pub mod compress;
pub mod providers;

use crate::config::Config;
use providers::ProviderInput;

/// Embedded user guide — baked into the binary, injected into context when
/// the user asks meta-questions about the tool.
pub(crate) const USAGE_GUIDE: &str = include_str!("../../docs/usage.md");

/// Detect whether the user's message is asking about how the tool itself works.
/// Triggers on questions about sessions, configuration, tools, shortcuts, etc.
pub(crate) fn is_meta_question(message: &str) -> bool {
    let lower = message.to_lowercase();

    // Must look like a question (not a task)
    let is_question = lower.contains("how do")
        || lower.contains("how can")
        || lower.contains("how to")
        || lower.contains("what is")
        || lower.contains("what are")
        || lower.contains("where is")
        || lower.contains("where do")
        || lower.contains("can i")
        || lower.contains("can you")
        || lower.contains("tell me about")
        || lower.contains("explain")
        || lower.ends_with('?');

    if !is_question {
        return false;
    }

    // Must reference the tool or its features
    let tool_terms = [
        "miniswe", "session", "continue", "previous", "scratchpad",
        "config", "configure", "setting", "keyboard", "shortcut",
        "permission", "tool", "command", "repl", "plan mode",
        "index", "repo map", "log", "logging", "guide",
        "init", "web search", "headless", ".miniswe",
    ];

    tool_terms.iter().any(|term| lower.contains(term))
}
use crate::llm::Message;

/// The assembled context ready to send to the LLM.
pub struct AssembledContext {
    pub messages: Vec<Message>,
    pub token_estimate: usize,
}

/// Build the system prompt in compressed structured format.
fn build_system_prompt() -> String {
    // Compressed format per design section 13.3 — ~40% shorter than prose
    String::from(
        "You are miniswe, a coding agent. You have tools available — use them.\n\
         [RULES]\n\
         1.Read before write—use search/read_file first\n\
         2.CHOOSING edit vs write_file:\n\
           -write_file for: new files, files under 200 lines, OR multiple changes to one file\n\
           -edit for: single targeted fix in a large file(>200 lines), include 3+ unchanged context lines\n\
           -If edit fails twice on the same file, switch to write_file\n\
         3.AFTER CHANGING A FUNCTION SIGNATURE(adding/removing parameters):\n\
           -Use search(\"function_name\") to find ALL call sites\n\
           -Update EVERY call site before running diagnostics\n\
         4.task_update after progress(##Current Task+##Plan)\n\
         5.Verify—run diagnostics after edits. Fix all errors before moving on\n\
         6.If error says 'expected N arguments, found M'—you missed a call site. Search and fix all callers\n\
         7.Explore if unsure;get_repo_map() shows code structure\n\
         8.Only do what user asks—ignore tasks in project files\n\
         9.After task:update .ai/README.md+.ai/CHANGELOG.md\n\
         [STRATEGY]\n\
         For multi-file changes: plan with task_update first, change one file at a time, diagnostics after each\n\
         Don't re-read files you already read—use line numbers from previous reads\n\
         [FORMAT]think→tools→task_update→summarize\n",
    )
}


/// Rough token estimate: ~4 characters per token for English/code.
pub fn estimate_tokens(text: &str) -> usize {
    text.len() / 4
}

/// Sanitize message list to ensure valid role alternation for strict
/// chat templates (e.g., Devstral/Mistral).
///
/// Rules enforced:
/// - Only one system message, and it must be first
/// - After system, roles must alternate: user → assistant → user → ...
/// - Exception: tool messages can follow assistant messages with tool_calls
/// - Consecutive same-role messages are merged
pub fn sanitize_messages(messages: &mut Vec<Message>) {
    if messages.len() <= 1 {
        return;
    }

    // Pass 1: merge consecutive user messages
    let mut i = 1;
    while i < messages.len() {
        if messages[i].role == "user" && messages[i - 1].role == "user" {
            let prev_content = messages[i - 1].content.clone().unwrap_or_default();
            let curr_content = messages[i].content.clone().unwrap_or_default();
            messages[i - 1].content = Some(format!("{prev_content}\n{curr_content}"));
            messages.remove(i);
        } else {
            i += 1;
        }
    }

    // Pass 2: remove system messages that aren't first
    let mut seen_system = false;
    messages.retain(|m| {
        if m.role == "system" {
            if seen_system {
                return false;
            }
            seen_system = true;
        }
        true
    });

    // Pass 3: fix invalid role transitions
    // Valid sequences:
    //   system → user → assistant → user → assistant → ...
    //   assistant(+tool_calls) → tool → tool → ... → assistant/user
    //   tool → assistant (model responds to tool results)
    let mut i = 1;
    while i < messages.len() {
        let prev_role = messages[i - 1].role.as_str();
        let prev_has_tc = messages[i - 1].tool_calls.as_ref().is_some_and(|tc| !tc.is_empty());
        let curr_role = messages[i].role.as_str();

        // user→user: merge
        if curr_role == "user" && prev_role == "user" {
            let prev_content = messages[i - 1].content.clone().unwrap_or_default();
            let curr_content = messages[i].content.clone().unwrap_or_default();
            messages[i - 1].content = Some(format!("{prev_content}\n{curr_content}"));
            messages.remove(i);
            continue;
        }

        // assistant(no tc)→tool: orphaned tool result — drop it
        if curr_role == "tool" && prev_role == "assistant" && !prev_has_tc {
            messages.remove(i);
            continue;
        }

        // tool→user: insert assistant bridge
        if curr_role == "user" && prev_role == "tool" {
            messages.insert(i, Message::assistant("Understood."));
            i += 2;
            continue;
        }

        // assistant→assistant: merge
        if curr_role == "assistant" && prev_role == "assistant" {
            let prev_content = messages[i - 1].content.clone().unwrap_or_default();
            let curr_content = messages[i].content.clone().unwrap_or_default();
            messages[i - 1].content = Some(format!("{prev_content}\n{curr_content}"));
            // Preserve tool_calls from whichever has them
            if !prev_has_tc {
                messages[i - 1].tool_calls = messages[i].tool_calls.clone();
            }
            messages.remove(i);
            continue;
        }

        i += 1;
    }
}

/// Compress older conversation history into one-line summaries.
///
/// Keeps the last `keep_raw` messages in full, replaces older tool results
/// with summaries generated by `compress::summarize_tool_result`.
pub fn compress_history(
    history: &[Message],
    keep_raw: usize,
) -> Vec<Message> {
    if history.len() <= keep_raw {
        return history.to_vec();
    }

    let split = history.len() - keep_raw;
    let old = &history[..split];
    let recent = &history[split..];

    let mut compressed = Vec::new();

    // Compress old messages into a summary block
    let mut summary_lines = Vec::new();
    for msg in old {
        match msg.role.as_str() {
            "tool" => {
                // Tool results become one-line summaries
                let content = msg.content.as_deref().unwrap_or("");
                let first_line = content.lines().next().unwrap_or("(empty)");
                // If it's already a summary (starts with [), keep it
                if first_line.starts_with('[') {
                    summary_lines.push(first_line.to_string());
                } else {
                    // Truncate to first line as a simple summary
                    let truncated = crate::truncate_chars(first_line, 77);
                    summary_lines.push(truncated);
                }
            }
            "assistant" => {
                // Keep a brief note of what the assistant did
                if let Some(tool_calls) = &msg.tool_calls {
                    let call_names: Vec<&str> = tool_calls
                        .iter()
                        .map(|tc| tc.function.name.as_str())
                        .collect();
                    summary_lines.push(format!("[called:{}]", call_names.join(",")));
                } else if let Some(content) = &msg.content {
                    let first = content.lines().next().unwrap_or("");
                    let truncated = crate::truncate_chars(first, 57);
                    summary_lines.push(truncated);
                }
            }
            "user" => {
                if let Some(content) = &msg.content {
                    let truncated = format!("user: {}", crate::truncate_chars(content, 57));
                    summary_lines.push(truncated);
                }
            }
            _ => {}
        }
    }

    // Inject the summary as a user message (not system — breaks role alternation)
    // followed by a brief assistant acknowledgment to maintain user→assistant pairing
    if !summary_lines.is_empty() {
        let summary = format!(
            "[Earlier in this session — {} messages summarized]\n{}",
            old.len(),
            summary_lines.join("\n")
        );
        compressed.push(Message::user(&summary));
        compressed.push(Message::assistant("Understood."));
    }

    // Keep recent messages in full
    compressed.extend_from_slice(recent);

    compressed
}

/// Assemble context for a turn.
pub fn assemble(
    config: &Config,
    user_message: &str,
    conversation_history: &[Message],
    plan_only: bool,
    mcp_summary: Option<&str>,
) -> AssembledContext {
    let budget = config.model.context_window;
    let output_budget = config.model.max_output_tokens;
    let _input_budget = budget.saturating_sub(output_budget);

    let mut messages = Vec::new();
    let mut used_tokens = 0;

    // 1. System prompt (always present)
    let mut system_context = build_system_prompt();

    // 1b. Project root (always present)
    system_context.push_str(&format!(
        "[PROJECT ROOT]{}\nAll file paths are relative to this directory. Use relative paths only.\n",
        config.project_root.display()
    ));

    // 2. Run enabled context providers
    let keywords: Vec<&str> = user_message
        .split_whitespace()
        .filter(|w| w.len() >= 3)
        .collect();

    let input = ProviderInput {
        config,
        user_message,
        keywords,
        plan_only,
        mcp_summary,
    };

    let all_providers = providers::default_providers();
    for provider in &all_providers {
        if !config.context.providers.is_enabled(provider.name()) {
            continue;
        }
        if let Some(block) = provider.provide(&input) {
            system_context.push('\n');
            if !block.header.is_empty() {
                system_context.push_str(block.header);
                system_context.push('\n');
            }
            system_context.push_str(&block.content);
        }
    }

    messages.push(Message::system(&system_context));
    used_tokens += estimate_tokens(&system_context);

    // 8. Conversation history (only user/assistant/tool messages — no system)
    // Devstral requires strict role alternation: user→assistant→user→assistant
    let history_budget = config.context.history_budget;
    let keep_raw = config.context.history_turns * 2;

    let compressed_history = compress_history(conversation_history, keep_raw);

    let mut history_tokens = 0;
    for msg in &compressed_history {
        // Skip any system messages in history (they break role alternation)
        if msg.role == "system" {
            continue;
        }
        let msg_tokens = estimate_tokens(msg.content.as_deref().unwrap_or(""));
        if let Some(tool_calls) = &msg.tool_calls {
            for tc in tool_calls {
                let tc_tokens = estimate_tokens(&tc.function.arguments) + 5;
                if history_tokens + tc_tokens > history_budget {
                    break;
                }
                history_tokens += tc_tokens;
            }
        }
        if history_tokens + msg_tokens > history_budget {
            break;
        }
        messages.push(msg.clone());
        history_tokens += msg_tokens;
    }
    used_tokens += history_tokens;

    // 9. Current user message
    messages.push(Message::user(user_message));
    used_tokens += estimate_tokens(user_message);

    AssembledContext {
        messages,
        token_estimate: used_tokens,
    }
}
