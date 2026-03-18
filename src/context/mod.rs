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

use crate::config::Config;
use crate::knowledge::graph::DependencyGraph;
use crate::knowledge::repo_map;
use crate::knowledge::ProjectIndex;
use crate::llm::Message;
use std::fs;

/// The assembled context ready to send to the LLM.
pub struct AssembledContext {
    pub messages: Vec<Message>,
    pub token_estimate: usize,
}

/// Build the system prompt in compressed structured format.
fn build_system_prompt() -> String {
    // Compressed format per design section 13.3 — ~40% shorter than prose
    String::from(
        "You are miniswe, a coding agent in a terminal.\n\
         \n\
         [RULES]\n\
         1.Read before write:use read_symbol/search before edits\n\
         2.Small edits:one change per edit call;tool validates syntax\n\
         3.Update state:call task_update after progress(memory between turns)\n\
         4.Verify:run tests/typecheck after edits\n\
         5.Step by step:follow scratchpad plan,one step at a time\n\
         6.If unsure:explore with search/read_symbol;repo map shows structure\n\
         7.File headers:every file must start with a brief comment describing its purpose\n\
         8.Keep files<200 lines;split into focused modules when growing\n\
         \n\
         [TOOLS]\n\
         read_symbol(name,follow_deps?)→function/class/type source\n\
         read_file(path,start_line?,end_line?)→file contents\n\
         search(query,scope?,max_results?)→grep matches\n\
         edit(path,old,new)→search-and-replace(best for surgical edits in large files)\n\
         write_file(path,content)→write complete file(preferred for files<200 lines)\n\
         shell(cmd,timeout?)→execute command\n\
         task_update(content)→rewrite scratchpad(must have ##Current Task,##Plan)\n\
         diagnostics(path?)→compiler/linter errors\n\
         web_search(query,max_results?)→DuckDuckGo snippets\n\
         web_fetch(url,selector?)→URL as markdown\n\
         docs_lookup(library,topic?)→local docs cache\n\
         \n\
         [WEB]check docs_lookup first→web_search snippets→web_fetch if needed\n\
         \n\
         [FORMAT]think→act with tools→task_update→summarize when done\n",
    )
}

/// Load and compress the project profile.
fn load_profile(config: &Config) -> Option<String> {
    let path = config.miniswe_path("profile.md");
    let content = fs::read_to_string(path).ok()?;
    Some(compress::compress_profile(&content))
}

/// Load the user guide from `.miniswe/guide.md`.
fn load_guide(config: &Config) -> Option<String> {
    let path = config.miniswe_path("guide.md");
    let content = fs::read_to_string(path).ok()?;
    // Skip if it's just the template
    if content.contains("<!-- Add project-specific instructions") && content.lines().count() <= 5 {
        return None;
    }
    Some(content)
}

/// Load the scratchpad from `.miniswe/scratchpad.md`.
fn load_scratchpad(config: &Config) -> Option<String> {
    let path = config.miniswe_path("scratchpad.md");
    fs::read_to_string(path).ok()
}

/// Load the plan from `.miniswe/plan.md`.
fn load_plan(config: &Config) -> Option<String> {
    let path = config.miniswe_path("plan.md");
    fs::read_to_string(path).ok()
}

/// Load the repo map slice, personalized for the current task.
fn load_repo_map(config: &Config, task_keywords: &[&str]) -> Option<String> {
    let miniswe_dir = config.miniswe_dir();
    let index = ProjectIndex::load(&miniswe_dir).ok()?;
    let graph = DependencyGraph::load(&miniswe_dir).ok()?;

    let map = repo_map::render(
        &index,
        &graph,
        config.context.repo_map_budget,
        task_keywords,
    );

    if map.is_empty() {
        None
    } else {
        Some(map)
    }
}

/// Load relevant lessons from `.miniswe/lessons.md` based on keywords.
fn load_relevant_lessons(config: &Config, keywords: &[&str]) -> Option<String> {
    let path = config.miniswe_path("lessons.md");
    let content = fs::read_to_string(path).ok()?;

    // Skip if it's just the template
    if content.contains("<!-- Accumulated tips") && content.lines().count() <= 5 {
        return None;
    }

    if keywords.is_empty() {
        return Some(content);
    }

    // Extract sections that match any keyword
    let mut relevant = String::new();
    let mut in_section = false;

    for line in content.lines() {
        if line.starts_with("## ") {
            let heading_lower = line.to_lowercase();
            in_section = keywords
                .iter()
                .any(|kw| kw.len() >= 3 && heading_lower.contains(&kw.to_lowercase()));
        }

        if in_section {
            relevant.push_str(line);
            relevant.push('\n');
        }
    }

    if relevant.is_empty() {
        None
    } else {
        Some(relevant)
    }
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

    // Pass 3: ensure no user→user or assistant→assistant sequences remain
    // (after tool messages, insert a synthetic assistant if needed before user)
    let mut i = 1;
    while i < messages.len() {
        let prev_role = &messages[i - 1].role;
        let curr_role = &messages[i].role;

        // user after user (shouldn't happen after pass 1, but safety)
        if curr_role == "user" && prev_role == "user" {
            messages.insert(i, Message::assistant("Understood."));
            i += 2;
            continue;
        }

        // user directly after tool (need an assistant acknowledgment in between)
        // Actually: tool messages should follow an assistant with tool_calls,
        // and after all tool results the next message should be from assistant.
        // But if a user message follows tool results, that's fine in practice
        // for most templates — the tool results are "completing" the assistant's
        // tool calls, and then it's the assistant's turn again.

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
                    let truncated = if first_line.len() > 80 {
                        format!("{}...", &first_line[..77])
                    } else {
                        first_line.to_string()
                    };
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
                    let truncated = if first.len() > 60 {
                        format!("{}...", &first[..57])
                    } else {
                        first.to_string()
                    };
                    summary_lines.push(truncated);
                }
            }
            "user" => {
                if let Some(content) = &msg.content {
                    let truncated = if content.len() > 60 {
                        format!("user: {}...", &content[..57])
                    } else {
                        format!("user: {content}")
                    };
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
        compressed.push(Message::assistant("Understood, continuing from where we left off."));
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
    let input_budget = budget.saturating_sub(output_budget);

    let mut messages = Vec::new();
    let mut used_tokens = 0;

    // 1. System prompt (compressed)
    let mut system_context = build_system_prompt();

    // 2. Project profile (compressed to structured format)
    if let Some(profile) = load_profile(config) {
        system_context.push_str("\n");
        system_context.push_str(&profile);
    }

    // 3. User guide
    if let Some(guide) = load_guide(config) {
        system_context.push_str("\n[GUIDE]\n");
        system_context.push_str(&guide);
    }

    // 4. Active plan
    if let Some(plan) = load_plan(config) {
        system_context.push_str("\n[PLAN]\n");
        system_context.push_str(&plan);
    }

    // 5. Relevant lessons (keyword-matched)
    let keywords: Vec<&str> = user_message
        .split_whitespace()
        .filter(|w| w.len() >= 3)
        .collect();
    if let Some(lessons) = load_relevant_lessons(config, &keywords) {
        system_context.push_str("\n[LESSONS]\n");
        system_context.push_str(&lessons);
    }

    // 6. Repo map (task-personalized, budget-controlled)
    if let Some(repo_map) = load_repo_map(config, &keywords) {
        system_context.push_str("\n[REPO MAP]\n");
        system_context.push_str(&repo_map);
    }

    // 7. MCP server summaries (one line each — lazy loading)
    if let Some(mcp) = mcp_summary {
        system_context.push_str("\n[MCP SERVERS]\n");
        system_context.push_str(mcp);
        system_context.push_str("\nUse mcp_use(server,tool,arguments) to call MCP tools.\n");
    }

    // 7. Scratchpad (folded into system message to avoid role alternation issues)
    if let Some(scratchpad) = load_scratchpad(config) {
        system_context.push_str("\n[SCRATCHPAD]\n");
        system_context.push_str(&scratchpad);
    }

    if plan_only {
        system_context.push_str(
            "\n[MODE:PLAN]\n\
             Read-only. Explore codebase, produce plan. NO edits/shell.\n\
             Write plan to .miniswe/plan.md via task_update.\n",
        );
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
