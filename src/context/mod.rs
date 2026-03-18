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
        "You are minime, a coding agent in a terminal.\n\
         \n\
         [RULES]\n\
         1.Read before write:use read_symbol/search before edits\n\
         2.Small edits:one change per edit call;tool validates syntax\n\
         3.Update state:call task_update after progress(memory between turns)\n\
         4.Verify:run tests/typecheck after edits\n\
         5.Step by step:follow scratchpad plan,one step at a time\n\
         6.If unsure:explore with search/read_symbol;repo map shows structure\n\
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
    let path = config.minime_path("profile.md");
    let content = fs::read_to_string(path).ok()?;
    Some(compress::compress_profile(&content))
}

/// Load the user guide from `.minime/guide.md`.
fn load_guide(config: &Config) -> Option<String> {
    let path = config.minime_path("guide.md");
    let content = fs::read_to_string(path).ok()?;
    // Skip if it's just the template
    if content.contains("<!-- Add project-specific instructions") && content.lines().count() <= 5 {
        return None;
    }
    Some(content)
}

/// Load the scratchpad from `.minime/scratchpad.md`.
fn load_scratchpad(config: &Config) -> Option<String> {
    let path = config.minime_path("scratchpad.md");
    fs::read_to_string(path).ok()
}

/// Load the plan from `.minime/plan.md`.
fn load_plan(config: &Config) -> Option<String> {
    let path = config.minime_path("plan.md");
    fs::read_to_string(path).ok()
}

/// Load the repo map slice, personalized for the current task.
fn load_repo_map(config: &Config, task_keywords: &[&str]) -> Option<String> {
    let minime_dir = config.minime_dir();
    let index = ProjectIndex::load(&minime_dir).ok()?;
    let graph = DependencyGraph::load(&minime_dir).ok()?;

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

/// Load relevant lessons from `.minime/lessons.md` based on keywords.
fn load_relevant_lessons(config: &Config, keywords: &[&str]) -> Option<String> {
    let path = config.minime_path("lessons.md");
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

    if !summary_lines.is_empty() {
        let summary = format!(
            "[HISTORY SUMMARY — {} earlier messages]\n{}",
            old.len(),
            summary_lines.join("\n")
        );
        compressed.push(Message::system(&summary));
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

    if plan_only {
        system_context.push_str(
            "\n[MODE:PLAN]\n\
             Read-only. Explore codebase, produce plan. NO edits/shell.\n\
             Write plan to .minime/plan.md via task_update.\n",
        );
    }

    messages.push(Message::system(&system_context));
    used_tokens += estimate_tokens(&system_context);

    // 7. Conversation history (compressed: old turns summarized, recent kept raw)
    let history_budget = config.context.history_budget;
    let keep_raw = config.context.history_turns * 2; // user + assistant pairs

    let compressed_history = compress_history(conversation_history, keep_raw);

    let mut history_tokens = 0;
    for msg in &compressed_history {
        let msg_tokens = estimate_tokens(msg.content.as_deref().unwrap_or(""));
        // Also account for tool call content
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

    // 8. Scratchpad at the tail (exploiting recency bias)
    if let Some(scratchpad) = load_scratchpad(config) {
        let scratchpad_msg = format!("[SCRATCHPAD]\n{scratchpad}");
        let scratchpad_tokens = estimate_tokens(&scratchpad_msg);
        if used_tokens + scratchpad_tokens < input_budget {
            messages.push(Message::system(&scratchpad_msg));
            used_tokens += scratchpad_tokens;
        }
    }

    // 9. Current user message
    messages.push(Message::user(user_message));
    used_tokens += estimate_tokens(user_message);

    AssembledContext {
        messages,
        token_estimate: used_tokens,
    }
}
