//! Context assembler — builds per-turn context within token budget.
//!
//! Each turn, assembles a fresh context from:
//! 1. System prompt (~2K tokens)
//! 2. Project profile (~500-800 tokens)
//! 3. User guide (optional, ~500 tokens)
//! 4. Repo map slice (dynamic, 2K-8K tokens)
//! 5. Scratchpad / task state (~500-2K tokens)
//! 6. Retrieved snippets (budget-controlled)
//! 7. Conversation history (summary + last N raw turns)
//! 8. Active lessons (contextual tips)
//! 9. Current user message

use crate::config::Config;
use crate::llm::Message;
use std::fs;

/// The assembled context ready to send to the LLM.
pub struct AssembledContext {
    pub messages: Vec<Message>,
    pub token_estimate: usize,
}

/// Build the system prompt for minime.
fn build_system_prompt(_config: &Config) -> String {
    let mut prompt = String::new();

    prompt.push_str(
        "You are minime, a coding agent operating in a terminal. You work on the \
         project described in the Profile below.\n\n\
         ## How You Work\n\
         You operate in a loop: read context → reason → act → update state.\n\
         Each turn, you receive a fresh context with the project profile, a \
         relevant slice of the codebase map, your task scratchpad, and any \
         code snippets retrieved for the current step.\n\n\
         ## Rules\n\
         1. Read before writing. Use read_symbol or search to understand code \
         before making edits. Never guess at APIs or function signatures.\n\
         2. Small, focused edits. One logical change per edit call. The edit \
         tool validates syntax — if it rejects your edit, fix the syntax.\n\
         3. Update state. After meaningful progress, call task_update to save \
         what you learned and what's next. This is your memory between turns.\n\
         4. Verify changes. After edits, run tests or type-check. Don't assume \
         an edit worked — confirm it.\n\
         5. Work step by step. Follow the plan in your scratchpad. Complete one \
         step fully before moving to the next.\n\
         6. If uncertain, explore. Use search and read_symbol to understand the \
         codebase. The repo map shows you the structure — use it to find \
         the right files.\n\n",
    );

    prompt.push_str(
        "## Tools\n\
         - read_symbol(name, follow_deps?) → source code of a function/class/type\n\
         - read_file(path, start_line?, end_line?) → file contents or range\n\
         - search(query, scope?, max_results?) → grep matches with context\n\
         - edit(path, old, new) → validated search-and-replace in file\n\
         - shell(cmd, timeout?) → execute command, return output\n\
         - task_update(content) → rewrite your task scratchpad\n\
         - diagnostics(path?) → LSP errors/warnings\n\
         - web_search(query, max_results?) → DuckDuckGo snippets\n\
         - web_fetch(url, selector?) → fetch URL as clean markdown\n\
         - docs_lookup(library, topic?) → search local docs cache\n\n",
    );

    prompt.push_str(
        "## Response Format\n\
         Think through your approach, then act using tools. After each \
         meaningful step, call task_update with your updated state.\n\
         When the task is complete, summarize what you changed.\n",
    );

    prompt
}

/// Load the project profile from `.minime/profile.md`.
fn load_profile(config: &Config) -> Option<String> {
    let path = config.minime_path("profile.md");
    fs::read_to_string(path).ok()
}

/// Load the user guide from `.minime/guide.md`.
fn load_guide(config: &Config) -> Option<String> {
    let path = config.minime_path("guide.md");
    fs::read_to_string(path).ok()
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

/// Load relevant lessons from `.minime/lessons.md` based on keywords.
fn load_relevant_lessons(config: &Config, keywords: &[&str]) -> Option<String> {
    let path = config.minime_path("lessons.md");
    let content = fs::read_to_string(path).ok()?;

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
                .any(|kw| heading_lower.contains(&kw.to_lowercase()));
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
fn estimate_tokens(text: &str) -> usize {
    text.len() / 4
}

/// Assemble context for a turn.
pub fn assemble(
    config: &Config,
    user_message: &str,
    conversation_history: &[Message],
    plan_only: bool,
) -> AssembledContext {
    let budget = config.model.context_window;
    let mut messages = Vec::new();
    let mut used_tokens = 0;

    // 1. System prompt
    let system_prompt = build_system_prompt(config);
    let _system_tokens = estimate_tokens(&system_prompt);

    // 2. Build the full system context
    let mut system_context = system_prompt;

    // Add project profile
    if let Some(profile) = load_profile(config) {
        system_context.push_str("\n---\n## Project Profile\n");
        system_context.push_str(&profile);
    }

    // Add user guide
    if let Some(guide) = load_guide(config) {
        system_context.push_str("\n---\n## Project Guide\n");
        system_context.push_str(&guide);
    }

    // Add plan if it exists
    if let Some(plan) = load_plan(config) {
        system_context.push_str("\n---\n## Active Plan\n");
        system_context.push_str(&plan);
    }

    // Add relevant lessons
    let keywords: Vec<&str> = user_message.split_whitespace().collect();
    if let Some(lessons) = load_relevant_lessons(config, &keywords) {
        system_context.push_str("\n---\n## Relevant Lessons\n");
        system_context.push_str(&lessons);
    }

    if plan_only {
        system_context.push_str(
            "\n---\n## Mode: PLAN ONLY\n\
             You are in plan mode. Explore the codebase and produce a plan. \
             Do NOT make any edits. Write your plan to .minime/plan.md using task_update.\n",
        );
    }

    messages.push(Message::system(&system_context));
    used_tokens += estimate_tokens(&system_context);

    // 3. Conversation history (last N turns within budget)
    let history_budget = config.context.history_budget;
    let max_turns = config.context.history_turns;
    let mut history_tokens = 0;

    let recent_turns: Vec<&Message> = conversation_history
        .iter()
        .rev()
        .take(max_turns * 2) // user + assistant pairs
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();

    for msg in &recent_turns {
        let msg_tokens = estimate_tokens(msg.content.as_deref().unwrap_or(""));
        if history_tokens + msg_tokens > history_budget {
            break;
        }
        messages.push((*msg).clone());
        history_tokens += msg_tokens;
    }
    used_tokens += history_tokens;

    // 4. Scratchpad (appended to the tail for recency bias)
    if let Some(scratchpad) = load_scratchpad(config) {
        let scratchpad_msg = format!("[SCRATCHPAD]\n{scratchpad}");
        let scratchpad_tokens = estimate_tokens(&scratchpad_msg);
        if used_tokens + scratchpad_tokens < budget / 2 {
            messages.push(Message::system(&scratchpad_msg));
            used_tokens += scratchpad_tokens;
        }
    }

    // 5. Current user message
    messages.push(Message::user(user_message));
    used_tokens += estimate_tokens(user_message);

    AssembledContext {
        messages,
        token_estimate: used_tokens,
    }
}
