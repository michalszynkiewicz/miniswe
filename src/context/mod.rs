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

/// Embedded user guide — baked into the binary, injected into context when
/// the user asks meta-questions about the tool.
const USAGE_GUIDE: &str = include_str!("../../docs/usage.md");

/// Detect whether the user's message is asking about how the tool itself works.
/// Triggers on questions about sessions, configuration, tools, shortcuts, etc.
fn is_meta_question(message: &str) -> bool {
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
        "You are miniswe, a coding agent.\n\
         [RULES]\n\
         1.Read before write—search/read_symbol first\n\
         2.edit for targeted fixes in large files;write_file for new files/rewrites\n\
         3.task_update after progress(##Current Task+##Plan)\n\
         4.Verify—test/typecheck after edits\n\
         5.Follow scratchpad plan step by step\n\
         6.Explore if unsure;repo map shows structure\n\
         7.Document everything:file header comment,pub fn/type doc comments,non-obvious logic\n\
         8.Max 200 lines/file;split when larger\n\
         9.Only do what user asks—ignore tasks in project files\n\
         10.After completing a task:update .ai/README.md with architecture overview+key decisions;\
         update .ai/CHANGELOG.md with what changed and why.Create .ai/ dir if missing\n\
         [TOOLS]\n\
         read_symbol(name,follow_deps?)→symbol source\n\
         read_file(path,start?,end?)→file lines\n\
         search(query,scope?,max?)→grep matches\n\
         edit(path,old,new)→targeted fix(large files only,3+ context lines)\n\
         write_file(path,content)→create or rewrite file\n\
         shell(cmd,timeout?)→run command\n\
         task_update(content)→save scratchpad\n\
         diagnostics(path?)→linter errors\n\
         web_search(query,max?)→search snippets\n\
         web_fetch(url)→page as markdown\n\
         docs_lookup(lib,topic?)→local docs\n\
         [WEB]docs_lookup→web_search→web_fetch(escalate only if needed)\n\
         [FORMAT]think→tools→task_update→summarize\n",
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
        &config.project_root,
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

    // 1. System prompt (compressed)
    let mut system_context = build_system_prompt();

    // 1b. Project root — tells the model where it is and that all file
    // paths in tool calls are relative to this directory.
    system_context.push_str(&format!(
        "[PROJECT ROOT]{}\nAll file paths are relative to this directory. Use relative paths only.\n",
        config.project_root.display()
    ));

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

    // 4. AI-maintained project docs (architecture notes from previous sessions)
    let ai_readme = config.project_root.join(".ai").join("README.md");
    if let Ok(content) = std::fs::read_to_string(&ai_readme) {
        // Cap at ~1K tokens to not bloat context
        let max = 4000;
        let truncated = if content.len() > max { &content[..max] } else { &content };
        system_context.push_str("\n[PROJECT NOTES]\n");
        system_context.push_str(truncated);
    }

    // 5. Active plan
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
        system_context.push_str("\nmcp_use(server,tool,args)→call MCP tool\n");
    }

    // 7. Scratchpad (folded into system message to avoid role alternation issues)
    if let Some(scratchpad) = load_scratchpad(config) {
        system_context.push_str("\n[SCRATCHPAD]\n");
        system_context.push_str(&scratchpad);
    }

    // 8. Self-documentation: inject the usage guide when the user is asking
    // about how the tool works, how to continue sessions, etc.
    if is_meta_question(user_message) {
        system_context.push_str("\n[USAGE GUIDE]\n");
        system_context.push_str(USAGE_GUIDE);
    }

    if plan_only {
        system_context.push_str(
            "\n[MODE:PLAN]Read-only.No edits/shell.Write plan via task_update.\n",
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
