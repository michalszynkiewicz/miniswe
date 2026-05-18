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
pub mod compressor;
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
        "miniswe",
        "session",
        "continue",
        "previous",
        "scratchpad",
        "config",
        "configure",
        "setting",
        "keyboard",
        "shortcut",
        "permission",
        "tool",
        "command",
        "repl",
        "plan mode",
        "index",
        "repo map",
        "log",
        "logging",
        "guide",
        "init",
        "web search",
        "headless",
        ".miniswe",
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
///
/// The edit-tool guidance varies by `edit_mode`: Smart mode points the model
/// at `edit_file`; Fast mode points at the primitive surface
/// (`replace_range` / `insert_at` / `revert` / `show_rev`). Telling the model
/// to use a tool that isn't in its tool list wastes rounds, so we branch here.
///
/// `refactor_available` / `edit_file_available` toggle which edit-tool
/// guidance is emitted (must match the actual visible tool list, or the
/// model wastes rounds calling a tool that isn't there). Production now
/// sends `(true, false)` for every model; the other cells are kept for
/// completeness and tests.
///
/// `plan_set` selects the preamble: a pre-plan "explore → plan" workflow
/// vs a post-plan "you are EDITING now" routing-led preamble. This split
/// is the key refactor-adoption lever — see the `preamble` binding below.
fn build_system_prompt(
    edit_mode: crate::config::EditMode,
    refactor_available: bool,
    edit_file_available: bool,
    plan_set: bool,
) -> String {
    let refactor_blurb = "\
refactor adds/drops a parameter (action='add_param'/'drop_param') or renames a symbol (action='rename') and updates every callsite in ONE atomic call. Give the target NAME — the tool resolves the location via LSP. add_param example: {\"action\":\"add_param\",\"path\":\"src/lib.rs\",\"name\":\"assemble\",\"new_param\":\"x: u32\",\"position\":\"after:b\",\"callsite_fill_in\":\"0\"}. rename example: {\"action\":\"rename\",\"path\":\"src/lib.rs\",\"line\":42,\"name\":\"assemble\",\"new_name\":\"build_context\"}.\n";
    let cs_smart = match (refactor_available, edit_file_available) {
        (true, true) => format!(
            "{refactor_blurb}\
Reach for refactor BEFORE doing per-callsite edit_file calls when a function signature changes or a name changes — it handles the fan-out in one round.\n"
        ),
        (true, false) => format!(
            "{refactor_blurb}\
Reach for refactor for any signature change or rename — it handles the fan-out across callsites in one round.\n"
        ),
        (false, true) => "\
For signature changes (adding/dropping a parameter) or renames, edit the function definition with edit_file FIRST, then update each callsite — definitions before callers, never the other way around.\n".to_string(),
        (false, false) => "\
For signature changes (adding/dropping a parameter) or renames, edit the function definition with replace_range FIRST, then update each callsite — definitions before callers, never the other way around.\n".to_string(),
    };
    let cs_fast = match (refactor_available, edit_file_available) {
        (true, true) => format!(
            "{refactor_blurb}\
Reach for refactor BEFORE doing per-callsite edit_file or replace_range edits when a function signature changes or a name changes — it handles the fan-out in one round.\n"
        ),
        (true, false) => format!(
            "{refactor_blurb}\
Reach for refactor BEFORE doing per-callsite replace_range edits when a function signature changes or a name changes — it handles the fan-out in one round.\n"
        ),
        (false, true) => "\
For signature changes (adding/dropping a parameter) or renames, edit the function definition with replace_range or edit_file FIRST, then update each callsite — definitions before callers, never the other way around.\n".to_string(),
        (false, false) => "\
For signature changes (adding/dropping a parameter) or renames, edit the function definition with replace_range FIRST, then update each callsite — definitions before callers, never the other way around.\n".to_string(),
    };
    let edit_file_smart_line = if edit_file_available {
        "edit_file applies a semantic patch to one file: {{\"path\":\"src/lib.rs\",\"task\":\"rename foo to bar throughout the file\"}}\n"
    } else {
        ""
    };
    let edit_file_fast_line = if edit_file_available {
        "edit_file applies a semantic patch to one file using a focused inner LLM (best for non-trivial body edits like wrapping a block in if-let, restructuring brace nesting, or any change where line-precise replace_range is fiddly): {{\"path\":\"src/lib.rs\",\"task\":\"wrap the assemble body in an if-let so override_text replaces system_context when system_prompt_override is Some\"}}\nPrefer edit_file for structural rewrites; use replace_range / insert_at for surgical line-precise edits.\n"
    } else {
        ""
    };
    let smart_tail_line = if edit_file_available {
        "For any partial file edit (single line or multi-line) that isn't a signature change or rename, use edit_file with a clear task description."
    } else {
        "For partial file edits use replace_range / insert_at; refactor handles signature changes and renames; write_file is for whole-file overwrites."
    };
    let edit_contract = match edit_mode {
        crate::config::EditMode::Smart => {
            format!(
                "{cs_smart}\
{edit_file_smart_line}\
write_file with content replaces the whole file: {{\"path\":\"notes/todo.txt\",\"content\":\"first line\\nsecond line\\n\"}}\n\
write_file without content creates a new empty file: {{\"path\":\"tmp/placeholder.txt\"}}\n\
file shell: {{\"action\":\"shell\",\"command\":\"ls\",\"timeout\":60}}\n\
{smart_tail_line}"
            )
        }
        crate::config::EditMode::Fast => {
            format!(
                "{cs_fast}\
{edit_file_fast_line}\
replace_range replaces lines [start..=end] (1-based, inclusive) with content: {{\"path\":\"src/lib.rs\",\"start\":10,\"end\":15,\"content\":\"...\"}}\n\
insert_at inserts content after a line (0=top, last line = append): {{\"path\":\"src/lib.rs\",\"after_line\":0,\"content\":\"use std::fs;\\n\"}}\n\
write_file with content replaces the whole file: {{\"path\":\"notes/todo.txt\",\"content\":\"first line\\nsecond line\\n\"}}\n\
write_file without content creates a new empty file: {{\"path\":\"tmp/placeholder.txt\"}}\n\
file shell: {{\"action\":\"shell\",\"command\":\"ls\",\"timeout\":60}}\n\
Every edit returns a revision table; if an edit regresses, call revert {{\"path\":...,\"rev\":N}} to roll back — do not layer more edits on top."
            )
        }
    };
    let mut unlock_tools: Vec<&str> = Vec::new();
    if refactor_available {
        unlock_tools.push("refactor (preferred for adding/dropping a parameter or renaming a symbol — updates definition + all callsites atomically)");
    }
    if edit_file_available {
        unlock_tools.push("edit_file");
    }
    unlock_tools.extend(["replace_range", "insert_at", "write_file"]);
    // Which tool the "signature change / rename" intent routes to. With
    // refactor available it's refactor; the fallbacks exist only for
    // configs that hide it.
    let sig_route = if refactor_available {
        "refactor"
    } else if edit_file_available {
        "edit_file"
    } else {
        "replace_range (definition first, then each callsite)"
    };

    // Phase-aware preamble. The pre-plan vs post-plan split is the key
    // finding from the refactor-adoption investigation: a *static* prompt
    // whose pre-plan scaffolding ("explore → plan → you'll GET edit tools
    // later") persists into the editing phase suppresses edit-tool
    // selection — replay probes showed Devstral go 0/10 → 10/10 on
    // refactor adoption purely by swapping to an editing-phase preamble
    // once the plan exists. assemble() recomputes this every turn, so the
    // prompt flips automatically at the plan(action='set') boundary (same
    // boundary visible_tool_defs uses to unlock the write tools).
    let preamble = if plan_set {
        format!(
            "You are miniswe, a coding agent. A plan is set — you are in the EDITING phase. Make the changes now.\n\
             Tool routing — pick by intent:\n\
             - Add/remove a parameter, or rename a function/method/type/variable across callsites -> {sig_route}\n\
             - Insert new lines/code -> insert_at\n\
             - Replace or delete existing lines -> replace_range\n\
             - Whole-file rewrite or new file -> write_file\n\
             For a signature change or rename, {sig_route} updates the definition AND every callsite in ONE call — do NOT hand-edit callsites yourself, that is exactly what it exists to avoid.\n\
             Mark a finished step with plan(action='check', step=N); use plan(action='refine') if the work split changed."
        )
    } else {
        let workflow_unlock_preview = format!(
            "After plan is set you'll get these edit tools: {}.",
            unlock_tools.join(", ")
        );
        format!(
            "You are miniswe, a coding agent. Use your tools to complete the task.\n\
             WORKFLOW: explore briefly to find the relevant files (use file(action='search') or code(action='repo_map') for an overview), then call plan(action='set') with your step-by-step approach, then edit. {workflow_unlock_preview} Use plan(action='refine') to adjust the plan as you learn more; reserve plan(action='set') for the initial plan and rare full restarts."
        )
    };
    format!(
        "{preamble}\n\
         Emit ONE tool call per response. Wait for its result before issuing the next one — chaining multiple tool calls in a single response can confuse the parser and you will lose work.\n\
         Tool contract: grouped tools require action plus action-specific params.\n\
         file read: {{\"action\":\"read\",\"path\":\"README.md\"}}\n\
         {edit_contract}\n\
         If a tool says a parameter is missing, retry with the exact required parameter names.\n\
         Background servers: spawn with `& echo $! > .pid` and kill via that pid before respawning — don't pkill/grep ps.\n\
         Bound port with no matching process under you: switch ports, don't escalate kills.\n"
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
        let prev_has_tc = messages[i - 1]
            .tool_calls
            .as_ref()
            .is_some_and(|tc| !tc.is_empty());
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

        // assistant(+tool_calls)→user: interrupted/abandoned tool phase.
        // Drop the dangling tool_calls message so the next user turn stays valid
        // for strict chat templates (e.g. llama.cpp / Mistral-style alternation).
        if curr_role == "user" && prev_role == "assistant" && prev_has_tc {
            messages.remove(i - 1);
            i = i.saturating_sub(1);
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
pub fn compress_history(history: &[Message], keep_raw: usize) -> Vec<Message> {
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
    // All models now get the same surface: refactor available, edit_file
    // hidden. The old Devstral special-case (hide refactor, keep edit_file)
    // was protecting against `position`-arg mangling that the rename to
    // `refactor` already fixed, and which suppressed refactor adoption.
    // The real lever for adoption is the phase-aware prompt below, not the
    // gate — see build_system_prompt's plan_set note.
    let mut system_context = build_system_prompt(
        config.tools.edit_mode,
        true,  // refactor available for all models
        false, // edit_file hidden for all models
        crate::tools::plan::plan_exists(config),
    );

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

    // Inject structured plan if it exists
    if let Some(plan_content) = crate::tools::plan::load_plan(config) {
        system_context.push_str("\n[PLAN]\n");
        system_context.push_str(&plan_content);
        system_context.push('\n');
    }

    messages.push(Message::system(&system_context));
    used_tokens += estimate_tokens(&system_context);

    // 8. Conversation history — add directly, unified compressor handles compression
    // (compress_history was previously called here but is now redundant since
    // maybe_compress() in the agent loop handles all compression)
    let mut history_tokens = 0;
    for msg in conversation_history {
        if msg.role == "system" {
            continue;
        }
        let msg_tokens = estimate_tokens(msg.content.as_deref().unwrap_or(""));
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

#[cfg(test)]
mod prompt_phase_tests {
    use super::build_system_prompt;
    use crate::config::EditMode;

    // Production sends (refactor=true, edit_file=false) for every model.
    fn pre() -> String {
        build_system_prompt(EditMode::Fast, true, false, false)
    }
    fn post() -> String {
        build_system_prompt(EditMode::Fast, true, false, true)
    }

    #[test]
    fn pre_plan_is_explore_then_plan() {
        let p = pre();
        assert!(
            p.contains("WORKFLOW: explore"),
            "pre-plan prompt must lead with the explore→plan workflow"
        );
        assert!(
            !p.contains("EDITING phase"),
            "pre-plan prompt must NOT claim the editing phase yet"
        );
    }

    #[test]
    fn post_plan_is_editing_phase_with_routing() {
        let p = post();
        assert!(
            p.contains("EDITING phase"),
            "post-plan prompt must announce the editing phase"
        );
        assert!(
            p.contains("Tool routing — pick by intent:"),
            "post-plan prompt must include the routing table"
        );
        assert!(
            !p.contains("WORKFLOW: explore"),
            "post-plan prompt must drop the stale pre-plan scaffolding"
        );
    }

    #[test]
    fn post_plan_routes_signature_changes_to_refactor() {
        // With refactor available, the sig-change/rename intent must point
        // at refactor (the whole point of the phase-aware change).
        let p = post();
        assert!(
            p.contains("rename a function/method/type/variable across callsites -> refactor"),
            "signature/rename intent must route to refactor when available"
        );
    }

    #[test]
    fn switching_plan_set_changes_the_prompt() {
        // The exact invariant the run-loop relies on: same args, only
        // plan_set differs → materially different prompt.
        assert_ne!(pre(), post(), "plan_set must change the system prompt");
    }

    #[test]
    fn pre_and_post_share_the_operational_tail() {
        // Phase only swaps the preamble; the tool-contract / ops rules
        // must be present in both so behavior other than routing is stable.
        for p in [pre(), post()] {
            assert!(p.contains("Emit ONE tool call per response."));
            assert!(p.contains("Tool contract: grouped tools require action"));
            assert!(p.contains("Background servers: spawn with"));
        }
    }
}
