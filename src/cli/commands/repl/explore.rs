//! Per-turn explore/coding intent router and the read-only EXPLORE
//! tool surface.

use super::*;

/// Per-turn intent router (REPL only). One isolated, tool-less LLM
/// round-trip — NOT added to conversation history — classifying the
/// user's message as code-editing vs read-only investigation.
///
/// Biased toward EXPLORE for *questions* (the prompt routes
/// explain/summarize/why/what/where/how to EXPLORE) so a plain question
/// doesn't tip into execution; CODING is reserved for a clear instruction
/// to change code. The one hard safety rule kept from the original: a
/// parse failure / empty / LLM error → `false` (CODING), so on genuine
/// uncertainty we never silently swallow an edit request into read-only
/// mode. Normal-path bias is the prompt's job; the error-path default is
/// CODING.
pub(super) async fn classify_is_explore(
    llm_worker: &LlmWorkerHandle,
    user_message: &str,
    cancelled: &Arc<AtomicBool>,
) -> bool {
    // Artifact-mutation framing (validated 2026-06-30 vs gemma, isolated
    // battery in scripts/classifier-prompt-probe family, 27 cases × 3 reps):
    // key on whether the user wants an artifact created/altered/dropped vs a
    // question answered. This generalizes to descriptive phrasings the earlier
    // verb-based prompt ("tells you to change/add/fix code") misrouted to
    // EXPLORE — "create an app …", "I want a CLI that …", "… sort it out" were
    // all 3/3 EXPLORE under the old prompt and are 0/17 dangerous under this
    // one, with 0/10 EXPLORE-side regressions. Don't trim without re-running.
    let sys = "Reply one word: CODING or EXPLORE. \
        Determine if this is a pure exploration task that leads to answering a \
        question (reply EXPLORE), or a task that creates, alters, or drops an \
        artifact — file, app, project, feature, etc. (reply CODING). \
        Default EXPLORE.";
    let request = ChatRequest {
        messages: vec![Message::system(sys), Message::user(user_message)],
        tools: None,
        tool_choice: None,
        max_tokens_override: Some(8),
        chat_template_kwargs: Some(serde_json::json!({"enable_thinking": false})),
        temperature_override: None,
        cache_prompt: None,
    };
    let mut events = llm_worker.submit(ModelRole::Default, request, cancelled.clone());
    let mut out = String::new();
    while let Some(ev) = events.recv().await {
        match ev {
            LlmWorkerEvent::Completed(Ok(r)) => {
                out = r
                    .choices
                    .first()
                    .and_then(|c| c.message.content.clone())
                    .unwrap_or_default();
                break;
            }
            LlmWorkerEvent::Completed(Err(_)) => break, // fail-safe → CODING
            _ => {}
        }
    }
    is_explore_reply(&out)
}

/// Fail-safe classifier parse: EXPLORE only on a clean leading
/// EXPLORE; everything else (incl. empty, prose-wrapped, CODING) →
/// false = CODING. The asymmetric bias is the safety property.
pub(super) fn is_explore_reply(s: &str) -> bool {
    s.trim().to_ascii_uppercase().starts_with("EXPLORE")
}

/// Read-only tool subset for EXPLORE turns: drop every writer/mutator
/// (so the model is never *offered* an edit tool) and the plan tool
/// (no planning in Q&A). Keeps file:read/search, code:* (LSP/repo
/// map), web, show_rev/check.
pub(super) fn read_only_tool_defs(
    all: &[crate::llm::ToolDefinition],
) -> Vec<crate::llm::ToolDefinition> {
    all.iter()
        .filter(|t| {
            let n = t.function.name.as_str();
            !(is_file_write(n) || matches!(n, "revert" | "delete_file" | "plan" | "spawn_agents"))
        })
        .cloned()
        .collect()
}

/// Conservative read-only shell classifier for explore mode. Returns true ONLY
/// when every command in the (possibly compound) line is a known read-only
/// command with no output redirection, command substitution, or mutating flags.
/// Errs toward `false` (block) on anything unrecognized. NB: a heuristic — the
/// load-bearing rule is "unknown ⇒ block", plus hard-rejecting write constructs.
pub(super) fn shell_is_read_only(command: &str) -> bool {
    let cmd = command.trim();
    if cmd.is_empty() {
        return false;
    }
    // Allow the two common stderr redirects; any remaining `>` (or process
    // substitution) is a real write → block.
    let scrub = cmd
        .replace("2>/dev/null", "")
        .replace("2>&1", "")
        .replace(">/dev/null", "");
    const DANGER: &[&str] = &[
        ">",
        "$(",
        "`",
        "<(",
        ">(",
        "&>",
        "|&",
        " -exec",
        " -execdir",
        " -delete",
        " -ok",
        " -fprint",
        "xargs",
        "eval ",
        "source ",
        "sudo ",
        "chmod",
        "chown",
        "tee ",
    ];
    if DANGER.iter().any(|d| scrub.contains(d)) {
        return false;
    }
    if cmd.contains("sed") && (cmd.contains(" -i") || cmd.contains("--in-place")) {
        return false;
    }
    const READ_CMDS: &[&str] = &[
        "ls", "cat", "head", "tail", "grep", "egrep", "fgrep", "rg", "ag", "find", "fd", "wc",
        "stat", "file", "tree", "pwd", "echo", "printf", "sort", "uniq", "cut", "tr", "awk", "sed",
        "which", "type", "basename", "dirname", "realpath", "readlink", "du", "df", "env",
        "printenv", "date", "whoami", "hostname", "uname", "nl", "tac", "column", "jq", "yq",
        "xxd", "od", "strings", "diff", "cmp", "comm", "less", "more", "true", "test", "cd",
    ];
    const GIT_READ: &[&str] = &[
        "status",
        "log",
        "diff",
        "show",
        "branch",
        "ls-files",
        "ls-tree",
        "blame",
        "describe",
        "rev-parse",
        "cat-file",
        "grep",
        "shortlog",
        "reflog",
        "remote",
        "config",
        "tag",
        "whatchanged",
        "name-rev",
    ];
    let normalized = scrub
        .replace("&&", "\n")
        .replace("||", "\n")
        .replace([';', '|', '&'], "\n");
    for seg in normalized.lines() {
        let mut toks = seg.split_whitespace().peekable();
        // skip leading VAR=val env assignments
        while toks
            .peek()
            .is_some_and(|t| t.contains('=') && !t.starts_with('-'))
        {
            toks.next();
        }
        let Some(c0) = toks.next() else { continue };
        let base = c0.rsplit('/').next().unwrap_or(c0);
        if base == "git" {
            if !GIT_READ.contains(&toks.next().unwrap_or("")) {
                return false;
            }
        } else if !READ_CMDS.contains(&base) {
            return false;
        }
    }
    true
}

/// In read-only (explore) mode, decide whether a tool call must be blocked.
/// `Some(reason)` ⇒ mutating, block it; `None` ⇒ read-only, allow. This is the
/// load-bearing runtime guard: the tool-def filter and the prompt are advisory,
/// but shell can mutate and the model can emit tools that aren't in the list.
pub(super) fn explore_block_reason(
    name: &str,
    file_action: &str,
    args: &serde_json::Value,
) -> Option<String> {
    if is_file_write(name) || matches!(name, "revert" | "delete_file" | "spawn_agents") {
        return Some(format!("`{name}` can modify files"));
    }
    if name == "shell" {
        let cmd = args["command"].as_str().unwrap_or("");
        if args["action"].as_str() == Some("run") && !shell_is_read_only(cmd) {
            return Some(format!(
                "shell command is not read-only: `{}`",
                crate::truncate_chars(cmd, 80)
            ));
        }
    }
    if name == "file" {
        match file_action {
            "shell" => {
                let cmd = args["command"].as_str().unwrap_or("");
                if !shell_is_read_only(cmd) {
                    return Some(format!(
                        "shell command is not read-only: `{}`",
                        crate::truncate_chars(cmd, 80)
                    ));
                }
            }
            a if is_file_write(a) || matches!(a, "write_file" | "delete" | "delete_file") => {
                return Some(format!("file action `{a}` can modify files"));
            }
            _ => {}
        }
    }
    None
}
