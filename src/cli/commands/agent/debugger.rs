//! Reactive debugger sub-agent (experimental, opt-in via
//! `tools.reactive_debugger`). See GitHub #40.
//!
//! When the behavioral done-gate (`[validation]`) blocks completion several
//! times in a turn, the primary agent is "stuck-but-knows": its own
//! test/check is failing and it can't recover within the attempt. This spins
//! up a **read-only, fresh-context** sub-agent that investigates the specific
//! failure and produces a DIAGNOSIS REPORT — root cause + the concrete fix
//! (file:line + what to change). The primary agent then applies the fix.
//!
//! Design (deliberately narrow — an earlier edit-capable version flailed:
//! it thrashed with edits/reverts, fiddled with `plan`, and never produced a
//! usable report):
//! - **No edits.** Only read/search/inspect tools are offered, and the `file`
//!   tool is hard-gated to `read`/`search` at execution. The debugger cannot
//!   change code or run shell — it can only look.
//! - **No plan.** The plan tool is not offered; this is a one-shot diagnosis,
//!   not a multi-step build.
//! - **Always reports.** The bounded read loop is followed by a forced
//!   text-only turn, so the deliverable (the report) is guaranteed.
//!
//! The value is *attention reset / fresh eyes on the diagnosis*, not extra
//! capability (same weights) and not an extra editor.
//!
//! `tools.debugger_judge` extends this into a ROUTER: given the goal + the FULL
//! diff, the same fresh-context sub-agent first DECIDES `SCRAP` vs `CONTINUE`.
//! SCRAP → the loop reverts the tree to the clean baseline and restarts from
//! scratch (a stuck/off-path attempt is negative equity — cheaper to redo than
//! recover); CONTINUE → it emits the diagnosis + an anchored plan. The stuck
//! agent never makes the call; a fresh judge does and the loop executes it. The
//! full diff is essential — investigating only the failing location makes the
//! judge myopic (it sees a small local fix and votes CONTINUE on an off-path
//! attempt).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::Mutex;

use crate::config::{Config, ModelRole};
use crate::context;
use crate::llm::{ChatRequest, ChatResponse, Message, ToolDefinition};
use crate::lsp::LspClient;
use crate::mcp::McpRegistry;
use crate::runtime::{LlmWorkerEvent, LlmWorkerHandle, ToolWorkerPool};
use crate::tools;
use crate::tools::ToolResult;
use crate::tools::permissions::PermissionManager;

/// Number of done-gate blocks in a turn after which the debugger fires once.
/// Two lets the primary agent take its own swings first (the cheap path);
/// the debugger is the escalation when those swings keep missing.
pub const DEBUGGER_TRIGGER_BLOCKS: usize = 2;

/// Max debugger fires per turn when `debugger_multifire` is on. Bounds the
/// walk down the failure chain (compile → smoke → …) so a pathological
/// flapping failure can't spawn unbounded sub-agents.
pub const MAX_DEBUGGER_FIRES: usize = 3;

/// Read-only investigation budget. Diagnosis doesn't need many rounds; a
/// final forced report turn happens after this regardless.
const MAX_DEBUGGER_ROUNDS: usize = 8;

/// Tools the debugger may use — strictly read-only. `file` is additionally
/// gated to read/search at execution (see `run_readonly_tool`). Notably
/// excludes every write tool, `revert`, `plan`, and `spawn_agents`.
const READONLY_TOOLS: &[&str] = &["file", "code", "check", "show_rev"];

/// The debugger's OWN system prompt. We deliberately do NOT use
/// `context::assemble` here: that builds the full coding-agent prompt
/// (strict-ceremony plan-first workflow, edit instructions, scratchpad) —
/// which would tell the debugger to `plan(action='set')` and edit, neither of
/// which it can or should do. This minimal prompt keeps it a pure
/// read-only diagnostician.
const DEBUGGER_SYSTEM_PROMPT: &str = "\
You are a READ-ONLY debugging analyst with fresh eyes on a stuck task. You have ONLY \
read/search/inspect tools — you CANNOT edit files, run shell, set a plan, or use a scratchpad, \
and you must not try (you have no such tools). Do NOT plan; just investigate and report.\n\
Your sole deliverable is a DIAGNOSIS REPORT the main agent will use to fix the problem. \
Investigate efficiently: read the exact location the failure names and the relevant \
definitions/callsites. Find the REAL root cause (a value plumbed but not consumed, a missing \
guard, a broken default path, a compile error — read, do not assume). Then output a tight report:\n\
ROOT CAUSE: <the precise reason the check fails>\n\
FIX: <the specific change(s) — file:line and exactly what to change>\n\
Describe the fix CONCEPTUALLY — name the location and what must change and why. Do NOT write \
verbatim replacement code: you are read-only and cannot compile-check, so exact code you guess \
(types, method calls, signatures, argument shapes) may be wrong and mislead the main agent into \
a broken edit. Say e.g. \"thread the override into the assemble() call instead of None, matching \
that parameter's type\" — NOT a literal code snippet. Let the main agent write code that matches \
the real signatures. Do not edit anything.";

/// The debugger's JUDGE prompt (`tools.debugger_judge`): same read-only, fresh-
/// eyes stance, but it first DECIDES whether the attempt is salvageable. SCRAP
/// (loop reverts + restarts) vs CONTINUE (emit the recovery report + plan).
/// Validated to discriminate: SCRAP a poisoned/off-path state, CONTINUE a
/// healthy on-path one.
const DEBUGGER_JUDGE_PROMPT: &str = "\
You are a READ-ONLY analyst with fresh eyes on a STUCK coding task. You have ONLY \
read/search/inspect tools — you CANNOT edit files, run shell, set a plan, or use a scratchpad. \
Do NOT plan and do NOT try to edit.\n\
Investigate the failure and the changes made so far (the failing location, the relevant \
definitions/callsites, and whether the changes are even in the right place for the GOAL). Then \
DECIDE whether this attempt is worth continuing:\n\
- SCRAP: the changes are misdirected, damaged, or off-path — editing the wrong places for the \
GOAL, or broken in ways the GOAL did not require. Reverting everything to the clean original and \
starting fresh would be faster and more reliable. IGNORE effort already spent.\n\
- CONTINUE: the changes are on the path to the GOAL and nearly working; only a focused fix remains.\n\
Output your decision on the FIRST line, exactly one of:\n\
DECISION: SCRAP\n\
DECISION: CONTINUE\n\
If SCRAP: add one line — REASON: <the single most important reason> — and STOP.\n\
If CONTINUE: produce the recovery report the main agent will apply —\n\
ROOT CAUSE: <the precise reason the check fails>\n\
FIX: <where and what must change, described conceptually — NOT verbatim code you cannot compile-check>\n\
PLAN: <the concrete remaining steps to finish the GOAL, including the step that makes the feature \
actually work at runtime, not merely compile>";

/// Run the read-only debugger sub-agent against a blocking check failure.
/// Returns its diagnosis report (to inject into the primary agent), or `None`
/// if it produced nothing usable. It makes **no edits** — the report is the
/// entire deliverable.
#[allow(clippy::too_many_arguments)]
pub async fn run_debugger(
    failure_output: &str,
    goal: &str,
    config: &Config,
    llm_worker: &LlmWorkerHandle,
    _tool_pool: &ToolWorkerPool,
    parent_tool_defs: &[ToolDefinition],
    perms: &Arc<PermissionManager>,
    _mcp_registry: &Option<Arc<Mutex<McpRegistry>>>,
    lsp: &Option<Arc<LspClient>>,
    fast_revisions: &Option<Arc<tools::RevisionStore>>,
    fast_baseline_errors: usize,
    cancelled: &Arc<AtomicBool>,
) -> Option<String> {
    let tool_defs = readonly_tools(parent_tool_defs);
    let changed = changed_files(config);
    // Minimal, debugger-only context — NOT context::assemble (which would
    // inject the full plan-first/edit ceremony, see DEBUGGER_SYSTEM_PROMPT).
    let system_prompt = if config.tools.debugger_judge {
        DEBUGGER_JUDGE_PROMPT
    } else {
        DEBUGGER_SYSTEM_PROMPT
    };
    // The judge needs the whole diff to see off-path-ness; the plain
    // diagnostician stays failure-location-focused (empty diff).
    let diff = if config.tools.debugger_judge {
        changed_diff(config, 500)
    } else {
        String::new()
    };
    let mut messages = vec![
        Message::system(system_prompt),
        Message::user(&build_prompt(goal, failure_output, &changed, &diff)),
    ];
    let mut report = String::new();

    for _round in 0..MAX_DEBUGGER_ROUNDS {
        if cancelled.load(Ordering::Relaxed) {
            break;
        }
        context::sanitize_messages(&mut messages);
        let request = ChatRequest {
            messages: messages.clone(),
            tools: Some(tool_defs.clone()),
            tool_choice: None,
            max_tokens_override: None,
            chat_template_kwargs: Some(serde_json::json!({"enable_thinking": false})),
        };
        let Some(resp) = drain(llm_worker, request, cancelled).await else {
            break;
        };
        let Some(choice) = resp.choices.first() else {
            break;
        };
        let msg = &choice.message;
        if let Some(c) = &msg.content
            && !c.trim().is_empty()
        {
            report = c.clone();
        }
        if msg.is_meaningful() {
            messages.push(msg.clone());
        }
        match &msg.tool_calls {
            Some(tc) if !tc.is_empty() => {
                for call in tc {
                    let args: serde_json::Value = serde_json::from_str(&call.function.arguments)
                        .unwrap_or(serde_json::json!({}));
                    let result = run_readonly_tool(
                        &call.function.name,
                        &args,
                        config,
                        perms,
                        lsp,
                        fast_revisions,
                        fast_baseline_errors,
                    )
                    .await;
                    messages.push(Message::tool_result(&call.id, &result.content));
                }
            }
            // No tool calls → the model has concluded; its text is the report.
            _ => break,
        }
    }

    // Guarantee a report: if the investigation never emitted usable text,
    // ask once explicitly with no tools so the deliverable always exists.
    if report.trim().is_empty() && !cancelled.load(Ordering::Relaxed) {
        messages.push(Message::user(
            "You've finished investigating — no more reads. Write your final report now and \
             nothing else: ROOT CAUSE (precise reason the check fails) and FIX (where and what \
             must change, described conceptually — not verbatim code you cannot compile-check).",
        ));
        let request = ChatRequest {
            messages,
            tools: None,
            tool_choice: None,
            max_tokens_override: None,
            chat_template_kwargs: Some(serde_json::json!({"enable_thinking": false})),
        };
        if let Some(resp) = drain(llm_worker, request, cancelled).await
            && let Some(c) = resp.choices.first().and_then(|c| c.message.content.clone())
        {
            report = c;
        }
    }

    let report = report.trim().to_string();
    (!report.is_empty()).then_some(report)
}

/// Drain one streamed LLM call to a single response. `None` on error/abort.
async fn drain(
    llm_worker: &LlmWorkerHandle,
    request: ChatRequest,
    cancelled: &Arc<AtomicBool>,
) -> Option<ChatResponse> {
    let mut events = llm_worker.submit(ModelRole::Default, request, cancelled.clone());
    loop {
        match events.recv().await {
            Some(LlmWorkerEvent::Token(_)) => {}
            Some(LlmWorkerEvent::Completed(Ok(r))) => return Some(r),
            Some(LlmWorkerEvent::Completed(Err(_))) => return None,
            None => return None,
        }
    }
}

/// Execute one read-only tool. Hard-blocks any mutating `file` action so the
/// debugger physically cannot edit even if the model tries.
async fn run_readonly_tool(
    name: &str,
    args: &serde_json::Value,
    config: &Config,
    perms: &Arc<PermissionManager>,
    lsp: &Option<Arc<LspClient>>,
    fast_revisions: &Option<Arc<tools::RevisionStore>>,
    fast_baseline_errors: usize,
) -> ToolResult {
    if name == "file" {
        let action = args
            .get("action")
            .and_then(|a| a.as_str())
            .unwrap_or("read");
        if !matches!(action, "read" | "search") {
            return ToolResult::err(format!(
                "Debugger is READ-ONLY — file action '{action}' is blocked. \
                 Use action='read' or 'search'. Do not edit or run shell; produce your report instead."
            ));
        }
    }
    let res = match name {
        "check" | "show_rev" => match fast_revisions {
            Some(rev) => {
                tools::execute_fast_tool(
                    name,
                    args,
                    config,
                    perms.as_ref(),
                    lsp.as_deref(),
                    rev.as_ref(),
                    fast_baseline_errors,
                )
                .await
            }
            None => Ok(ToolResult::err("revision store unavailable".into())),
        },
        _ => tools::execute_tool(name, args, config, perms.as_ref(), lsp.as_deref()).await,
    };
    res.unwrap_or_else(|e| ToolResult::err(e.to_string()))
}

/// Filter the parent tool list down to the read-only diagnostic set.
fn readonly_tools(all: &[ToolDefinition]) -> Vec<ToolDefinition> {
    all.iter()
        .filter(|t| READONLY_TOOLS.contains(&t.function.name.as_str()))
        .cloned()
        .collect()
}

/// Best-effort list of files changed in the working tree (modified + staged +
/// untracked), so the debugger knows where to look without re-discovering the
/// whole task. Empty vec if git is unavailable — the debugger can still search.
fn changed_files(config: &Config) -> Vec<String> {
    let Ok(out) = std::process::Command::new("git")
        .arg("-C")
        .arg(&config.project_root)
        .args(["status", "--porcelain"])
        .output()
    else {
        return Vec::new();
    };
    if !out.status.success() {
        return Vec::new();
    }
    parse_porcelain(&String::from_utf8_lossy(&out.stdout))
}

/// Parse `git status --porcelain` output into a list of paths. Strips the
/// two-column status prefix and handles rename arrows (`old -> new` keeps new).
fn parse_porcelain(stdout: &str) -> Vec<String> {
    stdout
        .lines()
        .filter_map(|line| {
            if line.len() < 4 {
                return None;
            }
            let path = line[3..].trim();
            let path = path.rsplit(" -> ").next().unwrap_or(path);
            (!path.is_empty()).then(|| path.trim_matches('"').to_string())
        })
        .collect()
}

fn build_prompt(goal: &str, failure_output: &str, changed: &[String], diff: &str) -> String {
    let files = if changed.is_empty() {
        "(could not list changed files — use file(action='search') to locate the relevant code)"
            .to_string()
    } else {
        changed
            .iter()
            .map(|f| format!("  - {f}"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    // The JUDGE needs the WHOLE diff to assess on-path-ness — investigating only
    // the failing location makes it myopic (it sees a small local fix and votes
    // CONTINUE even when the overall attempt is off-path). Empty for the plain
    // diagnostician, which is deliberately failure-location-focused.
    let diff_section = if diff.trim().is_empty() {
        String::new()
    } else {
        format!(
            "\nThe FULL diff of the changes so far (vs the clean original) — review ALL of it to \
             judge whether the work is on-path for the GOAL, not just the failing spot:\n\
             ```diff\n{diff}\n```\n"
        )
    };

    format!(
        "GOAL (the task the agent is trying to accomplish):\n\
         {goal}\n\
         \n\
         A verification check is BLOCKING task completion. It failed with this output:\n\
         ----------------------------------------\n\
         {failure_output}\n\
         ----------------------------------------\n\
         \n\
         Files changed so far this session:\n\
         {files}\n\
         {diff_section}\
         \n\
         Investigate the cause, then act per your instructions."
    )
}

/// The full working-tree diff vs the session baseline commit, capped to keep the
/// sub-agent prompt bounded. Lets the JUDGE see the whole scope of changes (not
/// just the failing location) to assess on-path-ness. Empty on error.
fn changed_diff(config: &Config, max_lines: usize) -> String {
    let Ok(out) = std::process::Command::new("git")
        .arg("-C")
        .arg(&config.project_root)
        .args(["diff", "--no-color"])
        .output()
    else {
        return String::new();
    };
    if !out.status.success() {
        return String::new();
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() > max_lines {
        let mut s = lines[..max_lines].join("\n");
        s.push_str(&format!(
            "\n… (diff truncated at {max_lines} of {} lines)",
            lines.len()
        ));
        s
    } else {
        text.into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn td(name: &str) -> ToolDefinition {
        ToolDefinition {
            r#type: "function".into(),
            function: crate::llm::FunctionDefinition {
                name: name.into(),
                description: String::new(),
                parameters: serde_json::json!({}),
            },
        }
    }

    #[test]
    fn readonly_tools_keeps_only_inspection_tools() {
        let all = vec![
            td("file"),
            td("code"),
            td("check"),
            td("show_rev"),
            td("replace_range"),
            td("insert_at"),
            td("write_file"),
            td("refactor"),
            td("edit_file"),
            td("revert"),
            td("plan"),
            td("spawn_agents"),
        ];
        let kept: Vec<String> = readonly_tools(&all)
            .iter()
            .map(|t| t.function.name.clone())
            .collect();
        assert_eq!(kept, ["file", "code", "check", "show_rev"]);
    }

    #[tokio::test]
    async fn readonly_tool_blocks_file_writes_and_shell() {
        let config = Config::default();
        let perms = Arc::new(PermissionManager::headless(&config));
        for action in ["write", "replace", "shell", "insert"] {
            let r = run_readonly_tool(
                "file",
                &serde_json::json!({"action": action, "path": "x", "content": "y"}),
                &config,
                &perms,
                &None,
                &None,
                0,
            )
            .await;
            assert!(!r.success, "file action '{action}' should be blocked");
            assert!(r.content.contains("READ-ONLY"));
        }
    }

    #[test]
    fn parse_porcelain_extracts_paths() {
        let out = " M src/cli/mod.rs\n?? new_file.rs\nA  staged.rs\n";
        let paths = parse_porcelain(out);
        assert_eq!(paths, vec!["src/cli/mod.rs", "new_file.rs", "staged.rs"]);
    }

    #[test]
    fn parse_porcelain_handles_rename_arrow() {
        let out = "R  old/name.rs -> new/name.rs\n";
        assert_eq!(parse_porcelain(out), vec!["new/name.rs"]);
    }

    #[test]
    fn parse_porcelain_empty() {
        assert!(parse_porcelain("").is_empty());
    }

    #[test]
    fn build_prompt_carries_failure_and_files() {
        let p = build_prompt(
            "BUILD A WIDGET",
            "EXPECTED foo GOT bar",
            &["src/a.rs".to_string()],
            "",
        );
        assert!(p.contains("EXPECTED foo GOT bar"));
        assert!(p.contains("src/a.rs"));
        assert!(p.contains("BUILD A WIDGET"));
    }

    #[test]
    fn system_prompt_forbids_plan_and_edits() {
        // Planning must be hidden from the debugger at the PROMPT level too,
        // not just by withholding the tool.
        assert!(DEBUGGER_SYSTEM_PROMPT.contains("READ-ONLY"));
        assert!(DEBUGGER_SYSTEM_PROMPT.contains("CANNOT edit"));
        assert!(DEBUGGER_SYSTEM_PROMPT.contains("Do NOT plan"));
        assert!(!DEBUGGER_SYSTEM_PROMPT.contains("plan(action"));
    }

    #[test]
    fn build_prompt_handles_no_changed_files() {
        let p = build_prompt("goal", "boom", &[], "");
        assert!(p.contains("could not list changed files"));
    }
}
