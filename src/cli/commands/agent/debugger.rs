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

/// The debugger's JUDGE+REWIND prompt (`tools.debugger_judge_rewind`, requires
/// `debugger_judge`): used INSTEAD of `DEBUGGER_JUDGE_PROMPT` only when a
/// mechanical scan (`find_rewind_candidate`) found a single-file rewind
/// candidate. A tier-1 probe found the judge basically never proposes a
/// targeted revert itself even when explicitly offered the option and shown
/// the full revision tables (0/24) — but given a MECHANICALLY computed
/// candidate and asked only to accept or reject it, hit rate rose to 13/24.
/// So the candidate is computed in code and named here; the model only picks.
const DEBUGGER_JUDGE_REWIND_PROMPT: &str = "\
You are a READ-ONLY analyst with fresh eyes on a STUCK coding task. You have ONLY \
read/search/inspect tools — you CANNOT edit files, run shell, set a plan, or use a scratchpad. \
Do NOT plan and do NOT try to edit.\n\
Investigate the failure and the changes made so far. A mechanical scan of the edit history has \
already found ONE candidate: a specific file that regressed from a much cleaner earlier revision \
to its current, more broken state (shown below, under CANDIDATE REWIND POINT). You do not need to \
find it yourself — decide whether taking it is the right move.\n\
Choose EXACTLY ONE:\n\
(a) RESET — the damage is not limited to that one file; the whole attempt is misdirected or \
damaged everywhere, and reverting the ENTIRE tree to the clean original and starting fresh would \
be faster and more reliable. IGNORE effort already spent.\n\
(b) REWIND — the candidate is correct: reverting JUST that file to the proposed revision recovers \
real progress, and the rest of the tree (outside that file) is fine or close to fine as it stands.\n\
(c) CONTINUE — no revert is needed anywhere; the current state (including that file as-is) is on \
the path to the GOAL and only a focused forward fix remains. A high current error count or a \
compile failure is evidence AGAINST this option, not something to wave away as \"simple\" — the \
same forward-fix instinct already failed once to reach this diagnosis.\n\
Output your choice on the FIRST line, exactly one of:\n\
CHOICE: (a)\n\
CHOICE: (b)\n\
CHOICE: (c)\n\
Then one line — REASON: <the single most important reason>.\n\
If (a): STOP after REASON.\n\
If (b): STOP after REASON — the loop performs the revert, you do not.\n\
If (c): produce the recovery report the main agent will apply —\n\
ROOT CAUSE: <the precise reason the check fails>\n\
FIX: <where and what must change, described conceptually — NOT verbatim code you cannot compile-check>\n\
PLAN: <the concrete remaining steps to finish the GOAL, including the step that makes the feature \
actually work at runtime, not merely compile>";

/// The STEP judge's prompt: fired on a `stuck_check` Red freeze while a
/// skill step is active — the replacement for the removed fixed per-step
/// round budget (rounds-only triggers: 3/3 false positives in the 08-24
/// trigger eval; a fixed budget fires UNAVOIDABLY on a step whose
/// instructions the environment cannot satisfy, e.g. the 2026-08-31
/// ConfigureSSO abandonment). The satisfiability framing and the narrowed
/// (a)/(b)/(c) choice (free-form 0/24 vs narrowed 13/24) are load-bearing.
const STEP_JUDGE_PROMPT: &str = "\
You are a READ-ONLY analyst with fresh eyes on a STUCK skill step. You have ONLY \
read/search/inspect tools — you CANNOT edit files, run shell, set a plan, or use a scratchpad. \
Do NOT plan and do NOT try to edit.\n\
The main agent is executing one step of a skill (a step-by-step operating procedure) and its \
observable state (files, checks, failures) has been frozen for many rounds. Investigate: read \
the step's instructions below, the relevant files, and the changes so far. Pay particular \
attention to whether the step is SATISFIABLE at all in this environment — distilled \
instructions sometimes demand something the environment forbids or lacks (a config the app \
does not support, a file that must not be modified, a secret with no consumer). An \
unsatisfiable step can never be finished, no matter how many rounds are spent on it.\n\
Choose EXACTLY ONE:\n\
(a) CONTINUE — the step is doable and the work is close; a focused diagnosis will unstick it.\n\
(b) RETRY — the step is doable but the agent's context is poisoned (grinding one dead-end \
approach, looping on reads); the loop will compact the conversation and the agent will \
re-approach the step carrying your diagnosis.\n\
(c) ABANDON — the step as written cannot be satisfied in this environment, or its intent is \
already effectively satisfied and the agent cannot see it; the loop will mark the step \
abandoned (recorded as NOT done) and move on, revisiting it at the end of the skill if \
possible.\n\
Output your choice on the FIRST line, exactly one of:\n\
CHOICE: (a)\n\
CHOICE: (b)\n\
CHOICE: (c)\n\
Then one line — REASON: <the single most important reason>.\n\
If (a) or (b): produce the report the main agent will use —\n\
ROOT CAUSE: <the precise reason the step is not converging>\n\
FIX: <where and what must change, described conceptually — NOT verbatim code you cannot \
compile-check>\n\
If (c): add one line — WHAT WAS MISSING: <what the environment lacks or forbids that the step \
demands, or what already satisfies its intent> — and STOP.";

/// What the debugger sub-agent decided the primary agent (or the loop)
/// should do next.
#[derive(Debug, Clone)]
pub enum DebuggerVerdict {
    /// Inject this text for the primary agent: a diagnosis report (plain
    /// mode), a judge CONTINUE, or a judge (c) after a rewind was offered
    /// and declined.
    Report(String),
    /// Judge voted SCRAP, or (a) RESET in rewind mode: the loop reverts the
    /// WHOLE tree and restarts.
    Scrap,
    /// Judge voted (b) REWIND: the loop reverts JUST this one file to the
    /// mechanically-computed candidate revision.
    Rewind(tools::RewindCandidate),
}

/// The step judge's verdict on a frozen skill step. Every variant carries
/// the judge's report — the caller injects it in place of the generic stuck
/// note, so even CONTINUE changes what the model sees next round.
#[derive(Debug, Clone)]
pub enum StepVerdict {
    /// The step is doable and close — inject the diagnosis and keep going.
    Continue(String),
    /// The step is doable but the context is poisoned — the loop forces a
    /// compaction and the agent re-approaches carrying the diagnosis.
    Retry(String),
    /// The step cannot be satisfied as written in this environment (or its
    /// intent is already met) — the loop marks it abandoned, NOT done.
    Abandon(String),
}

/// Run the read-only debugger sub-agent against a blocking check failure.
/// It makes **no edits** — its output is a verdict for the loop to execute
/// (SCRAP/Rewind) or a report to inject (Report).
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
) -> DebuggerVerdict {
    let tool_defs = readonly_tools(parent_tool_defs);
    let changed = changed_files(config);

    // Mechanically find a single-file rewind candidate (opt-in, requires the
    // judge). A free-form "notice AND name it" ask scored 0/24 in a tier-1
    // probe; computing it in code and narrowing the ask to accept/reject
    // raised that to 13/24 — see `tools::find_rewind_candidate`.
    let candidate = if config.tools.debugger_judge && config.tools.debugger_judge_rewind {
        fast_revisions
            .as_deref()
            .and_then(|revs| tools::find_rewind_candidate(&changed, revs))
    } else {
        None
    };

    // Minimal, debugger-only context — NOT context::assemble (which would
    // inject the full plan-first/edit ceremony, see DEBUGGER_SYSTEM_PROMPT).
    let system_prompt = if candidate.is_some() {
        DEBUGGER_JUDGE_REWIND_PROMPT
    } else if config.tools.debugger_judge {
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
    let mut user_prompt = build_prompt(goal, failure_output, &changed, &diff);
    if let Some(ref c) = candidate {
        user_prompt.push_str(&format!(
            "\n\nCANDIDATE REWIND POINT: {} rev_{} (ast=ok, file_errors={}) — the file's CURRENT \
             state (file_errors={}) is a regression from this revision, reached via edits that \
             were never reverted.",
            c.path, c.rev, c.file_errors_then, c.file_errors_now
        ));
    }
    let report = investigate(
        system_prompt,
        &user_prompt,
        "You've finished investigating — no more reads. Write your final report now and \
         nothing else: ROOT CAUSE (precise reason the check fails) and FIX (where and what \
         must change, described conceptually — not verbatim code you cannot compile-check).",
        &tool_defs,
        config,
        llm_worker,
        perms,
        lsp,
        fast_revisions,
        fast_baseline_errors,
        cancelled,
    )
    .await;
    if report.is_empty() {
        return DebuggerVerdict::Report("(the debugger produced no diagnosis)".to_string());
    }
    parse_verdict(&report, candidate)
}

/// Judge a stuck skill step: either `stuck_check` fired Red while a skill
/// cursor is active, or the model repeatedly tried to declare the whole
/// task finished while steps remained (the stop-valve escalates instead of
/// nudging forever). A fresh-context read-only sub-agent decides CONTINUE
/// (inject a diagnosis), RETRY (compact + re-approach), or ABANDON (mark
/// the step abandoned, NOT done). `trigger` is one preformatted sentence
/// describing what tripped the escalation, so the judge sees the real
/// signal for either path. `None` when the judge produced nothing
/// (cancelled or LLM failure) — the caller falls back to its plain nudge.
#[allow(clippy::too_many_arguments)]
pub async fn run_step_judge(
    goal: &str,
    skill_name: &str,
    step_name: &str,
    step_instructions: &str,
    step_check: Option<&str>,
    rounds_on_step: usize,
    trigger: &str,
    config: &Config,
    llm_worker: &LlmWorkerHandle,
    parent_tool_defs: &[ToolDefinition],
    perms: &Arc<PermissionManager>,
    lsp: &Option<Arc<LspClient>>,
    fast_revisions: &Option<Arc<tools::RevisionStore>>,
    fast_baseline_errors: usize,
    cancelled: &Arc<AtomicBool>,
) -> Option<StepVerdict> {
    let tool_defs = readonly_tools(parent_tool_defs);
    let changed = changed_files(config);
    // Like the SCRAP/CONTINUE judge, this one needs the whole diff: whether
    // a step is unsatisfiable or merely mid-flight only shows in what the
    // work so far actually produced, not in the frozen signature.
    let diff = changed_diff(config, 500);
    let check_section = match step_check {
        Some(cmd) => format!("The step's DONE WHEN check: `{cmd}`\n\n"),
        None => String::new(),
    };
    let user_prompt = format!(
        "GOAL (the overall task):\n\
         {goal}\n\
         \n\
         The agent is executing skill '{skill_name}', step '{step_name}'. It has spent \
         {rounds_on_step} rounds on this step. {trigger}\n\
         \n\
         THE STEP'S INSTRUCTIONS (exactly as given to the agent):\n\
         ----------------------------------------\n\
         {step_instructions}\n\
         ----------------------------------------\n\
         \n\
         {check_section}\
         Files changed so far this session:\n\
         {files}\n\
         {diff_section}\
         \n\
         Investigate, then act per your instructions.",
        files = format_changed_files(&changed),
        diff_section = format_diff_section(&diff),
    );
    let report = investigate(
        STEP_JUDGE_PROMPT,
        &user_prompt,
        "You've finished investigating — no more reads. Write your final answer now and nothing \
         else, per your instructions: first line CHOICE: (a), (b) or (c); then REASON:; then \
         ROOT CAUSE and FIX for (a)/(b), or WHAT WAS MISSING for (c).",
        &tool_defs,
        config,
        llm_worker,
        perms,
        lsp,
        fast_revisions,
        fast_baseline_errors,
        cancelled,
    )
    .await;
    if report.is_empty() {
        return None;
    }
    Some(parse_step_verdict(&report))
}

/// Interpret the step judge's raw text. An unmarked report defaults to
/// CONTINUE — the do-nothing verdict (inject text, change no state).
fn parse_step_verdict(report: &str) -> StepVerdict {
    let head = decision_head(report);
    if head.contains("choice") && head.contains("(b)") {
        return StepVerdict::Retry(report.to_string());
    }
    if head.contains("choice") && head.contains("(c)") {
        return StepVerdict::Abandon(report.to_string());
    }
    StepVerdict::Continue(report.to_string())
}

/// The first few lines of a report, lowercased — where every judge flavor
/// puts its decision marker.
fn decision_head(report: &str) -> String {
    report
        .lines()
        .take(3)
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

/// Bounded read-only investigation shared by every debugger flavor: alternate
/// LLM turns and read-only tool executions until the model concludes (a turn
/// with no tool calls) or the round budget runs out; if no usable text was
/// produced, force one text-only turn with `final_ask` so the deliverable
/// always exists. Returns the trimmed report — empty on cancel/LLM failure.
#[allow(clippy::too_many_arguments)]
async fn investigate(
    system_prompt: &str,
    user_prompt: &str,
    final_ask: &str,
    tool_defs: &[ToolDefinition],
    config: &Config,
    llm_worker: &LlmWorkerHandle,
    perms: &Arc<PermissionManager>,
    lsp: &Option<Arc<LspClient>>,
    fast_revisions: &Option<Arc<tools::RevisionStore>>,
    fast_baseline_errors: usize,
    cancelled: &Arc<AtomicBool>,
) -> String {
    let mut messages = vec![Message::system(system_prompt), Message::user(user_prompt)];
    let mut report = String::new();

    for _round in 0..MAX_DEBUGGER_ROUNDS {
        if cancelled.load(Ordering::Relaxed) {
            break;
        }
        context::sanitize_messages(&mut messages);
        // Diagnosis benefits from reasoning: follow the main loop's
        // `model.thinking` opt-in (mechanical sub-roles stay non-thinking).
        let request = ChatRequest {
            messages: messages.clone(),
            tools: Some(tool_defs.to_vec()),
            tool_choice: None,
            max_tokens_override: None,
            chat_template_kwargs: Some(
                serde_json::json!({"enable_thinking": config.model.thinking}),
            ),
            temperature_override: config
                .model
                .thinking
                .then_some(config.model.thinking_temperature),
            cache_prompt: None,
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
        messages.push(Message::user(final_ask));
        // The loop above sanitizes before every request; this one is built
        // outside it and used to be sent raw. Its shape ends tool→user,
        // which Mistral-family templates count as user→user (tool messages
        // are invisible to the alternation check) → HTTP 500 on every
        // attempt, and the diagnosis was lost after 7 retries (Devstral,
        // 2026-08-23 bench). sanitize inserts the assistant bridge.
        context::sanitize_messages(&mut messages);
        let request = ChatRequest {
            messages,
            tools: None,
            tool_choice: None,
            max_tokens_override: None,
            chat_template_kwargs: Some(
                serde_json::json!({"enable_thinking": config.model.thinking}),
            ),
            temperature_override: config
                .model
                .thinking
                .then_some(config.model.thinking_temperature),
            cache_prompt: None,
        };
        if let Some(resp) = drain(llm_worker, request, cancelled).await
            && let Some(c) = resp.choices.first().and_then(|c| c.message.content.clone())
        {
            report = c;
        }
    }

    report.trim().to_string()
}

/// Interpret the debugger's raw text into a structured verdict. Only the
/// first few lines are inspected for the decision marker — the rest is the
/// report body, injected verbatim by the caller when it's a `Report`.
fn parse_verdict(report: &str, candidate: Option<tools::RewindCandidate>) -> DebuggerVerdict {
    let head = decision_head(report);
    if let Some(cand) = candidate {
        if head.contains("choice") && head.contains("(a)") {
            return DebuggerVerdict::Scrap;
        }
        if head.contains("choice") && head.contains("(b)") {
            return DebuggerVerdict::Rewind(cand);
        }
        return DebuggerVerdict::Report(report.to_string());
    }
    if head.contains("decision") && head.contains("scrap") {
        return DebuggerVerdict::Scrap;
    }
    DebuggerVerdict::Report(report.to_string())
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

/// The "files changed" prompt section, shared by every debugger flavor.
fn format_changed_files(changed: &[String]) -> String {
    if changed.is_empty() {
        "(could not list changed files — use file(action='search') to locate the relevant code)"
            .to_string()
    } else {
        changed
            .iter()
            .map(|f| format!("  - {f}"))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// The full-diff prompt section; empty input yields an empty section. The
/// JUDGE flavors need the WHOLE diff to assess on-path-ness — investigating
/// only the failing location makes them myopic (a small local fix looks
/// fixable even when the overall attempt is off-path). The plain
/// diagnostician passes an empty diff, staying failure-location-focused.
fn format_diff_section(diff: &str) -> String {
    if diff.trim().is_empty() {
        String::new()
    } else {
        format!(
            "\nThe FULL diff of the changes so far (vs the clean original) — review ALL of it to \
             judge whether the work is on-path for the GOAL, not just the failing spot:\n\
             ```diff\n{diff}\n```\n"
        )
    }
}

fn build_prompt(goal: &str, failure_output: &str, changed: &[String], diff: &str) -> String {
    let files = format_changed_files(changed);
    let diff_section = format_diff_section(diff);

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

    fn candidate() -> tools::RewindCandidate {
        tools::RewindCandidate {
            path: "src/f.rs".to_string(),
            rev: 4,
            file_errors_now: 31,
            file_errors_then: 1,
        }
    }

    #[test]
    fn parse_verdict_binary_scrap() {
        let v = parse_verdict("DECISION: SCRAP\nREASON: off-path", None);
        assert!(matches!(v, DebuggerVerdict::Scrap));
    }

    #[test]
    fn parse_verdict_binary_continue_is_a_report() {
        let v = parse_verdict("DECISION: CONTINUE\nROOT CAUSE: x", None);
        assert!(matches!(v, DebuggerVerdict::Report(s) if s.contains("ROOT CAUSE")));
    }

    #[test]
    fn parse_verdict_rewind_choice_a_is_scrap() {
        let v = parse_verdict(
            "CHOICE: (a)\nREASON: everywhere is broken",
            Some(candidate()),
        );
        assert!(matches!(v, DebuggerVerdict::Scrap));
    }

    #[test]
    fn parse_verdict_rewind_choice_b_is_rewind_with_the_candidate() {
        let v = parse_verdict("CHOICE: (b)\nREASON: take it", Some(candidate()));
        match v {
            DebuggerVerdict::Rewind(c) => assert_eq!(c, candidate()),
            other => panic!("expected Rewind, got {other:?}"),
        }
    }

    #[test]
    fn parse_verdict_rewind_choice_c_is_a_report() {
        let v = parse_verdict("CHOICE: (c)\nROOT CAUSE: y", Some(candidate()));
        assert!(matches!(v, DebuggerVerdict::Report(s) if s.contains("ROOT CAUSE")));
    }

    #[test]
    fn rewind_prompt_offers_all_three_choices_and_forbids_edits() {
        assert!(DEBUGGER_JUDGE_REWIND_PROMPT.contains("READ-ONLY"));
        assert!(DEBUGGER_JUDGE_REWIND_PROMPT.contains("CANNOT edit"));
        assert!(DEBUGGER_JUDGE_REWIND_PROMPT.contains("CHOICE: (a)"));
        assert!(DEBUGGER_JUDGE_REWIND_PROMPT.contains("CHOICE: (b)"));
        assert!(DEBUGGER_JUDGE_REWIND_PROMPT.contains("CHOICE: (c)"));
    }

    #[test]
    fn parse_step_verdict_maps_choices_and_defaults_to_continue() {
        use StepVerdict::*;
        let v = parse_step_verdict("CHOICE: (a)\nREASON: close\nROOT CAUSE: x");
        assert!(matches!(v, Continue(s) if s.contains("ROOT CAUSE")));
        let v = parse_step_verdict("CHOICE: (b)\nREASON: poisoned context");
        assert!(matches!(v, Retry(_)));
        let v = parse_step_verdict("CHOICE: (c)\nREASON: no\nWHAT WAS MISSING: y");
        assert!(matches!(v, Abandon(s) if s.contains("WHAT WAS MISSING")));
        // No CHOICE marker → the do-nothing verdict: inject text, change no
        // cursor or compaction state.
        let v = parse_step_verdict("ROOT CAUSE: something\nFIX: somewhere");
        assert!(matches!(v, Continue(_)));
    }

    #[test]
    fn step_judge_prompt_offers_all_three_choices_and_forbids_edits() {
        assert!(STEP_JUDGE_PROMPT.contains("READ-ONLY"));
        assert!(STEP_JUDGE_PROMPT.contains("CANNOT edit"));
        assert!(STEP_JUDGE_PROMPT.contains("CHOICE: (a)"));
        assert!(STEP_JUDGE_PROMPT.contains("CHOICE: (b)"));
        assert!(STEP_JUDGE_PROMPT.contains("CHOICE: (c)"));
        // The satisfiability framing is the load-bearing part of the prompt.
        assert!(STEP_JUDGE_PROMPT.contains("SATISFIABLE"));
    }
}
