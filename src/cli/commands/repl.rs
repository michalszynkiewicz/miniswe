//! Interactive REPL mode with ratatui TUI.

use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::Mutex;

use anyhow::Result;
use crossterm::ExecutableCommand;
use crossterm::event::{KeyCode, KeyModifiers};
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::Terminal;
use ratatui::backend::Backend;
use ratatui::backend::CrosstermBackend;
use tokio::sync::mpsc;

use crate::cli::commands::agent::debugger;
use crate::cli::commands::agent::display::summarize_args;
use crate::cli::commands::agent::hints::{
    PLAN_CHECKPOINT_AFTER_EDITS, PLAN_CHECKPOINT_WARNING, PLAN_PROGRESS_NUDGE,
    PREMATURE_EXIT_NUDGE, REPEATED_READ_ESCALATION, REPEATED_READ_NUDGE, cycle_loop_hint,
    is_file_write, is_prunable_refactor_failure, loop_detected_hint, truncated_tool_call_hint,
    visible_tool_defs,
};
use crate::cli::commands::agent::loop_detector::{
    cycle_period, is_mutating_call, key_is_mutating, loop_call_key,
};
use crate::cli::commands::agent::permissions::permission_action;
use crate::cli::commands::agent::spiral;
use crate::cli::commands::agent::validation;
use crate::config::{CeremonyMode, Config, EditMode, ModelRole};
use crate::context;
use crate::llm::{
    ChatRequest, Message, ModelRouter, TRUNCATED_CALL_ABORT_AFTER, is_context_exceeded_error,
    is_context_truncated_response, is_tool_call_args_cap_error, is_truncated_tool_call_error,
    sanitize_truncated_tool_calls, scrub_unparseable_tool_calls, truncated_args_info,
    truncated_args_tool_result,
};
use crate::logging::SessionLog;
use crate::lsp::LspClient;
use crate::mcp::{McpConfig, McpRegistry};
use crate::runtime::{
    LlmWorkerEvent, LlmWorkerHandle, ShellControl, ShellWorkerEvent, ToolWorkerPool,
};
use crate::tools;
use crate::tools::permissions::{Action, PermissionManager};
use crate::tui::app::{App, AppMode, LineStyle, PlanStepView};
use crate::tui::event::{self, AppEvent};
use crate::tui::ui;

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
async fn classify_is_explore(
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
fn is_explore_reply(s: &str) -> bool {
    s.trim().to_ascii_uppercase().starts_with("EXPLORE")
}

/// Read-only tool subset for EXPLORE turns: drop every writer/mutator
/// (so the model is never *offered* an edit tool) and the plan tool
/// (no planning in Q&A). Keeps file:read/search, code:* (LSP/repo
/// map), web, show_rev/check.
fn read_only_tool_defs(all: &[crate::llm::ToolDefinition]) -> Vec<crate::llm::ToolDefinition> {
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
fn shell_is_read_only(command: &str) -> bool {
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
fn explore_block_reason(name: &str, file_action: &str, args: &serde_json::Value) -> Option<String> {
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

struct ReplTerminalGuard;

impl ReplTerminalGuard {
    fn enter() -> Result<Self> {
        terminal::enable_raw_mode()?;
        io::stdout().execute(EnterAlternateScreen)?;
        Ok(Self)
    }
}

impl Drop for ReplTerminalGuard {
    fn drop(&mut self) {
        let _ = terminal::disable_raw_mode();
        let _ = io::stdout().execute(LeaveAlternateScreen);
    }
}

/// Run the interactive REPL with TUI.
pub async fn run(mut config: Config, headless: bool, continue_session: bool) -> Result<()> {
    let log = Arc::new(SessionLog::new(&config));

    let router = Arc::new(ModelRouter::new(&config));
    // Probe server for the actual model identity (see run.rs for rationale).
    config.model.probed_model = router.probe_default_model().await.ok();
    let llm_worker = LlmWorkerHandle::new(router.clone(), config.runtime.llm_concurrency);
    let perms = Arc::new(if headless {
        PermissionManager::headless(&config)
    } else {
        PermissionManager::new(&config)
    });
    let tool_pool = ToolWorkerPool::new(config.runtime.tool_worker_pool_size);
    let mut tool_defs = tools::tool_definitions(config.tools.edit_mode);
    // Filter tools based on config
    {
        let mut disabled = Vec::new();
        if !config.tools.web_tools {
            disabled.push("web");
        }
        if !config.tools.plan {
            disabled.push("plan");
        }
        // Uniform across all models: refactor available, edit_file hidden.
        // Devstral carve-out removed — see run.rs for the rationale.
        disabled.push("edit_file");
        // Fast mode keeps `edit_file` available alongside the
        // primitives — see run.rs for rationale.
        tool_defs.retain(|t| !disabled.contains(&t.function.name.as_str()));
        if config.tools.edit_mode == EditMode::Fast {
            tool_defs.extend(tools::fast_mode_tool_definitions());
        }
        // Only expose spawn_agents when concurrency makes it useful (see run.rs).
        if config.runtime.llm_concurrency > 1 {
            tool_defs.push(tools::definitions::spawn_agents_tool_definition());
        }
        // Background jobs: explicit file(shell background=true) start +
        // jobs(wait/status/kill) management. Session-scoped registry so a
        // server started in one turn is manageable in later turns.
        tool_defs.push(tools::definitions::shell_tool_definition());
    }
    let job_registry = Arc::new(tools::jobs::JobRegistry::default());

    // Spawn LSP client (non-blocking)
    let lsp_client: Option<Arc<LspClient>> = if config.lsp.enabled {
        match LspClient::spawn(config.project_root.clone()).await {
            Ok(client) => Some(Arc::new(client)),
            Err(_) => None,
        }
    } else {
        None
    };

    // Fast-mode state: per-file revisions + project-wide LSP baseline.
    // Same structure as the one in run.rs — see its comment for rationale.
    let fast_revisions: Option<Arc<tools::RevisionStore>> =
        if config.tools.edit_mode == EditMode::Fast {
            let miniswe_dir = config.miniswe_path("revisions");
            tools::RevisionStore::new(&miniswe_dir).ok().map(Arc::new)
        } else {
            None
        };
    let fast_baseline_errors: usize = if config.tools.edit_mode == EditMode::Fast {
        tools::fast::project_error_count(lsp_client.as_deref()).await
    } else {
        0
    };

    // Snapshot manager for whole-tree revert support (SCRAP restart,
    // revert-to-green). Unlike run.rs (where one session IS one task, so a
    // single session-scoped instance is correct), REPL is a persistent
    // multi-turn surface — (re-)initialized fresh at the start of every turn
    // below, right before run_agent_loop, so SCRAP's revert-to-round-0 only
    // ever reverts the CURRENT turn's changes, never prior turns' work.
    // `SnapshotManager::init` wipes and recreates the shadow-git repo each
    // call, so this is just a relocation, not new plumbing. No instance
    // exists yet before the first turn runs — always overwritten before use.
    #[allow(unused_assignments)]
    let mut snapshots: Option<Arc<Mutex<tools::snapshots::SnapshotManager>>> = None;

    // Session working state (plan.md, scratchpad.md) lives in a private
    // per-session directory, so there is nothing stale to clear and no
    // shared path a concurrent or nested run could wipe out from under us.
    // `--continue` adopts the previous session's directory rather than
    // opening a fresh one.
    let sessions_dir = config.sessions_dir();
    if continue_session && let Some(previous) = crate::config::session::last_id(&sessions_dir) {
        config.session_id = previous;
    }
    let _ = config.ensure_session_dir();
    crate::config::session::record_last(&sessions_dir, &config.session_id);
    crate::config::session::prune(
        &sessions_dir,
        crate::config::session::RETENTION,
        &config.session_id,
    );

    // Initialize MCP
    let mcp_config = McpConfig::load(&config.project_root)?;
    let mcp_registry = if mcp_config.has_servers() {
        let cache_dir = config.miniswe_path("mcp");
        match McpRegistry::connect(&mcp_config, &cache_dir) {
            Ok(registry) => {
                if registry.has_servers() {
                    tool_defs.push(tools::definitions::mcp_tool_definition());
                }
                Some(Arc::new(Mutex::new(registry)))
            }
            Err(_) => None,
        }
    } else {
        None
    };

    let mcp_summary = mcp_registry
        .as_ref()
        .and_then(|r| r.lock().context_summary());

    // Token budget for compression decisions. Tool definitions are a fixed
    // overhead per request, so compute once.
    let tool_def_tokens =
        context::estimate_tokens(&serde_json::to_string(&tool_defs).unwrap_or_default());

    // Set up terminal
    let _terminal_guard = ReplTerminalGuard::enter()?;
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;

    // Set up app state
    let mut app = App::new();
    let history_file = config.miniswe_path("sessions/repl_history.txt");
    app.load_history(&history_file);

    // Welcome message — probe what the server is actually serving rather
    // than parroting config.toml, which can disagree with reality when a
    // llama-swap/llama-cpp in front of the endpoint is loading a
    // different gguf than named in config.
    for line in router.startup_summary().await {
        app.push_output(&format!("miniswe — {line}"), LineStyle::Status);
    }
    if let Some(ref mcp) = mcp_registry {
        let guard = mcp.lock();
        if guard.has_servers() {
            app.push_output(
                &format!(
                    "MCP: {} servers, {} tools",
                    guard.servers.len(),
                    guard.tool_count()
                ),
                LineStyle::Status,
            );
        }
    }
    app.push_output(
        "Type your message. Ctrl+O: details, Ctrl+C: interrupt, Ctrl+D: quit",
        LineStyle::Status,
    );
    app.push_output(
        "────────────────────────────────────────────────",
        LineStyle::Separator,
    );

    // Event channel
    let (tx, mut rx) = mpsc::unbounded_channel::<AppEvent>();
    perms.set_prompt_event_tx(tx.clone());

    // Cancellation flag for LLM
    let cancelled = Arc::new(AtomicBool::new(false));

    // Spawn keyboard reader (passes cancel flag for direct Ctrl+C handling)
    event::spawn_key_reader(tx.clone(), cancelled.clone());

    let mut conversation_history: Vec<Message> = Vec::new();

    // Main event loop
    loop {
        // Render
        terminal
            .draw(|frame| ui::draw(frame, &app))
            .map_err(io::Error::other)?;

        // Wait for next event
        let evt = match rx.recv().await {
            Some(e) => e,
            None => break,
        };

        match evt {
            AppEvent::Tick => {
                // Just triggers a re-render for spinner animation
            }

            AppEvent::Key(key) => {
                match app.mode {
                    AppMode::Detail => {
                        // In detail view: Esc, Ctrl+O, or q closes it
                        if key.code == KeyCode::Esc
                            || event::is_ctrl_o(&key)
                            || key.code == KeyCode::Char('q')
                        {
                            app.close_detail();
                        }
                    }
                    AppMode::Normal => {
                        if event::is_ctrl_d(&key) {
                            break;
                        }

                        if event::is_ctrl_c(&key) {
                            if app.is_thinking {
                                cancelled.store(true, Ordering::Relaxed);
                                app.push_output("(interrupted)", LineStyle::Status);
                                app.is_thinking = false;
                            }
                            continue;
                        }

                        if event::is_ctrl_o(&key) {
                            app.open_detail();
                            continue;
                        }

                        if app.is_thinking {
                            // Ignore input while LLM is generating
                            continue;
                        }

                        match key.code {
                            KeyCode::Enter | KeyCode::Char('\n') | KeyCode::Char('\r') => {
                                let input = app.submit_input();
                                if input.is_empty() {
                                    continue;
                                }

                                // Handle commands
                                if input == "quit" || input == "exit" || input == "/quit" {
                                    break;
                                }

                                if input == "/clear" || input == "/new" {
                                    conversation_history.clear();
                                    if input == "/new" {
                                        let _ = std::fs::remove_file(
                                            config.session_path("scratchpad.md"),
                                        );
                                        let _ =
                                            std::fs::remove_file(config.session_path("plan.md"));
                                        app.push_output(
                                            "Cleared history, scratchpad, and plan.",
                                            LineStyle::Status,
                                        );
                                    } else {
                                        app.push_output(
                                            "Cleared conversation history.",
                                            LineStyle::Status,
                                        );
                                    }
                                    continue;
                                }

                                if input == "/help" {
                                    app.push_output(
                                        "/clear — clear conversation history",
                                        LineStyle::Status,
                                    );
                                    app.push_output(
                                        "/new   — clear history + scratchpad + plan",
                                        LineStyle::Status,
                                    );
                                    app.push_output(
                                        "/skills list       — list available skills",
                                        LineStyle::Status,
                                    );
                                    app.push_output(
                                        "/skills <name> help — show skill details",
                                        LineStyle::Status,
                                    );
                                    app.push_output("/help  — show this help", LineStyle::Status);
                                    app.push_output("quit   — exit", LineStyle::Status);
                                    continue;
                                }

                                if input == "/skills" || input == "/skills list" {
                                    let entries = crate::skills::discover(&config.project_root);
                                    if entries.is_empty() {
                                        app.push_output(
                                            "No skills found in .ai/skills/",
                                            LineStyle::Status,
                                        );
                                    } else {
                                        for entry in &entries {
                                            if let Ok(skill) = crate::skills::load(&entry.path) {
                                                app.push_output(
                                                    &crate::skills::format_list_entry(&skill),
                                                    LineStyle::Status,
                                                );
                                            }
                                        }
                                    }
                                    continue;
                                }

                                if let Some(rest) = input.strip_prefix("/skills ") {
                                    let name = rest
                                        .trim_end_matches(" help")
                                        .trim_end_matches(" --help")
                                        .trim();
                                    if rest.ends_with(" help") || rest.ends_with(" --help") {
                                        match crate::skills::load_by_name(
                                            name,
                                            &config.project_root,
                                        ) {
                                            Some(skill) => {
                                                for line in crate::skills::format_help(
                                                    &skill,
                                                    &config.project_root,
                                                ) {
                                                    app.push_output(&line, LineStyle::Status);
                                                }
                                            }
                                            None => app.push_output(
                                                &format!("skill '{name}' not found"),
                                                LineStyle::Error,
                                            ),
                                        }
                                        continue;
                                    }
                                }

                                // Check for skill invocation: /skill-name [args]
                                let (user_message, active_skill_reminder) = if let Some(
                                    slash_rest,
                                ) =
                                    input.strip_prefix('/')
                                {
                                    let (name, args) =
                                        slash_rest.split_once(' ').unwrap_or((slash_rest, ""));
                                    if let Some(skill) =
                                        crate::skills::load_by_name(name, &config.project_root)
                                    {
                                        let skill_path = skill.path.clone();
                                        let perms_for_skill = perms.clone();
                                        let authorize =
                                            move || perms_for_skill.check_skill_shell(&skill_path);
                                        match crate::skills::render(&skill, args, authorize) {
                                            Ok(rendered) => {
                                                let display = if args.is_empty() {
                                                    format!("/{}", skill.name)
                                                } else {
                                                    format!("/{} {args}", skill.name)
                                                };
                                                app.push_output(
                                                    &format!("you> {display}"),
                                                    LineStyle::Normal,
                                                );
                                                let reminder = format!(
                                                    "Follow the instructions from {} (already provided as your task).",
                                                    skill.display_path(&config.project_root)
                                                );
                                                (rendered, Some(reminder))
                                            }
                                            Err(e) => {
                                                app.push_output(
                                                    &format!("skill error: {e}"),
                                                    LineStyle::Error,
                                                );
                                                continue;
                                            }
                                        }
                                    } else {
                                        app.push_output(
                                            &format!("you> {input}"),
                                            LineStyle::Normal,
                                        );
                                        (input.clone(), None)
                                    }
                                } else {
                                    app.push_output(&format!("you> {input}"), LineStyle::Normal);
                                    (input.clone(), None)
                                };

                                // Per-turn intent router (fail-safe to
                                // CODING). EXPLORE turns run a read-only,
                                // no-plan, Q&A-directed variant; the proven
                                // coding path is byte-unchanged otherwise.
                                // Pure model-driven: the artifact-mutation
                                // prompt routes build/change requests to CODING
                                // on its own (validated battery), so there is no
                                // keyword pre-route — on any parse failure the
                                // classifier still fails safe to CODING.
                                // Pre-turn skill router (fail-safe; skipped when the
                                // user already invoked a skill via /name). See
                                // agent::skill_router — probe: adoption 0/8 -> 8/8.
                                let user_message = if active_skill_reminder.is_none() {
                                    match crate::cli::commands::agent::skill_router::route_task_to_skill(
                                        &llm_worker,
                                        &config.project_root,
                                        &user_message,
                                        &cancelled,
                                    )
                                    .await
                                    {
                                        Some(skill) => {
                                            app.push_output(
                                                &format!("  · task routed to skill '{skill}'"),
                                                LineStyle::Status,
                                            );
                                            crate::cli::commands::agent::skill_router::rewrite_task_for_skill(
                                                &skill,
                                                &user_message,
                                            )
                                        }
                                        None => user_message,
                                    }
                                } else {
                                    user_message
                                };

                                let is_explore =
                                    classify_is_explore(&llm_worker, &user_message, &cancelled)
                                        .await;
                                let turn_cfg = if is_explore {
                                    let mut c = config.clone();
                                    c.tools.plan = false;
                                    c.tools.ceremony = crate::config::CeremonyMode::Off;
                                    c
                                } else {
                                    config.clone()
                                };
                                let turn_tools = if is_explore {
                                    read_only_tool_defs(&tool_defs)
                                } else {
                                    tool_defs.clone()
                                };
                                if is_explore {
                                    app.push_output(
                                        "[explore] read-only investigation — no edits. \
                                         Say e.g. \"actually, change it\" to switch to coding.",
                                        LineStyle::Status,
                                    );
                                }

                                // Run the agent loop
                                let mcp_summary_clone = mcp_summary.clone();

                                // Assemble context
                                let assembled = context::assemble(
                                    &turn_cfg,
                                    &user_message,
                                    &conversation_history,
                                    false,
                                    mcp_summary_clone.as_deref(),
                                );
                                conversation_history.push(Message::user(&user_message));

                                // Inject skill reminder into system prompt so every LLM call
                                // in this turn is reminded which skill is being executed.
                                let mut messages = assembled.messages;
                                if let Some(ref reminder) = active_skill_reminder
                                    && let Some(sys_msg) = messages.first_mut()
                                    && let Some(ref mut content) = sys_msg.content
                                {
                                    content.push_str("\n[ACTIVE SKILL]\n");
                                    content.push_str(reminder);
                                }
                                if is_explore
                                    && let Some(sys_msg) = messages.first_mut()
                                    && let Some(ref mut content) = sys_msg.content
                                {
                                    content.push_str(
                                        "\n[INVESTIGATION MODE] Read-only. Investigate with the \
                                         read tools and answer the question precisely, citing \
                                         file:line. Do NOT modify code or files. If a code change \
                                         is actually wanted, state that and ask the user to \
                                         rephrase as an edit request.\n",
                                    );
                                }

                                app.is_thinking = true;
                                // Live plan panel: show it for coding turns
                                // (it fills in as the model sets/checks steps).
                                // Suppressed for read-only Q&A (no plan there).
                                app.plan_task = if is_explore {
                                    None
                                } else {
                                    Some(input.clone())
                                };
                                app.plan_steps.clear();

                                let max_rounds = config.context.max_rounds;
                                let perms_ref = &perms;
                                let mcp_ref = &mcp_registry;
                                let conv_ref = &mut conversation_history;

                                log.user_message(&input);

                                // Fresh snapshot baseline for THIS turn: SCRAP's
                                // revert-to-round-0 must only undo this turn's
                                // changes, never prior turns' work (see the
                                // `snapshots` declaration above for why this
                                // can't be session-scoped the way run.rs's is).
                                snapshots =
                                    tools::snapshots::SnapshotManager::init(&config.project_root)
                                        .ok()
                                        .map(|s| Arc::new(Mutex::new(s)));

                                // Run agent loop inline (not spawned — needs mutable refs).
                                // Context compaction now happens EVERY round inside
                                // run_agent_loop (matching run.rs) rather than once
                                // per turn out here.
                                run_agent_loop(
                                    &mut app,
                                    &mut rx,
                                    &mut terminal,
                                    &router,
                                    &llm_worker,
                                    &tool_pool,
                                    &turn_tools,
                                    &turn_cfg,
                                    is_explore,
                                    perms_ref,
                                    mcp_ref,
                                    &cancelled,
                                    &mut messages,
                                    conv_ref,
                                    max_rounds,
                                    log.clone(),
                                    &lsp_client,
                                    &fast_revisions,
                                    fast_baseline_errors,
                                    &snapshots,
                                    tool_def_tokens,
                                    mcp_summary.as_deref(),
                                    &user_message,
                                    &job_registry,
                                )
                                .await;

                                // The agent loop may exit via early-break
                                // paths (empty choices, errors) that skip
                                // flush_tokens. Flush here so stray tokens
                                // don't sit in the buffer.
                                app.flush_tokens();

                                // The turn and any post-turn work are fully
                                // done — flush tokens, draw the separator, flip
                                // `is_thinking=false`, redraw.
                                finish_completed_turn(&mut app, &mut terminal, None, None)?;

                                // Turns that never produced a plan (the model
                                // answered without planning) shouldn't leave an
                                // empty "(exploring…)" panel lingering. Keep the
                                // panel only when a real plan exists.
                                if app.plan_steps.is_empty() {
                                    app.clear_plan();
                                }

                                // Discard any keys / ticks that queued up
                                // while `is_thinking` was true. If we don't,
                                // a paste or impatient typing during the
                                // compressor await would replay against the
                                // freshly-idle input box and appear to
                                // submit the next prompt "on its own".
                                drain_stale_key_events(&mut rx);
                            }
                            KeyCode::Backspace => app.delete_char(),
                            KeyCode::Left => app.cursor_left(),
                            KeyCode::Right => app.cursor_right(),
                            KeyCode::Up => {
                                if app.input.is_empty() {
                                    app.scroll_up(1);
                                } else {
                                    app.history_up();
                                }
                            }
                            KeyCode::Down => {
                                if app.input.is_empty() {
                                    app.scroll_down(1);
                                } else {
                                    app.history_down();
                                }
                            }
                            KeyCode::PageUp => app.scroll_up(10),
                            KeyCode::PageDown => app.scroll_down(10),
                            KeyCode::Home => {
                                if key.modifiers.contains(KeyModifiers::CONTROL) {
                                    app.scroll_offset = app.output.len().saturating_sub(1) as u16;
                                } else {
                                    app.cursor = 0;
                                }
                            }
                            KeyCode::End => {
                                if key.modifiers.contains(KeyModifiers::CONTROL) {
                                    app.scroll_offset = 0;
                                } else {
                                    app.cursor = app.input.len();
                                }
                            }
                            KeyCode::Char(c) => app.insert_char(c),
                            _ => {}
                        }
                    }
                }
            }

            AppEvent::Mouse(_) => {}
            AppEvent::PermissionRequest(prompt, response_tx) => {
                let response =
                    fulfill_permission_request(&mut app, &mut rx, &mut terminal, prompt).await;
                let _ = response_tx.send(response);
            }

            // Events from agent loop
            AppEvent::Token(token) => {
                app.push_token(&token);
            }
            AppEvent::ToolCall(name, summary) => {
                app.push_output(&format!("  → {name}({summary})"), LineStyle::ToolCall);
            }
            AppEvent::ToolResult(name, success, summary, full_content) => {
                let style = if success {
                    LineStyle::ToolOk
                } else {
                    LineStyle::ToolErr
                };
                let icon = if success { "✓" } else { "✗" };
                app.push_output(&format!("  {icon} {name}: {summary}"), style);
                app.store_tool_result(&name, &full_content);
            }
            AppEvent::Status(msg) => {
                app.push_output(&msg, LineStyle::Status);
            }
            AppEvent::LlmError(msg) => {
                app.push_output(&format!("error: {msg}"), LineStyle::Error);
                app.is_thinking = false;
            }
            AppEvent::LlmDone | AppEvent::AgentDone => {
                app.is_thinking = false;
                app.flush_tokens();
            }
        }
    }

    // Cleanup
    app.save_history(&history_file);

    // Shut down LSP
    if let Some(lsp) = lsp_client
        && let Ok(lsp) = Arc::try_unwrap(lsp)
    {
        lsp.shutdown().await;
    }

    Ok(())
}

/// Run the agent loop (LLM call → tool execution → repeat).
/// Refresh the live plan panel from `plan.md` (the single source of truth).
/// No-op when no task is active (Q&A turns). Called both at the top of each
/// round and immediately after the plan tool runs, so a checked-off step shows
/// the instant `plan(check)` returns rather than lagging to the next round.
fn refresh_plan_panel(app: &mut App, config: &Config, round: usize) {
    if app.plan_task.is_none() {
        return;
    }
    app.plan_steps = tools::plan::parsed_steps(config)
        .into_iter()
        .map(|(checked, checked_round, text)| PlanStepView {
            checked,
            checked_round,
            text,
        })
        .collect();
    app.round = round;
}

/// `compressor::force_compress` driven through the same select!-with-redraw
/// pattern the round loop uses for `maybe_compress`, so a long LLM-based
/// summarization during reactive context-exhaustion recovery keeps the TUI
/// responsive. Returns force_compress's "did anything shrink" result.
#[allow(clippy::too_many_arguments)]
async fn force_compress_responsive(
    app: &mut App,
    rx: &mut mpsc::UnboundedReceiver<AppEvent>,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    messages: &mut Vec<Message>,
    config: &Config,
    router: &ModelRouter,
    llm_worker: &LlmWorkerHandle,
    tool_def_tokens: usize,
) -> bool {
    let fut =
        context::compressor::force_compress(messages, config, router, llm_worker, tool_def_tokens);
    let mut fut = std::pin::pin!(fut);
    loop {
        tokio::select! {
            biased;
            freed = &mut fut => break freed,
            evt = rx.recv() => {
                if matches!(evt, Some(AppEvent::Tick)) {
                    let _ = terminal.draw(|frame| ui::draw(frame, app));
                }
            }
        }
    }
}

/// This runs inline in the main loop, processing events between rounds.
#[allow(clippy::too_many_arguments)]
async fn run_agent_loop(
    app: &mut App,
    rx: &mut mpsc::UnboundedReceiver<AppEvent>,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    router: &Arc<ModelRouter>,
    llm_worker: &LlmWorkerHandle,
    tool_pool: &ToolWorkerPool,
    tool_defs: &[crate::llm::ToolDefinition],
    config: &Config,
    // Explore mode: hard-block mutating tool calls at runtime (read-only shell ok).
    read_only: bool,
    perms: &Arc<PermissionManager>,
    mcp_registry: &Option<Arc<Mutex<McpRegistry>>>,
    cancelled: &Arc<AtomicBool>,
    messages: &mut Vec<Message>,
    conversation_history: &mut Vec<Message>,
    max_rounds: usize,
    log: Arc<SessionLog>,
    lsp: &Option<Arc<LspClient>>,
    fast_revisions: &Option<Arc<tools::RevisionStore>>,
    fast_baseline_errors: usize,
    snapshots: &Option<Arc<Mutex<tools::snapshots::SnapshotManager>>>,
    tool_def_tokens: usize,
    mcp_summary: Option<&str>,
    // The original user message for this turn — the recovery goal used by the
    // done-gate re-anchor, the debugger sub-agent, and whole-tree SCRAP/reset
    // re-assembly.
    goal: &str,
    // Session-scoped background-job registry (file shell background=true).
    job_registry: &Arc<tools::jobs::JobRegistry>,
) {
    // Ceremony=Strict re-enables the legacy plan-first machinery (plan gate,
    // plan/no-plan nudges, hide-edit-tools-until-plan). Derived from the
    // per-turn config, which already has ceremony forced Off for explore turns.
    let strict = config.tools.ceremony == CeremonyMode::Strict;
    let pause_at = config.context.pause_after_rounds;

    let mut round = 0;
    let mut had_error = false;
    let mut user_continued = false;
    // Track consecutive identical tool calls to detect loops
    let mut last_call_key: Option<String> = None;
    let mut same_call_streak = 0u32;
    // Short rolling history of call keys for period-2 cycle detection
    // (edit↔revert oscillation — invisible to the consecutive detector).
    let mut recent_call_keys: Vec<String> = Vec::new();
    // Number of distinct loops the model has been pulled out of in this
    // turn. We give one recovery; a second loop ends the turn for real.
    let mut loop_recoveries = 0u32;
    // Read-loop escalation ladder (see run.rs for the full rationale): first
    // detection gets REPEATED_READ_NUDGE; a re-detection forces a context
    // compaction before the next request — breaking the cache-hot prefix is
    // what actually ends the loop. Resets after each escalation.
    let mut read_nudges = 0u32;
    let mut force_compact_next_round = false;
    let mut calls_since_last_edit = 0u32;
    let mut successful_edits_since_plan_update = 0u32;
    let mut plan_update_requested = false;
    let mut nudged_premature_exit = false;
    let mut nudged_no_plan = false;
    // Consecutive reactive-compaction retries (context exhaustion signaled
    // by the server — see compressor::force_compress). Reset whenever a
    // response is successfully consumed; bounds futile retries of one
    // failing request, not total compactions over a long turn.
    let mut context_compact_retries: usize = 0;
    // Consecutive LLM requests that died on a tool-call argument problem
    // (server-side parse error or our streaming size cap) with no completed
    // response in between. See run.rs for the escalation ladder.
    let mut truncated_call_errors_in_a_row: usize = 0;
    // How many times the behavioral done-gate has blocked completion this turn.
    let mut validation_blocks: usize = 0;
    // The model's stated rationale each time the gate blocked it (bounded, auditable).
    let mut validation_disputes: Vec<String> = Vec::new();
    // Reactive-debugger / restart / replan bookkeeping (each fires at most once
    // per turn; debugger_multifire walks the failure chain up to MAX fires).
    let mut replan_fired = false;
    let mut restart_fired = false;
    let mut debugger_fires = 0usize;
    let mut last_debugged_failure: Option<String> = None;
    // `plan_gate_debugger`: consecutive plan(check) failures on the SAME step.
    let mut same_plan_step_failures: u32 = 0;
    let mut last_failed_plan_step: Option<u64> = None;
    // Gate-triggered context resets fired this turn (bounded — don't loop).
    let mut gate_resets: usize = 0;
    // Spiral-reset: per-file revert counts + how many resets fired this turn.
    let mut revert_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    let mut spiral_resets: usize = 0;
    // revert-to-green state (opt-in `tools.revert_to_green`): the last round
    // whose start-of-round snapshot was green (project errors ≤ baseline) and
    // how many consecutive rounds the project has stayed broken.
    const REVERT_TO_GREEN_BLOCKS: usize = 6;
    let mut last_green_round: usize = 0;
    let mut red_streak: usize = 0;

    'round: loop {
        if had_error {
            break;
        }
        // Check cancellation at the top of every round
        if consume_interrupt(cancelled) {
            app.push_output("(interrupted)", LineStyle::Status);
            break;
        }

        round += 1;
        log.round_start(round);

        // Snapshot at the start of each round for revert support (SCRAP /
        // revert-to-green rely on these per-round commits in the shadow repo).
        if let Some(snap) = snapshots {
            let mut guard = snap.lock();
            let _ = guard.begin_round(round);
        }

        // revert-to-green: if the project has been broken above baseline for
        // REVERT_TO_GREEN_BLOCKS rounds, the agent is digging deeper, not
        // recovering — reset the whole tree to the last green snapshot.
        if config.tools.revert_to_green
            && config.tools.edit_mode == EditMode::Fast
            && let Some(snap) = snapshots
        {
            let errs = tools::fast::project_error_count(lsp.as_deref()).await;
            if errs <= fast_baseline_errors {
                last_green_round = round;
                red_streak = 0;
            } else {
                red_streak += 1;
                if red_streak >= REVERT_TO_GREEN_BLOCKS {
                    let result = {
                        let guard = snap.lock();
                        guard.revert_to_round(last_green_round)
                    };
                    match result {
                        Ok(m) => {
                            app.push_output(
                                &format!("[revert-to-green] stuck {red_streak} rounds; {m}"),
                                LineStyle::Status,
                            );
                            messages.push(Message::user(&format!(
                                "[auto-revert-to-green] The project has had compile errors for \
                                 {red_streak} rounds straight and you are not converging — you are \
                                 digging deeper, not recovering. I reverted the ENTIRE working tree \
                                 to round {last_green_round}, the last state that compiled cleanly. \
                                 Your edits since then are GONE; do not replay them. Start over from \
                                 this clean base: re-read the relevant code, make ONE small complete \
                                 change, and run a check before continuing."
                            )));
                            red_streak = 0;
                        }
                        Err(e) => {
                            app.push_output(
                                &format!("[revert-to-green] revert failed: {e}"),
                                LineStyle::Error,
                            );
                        }
                    }
                }
            }
        }

        if round > max_rounds {
            app.push_output("Maximum tool rounds reached.", LineStyle::Error);
            break;
        }

        // Ask the user whether to continue after pause_after_rounds rounds.
        if round == pause_at && !user_continued {
            app.pending_permission = Some(format!(
                "{pause_at} tool rounds used. Continue? [y]es / [n]o:"
            ));
            app.input.clear();
            app.cursor = 0;
            let response = wait_for_modal_input(app, rx, terminal, &['y', 'n']).await;
            app.pending_permission = None;
            match response.as_str() {
                "y" | "yes" | "" => user_continued = true,
                _ => messages.push(Message::user("[Stop now. Summarize what you've done.]")),
            }
        }

        // Warn the LLM when approaching the hard limit.
        if round == max_rounds.saturating_sub(5) {
            messages.push(Message::user(
                "[Approaching tool limit. Wrap up and summarize.]",
            ));
        }

        // Refresh the live plan panel from plan.md (single source of truth).
        refresh_plan_panel(app, config, round);

        // Unified context compression — handles both tool results and
        // conversation, every round (matching run.rs). Driven through a
        // select! so a long LLM-based summarization keeps the TUI responsive.
        {
            let pre = messages.len();
            // Read-loop escalation (see REPEATED_READ_ESCALATION): the loop
            // is sustained by the cache-hot prompt prefix, so break it
            // deliberately even though no budget pressure asks for it. Runs
            // before maybe_compress so refresh_current_state still lands on
            // the tail.
            if force_compact_next_round {
                force_compact_next_round = false;
                app.push_output(
                    "  ⚠ Read loop persisted — forcing context compaction",
                    LineStyle::Status,
                );
                force_compress_responsive(
                    app,
                    rx,
                    terminal,
                    messages,
                    config,
                    router,
                    llm_worker,
                    tool_def_tokens,
                )
                .await;
            }
            {
                let compress_fut = context::compressor::maybe_compress(
                    messages,
                    config,
                    router,
                    llm_worker,
                    tool_def_tokens,
                    &mut plan_update_requested,
                );
                let mut compress_fut = std::pin::pin!(compress_fut);
                let mut done = false;
                while !done {
                    tokio::select! {
                        biased;
                        () = &mut compress_fut, if !done => { done = true; }
                        evt = rx.recv() => {
                            if matches!(evt, Some(AppEvent::Tick)) {
                                let _ = terminal.draw(|frame| ui::draw(frame, app));
                            }
                        }
                    }
                }
            }
            log.masking_applied(pre.saturating_sub(messages.len()), pre);
        }

        // Sanitize messages
        context::sanitize_messages(messages);

        // Hide edit tools until a plan exists; see visible_tool_defs.
        let plan_set = tools::plan::plan_exists(config);
        // Off: never hide edit tools (pass plan_exists=true). Strict: legacy
        // hide-until-plan behavior.
        let visible = visible_tool_defs(tool_defs, plan_set || !strict);
        // Build request. See run.rs for the per-model reasoning_effort and
        // thinking-mode logic.
        let (chat_template_kwargs, temperature_override) =
            if config.model.is_mistral_small_4_family() {
                let effort = if plan_set { "none" } else { "high" };
                (serde_json::json!({"reasoning_effort": effort}), None)
            } else if config.model.thinking {
                (
                    serde_json::json!({"enable_thinking": true}),
                    Some(config.model.thinking_temperature),
                )
            } else {
                (serde_json::json!({"enable_thinking": false}), None)
            };
        // Bump output budget for Mistral 4 — see run.rs for rationale
        // (probe data: 8K truncates with empty content, 16K emits clean
        // correct output at ~6K tokens used).
        let max_tokens_override = if config.model.is_mistral_small_4_family() {
            Some(16384)
        } else {
            None
        };
        let request = ChatRequest {
            messages: messages.clone(),
            tools: Some(visible),
            tool_choice: None,
            max_tokens_override,
            chat_template_kwargs: Some(chat_template_kwargs),
            temperature_override,
            cache_prompt: None,
        };
        log.llm_request(&request);

        cancelled.store(false, Ordering::Relaxed);
        app.is_thinking = true;
        app.set_active_job("llm");

        // Render before LLM call so spinner is visible immediately
        let _ = terminal.draw(|frame| ui::draw(frame, app));

        // Call LLM with streaming — render on each token
        let mut rendered_assistant_text = String::new();
        // Set when the server rejected the model's tool call as
        // truncated JSON. In that case we inject a synthetic user-role
        // hint and continue the outer loop so the agent can recover
        // with a smaller operation, instead of aborting the session.
        let mut truncated_tool_call_hint_pushed = false;
        // Set when the failure is really CONTEXT EXHAUSTION (the server
        // rejected an over-size request, or clipped a tool call because the
        // prompt sits near the window). Handled after the select loop —
        // force_compress is a long await that must not run inside it.
        let mut context_ceiling_hit = false;
        let response = {
            let mut token_count = 0u32;
            let mut llm_events =
                llm_worker.submit(ModelRole::Default, request.clone(), cancelled.clone());
            loop {
                tokio::select! {
                    evt = llm_events.recv() => {
                        match evt {
                            Some(LlmWorkerEvent::Token(token)) => {
                                app.push_token(&token);
                                rendered_assistant_text.push_str(&token);
                                token_count += 1;
                                if token_count.is_multiple_of(3) {
                                    let _ = terminal.draw(|frame| ui::draw(frame, app));
                                }
                            }
                            Some(LlmWorkerEvent::Completed(Ok(r))) => break Some(r),
                            Some(LlmWorkerEvent::Completed(Err(err_str))) => {
                                if err_str.contains("Interrupted") {
                                    cancelled.store(false, Ordering::Relaxed);
                                    app.push_output("Generation interrupted.", LineStyle::Status);
                                } else if is_context_exceeded_error(&err_str)
                                    && context_compact_retries
                                        < context::compressor::FORCE_COMPRESS_MAX_RETRIES
                                {
                                    // Prompt alone exceeds the context window
                                    // — recoverable by compacting + resending
                                    // (primary path for compaction="lazy",
                                    // safety net for every other strategy).
                                    context_ceiling_hit = true;
                                } else if is_tool_call_args_cap_error(&err_str) {
                                    // Our streaming assembler aborted the
                                    // generation: an anchor-only tool's
                                    // arguments outgrew the cap. Nothing was
                                    // persisted; hint and retry, or give up
                                    // when the model keeps doing it.
                                    truncated_call_errors_in_a_row += 1;
                                    if truncated_call_errors_in_a_row
                                        >= TRUNCATED_CALL_ABORT_AFTER
                                    {
                                        log.llm_error(&format!(
                                            "{truncated_call_errors_in_a_row} consecutive oversized tool calls — aborting turn"
                                        ));
                                        // Falls through to `break None` below:
                                        // no hint flag set, so the turn ends.
                                        app.push_output(
                                            "The model keeps emitting oversized tool-call arguments — giving up on this turn.",
                                            LineStyle::Error,
                                        );
                                    } else {
                                    log.llm_error(&format!(
                                        "tool call aborted by the argument size cap: {err_str}"
                                    ));
                                    app.push_output(
                                        "Tool call arguments exceeded the size cap — retrying with guidance.",
                                        LineStyle::Status,
                                    );
                                    let hint = Message::user(&format!(
                                        "{err_str}. Anchor-style tools take identifiers and short expressions only — \
                                         never paste code bodies into their arguments. {}",
                                        truncated_tool_call_hint(config.tools.edit_mode)
                                    ));
                                    messages.push(hint.clone());
                                    conversation_history.push(hint);
                                    truncated_tool_call_hint_pushed = true;
                                    }
                                } else if is_truncated_tool_call_error(&err_str) {
                                    // The server's chat template could not
                                    // parse some assistant tool call's
                                    // arguments as JSON: either this
                                    // response was cut off mid-call
                                    // (nothing persisted), or a previously
                                    // persisted call is broken and every
                                    // request will fail until it is gone.
                                    // Handle the second first — it is a
                                    // zero-progress spin otherwise.
                                    truncated_call_errors_in_a_row += 1;
                                    let scrubbed = if truncated_call_errors_in_a_row >= 2 {
                                        scrub_unparseable_tool_calls(messages)
                                            + scrub_unparseable_tool_calls(conversation_history)
                                    } else {
                                        0
                                    };
                                    if scrubbed > 0 {
                                        log.llm_error(&format!(
                                            "scrubbed {scrubbed} unparseable tool call(s) from history after repeated parse failures — retrying"
                                        ));
                                        app.push_output(
                                            "Repaired a truncated tool call left in history — retrying.",
                                            LineStyle::Status,
                                        );
                                        truncated_tool_call_hint_pushed = true;
                                    } else if truncated_call_errors_in_a_row
                                        >= TRUNCATED_CALL_ABORT_AFTER
                                    {
                                        log.llm_error(&format!(
                                            "{truncated_call_errors_in_a_row} consecutive tool-call parse failures with nothing left to repair — aborting turn"
                                        ));
                                        // Falls through to `break None`: turn ends.
                                        app.push_output(
                                            "The server keeps rejecting tool-call arguments — giving up on this turn.",
                                            LineStyle::Error,
                                        );
                                    } else if context::compressor::estimated_context_tokens(
                                        messages,
                                        tool_def_tokens,
                                    ) > config.model.context_window * 3 / 4
                                        && context_compact_retries
                                            < context::compressor::FORCE_COMPRESS_MAX_RETRIES
                                    {
                                        context_ceiling_hit = true;
                                    } else {
                                        // Clear the partial UI text (don't
                                        // persist the half-streamed output)
                                        // and push a user-role hint so the
                                        // agent retries with a smaller
                                        // operation.
                                        log.llm_error(
                                            "tool call JSON truncated (max_tokens) — \
                                             injecting hint and continuing",
                                        );
                                        app.push_output(
                                            "Previous tool call truncated — retrying with guidance.",
                                            LineStyle::Status,
                                        );
                                        let hint = Message::user(truncated_tool_call_hint(
                                            config.tools.edit_mode,
                                        ));
                                        messages.push(hint.clone());
                                        conversation_history.push(hint);
                                        truncated_tool_call_hint_pushed = true;
                                    }
                                } else {
                                    let clean = if err_str.contains('<') {
                                        err_str
                                            .split('<')
                                            .next()
                                            .unwrap_or(&err_str)
                                            .trim()
                                            .to_string()
                                    } else {
                                        err_str
                                    };
                                    log.llm_error(&clean);
                                    app.push_output(&format!("LLM error: {clean}"), LineStyle::Error);
                                }
                                app.clear_active_job();
                                break None;
                            }
                            None => {
                                app.push_output("LLM worker stopped unexpectedly.", LineStyle::Error);
                                app.clear_active_job();
                                break None;
                            }
                        }
                    }
                    app_evt = rx.recv() => {
                        match app_evt {
                            Some(AppEvent::Tick) => {
                                let _ = terminal.draw(|frame| ui::draw(frame, app));
                            }
                            Some(AppEvent::Key(key)) if handle_background_key(app, &key) => {
                                let _ = terminal.draw(|frame| ui::draw(frame, app));
                            }
                            Some(AppEvent::Key(key)) if event::is_ctrl_c(&key) => {
                                cancelled.store(true, Ordering::Relaxed);
                                app.push_output("(interrupted)", LineStyle::Status);
                                let _ = terminal.draw(|frame| ui::draw(frame, app));
                            }
                            Some(AppEvent::Mouse(_)) => {}
                            Some(AppEvent::PermissionRequest(prompt, response_tx)) => {
                                let response =
                                    fulfill_permission_request(app, rx, terminal, prompt).await;
                                let _ = response_tx.send(response);
                            }
                            Some(_) => {}
                            None => {
                                app.push_output("Event stream closed.", LineStyle::Error);
                                app.clear_active_job();
                                break None;
                            }
                        }
                    }
                }
            }
        };

        app.clear_active_job();

        // Re-render after LLM response
        let _ = terminal.draw(|frame| ui::draw(frame, app));

        let Some(response) = response else {
            if context_ceiling_hit {
                context_compact_retries += 1;
                app.push_output(
                    "Context window exceeded — compacting and retrying.",
                    LineStyle::Status,
                );
                if force_compress_responsive(
                    app,
                    rx,
                    terminal,
                    messages,
                    config,
                    router,
                    llm_worker,
                    tool_def_tokens,
                )
                .await
                {
                    log.llm_error("context window exceeded — compacted history, retrying");
                    continue;
                }
                // Nothing could be freed — retrying would fail identically.
                app.push_output(
                    "Compaction could not free any context — stopping this turn.",
                    LineStyle::Error,
                );
                break;
            }
            if truncated_tool_call_hint_pushed {
                // Hint was injected into `messages`; loop back and let
                // the agent try again with smaller operations.
                continue;
            }
            break;
        };

        // A 200 response can still be a context-exhaustion casualty:
        // finish_reason="length" with the completion well under the
        // requested cap means the server clipped generation at n_ctx (see
        // run.rs / is_context_truncated_response). Discard the partial
        // output, compact, regenerate.
        let effective_max_tokens =
            max_tokens_override.unwrap_or(config.model.max_output_tokens as u64) as usize;
        if is_context_truncated_response(&response, effective_max_tokens)
            && context::compressor::estimated_context_tokens(messages, tool_def_tokens)
                > config.model.context_window * 3 / 4
            && context_compact_retries < context::compressor::FORCE_COMPRESS_MAX_RETRIES
        {
            context_compact_retries += 1;
            app.push_output(
                "Generation truncated by context ceiling — compacting and regenerating.",
                LineStyle::Status,
            );
            if force_compress_responsive(
                app,
                rx,
                terminal,
                messages,
                config,
                router,
                llm_worker,
                tool_def_tokens,
            )
            .await
            {
                log.llm_error(
                    "generation truncated by context ceiling — compacted history, regenerating",
                );
                continue;
            }
        }

        let choice = match response.choices.first() {
            Some(c) => c,
            None => break,
        };
        // A response made it through whole — any prior reactive-compaction
        // retries resolved this request; reset the budget for the next one.
        context_compact_retries = 0;
        truncated_call_errors_in_a_row = 0;

        // Never persist an unparseable tool call (see run.rs): stub the
        // cut-off arguments; the tool loop answers the stub with guidance.
        let mut assistant_msg = choice.message.clone();
        let truncated_calls = sanitize_truncated_tool_calls(&mut assistant_msg);
        if truncated_calls > 0 {
            log.llm_error(&format!(
                "{truncated_calls} tool call(s) arrived with unparseable arguments (cut off by the output limit) — stubbed before persisting"
            ));
            app.push_output(
                "A tool call was cut off by the output limit — it will not be executed.",
                LineStyle::Status,
            );
        }
        let assistant_msg = &assistant_msg;

        // Flush any remaining tokens
        app.flush_tokens();

        if let Some(content) = &assistant_msg.content {
            log.llm_response(content);
            if let Some(missing) =
                reconcile_streamed_assistant_content(&rendered_assistant_text, content)
            {
                app.push_token(&missing);
                app.flush_tokens();
                let _ = terminal.draw(|frame| ui::draw(frame, app));
            }
        }
        if assistant_msg.is_meaningful() {
            conversation_history.push(assistant_msg.clone());
        }

        let tool_calls = match &assistant_msg.tool_calls {
            Some(tc) if !tc.is_empty() => tc.clone(),
            _ => {
                // See run.rs for rationale — nudge on both "mid-plan exit"
                // and "no-plan exit" (the latter caught Mistral Small 4
                // bailing during exploration before any meaningful work).
                // Strict/legacy only.
                if strict && !nudged_premature_exit && config.tools.plan {
                    let has_unchecked = tools::plan::has_unchecked_steps(config);
                    let plan_exists = tools::plan::plan_exists(config);
                    if has_unchecked || !plan_exists {
                        nudged_premature_exit = true;
                        let nudge_text = if plan_exists {
                            PREMATURE_EXIT_NUDGE.to_string()
                        } else {
                            "[You returned no tool call before setting a plan. \
                             Don't exit yet — call plan(action='set') with your \
                             step-by-step approach (or file/code if you need more \
                             exploration). The task isn't done.]"
                                .to_string()
                        };
                        let nudge = Message::user(&nudge_text);
                        messages.push(nudge.clone());
                        conversation_history.push(nudge);
                        continue;
                    }
                }

                // Behavioral done-gate: before accepting completion, verify the
                // change actually works at runtime. Default config has no
                // command → no-op. See docs/success-validation-design.md.
                // Skipped for read-only (explore) turns — a Q&A turn makes no
                // edits, so there is nothing to behaviorally verify and blocking
                // the answer would be nonsensical.
                if !read_only
                    && validation_blocks < config.validation.max_retries
                    && config.validation.command().is_some()
                {
                    match validation::run_behavioral_check(config).await {
                        validation::CheckOutcome::Fail(output) => {
                            validation_blocks += 1;
                            // Record the model's completion rationale (its
                            // no-tool-call exit content) — a bounded, auditable
                            // voice, not a silent free pass.
                            if let Some(rationale) = assistant_msg
                                .content
                                .as_deref()
                                .map(str::trim)
                                .filter(|c| !c.is_empty())
                            {
                                tracing::warn!(
                                    "[validation] blocked completion (attempt {validation_blocks}); model rationale: {}",
                                    crate::truncate_chars(rationale, 300)
                                );
                                validation_disputes.push(rationale.to_string());
                            }
                            app.push_output(
                                "Behavioral check failed — not done yet.",
                                LineStyle::Status,
                            );

                            // Full restart (opt-in `gate_restart`): on the FIRST
                            // gate block, abandon the possibly-poisoned attempt —
                            // revert the WHOLE tree to the clean baseline AND
                            // reset the context. Fires once per turn.
                            if config.tools.gate_restart && !restart_fired {
                                restart_fired = true;
                                *messages =
                                    scrap_restart(app, config, goal, mcp_summary, snapshots, false);
                                conversation_history.clear();
                                validation_blocks = 0;
                                same_plan_step_failures = 0;
                                last_failed_plan_step = None;
                                continue;
                            }

                            // Goal re-anchor (opt-in `gate_replan`): re-anchor on
                            // the ORIGINAL goal and force a fresh plan — but skip
                            // when the block is a COMPILE failure (re-anchoring on
                            // a broken tree just digs deeper).
                            let is_compile_fail = output.contains("DOES NOT COMPILE")
                                || output.contains("could not compile")
                                || output.contains("error[E");
                            if config.tools.gate_replan && !replan_fired && !is_compile_fail {
                                replan_fired = true;
                                app.push_output(
                                    "Re-anchoring on the original goal — re-plan from the task…",
                                    LineStyle::Status,
                                );
                                let msg = Message::user(&format!(
                                    "[A check that exercises the change end-to-end FAILED — it \
                                     COMPILES but does not yet BEHAVE as required. After fixing \
                                     errors it is easy to lose the original goal and stop at \"it \
                                     compiles\". Re-anchor on the task: \"{goal}\". Use \
                                     plan(action='set') to re-derive the FULL plan from that goal — \
                                     list every step the feature needs end-to-end, INCLUDING the \
                                     code that actually USES the new input to change behavior (not \
                                     just declaring or plumbing it). For each step, confirm it is \
                                     DONE in the code, not merely compiling — then implement \
                                     whatever is missing before finishing.\nCheck output:\n{output}]"
                                ));
                                messages.push(msg.clone());
                                conversation_history.push(msg);
                                continue;
                            }

                            // Reactive debugger (opt-in): hand the SPECIFIC
                            // failure to a fresh-context sub-agent once the
                            // primary agent has failed the gate a couple times.
                            let fkey = crate::cli::commands::run::failure_key(&output);
                            let may_fire = if config.tools.debugger_multifire {
                                debugger_fires < debugger::MAX_DEBUGGER_FIRES
                                    && last_debugged_failure.as_deref() != Some(fkey.as_str())
                            } else {
                                debugger_fires == 0
                            };
                            if (config.tools.reactive_debugger || config.tools.debugger_judge)
                                && may_fire
                                && validation_blocks >= debugger::DEBUGGER_TRIGGER_BLOCKS
                            {
                                debugger_fires += 1;
                                last_debugged_failure = Some(fkey);
                                app.push_output(
                                    "Still failing — spinning up a fresh-context debugger sub-agent…",
                                    LineStyle::Status,
                                );
                                let verdict = debugger::run_debugger(
                                    &output,
                                    goal,
                                    config,
                                    llm_worker,
                                    tool_pool,
                                    tool_defs,
                                    perms,
                                    mcp_registry,
                                    lsp,
                                    fast_revisions,
                                    fast_baseline_errors,
                                    cancelled,
                                )
                                .await;

                                let msg = match verdict {
                                    debugger::DebuggerVerdict::Scrap if !restart_fired => {
                                        restart_fired = true;
                                        *messages = scrap_restart(
                                            app,
                                            config,
                                            goal,
                                            mcp_summary,
                                            snapshots,
                                            true,
                                        );
                                        conversation_history.clear();
                                        validation_blocks = 0;
                                        same_plan_step_failures = 0;
                                        last_failed_plan_step = None;
                                        continue;
                                    }
                                    debugger::DebuggerVerdict::Scrap => Message::user(
                                        "[A fresh-context review voted to reset again, but the \
                                         tree was already reset once this turn. Keep going: read \
                                         the current failure carefully and fix it directly.]",
                                    ),
                                    debugger::DebuggerVerdict::Rewind(candidate) => {
                                        rewind_message_repl(
                                            app,
                                            &candidate,
                                            config,
                                            perms,
                                            lsp,
                                            fast_revisions,
                                            fast_baseline_errors,
                                            &output,
                                        )
                                        .await
                                    }
                                    debugger::DebuggerVerdict::Report(body) => {
                                        let output_note =
                                            crate::cli::commands::run::write_gate_failure_output(
                                                config, &output,
                                            )
                                            .map(|path| {
                                                format!(
                                                    "\nFull raw check output: read(\"{path}\")."
                                                )
                                            })
                                            .unwrap_or_default();
                                        Message::user(&format!(
                                            "[A read-only debugger with fresh eyes investigated the failing \
                                         check and produced this DIAGNOSIS. It did not edit anything — \
                                         YOU must apply the fix and finish the plan it lays out:\n{body}\n\
                                         Make the change(s), then finish; the verification will re-run.{output_note}]"
                                        ))
                                    }
                                };
                                messages.push(msg.clone());
                                conversation_history.push(msg);
                                continue;
                            }

                            // Gate context-reset (opt-in): drop the polluted
                            // history and re-assemble a clean context (files
                            // persist on disk). Bounded per turn.
                            if config.tools.gate_context_reset
                                && gate_resets < spiral::MAX_GATE_RESETS
                                && validation_blocks >= spiral::GATE_RESET_AFTER_BLOCKS
                            {
                                gate_resets += 1;
                                validation_blocks = 0;
                                let fresh = spiral::build_gate_reset_prompt(goal, &output);
                                let assembled =
                                    context::assemble(config, &fresh, &[], false, mcp_summary);
                                *messages = assembled.messages;
                                app.push_output(
                                    "Gate context-reset — fresh start (history cleared, files kept).",
                                    LineStyle::Status,
                                );
                                log.tool_debug(
                                    "agent",
                                    "gate context-reset: re-assembled clean context after repeated gate blocks",
                                );
                                continue;
                            }

                            let msg = Message::user(&format!(
                                "[Verification failed — do NOT finish yet. A check that exercises \
                                 the change end-to-end exited non-zero; the output below shows what \
                                 is actually wrong. Read it carefully and fix the SPECIFIC problem \
                                 it reports (it may be a compile error, not a logic error), then \
                                 continue. (If you are certain the check itself is wrong, finish \
                                 anyway and state the specific reason — it will be recorded.)\n\
                                 Check output:\n{output}]"
                            ));
                            messages.push(msg.clone());
                            conversation_history.push(msg);
                            continue;
                        }
                        validation::CheckOutcome::Pass | validation::CheckOutcome::Skipped => {}
                    }
                }
                // Exiting now. Surface any recorded gate rationale(s) for audit.
                if !validation_disputes.is_empty() {
                    app.push_output(
                        &format!(
                            "Completed after {} blocked verification(s); model's reasons recorded in the log.",
                            validation_disputes.len()
                        ),
                        LineStyle::Status,
                    );
                    tracing::warn!(
                        "[validation] turn completed over {} blocked check(s); model rationale(s): {}",
                        validation_disputes.len(),
                        validation_disputes.join(" | ")
                    );
                }
                break;
            }
        };

        messages.push(assistant_msg.clone());

        // See run.rs for the rationale — both buffers' last entry is the
        // assistant_msg we just pushed, so truncate one before to also
        // drop it. If every tool call in this assistant message turns out
        // to be a prunable validator failure, we rewind here and replace
        // with a single user-role corrective.
        let messages_pre = messages.len() - 1;
        let history_pre = if conversation_history
            .last()
            .is_some_and(|m| m.role == "assistant")
        {
            conversation_history.len() - 1
        } else {
            conversation_history.len()
        };
        let mut all_prunable_failures = !tool_calls.is_empty();
        let mut prunable_errors: Vec<String> = Vec::new();

        // Execute tool calls
        for tc in &tool_calls {
            // Check cancellation between tool calls
            if consume_interrupt(cancelled) {
                app.push_output("(interrupted)", LineStyle::Status);
                break 'round;
            }
            let args: serde_json::Value = match serde_json::from_str(&tc.function.arguments) {
                Ok(v) => v,
                Err(e) => {
                    let result_msg = Message::tool_result(
                        &tc.id,
                        &format!("Invalid JSON in tool arguments: {e}"),
                    );
                    messages.push(result_msg.clone());
                    conversation_history.push(result_msg);
                    app.push_output(
                        &format!("  ✗ {}: invalid JSON args", tc.function.name),
                        LineStyle::ToolErr,
                    );
                    continue;
                }
            };
            if let Some(info) = truncated_args_info(&args) {
                // Stubbed by sanitize_truncated_tool_calls: nothing to run.
                let result_msg = Message::tool_result(
                    &tc.id,
                    &format!(
                        "{}\n\n{}",
                        truncated_args_tool_result(&tc.function.name, &info),
                        truncated_tool_call_hint(config.tools.edit_mode)
                    ),
                );
                messages.push(result_msg.clone());
                conversation_history.push(result_msg);
                app.push_output(
                    &format!(
                        "  ✗ {}: arguments cut off by the output limit after {} chars — not executed",
                        tc.function.name, info.original_chars
                    ),
                    LineStyle::ToolErr,
                );
                continue;
            }

            let args_summary = summarize_args(&tc.function.name, &args);

            // Detect tool call loops: identical calls repeated consecutively
            // (period-1), or the SAME two calls alternating (period-2 — the
            // edit↔revert oscillation the streak counter is blind to).
            let call_key = loop_call_key(&tc.function.name, &args);
            if last_call_key.as_ref() == Some(&call_key) {
                same_call_streak += 1;
            } else {
                last_call_key = Some(call_key.clone());
                same_call_streak = 1;
            }
            recent_call_keys.push(call_key.clone());
            if recent_call_keys.len() > 12 {
                recent_call_keys.remove(0);
            }
            let cycle = cycle_period(&recent_call_keys);
            if same_call_streak >= 3 || cycle.is_some() {
                // Cycle-only detection (not also a plain streak).
                let cycle_only = cycle.filter(|_| same_call_streak < 3);
                // A cycle is harmful if ANY member mutates.
                let mutating = if let Some(period) = cycle_only {
                    let tail = &recent_call_keys[recent_call_keys.len().saturating_sub(period)..];
                    tail.iter().any(|k| key_is_mutating(k))
                } else {
                    is_mutating_call(&tc.function.name, &args)
                };
                log.loop_detected(&tc.function.name, &args_summary, same_call_streak as usize);

                // Read-only repetition: harmless per call, just wasted tokens.
                // First detection: polite nudge, let processing continue.
                // Re-detection: escalate — the nudge can't reach a
                // cache-numerics rut, so force a compaction next round.
                if !mutating {
                    read_nudges += 1;
                    let escalate = read_nudges >= 2;
                    let text = if escalate {
                        read_nudges = 0;
                        force_compact_next_round = true;
                        REPEATED_READ_ESCALATION
                    } else {
                        REPEATED_READ_NUDGE
                    };
                    let result_msg = Message::tool_result(&tc.id, text);
                    messages.push(result_msg.clone());
                    conversation_history.push(result_msg);
                    app.push_output(
                        &format!(
                            "  ⓘ Repeated read: {}({}) — {}, continuing",
                            tc.function.name,
                            args_summary,
                            if escalate {
                                "nudge failed, forcing compaction next round"
                            } else {
                                "nudge sent"
                            }
                        ),
                        LineStyle::Status,
                    );
                    last_call_key = None;
                    same_call_streak = 0;
                    recent_call_keys.clear();
                    continue;
                }

                let hint = if let Some(period) = cycle_only {
                    cycle_loop_hint(period)
                } else {
                    loop_detected_hint(config.tools.edit_mode).to_string()
                };
                let result_msg = Message::tool_result(&tc.id, &hint);
                messages.push(result_msg.clone());
                conversation_history.push(result_msg);

                // First mutating loop in this turn: surface the hint, reset
                // the streak, and let the model try a different approach.
                if loop_recoveries == 0 {
                    loop_recoveries += 1;
                    last_call_key = None;
                    same_call_streak = 0;
                    recent_call_keys.clear();
                    app.push_output(
                        &format!(
                            "  ⚠ Loop detected: {}({}) {} — surfacing a hint, giving the model one more round",
                            tc.function.name,
                            args_summary,
                            if let Some(period) = cycle_only {
                                format!("cycling through the same {period} calls (period-{period} cycle)")
                            } else {
                                "repeated 3 times".to_string()
                            }
                        ),
                        LineStyle::Status,
                    );
                    break;
                }

                // Second mutating loop after the recovery hint. With a
                // behavioral done-gate configured this is NOT a dead end — it
                // is the same "stuck but the task isn't done" state as a
                // premature exit, so route it through the gate ladder instead
                // of dying with the whole recovery stack idle.
                if !read_only
                    && config.validation.command().is_some()
                    && validation_blocks < config.validation.max_retries
                {
                    app.push_output(
                        &format!(
                            "  Loop detected again ({}({})) — routing through the done-gate instead of stopping",
                            tc.function.name, args_summary
                        ),
                        LineStyle::Status,
                    );
                    if let validation::CheckOutcome::Fail(output) =
                        validation::run_behavioral_check(config).await
                    {
                        validation_blocks += 1;
                        // Fresh recovery budget for the rounds the gate grants.
                        loop_recoveries = 0;
                        last_call_key = None;
                        same_call_streak = 0;
                        recent_call_keys.clear();

                        let fkey = crate::cli::commands::run::failure_key(&output);
                        let may_fire = if config.tools.debugger_multifire {
                            debugger_fires < debugger::MAX_DEBUGGER_FIRES
                                && last_debugged_failure.as_deref() != Some(fkey.as_str())
                        } else {
                            debugger_fires == 0
                        };
                        if (config.tools.reactive_debugger || config.tools.debugger_judge)
                            && may_fire
                            && validation_blocks >= debugger::DEBUGGER_TRIGGER_BLOCKS
                        {
                            debugger_fires += 1;
                            last_debugged_failure = Some(fkey);
                            app.push_output(
                                "Looping + failing gate — spinning up a fresh-context debugger sub-agent…",
                                LineStyle::Status,
                            );
                            let verdict = debugger::run_debugger(
                                &output,
                                goal,
                                config,
                                llm_worker,
                                tool_pool,
                                tool_defs,
                                perms,
                                mcp_registry,
                                lsp,
                                fast_revisions,
                                fast_baseline_errors,
                                cancelled,
                            )
                            .await;

                            let msg = match verdict {
                                debugger::DebuggerVerdict::Scrap if !restart_fired => {
                                    restart_fired = true;
                                    *messages = scrap_restart(
                                        app,
                                        config,
                                        goal,
                                        mcp_summary,
                                        snapshots,
                                        true,
                                    );
                                    conversation_history.clear();
                                    validation_blocks = 0;
                                    same_plan_step_failures = 0;
                                    last_failed_plan_step = None;
                                    continue 'round;
                                }
                                debugger::DebuggerVerdict::Scrap => Message::user(
                                    "[A fresh-context review voted to reset again, but the tree \
                                     was already reset once this turn. Keep going: read the \
                                     current failure carefully and fix it directly.]",
                                ),
                                debugger::DebuggerVerdict::Rewind(candidate) => {
                                    rewind_message_repl(
                                        app,
                                        &candidate,
                                        config,
                                        perms,
                                        lsp,
                                        fast_revisions,
                                        fast_baseline_errors,
                                        &output,
                                    )
                                    .await
                                }
                                debugger::DebuggerVerdict::Report(body) => {
                                    let output_note =
                                        crate::cli::commands::run::write_gate_failure_output(
                                            config, &output,
                                        )
                                        .map(|path| {
                                            format!("\nFull raw check output: read(\"{path}\").")
                                        })
                                        .unwrap_or_default();
                                    Message::user(&format!(
                                        "[A read-only debugger with fresh eyes investigated the failing \
                                     check and produced this DIAGNOSIS. It did not edit anything — \
                                     YOU must apply the fix and finish the plan it lays out:\n{body}\n\
                                     Make the change(s), then finish; the verification will re-run.{output_note}]"
                                    ))
                                }
                            };
                            messages.push(msg.clone());
                            conversation_history.push(msg);
                            continue 'round;
                        }

                        let msg = Message::user(&format!(
                            "[Your repeated failing tool call was aborted, and the task is NOT \
                             done — the verification check failed:\n{output}\nRead the check \
                             output and your tool errors carefully, fix the SPECIFIC problem, \
                             and use a correctly-formed call (include every required parameter) \
                             or a different tool.]"
                        ));
                        messages.push(msg.clone());
                        conversation_history.push(msg);
                        continue 'round;
                    }
                    // Gate passed (or skipped): the loop was on something the
                    // check doesn't care about — fall through to the stop.
                }
                app.push_output(
                    &format!(
                        "  ✗ Loop detected again ({}({})) after the recovery hint — stopping this turn",
                        tc.function.name, args_summary
                    ),
                    LineStyle::Error,
                );
                had_error = true;
                break;
            }

            log.tool_call_detail(&tc.function.name, &args);
            app.push_output(
                &format!("  → {}({})", tc.function.name, args_summary),
                LineStyle::ToolCall,
            );

            // Re-render to show tool call
            let _ = terminal.draw(|frame| ui::draw(frame, app));

            // Read-only investigation (explore) mode: hard-block any mutating
            // tool call at runtime, BEFORE any permission prompt. The def filter
            // and the prompt are advisory — shell can still mutate and the model
            // can emit tools that aren't in the list. Read-only shell is allowed.
            if read_only {
                let file_action = args["action"].as_str().unwrap_or("");
                if let Some(reason) = explore_block_reason(&tc.function.name, file_action, &args) {
                    let msg = Message::tool_result(
                        &tc.id,
                        &format!(
                            "[blocked: read-only investigation mode] {reason}. To change code or \
                             run mutating shell commands, switch to coding (e.g. say \"actually, \
                             change it\")."
                        ),
                    );
                    messages.push(msg.clone());
                    conversation_history.push(msg);
                    app.push_output(
                        &format!("  ⛔ {}: blocked — read-only mode", tc.function.name),
                        LineStyle::ToolErr,
                    );
                    continue;
                }
            }

            // Determine if this tool call needs a permission prompt
            let perm_action = permission_action(&tc.function.name, &args);

            // Check permission via TUI prompt (not raw stderr)
            let mut perm_denied = false;
            if let Some(ref action) = perm_action
                && matches!(action, Action::Shell(_) | Action::McpUse(_, _))
            {
                match perms.check_needs_prompt(action) {
                    Err(e) => {
                        // Blocklisted — skip this tool call
                        let result_msg = Message::tool_result(&tc.id, &e);
                        messages.push(result_msg.clone());
                        conversation_history.push(result_msg);
                        app.push_output(
                            &format!("  ✗ {}: {e}", tc.function.name),
                            LineStyle::ToolErr,
                        );
                        continue;
                    }
                    Ok(Some(prompt)) => {
                        // Needs user approval — show prompt in TUI
                        app.pending_permission = Some(prompt);
                        app.input.clear();
                        app.cursor = 0;
                        let _ = terminal.draw(|frame| ui::draw(frame, app));

                        // Wait for user input (y/n/a)
                        let response = wait_for_permission_input(app, rx, terminal).await;
                        app.pending_permission = None;

                        match response.as_str() {
                            "y" | "yes" => {
                                perms.approve(action, false);
                                app.push_output(
                                    "  · Permission granted, running tool...",
                                    LineStyle::Status,
                                );
                            }
                            "a" | "always" => {
                                perms.approve(action, true);
                                app.push_output(
                                    "  · Permission granted and saved, running tool...",
                                    LineStyle::Status,
                                );
                            }
                            _ => {
                                perm_denied = true;
                                app.push_output("  · Permission denied.", LineStyle::Status);
                            }
                        }

                        let _ = terminal.draw(|frame| ui::draw(frame, app));
                    }
                    Ok(None) => {} // No prompt needed
                }
            }

            if perm_denied {
                let result_msg =
                    Message::tool_result(&tc.id, &format!("{} denied by user", tc.function.name));
                messages.push(result_msg.clone());
                conversation_history.push(result_msg);
                app.push_output(
                    &format!("  ✗ {}: denied", tc.function.name),
                    LineStyle::ToolErr,
                );
                continue;
            }

            let file_action = args["action"].as_str().unwrap_or("");

            // Write gating: require a plan before write tools (strict only).
            let is_write_action = is_file_write(tc.function.name.as_str());
            if strict && config.tools.plan && !tools::plan::plan_exists(config) && is_write_action {
                let result_msg = Message::tool_result(
                    &tc.id,
                    "Create a plan first: use plan(action='set') with your step-by-step approach before making changes.",
                );
                messages.push(result_msg.clone());
                conversation_history.push(result_msg);
                app.push_output(
                    &format!("  ✗ {}: blocked — no plan", tc.function.name),
                    LineStyle::ToolErr,
                );
                continue;
            }
            // (Plan-checkpoint used to hard-block writes after N edits without
            //  a plan action; that interacted poorly with the compile-gate on
            //  `plan(check)` — if the project didn't compile, the model
            //  couldn't escape the block, couldn't fix the project, deadlock.
            //  Now we just warn at the threshold via PLAN_CHECKPOINT_WARNING
            //  appended to the tool result; the model decides what to do.)

            // Execute tool (permissions already checked above for shell/web/mcp)
            let mut result = if matches!(
                tc.function.name.as_str(),
                "replace_range" | "insert_at" | "revert" | "show_rev" | "check"
            ) && config.tools.edit_mode == EditMode::Fast
            {
                let tool_name = tc.function.name.clone();
                let args = args.clone();
                let config = config.clone();
                let perms = perms.clone();
                let lsp = lsp.clone();
                let revisions = fast_revisions.clone();
                let baseline = fast_baseline_errors;
                let mut result_rx = tool_pool.submit(move || {
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .map_err(|e| e.to_string())?;
                    let Some(revisions) = revisions else {
                        return Ok(crate::tools::ToolResult::err(
                            "fast mode: revision store unavailable".into(),
                        ));
                    };
                    runtime
                        .block_on(async move {
                            tools::execute_fast_tool(
                                &tool_name,
                                &args,
                                &config,
                                perms.as_ref(),
                                lsp.as_deref(),
                                revisions.as_ref(),
                                baseline,
                            )
                            .await
                        })
                        .map_err(|e| format!("fast tool error: {e}"))
                });
                await_tool_job_ui(
                    rx,
                    terminal,
                    app,
                    &tc.function.name,
                    &mut result_rx,
                    cancelled,
                )
                .await
            } else if tc.function.name == "mcp_use" {
                let server = args["server"].as_str().unwrap_or("").to_string();
                let tool = args["tool"].as_str().unwrap_or("").to_string();
                let tool_args = args.get("arguments").cloned().unwrap_or_default();
                if server.is_empty() || tool.is_empty() {
                    crate::tools::ToolResult::err(
                        "mcp_use requires top-level 'server' and 'tool' string fields. \
                         Example: {\"server\": \"my-server\", \"tool\": \"my-tool\", \"arguments\": {}}".into(),
                    )
                } else {
                    let registry = mcp_registry.clone();
                    let mut result_rx = tool_pool.submit(move || match registry {
                        Some(registry) => {
                            let mut guard = registry.lock();
                            guard
                                .call_tool(&server, &tool, tool_args)
                                .map(crate::tools::ToolResult::ok)
                                .map_err(|e| format!("MCP error: {e}"))
                        }
                        None => Ok(crate::tools::ToolResult::err(
                            "No MCP servers connected".into(),
                        )),
                    });
                    await_tool_job_ui(rx, terminal, app, "mcp_use", &mut result_rx, cancelled).await
                }
            } else if tc.function.name == "spawn_agents" {
                let tasks = crate::cli::commands::agent::subagent::parse_tasks(&args);
                if tasks.is_empty() {
                    crate::tools::ToolResult::err(
                        "spawn_agents: 'agents' must be a non-empty array of {label, prompt}"
                            .into(),
                    )
                } else {
                    let (out_tx, mut out_rx) =
                        tokio::sync::mpsc::unbounded_channel::<(String, LineStyle)>();
                    let subagents_fut = crate::cli::commands::agent::subagent::run_subagents(
                        tasks,
                        config,
                        llm_worker,
                        tool_pool,
                        tool_defs,
                        perms,
                        mcp_registry,
                        lsp,
                        fast_revisions,
                        fast_baseline_errors,
                        cancelled,
                        Some(out_tx),
                    );
                    let mut subagents_fut = std::pin::pin!(subagents_fut);
                    let mut outputs = None;
                    while outputs.is_none() {
                        tokio::select! {
                            biased;
                            r = &mut subagents_fut, if outputs.is_none() => { outputs = Some(r); }
                            line = out_rx.recv() => {
                                if let Some((text, style)) = line {
                                    app.push_output(&text, style);
                                    let _ = terminal.draw(|frame| ui::draw(frame, app));
                                }
                            }
                            evt = rx.recv() => {
                                if matches!(evt, Some(AppEvent::Tick)) {
                                    let _ = terminal.draw(|frame| ui::draw(frame, app));
                                }
                            }
                        }
                    }
                    let combined =
                        crate::cli::commands::agent::subagent::format_outputs(outputs.unwrap());
                    crate::tools::ToolResult::ok(combined)
                }
            } else if tc.function.name == "edit_file" {
                let args = args.clone();
                let config = config.clone();
                let perms = perms.clone();
                let router = router.clone();
                let lsp = lsp.clone();
                let cancelled_for_job = cancelled.clone();
                let log_for_job = log.clone();
                let mut result_rx = tool_pool.submit(move || {
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .map_err(|e| e.to_string())?;
                    runtime
                        .block_on(async move {
                            crate::tools::execute_edit_file_tool(
                                &args,
                                &config,
                                perms.as_ref(),
                                router.as_ref(),
                                lsp.as_deref(),
                                Some(cancelled_for_job.as_ref()),
                                Some(log_for_job.as_ref()),
                            )
                            .await
                        })
                        .map_err(|e| format!("edit_file error: {e}"))
                });
                await_tool_job_ui(rx, terminal, app, "edit_file", &mut result_rx, cancelled).await
            } else if tc.function.name == "plan" {
                let args = args.clone();
                let config_for_job = config.clone();
                let mut result_rx = tool_pool.submit(move || {
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .map_err(|e| e.to_string())?;
                    runtime
                        .block_on(async move {
                            tools::plan::execute(&args, &config_for_job, round).await
                        })
                        .map_err(|e| format!("plan error: {e}"))
                });
                let r =
                    await_tool_job_ui(rx, terminal, app, "plan", &mut result_rx, cancelled).await;
                // The plan tool just mutated plan.md mid-round — refresh the
                // panel and redraw now so a checked/added/refined step appears
                // immediately instead of lagging to the next round's refresh.
                refresh_plan_panel(app, config, round);
                let _ = terminal.draw(|frame| ui::draw(frame, app));
                r
            } else if tc.function.name == "refactor" {
                let args = args.clone();
                let config = config.clone();
                let router = router.clone();
                let lsp = lsp.clone();
                let log_for_job = log.clone();
                let revisions_for_job = fast_revisions.clone();
                let cancelled_for_job = cancelled.clone();
                let mut result_rx = tool_pool.submit(move || {
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .map_err(|e| e.to_string())?;
                    runtime
                        .block_on(async move {
                            crate::tools::execute_refactor_tool(
                                &args,
                                &config,
                                router.as_ref(),
                                lsp.as_deref(),
                                Some(log_for_job.as_ref()),
                                revisions_for_job.as_deref(),
                                Some(cancelled_for_job.as_ref()),
                            )
                            .await
                        })
                        .map_err(|e| format!("refactor error: {e}"))
                });
                await_tool_job_ui(rx, terminal, app, "refactor", &mut result_rx, cancelled).await
            } else if (tc.function.name == "shell" && args["action"].as_str() == Some("run"))
                || (tc.function.name == "file" && file_action == "shell")
            {
                if args["background"].as_bool() == Some(true) {
                    // Explicit background start (cheap, non-blocking) —
                    // registered in the session job registry, managed via
                    // the jobs tool in this or any later turn.
                    tools::jobs::start_background(&args, config, job_registry.as_ref())
                } else {
                    await_shell_job_repl(
                        tool_pool.submit_shell(args.clone(), config.clone(), cancelled.clone()),
                        app,
                        rx,
                        terminal,
                        cancelled,
                    )
                    .await
                }
            } else if tc.function.name == "shell" {
                // Runs on the pool (own runtime) so jobs(wait) keeps the TUI
                // responsive via await_tool_job_ui, like other pooled tools.
                let args_for_job = args.clone();
                let config_for_job = config.clone();
                let perms_for_job = perms.clone();
                let registry_for_job = job_registry.clone();
                let cancelled_for_job = cancelled.clone();
                let mut result_rx = tool_pool.submit(move || {
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .map_err(|e| e.to_string())?;
                    Ok(runtime.block_on(async {
                        tools::jobs::execute(
                            &args_for_job,
                            &config_for_job,
                            perms_for_job.as_ref(),
                            registry_for_job.as_ref(),
                            Some(cancelled_for_job.as_ref()),
                        )
                        .await
                    }))
                });
                await_tool_job_ui(rx, terminal, app, "jobs", &mut result_rx, cancelled).await
            } else {
                let tool_name = tc.function.name.clone();
                let args = args.clone();
                let config = config.clone();
                let perms = perms.clone();
                let lsp = lsp.clone();
                let mut result_rx = tool_pool.submit(move || {
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .map_err(|e| e.to_string())?;
                    runtime
                        .block_on(async move {
                            tools::execute_tool(
                                &tool_name,
                                &args,
                                &config,
                                perms.as_ref(),
                                lsp.as_deref(),
                            )
                            .await
                        })
                        .map_err(|e| format!("Tool error: {e}"))
                });
                await_tool_job_ui(
                    rx,
                    terminal,
                    app,
                    &tc.function.name,
                    &mut result_rx,
                    cancelled,
                )
                .await
            };

            if !result.success
                && let Some(hint) = tools::plan::failure_hint(config)
            {
                result.content.push('\n');
                result.content.push_str(&hint);
            }

            // Append round number to every tool result.
            result
                .content
                .push_str(&format!("\n[round {round}/{max_rounds}]"));

            let first_line = result.content.lines().next().unwrap_or("(empty)");
            log.tool_call(&tc.function.name, &args_summary, result.success, first_line);
            log.tool_result_detail(&tc.function.name, result.success, &result.content);
            let style = if result.success {
                LineStyle::ToolOk
            } else {
                LineStyle::ToolErr
            };
            let icon = if result.success { "✓" } else { "✗" };
            app.push_output(
                &format!("  {icon} {}: {first_line}", tc.function.name),
                style,
            );
            app.store_tool_result(&tc.function.name, &result.content);

            if result.success && tc.function.name == "plan" {
                successful_edits_since_plan_update = 0;
            }

            // Successful file write = code changed, reset loop/stall trackers.
            if result.success && is_file_write(tc.function.name.as_str()) {
                last_call_key = None;
                same_call_streak = 0;
                calls_since_last_edit = 0;
                if strict && config.tools.plan {
                    if tools::plan::plan_exists(config) {
                        result.content.push('\n');
                        result.content.push_str(PLAN_PROGRESS_NUDGE);
                    }
                    successful_edits_since_plan_update += 1;
                    if successful_edits_since_plan_update == PLAN_CHECKPOINT_AFTER_EDITS {
                        result.content.push('\n');
                        result.content.push_str(PLAN_CHECKPOINT_WARNING);
                    }
                }
            } else {
                calls_since_last_edit += 1;
            }

            if !is_prunable_refactor_failure(&result.content, result.success) {
                all_prunable_failures = false;
            } else {
                prunable_errors.push(result.content.clone());
            }

            let result_msg = Message::tool_result(&tc.id, &result.content);
            messages.push(result_msg.clone());
            conversation_history.push(result_msg);

            // `plan_gate_debugger`: the plan tool's OWN compile gate repeatedly
            // blocking the SAME step is a distinct stall signature from the
            // behavioral done-gate (`validation_blocks`) — the primary agent is
            // re-litigating one step rather than making forward progress.
            if tc.function.name == "plan"
                && args.get("action").and_then(|a| a.as_str()) == Some("check")
            {
                if result.success {
                    same_plan_step_failures = 0;
                    last_failed_plan_step = None;
                } else if let Some(step) = args.get("step").and_then(|s| s.as_u64()) {
                    crate::cli::commands::run::track_plan_step_failure(
                        &mut last_failed_plan_step,
                        &mut same_plan_step_failures,
                        step,
                    );

                    let fkey = crate::cli::commands::run::failure_key(&result.content);
                    let may_fire = if config.tools.debugger_multifire {
                        debugger_fires < debugger::MAX_DEBUGGER_FIRES
                            && last_debugged_failure.as_deref() != Some(fkey.as_str())
                    } else {
                        debugger_fires == 0
                    };
                    if config.tools.plan_gate_debugger
                        && may_fire
                        && same_plan_step_failures as usize >= debugger::DEBUGGER_TRIGGER_BLOCKS
                    {
                        debugger_fires += 1;
                        last_debugged_failure = Some(fkey);
                        app.push_output(
                            "Plan-check gate failing repeatedly on the same step — spinning up a fresh-context debugger sub-agent…",
                            LineStyle::Status,
                        );
                        let verdict = debugger::run_debugger(
                            &result.content,
                            goal,
                            config,
                            llm_worker,
                            tool_pool,
                            tool_defs,
                            perms,
                            mcp_registry,
                            lsp,
                            fast_revisions,
                            fast_baseline_errors,
                            cancelled,
                        )
                        .await;

                        let extra_msg = match verdict {
                            debugger::DebuggerVerdict::Scrap if !restart_fired => {
                                restart_fired = true;
                                *messages =
                                    scrap_restart(app, config, goal, mcp_summary, snapshots, true);
                                conversation_history.clear();
                                validation_blocks = 0;
                                same_plan_step_failures = 0;
                                last_failed_plan_step = None;
                                continue 'round;
                            }
                            debugger::DebuggerVerdict::Scrap => Message::user(
                                "[A fresh-context review voted to reset again, but the tree \
                                 was already reset once this turn. Keep going: read the \
                                 current failure carefully and fix it directly.]",
                            ),
                            debugger::DebuggerVerdict::Rewind(candidate) => {
                                rewind_message_repl(
                                    app,
                                    &candidate,
                                    config,
                                    perms,
                                    lsp,
                                    fast_revisions,
                                    fast_baseline_errors,
                                    &result.content,
                                )
                                .await
                            }
                            debugger::DebuggerVerdict::Report(body) => {
                                let output_note =
                                    crate::cli::commands::run::write_gate_failure_output(
                                        config,
                                        &result.content,
                                    )
                                    .map(|path| {
                                        format!("\nFull raw check output: read(\"{path}\").")
                                    })
                                    .unwrap_or_default();
                                Message::user(&format!(
                                    "[A read-only debugger with fresh eyes investigated the failing \
                                 plan-check step and produced this DIAGNOSIS. It did not edit \
                                 anything — YOU must apply the fix and finish the step it lays \
                                 out:\n{body}\nMake the change(s), then re-check the step.{output_note}]"
                                ))
                            }
                        };
                        messages.push(extra_msg.clone());
                        conversation_history.push(extra_msg);
                    }
                }
            }

            // Spiral-reset: a revert-loop (same file reverted repeatedly) means
            // the agent is cycling on the same failing edits. Inject a cognitive
            // reset (names what failed + forces a replan + concrete redirection).
            if config.tools.spiral_reset
                && result.success
                && tc.function.name == "revert"
                && config.tools.edit_mode == EditMode::Fast
                && spiral_resets < spiral::MAX_RESETS_PER_TURN
                && let Some(path) = args.get("path").and_then(|p| p.as_str())
            {
                let count = revert_counts.entry(path.to_string()).or_insert(0);
                *count += 1;
                if *count >= spiral::SPIRAL_REVERT_THRESHOLD {
                    let n = *count;
                    *count = 0;
                    spiral_resets += 1;
                    let tried = fast_revisions
                        .as_deref()
                        .map(|r| spiral::tried_edit_labels(r, path, 4))
                        .unwrap_or_default();
                    let reset = Message::user(&spiral::build_reset_message(path, n, &tried));
                    messages.push(reset.clone());
                    conversation_history.push(reset);
                    app.push_output(
                        "Spiral detected (revert-loop) — reset + replan injected.",
                        LineStyle::Status,
                    );
                    log.tool_debug(
                        "agent",
                        &format!("spiral-reset fired for {path} after {n} reverts"),
                    );
                }
            }

            // Re-render after tool result
            let _ = terminal.draw(|frame| ui::draw(frame, app));
        }

        // History pruning — see run.rs for rationale.
        if all_prunable_failures && !prunable_errors.is_empty() {
            messages.truncate(messages_pre);
            conversation_history.truncate(history_pre);
            let hint = Message::user(&format!(
                "Your previous refactor call(s) were rejected:\n\n{}\n\n\
                 Retry with all required parameters and a clean position value \
                 (one of 'start' or 'after:<single_param_name>').",
                prunable_errors.join("\n\n---\n\n")
            ));
            messages.push(hint.clone());
            conversation_history.push(hint);
            log.tool_debug(
                "agent",
                &format!(
                    "history pruned: dropped {} tool_result(s) after refactor validator failure",
                    prunable_errors.len()
                ),
            );
        }

        // Early no-plan nudge (strict only): edit tools are hidden until
        // plan(action='set'). Nudge around round 12 so a model that ignores the
        // system prompt gets a course correction before it's deeply stuck.
        if strict && round >= 12 && !nudged_no_plan && !tools::plan::plan_exists(config) {
            let unlock_tools = "refactor, replace_range, insert_at, write_file";
            messages.push(Message::user(&format!(
                "[Reminder: you've explored for several rounds without a plan. \
                 Call plan(action='set') with your step-by-step approach now — \
                 the edit tools ({unlock_tools}) are hidden until you do, and \
                 you'll need them to make changes.]"
            )));
            nudged_no_plan = true;
        }

        // Stall detection: too many tool calls without any edits. Content is
        // plan-state aware — without a plan the edit tools are hidden, so
        // re-fire the plan nudge instead of pointing at hidden tools.
        if calls_since_last_edit >= 20 && calls_since_last_edit.is_multiple_of(20) {
            let body = if strict && !tools::plan::plan_exists(config) {
                "Still no plan set after 20+ exploration calls. \
                 Edit tools cannot appear in your tool list until plan(action='set') is called. \
                 Stop exploring and set a plan now — even an imperfect plan can be refined later. \
                 If something is blocking you from planning, say so."
                    .to_string()
            } else {
                let edit_hint = match config.tools.edit_mode {
                    EditMode::Smart => "Use edit_file for semantic file edits.",
                    EditMode::Fast => "Use replace_range or insert_at to land targeted edits.",
                };
                format!(
                    "You have used 20+ tool calls without making any edits. \
                     You likely have enough information. Start making changes now. \
                     {edit_hint} \
                     If you're stuck, explain what's blocking you."
                )
            };
            messages.push(Message::user(&format!("[WARNING: {body}]")));
        }
    }

    log.session_end(round, had_error);
}

/// Execute the debugger's proposed single-file rewind (`debugger_judge_rewind`)
/// and build the message to inject afterward. Best-effort: on failure the tree
/// is left as-is and the model just sees the original verification failure.
#[allow(clippy::too_many_arguments)]
async fn rewind_message_repl(
    app: &mut App,
    candidate: &tools::RewindCandidate,
    config: &Config,
    perms: &Arc<PermissionManager>,
    lsp: &Option<Arc<LspClient>>,
    fast_revisions: &Option<Arc<tools::RevisionStore>>,
    fast_baseline_errors: usize,
    output: &str,
) -> Message {
    let Some(revisions) = fast_revisions.as_deref() else {
        return Message::user(&format!(
            "[Verification failed — do NOT finish yet. Check output:\n{output}]"
        ));
    };
    let args = serde_json::json!({"path": candidate.path, "rev": candidate.rev});
    let ok = tools::execute_fast_tool(
        "revert",
        &args,
        config,
        perms.as_ref(),
        lsp.as_deref(),
        revisions,
        fast_baseline_errors,
    )
    .await
    .is_ok_and(|r| r.success);

    if ok {
        app.push_output(
            &format!(
                "[debugger-judge] REWIND — reverted {} to rev_{} (file_errors {} → {})",
                candidate.path,
                candidate.rev,
                candidate.file_errors_now,
                candidate.file_errors_then
            ),
            LineStyle::Status,
        );
        // REWIND fixes the ONE regressed file the debugger flagged — it doesn't
        // mean the gate's original failure is fully resolved. Point at the raw
        // check output so the model isn't left guessing what "the remaining
        // problem" actually is from the rewind summary alone.
        let output_note = crate::cli::commands::run::write_gate_failure_output(config, output)
            .map(|path| format!(" Full check output that triggered this: read(\"{path}\")."))
            .unwrap_or_default();
        Message::user(&format!(
            "[A read-only debugger with fresh eyes found that {} had regressed from a much \
             cleaner earlier revision. The loop has ALREADY reverted it to rev_{} for you \
             (file_errors {} → {}) — do NOT redo the discarded edits the same way. Re-read the \
             file to see its current (reverted) content, then continue the plan, fixing the \
             remaining problem differently. Everything outside this file is untouched.{output_note}]",
            candidate.path, candidate.rev, candidate.file_errors_now, candidate.file_errors_then
        ))
    } else {
        app.push_output(
            &format!(
                "[debugger-judge] REWIND — revert of {} to rev_{} failed; continuing without it",
                candidate.path, candidate.rev
            ),
            LineStyle::Status,
        );
        Message::user(&format!(
            "[Verification failed — do NOT finish yet. Check output:\n{output}]"
        ))
    }
}

/// Whole-tree SCRAP restart: revert the working tree to the clean round-0
/// baseline, resync the symbol index, clear plan/scratchpad, and return a
/// freshly-assembled context. `judge` selects the debugger-judge vs
/// gate-restart status wording. The caller resets the loop counters and
/// `continue`s. Mirrors run.rs's SCRAP/gate-restart blocks.
fn scrap_restart(
    app: &mut App,
    config: &Config,
    goal: &str,
    mcp_summary: Option<&str>,
    snapshots: &Option<Arc<Mutex<tools::snapshots::SnapshotManager>>>,
    judge: bool,
) -> Vec<Message> {
    let (ok_prefix, err_prefix, done) = if judge {
        (
            "[debugger-judge] SCRAP — ",
            "[debugger-judge] SCRAP — tree revert failed: ",
            "[debugger-judge] scrapped the stuck state — clean baseline + fresh context; restarting from scratch.",
        )
    } else {
        (
            "[gate-restart] ",
            "[gate-restart] tree revert failed: ",
            "[gate-restart] scrapped the stuck state — tree at clean baseline + fresh context; restarting from scratch.",
        )
    };
    if let Some(snap) = snapshots {
        let guard = snap.lock();
        match guard.revert_to_round(0) {
            Ok(m) => app.push_output(&format!("{ok_prefix}{m}"), LineStyle::Status),
            Err(e) => app.push_output(&format!("{err_prefix}{e}"), LineStyle::Status),
        }
    }
    // Whole-tree revert touched many files outside the per-edit reindex path —
    // resync the symbol index / repo-map to the clean baseline.
    tools::reindex_project_incremental(config);
    let _ = std::fs::remove_file(config.session_path("plan.md"));
    let _ = std::fs::remove_file(config.session_path("scratchpad.md"));
    let assembled = context::assemble(config, goal, &[], false, mcp_summary);
    app.push_output(done, LineStyle::Status);
    assembled.messages
}

fn handle_background_key(app: &mut App, key: &crossterm::event::KeyEvent) -> bool {
    match key.code {
        KeyCode::PageUp => {
            app.scroll_up(10);
            true
        }
        KeyCode::PageDown => {
            app.scroll_down(10);
            true
        }
        KeyCode::Up if app.input.is_empty() => {
            app.scroll_up(1);
            true
        }
        KeyCode::Down if app.input.is_empty() => {
            app.scroll_down(1);
            true
        }
        KeyCode::Home if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.scroll_offset = app.output.len().saturating_sub(1) as u16;
            true
        }
        KeyCode::End if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.scroll_offset = 0;
            true
        }
        _ => false,
    }
}

fn consume_interrupt(cancelled: &AtomicBool) -> bool {
    cancelled.swap(false, Ordering::Relaxed)
}

/// Drop any input-kind events queued in `rx` without processing them.
///
/// Called at the end of the Enter handler to discard keystrokes the user
/// typed while the "working" indicator was up (LLM streaming + post-turn
/// compression). Those keys were meant to be ignored per `is_thinking`, but
/// because the main loop is `await`-ing the agent/compressor, keys queue in
/// the channel and would otherwise be replayed against the now-idle input
/// box — producing a visible desync where the user's next prompt appears to
/// fire "on its own".
///
/// We preserve non-key events (permission requests, status updates). Ctrl+C
/// was already handled inline by the key reader via the `cancelled` flag.
fn drain_stale_key_events(rx: &mut mpsc::UnboundedReceiver<AppEvent>) {
    while let Ok(evt) = rx.try_recv() {
        match evt {
            AppEvent::Key(_) | AppEvent::Mouse(_) | AppEvent::Tick => {}
            other => {
                // Put non-input events back via a small re-enqueue: we only
                // have a receiver here, so the cleanest option is to drop
                // them too. In practice, permission requests and status
                // messages are only emitted while an agent task is running,
                // which it isn't by the time we get here.
                let _ = other;
            }
        }
    }
}

fn reconcile_streamed_assistant_content(rendered: &str, final_content: &str) -> Option<String> {
    if final_content.is_empty() || rendered == final_content {
        return None;
    }
    if let Some(suffix) = final_content.strip_prefix(rendered) {
        return (!suffix.is_empty()).then(|| suffix.to_string());
    }
    if rendered.is_empty() {
        return Some(final_content.to_string());
    }
    Some(format!(
        "\n[final response continuation]\n{}",
        final_content
    ))
}

fn finish_completed_turn(
    app: &mut App,
    terminal: &mut Terminal<impl Backend>,
    final_content: Option<&str>,
    rendered_assistant_text: Option<&str>,
) -> io::Result<()> {
    app.is_thinking = false;
    if let (Some(final_content), Some(rendered)) = (final_content, rendered_assistant_text)
        && let Some(missing) = reconcile_streamed_assistant_content(rendered, final_content)
    {
        app.push_token(&missing);
    }
    app.flush_tokens();
    app.push_output(
        "────────────────────────────────────────────────",
        LineStyle::Separator,
    );
    terminal
        .draw(|frame| ui::draw(frame, app))
        .map_err(|e| io::Error::other(e.to_string()))?;
    Ok(())
}

async fn await_tool_job_ui(
    rx: &mut mpsc::UnboundedReceiver<AppEvent>,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
    job_label: &str,
    result_rx: &mut tokio::sync::oneshot::Receiver<Result<crate::tools::ToolResult, String>>,
    cancelled: &Arc<AtomicBool>,
) -> crate::tools::ToolResult {
    app.set_active_job(job_label);
    loop {
        tokio::select! {
            result = &mut *result_rx => {
                app.clear_active_job();
                return match result {
                    Ok(Ok(tool_result)) => tool_result,
                    Ok(Err(err)) => crate::tools::ToolResult::err(err),
                    Err(_) => crate::tools::ToolResult::err("Tool worker dropped job".into()),
                };
            }
            evt = rx.recv() => {
                match evt {
                    Some(AppEvent::Tick) => {
                        let _ = terminal.draw(|frame| ui::draw(frame, app));
                    }
                    Some(AppEvent::Key(key)) if handle_background_key(app, &key) => {
                        let _ = terminal.draw(|frame| ui::draw(frame, app));
                    }
                    Some(AppEvent::Key(key)) if event::is_ctrl_c(&key) => {
                        cancelled.store(true, Ordering::Relaxed);
                        app.push_output("(interrupted)", LineStyle::Status);
                        let _ = terminal.draw(|frame| ui::draw(frame, app));
                    }
                    Some(AppEvent::Mouse(_)) => {}
                    Some(AppEvent::PermissionRequest(prompt, response_tx)) => {
                        let response = fulfill_permission_request(app, rx, terminal, prompt).await;
                        let _ = response_tx.send(response);
                    }
                    Some(_) => {}
                    None => {
                        app.clear_active_job();
                        return crate::tools::ToolResult::err("Event stream closed.".into())
                    },
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyEvent, KeyEventKind, KeyEventState};
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use serde_json::json;
    use tokio::sync::mpsc;

    #[test]
    fn shell_read_only_allows_reads_blocks_writes() {
        for ok in [
            "ls -R",
            "cat src/main.rs",
            "grep -rn foo .",
            "rg pattern",
            "find . -name '*.rs'",
            "git status",
            "git log --oneline",
            "cd app && ls",
            "wc -l file",
            "grep x 2>/dev/null",
            "git diff | grep foo",
        ] {
            assert!(shell_is_read_only(ok), "should ALLOW read-only: {ok}");
        }
        for bad in [
            "echo hi > foo.rs",
            "rm -rf x",
            "mv a b",
            "cargo build",
            "cargo check",
            "git commit -m x",
            "git add .",
            "sed -i 's/a/b/' f",
            "touch new",
            "mkdir d",
            "find . -delete",
            "find . -exec rm {} \\;",
            "ls && rm x",
            "echo $(rm x)",
            "bash -c 'rm x'",
            "python -c 'import os'",
            "x > y",
        ] {
            assert!(!shell_is_read_only(bad), "should BLOCK mutating: {bad}");
        }
    }

    #[test]
    fn explore_block_reason_blocks_mutations_allows_reads() {
        let empty = json!({});
        // edit/delete/spawn tools → blocked
        for t in [
            "write_file",
            "replace_range",
            "refactor",
            "revert",
            "delete_file",
            "spawn_agents",
        ] {
            assert!(explore_block_reason(t, "", &empty).is_some(), "block {t}");
        }
        // read/intel tools → allowed
        for t in ["read_symbol", "search", "code"] {
            assert!(
                explore_block_reason(t, "repo_map", &empty).is_none(),
                "allow {t}"
            );
        }
        // file: read ok, read-only shell ok, mutating shell blocked
        assert!(explore_block_reason("file", "read", &json!({"path": "x"})).is_none());
        assert!(
            explore_block_reason("file", "shell", &json!({"command": "ls -R"})).is_none(),
            "read-only shell allowed"
        );
        assert!(
            explore_block_reason("file", "shell", &json!({"command": "echo x > f"})).is_some(),
            "mutating shell blocked"
        );
    }

    #[test]
    fn classifier_parse_is_fail_safe_to_coding() {
        // EXPLORE only on a clean leading EXPLORE.
        assert!(is_explore_reply("EXPLORE"));
        assert!(is_explore_reply("  explore \n"));
        assert!(is_explore_reply("EXPLORE."));
        assert!(is_explore_reply("EXPLORE - read only"));
        // Everything else → CODING (fail-safe), incl. prose-wrapped,
        // empty, the other label, garbage.
        assert!(!is_explore_reply("CODING"));
        assert!(!is_explore_reply(""));
        assert!(!is_explore_reply("I think this is EXPLORE"));
        assert!(!is_explore_reply("probably explore?"));
        assert!(!is_explore_reply("</think> blah"));
    }

    #[test]
    fn read_only_filter_drops_writers_and_plan_keeps_readers() {
        use crate::llm::{FunctionDefinition, ToolDefinition};
        let td = |n: &str| ToolDefinition {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: n.into(),
                description: String::new(),
                parameters: json!({}),
            },
        };
        let all = vec![
            td("file"),
            td("code"),
            td("web"),
            td("show_rev"),
            td("check"),
            td("write_file"),
            td("edit_file"),
            td("refactor"),
            td("add_function_param"),
            td("replace_range"),
            td("insert_at"),
            td("revert"),
            td("delete_file"),
            td("plan"),
            td("spawn_agents"),
        ];
        let ro: Vec<String> = read_only_tool_defs(&all)
            .iter()
            .map(|t| t.function.name.clone())
            .collect();
        assert_eq!(ro, ["file", "code", "web", "show_rev", "check"]);
    }

    #[test]
    fn file_search_summary_uses_pattern_and_path() {
        let args = json!({
            "action": "search",
            "path": "src/context/mod.rs",
            "pattern": "pub fn assemble",
        });

        assert_eq!(
            summarize_args("file", &args),
            "search \"pub fn assemble\" in src/context/mod.rs"
        );
    }

    #[test]
    fn plan_refine_summary_includes_step() {
        let args = json!({
            "action": "refine",
            "step": 2,
        });

        assert_eq!(summarize_args("plan", &args), "refine step 2");
    }

    #[test]
    fn web_search_summary_includes_query() {
        let args = json!({
            "action": "search",
            "query": "Michał Szynkiewicz",
        });

        assert_eq!(
            summarize_args("web", &args),
            "search \"Michał Szynkiewicz\""
        );
    }

    #[test]
    fn grouped_file_shell_maps_to_shell_permission_action() {
        let args = json!({
            "action": "shell",
            "command": "python -m http.server",
        });

        match permission_action("file", &args) {
            Some(Action::Shell(cmd)) => assert_eq!(cmd, "python -m http.server"),
            _ => panic!("expected grouped file shell to require shell permission"),
        }
    }

    #[test]
    fn loop_hint_smart_mentions_edit_file() {
        let hint = loop_detected_hint(EditMode::Smart);
        assert!(hint.contains("edit_file"));
    }

    #[test]
    fn loop_hint_fast_mentions_revision_table_tools() {
        let hint = loop_detected_hint(EditMode::Fast);
        assert!(hint.contains("show_rev"));
        assert!(hint.contains("revert"));
        // Fast mode now exposes edit_file, so the loop hint suggests it
        // as a structural-rewrite escape hatch.
        assert!(hint.contains("edit_file"));
    }

    #[test]
    fn consume_interrupt_clears_flag_after_first_read() {
        let cancelled = AtomicBool::new(true);
        assert!(consume_interrupt(&cancelled));
        assert!(!consume_interrupt(&cancelled));
        assert!(!cancelled.load(Ordering::Relaxed));
    }

    #[test]
    fn reconcile_streamed_assistant_content_appends_missing_suffix() {
        assert_eq!(
            reconcile_streamed_assistant_content("Hello", "Hello world"),
            Some(" world".into())
        );
    }

    #[test]
    fn reconcile_streamed_assistant_content_returns_none_when_complete() {
        assert_eq!(
            reconcile_streamed_assistant_content("Hello world", "Hello world"),
            None
        );
    }

    #[test]
    fn reconcile_streamed_assistant_content_uses_full_content_when_nothing_rendered() {
        assert_eq!(
            reconcile_streamed_assistant_content("", "Hello world"),
            Some("Hello world".into())
        );
    }

    #[test]
    fn finish_completed_turn_draws_final_text_and_separator() {
        let backend = TestBackend::new(80, 20);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new();
        app.push_token("Final answer");

        finish_completed_turn(&mut app, &mut terminal, Some("Final answer"), Some("")).unwrap();

        let text = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(text.contains("Final answer"));
        assert!(text.contains("────────────────────────────────────────────────"));
        assert!(!app.is_thinking);
    }

    #[test]
    fn finish_completed_turn_appends_missing_suffix_before_separator() {
        let backend = TestBackend::new(100, 20);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new();
        app.push_token("Hello");

        finish_completed_turn(&mut app, &mut terminal, Some("Hello world"), Some("Hello")).unwrap();

        let joined = app
            .output
            .iter()
            .map(|line| line.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(joined.contains("Hello world"));
        assert!(joined.ends_with("────────────────────────────────────────────────"));
    }

    #[tokio::test]
    async fn permission_prompt_accepts_single_key_without_enter() {
        let mut app = App::new();
        app.pending_permission = Some("Allow shell command?".into());
        let backend = TestBackend::new(80, 20);
        let mut terminal = Terminal::new(backend).unwrap();
        let (tx, mut rx) = mpsc::unbounded_channel();

        tx.send(AppEvent::Key(KeyEvent {
            code: KeyCode::Char('y'),
            modifiers: KeyModifiers::empty(),
            kind: KeyEventKind::Press,
            state: KeyEventState::empty(),
        }))
        .unwrap();

        let response = wait_for_permission_input(&mut app, &mut rx, &mut terminal).await;
        assert_eq!(response, "y");
        assert!(app.input.is_empty());
        assert_eq!(app.cursor, 0);
    }

    #[tokio::test]
    async fn permission_prompt_accepts_raw_carriage_return_as_enter() {
        let mut app = App::new();
        app.pending_permission = Some("Allow shell command?".into());
        app.input = "yes".into();
        app.cursor = app.input.len();
        let backend = TestBackend::new(80, 20);
        let mut terminal = Terminal::new(backend).unwrap();
        let (tx, mut rx) = mpsc::unbounded_channel();

        tx.send(AppEvent::Key(KeyEvent {
            code: KeyCode::Char('\r'),
            modifiers: KeyModifiers::empty(),
            kind: KeyEventKind::Press,
            state: KeyEventState::empty(),
        }))
        .unwrap();

        let response = wait_for_permission_input(&mut app, &mut rx, &mut terminal).await;
        assert_eq!(response, "yes");
        assert!(app.input.is_empty());
        assert_eq!(app.cursor, 0);
    }
}

/// Wait for the user to respond to a permission prompt in the TUI.
/// Blocks until Enter is pressed, returns the trimmed input (e.g., "y", "n", "a").
async fn wait_for_modal_input(
    app: &mut App,
    rx: &mut mpsc::UnboundedReceiver<AppEvent>,
    terminal: &mut Terminal<impl Backend>,
    instant_keys: &[char],
) -> String {
    loop {
        let _ = terminal.draw(|frame| ui::draw(frame, app));

        let evt = match rx.recv().await {
            Some(e) => e,
            None => return "n".into(),
        };

        match evt {
            AppEvent::Key(key) => match key.code {
                KeyCode::Enter => {
                    let response = app.input.trim().to_lowercase();
                    app.input.clear();
                    app.cursor = 0;
                    return response;
                }
                KeyCode::Char('\n') | KeyCode::Char('\r') => {
                    let response = app.input.trim().to_lowercase();
                    app.input.clear();
                    app.cursor = 0;
                    return response;
                }
                KeyCode::Char(c) => {
                    if key.modifiers.is_empty() {
                        let lower = c.to_ascii_lowercase();
                        if app.input.is_empty() && instant_keys.contains(&lower) {
                            app.input.clear();
                            app.cursor = 0;
                            return lower.to_string();
                        }
                    }
                    app.insert_char(c);
                }
                KeyCode::Backspace => app.delete_char(),
                KeyCode::Esc => {
                    app.input.clear();
                    app.cursor = 0;
                    return "n".into();
                }
                _ => {}
            },
            AppEvent::Tick => {} // re-render
            _ => {}
        }
    }
}

async fn wait_for_permission_input(
    app: &mut App,
    rx: &mut mpsc::UnboundedReceiver<AppEvent>,
    terminal: &mut Terminal<impl Backend>,
) -> String {
    wait_for_modal_input(app, rx, terminal, &['y', 'n', 'a']).await
}

async fn fulfill_permission_request(
    app: &mut App,
    rx: &mut mpsc::UnboundedReceiver<AppEvent>,
    terminal: &mut Terminal<impl Backend>,
    prompt: String,
) -> String {
    app.pending_permission = Some(prompt);
    app.input.clear();
    app.cursor = 0;
    let response = wait_for_permission_input(app, rx, terminal).await;
    app.pending_permission = None;
    let _ = terminal.draw(|frame| ui::draw(frame, app));
    response
}

async fn await_shell_job_repl(
    mut shell_job: crate::runtime::ShellJobHandle,
    app: &mut App,
    rx: &mut mpsc::UnboundedReceiver<AppEvent>,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    cancelled: &Arc<AtomicBool>,
) -> crate::tools::ToolResult {
    app.set_active_job("shell");
    loop {
        tokio::select! {
            event = shell_job.events_rx.recv() => {
                match event {
                    Some(ShellWorkerEvent::TimedOut { command, timeout_secs }) => {
                        app.pending_permission = Some(format!(
                            "Shell command has been running for {timeout_secs}s:\n  $ {command}\nChoose: [c]ontinue waiting or [k]ill the command."
                        ));
                        app.input.clear();
                        app.cursor = 0;
                        let _ = terminal.draw(|frame| ui::draw(frame, app));
                        let response = wait_for_modal_input(app, rx, terminal, &['c', 'k']).await;
                        app.pending_permission = None;
                        let control = match response.as_str() {
                            "c" => {
                                app.push_output("  · Continuing to wait for shell command...", LineStyle::Status);
                                ShellControl::Continue
                            }
                            _ => {
                                app.push_output("  · Shell command killed.", LineStyle::Status);
                                ShellControl::Kill
                            }
                        };
                        if shell_job.send_control(control).is_err() {
                            app.clear_active_job();
                            return crate::tools::ToolResult::err("Shell worker dropped control channel".into());
                        }
                        let _ = terminal.draw(|frame| ui::draw(frame, app));
                    }
                    Some(ShellWorkerEvent::Detached { running, command }) => {
                        // The REPL never sends ShellControl::Detach (jobs are
                        // a headless-only surface) — if this arrives anyway,
                        // fail safe: kill the command rather than leak it.
                        app.clear_active_job();
                        let _ = crate::tools::shell::kill(running, 0);
                        return crate::tools::ToolResult::err(format!(
                            "Shell command detached unexpectedly and was killed: {command}"
                        ));
                    }
                    Some(ShellWorkerEvent::Completed(result)) => {
                        app.clear_active_job();
                        if cancelled.load(Ordering::Relaxed) {
                            cancelled.store(false, Ordering::Relaxed);
                        }
                        if matches!(&result, Ok(tool_result) if !tool_result.success && tool_result.content == "Command interrupted by user.") {
                            app.push_output("  · Shell command interrupted.", LineStyle::Status);
                        }
                        return match result {
                            Ok(tool_result) => tool_result,
                            Err(err) => crate::tools::ToolResult::err(err),
                        };
                    }
                    None => {
                        app.clear_active_job();
                        if cancelled.load(Ordering::Relaxed) {
                            cancelled.store(false, Ordering::Relaxed);
                        }
                        return crate::tools::ToolResult::err("Shell worker dropped before reporting a result".into());
                    }
                }
            }
            evt = rx.recv() => {
                match evt {
                    Some(AppEvent::Tick) => {
                        let _ = terminal.draw(|frame| ui::draw(frame, app));
                    }
                    Some(AppEvent::Key(key)) if handle_background_key(app, &key) => {
                        let _ = terminal.draw(|frame| ui::draw(frame, app));
                    }
                    Some(AppEvent::Key(key)) if event::is_ctrl_c(&key) => {
                        cancelled.store(true, Ordering::Relaxed);
                        app.push_output("(interrupted)", LineStyle::Status);
                        let _ = terminal.draw(|frame| ui::draw(frame, app));
                    }
                    Some(AppEvent::Mouse(_)) => {}
                    Some(AppEvent::PermissionRequest(prompt, response_tx)) => {
                        let response = fulfill_permission_request(app, rx, terminal, prompt).await;
                        let _ = response_tx.send(response);
                    }
                    Some(_) => {}
                    None => {
                        app.clear_active_job();
                        if cancelled.load(Ordering::Relaxed) {
                            cancelled.store(false, Ordering::Relaxed);
                        }
                        return crate::tools::ToolResult::err("Event stream closed.".into());
                    }
                }
            }
        }
    }
}
