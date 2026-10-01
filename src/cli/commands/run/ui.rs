//! [`AgentUi`] for the headless/one-shot loop: plain ANSI output through the
//! `crate::tui` print helpers, stdin prompts, and deadline-bounded job waits
//! (nobody is watching — a wedged worker must not hang the run).
//!
//! `await_tool_job` / `await_shell_job_run` moved here verbatim from
//! `main_loop.rs`; the loop keeps calling them directly until it adopts the
//! trait.

use super::*;

use std::future::Future;

use crate::cli::commands::agent::subagent::{AgentOutput, AgentTask, run_subagents};
use crate::cli::commands::agent::ui::{
    AgentUi, LlmOutcome, PauseDecision, PreflightPermission, UiEvent,
};

pub(super) struct HeadlessUi {
    /// True for `--headless` runs: interaction points auto-continue with a
    /// logged notice instead of blocking on stdin.
    pub(super) headless: bool,
}

impl AgentUi for HeadlessUi {
    fn status(&mut self, line: &str) {
        tui::print_status(line);
    }

    fn error(&mut self, line: &str) {
        tui::print_error(line);
    }

    fn event(&mut self, ev: UiEvent) {
        match ev {
            UiEvent::MaxRoundsReached => {
                tui::print_error("Maximum tool rounds reached. Stopping.");
            }
            UiEvent::EmptyLlmResponse => {
                tui::print_error("Empty response from LLM");
            }
            UiEvent::RevertToGreenFailed { error } => {
                tui::print_status(&format!("[revert-to-green] revert failed: {error}"));
            }
            UiEvent::ReadsPruned {
                pairs,
                keys,
                deepest,
            } => {
                tui::print_status(&format!(
                    "[prune] dropped {} repeated {} from context{}",
                    pairs,
                    if keys == 1 { "call" } else { "calls" },
                    deepest
                        .as_deref()
                        .map(|d| format!(" (deepest: {d})"))
                        .unwrap_or_default()
                ));
            }
            UiEvent::ForcingCompaction => {
                tui::print_status("Loop persisted past the nudge — forcing context compaction.");
            }
            UiEvent::LlmErrorEndpointHint { endpoint } => {
                tui::print_status(&format!(
                    "Check that your LLM server is running at {endpoint}"
                ));
            }
            UiEvent::TruncatedArgs { name, .. } => {
                tui::print_tool_result(
                    &name,
                    false,
                    "arguments cut off by the output limit — not executed",
                );
            }
            UiEvent::RepeatedRead {
                name,
                args_summary,
                escalate,
            } => {
                tui::print_status(&format!(
                    "Repeated read: {name}({args_summary}) — {}, continuing",
                    if escalate {
                        "nudge failed, forcing compaction next round"
                    } else {
                        "nudge sent"
                    }
                ));
            }
            UiEvent::LoopDetected {
                name,
                args_summary,
                cycle_period,
            } => {
                let how = match cycle_period {
                    Some(period) => {
                        format!("cycling through the same {period} calls (period-{period} cycle)")
                    }
                    None => "repeated 3 times".to_string(),
                };
                tui::print_error(&format!(
                    "Loop detected: {name}({args_summary}) {how} — surfacing a hint, giving the model one more round"
                ));
            }
            UiEvent::LoopRecovering { name, args_summary } => {
                tui::print_error(&format!(
                    "Loop detected again ({name}({args_summary})) — routing through the recovery ladder instead of stopping"
                ));
            }
            UiEvent::LoopStopping { name, args_summary } => {
                tui::print_error(&format!(
                    "Loop detected again ({name}({args_summary})) after the recovery hint — stopping this turn"
                ));
            }
            UiEvent::ExploreBlocked { name } => {
                // Never fires headless (no explore mode) — rendering chosen
                // for consistency with the other tool_result-style events.
                tui::print_tool_result(&name, false, "blocked — read-only mode");
            }
            UiEvent::WriteBlockedNoPlan { name } => {
                tui::print_tool_result(&name, false, "blocked: no plan");
            }
        }
    }

    fn tool_call_started(&mut self, name: &str, args_summary: &str) {
        tui::print_tool_call(name, args_summary);
    }

    fn tool_result(&mut self, name: &str, ok: bool, first_line: &str) {
        tui::print_tool_result(name, ok, first_line);
    }

    fn store_tool_result(&mut self, _name: &str, _content: &str) {}

    fn separator(&mut self) {
        tui::print_separator();
    }

    fn refresh_plan(&mut self, _config: &Config, _round: usize) {}

    /// No-op: headless has no live plan panel to refresh.
    fn after_plan_tool(&mut self, _config: &Config, _round: usize) {}

    fn spawning_subagents(&mut self, count: usize) {
        tui::print_status(&format!("spawning {} subagents...", count));
    }

    fn notify_interrupted(&mut self) {}

    async fn pump<T>(&mut self, fut: impl Future<Output = T>) -> T {
        fut.await
    }

    async fn stream_llm(
        &mut self,
        llm_worker: &LlmWorkerHandle,
        role: ModelRole,
        request: crate::llm::ChatRequest,
        cancelled: &Arc<AtomicBool>,
    ) -> LlmOutcome {
        eprint!("\x1b[2m⠋ thinking...\x1b[0m");
        std::io::stderr().flush().ok();
        let mut thinking = true;

        let mut llm_events = llm_worker.submit(role, request, cancelled.clone());
        loop {
            match llm_events.recv().await {
                Some(LlmWorkerEvent::Token(token)) => {
                    if thinking {
                        thinking = false;
                        eprint!("\r\x1b[2K");
                        std::io::stderr().flush().ok();
                    }
                    tui::print_token(&token);
                }
                Some(LlmWorkerEvent::Completed(Ok(r))) => {
                    if thinking {
                        eprint!("\r\x1b[2K");
                        std::io::stderr().flush().ok();
                    }
                    return LlmOutcome::Response(r);
                }
                Some(LlmWorkerEvent::Completed(Err(e))) => {
                    eprint!("\r\x1b[2K");
                    std::io::stderr().flush().ok();
                    return LlmOutcome::Error(e);
                }
                None => {
                    eprint!("\r\x1b[2K");
                    std::io::stderr().flush().ok();
                    return LlmOutcome::WorkerStopped;
                }
            }
        }
    }

    fn finish_assistant_text(&mut self, content: Option<&str>) {
        if content.is_some() {
            println!();
        }
    }

    async fn await_tool_job(
        &mut self,
        result_rx: tokio::sync::oneshot::Receiver<Result<crate::tools::ToolResult, String>>,
        label: &str,
        _cancelled: &Arc<AtomicBool>,
    ) -> crate::tools::ToolResult {
        await_tool_job(result_rx, label).await
    }

    async fn await_shell_job(
        &mut self,
        shell_job: crate::runtime::ShellJobHandle,
        cancelled: &Arc<AtomicBool>,
        promote_to: Option<&tools::jobs::JobRegistry>,
    ) -> crate::tools::ToolResult {
        await_shell_job_run(shell_job, cancelled, promote_to).await
    }

    #[allow(clippy::too_many_arguments)]
    async fn drive_subagents(
        &mut self,
        tasks: Vec<AgentTask>,
        config: &Config,
        llm_worker: &LlmWorkerHandle,
        tool_pool: &ToolWorkerPool,
        tool_defs: &[crate::llm::ToolDefinition],
        perms: &Arc<crate::tools::permissions::PermissionManager>,
        mcp_registry: &Option<Arc<parking_lot::Mutex<McpRegistry>>>,
        lsp: &Option<Arc<LspClient>>,
        fast_revisions: &Option<Arc<tools::RevisionStore>>,
        fast_baseline_errors: usize,
        cancelled: &Arc<AtomicBool>,
    ) -> Vec<AgentOutput> {
        run_subagents(
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
            None,
        )
        .await
    }

    async fn confirm_continue(&mut self, pause_at: usize) -> PauseDecision {
        if self.headless {
            tui::print_status(&format!(
                "{pause_at} tool rounds used — headless, continuing without prompt."
            ));
            return PauseDecision::Continue;
        }
        tui::print_status(&format!("{pause_at} tool rounds used."));
        let response = tui::read_input("Continue? [y]es / [n]o:");
        match response.as_deref() {
            Some("y") | Some("yes") | Some("") => PauseDecision::Continue,
            _ => PauseDecision::WrapUp,
        }
    }

    async fn preflight_permission(
        &mut self,
        _perms: &crate::tools::permissions::PermissionManager,
        _action: &crate::tools::permissions::Action,
    ) -> PreflightPermission {
        // Headless has no preflight modal; permission prompts (if any) are
        // handled lazily on stdin inside tool execution.
        PreflightPermission::Allowed
    }
}

/// Hard ceiling on how long the agent loop waits for a worker-pool tool
/// job. Every wait inside a tool is individually bounded (LLM calls 600s,
/// LSP requests 10-30s, LSP stdin writes 5s), so a job that outlives this
/// ceiling is stuck in something unpreemptable. Returning an error keeps
/// the run alive — the alternative was the 08-22/08-24 harness hang,
/// where one wedged refactor silently voided the rest of a bench run.
/// Generous enough that a legitimate many-callsite refactor on a slow
/// local model stays well under it.
const TOOL_JOB_DEADLINE_SECS: u64 = 1200;

/// Await a worker-pool tool job, bounded by [`TOOL_JOB_DEADLINE_SECS`].
/// On expiry the job is abandoned: the worker thread finishes (or errors
/// out of its own bounded waits) later, and its late result is discarded
/// along with the dropped oneshot receiver.
pub(super) async fn await_tool_job(
    rx: tokio::sync::oneshot::Receiver<Result<crate::tools::ToolResult, String>>,
    tool_name: &str,
) -> crate::tools::ToolResult {
    match tokio::time::timeout(std::time::Duration::from_secs(TOOL_JOB_DEADLINE_SECS), rx).await {
        Ok(Ok(Ok(r))) => r,
        Ok(Ok(Err(e))) => crate::tools::ToolResult::err(e),
        Ok(Err(_)) => crate::tools::ToolResult::err(format!("Tool worker dropped {tool_name} job")),
        Err(_) => {
            eprintln!(
                "[tool] {tool_name} exceeded {TOOL_JOB_DEADLINE_SECS}s — abandoning the job \
                 (worker presumed wedged)"
            );
            crate::tools::ToolResult::err(format!(
                "✗ {tool_name} timed out after {TOOL_JOB_DEADLINE_SECS}s and was abandoned. Its \
                 edits may or may not have landed — re-read the affected file(s) before editing \
                 further, and do not repeat the same call."
            ))
        }
    }
}

pub(super) async fn await_shell_job_run(
    mut shell_job: crate::runtime::ShellJobHandle,
    cancelled: &AtomicBool,
    // Some(registry) = headless: long-running commands auto-promote to
    // background jobs (nobody can answer the continue/kill prompt — real
    // e2e runs hung at it for their full timeout, pkg-mcp 2026-07-13).
    // None = interactive: the human prompt stays.
    promote_to: Option<&tools::jobs::JobRegistry>,
) -> crate::tools::ToolResult {
    while let Some(event) = shell_job.events_rx.recv().await {
        match event {
            ShellWorkerEvent::TimedOut {
                command,
                timeout_secs,
            } => {
                let control = if promote_to.is_some() {
                    crate::tui::print_status(&format!(
                        "shell still running after {timeout_secs}s — promoting to background job: $ {command}"
                    ));
                    ShellControl::Detach
                } else {
                    let prompt = format!(
                        "Shell command still running after {timeout_secs}s.\n  $ {command}\n[c]ontinue waiting / [k]ill: "
                    );
                    let response = crate::tui::read_input(&prompt)
                        .unwrap_or_else(|| "k".into())
                        .trim()
                        .to_lowercase();
                    if response == "c" || response == "continue" {
                        crate::tui::print_status("continuing to wait for shell command...");
                        ShellControl::Continue
                    } else {
                        ShellControl::Kill
                    }
                };
                let is_detach = matches!(control, ShellControl::Detach);
                if shell_job.send_control(control).is_err() {
                    return crate::tools::ToolResult::err(
                        "Shell worker dropped control channel".into(),
                    );
                }
                if is_detach {
                    // Worker responds with Detached carrying the live command.
                    match shell_job.events_rx.recv().await {
                        Some(ShellWorkerEvent::Detached { running, command }) => {
                            let registry = promote_to.expect("Detach only sent when Some");
                            let (id, so_far) = registry.register(&command, running);
                            return crate::tools::ToolResult::ok(tools::jobs::promotion_message(
                                id,
                                &command,
                                timeout_secs,
                                &so_far,
                            ));
                        }
                        Some(ShellWorkerEvent::Completed(result)) => {
                            // Raced completion between TimedOut and Detach —
                            // the finished result wins.
                            return match result {
                                Ok(r) => r,
                                Err(e) => crate::tools::ToolResult::err(e),
                            };
                        }
                        _ => {
                            return crate::tools::ToolResult::err(
                                "Shell worker dropped during job promotion".into(),
                            );
                        }
                    }
                }
            }
            ShellWorkerEvent::Detached { running, command } => {
                // Defensive: only expected right after a Detach request.
                if let Some(registry) = promote_to {
                    let (id, so_far) = registry.register(&command, running);
                    return crate::tools::ToolResult::ok(tools::jobs::promotion_message(
                        id, &command, 0, &so_far,
                    ));
                }
                return crate::tools::ToolResult::err(
                    "Shell worker detached without a registry".into(),
                );
            }
            ShellWorkerEvent::Completed(result) => {
                if cancelled.load(Ordering::Relaxed) {
                    cancelled.store(false, Ordering::Relaxed);
                }
                return match result {
                    Ok(tool_result) => tool_result,
                    Err(err) => crate::tools::ToolResult::err(err),
                };
            }
        }
    }
    if cancelled.load(Ordering::Relaxed) {
        cancelled.store(false, Ordering::Relaxed);
    }
    crate::tools::ToolResult::err("Shell worker dropped before reporting a result".into())
}
