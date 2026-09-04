//! [`AgentUi`] for the REPL: pushes lines into the ratatui `App` buffer and
//! keeps the TUI responsive by racing every long await against the event
//! channel (Tick redraws, background scroll keys, ctrl-c, in-band permission
//! requests) — the same select! shapes the loop uses today, delegated to the
//! `support` pumps where they already exist.
//!
//! Fields are `pub(super)` on purpose: loop paths that stay REPL-only
//! (explore gate, permission modals, interrupt checkpoints) keep direct
//! `ui.app` / `ui.rx` / `ui.terminal` access instead of growing trait
//! methods they'd never share.

use super::*;

use std::future::Future;

use crate::cli::commands::agent::subagent::{AgentOutput, AgentTask, run_subagents};
use crate::cli::commands::agent::ui::{AgentUi, LlmOutcome, PauseDecision, UiEvent};

pub(super) struct TuiUi<'a> {
    pub(super) app: &'a mut App,
    pub(super) rx: &'a mut mpsc::UnboundedReceiver<AppEvent>,
    pub(super) terminal: &'a mut Terminal<CrosstermBackend<io::Stdout>>,
    /// The assistant text streamed this round, so `finish_assistant_text`
    /// can reconcile it against the final message content
    /// (`reconcile_streamed_assistant_content`). Reset by `stream_llm`.
    rendered_assistant_text: String,
}

impl<'a> TuiUi<'a> {
    pub(super) fn new(
        app: &'a mut App,
        rx: &'a mut mpsc::UnboundedReceiver<AppEvent>,
        terminal: &'a mut Terminal<CrosstermBackend<io::Stdout>>,
    ) -> Self {
        Self {
            app,
            rx,
            terminal,
            rendered_assistant_text: String::new(),
        }
    }

    /// Reborrow the fields at one level of `&mut` — the
    /// `terminal.draw(|f| ui::draw(f, app))` closure needs `terminal` and
    /// `app` mutably at once, which only works through direct field borrows.
    #[allow(clippy::type_complexity)]
    fn parts(
        &mut self,
    ) -> (
        &mut App,
        &mut mpsc::UnboundedReceiver<AppEvent>,
        &mut Terminal<CrosstermBackend<io::Stdout>>,
        &mut String,
    ) {
        (
            &mut *self.app,
            &mut *self.rx,
            &mut *self.terminal,
            &mut self.rendered_assistant_text,
        )
    }
}

impl AgentUi for TuiUi<'_> {
    fn status(&mut self, line: &str) {
        self.app.push_output(line, LineStyle::Status);
    }

    fn error(&mut self, line: &str) {
        self.app.push_output(line, LineStyle::Error);
    }

    fn event(&mut self, ev: UiEvent) {
        match ev {
            UiEvent::MaxRoundsReached => {
                self.app
                    .push_output("Maximum tool rounds reached.", LineStyle::Error);
            }
            // The REPL ends the turn silently on an empty response.
            UiEvent::EmptyLlmResponse => {}
            UiEvent::RevertToGreenFailed { error } => {
                self.app.push_output(
                    &format!("[revert-to-green] revert failed: {error}"),
                    LineStyle::Error,
                );
            }
            UiEvent::ReadsPruned { pairs, .. } => {
                self.app.push_output(
                    &format!("  ⋯ pruned {pairs} repeated calls from context"),
                    LineStyle::Status,
                );
            }
            UiEvent::ForcingCompaction => {
                self.app.push_output(
                    "  ⚠ Read loop persisted — forcing context compaction",
                    LineStyle::Status,
                );
            }
            // The REPL's stream_llm/error lines already said everything.
            UiEvent::LlmErrorEndpointHint { .. } => {}
        }
    }

    fn tool_call_started(&mut self, name: &str, args_summary: &str) {
        let (app, _, terminal, _) = self.parts();
        app.push_output(&format!("  → {name}({args_summary})"), LineStyle::ToolCall);
        let _ = terminal.draw(|frame| ui::draw(frame, app));
    }

    fn tool_result(&mut self, name: &str, ok: bool, first_line: &str) {
        let icon = if ok { "✓" } else { "✗" };
        let style = if ok {
            LineStyle::ToolOk
        } else {
            LineStyle::ToolErr
        };
        self.app
            .push_output(&format!("  {icon} {name}: {first_line}"), style);
    }

    fn store_tool_result(&mut self, name: &str, content: &str) {
        self.app.store_tool_result(name, content);
    }

    /// The TUI renders its separator once at end-of-turn
    /// (`finish_completed_turn`), not per round.
    fn separator(&mut self) {}

    fn refresh_plan(&mut self, config: &Config, round: usize) {
        refresh_plan_panel(self.app, config, round);
    }

    fn notify_interrupted(&mut self) {
        self.app.push_output("(interrupted)", LineStyle::Status);
    }

    async fn pump<T>(&mut self, fut: impl Future<Output = T>) -> T {
        let (app, rx, terminal, _) = self.parts();
        let mut fut = std::pin::pin!(fut);
        loop {
            tokio::select! {
                biased;
                value = &mut fut => break value,
                evt = rx.recv() => {
                    if matches!(evt, Some(AppEvent::Tick)) {
                        let _ = terminal.draw(|frame| ui::draw(frame, app));
                    }
                }
            }
        }
    }

    async fn stream_llm(
        &mut self,
        llm_worker: &LlmWorkerHandle,
        role: ModelRole,
        request: ChatRequest,
        cancelled: &Arc<AtomicBool>,
    ) -> LlmOutcome {
        let (app, rx, terminal, rendered_assistant_text) = self.parts();
        app.is_thinking = true;
        app.set_active_job("llm");

        // Render before the LLM call so the spinner is visible immediately.
        let _ = terminal.draw(|frame| ui::draw(frame, app));

        rendered_assistant_text.clear();
        let mut token_count = 0u32;
        let mut llm_events = llm_worker.submit(role, request, cancelled.clone());
        let outcome = loop {
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
                        Some(LlmWorkerEvent::Completed(Ok(r))) => break LlmOutcome::Response(r),
                        Some(LlmWorkerEvent::Completed(Err(err_str))) => {
                            break LlmOutcome::Error(err_str);
                        }
                        None => {
                            app.push_output("LLM worker stopped unexpectedly.", LineStyle::Error);
                            break LlmOutcome::WorkerStopped;
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
                            break LlmOutcome::UiClosed;
                        }
                    }
                }
            }
        };

        app.clear_active_job();
        // Re-render after the LLM response.
        let _ = terminal.draw(|frame| ui::draw(frame, app));
        outcome
    }

    fn finish_assistant_text(&mut self, content: Option<&str>) {
        let (app, _, terminal, rendered_assistant_text) = self.parts();
        app.flush_tokens();
        if let Some(content) = content
            && let Some(missing) =
                reconcile_streamed_assistant_content(rendered_assistant_text, content)
        {
            app.push_token(&missing);
            app.flush_tokens();
            let _ = terminal.draw(|frame| ui::draw(frame, app));
        }
    }

    async fn await_tool_job(
        &mut self,
        result_rx: tokio::sync::oneshot::Receiver<Result<crate::tools::ToolResult, String>>,
        label: &str,
        cancelled: &Arc<AtomicBool>,
    ) -> crate::tools::ToolResult {
        let (app, rx, terminal, _) = self.parts();
        let mut result_rx = result_rx;
        await_tool_job_ui(rx, terminal, app, label, &mut result_rx, cancelled).await
    }

    async fn await_shell_job(
        &mut self,
        shell_job: crate::runtime::ShellJobHandle,
        cancelled: &Arc<AtomicBool>,
        // Jobs are a headless-only surface — the human answers the
        // continue/kill modal instead.
        _promote_to: Option<&tools::jobs::JobRegistry>,
    ) -> crate::tools::ToolResult {
        let (app, rx, terminal, _) = self.parts();
        await_shell_job_repl(shell_job, app, rx, terminal, cancelled).await
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
        mcp_registry: &Option<Arc<Mutex<McpRegistry>>>,
        lsp: &Option<Arc<LspClient>>,
        fast_revisions: &Option<Arc<tools::RevisionStore>>,
        fast_baseline_errors: usize,
        cancelled: &Arc<AtomicBool>,
    ) -> Vec<AgentOutput> {
        let (app, rx, terminal, _) = self.parts();
        let (out_tx, mut out_rx) = tokio::sync::mpsc::unbounded_channel::<(String, LineStyle)>();
        let subagents_fut = run_subagents(
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
        outputs.unwrap()
    }

    async fn confirm_continue(&mut self, pause_at: usize) -> PauseDecision {
        let (app, rx, terminal, _) = self.parts();
        app.pending_permission = Some(format!(
            "{pause_at} tool rounds used. Continue? [y]es / [n]o:"
        ));
        app.input.clear();
        app.cursor = 0;
        let response = wait_for_modal_input(app, rx, terminal, &['y', 'n']).await;
        app.pending_permission = None;
        match response.as_str() {
            "y" | "yes" | "" => PauseDecision::Continue,
            _ => PauseDecision::WrapUp,
        }
    }
}
