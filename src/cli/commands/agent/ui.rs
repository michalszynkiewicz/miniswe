//! The frontend seam between the agent round loops and their two UIs.
//!
//! Both round loops — headless `run()` (`run/main_loop.rs`) and the REPL's
//! `run_agent_loop()` (`repl/agent_loop.rs`) — do the same work but render
//! it differently: plain ANSI lines on stderr/stdout vs a ratatui frame
//! that must keep redrawing while long awaits are in flight (every long
//! await on the TUI side races the event channel for Tick redraws,
//! background scroll keys, ctrl-c, and in-band permission requests). This
//! trait captures every point where the loops touch their frontend, so the
//! loop bodies can converge without flattening real presentation
//! differences.
//!
//! String strategy (the drift defense): a line that is byte-identical in
//! both frontends is passed through [`AgentUi::status`] / [`AgentUi::error`]
//! by the caller; a line whose wording, style, or channel differs per
//! frontend becomes a [`UiEvent`] variant and each impl renders its own
//! exact bytes. The `UiEvent` inventory is therefore the audit trail of
//! every deliberate presentation divergence between the two loops.
//!
//! Deliberately NOT part of the trait (they stay per-loop for now): the LLM
//! error-recovery ladders (`had_error` marking and the endpoint hint are
//! headless-only), tool permission prompting, and the REPL-only interrupt
//! checkpoints — see the behavior-delta table in the unification plan.
//!
//! Static dispatch only — two construction sites, no `dyn`.

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use parking_lot::Mutex;

use crate::cli::commands::agent::subagent::{AgentOutput, AgentTask};
use crate::config::{Config, ModelRole};
use crate::llm::{ChatRequest, ChatResponse};
use crate::lsp::LspClient;
use crate::mcp::McpRegistry;
use crate::runtime::{LlmWorkerHandle, ShellJobHandle, ToolWorkerPool};
use crate::tools::{self, ToolResult};

/// How an LLM streaming call ended (see [`AgentUi::stream_llm`]).
pub(crate) enum LlmOutcome {
    /// The stream completed with a full response.
    Response(ChatResponse),
    /// The stream failed; the caller owns the recovery ladder (interrupt /
    /// context-exceeded / oversized-args / truncated-call / generic — the
    /// two loops' ladders genuinely differ, so they stay per-loop).
    Error(String),
    /// The LLM worker channel closed. The TUI impl has already rendered its
    /// own notice; the headless caller routes this through its generic
    /// error ladder as "LLM worker stopped unexpectedly".
    WorkerStopped,
    /// The UI's own event stream closed (TUI teardown) — end the turn.
    /// Never returned by the headless impl.
    UiClosed,
}

/// Answer to the pause-after-N-rounds interaction point
/// (see [`AgentUi::confirm_continue`]).
pub(crate) enum PauseDecision {
    /// Keep going (also the automatic headless answer).
    Continue,
    /// User declined — the loop tells the model to wrap up.
    WrapUp,
}

/// A loop notification whose rendering deliberately differs per frontend
/// (wording, style, or channel). One variant per divergence; each impl owns
/// its exact bytes.
pub(crate) enum UiEvent {
    /// Round counter passed `max_rounds` — the turn is being cut off.
    MaxRoundsReached,
    /// The LLM response carried no choices (headless prints an error; the
    /// REPL ends the turn silently).
    EmptyLlmResponse,
    /// revert-to-green found a broken tree but the snapshot revert failed.
    RevertToGreenFailed { error: String },
    /// `prune_repeated_reads` dropped `pairs` call/result pairs across
    /// `keys` distinct calls from context.
    ReadsPruned {
        pairs: usize,
        keys: usize,
        deepest: Option<String>,
    },
    /// Read-loop escalation is forcing a context compaction this round.
    ForcingCompaction,
    /// A fatal LLM error was just surfaced (headless adds a check-your-
    /// server hint; the REPL shows nothing extra).
    LlmErrorEndpointHint { endpoint: String },
}

/// Everything an agent round loop needs from its frontend: line output,
/// LLM streaming, long-await pumping, job waits, and the interaction
/// points. Implemented by `run::ui::HeadlessUi` (plain ANSI + stdin) and
/// `repl::tui_ui::TuiUi` (ratatui `App` + event-channel pumps).
pub(crate) trait AgentUi {
    /// A status line whose bytes are identical in both frontends.
    fn status(&mut self, line: &str);
    /// An error line whose bytes are identical in both frontends.
    fn error(&mut self, line: &str);
    /// A notification whose rendering differs per frontend.
    fn event(&mut self, ev: UiEvent);
    /// Announce a tool call about to execute.
    fn tool_call_started(&mut self, name: &str, args_summary: &str);
    /// Render a tool result summary line.
    fn tool_result(&mut self, name: &str, ok: bool, first_line: &str);
    /// Record the full tool result for the TUI's detail view (no-op
    /// headless).
    fn store_tool_result(&mut self, name: &str, content: &str);
    /// Per-round separator before the LLM call. The TUI renders its
    /// separator once at end-of-turn instead (`finish_completed_turn`), so
    /// its impl is a no-op.
    fn separator(&mut self);
    /// Refresh the live plan panel from `plan.md` (no-op headless).
    fn refresh_plan(&mut self, config: &Config, round: usize);
    /// The user interrupted the turn (REPL ctrl-c checkpoints; the headless
    /// loop has no interrupt checkpoints and never calls this).
    fn notify_interrupted(&mut self);

    /// Drive a long non-tool await (context compaction) to completion while
    /// keeping the frontend alive — the TUI impl races the event channel
    /// for Tick redraws; headless just awaits.
    async fn pump<T>(&mut self, fut: impl Future<Output = T>) -> T;

    /// Submit `request` to the LLM worker and stream the response to the
    /// frontend (spinner-then-tokens headless; token pushes with periodic
    /// redraws in the TUI). Covers the whole per-round streaming lifecycle,
    /// including the TUI's active-job marker and post-stream redraw.
    async fn stream_llm(
        &mut self,
        llm_worker: &LlmWorkerHandle,
        role: ModelRole,
        request: ChatRequest,
        cancelled: &Arc<AtomicBool>,
    ) -> LlmOutcome;

    /// Finalize the streamed assistant text once the full message is known:
    /// headless prints the trailing newline; the TUI flushes its token
    /// buffer and reconciles the streamed text against the final content.
    fn finish_assistant_text(&mut self, content: Option<&str>);

    /// Await a worker-pool tool job. The headless impl bounds the wait by
    /// its hard deadline (nobody is watching — a wedged worker must not
    /// hang the run); the TUI impl pumps events instead and lets the human
    /// interrupt.
    async fn await_tool_job(
        &mut self,
        result_rx: tokio::sync::oneshot::Receiver<Result<ToolResult, String>>,
        label: &str,
        cancelled: &Arc<AtomicBool>,
    ) -> ToolResult;

    /// Await a shell job, handling its timeout interaction: headless
    /// auto-promotes to a background job when `promote_to` is set (or falls
    /// back to a stdin prompt); the TUI asks continue/kill in a modal and
    /// ignores `promote_to` (jobs are a headless-only surface).
    async fn await_shell_job(
        &mut self,
        shell_job: ShellJobHandle,
        cancelled: &Arc<AtomicBool>,
        promote_to: Option<&tools::jobs::JobRegistry>,
    ) -> ToolResult;

    /// Run `spawn_agents` sub-agents to completion, forwarding their live
    /// output lines to the frontend where one exists (TUI only).
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
    ) -> Vec<AgentOutput>;

    /// The pause-after-N-rounds interaction point. Each impl owns its
    /// prompt wording and input mechanism (stdin line vs TUI modal); the
    /// headless auto-continue notice lives in the headless impl.
    async fn confirm_continue(&mut self, pause_at: usize) -> PauseDecision;
}
