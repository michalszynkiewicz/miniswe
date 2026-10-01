//! The frontend seam between the agent round loops and their two UIs.
//!
//! Both frontends — headless `run()` (`run/main_loop.rs`) and the REPL's
//! `repl/agent_turn.rs` — drive the same round loop, `agent/turn/driver.rs::run_turn`,
//! but render it differently: plain ANSI lines on stderr/stdout vs a ratatui frame
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
//! headless-only) and tool permission prompting — see the behavior-delta
//! table in the unification plan.
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

/// Outcome of a permission preflight for one shell/MCP tool call (see
/// [`AgentUi::preflight_permission`]).
pub(crate) enum PreflightPermission {
    /// No prompt was needed, or the user approved it.
    Allowed,
    /// The user declined the prompt.
    Denied,
    /// Blocklisted — `check_needs_prompt` returned this explanatory error
    /// instead of a prompt.
    Blocked(String),
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
    /// A tool call's arguments were cut off by the output limit and the
    /// call was not executed (headless's summary omits the char count; the
    /// REPL's names it).
    TruncatedArgs { name: String, original_chars: usize },
    /// A read/inspection tool call repeated 3x with identical args — a
    /// nudge was sent (or, on re-detection, `escalate` is true and the
    /// escalated nudge also forced a compaction next round).
    RepeatedRead {
        name: String,
        args_summary: String,
        escalate: bool,
    },
    /// First loop detection this turn: a hint was surfaced to the model and
    /// it gets one more round. `cycle_period` is `Some` for a short
    /// edit/revert-style cycle, `None` for a plain 3x-identical streak.
    LoopDetected {
        name: String,
        args_summary: String,
        cycle_period: Option<usize>,
    },
    /// Second loop detection this turn: routing through the behavioral
    /// done-gate / recovery ladder instead of stopping.
    LoopRecovering { name: String, args_summary: String },
    /// Second loop detection this turn with no recovery path available (or
    /// the recovery budget already exhausted) — the turn stops.
    LoopStopping { name: String, args_summary: String },
    /// REPL explore mode blocked a mutating tool call before dispatch
    /// (never fires headless, which has no explore mode).
    ExploreBlocked { name: String },
    /// A write tool was blocked: strict ceremony requires a plan first.
    WriteBlockedNoPlan { name: String },
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
    /// After a `plan` tool call returns: refresh the live plan panel and
    /// redraw immediately, so a checked/added/refined step appears right
    /// away instead of lagging to the next round's `refresh_plan`. No-op
    /// headless (same reason `refresh_plan` is a no-op there).
    fn after_plan_tool(&mut self, config: &Config, round: usize);
    /// Announce a `spawn_agents` dispatch is starting — byte-identical to
    /// the line headless has always printed here. No-op in the REPL:
    /// `drive_subagents` already streams each subagent's own output live
    /// as it runs, so a separate announcement line was never added.
    fn spawning_subagents(&mut self, count: usize);
    /// The user interrupted the turn (REPL ctrl-c checkpoints; the headless
    /// loop has no interrupt checkpoints and never calls this).
    fn notify_interrupted(&mut self);
    /// Re-render after a tool result landed — the TUI redraws its frame so
    /// the result shows before the next call starts; no-op headless.
    fn after_tool_call(&mut self);

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

    /// Permission preflight for a shell/MCP tool call, run before dispatch.
    /// The REPL shows a blocking TUI modal (`check_needs_prompt` → y/n/a)
    /// here; headless has no preflight — permission prompts, if any, are
    /// handled lazily on stdin inside tool execution — so it always
    /// returns `Allowed`.
    async fn preflight_permission(
        &mut self,
        perms: &crate::tools::permissions::PermissionManager,
        action: &crate::tools::permissions::Action,
    ) -> PreflightPermission;
}
