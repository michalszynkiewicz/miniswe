//! Shared round phases extracted from the two agent loops (headless
//! `run()` and the REPL). Phases return [`RoundFlow`] instead of doing
//! raw control flow — the loop skeletons alone own `break`/`continue`.

use std::sync::Arc;

use parking_lot::Mutex;

use std::sync::atomic::AtomicBool;

use crate::config::{Config, ModelRole};
use crate::llm::{Message, ModelRouter, ToolDefinition};
use crate::logging::SessionLog;
use crate::lsp::LspClient;
use crate::mcp::McpRegistry;
use crate::runtime::{LlmWorkerHandle, ToolWorkerPool};
use crate::tools;
use crate::tools::permissions::PermissionManager;

pub(crate) mod dispatch;
pub(crate) mod done_gate;
pub(crate) mod llm_call;
pub(crate) mod preamble;
pub(crate) mod restart;

/// Borrowed per-turn view of the services both loops thread through every
/// phase. `Copy` — call sites build it inline and pass it by value.
#[derive(Clone, Copy)]
pub(crate) struct TurnCtx<'a> {
    pub config: &'a Config,
    pub router: &'a ModelRouter,
    pub llm_worker: &'a LlmWorkerHandle,
    pub lsp: &'a Option<Arc<LspClient>>,
    pub snapshots: &'a Option<Arc<Mutex<tools::snapshots::SnapshotManager>>>,
    pub log: &'a SessionLog,
    pub tool_defs: &'a [ToolDefinition],
    pub cancelled: &'a Arc<AtomicBool>,
    pub model_role: ModelRole,
    pub fast_baseline_errors: usize,
    pub tool_def_tokens: usize,
    pub max_rounds: usize,
    pub perms: &'a Arc<PermissionManager>,
    pub tool_pool: &'a ToolWorkerPool,
    pub mcp_registry: &'a Option<Arc<Mutex<McpRegistry>>>,
    pub fast_revisions: &'a Option<Arc<tools::RevisionStore>>,
    pub job_registry: &'a Arc<tools::jobs::JobRegistry>,
    /// The original user message for this turn — the recovery goal used by the
    /// done-gate re-anchor, the debugger sub-agent, and SCRAP re-assembly.
    pub task: &'a str,
    pub mcp_summary: Option<&'a str>,
    /// `context::assemble`'s plan-only flag; the REPL always passes `false`.
    pub plan_only: bool,
}

/// Explicit behavior deltas between the two loops — never silently
/// unified. Constant per loop; built once above the round loop.
#[derive(Clone, Copy)]
pub(crate) struct TurnOptions {
    /// Headless: expose the skill(done) advance control while a
    /// step-cursor is active (that surface registers the skill tool;
    /// the REPL does not).
    pub skill_steps: bool,
    /// REPL: consume the cancel flag before ending the turn on interrupt.
    pub clear_cancel_on_interrupt: bool,
    /// REPL: a stopped worker (or closed UI) ends the turn quietly —
    /// stream_llm already surfaced the error line. Headless instead folds
    /// WorkerStopped into the generic LLM error ladder.
    pub worker_stopped_ends_turn: bool,
    /// Compact-retry UX in the LLM error ladder.
    pub compaction: CompactionUx,
    /// REPL explore turns: skip the behavioral done-gate entirely. A read-only
    /// Q&A turn makes no edits, so there is nothing to behaviorally verify and
    /// blocking the answer would be nonsensical.
    pub read_only: bool,
    /// Headless: nudge once if background jobs are still running at finish time
    /// (session end kills them). The REPL has no such gate.
    pub live_jobs_gate: bool,
    /// Headless: record the last failing tool call and background-job failure
    /// banners. Both feed the loop-recovery ladder's `recover_output` chain,
    /// which only the headless loop has.
    pub failure_tracking: bool,
    /// Headless: feed the stuck-signature tracker, read by the
    /// `tools.stuck_check` fire in the headless postamble.
    pub stuck_tracking: bool,
    /// Headless: the `file(action='revert')` arm (snapshot-manager revert)
    /// exists at all. The REPL has no such arm — a revert call falls
    /// through to the generic dispatcher instead.
    pub snapshot_revert_arm: bool,
    /// Headless: the refactor arm also accepts the flat single-purpose
    /// aliases (`add_function_param`, `drop_function_param`,
    /// `rename_symbol`), normalizing their args into the grouped
    /// `refactor` shape via `flat_to_refactor_args`. The REPL's refactor
    /// arm matches only `"refactor"`; calls under the flat names fall
    /// through to the generic dispatcher instead.
    pub flat_refactor_aliases: bool,
    /// Headless: the `mcp_use` arm checks the MCP permission inline
    /// (`perms.check(&Action::McpUse(..))`) before submitting to the tool
    /// pool, and awaits the job by matching `tool_pool.submit(..)`
    /// directly instead of through `AgentUi::await_tool_job`. The REPL's
    /// permission modal already ran upstream of dispatch, so it skips the
    /// inline check and always awaits via
    /// `ui.await_tool_job(.., "mcp_use", ..)`.
    pub inline_mcp_permission_check: bool,
    /// Headless: a shell-run whose foreground wait times out may
    /// auto-promote to a tracked background job (`Some(job_registry)`
    /// passed as `await_shell_job`'s `promote_to`) — but only in true
    /// `--headless` runs (nobody can answer the continue/kill prompt);
    /// a plain one-shot `run()` invocation still prompts. Set to the
    /// loop's own `headless` flag, not a constant. The REPL always passes
    /// `None` — jobs are a headless-only surface; the human answers the
    /// continue/kill modal instead.
    pub register_shell_jobs: bool,
    /// REPL: the `shell` (jobs) arm submits `tools::jobs::execute` to the
    /// tool-worker pool and awaits it via `ui.await_tool_job(.., "jobs",
    /// ..)`, keeping the TUI responsive. Headless awaits
    /// `tools::jobs::execute(..)` inline on the calling task instead.
    pub jobs_on_pool: bool,
    /// Headless: the `plan` tool job is awaited by matching
    /// `tool_pool.submit(..)` directly instead of going through
    /// `AgentUi::await_tool_job`. Cost: unlike every other pooled tool,
    /// headless's plan job is NOT bounded by `TOOL_JOB_DEADLINE_SECS`
    /// (that timeout lives inside headless's `await_tool_job`, in
    /// `run/ui.rs`). Preserved as-is — this is a refactor, not a behavior
    /// change; fixing the missing deadline is a separate decision.
    pub plan_job_direct_await: bool,
}

/// How the error ladder's compact-retry branches announce themselves and
/// handle a compaction that frees nothing.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum CompactionUx {
    /// Headless: announce after a successful compaction; a stuck
    /// compaction falls through to the rest of the ladder.
    Batch,
    /// REPL: announce before the (long) pump — the user is watching; a
    /// stuck compaction ends the turn with its own error line.
    Interactive,
}

/// What the round does next; only the loop skeletons translate this into
/// actual control flow.
pub(crate) enum RoundFlow {
    Continue,
    /// End the turn; `error` feeds the headless `had_error` exit path
    /// (the REPL ignores it).
    EndTurn {
        error: bool,
    },
}

/// What the round does after finishing one tool call in the batch.
pub(crate) enum CallFlow {
    /// Move on to the next tool call in this round's batch.
    NextCall,
    /// Abandon the rest of the batch and restart the round loop — the
    /// context was re-assembled underneath us (debugger SCRAP).
    RestartRound,
}

/// Outcome of the LLM-call phase.
pub(crate) enum LlmFlow {
    /// The sanitized assistant message — already logged, streamed text
    /// reconciled, and pushed to conversation_history when meaningful.
    Ready(Message),
    /// Retry from the top of the round (compaction ran or a hint was
    /// injected).
    Retry,
    /// End the turn; same `error` contract as [`RoundFlow::EndTurn`].
    EndTurn { error: bool },
}
