//! Thin REPL adapter over the shared turn driver: builds the REPL's
//! `TuiUi`, per-turn state, and `TurnOptions`/`TurnCtx`, then hands them to
//! `turn::driver::run_turn` — the single `'round`/`for tc` loop shared with
//! the headless `run()`.

use super::*;

/// Run one agent turn (LLM call → tool execution → repeat) inside the TUI.
#[allow(clippy::too_many_arguments)]
pub(super) async fn run_agent_turn(
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
    // Fed into `TurnCtx` but never read on this side — `opts.stuck_tracking`
    // is always false in the REPL, so the headless-only `tools.stuck_check`
    // fire in `postamble::finish_round` never evaluates it.
    let session_start = std::time::Instant::now();
    // Per-turn agent-loop state shared with the headless loop (see the field
    // docs on `turn_state::TurnState` and its sub-structs).
    let mut state = turn_state::TurnState::default();
    // Unused in the REPL (`opts.skill_steps` is false, which short-circuits every
    // read of it); the shared phases take it by reference.
    let mut skill_state = turn_state::SkillTurnState::default();
    let mut ui = TuiUi::new(app, rx, terminal);

    // Explicit behavior deltas for the shared turn phases — see
    // `turn::TurnOptions` field docs.
    let opts = turn::TurnOptions {
        skill_steps: false,
        clear_cancel_on_interrupt: true,
        worker_stopped_ends_turn: true,
        compaction: turn::CompactionUx::Interactive,
        read_only,
        live_jobs_gate: false,
        failure_tracking: false,
        stuck_tracking: false,
        snapshot_revert_arm: false,
        flat_refactor_aliases: false,
        inline_mcp_permission_check: false,
        register_shell_jobs: false,
        jobs_on_pool: true,
        plan_job_direct_await: false,
        window_repeat_detector: false,
        jobs_poll_redirect: false,
        interrupt_checkpoints: true,
        fatal_marks_error: false,
    };

    let ctx = turn::TurnCtx {
        config,
        router,
        llm_worker,
        lsp,
        snapshots,
        log: &log,
        tool_defs,
        cancelled,
        model_role: ModelRole::Default,
        fast_baseline_errors,
        tool_def_tokens,
        max_rounds,
        perms,
        tool_pool,
        mcp_registry,
        fast_revisions,
        job_registry,
        task: goal,
        mcp_summary,
        plan_only: false,
        session_start,
    };

    // The REPL never reads `had_error`/`rounds` after the loop — the turn's
    // completion is already reflected live in `app`.
    let _ = turn::driver::run_turn(
        ctx,
        opts,
        &mut state,
        &mut skill_state,
        &mut ui,
        messages,
        conversation_history,
        router,
        &log,
    )
    .await;
}
