//! Shared round phases extracted from the two agent loops (headless
//! `run()` and the REPL). Phases return [`RoundFlow`] instead of doing
//! raw control flow — the loop skeletons alone own `break`/`continue`.

use std::sync::Arc;

use parking_lot::Mutex;

use crate::config::Config;
use crate::llm::ModelRouter;
use crate::logging::SessionLog;
use crate::lsp::LspClient;
use crate::runtime::LlmWorkerHandle;
use crate::tools;

pub(crate) mod preamble;

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
    pub fast_baseline_errors: usize,
    pub tool_def_tokens: usize,
    pub max_rounds: usize,
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
