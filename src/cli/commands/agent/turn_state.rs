//! Per-turn agent-loop state, shared by the headless run loop and the REPL
//! agent loop. Plain `pub(crate)` fields (no accessors): call sites reset
//! different subsets at different points and the split borrows must keep
//! working.

use super::{debugger, loop_detector, spiral, validation};

/// The mutable per-turn state both agent loops thread through a turn. The
/// round counter and `had_error` stay loop-locals — they belong to the round
/// driver itself, not to the state the phases share.
#[derive(Default)]
pub(crate) struct TurnState {
    /// Whether the user chose to continue past a pause-after-N checkpoint.
    pub(crate) user_continued: bool,
    /// Per-turn loop-detection state (see the field docs).
    pub(crate) loops: loop_detector::LoopTracker,
    /// Per-turn behavioral-gate state (see the field docs).
    pub(crate) gate: validation::GateState,
    /// Per-turn reactive-debugger state (see the field docs).
    pub(crate) debugger: debugger::DebuggerState,
    /// Per-turn spiral-reset state (see the field docs).
    pub(crate) spiral: spiral::SpiralState,
    /// Per-turn revert-to-green state (see the field docs).
    pub(crate) green: spiral::GreenState,
    pub(crate) calls_since_last_edit: u32,
    pub(crate) successful_edits_since_plan_update: u32,
    pub(crate) plan_update_requested: bool,
    pub(crate) nudged_premature_exit: bool,
    pub(crate) nudged_no_plan: bool,
    /// Force a context compaction before the next LLM request (the read-loop
    /// ladder's escalation — see `LoopTracker::read_nudges`).
    pub(crate) force_compact_next_round: bool,
    /// Consecutive reactive-compaction retries (context exhaustion signaled
    /// by the server — see compressor::force_compress). Reset whenever a
    /// response is successfully consumed, so this bounds futile retries of
    /// one failing request, not total compactions over a long run.
    pub(crate) context_compact_retries: usize,
    /// Consecutive LLM requests that died on a tool-call argument problem
    /// (server-side "Failed to parse tool call arguments" or our own
    /// streaming size cap) with no completed response in between. Each one
    /// costs a round but no model turn — Devstral spun 436 rounds in two
    /// seconds this way (2026-08-23 bench) once a truncated call sat in
    /// history. Escalates: scrub history → compact → abort the turn.
    pub(crate) truncated_call_errors_in_a_row: usize,
    /// Ceremony-gate latch: whether this context segment has ever had a plan.
    /// Tool *visibility* only ever widens within a segment — see
    /// `visible_tool_defs`. Reset wherever the context is scrapped and
    /// reassembled, because that restarts the ceremony deliberately.
    pub(crate) plan_ever_set: bool,
}

/// Skill-cursor finish-gate state (headless-only): the step we're currently
/// blocking premature-finish on, and how many times the model has tried to
/// stop on it. A cursor with steps remaining means the task is NOT done, so
/// the model must not be allowed to finish — but after it insists a few
/// times we take the step as done and advance (anti-spin). Reset when the
/// step changes.
#[derive(Default)]
pub(crate) struct SkillTurnState {
    pub(crate) exit_step: Option<String>,
    pub(crate) exit_stops: usize,
    /// How many times the blocked-stop path has escalated to the step judge on
    /// the CURRENT step (reset with `exit_stops` on step change) — the
    /// judge fires on every SKILL_EXIT_MAX_STOPS-th blocked stop, capped.
    pub(crate) stop_judge_fires: usize,
    /// Last skill-step judge "not done" reason surfaced to the model, so we
    /// don't repeat an identical nudge every judge cycle (repeats feed loops).
    pub(crate) last_judge_nudge: Option<String>,
    /// Log-only judge-veto observation (2026-09-01 e2e: skill(done) bypassed a
    /// standing not-done verdict on 4 unchecked steps): the completion judge's
    /// last not-done verdict as (skill::step key, reason, edits_total at the
    /// time). When skill(done) lands on an UNCHECKED step with a standing
    /// verdict and no mutating edit since, we LOG what an enforced veto would
    /// have done — enforcement waits on measured live judge quality.
    pub(crate) last_judge_block: Option<(String, String, usize)>,
    /// Running count of successful mutating edits (stuck_check::is_mutating_edit)
    /// — the freshness clock for `last_judge_block`.
    pub(crate) edits_total: usize,
}
