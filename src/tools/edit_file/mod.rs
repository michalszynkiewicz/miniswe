//! edit_file tool — LLM plans and applies bounded edits atomically.
//!
//! The model describes the task, miniswe asks the inner LLM for a structured
//! edit plan, and then executes the planned steps against an in-memory working
//! copy. If some steps succeed and later ones fail, the successful progress is
//! preserved in memory and the tool asks for a repaired plan against the updated
//! working copy. Only the final validated result is written to disk.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Result, bail};
use lsp_types::{Diagnostic, DiagnosticSeverity};
use serde_json::Value;

use super::ToolResult;
use super::permissions::PermissionManager;
use crate::config::{Config, ModelRole};
use crate::llm::{ChatRequest, Message, ModelRouter};
use crate::logging::SessionLog;
use crate::lsp::LspClient;

mod apply;
mod parse;

pub use apply::{apply_literal_replace_in_scope, apply_patch_dry_run};
pub use parse::{EditPlanStep, EditRegion, PatchOp, parse_edit_plan, parse_patch};

use apply::{RelocateOutcome, apply_patch_dry_run_in_region, try_relocate_and_replace};
use parse::{
    MAX_FAILED_REASON_CHARS, format_completed_steps_compact, format_edit_plan_steps,
    format_preplan_log, looks_like_complete, parse_failed, parse_needs_clarification,
    partition_overlapping_steps, truncate_multiline,
};

/// Max lines per window for reliable LLM recall.
const WINDOW_SIZE: usize = 800;
/// Overlap between windows to catch edits at boundaries.
const PREPLAN_READ_OVERLAP: usize = 60;
/// Files up to this many lines skip the windowed observation pass and feed
/// their full content directly into the finalize prompt. Below this size
/// the whole file already fits comfortably in a single slice, so the
/// observation round is pure latency overhead.
const SMALL_FILE_THRESHOLD: usize = 200;
const MAX_PLAN_ATTEMPTS: usize = 4;
/// Additional plan attempts granted when the failing trajectory is
/// strictly improving (new LSP error count < best-so-far). Bounds the
/// "promising-fix-loop" retry credit introduced to stop cutting off
/// slow converging runs at `MAX_PLAN_ATTEMPTS`.
const MAX_EXTRA_ATTEMPTS: usize = 2;
pub(super) const MAX_PREPLAN_STEPS: usize = 100;
pub(super) const MAX_PREPLAN_LOG_CHARS: usize = 20000;
const LARGE_TRUNCATION_MIN_LINES: usize = 50;

mod execute;
mod lsp_validation;
mod preplan;
mod repair;
#[cfg(test)]
mod tests;

pub use execute::{build_windows, execute};

use execute::*;
use lsp_validation::*;
use preplan::*;
use repair::*;

struct SplitResult {
    content: String,
    message: String,
}

/// A step that was rejected before execution because it overlapped an
/// earlier step. The kept step is applied normally; the dropped one is
/// reported as a per-step failure in the result so the agent sees both
/// the success and the failure side-by-side.
pub(super) struct DroppedStep {
    pub(super) step: EditPlanStep,
    pub(super) reason: String,
}

/// What one planning attempt produced. The model can return a concrete
/// edit plan, or at any phase ask for clarification via
/// `NEEDS_CLARIFICATION: <question>` when the task is too vague or
/// contradicts the file to execute without guessing. Clarification
/// requests short-circuit the entire repair retry loop — the whole point
/// is to stop burning attempts on guesses.
///
/// The pre-plan model's verdict for one iteration against the current
/// file state. Every round, the inner model sees the task + current file
/// + prior-attempt history and must pick one of these outputs.
///
/// `Continue` carries `dropped` so the executor can report overlapping
/// steps as failed steps in the per-step output instead of as opaque
/// warnings.
enum PreplanOutcome {
    /// More edits needed. The model emitted edit-plan blocks that should
    /// be executed against the current file state.
    Continue {
        steps: Vec<EditPlanStep>,
        dropped: Vec<DroppedStep>,
    },
    /// The model says the task is satisfied by the current file state.
    /// Either nothing needed to change (first iteration), or prior
    /// iterations already completed the task. Terminal verdict — the
    /// retry loop stops and reports success.
    Complete,
    /// The model decided the task CANNOT be completed — e.g. it requires
    /// cross-file changes, it contradicts file invariants, or prior
    /// attempts hit an obstacle the model can't plan around. Carries a
    /// short reason from the model itself. Terminal verdict — the retry
    /// loop stops and reports the failure.
    Failed(String),
    /// The pre-plan model decided the task is too ambiguous, under-specified,
    /// or contradictory to act on. Carries the model's question back to the
    /// caller so the outer agent can rephrase or split the task.
    NeedsClarification(String),
    /// The model returned empty or whitespace-only text. Treated as a
    /// transient pathology (stalled inference, template misfire) and
    /// retried with explicit feedback via `RepairContext`.
    EmptyResponse,
    /// The model emitted something that `parse_edit_plan` rejected
    /// (unknown token, missing `OLD:` after we removed the no-OLD
    /// shortcut, malformed SCOPE, etc). Carry the parser's error
    /// verbatim so the repair prompt can tell the model what to fix.
    ParseError(String),
}

/// What the whole pre-plan retry loop produced. The outer caller wires
/// each variant to a one-line agent-facing message, no per-step trail.
enum PreplanResult {
    /// Task is done; file content should be written to disk.
    Applied(String),
    /// Task is already satisfied; file was not modified.
    NothingToDo,
    /// Task could not be completed; carries a short reason (from the
    /// inner model if available, or from the retry loop on exhaustion).
    Failed(String),
    /// Task is too vague to act on; the agent must rephrase.
    NeedsClarification(String),
}

struct PlannedExecutionFailure {
    current_content: String,
    message: String,
    error: String,
    /// Steps from the plan that already applied successfully to
    /// `current_content`. Recorded in execution order (descending source
    /// line). Empty when the very first step blew up.
    completed_steps: Vec<EditPlanStep>,
    /// The step that failed, if any. `None` means the plan executed
    /// fully but then post-validation (e.g. LSP) rejected the result, so
    /// every step in `completed_steps` succeeded individually.
    failed_step: Option<EditPlanStep>,
    /// When the failure is an LSP regression, structured information
    /// about each error location plus the post-edit file snapshot so
    /// the replanner can see *which lines* of its patch produced the
    /// new errors instead of reading a truncated error blob.
    lsp_regression: Option<LspRegression>,
}

/// One diagnostic entry carried inside `LspRegression`. Lines and columns
/// are 1-based, matching the `file:line:col` rendering used throughout
/// the tool output.
#[derive(Clone, Debug)]
struct LspErrorLocation {
    line: usize,
    column: usize,
    message: String,
}

/// Structured LSP regression captured when post-edit validation fails.
/// Unlike the opaque error string, this carries the broken candidate
/// content so the repair prompt can show the planner the *exact lines*
/// its patch produced around every new error.
#[derive(Clone, Debug)]
struct LspRegression {
    baseline_count: usize,
    errors: Vec<LspErrorLocation>,
    /// Extra suffix "... and N more error(s)" — tracked separately so
    /// the rendered block can repeat the truncation note instead of
    /// losing it.
    extra_error_count: usize,
    /// The candidate file content at the moment validation ran. Used
    /// by `format_lsp_regression_for_planner` to extract ±5-line
    /// snippets around each error location.
    candidate_content: String,
}

/// Validation outcome when `validate_candidate_for_write` (or its LSP
/// sub-step) rejects a candidate. Separating `LspRegression` from
/// `Other` lets the retry loop capture structured diagnostic info to
/// feed back into the planner, while still allowing non-LSP failures
/// (truncation gate, IO errors, …) to surface as opaque errors.
enum ValidationError {
    LspRegression(LspRegression),
    Other(anyhow::Error),
}

/// Structured information passed to `request_preplan_steps` describing
/// the most recent iteration's attempt. The planner sees it at the top
/// of every verdict prompt so every round is explicitly aware of what
/// was just tried and what happened.
///
/// Used both for failure context (step blew up, LSP regression, parse
/// error) and for success context (attempt applied cleanly, time to
/// decide if the task is now done). `cleanly_applied` flips the
/// rendering between "failure — plan the recovery" and "success —
/// decide COMPLETE or CONTINUE".
struct RepairContext {
    previous_plan: Vec<EditPlanStep>,
    completed_steps: Vec<EditPlanStep>,
    failed_step: Option<EditPlanStep>,
    failure_reason: String,
    /// When the failure was an LSP regression, carries the structured
    /// diagnostic info so `format_repair_context` can render post-edit
    /// snippets around each error instead of a truncated blob.
    lsp_regression: Option<LspRegression>,
    /// True when the previous iteration applied cleanly (no step failed,
    /// no validation regression). In that case the verdict prompt should
    /// frame the prior attempt as progress toward the task, not as a
    /// failure — the model is being asked to decide whether the task is
    /// now done (COMPLETE) or still needs more work (CONTINUE).
    cleanly_applied: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LspValidationMode {
    Auto,
    Require,
    Off,
}

struct PatchResponse {
    ops: Vec<PatchOp>,
    output: String,
    raw_text: String,
}

#[derive(Debug)]
struct PreplanWindowResponse {
    notes: Vec<String>,
    commands: Vec<InspectionCommand>,
    /// Set to `Some(question)` if any line in the response is a
    /// `NEEDS_CLARIFICATION:` sentinel. Short-circuits the windowed pass:
    /// the model already knows the task is ambiguous and there's no point
    /// scanning more slices.
    clarification: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum InspectionCommand {
    Search(String),
    Read { start: usize, end: usize },
}

/// Per-edit totals for SEARCH and READ so that the batch inspection pass
/// can enforce the same caps whether the model requested one command or
/// many across several windows.
struct InspectionCounters {
    search_count: usize,
    read_count: usize,
    max_reads: usize,
}
