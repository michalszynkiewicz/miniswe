//! Whole-tree SCRAP restart and single-file REWIND messaging — shared by
//! both loops' debugger-recovery and gate-restart blocks, unified behind
//! [`AgentUi`] so the two near-identical copies (`repl/support.rs`,
//! `run/support.rs`) collapse into one.

use std::sync::Arc;

use parking_lot::Mutex;

use crate::cli::commands::agent::ui::AgentUi;
use crate::cli::commands::agent::validation;
use crate::config::Config;
use crate::context;
use crate::llm::Message;
use crate::lsp::LspClient;
use crate::tools;
use crate::tools::permissions::PermissionManager;

/// Execute the debugger's proposed single-file rewind (`debugger_judge_rewind`)
/// and build the message to inject afterward. Best-effort: on failure the tree
/// is left as-is and the model just sees the original verification failure.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn rewind_message(
    ui: &mut impl AgentUi,
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
        ui.status(&format!(
            "[debugger-judge] REWIND — reverted {} to rev_{} (file_errors {} → {})",
            candidate.path, candidate.rev, candidate.file_errors_now, candidate.file_errors_then
        ));
        // REWIND fixes the ONE regressed file the debugger flagged — it doesn't
        // mean the gate's original failure is fully resolved (the behavioral
        // check may still be failing for an unrelated reason elsewhere). Point
        // at the raw check output so the model isn't left guessing what
        // "the remaining problem" actually is from the rewind summary alone.
        let output_note = validation::write_gate_failure_output(config, output)
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
        ui.status(&format!(
            "[debugger-judge] REWIND — revert of {} to rev_{} failed; continuing without it",
            candidate.path, candidate.rev
        ));
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
///
/// `plan_only` is `context::assemble`'s plan-only flag; the REPL always
/// passes `false`.
pub(crate) fn scrap_restart(
    ui: &mut impl AgentUi,
    config: &Config,
    task: &str,
    mcp_summary: Option<&str>,
    snapshots: &Option<Arc<Mutex<tools::snapshots::SnapshotManager>>>,
    plan_only: bool,
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
            Ok(m) => ui.status(&format!("{ok_prefix}{m}")),
            Err(e) => ui.status(&format!("{err_prefix}{e}")),
        }
    }
    // Whole-tree revert touched many files outside the per-edit reindex path —
    // resync the symbol index / repo-map to the clean baseline.
    tools::reindex_project_incremental(config);
    let _ = std::fs::remove_file(config.session_path("plan.md"));
    let _ = std::fs::remove_file(config.session_path("scratchpad.md"));
    let assembled = context::assemble(config, task, &[], plan_only, mcp_summary);
    ui.status(done);
    assembled.messages
}
