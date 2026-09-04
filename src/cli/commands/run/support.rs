//! Support helpers for the run loop: patch-path parsing, job-banner
//! tracking, recent-activity digests, rewind messaging, and bounded
//! awaiting of worker-pool and shell jobs.

use super::*;

/// Project-relative paths touched by a unified-diff patch file (parsed from
/// its `+++ b/<path>` headers). Used by replay mode to notify the LSP of files
/// changed out-of-band by `--replay-apply`. Best-effort: returns `[]` on read error.
pub(super) fn changed_paths_in_patch(patch: &std::path::Path) -> Vec<String> {
    let Ok(text) = std::fs::read_to_string(patch) else {
        return Vec::new();
    };
    text.lines()
        .filter_map(|l| l.strip_prefix("+++ b/"))
        .map(|p| p.trim().to_string())
        .filter(|p| p != "/dev/null")
        .collect()
}

/// Normalize a shell command for cross-invocation matching: drop a leading
/// `cd … &&` working-dir prefix and collapse whitespace, so the same
/// underlying command matches regardless of where it was launched from.
pub(super) fn normalize_command(cmd: &str) -> String {
    let tail = cmd.rsplit("&&").next().unwrap_or(cmd).trim();
    tail.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The normalized command out of a single job banner line
/// (`[job N FINISHED|FAILED]  $ <cmd>`).
pub(super) fn job_banner_command(line: &str) -> Option<String> {
    let cmd = line.split("$ ").nth(1)?.trim();
    (!cmd.is_empty()).then(|| normalize_command(cmd))
}

/// Apply every job banner in a tool result to the failed-command map: a
/// FAILED banner records its (normalized) command with the result text, a
/// clean FINISHED banner clears any stale failure for it. Background jobs
/// fail LATER, in a status/wait result, not on the launch call, so keying
/// their failure by command is the only way a re-launched-and-re-failing
/// deploy reaches the debugger. Parsed per banner line — an aggregate
/// status result can carry several banners with different verdicts, so the
/// wrapper's overall ok/err cannot attribute them.
pub(super) fn note_job_banners(
    content: &str,
    failed_job_commands: &mut std::collections::HashMap<String, String>,
) {
    for line in content.lines() {
        if line.contains(" FAILED]  $ ") {
            if let Some(cmd) = job_banner_command(line) {
                failed_job_commands.insert(cmd, crate::truncate_chars(content.trim(), 2000));
            }
        } else if line.contains(" FINISHED]  $ ")
            && let Some(cmd) = job_banner_command(line)
        {
            failed_job_commands.remove(&cmd);
        }
    }
}

/// The recorded failure output for a shell tool call whose (normalized)
/// command has a failing background job on record. Lets the loop-recovery
/// ladder route a repeated failing deploy to the debugger.
pub(super) fn failing_job_output<'a>(
    name: &str,
    args: &serde_json::Value,
    failed: &'a std::collections::HashMap<String, String>,
) -> Option<(String, &'a str)> {
    if name != "shell" {
        return None;
    }
    let cmd = normalize_command(args.get("command")?.as_str()?);
    failed.get(&cmd).map(|out| (cmd, out.as_str()))
}

/// Render the tail of the conversation into a compact transcript — the
/// evidence of what the model just did and observed. Fed to the skill-step
/// completion judge (see `skill_router::judge_step_done`) so it decides from
/// actual recent activity, not the whole (possibly stale) history.
pub(super) fn recent_activity(messages: &[Message], n: usize, cap: usize) -> String {
    let tail: Vec<&Message> = messages.iter().rev().take(n).collect();
    let mut lines: Vec<String> = Vec::new();
    for m in tail.into_iter().rev() {
        if let Some(c) = &m.content {
            let c = c.trim();
            if !c.is_empty() {
                lines.push(format!("[{}] {}", m.role, crate::truncate_chars(c, 500)));
            }
        }
        for tc in m.tool_calls.iter().flatten() {
            lines.push(format!(
                "[{} call] {}({})",
                m.role,
                tc.function.name,
                crate::truncate_chars(&tc.function.arguments, 160)
            ));
        }
    }
    crate::truncate_chars(&lines.join("\n"), cap)
}

/// debugger_judge_rewind`) and build the message to inject afterward.
/// Best-effort, matching the codebase's other auto-recovery guards: on
/// failure the tree is left as-is and the model just sees the original
/// verification failure, no different from a plain Report verdict.
#[allow(clippy::too_many_arguments)]
pub(super) async fn rewind_message(
    candidate: &tools::RewindCandidate,
    config: &Config,
    perms: &Arc<PermissionManager>,
    lsp_client: &Option<Arc<LspClient>>,
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
        lsp_client.as_deref(),
        revisions,
        fast_baseline_errors,
    )
    .await
    .is_ok_and(|r| r.success);

    if ok {
        tui::print_status(&format!(
            "[debugger-judge] REWIND — reverted {} to rev_{} (file_errors {} → {})",
            candidate.path, candidate.rev, candidate.file_errors_now, candidate.file_errors_then
        ));
        // REWIND fixes the ONE regressed file the debugger flagged — it doesn't
        // mean the gate's original failure is fully resolved (the behavioral
        // check may still be failing for an unrelated reason elsewhere). Point
        // at the raw check output so the model isn't left guessing what
        // "the remaining problem" actually is from the rewind summary alone.
        let output_note = write_gate_failure_output(config, output)
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
        tui::print_status(&format!(
            "[debugger-judge] REWIND — revert of {} to rev_{} failed; continuing without it",
            candidate.path, candidate.rev
        ));
        Message::user(&format!(
            "[Verification failed — do NOT finish yet. Check output:\n{output}]"
        ))
    }
}
