//! Gate-failure bookkeeping: stable failure signatures, per-step failure
//! streak tracking, and the on-disk gate-failure pointer.

use super::*;

/// Stable signature of a gate failure, used by `debugger_multifire` to decide
/// whether the failure CHANGED since the last diagnosis (so the debugger walks
/// compile→smoke rather than re-diagnosing the same failure). Using only the
/// first line breaks this for commands that wrap every failure in a constant
/// banner (e.g. `echo "DOES NOT COMPILE:"; echo "$out" | tail -20`) — every
/// compile error then hashes to the identical string, so multifire silently
/// refuses to refire when the underlying error actually changed. Join enough
/// of the (non-empty) output to reach past such banners into the real detail.
pub(crate) fn failure_key(output: &str) -> String {
    let joined = output
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    joined.to_ascii_lowercase().chars().take(400).collect()
}

#[cfg(test)]
mod failure_key_tests {
    use super::failure_key;

    #[test]
    fn distinguishes_different_errors_behind_the_same_banner() {
        let a = "DOES NOT COMPILE:\nerror[E0599]: no method named `call_chat`\n --> src/cli/commands/run.rs:222:41";
        let b = "DOES NOT COMPILE:\nerror[E0382]: borrow of partially moved value: `system_prompt_override`\n --> src/context/mod.rs:306:8";
        assert_ne!(failure_key(a), failure_key(b));
    }

    #[test]
    fn treats_the_same_error_as_the_same_key() {
        let a = "DOES NOT COMPILE:\nerror[E0382]: borrow of partially moved value: `x`\n --> src/context/mod.rs:306:8";
        let b = "DOES NOT COMPILE:\nerror[E0382]: borrow of partially moved value: `x`\n --> src/context/mod.rs:306:8";
        assert_eq!(failure_key(a), failure_key(b));
    }

    #[test]
    fn ignores_blank_lines_and_case() {
        let a = "\n\nDOES NOT COMPILE:\n\nError[E0308]\n";
        let b = "DOES NOT COMPILE:\nerror[e0308]";
        assert_eq!(failure_key(a), failure_key(b));
    }
}

/// Track consecutive `plan(action='check')` failures on the SAME step
/// (`tools.plan_gate_debugger`'s trigger). A failure on a step other than the
/// last-failed one (or the first ever) resets the streak to 1 — only
/// *repeated* blocking on one step signals a stall worth escalating.
pub(crate) fn track_plan_step_failure(
    last_failed_step: &mut Option<u64>,
    streak: &mut u32,
    step: u64,
) {
    if *last_failed_step == Some(step) {
        *streak += 1;
    } else {
        *last_failed_step = Some(step);
        *streak = 1;
    }
}

#[cfg(test)]
mod job_failure_tests {
    use super::{failing_job_output, job_banner_command, normalize_command, note_job_banners};

    #[test]
    fn normalize_strips_cd_prefix_and_whitespace() {
        assert_eq!(
            normalize_command("cd demo-e2e-task-package &&  pkg   run dev"),
            "pkg run dev"
        );
        assert_eq!(normalize_command("pkg run dev"), "pkg run dev");
    }

    #[test]
    fn job_banner_command_extracts_normalized_command() {
        assert_eq!(
            job_banner_command("[job 1 FAILED]  $ cd pkg && pkg run dev").as_deref(),
            Some("pkg run dev")
        );
        assert_eq!(job_banner_command("just some output"), None);
    }

    #[test]
    fn failed_banner_records_and_clean_finish_clears() {
        let mut failed = std::collections::HashMap::new();
        note_job_banners(
            "[job 1 FAILED]  $ cd pkg && pkg run dev\n[shell: exit 1]\nunable to pull chart",
            &mut failed,
        );
        assert!(failed["pkg run dev"].contains("unable to pull chart"));
        // Same command later finishes cleanly → stale failure cleared.
        note_job_banners("[job 2 FINISHED]  $ pkg run dev\nok", &mut failed);
        assert!(failed.is_empty());
    }

    #[test]
    fn aggregate_result_with_mixed_banners_attributes_per_job() {
        // One result carrying two banners (aggregate status): only the FAILED
        // job's command must be recorded, regardless of result-level success.
        let mut failed = std::collections::HashMap::new();
        note_job_banners(
            "[job 1 FINISHED]  $ ls\nok\n[job 2 FAILED]  $ pkg run dev\n[shell: exit 128]",
            &mut failed,
        );
        assert!(!failed.contains_key("ls"));
        assert!(failed.contains_key("pkg run dev"));
    }

    #[test]
    fn failing_job_output_matches_normalized_shell_command() {
        let mut failed = std::collections::HashMap::new();
        failed.insert(
            "pkg run dev".to_string(),
            "exit 128: bad chart ref".to_string(),
        );
        let args = serde_json::json!({"action": "run", "command": "cd pkg && pkg run dev"});
        let hit = failing_job_output("shell", &args, &failed);
        assert_eq!(hit.map(|(c, _)| c), Some("pkg run dev".to_string()));
        // non-shell tool, or unrecorded command → no match
        assert!(failing_job_output("file", &args, &failed).is_none());
        let other = serde_json::json!({"action": "run", "command": "ls"});
        assert!(failing_job_output("shell", &other, &failed).is_none());
    }
}

#[cfg(test)]
mod track_plan_step_failure_tests {
    use super::track_plan_step_failure;

    #[test]
    fn same_step_repeated_increments_streak() {
        let mut last = None;
        let mut streak = 0;
        track_plan_step_failure(&mut last, &mut streak, 1);
        assert_eq!(streak, 1);
        track_plan_step_failure(&mut last, &mut streak, 1);
        assert_eq!(streak, 2);
        track_plan_step_failure(&mut last, &mut streak, 1);
        assert_eq!(streak, 3);
        assert_eq!(last, Some(1));
    }

    #[test]
    fn different_step_resets_streak_to_one() {
        let mut last = None;
        let mut streak = 0;
        track_plan_step_failure(&mut last, &mut streak, 1);
        track_plan_step_failure(&mut last, &mut streak, 1);
        assert_eq!(streak, 2);
        track_plan_step_failure(&mut last, &mut streak, 2);
        assert_eq!(streak, 1);
        assert_eq!(last, Some(2));
    }

    #[test]
    fn first_failure_ever_sets_streak_to_one() {
        let mut last = None;
        let mut streak = 0;
        track_plan_step_failure(&mut last, &mut streak, 7);
        assert_eq!(streak, 1);
        assert_eq!(last, Some(7));
    }
}

/// Execute the debugger's proposed single-file rewind (`tools.
/// Persist the behavioral gate's raw failure output (stdout+stderr from the
/// configured validation command) to a file the model can `read` directly.
/// The reactive debugger's Report/Rewind verdicts summarize this output in
/// their own words before handing it to the primary agent — a paraphrase
/// that can lose precision the raw text had (e.g. the exact "GOT: <value>"
/// line pointing at which callsite is actually broken). Writing the raw
/// text alongside the summary, rather than instead of it, lets the model
/// fall back to ground truth when the summary steers it wrong. Best-effort:
/// a write failure just means no pointer gets appended, never a hard error.
pub(crate) fn write_gate_failure_output(config: &Config, output: &str) -> Option<String> {
    let rel = "last_gate_failure.txt";
    std::fs::write(config.miniswe_path(rel), output).ok()?;
    Some(format!(".miniswe/{rel}"))
}
