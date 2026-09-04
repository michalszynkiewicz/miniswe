//! Behavioral "done-gate": run a configured command that exercises the
//! feature end-to-end before the agent is allowed to finish.
//!
//! This catches the documented failure where a value is *plumbed* through
//! signatures (so it compiles and tests pass) but never *consumed* at
//! runtime — the change looks done on every structural signal yet doesn't
//! actually work. See `docs/success-validation-design.md`.
//!
//! Disabled by default (`[validation] command` empty) — a no-op unless a
//! project/bench opts in, so it cannot regress existing behavior.

use std::time::Duration;

use crate::config::Config;

/// Outcome of a behavioral validation run.
pub enum CheckOutcome {
    /// Command exited 0 — the feature works; allow completion.
    Pass,
    /// Command exited non-zero — block completion; carries combined output.
    Fail(String),
    /// No command configured, or it could not be run / timed out — don't block.
    Skipped,
}

/// Run the configured behavioral check in the project root.
///
/// `Skipped` (no command, spawn failure, or timeout) never blocks the agent —
/// the gate is best-effort and must degrade to the prior behavior.
pub async fn run_behavioral_check(config: &Config) -> CheckOutcome {
    let Some(cmd) = config.validation.command() else {
        return CheckOutcome::Skipped;
    };
    run_check_command(config, cmd).await
}

/// Run an explicit check command in the project root (shared by the
/// configured task-level gate and the per-skill-step completion check —
/// see `skill_cursor::current_check_command`). Same best-effort contract:
/// spawn failure or timeout → `Skipped`, never a block.
pub async fn run_check_command(config: &Config, cmd: &str) -> CheckOutcome {
    // Recursion guard: the check typically runs the project's own binary,
    // which may itself be a miniswe build that would re-enter this gate (and
    // re-build, and re-run …). Nested invocations set this env var to opt out.
    if std::env::var_os("MINISWE_SKIP_VALIDATION").is_some() {
        return CheckOutcome::Skipped;
    }
    let timeout = Duration::from_secs(config.validation.timeout_secs);
    let mut command = tokio::process::Command::new("sh");
    command
        .arg("-c")
        .arg(cmd)
        .current_dir(&config.project_root)
        .stdin(std::process::Stdio::null());

    let output = match tokio::time::timeout(timeout, command.output()).await {
        Ok(Ok(o)) => o,
        Ok(Err(e)) => {
            tracing::warn!("behavioral check failed to spawn: {e}");
            return CheckOutcome::Skipped;
        }
        Err(_) => {
            tracing::warn!(
                "behavioral check timed out after {}s — not blocking",
                timeout.as_secs()
            );
            return CheckOutcome::Skipped;
        }
    };

    if output.status.success() {
        CheckOutcome::Pass
    } else {
        let mut combined = String::from_utf8_lossy(&output.stdout).into_owned();
        if !output.stderr.is_empty() {
            if !combined.is_empty() {
                combined.push('\n');
            }
            combined.push_str(&String::from_utf8_lossy(&output.stderr));
        }
        let combined = crate::truncate_chars(combined.trim(), config.tool_output_budget_chars());
        CheckOutcome::Fail(combined)
    }
}

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

/// Per-turn behavioral-gate state, shared by both agent loops. Plain
/// `pub(crate)` fields (no accessors): call sites reset different subsets at
/// different points and the split borrows must keep working.
#[derive(Default)]
pub(crate) struct GateState {
    /// How many times the behavioral done-gate has blocked completion this turn.
    pub(crate) validation_blocks: usize,
    /// The model's stated rationale each time the gate blocked it — so a model
    /// that believes the check is wrong has an auditable voice (bounded by
    /// max_retries; never a silent free pass).
    pub(crate) validation_disputes: Vec<String>,
    /// The reactive replan/restart sub-agents fire at most once per turn.
    pub(crate) replan_fired: bool,
    pub(crate) restart_fired: bool,
    /// Gate-triggered context resets fired this turn (bounded — don't loop).
    pub(crate) context_resets: usize,
    /// `tools.plan_gate_debugger`: consecutive plan(check) failures on the SAME
    /// step. Distinct from `validation_blocks` (the behavioral done-gate) — this
    /// is the plan tool's OWN compile gate repeatedly blocking one step.
    pub(crate) plan_step_failures: PlanStepFailures,
}

/// Consecutive `plan(action='check')` failures on the SAME step
/// (`tools.plan_gate_debugger`'s trigger). A failure on a step other than the
/// last-failed one (or the first ever) resets the streak to 1 — only
/// *repeated* blocking on one step signals a stall worth escalating.
#[derive(Default)]
pub(crate) struct PlanStepFailures {
    last_failed_step: Option<u64>,
    streak: u32,
}

impl PlanStepFailures {
    /// Record a plan-check failure on `step`.
    pub(crate) fn note(&mut self, step: u64) {
        if self.last_failed_step == Some(step) {
            self.streak += 1;
        } else {
            self.last_failed_step = Some(step);
            self.streak = 1;
        }
    }

    /// Consecutive failures on the current step.
    pub(crate) fn streak(&self) -> u32 {
        self.streak
    }

    /// Clear the streak (plan-check success, or a tree/context reset).
    pub(crate) fn reset(&mut self) {
        self.last_failed_step = None;
        self.streak = 0;
    }
}

#[cfg(test)]
mod plan_step_failures_tests {
    use super::PlanStepFailures;

    #[test]
    fn same_step_repeated_increments_streak() {
        let mut f = PlanStepFailures::default();
        f.note(1);
        assert_eq!(f.streak(), 1);
        f.note(1);
        assert_eq!(f.streak(), 2);
        f.note(1);
        assert_eq!(f.streak(), 3);
    }

    #[test]
    fn different_step_resets_streak_to_one() {
        let mut f = PlanStepFailures::default();
        f.note(1);
        f.note(1);
        assert_eq!(f.streak(), 2);
        f.note(2);
        assert_eq!(f.streak(), 1);
    }

    #[test]
    fn first_failure_ever_sets_streak_to_one() {
        let mut f = PlanStepFailures::default();
        f.note(7);
        assert_eq!(f.streak(), 1);
    }

    #[test]
    fn reset_clears_streak_and_step_memory() {
        let mut f = PlanStepFailures::default();
        f.note(3);
        f.note(3);
        f.reset();
        assert_eq!(f.streak(), 0);
        // After a reset the next failure on the same step starts a NEW streak.
        f.note(3);
        assert_eq!(f.streak(), 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(cmd: &str) -> Config {
        let mut c = Config::default();
        c.validation.command = cmd.to_string();
        c.validation.timeout_secs = 10;
        c
    }

    #[tokio::test]
    async fn passing_command_is_pass() {
        assert!(matches!(
            run_behavioral_check(&cfg("true")).await,
            CheckOutcome::Pass
        ));
    }

    #[tokio::test]
    async fn failing_command_is_fail_with_output() {
        match run_behavioral_check(&cfg("echo nope 1>&2; exit 1")).await {
            CheckOutcome::Fail(s) => assert!(s.contains("nope"), "output was: {s:?}"),
            _ => panic!("expected Fail"),
        }
    }

    #[tokio::test]
    async fn empty_command_is_skipped() {
        assert!(matches!(
            run_behavioral_check(&Config::default()).await,
            CheckOutcome::Skipped
        ));
    }
}
