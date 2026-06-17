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
    // Recursion guard: the check typically runs the project's own binary,
    // which may itself be a miniswe build that would re-enter this gate (and
    // re-build, and re-run …). Nested invocations set this env var to opt out.
    if std::env::var_os("MINISWE_SKIP_VALIDATION").is_some() {
        return CheckOutcome::Skipped;
    }
    let Some(cmd) = config.validation.command() else {
        return CheckOutcome::Skipped;
    };
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
