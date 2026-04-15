//! Verify that a candidate LSP binary actually runs. Catches the common
//! case where `rust-analyzer` is on `PATH` only as a rustup proxy for a
//! toolchain that doesn't have the component installed.

use std::path::Path;
use std::process::Command;

/// Result of verifying that a binary on disk actually runs.
#[derive(Debug, PartialEq, Eq)]
pub enum VerifyResult {
    Ok,
    Failed { reason: String },
}

/// Verify a binary actually works by running it with version args.
/// Captures stderr/stdout so failures can be explained instead of swallowed.
pub fn verify_binary_verbose(path: &Path, args: &[&str]) -> VerifyResult {
    let output = match Command::new(path)
        .args(args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
    {
        Ok(out) => out,
        Err(e) => {
            return VerifyResult::Failed {
                reason: format!("spawn failed: {e}"),
            };
        }
    };

    if output.status.success() {
        return VerifyResult::Ok;
    }

    // Surface the first useful line of stderr so the user can diagnose
    // (rustup proxy errors look like: "error: 'rust-analyzer' is not installed for the toolchain ...").
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let snippet = stderr
        .lines()
        .chain(stdout.lines())
        .find(|l| !l.trim().is_empty())
        .unwrap_or("no output")
        .trim()
        .chars()
        .take(160)
        .collect::<String>();

    let exit = output
        .status
        .code()
        .map(|c| format!("exit {c}"))
        .unwrap_or_else(|| "killed by signal".to_string());

    VerifyResult::Failed {
        reason: format!("{exit}: {snippet}"),
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    // We deliberately do *not* write our own scripts and exec them
    // here. Doing so in a multi-threaded `cargo test` run is racy on
    // Linux: a concurrent fork in another test can inherit our write
    // FD and trip `ETXTBSY` when the child tries to exec. Driving a
    // pre-existing shell (`/bin/sh -c '…'`) skips the write step
    // entirely and the tests become completely deterministic.

    #[test]
    fn verify_binary_ok_on_zero_exit() {
        // `sh -c 'exit 0'` is the smallest "binary runs and returns
        // success" shape we can construct without creating files.
        assert_eq!(
            verify_binary_verbose(Path::new("/bin/sh"), &["-c", "echo 1.2.3; exit 0"]),
            VerifyResult::Ok
        );
    }

    #[test]
    fn verify_binary_failed_captures_stderr_snippet() {
        // Mimic a rustup proxy failure: nonzero exit + stderr line
        // that the user will want to see verbatim in logs.
        let rustup_msg = "error: 'rust-analyzer' is not installed for the toolchain 'stable-x86_64-unknown-linux-gnu'";
        match verify_binary_verbose(
            Path::new("/bin/sh"),
            &["-c", &format!("echo \"{rustup_msg}\" >&2; exit 1")],
        ) {
            VerifyResult::Ok => panic!("expected failure"),
            VerifyResult::Failed { reason } => {
                assert!(reason.contains("exit 1"), "reason was: {reason}");
                assert!(
                    reason.contains("rust-analyzer") && reason.contains("not installed"),
                    "reason was: {reason}"
                );
            }
        }
    }

    #[test]
    fn verify_binary_failed_when_missing_executable() {
        let result = verify_binary_verbose(
            Path::new("/nonexistent/definitely-missing-binary"),
            &["--version"],
        );
        match result {
            VerifyResult::Ok => panic!("expected failure for missing binary"),
            VerifyResult::Failed { reason } => {
                assert!(reason.starts_with("spawn failed"), "reason was: {reason}");
            }
        }
    }

    #[test]
    fn verify_binary_truncates_long_lines() {
        // 500 chars of 'x' on stderr — the snippet that ends up in
        // the reason string should be truncated to ≤160 chars.
        match verify_binary_verbose(
            Path::new("/bin/sh"),
            &["-c", "printf 'x%.0s' $(seq 1 500) >&2; exit 2"],
        ) {
            VerifyResult::Ok => panic!("expected failure"),
            VerifyResult::Failed { reason } => {
                // exit prefix + ": " + up-to-160-char snippet
                assert!(reason.starts_with("exit 2: "), "reason was: {reason}");
                let snippet = &reason["exit 2: ".len()..];
                assert!(snippet.len() <= 160, "snippet too long: {}", snippet.len());
            }
        }
    }
}
