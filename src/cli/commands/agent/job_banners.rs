//! Background-job banner tracking: parses `[job N FINISHED|FAILED]  $ <cmd>`
//! lines out of a tool result and keeps a map of currently-failing
//! commands, so a repeated failing background job (e.g. a re-launched
//! `pkg run dev` deploy) can be routed to the debugger by the loop-recovery
//! ladder — the same way a foreground command that keeps failing does.

/// Normalize a shell command for cross-invocation matching: drop a leading
/// `cd … &&` working-dir prefix and collapse whitespace, so the same
/// underlying command matches regardless of where it was launched from.
fn normalize_command(cmd: &str) -> String {
    let tail = cmd.rsplit("&&").next().unwrap_or(cmd).trim();
    tail.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The normalized command out of a single job banner line
/// (`[job N FINISHED|FAILED]  $ <cmd>`).
fn job_banner_command(line: &str) -> Option<String> {
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
pub(crate) fn note_job_banners(
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
pub(crate) fn failing_job_output<'a>(
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
