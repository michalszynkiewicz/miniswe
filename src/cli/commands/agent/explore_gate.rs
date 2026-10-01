//! Runtime read-only guard for explore (investigation) mode, shared by the
//! REPL's explore turns today and by [`super::turn::call_gate::admit`] once
//! headless gains an equivalent mode. Moved verbatim from `repl/explore.rs`
//! (pure functions, no REPL-specific state) so both call sites can use them.

use super::hints::is_file_write;

/// Conservative read-only shell classifier for explore mode. Returns true ONLY
/// when every command in the (possibly compound) line is a known read-only
/// command with no output redirection, command substitution, or mutating flags.
/// Errs toward `false` (block) on anything unrecognized. NB: a heuristic — the
/// load-bearing rule is "unknown ⇒ block", plus hard-rejecting write constructs.
pub(crate) fn shell_is_read_only(command: &str) -> bool {
    let cmd = command.trim();
    if cmd.is_empty() {
        return false;
    }
    // Allow the two common stderr redirects; any remaining `>` (or process
    // substitution) is a real write → block.
    let scrub = cmd
        .replace("2>/dev/null", "")
        .replace("2>&1", "")
        .replace(">/dev/null", "");
    const DANGER: &[&str] = &[
        ">",
        "$(",
        "`",
        "<(",
        ">(",
        "&>",
        "|&",
        " -exec",
        " -execdir",
        " -delete",
        " -ok",
        " -fprint",
        "xargs",
        "eval ",
        "source ",
        "sudo ",
        "chmod",
        "chown",
        "tee ",
    ];
    if DANGER.iter().any(|d| scrub.contains(d)) {
        return false;
    }
    if cmd.contains("sed") && (cmd.contains(" -i") || cmd.contains("--in-place")) {
        return false;
    }
    const READ_CMDS: &[&str] = &[
        "ls", "cat", "head", "tail", "grep", "egrep", "fgrep", "rg", "ag", "find", "fd", "wc",
        "stat", "file", "tree", "pwd", "echo", "printf", "sort", "uniq", "cut", "tr", "awk", "sed",
        "which", "type", "basename", "dirname", "realpath", "readlink", "du", "df", "env",
        "printenv", "date", "whoami", "hostname", "uname", "nl", "tac", "column", "jq", "yq",
        "xxd", "od", "strings", "diff", "cmp", "comm", "less", "more", "true", "test", "cd",
    ];
    const GIT_READ: &[&str] = &[
        "status",
        "log",
        "diff",
        "show",
        "branch",
        "ls-files",
        "ls-tree",
        "blame",
        "describe",
        "rev-parse",
        "cat-file",
        "grep",
        "shortlog",
        "reflog",
        "remote",
        "config",
        "tag",
        "whatchanged",
        "name-rev",
    ];
    let normalized = scrub
        .replace("&&", "\n")
        .replace("||", "\n")
        .replace([';', '|', '&'], "\n");
    for seg in normalized.lines() {
        let mut toks = seg.split_whitespace().peekable();
        // skip leading VAR=val env assignments
        while toks
            .peek()
            .is_some_and(|t| t.contains('=') && !t.starts_with('-'))
        {
            toks.next();
        }
        let Some(c0) = toks.next() else { continue };
        let base = c0.rsplit('/').next().unwrap_or(c0);
        if base == "git" {
            if !GIT_READ.contains(&toks.next().unwrap_or("")) {
                return false;
            }
        } else if !READ_CMDS.contains(&base) {
            return false;
        }
    }
    true
}

/// In read-only (explore) mode, decide whether a tool call must be blocked.
/// `Some(reason)` ⇒ mutating, block it; `None` ⇒ read-only, allow. This is the
/// load-bearing runtime guard: the tool-def filter and the prompt are advisory,
/// but shell can mutate and the model can emit tools that aren't in the list.
pub(crate) fn explore_block_reason(
    name: &str,
    file_action: &str,
    args: &serde_json::Value,
) -> Option<String> {
    if is_file_write(name) || matches!(name, "revert" | "delete_file" | "spawn_agents") {
        return Some(format!("`{name}` can modify files"));
    }
    if name == "shell" {
        let cmd = args["command"].as_str().unwrap_or("");
        if args["action"].as_str() == Some("run") && !shell_is_read_only(cmd) {
            return Some(format!(
                "shell command is not read-only: `{}`",
                crate::truncate_chars(cmd, 80)
            ));
        }
    }
    if name == "file" {
        match file_action {
            "shell" => {
                let cmd = args["command"].as_str().unwrap_or("");
                if !shell_is_read_only(cmd) {
                    return Some(format!(
                        "shell command is not read-only: `{}`",
                        crate::truncate_chars(cmd, 80)
                    ));
                }
            }
            a if is_file_write(a) || matches!(a, "write_file" | "delete" | "delete_file") => {
                return Some(format!("file action `{a}` can modify files"));
            }
            _ => {}
        }
    }
    None
}
