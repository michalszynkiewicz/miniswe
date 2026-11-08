//! Built-in REPL commands: classification and `/help` text.

/// `quit`, `exit` or `/quit` — leave the REPL.
pub(super) fn is_quit_command(input: &str) -> bool {
    matches!(input, "quit" | "exit" | "/quit")
}

/// Commands handled by the REPL itself (everything but a model turn or a
/// skill invocation, which echo on their own).
pub(super) fn is_builtin_command(input: &str) -> bool {
    matches!(
        input,
        "/help" | "/clear" | "/new" | "/skills" | "/skills list"
    ) || (input.starts_with("/skills ") && (input.ends_with(" help") || input.ends_with(" --help")))
}

/// The `/help` output: commands, then keys.
pub(super) fn help_lines() -> Vec<&'static str> {
    vec![
        "/clear — clear conversation history",
        "/new   — clear history + scratchpad + plan",
        "/skills list       — list available skills",
        "/skills <name> help — show skill details",
        "/help  — show this help",
        "quit   — exit",
        "Enter: send · Ctrl+C: interrupt · Ctrl+D or quit: exit",
        "Ctrl+O: tool-result details · PgUp/PgDn, Ctrl+Home/End: scroll",
        "Up/Down: input history",
    ]
}
