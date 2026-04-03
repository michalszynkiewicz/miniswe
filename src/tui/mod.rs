//! Terminal UI for miniswe.
//!
//! Two modes:
//! - Non-interactive (run.rs): uses print_* functions with ANSI codes
//! - Interactive (repl.rs): uses ratatui TUI with split panes

pub mod app;
pub mod event;
pub mod ui;

use std::io::{self, Write};

// === Non-interactive output functions (used by run.rs and headless mode) ===

/// Print a styled header.
pub fn print_header(text: &str) {
    eprintln!("\x1b[1;36m┌─ miniswe ─────────────────────────────────────────┐\x1b[0m");
    eprintln!("\x1b[1;36m│\x1b[0m {:<49}\x1b[1;36m│\x1b[0m", text);
    eprintln!("\x1b[1;36m└─────────────────────────────────────────────────────┘\x1b[0m");
}

/// Print a tool call notification.
pub fn print_tool_call(name: &str, args_summary: &str) {
    eprintln!(
        "\x1b[33m  → {}\x1b[0m({})",
        name,
        truncate(args_summary, 60)
    );
}

/// Print a tool result summary.
pub fn print_tool_result(name: &str, success: bool, summary: &str) {
    let icon = if success { "✓" } else { "✗" };
    let color = if success { "32" } else { "31" };
    eprintln!(
        "\x1b[{color}m  {icon} {name}\x1b[0m: {}",
        truncate(summary, 70)
    );
}

/// Print streaming token output.
pub fn print_token(token: &str) {
    print!("{token}");
    io::stdout().flush().ok();
}

/// Print a status message.
pub fn print_status(msg: &str) {
    eprintln!("\x1b[2m{msg}\x1b[0m");
}

/// Print an error message.
pub fn print_error(msg: &str) {
    eprintln!("\x1b[1;31merror\x1b[0m: {msg}");
}

/// Print completion message.
pub fn print_complete(msg: &str) {
    eprintln!("\x1b[1;32m✓ {msg}\x1b[0m");
}

/// Print a separator.
pub fn print_separator() {
    eprintln!("\x1b[2m──────────────────────────────────────────────────\x1b[0m");
}

/// Read a line of input from the user (for permission prompts).
pub fn read_input(prompt: &str) -> Option<String> {
    // Ensure terminal is in cooked mode for line input
    let _ = crossterm::terminal::disable_raw_mode();

    eprint!("\x1b[1;35m{prompt}\x1b[0m ");
    io::stderr().flush().ok();

    let mut input = String::new();
    match io::stdin().read_line(&mut input) {
        Ok(0) => None,
        Ok(_) => Some(input.trim().to_string()),
        Err(_) => None,
    }
}

fn truncate(s: &str, max: usize) -> String {
    crate::truncate_chars(s, max.saturating_sub(3))
}
