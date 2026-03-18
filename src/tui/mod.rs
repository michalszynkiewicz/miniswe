//! Terminal UI for miniswe.
//!
//! Provides a simple streaming output display for the agent loop.
//! Phase 1: Basic terminal output with colors + spinner.
//! Phase 2 (future): Full ratatui TUI with panels.

use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

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

/// Read a line of input from the user.
pub fn read_input(prompt: &str) -> Option<String> {
    eprint!("\x1b[1;35m{prompt}\x1b[0m ");
    io::stderr().flush().ok();

    let mut input = String::new();
    match io::stdin().read_line(&mut input) {
        Ok(0) => None, // EOF
        Ok(_) => Some(input.trim().to_string()),
        Err(_) => None,
    }
}

/// A spinner that shows activity while waiting for the LLM.
/// Call `start()` before the LLM request, `stop()` when tokens start flowing.
pub struct Spinner {
    running: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Spinner {
    /// Start the spinner with a label.
    pub fn start(label: &str) -> Self {
        let running = Arc::new(AtomicBool::new(true));
        let running_clone = running.clone();
        let label = label.to_string();

        let handle = std::thread::spawn(move || {
            let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
            let mut i = 0;
            while running_clone.load(Ordering::Relaxed) {
                eprint!("\r\x1b[2m{} {}\x1b[0m", frames[i % frames.len()], label);
                io::stderr().flush().ok();
                std::thread::sleep(std::time::Duration::from_millis(80));
                i += 1;
            }
            // Clear the spinner line
            eprint!("\r\x1b[2K");
            io::stderr().flush().ok();
        });

        Self {
            running,
            handle: Some(handle),
        }
    }

    /// Stop the spinner.
    pub fn stop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for Spinner {
    fn drop(&mut self) {
        self.stop();
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max - 3])
    }
}
