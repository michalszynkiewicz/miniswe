//! Session logger — writes structured logs to `.miniswe/logs/`.
//!
//! Three verbosity levels:
//! - **info**: tool calls and outcomes (one-liner per action)
//! - **debug**: full interactions — LLM messages, tool args/results, file changes
//! - **trace**: everything + context assembly stats, token counts, masking decisions

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

use crate::config::Config;

/// Log verbosity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Info,
    Debug,
    Trace,
}

impl LogLevel {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "info" => LogLevel::Info,
            "trace" => LogLevel::Trace,
            _ => LogLevel::Debug,
        }
    }
}

/// Session logger that writes to a per-session log file.
pub struct SessionLog {
    file: Mutex<Option<File>>,
    level: LogLevel,
    path: PathBuf,
}

impl SessionLog {
    /// Create a new session log. Creates the log directory and file.
    /// Returns a no-op logger if logging is disabled or file creation fails.
    pub fn new(config: &Config) -> Self {
        if !config.logging.enabled {
            return Self {
                file: Mutex::new(None),
                level: LogLevel::from_str(&config.logging.level),
                path: PathBuf::new(),
            };
        }

        let level = LogLevel::from_str(&config.logging.level);
        let log_dir = config.miniswe_path("logs");
        let _ = fs::create_dir_all(&log_dir);

        let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
        let log_path = log_dir.join(format!("{timestamp}.log"));

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .ok();

        if let Some(ref f) = file {
            // Write header
            let mut f = f.try_clone().ok();
            if let Some(ref mut f) = f {
                let _ = writeln!(f, "# miniswe session log");
                let _ = writeln!(f, "# started: {}", chrono::Local::now().format("%Y-%m-%d %H:%M:%S"));
                let _ = writeln!(f, "# project: {}", config.project_root.display());
                let _ = writeln!(f, "# model: {} @ {}", config.model.model, config.model.endpoint);
                let _ = writeln!(f, "# level: {}", config.logging.level);
                let _ = writeln!(f, "---");
            }
        }

        Self {
            file: Mutex::new(file),
            level,
            path: log_path,
        }
    }

    /// Log path for display.
    pub fn path(&self) -> &PathBuf {
        &self.path
    }

    // ── Info level: tool outcomes ────────────────────────────────────

    /// Log a tool call and its outcome (info level).
    pub fn tool_call(&self, name: &str, args_summary: &str, success: bool, first_line: &str) {
        if self.level < LogLevel::Info {
            return;
        }
        let icon = if success { "✓" } else { "✗" };
        self.write(&format!("[tool] {icon} {name}({args_summary}) → {first_line}"));
    }

    /// Log the start of an agent round (info level).
    pub fn round_start(&self, round: usize) {
        if self.level < LogLevel::Info {
            return;
        }
        self.write(&format!("[round {round}]"));
    }

    /// Log agent completion or error (info level).
    pub fn session_end(&self, rounds: usize, had_error: bool) {
        let status = if had_error { "error" } else { "ok" };
        self.write(&format!("[end] {rounds} rounds, status={status}"));
    }

    /// Log a loop detection event (info level).
    pub fn loop_detected(&self, name: &str, args_summary: &str, count: usize) {
        self.write(&format!("[loop] {name}({args_summary}) repeated {count}x"));
    }

    // ── Debug level: full interactions ───────────────────────────────

    /// Log the user's message (debug level).
    pub fn user_message(&self, message: &str) {
        if self.level < LogLevel::Debug {
            return;
        }
        self.write(&format!("[user] {}", truncate(message, 500)));
    }

    /// Log the LLM's text response (debug level).
    pub fn llm_response(&self, content: &str) {
        if self.level < LogLevel::Debug {
            return;
        }
        self.write(&format!("[llm] {}", truncate(content, 1000)));
    }

    /// Log a tool call with full arguments (debug level).
    pub fn tool_call_detail(&self, name: &str, args: &serde_json::Value) {
        if self.level < LogLevel::Debug {
            return;
        }
        let args_str = serde_json::to_string(args).unwrap_or_default();
        self.write(&format!("[tool:call] {name} {}", truncate(&args_str, 500)));
    }

    /// Log a tool result with full content (debug level).
    pub fn tool_result_detail(&self, name: &str, success: bool, content: &str) {
        if self.level < LogLevel::Debug {
            return;
        }
        let icon = if success { "✓" } else { "✗" };
        self.write(&format!("[tool:result] {icon} {name}\n{}", truncate(content, 2000)));
    }

    /// Log a file modification (debug level).
    pub fn file_modified(&self, path: &str, action: &str) {
        if self.level < LogLevel::Debug {
            return;
        }
        self.write(&format!("[file] {action} {path}"));
    }

    /// Log an LLM error (debug level).
    pub fn llm_error(&self, error: &str) {
        self.write(&format!("[error:llm] {error}"));
    }

    // ── Trace level: internals ──────────────────────────────────────

    /// Log context assembly stats (trace level).
    pub fn context_assembled(&self, token_estimate: usize, message_count: usize) {
        if self.level < LogLevel::Trace {
            return;
        }
        self.write(&format!(
            "[context] ~{token_estimate} tokens, {message_count} messages"
        ));
    }

    /// Log observation masking decisions (trace level).
    pub fn masking_applied(&self, masked_count: usize, total: usize) {
        if self.level < LogLevel::Trace {
            return;
        }
        self.write(&format!("[masking] {masked_count}/{total} tool results compressed"));
    }

    /// Log a custom trace message (trace level).
    pub fn trace(&self, msg: &str) {
        if self.level < LogLevel::Trace {
            return;
        }
        self.write(&format!("[trace] {msg}"));
    }

    // ── Internal ────────────────────────────────────────────────────

    fn write(&self, line: &str) {
        let Ok(mut guard) = self.file.lock() else {
            return;
        };
        let Some(ref mut file) = *guard else {
            return;
        };
        let ts = chrono::Local::now().format("%H:%M:%S%.3f");
        let _ = writeln!(file, "{ts} {line}");
    }
}

/// Truncate a string for logging, avoiding mid-line cuts.
fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.replace('\n', "\\n")
    } else {
        format!("{}...({}B total)", s[..max].replace('\n', "\\n"), s.len())
    }
}
