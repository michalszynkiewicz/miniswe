//! Session logger — writes structured logs to `.buzztr/logs/`.
//!
//! Three verbosity levels:
//! - **info**: tool calls and outcomes (one-liner per action)
//! - **debug**: full interactions — LLM messages, tool args/results, file changes
//! - **trace**: everything + context assembly stats, token counts, masking decisions

use parking_lot::Mutex;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

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

/// How a turn ended, as recorded in the `[end]` log line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionStatus {
    Ok,
    Error,
    Interrupted,
}

impl SessionStatus {
    /// Combine the log-only flags. Precedence: error > interrupted > ok.
    pub fn from_flags(errored: bool, interrupted: bool) -> Self {
        if errored {
            Self::Error
        } else if interrupted {
            Self::Interrupted
        } else {
            Self::Ok
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Error => "error",
            Self::Interrupted => "interrupted",
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
        let log_dir = config.buzztr_path("logs");
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
                let _ = writeln!(f, "# buzztr session log");
                let _ = writeln!(
                    f,
                    "# started: {}",
                    chrono::Local::now().format("%Y-%m-%d %H:%M:%S")
                );
                let _ = writeln!(f, "# project: {}", config.project_root.display());
                let _ = writeln!(
                    f,
                    "# model: {} @ {}",
                    config.model.model, config.model.endpoint
                );
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
        self.write(&format!(
            "[tool] {icon} {name}({args_summary}) → {first_line}"
        ));
    }

    /// Log the start of an agent round (info level).
    pub fn round_start(&self, round: usize) {
        if self.level < LogLevel::Info {
            return;
        }
        self.write(&format!("[round {round}]"));
    }

    /// Log agent completion or error (info level).
    pub fn session_end(&self, rounds: usize, status: SessionStatus) {
        self.write(&format!(
            "[end] {rounds} rounds, status={}",
            status.as_str()
        ));
    }

    /// Log the REPL's per-turn routing decision: `explore` or `coding`.
    pub fn route(&self, route: &str) {
        self.write(&format!("[route] {route}"));
    }

    /// Log that the skill router picked a skill for this turn.
    pub fn route_skill(&self, skill: &str) {
        self.write(&format!("[route] skill={skill}"));
    }

    /// Log the partial assistant text of an interrupted generation.
    pub fn interrupted_partial(&self, partial: &str) {
        self.llm_response(&format!("(interrupted) {partial}"));
    }

    /// Log one LLM call's token usage (info level). Called after a
    /// successful response that carries a `usage` block — hosted providers
    /// and llama.cpp both send one; a bare OpenAI-compatible stub may not,
    /// in which case the caller simply never calls this.
    pub fn llm_usage(&self, usage: &crate::llm::Usage) {
        if self.level < LogLevel::Info {
            return;
        }
        self.write(&format!(
            "[usage] prompt={} completion={} cached={} cache_write={} total={}",
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.cached_tokens(),
            usage.cache_write_tokens(),
            usage.total_tokens
        ));
    }

    /// Log the session-wide token usage total, folded across every LLM
    /// client the router used. Call once, right before `session_end`.
    pub fn usage_total(&self, usage: &crate::llm::providers::UsageSnapshot) {
        self.write(&format!(
            "[usage:total] prompt={} completion={} cached={} cache_write={} total={} calls={}",
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.cached_tokens,
            usage.cache_write_tokens,
            usage.total_tokens(),
            usage.calls
        ));
    }

    /// Log the `runtime.max_session_input_tokens` budget guard tripping
    /// (info level) — the turn is about to stop because of it.
    pub fn budget_exceeded(&self, used: u64, limit: u64) {
        self.write(&format!(
            "[budget] session input-token budget exceeded: used={used} limit={limit}"
        ));
    }

    /// Log a loop detection event (info level).
    pub fn loop_detected(&self, name: &str, args_summary: &str, count: usize) {
        self.write(&format!("[loop] {name}({args_summary}) repeated {count}x"));
    }

    /// Log a read-pruning pass. Same level as `loop_detected`: this is a loop
    /// intervention, and a bench post-mortem has to be able to count how often
    /// it fired without turning on trace logging for a whole run.
    pub fn reads_pruned(&self, pairs: usize, keys: usize, deepest: &str) {
        self.write(&format!(
            "[prune] dropped {pairs} repeated call/result pairs across {keys} key(s); deepest {deepest}"
        ));
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
        self.write(&format!(
            "[tool:result] {icon} {name}\n{}",
            truncate(content, 2000)
        ));
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

    /// Log an internal tool stage transition (debug level).
    pub fn tool_stage(&self, tool: &str, stage: &str) {
        if self.level < LogLevel::Debug {
            return;
        }
        self.write(&format!("[tool:stage] {tool} {stage}"));
    }

    /// Log additional tool diagnostics (debug level).
    pub fn tool_debug(&self, tool: &str, detail: &str) {
        if self.level < LogLevel::Debug {
            return;
        }
        self.write(&format!("[tool:debug] {tool} {}", truncate(detail, 20000)));
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
        self.write(&format!(
            "[masking] {masked_count}/{total} tool results compressed"
        ));
    }

    /// Log a custom trace message (trace level).
    pub fn trace(&self, msg: &str) {
        if self.level < LogLevel::Trace {
            return;
        }
        self.write(&format!("[trace] {msg}"));
    }

    /// Log the full LLM request payload (trace level).
    pub fn llm_request(&self, request: &crate::llm::ChatRequest) {
        if self.level < LogLevel::Trace {
            return;
        }
        let body = serde_json::to_string(request).unwrap_or_else(|_| "{}".into());
        self.write(&format!("[llm:request] {}", truncate(&body, 6000)));
    }

    // ── Internal ────────────────────────────────────────────────────

    fn write(&self, line: &str) {
        let mut guard = self.file.lock();
        let Some(ref mut file) = *guard else {
            return;
        };
        let ts = chrono::Local::now().format("%H:%M:%S%.3f");
        let _ = writeln!(file, "{ts} {line}");
    }
}

/// Truncate a string for logging, avoiding mid-line cuts.
fn truncate(s: &str, max_chars: usize) -> String {
    let flat = s.replace('\n', "\\n");
    let char_count = flat.chars().count();
    if char_count <= max_chars {
        flat
    } else {
        let truncated: String = flat.chars().take(max_chars).collect();
        format!("{truncated}...({}B total)", s.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_strings() {
        assert_eq!(SessionStatus::Ok.as_str(), "ok");
        assert_eq!(SessionStatus::Error.as_str(), "error");
        assert_eq!(SessionStatus::Interrupted.as_str(), "interrupted");
    }

    #[test]
    fn status_precedence_error_over_interrupted_over_ok() {
        assert_eq!(SessionStatus::from_flags(false, false), SessionStatus::Ok);
        assert_eq!(
            SessionStatus::from_flags(false, true),
            SessionStatus::Interrupted
        );
        assert_eq!(SessionStatus::from_flags(true, false), SessionStatus::Error);
        assert_eq!(SessionStatus::from_flags(true, true), SessionStatus::Error);
    }

    #[test]
    fn session_end_and_route_lines_are_written() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = Config::default();
        config.project_root = dir.path().to_path_buf();
        config.logging.enabled = true;
        let log = SessionLog::new(&config);
        log.route("explore");
        log.route_skill("scaffold");
        log.session_end(3, SessionStatus::Interrupted);
        log.session_end(1, SessionStatus::Error);
        log.session_end(2, SessionStatus::Ok);
        let text = fs::read_to_string(log.path()).unwrap();
        assert!(text.contains("[route] explore"));
        assert!(text.contains("[route] skill=scaffold"));
        assert!(text.contains("[end] 3 rounds, status=interrupted"));
        assert!(text.contains("[end] 1 rounds, status=error"));
        assert!(text.contains("[end] 2 rounds, status=ok"));
    }
}
