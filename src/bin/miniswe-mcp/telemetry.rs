//! Per-project MCP telemetry: an append-only JSONL event log plus running
//! success/error counts per tool.
//!
//! Written to `~/.miniswe/projects/<slug>/mcp.log` (not the project's own
//! `.miniswe/`) so usage across many different project directories collects
//! in one place — one `miniswe-mcp` config in Claude Code can serve any repo,
//! and this keeps their logs findable without knowing which repo you were in.
//! `<slug>` is the project's absolute path with path separators replaced by
//! `-`, matching the convention already used for this project's other
//! per-directory state outside the repo itself.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use serde_json::json;

#[derive(Default, Clone, Copy)]
pub struct Counts {
    pub ok: u64,
    pub error: u64,
}

pub struct Telemetry {
    file: File,
    counts: HashMap<String, Counts>,
}

impl Telemetry {
    /// Open (creating if needed) the telemetry log for `project_root`.
    pub fn open(project_root: &Path) -> Result<Self> {
        let path = log_path(project_root)?;
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)
                .with_context(|| format!("failed to create {}", dir.display()))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("failed to open {}", path.display()))?;
        Ok(Self {
            file,
            counts: HashMap::new(),
        })
    }

    /// Record one tool-call outcome: updates in-memory counts and appends a
    /// JSONL event line. `detail` is a short, already-truncated error
    /// summary — omit for successful calls.
    pub fn record_call(&mut self, tool: &str, ok: bool, duration: Duration, detail: Option<&str>) {
        let entry = self.counts.entry(tool.to_string()).or_default();
        if ok {
            entry.ok += 1;
        } else {
            entry.error += 1;
        }

        self.write_event(json!({
            "event": "tool_call",
            "tool": tool,
            "outcome": if ok { "ok" } else { "error" },
            "duration_ms": duration.as_millis(),
            "detail": detail,
        }));
    }

    /// Log server startup.
    pub fn log_start(&mut self, lsp_ready: bool) {
        self.write_event(json!({
            "event": "start",
            "pid": std::process::id(),
            "lsp_ready": lsp_ready,
        }));
    }

    /// Log server shutdown with a summary of this session's counts.
    pub fn log_stop(&mut self) {
        self.write_event(json!({
            "event": "stop",
            "summary": self.summary(),
        }));
    }

    fn summary(&self) -> serde_json::Value {
        let per_tool: serde_json::Map<String, serde_json::Value> = self
            .counts
            .iter()
            .map(|(name, c)| (name.clone(), json!({ "ok": c.ok, "error": c.error })))
            .collect();
        json!(per_tool)
    }

    fn write_event(&mut self, mut event: serde_json::Value) {
        if let Some(map) = event.as_object_mut() {
            map.insert("ts".into(), json!(chrono::Local::now().to_rfc3339()));
        }
        // Telemetry must never take the server down — best-effort only.
        let _ = writeln!(self.file, "{event}");
        let _ = self.file.flush();
    }
}

fn log_path(project_root: &Path) -> Result<PathBuf> {
    let home = dirs::home_dir().context("could not determine home directory")?;
    let slug = project_root.to_string_lossy().replace(['/', '\\'], "-");
    Ok(home
        .join(".miniswe")
        .join("projects")
        .join(slug)
        .join("mcp.log"))
}
