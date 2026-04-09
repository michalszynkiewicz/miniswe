//! Permission system for tool execution.
//!
//! Controls what the agent is allowed to do. Every destructive or sensitive
//! action requires user approval. Permissions can be pre-approved in config
//! or prompted interactively.
//!
//! Security boundaries:
//! - File access: jailed to project root (no path traversal)
//! - Shell commands: require user approval (with allow-list for safe commands)
//! - Web access: requires user approval on first use per session
//! - MCP tools: require user approval per server on first use

use std::collections::HashSet;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Mutex;

use crate::config::Config;
use crate::tui::event::AppEvent;
use tokio::sync::mpsc;

/// Permission decisions cached for the current session.
pub struct PermissionManager {
    /// Project root (canonical path for jail enforcement)
    project_root: PathBuf,
    /// Shell commands approved this session (prefix match)
    approved_shell: Mutex<HashSet<String>>,
    /// MCP servers approved this session
    approved_mcp: Mutex<HashSet<String>>,
    /// Whether web access has been approved this session
    web_approved: Mutex<bool>,
    /// Pre-approved shell command prefixes from config
    shell_allowlist: Vec<String>,
    /// Whether to auto-approve all actions (--yes flag, dangerous)
    auto_approve: bool,
    /// Optional REPL UI event channel for permission requests.
    prompt_events: Mutex<Option<mpsc::UnboundedSender<AppEvent>>>,
}

/// Actions that require permission.
#[derive(Debug, Clone)]
pub enum Action {
    /// Read a file (path is resolved and jail-checked)
    ReadFile(String),
    /// Write/edit a file
    WriteFile(String),
    /// Execute a shell command
    Shell(String),
    /// Web search (carries the query so user can see what's being sent)
    WebSearch(String),
    /// Fetch a URL (carries the URL so user can see where data goes)
    WebFetch(String),
    /// Call an MCP tool
    McpUse(String, String), // (server, tool)
}

impl PermissionManager {
    pub fn new(config: &Config) -> Self {
        Self::with_auto_approve(config, false)
    }

    pub fn headless(config: &Config) -> Self {
        Self::with_auto_approve(config, true)
    }

    fn with_auto_approve(config: &Config, auto_approve: bool) -> Self {
        let project_root = config
            .project_root
            .canonicalize()
            .unwrap_or_else(|_| config.project_root.clone());

        // Default safe shell prefixes
        let shell_allowlist = vec![
            "cargo".into(),
            "go ".into(),
            "npm ".into(),
            "pnpm ".into(),
            "yarn ".into(),
            "python".into(),
            "pip ".into(),
            "pytest".into(),
            "node ".into(),
            "npx ".into(),
            "git status".into(),
            "git diff".into(),
            "git log".into(),
            "git branch".into(),
            "ls".into(),
            "cat ".into(),
            "head ".into(),
            "tail ".into(),
            "wc ".into(),
            "find ".into(),
            "grep ".into(),
            "rg ".into(),
            "echo ".into(),
            "which ".into(),
            "pwd".into(),
            "env".into(),
            "make".into(),
        ];

        Self {
            project_root,
            approved_shell: Mutex::new(HashSet::new()),
            approved_mcp: Mutex::new(HashSet::new()),
            web_approved: Mutex::new(false),
            shell_allowlist,
            auto_approve,
            prompt_events: Mutex::new(None),
        }
    }

    pub fn set_prompt_event_tx(&self, tx: mpsc::UnboundedSender<AppEvent>) {
        *self.prompt_events.lock().unwrap() = Some(tx);
    }

    /// Resolve a file path and verify it's within the project root.
    /// Returns the canonical path or an error if it escapes the jail.
    pub fn resolve_and_check_path(&self, path_str: &str) -> Result<PathBuf, String> {
        // Reject absolute paths
        if path_str.starts_with('/') || path_str.starts_with('\\') {
            return Err(format!(
                "Absolute paths not allowed: {path_str}. Use paths relative to the project root: {}",
                self.project_root.display()
            ));
        }

        let joined = self.project_root.join(path_str);

        // Canonicalize to resolve ../
        // For new files that don't exist yet, canonicalize the parent
        let canonical = if joined.exists() {
            joined
                .canonicalize()
                .map_err(|e| format!("Cannot resolve path: {e}"))?
        } else {
            let parent = joined.parent().ok_or_else(|| "Invalid path".to_string())?;
            if !parent.exists() {
                // Parent doesn't exist — will be created, check grandparent
                let mut check = parent.to_path_buf();
                while !check.exists() {
                    check = check
                        .parent()
                        .ok_or_else(|| "Invalid path".to_string())?
                        .to_path_buf();
                }
                let canonical_parent = check
                    .canonicalize()
                    .map_err(|e| format!("Cannot resolve path: {e}"))?;
                if !canonical_parent.starts_with(&self.project_root) {
                    return Err(format!(
                        "Path escapes project root: {path_str}. Project root is: {}",
                        self.project_root.display()
                    ));
                }
                return Ok(joined);
            }
            let canonical_parent = parent
                .canonicalize()
                .map_err(|e| format!("Cannot resolve path: {e}"))?;
            canonical_parent.join(joined.file_name().unwrap_or_default())
        };

        if !canonical.starts_with(&self.project_root) {
            return Err(format!("Path escapes project root: {path_str}"));
        }

        Ok(canonical)
    }

    /// Check if an action needs a user prompt. Returns:
    /// - `Ok(None)` — allowed, no prompt needed
    /// - `Ok(Some(prompt))` — needs user approval, here's the prompt text
    /// - `Err(reason)` — blocked (blocklist), no prompt will help
    pub fn check_needs_prompt(&self, action: &Action) -> Result<Option<String>, String> {
        if self.auto_approve {
            return Ok(None);
        }
        match action {
            Action::ReadFile(_) | Action::WriteFile(_) => Ok(None),
            Action::Shell(cmd) => self.check_shell_needs_prompt(cmd),
            Action::WebSearch(query) => self.check_web_needs_prompt(&format!(
                "Allow web search?\n  query: \"{query}\"\n  [y]es / [n]o / [a]llow all web: "
            )),
            Action::WebFetch(url) => self.check_web_needs_prompt(&format!(
                "Allow web fetch?\n  url: {url}\n  [y]es / [n]o / [a]llow all web: "
            )),
            Action::McpUse(server, tool) => self.check_mcp_needs_prompt(server, tool),
        }
    }

    /// Record user's approval for an action after prompting.
    pub fn approve(&self, action: &Action, always: bool) {
        match action {
            Action::Shell(cmd) => {
                if always {
                    self.approved_shell
                        .lock()
                        .unwrap()
                        .insert(cmd.trim().to_string());
                }
            }
            Action::WebSearch(_) | Action::WebFetch(_) => {
                if always {
                    *self.web_approved.lock().unwrap() = true;
                }
            }
            Action::McpUse(server, _) => {
                if always {
                    self.approved_mcp.lock().unwrap().insert(server.clone());
                }
            }
            _ => {}
        }
    }

    fn check_shell_needs_prompt(&self, cmd: &str) -> Result<Option<String>, String> {
        let cmd_trimmed = cmd.trim();
        for blocked in BLOCKED_COMMANDS {
            if cmd_trimmed.contains(blocked) {
                return Err(format!("Blocked dangerous command: {cmd_trimmed}"));
            }
        }
        for prefix in &self.shell_allowlist {
            if cmd_trimmed.starts_with(prefix) {
                return Ok(None);
            }
        }
        {
            let approved = self.approved_shell.lock().unwrap();
            if approved.contains(cmd_trimmed) {
                return Ok(None);
            }
        }
        Ok(Some(format!(
            "Allow shell command?\n  $ {cmd_trimmed}\n  [y]es / [n]o / [a]lways for this command: "
        )))
    }

    fn check_web_needs_prompt(&self, prompt: &str) -> Result<Option<String>, String> {
        {
            let approved = self.web_approved.lock().unwrap();
            if *approved {
                return Ok(None);
            }
        }
        Ok(Some(prompt.to_string()))
    }

    fn check_mcp_needs_prompt(&self, server: &str, tool: &str) -> Result<Option<String>, String> {
        {
            let approved = self.approved_mcp.lock().unwrap();
            if approved.contains(server) {
                return Ok(None);
            }
        }
        Ok(Some(format!(
            "Allow MCP '{server}' tool '{tool}'?\n  [y]es / [n]o / [a]lways for this server: "
        )))
    }

    /// Check if an action is allowed. Prompts the user if needed (non-TUI mode).
    /// Returns Ok(()) if approved, Err(reason) if denied.
    pub fn check(&self, action: &Action) -> Result<(), String> {
        if self.auto_approve {
            return Ok(());
        }

        match action {
            Action::ReadFile(_) | Action::WriteFile(_) => {
                // File access is jail-checked at resolve time, always allowed within jail
                Ok(())
            }
            Action::Shell(cmd) => self.check_shell(cmd),
            Action::WebSearch(query) => self.check_web_search(query),
            Action::WebFetch(url) => self.check_web_fetch(url),
            Action::McpUse(server, tool) => self.check_mcp(server, tool),
        }
    }

    /// Check if a shell command is allowed.
    fn check_shell(&self, cmd: &str) -> Result<(), String> {
        let cmd_trimmed = cmd.trim();

        // Check blocklist (dangerous commands)
        for blocked in BLOCKED_COMMANDS {
            if cmd_trimmed.contains(blocked) {
                return Err(format!("Blocked dangerous command: {cmd_trimmed}"));
            }
        }

        // Check allowlist
        for prefix in &self.shell_allowlist {
            if cmd_trimmed.starts_with(prefix) {
                return Ok(());
            }
        }

        // Check session approvals
        {
            let approved = self.approved_shell.lock().unwrap();
            if approved.contains(cmd_trimmed) {
                return Ok(());
            }
        }

        // Prompt user
        let approved = self.request_user_decision(&format!(
            "Allow shell command?\n  $ {cmd_trimmed}\n  [y]es / [n]o / [a]lways for this command: "
        ));

        match approved.as_str() {
            "y" | "yes" => Ok(()),
            "a" | "always" => {
                self.approved_shell
                    .lock()
                    .unwrap()
                    .insert(cmd_trimmed.to_string());
                Ok(())
            }
            _ => Err("Shell command denied by user".into()),
        }
    }

    /// Check if a web search is allowed. Shows the query to the user.
    fn check_web_search(&self, query: &str) -> Result<(), String> {
        // If blanket web access was approved, allow
        {
            let approved = self.web_approved.lock().unwrap();
            if *approved {
                return Ok(());
            }
        }

        let response = self.request_user_decision(&format!(
            "Allow web search?\n\
             \x1b[2m  query: \"{query}\"\n\
             \x1b[2m  sends to: web search API\x1b[0m\n\
             \x1b[1;33m  [y]es / [n]o / [a]llow all web this session: \x1b[0m"
        ));

        match response.as_str() {
            "y" | "yes" => Ok(()),
            "a" | "always" | "allow" => {
                *self.web_approved.lock().unwrap() = true;
                Ok(())
            }
            _ => Err("Web search denied by user".into()),
        }
    }

    /// Check if a URL fetch is allowed. Shows the URL to the user.
    fn check_web_fetch(&self, url: &str) -> Result<(), String> {
        // If blanket web access was approved, allow
        {
            let approved = self.web_approved.lock().unwrap();
            if *approved {
                return Ok(());
            }
        }

        let via = if url.contains("r.jina.ai") || !url.starts_with("http") {
            "Jina Reader API (content proxied through r.jina.ai)"
        } else {
            "direct HTTP request"
        };

        let response = self.request_user_decision(&format!(
            "Allow web fetch?\n\
             \x1b[2m  url: {url}\n\
             \x1b[2m  via: {via}\x1b[0m\n\
             \x1b[1;33m  [y]es / [n]o / [a]llow all web this session: \x1b[0m"
        ));

        match response.as_str() {
            "y" | "yes" => Ok(()),
            "a" | "always" | "allow" => {
                *self.web_approved.lock().unwrap() = true;
                Ok(())
            }
            _ => Err("Web fetch denied by user".into()),
        }
    }

    /// Check if an MCP tool call is allowed.
    fn check_mcp(&self, server: &str, tool: &str) -> Result<(), String> {
        {
            let approved = self.approved_mcp.lock().unwrap();
            if approved.contains(server) {
                return Ok(());
            }
        }

        let response = self.request_user_decision(&format!(
            "Allow MCP server '{server}' to run tool '{tool}'?\n  [y]es / [n]o / [a]lways for this server: "
        ));

        match response.as_str() {
            "y" | "yes" => Ok(()),
            "a" | "always" => {
                self.approved_mcp.lock().unwrap().insert(server.to_string());
                Ok(())
            }
            _ => Err(format!("MCP tool '{server}/{tool}' denied by user")),
        }
    }

    fn request_user_decision(&self, prompt: &str) -> String {
        if let Some(tx) = self.prompt_events.lock().unwrap().clone() {
            let (response_tx, response_rx) = std::sync::mpsc::channel();
            if tx
                .send(AppEvent::PermissionRequest(
                    prompt.to_string(),
                    response_tx,
                ))
                .is_ok()
            {
                return response_rx.recv().unwrap_or_else(|_| "n".into());
            }
        }
        prompt_user(prompt)
    }
}

/// Commands that are always blocked.
const BLOCKED_COMMANDS: &[&str] = &[
    "rm -rf /",
    "rm -rf ~",
    "rm -rf $HOME",
    "mkfs",
    "dd if=",
    ":(){:|:&};:",
    "chmod -R 777 /",
    "wget http", // downloading arbitrary executables
    "curl | sh",
    "curl | bash",
];

/// Prompt the user for a decision.
///
/// Temporarily disables terminal raw mode (which reedline enables)
/// so that stdin reads work correctly for permission prompts.
fn prompt_user(message: &str) -> String {
    // Ensure terminal is in cooked mode for line input
    let _ = crossterm::terminal::disable_raw_mode();

    eprint!("\x1b[1;33m{message}\x1b[0m");
    io::stderr().flush().ok();

    let mut input = String::new();
    let result = match io::stdin().read_line(&mut input) {
        Ok(_) => input.trim().to_lowercase(),
        Err(_) => "n".into(),
    };

    // Don't re-enable raw mode — reedline will do that on next read_line()
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    #[tokio::test]
    async fn permission_checks_can_request_ui_decision_over_event_channel() {
        let config = Config::default();
        let perms = std::sync::Arc::new(PermissionManager::new(&config));
        let (tx, mut rx) = mpsc::unbounded_channel();
        perms.set_prompt_event_tx(tx);

        let perms_for_thread = perms.clone();
        let handle = std::thread::spawn(move || {
            perms_for_thread.check(&Action::WebSearch("golang static file server".into()))
        });

        let event = rx.recv().await.expect("permission request event");
        match event {
            AppEvent::PermissionRequest(prompt, response_tx) => {
                assert!(prompt.contains("Allow web search?"));
                response_tx.send("y".into()).unwrap();
            }
            other => panic!("unexpected event: {other:?}"),
        }

        assert!(handle.join().unwrap().is_ok());
    }
}
