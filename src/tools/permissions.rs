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
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::config::Config;

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
}

/// Actions that require permission.
#[derive(Debug)]
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
        }
    }

    /// Resolve a file path and verify it's within the project root.
    /// Returns the canonical path or an error if it escapes the jail.
    pub fn resolve_and_check_path(&self, path_str: &str) -> Result<PathBuf, String> {
        // Reject absolute paths
        if path_str.starts_with('/') || path_str.starts_with('\\') {
            return Err(format!(
                "Absolute paths not allowed: {path_str}. Use paths relative to the project root."
            ));
        }

        let joined = self.project_root.join(path_str);

        // Canonicalize to resolve ../
        // For new files that don't exist yet, canonicalize the parent
        let canonical = if joined.exists() {
            joined.canonicalize().map_err(|e| format!("Cannot resolve path: {e}"))?
        } else {
            let parent = joined
                .parent()
                .ok_or_else(|| "Invalid path".to_string())?;
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
                        "Path escapes project root: {path_str}"
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
            return Err(format!(
                "Path escapes project root: {path_str}"
            ));
        }

        Ok(canonical)
    }

    /// Check if an action is allowed. Prompts the user if needed.
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
        let approved = prompt_user(&format!(
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

        let response = prompt_user(&format!(
            "Allow web search?\n\
             \x1b[2m  query: \"{query}\"\n\
             \x1b[2m  sends to: DuckDuckGo\x1b[0m\n\
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

        let response = prompt_user(&format!(
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

        let response = prompt_user(&format!(
            "Allow MCP server '{server}' to run tool '{tool}'?\n  [y]es / [n]o / [a]lways for this server: "
        ));

        match response.as_str() {
            "y" | "yes" => Ok(()),
            "a" | "always" => {
                self.approved_mcp
                    .lock()
                    .unwrap()
                    .insert(server.to_string());
                Ok(())
            }
            _ => Err(format!("MCP tool '{server}/{tool}' denied by user")),
        }
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
    "wget http",   // downloading arbitrary executables
    "curl | sh",
    "curl | bash",
];

/// Prompt the user for a decision.
fn prompt_user(message: &str) -> String {
    eprint!("\x1b[1;33m{message}\x1b[0m");
    io::stderr().flush().ok();

    let mut input = String::new();
    match io::stdin().read_line(&mut input) {
        Ok(_) => input.trim().to_lowercase(),
        Err(_) => "n".into(),
    }
}
