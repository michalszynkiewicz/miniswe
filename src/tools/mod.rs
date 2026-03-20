//! Tool system for miniswe.
//!
//! 12 built-in tools + MCP bridge. All file access is jailed to the project
//! root. Destructive actions (shell, web, MCP) require user permission.
//! After file edits, the index is incrementally updated.

mod edit;
mod read_file;
mod read_symbol;
mod search;
mod shell;
mod task_update;
mod web;
mod write_file;

pub mod definitions;
pub mod permissions;
pub use definitions::tool_definitions;
pub use permissions::PermissionManager;

use anyhow::{Result, bail};
use permissions::Action;
use serde_json::Value;

use crate::config::Config;
use crate::knowledge::ProjectIndex;
use crate::knowledge::indexer;

/// Result of executing a tool.
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub content: String,
    pub success: bool,
}

impl ToolResult {
    pub fn ok(content: String) -> Self {
        Self {
            content,
            success: true,
        }
    }

    pub fn err(content: String) -> Self {
        Self {
            content,
            success: false,
        }
    }
}

/// Execute a tool by name with the given arguments.
/// All file paths are jail-checked. Destructive actions require permission.
/// After successful file mutations (edit, write_file), the index is updated.
pub async fn execute_tool(
    name: &str,
    args: &Value,
    config: &Config,
    perms: &PermissionManager,
) -> Result<ToolResult> {
    match name {
        "read_file" => {
            let path = args["path"].as_str().unwrap_or("");
            if let Err(e) = perms.resolve_and_check_path(path) {
                return Ok(ToolResult::err(e));
            }
            read_file::execute(args, config).await
        }
        "read_symbol" => read_symbol::execute(args, config).await,
        "search" => search::execute(args, config).await,
        "edit" => {
            let path = args["path"].as_str().unwrap_or("");
            if let Err(e) = perms.resolve_and_check_path(path) {
                return Ok(ToolResult::err(e));
            }
            let mut result = edit::execute(args, config).await?;
            if result.success {
                reindex_changed_file(path, config);
                auto_check(path, config, &mut result).await;
            }
            Ok(result)
        }
        "write_file" => {
            let path = args["path"].as_str().unwrap_or("");
            if let Err(e) = perms.resolve_and_check_path(path) {
                return Ok(ToolResult::err(e));
            }
            let mut result = write_file::execute(args, config).await?;
            if result.success {
                reindex_changed_file(path, config);
                auto_check(path, config, &mut result).await;
            }
            Ok(result)
        }
        "shell" => {
            let cmd = args["command"].as_str().unwrap_or("");
            if let Err(e) = perms.check(&Action::Shell(cmd.into())) {
                return Ok(ToolResult::err(e));
            }
            shell::execute(args, config).await
        }
        "task_update" => task_update::execute(args, config).await,
        "diagnostics" => {
            let shell_args = serde_json::json!({
                "command": format!("cd {} && cargo check --message-format=short 2>&1 | head -50",
                    config.project_root.display()),
                "timeout": 30
            });
            shell::execute(&shell_args, config).await
        }
        "web_search" => {
            let query = args["query"].as_str().unwrap_or("");
            if let Err(e) = perms.check(&Action::WebSearch(query.into())) {
                return Ok(ToolResult::err(e));
            }
            web::search(args, config).await
        }
        "web_fetch" => {
            let url = args["url"].as_str().unwrap_or("");
            if let Err(e) = perms.check(&Action::WebFetch(url.into())) {
                return Ok(ToolResult::err(e));
            }
            web::fetch(args, config).await
        }
        "docs_lookup" => web::docs_lookup(args, config).await,
        _ => bail!("Unknown tool: {name}"),
    }
}

/// Re-index a single changed file. Best-effort — doesn't fail the tool call.
fn reindex_changed_file(rel_path: &str, config: &Config) {
    let miniswe_dir = config.miniswe_dir();
    let abs_path = config.project_root.join(rel_path);

    let mut index = match ProjectIndex::load(&miniswe_dir) {
        Ok(idx) => idx,
        Err(_) => return, // no index yet, skip
    };

    indexer::reindex_file(rel_path, &abs_path, &mut index, &miniswe_dir);
}

/// Auto-run type checker after editing a source file. Appends output + source
/// context around errors to the tool result. Runs in a blocking thread to
/// avoid stalling the async runtime.
async fn auto_check(path: &str, config: &Config, result: &mut ToolResult) {
    // Determine which checker to run based on file extension and project markers
    let (cmd, args) = if path.ends_with(".rs") && config.project_root.join("Cargo.toml").exists() {
        ("cargo", vec!["check", "--message-format=short"])
    } else if (path.ends_with(".ts") || path.ends_with(".tsx"))
        && config.project_root.join("tsconfig.json").exists()
    {
        ("npx", vec!["tsc", "--noEmit", "--pretty", "false"])
    } else if path.ends_with(".go") && config.project_root.join("go.mod").exists() {
        ("go", vec!["vet", "./..."])
    } else if path.ends_with(".py") {
        ("python3", vec!["-m", "py_compile", path])
    } else {
        return; // no checker for this language
    };

    let project_root = config.project_root.clone();
    let project_root2 = project_root.clone();
    let cmd = cmd.to_string();
    let args: Vec<String> = args.into_iter().map(|s| s.to_string()).collect();

    // Run in a blocking thread with timeout to avoid stalling the async runtime
    // and to prevent pipe deadlock (same pattern as shell.rs).
    let check_result = tokio::task::spawn_blocking(move || {
        run_check_with_timeout(&cmd, &args, &project_root2, 30)
    })
    .await;

    let (success, stderr) = match check_result {
        Ok(Some(r)) => r,
        Ok(None) | Err(_) => return, // checker not available or panicked
    };

    let checker_name = if path.ends_with(".rs") {
        "cargo check"
    } else if path.ends_with(".ts") || path.ends_with(".tsx") {
        "tsc"
    } else if path.ends_with(".go") {
        "go vet"
    } else if path.ends_with(".py") {
        "py_compile"
    } else {
        "check"
    };

    if success {
        result.content.push_str(&format!("\n[{checker_name}] OK"));
        return;
    }

    // Extract error/warning lines
    let relevant: Vec<&str> = stderr
        .lines()
        .filter(|l| {
            l.contains("error") || l.contains("warning") || l.starts_with("  ")
        })
        .take(30)
        .collect();

    if relevant.is_empty() {
        result
            .content
            .push_str(&format!("\n[{checker_name}] failed (no details captured)"));
        result.success = false;
        return;
    }

    result.content.push_str(&format!("\n[{checker_name}]\n"));
    result.content.push_str(&relevant.join("\n"));

    // Parse error locations and include source context so the model can fix
    // errors without a separate read_file call.
    let locations = extract_error_locations(&stderr);
    if !locations.is_empty() {
        result.content.push_str("\n[source context]\n");
        for (file, line_num) in &locations {
            if let Some(ctx) = read_source_context(file, *line_num, &project_root) {
                result.content.push_str(&ctx);
            }
        }
    }

    result.success = false;
}

/// Run a check command with a timeout, draining pipes to prevent deadlock.
/// Returns Some((success, stderr)) or None if the command couldn't be spawned.
fn run_check_with_timeout(
    cmd: &str,
    args: &[String],
    project_root: &std::path::Path,
    timeout_secs: u64,
) -> Option<(bool, String)> {
    use std::io::Read;
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};

    let mut child = Command::new(cmd)
        .args(args)
        .current_dir(project_root)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .ok()?;

    // Drain pipes in background threads to prevent deadlock
    let stdout_pipe = child.stdout.take();
    let stderr_pipe = child.stderr.take();

    let stdout_handle = std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(out) = stdout_pipe {
            let _ = out.take(512 * 1024).read_to_end(&mut buf);
        }
        buf
    });
    let stderr_handle = std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(err) = stderr_pipe {
            let _ = err.take(512 * 1024).read_to_end(&mut buf);
        }
        buf
    });

    // Poll for completion with timeout
    let deadline = Instant::now() + Duration::from_secs(timeout_secs);
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {
                if Instant::now() > deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Some((false, "Check timed out".into()));
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(_) => return None,
        }
    }

    let success = child.wait().map(|s| s.success()).unwrap_or(false);
    let _stdout = stdout_handle.join().unwrap_or_default();
    let stderr_bytes = stderr_handle.join().unwrap_or_default();
    let stderr = String::from_utf8_lossy(&stderr_bytes).to_string();

    Some((success, stderr))
}

/// Parse error locations (file:line) from compiler stderr output.
/// Returns up to 3 locations. Handles cargo, tsc, go vet, and python formats.
fn extract_error_locations(stderr: &str) -> Vec<(String, usize)> {
    let mut locations = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for line in stderr.lines() {
        if !line.contains("error") && !line.contains("Error") {
            continue;
        }
        // Common format: "path/file.ext:LINE:COL" (cargo, tsc, go vet)
        // Also: " --> path/file.ext:LINE:COL" (cargo verbose)
        let trimmed = line.trim().trim_start_matches("--> ");
        let parts: Vec<&str> = trimmed.splitn(4, ':').collect();
        if parts.len() >= 2 {
            let file_path = parts[0].trim();
            if let Ok(line_num) = parts[1].trim().parse::<usize>() {
                // Sanity check: looks like a source file path
                if file_path.contains('.')
                    && !file_path.starts_with("//")
                    && !file_path.contains(' ')
                {
                    let key = format!("{file_path}:{line_num}");
                    if seen.insert(key) {
                        locations.push((file_path.to_string(), line_num));
                    }
                }
            }
        }
        if locations.len() >= 3 {
            break;
        }
    }

    locations
}

/// Read ±5 lines of source around an error location for inline context.
fn read_source_context(
    file_path: &str,
    line_num: usize,
    project_root: &std::path::Path,
) -> Option<String> {
    let abs_path = project_root.join(file_path);
    let content = std::fs::read_to_string(&abs_path).ok()?;
    let lines: Vec<&str> = content.lines().collect();

    if line_num == 0 || line_num > lines.len() {
        return None;
    }

    let start = line_num.saturating_sub(6); // 5 lines before (0-indexed)
    let end = (line_num + 5).min(lines.len());

    let mut output = format!("  {file_path}:{line_num}:\n");
    for i in start..end {
        let marker = if i + 1 == line_num { ">" } else { " " };
        output.push_str(&format!("  {marker}{:>4}│{}\n", i + 1, lines[i]));
    }
    Some(output)
}
