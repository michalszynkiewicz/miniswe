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
use crate::lsp::LspClient;

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
    lsp: Option<&LspClient>,
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
                auto_check(path, config, &mut result, lsp).await;
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
                auto_check(path, config, &mut result, lsp).await;
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
            // Try LSP diagnostics first for Rust projects
            if let Some(lsp) = lsp {
                if lsp.is_ready() && !lsp.has_crashed() {
                    if let Some(result) = lsp_project_diagnostics(config, lsp).await {
                        return Ok(result);
                    }
                }
            }
            // Fallback: cargo check
            let shell_args = serde_json::json!({
                "command": format!("cd {} && cargo check --message-format=short 2>&1 | head -50",
                    config.project_root.display()),
                "timeout": 30
            });
            shell::execute(&shell_args, config).await
        }
        "goto_definition" => {
            let path = args["path"].as_str().unwrap_or("");
            let line = args["line"].as_u64().unwrap_or(1).saturating_sub(1) as u32;
            let column = args["column"].as_u64().unwrap_or(1).saturating_sub(1) as u32;
            lsp_goto_definition(path, line, column, config, lsp).await
        }
        "find_references" => {
            let path = args["path"].as_str().unwrap_or("");
            let line = args["line"].as_u64().unwrap_or(1).saturating_sub(1) as u32;
            let column = args["column"].as_u64().unwrap_or(1).saturating_sub(1) as u32;
            lsp_find_references(path, line, column, config, lsp).await
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
async fn auto_check(path: &str, config: &Config, result: &mut ToolResult, lsp: Option<&LspClient>) {
    // Try LSP diagnostics first — ~200ms vs 2-5s for cargo check
    if path.ends_with(".rs") {
        if let Some(lsp) = lsp {
            if lsp.is_ready() && !lsp.has_crashed() {
                let abs_path = config.project_root.join(path);
                if lsp.notify_file_changed(&abs_path).is_ok() {
                    let timeout = std::time::Duration::from_millis(config.lsp.diagnostic_timeout_ms);
                    let diags = lsp.get_diagnostics(&abs_path, timeout).await;
                    if !diags.is_empty() || timeout.as_millis() >= config.lsp.diagnostic_timeout_ms as u128 {
                        let errors: Vec<&lsp_types::Diagnostic> = diags.iter()
                            .filter(|d| d.severity == Some(lsp_types::DiagnosticSeverity::ERROR))
                            .collect();
                        if errors.is_empty() {
                            result.content.push_str("\n[rust-analyzer] OK");
                        } else {
                            result.content.push_str("\n[rust-analyzer]\n");
                            for diag in &errors {
                                let line = diag.range.start.line + 1;
                                let col = diag.range.start.character + 1;
                                result.content.push_str(&format!(
                                    "{}:{}:{}: error: {}\n",
                                    path, line, col, diag.message
                                ));
                            }
                            // Add source context around errors
                            let project_root = config.project_root.clone();
                            for diag in &errors {
                                let line = (diag.range.start.line + 1) as usize;
                                if let Some(ctx) = read_source_context(path, line, &project_root) {
                                    result.content.push_str(&ctx);
                                }
                            }
                            result.success = false;
                        }
                        return;
                    }
                }
            }
        }
    }
    // Fallback: cargo check
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

// ── LSP tool implementations ──────────────────────────────────────────

/// Gather project-wide diagnostics from LSP.
async fn lsp_project_diagnostics(config: &Config, lsp: &LspClient) -> Option<ToolResult> {
    // Trigger a full check by notifying on Cargo.toml (forces re-analysis)
    let cargo_toml = config.project_root.join("Cargo.toml");
    if cargo_toml.exists() {
        let _ = lsp.notify_file_changed(&cargo_toml);
    }

    let timeout = std::time::Duration::from_millis(config.lsp.diagnostic_timeout_ms);
    tokio::time::sleep(timeout).await;

    // Collect all diagnostics
    let mut output = String::new();
    let mut error_count = 0;
    let mut warning_count = 0;

    for entry in lsp.diagnostics_snapshot() {
        for diag in entry.1 {
            let severity = match diag.severity {
                Some(lsp_types::DiagnosticSeverity::ERROR) => { error_count += 1; "error" }
                Some(lsp_types::DiagnosticSeverity::WARNING) => { warning_count += 1; "warning" }
                _ => continue,
            };
            let line = diag.range.start.line + 1;
            let col = diag.range.start.character + 1;
            // Strip file:// prefix for readability
            let path = entry.0.strip_prefix("file://").unwrap_or(&entry.0);
            output.push_str(&format!("{path}:{line}:{col}: {severity}: {}\n", diag.message));
        }
    }

    if output.is_empty() {
        Some(ToolResult::ok("[rust-analyzer] No errors or warnings".into()))
    } else {
        let summary = format!("[rust-analyzer] {error_count} error(s), {warning_count} warning(s)\n");
        let success = error_count == 0;
        Some(ToolResult { content: format!("{summary}{output}"), success })
    }
}

/// goto_definition tool handler.
async fn lsp_goto_definition(
    path: &str,
    line: u32,
    column: u32,
    config: &Config,
    lsp: Option<&LspClient>,
) -> Result<ToolResult> {
    let lsp = match lsp {
        Some(l) if l.is_ready() && !l.has_crashed() => l,
        _ => return Ok(ToolResult::err("LSP not available. Use search or read_file instead.".into())),
    };

    let abs_path = config.project_root.join(path);
    // Ensure file is open in LSP
    let _ = lsp.notify_file_changed(&abs_path);

    match lsp.goto_definition(&abs_path, line, column).await {
        Ok(locations) if locations.is_empty() => {
            Ok(ToolResult::ok("No definition found at this location.".into()))
        }
        Ok(locations) => {
            let mut output = String::new();
            for loc in &locations {
                let def_path = crate::lsp::client::uri_to_path(&loc.uri);
                let def_line = loc.range.start.line + 1;
                let def_col = loc.range.start.character + 1;

                if let Some(ref p) = def_path {
                    // Make path relative to project root
                    let rel = p.strip_prefix(&config.project_root)
                        .unwrap_or(p);
                    output.push_str(&format!("Definition: {}:{}:{}\n", rel.display(), def_line, def_col));

                    // Include source context
                    if let Some(ctx) = read_source_context(
                        &rel.to_string_lossy(),
                        def_line as usize,
                        &config.project_root,
                    ) {
                        output.push_str(&ctx);
                    }
                } else {
                    output.push_str(&format!("Definition: {}:{}:{}\n", loc.uri.as_str(), def_line, def_col));
                }
            }
            Ok(ToolResult::ok(output))
        }
        Err(e) => Ok(ToolResult::err(format!("LSP error: {e}"))),
    }
}

/// find_references tool handler.
async fn lsp_find_references(
    path: &str,
    line: u32,
    column: u32,
    config: &Config,
    lsp: Option<&LspClient>,
) -> Result<ToolResult> {
    let lsp = match lsp {
        Some(l) if l.is_ready() && !l.has_crashed() => l,
        _ => return Ok(ToolResult::err("LSP not available. Use search instead.".into())),
    };

    let abs_path = config.project_root.join(path);
    let _ = lsp.notify_file_changed(&abs_path);

    match lsp.find_references(&abs_path, line, column).await {
        Ok(locations) if locations.is_empty() => {
            Ok(ToolResult::ok("No references found.".into()))
        }
        Ok(locations) => {
            let mut output = format!("{} reference(s) found:\n", locations.len());
            for loc in &locations {
                let ref_path = crate::lsp::client::uri_to_path(&loc.uri);
                let ref_line = loc.range.start.line + 1;

                if let Some(ref p) = ref_path {
                    let rel = p.strip_prefix(&config.project_root).unwrap_or(p);
                    // Read the actual line for context
                    let abs = config.project_root.join(rel);
                    let line_content = std::fs::read_to_string(&abs)
                        .ok()
                        .and_then(|content| {
                            content.lines().nth(ref_line as usize - 1).map(|l| l.trim().to_string())
                        })
                        .unwrap_or_default();
                    output.push_str(&format!("  {}:{}: {}\n", rel.display(), ref_line, line_content));
                } else {
                    output.push_str(&format!("  {}:{}\n", loc.uri.as_str(), ref_line));
                }
            }
            Ok(ToolResult::ok(output))
        }
        Err(e) => Ok(ToolResult::err(format!("LSP error: {e}"))),
    }
}
