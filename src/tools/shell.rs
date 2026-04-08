//! shell tool — Execute shell commands with output capture and timeout.
//!
//! Permission checks (blocklist + user approval) are handled by the
//! PermissionManager before this function is called.

use anyhow::{Result, anyhow};
use serde_json::Value;
use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use super::ToolResult;
use crate::config::Config;

/// Maximum characters per output line before truncation (display).
const MAX_LINE_CHARS: usize = 1000;

/// Maximum total bytes to read from stdout+stderr combined.
const MAX_OUTPUT_BYTES: usize = 512 * 1024; // 512KB

pub struct RunningShellCommand {
    child: Child,
    stdout_handle: JoinHandle<Vec<u8>>,
    stderr_handle: JoinHandle<Vec<u8>>,
}

pub enum ShellWaitOutcome {
    Completed(ToolResult),
    TimedOut(RunningShellCommand),
    Interrupted(RunningShellCommand),
}

pub async fn execute(args: &Value, config: &Config) -> Result<ToolResult> {
    let timeout_secs = args["timeout"]
        .as_u64()
        .unwrap_or(config.shell.default_timeout_secs);
    let running = match start(args, config) {
        Ok(r) => r,
        Err(e) => return Ok(ToolResult::err(e.to_string())),
    };
    match wait(running, timeout_secs, config, None) {
        Ok(ShellWaitOutcome::Completed(result)) => Ok(result),
        Ok(ShellWaitOutcome::TimedOut(mut running)) => {
            let _ = running.child.kill();
            Ok(render_killed_result(running, timeout_secs))
        }
        Ok(ShellWaitOutcome::Interrupted(mut running)) => {
            let _ = running.child.kill();
            Ok(render_interrupted_result(running))
        }
        Err(e) => Err(e),
    }
}

pub fn start(args: &Value, config: &Config) -> Result<RunningShellCommand> {
    let command = args["command"].as_str().unwrap_or("");
    if command.is_empty() {
        return Err(anyhow!(
            "Missing required parameter: command. Expected JSON arguments: {{\"action\":\"shell\",\"command\":\"cargo check\",\"timeout\":60}}."
        ));
    }

    // Spawn the child process
    let mut child = match Command::new("sh")
        .arg("-c")
        .arg(command)
        .current_dir(&config.project_root)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => return Err(anyhow!("Failed to execute command: {e}")),
    };

    // Drain stdout and stderr in background threads to prevent pipe deadlock.
    // If the child writes more than the OS pipe buffer (~64KB) and nobody reads,
    // the child blocks and try_wait() never sees it exit → false timeout.
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    let stdout_handle = std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(out) = stdout {
            let _ = out.take(MAX_OUTPUT_BYTES as u64).read_to_end(&mut buf);
        }
        buf
    });

    let stderr_handle = std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(err) = stderr {
            let _ = err.take(MAX_OUTPUT_BYTES as u64).read_to_end(&mut buf);
        }
        buf
    });

    Ok(RunningShellCommand {
        child,
        stdout_handle,
        stderr_handle,
    })
}

pub fn wait(
    mut running: RunningShellCommand,
    timeout_secs: u64,
    config: &Config,
    cancel: Option<&AtomicBool>,
) -> Result<ShellWaitOutcome> {
    let deadline = Instant::now() + Duration::from_secs(timeout_secs);
    loop {
        if cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
            return Ok(ShellWaitOutcome::Interrupted(running));
        }
        match running.child.try_wait() {
            Ok(Some(_)) => return Ok(ShellWaitOutcome::Completed(render_finished_result(
                running, config,
            ))),
            Ok(None) => {
                if Instant::now() > deadline {
                    return Ok(ShellWaitOutcome::TimedOut(running));
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(e) => {
                return Ok(ShellWaitOutcome::Completed(ToolResult::err(format!(
                    "Failed to wait for command: {e}"
                ))));
            }
        }
    }
}

pub fn kill(mut running: RunningShellCommand, timeout_secs: u64) -> ToolResult {
    let _ = running.child.kill();
    render_killed_result(running, timeout_secs)
}

fn render_killed_result(mut running: RunningShellCommand, timeout_secs: u64) -> ToolResult {
    let _ = running.child.wait();
    ToolResult::err(format!(
        "Command timed out after {timeout_secs}s and was killed by user."
    ))
}

fn render_interrupted_result(mut running: RunningShellCommand) -> ToolResult {
    let _ = running.child.wait();
    ToolResult::err("Command interrupted by user.".into())
}

fn render_finished_result(mut running: RunningShellCommand, config: &Config) -> ToolResult {
    let exit_code = running
        .child
        .wait()
        .map(|s| s.code().unwrap_or(-1))
        .unwrap_or(-1);

    // Collect output from drain threads
    let stdout_bytes = running.stdout_handle.join().unwrap_or_default();
    let stderr_bytes = running.stderr_handle.join().unwrap_or_default();

    let stdout = String::from_utf8_lossy(&stdout_bytes);
    let stderr = String::from_utf8_lossy(&stderr_bytes);

    let mut combined = String::new();
    if !stdout.is_empty() {
        combined.push_str(&stdout);
    }
    if !stderr.is_empty() {
        if !combined.is_empty() {
            combined.push('\n');
        }
        combined.push_str("[stderr]\n");
        combined.push_str(&stderr);
    }

    // Tail-truncate for error visibility, and cap long lines
    // Max output lines from context budget (~80 chars/line)
    let max_output_lines = config.tool_output_budget_chars() / 80;

    let lines: Vec<&str> = combined.lines().collect();
    let truncated = lines.len() > max_output_lines;

    let mut result = format!("[shell: exit {exit_code}");

    if truncated {
        // Save full output to file, show tail + pointer
        let cache_dir = config.miniswe_dir().join("shell_output");
        let _ = std::fs::create_dir_all(&cache_dir);
        let filename = format!("cmd_{}.txt", chrono::Local::now().format("%H%M%S"));
        let cache_path = cache_dir.join(&filename);
        let _ = std::fs::write(&cache_path, &combined);
        let rel_path = format!(".miniswe/shell_output/{filename}");

        result.push_str(&format!(
            ", showing last {} of {} lines",
            max_output_lines,
            lines.len()
        ));
        result.push_str("]\n");

        let skip = lines.len() - max_output_lines;
        for (i, line) in lines[skip..].iter().enumerate() {
            if i > 0 {
                result.push('\n');
            }
            if line.chars().count() > MAX_LINE_CHARS {
                result.push_str(&crate::truncate_chars(line, MAX_LINE_CHARS));
            } else {
                result.push_str(line);
            }
        }

        result.push_str(&format!(
            "\n\n[Full output saved to {rel_path} — use read_file(\"{rel_path}\") for more]"
        ));
    } else {
        result.push_str("]\n");
        for (i, line) in lines.iter().enumerate() {
            if i > 0 {
                result.push('\n');
            }
            if line.chars().count() > MAX_LINE_CHARS {
                result.push_str(&crate::truncate_chars(line, MAX_LINE_CHARS));
            } else {
                result.push_str(line);
            }
        }
    }

    if exit_code == 0 {
        ToolResult::ok(result)
    } else {
        ToolResult {
            content: result,
            success: false,
        }
    }
}
