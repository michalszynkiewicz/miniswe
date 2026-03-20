//! shell tool — Execute shell commands with output capture and timeout.
//!
//! Permission checks (blocklist + user approval) are handled by the
//! PermissionManager before this function is called.

use anyhow::Result;
use serde_json::Value;
use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::config::Config;
use super::ToolResult;

/// Maximum output lines (tail-priority for error visibility).
const MAX_OUTPUT_LINES: usize = 100;

/// Maximum characters per output line before truncation.
const MAX_LINE_CHARS: usize = 1000;

/// Maximum total bytes to read from stdout+stderr combined.
const MAX_OUTPUT_BYTES: usize = 512 * 1024; // 512KB

/// Default timeout in seconds.
const DEFAULT_TIMEOUT_SECS: u64 = 30;

pub async fn execute(args: &Value, config: &Config) -> Result<ToolResult> {
    let command = args["command"].as_str().unwrap_or("");
    let timeout_secs = args["timeout"]
        .as_u64()
        .unwrap_or(DEFAULT_TIMEOUT_SECS);

    if command.is_empty() {
        return Ok(ToolResult::err("Missing required parameter: command".into()));
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
        Err(e) => return Ok(ToolResult::err(format!("Failed to execute command: {e}"))),
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

    // Poll for completion with timeout
    let deadline = Instant::now() + Duration::from_secs(timeout_secs);
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break, // process exited
            Ok(None) => {
                if Instant::now() > deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Ok(ToolResult {
                        content: format!(
                            "Command timed out after {timeout_secs}s (killed).\n\
                             Tip: use a higher timeout parameter, or don't run \
                             long-lived servers with shell()."
                        ),
                        success: false,
                    });
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(e) => {
                return Ok(ToolResult::err(format!("Failed to wait for command: {e}")));
            }
        }
    }

    let exit_code = child.wait().map(|s| s.code().unwrap_or(-1)).unwrap_or(-1);

    // Collect output from drain threads
    let stdout_bytes = stdout_handle.join().unwrap_or_default();
    let stderr_bytes = stderr_handle.join().unwrap_or_default();

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
    let lines: Vec<&str> = combined.lines().collect();
    let truncated = lines.len() > MAX_OUTPUT_LINES;
    let display_lines = if truncated {
        let skip = lines.len() - MAX_OUTPUT_LINES;
        &lines[skip..]
    } else {
        &lines[..]
    };

    let mut result = format!("[shell: exit {exit_code}");
    if truncated {
        result.push_str(&format!(
            ", showing last {MAX_OUTPUT_LINES} of {} lines",
            lines.len()
        ));
    }
    result.push_str("]\n");

    // Join display lines, truncating any that exceed MAX_LINE_CHARS
    for (i, line) in display_lines.iter().enumerate() {
        if i > 0 {
            result.push('\n');
        }
        if line.len() > MAX_LINE_CHARS {
            result.push_str(&line[..MAX_LINE_CHARS]);
            result.push_str("...(truncated)");
        } else {
            result.push_str(line);
        }
    }

    if exit_code == 0 {
        Ok(ToolResult::ok(result))
    } else {
        Ok(ToolResult { content: result, success: false })
    }
}
