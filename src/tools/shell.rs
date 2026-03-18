//! shell tool — Execute shell commands with output capture.

use anyhow::Result;
use serde_json::Value;
use std::process::Command;

use crate::config::Config;
use super::ToolResult;

/// Maximum output lines (tail-priority for error visibility).
const MAX_OUTPUT_LINES: usize = 100;

/// Commands that are blocked for safety.
const BLOCKED_COMMANDS: &[&str] = &[
    "rm -rf /",
    "rm -rf ~",
    "mkfs",
    "dd if=",
    ":(){:|:&};:",
];

pub async fn execute(args: &Value, config: &Config) -> Result<ToolResult> {
    let command = args["command"].as_str().unwrap_or("");
    let _timeout = args["timeout"]
        .as_u64()
        .unwrap_or(60);

    if command.is_empty() {
        return Ok(ToolResult::err("Missing required parameter: command".into()));
    }

    // Safety check
    for blocked in BLOCKED_COMMANDS {
        if command.contains(blocked) {
            return Ok(ToolResult::err(format!(
                "Blocked dangerous command: {command}"
            )));
        }
    }

    let output = Command::new("sh")
        .arg("-c")
        .arg(command)
        .current_dir(&config.project_root)
        .output();

    match output {
        Ok(result) => {
            let stdout = String::from_utf8_lossy(&result.stdout);
            let stderr = String::from_utf8_lossy(&result.stderr);
            let exit_code = result.status.code().unwrap_or(-1);

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

            // Tail-truncate for error visibility
            let lines: Vec<&str> = combined.lines().collect();
            let truncated = lines.len() > MAX_OUTPUT_LINES;
            let display_lines = if truncated {
                let skip = lines.len() - MAX_OUTPUT_LINES;
                &lines[skip..]
            } else {
                &lines[..]
            };

            let mut output = format!("[shell: exit {exit_code}");
            if truncated {
                output.push_str(&format!(
                    ", showing last {MAX_OUTPUT_LINES} of {} lines",
                    lines.len()
                ));
            }
            output.push_str("]\n");
            output.push_str(&display_lines.join("\n"));

            if exit_code == 0 {
                Ok(ToolResult::ok(output))
            } else {
                Ok(ToolResult::ok(output)) // Still "ok" — the LLM needs to see the error
            }
        }
        Err(e) => Ok(ToolResult::err(format!("Failed to execute command: {e}"))),
    }
}
