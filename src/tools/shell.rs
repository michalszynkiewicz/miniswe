//! shell tool — Execute shell commands with output capture and timeout.
//!
//! Permission checks (blocklist + user approval) are handled by the
//! PermissionManager before this function is called.

use anyhow::{Result, anyhow};
use serde_json::Value;
use std::fs::File;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use super::ToolResult;
use crate::config::Config;

/// Maximum characters per output line before truncation (display).
const MAX_LINE_CHARS: usize = 1000;

pub struct RunningShellCommand {
    child: Child,
    stdout_path: PathBuf,
    stderr_path: PathBuf,
}

impl Drop for RunningShellCommand {
    /// Kill the child and clean up temp files if the struct is dropped
    /// without going through `kill`/`interrupt`/`render_finished_result`
    /// (e.g. a panic or early `?` return in an outer caller).
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = cleanup_temp_file(&self.stdout_path);
        let _ = cleanup_temp_file(&self.stderr_path);
    }
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
        Ok(ShellWaitOutcome::TimedOut(running)) => Ok(kill(running, timeout_secs)),
        Ok(ShellWaitOutcome::Interrupted(running)) => Ok(interrupt(running)),
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

    let temp_dir = config.miniswe_dir().join("shell_tmp");
    std::fs::create_dir_all(&temp_dir)?;
    let unique = format!(
        "{}_{}",
        chrono::Local::now().format("%Y%m%d_%H%M%S_%f"),
        std::process::id()
    );
    let stdout_path = temp_dir.join(format!("{unique}_stdout.txt"));
    let stderr_path = temp_dir.join(format!("{unique}_stderr.txt"));
    let stdout_file = File::create(&stdout_path)?;
    let stderr_file = File::create(&stderr_path)?;

    let mut command_builder = Command::new("sh");
    command_builder
        .arg("-c")
        .arg(command)
        .current_dir(&config.project_root)
        .stdout(Stdio::from(stdout_file))
        .stderr(Stdio::from(stderr_file));

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command_builder.process_group(0);
    }

    let child = command_builder
        .spawn()
        .map_err(|e| anyhow!("Failed to execute command: {e}"))?;

    Ok(RunningShellCommand {
        child,
        stdout_path,
        stderr_path,
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
            Ok(Some(_)) => {
                return Ok(ShellWaitOutcome::Completed(render_finished_result(
                    running, config,
                )));
            }
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
    terminate_process_tree(&mut running.child, libc::SIGTERM);
    std::thread::sleep(Duration::from_millis(200));
    terminate_process_tree(&mut running.child, libc::SIGKILL);
    let _ = running.child.wait();
    let _ = cleanup_temp_file(&running.stdout_path);
    let _ = cleanup_temp_file(&running.stderr_path);
    ToolResult::err(format!(
        "Command timed out after {timeout_secs}s and was killed by user."
    ))
}

pub fn interrupt(mut running: RunningShellCommand) -> ToolResult {
    terminate_process_tree(&mut running.child, libc::SIGTERM);
    std::thread::sleep(Duration::from_millis(200));
    terminate_process_tree(&mut running.child, libc::SIGKILL);
    let _ = running.child.wait();
    let _ = cleanup_temp_file(&running.stdout_path);
    let _ = cleanup_temp_file(&running.stderr_path);
    ToolResult::err("Command interrupted by user.".into())
}

fn render_finished_result(mut running: RunningShellCommand, config: &Config) -> ToolResult {
    let exit_code = running
        .child
        .wait()
        .map(|s| s.code().unwrap_or(-1))
        .unwrap_or(-1);

    let stdout = std::fs::read_to_string(&running.stdout_path).unwrap_or_default();
    let stderr = std::fs::read_to_string(&running.stderr_path).unwrap_or_default();
    let _ = cleanup_temp_file(&running.stdout_path);
    let _ = cleanup_temp_file(&running.stderr_path);

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
    let max_output_lines = config.tool_output_budget_chars() / 80;
    let lines: Vec<&str> = combined.lines().collect();
    let truncated = lines.len() > max_output_lines;

    let mut result = format!("[shell: exit {exit_code}");

    if truncated {
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
        ToolResult::err(result)
    }
}

fn cleanup_temp_file(path: &PathBuf) -> std::io::Result<()> {
    if path.exists() {
        std::fs::remove_file(path)
    } else {
        Ok(())
    }
}

fn terminate_process_tree(child: &mut Child, signal: i32) {
    #[cfg(unix)]
    {
        let pid = child.id() as i32;
        unsafe {
            let _ = libc::kill(-pid, signal);
            let _ = libc::kill(pid, signal);
        }
    }
    #[cfg(not(unix))]
    {
        let _ = child.kill();
    }
}
