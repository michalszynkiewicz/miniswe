//! Compiler/checker subprocess helpers shared by edit orchestration and
//! LSP-fallback diagnostics.
//!
//! These functions are intentionally generic — they run an external checker
//! (cargo, tsc, mvn, …) with a timeout, parse its stderr for error
//! locations, and pull source context around those lines.

/// Run a check command with a timeout, draining pipes to prevent deadlock.
/// Returns Some((success, stderr)) or None if the command couldn't be spawned.
pub fn run_check_with_timeout(
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
pub(super) fn extract_error_locations(stderr: &str) -> Vec<(String, usize)> {
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
pub(super) fn read_source_context(
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
