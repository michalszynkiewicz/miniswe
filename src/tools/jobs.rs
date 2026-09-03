//! `jobs` tool — manage long-running background shell commands.
//!
//! Jobs are created two ways:
//! - explicitly: `file(shell ..., background=true)` — the sanctioned form
//!   of the model's "cmd & echo $! > .pid" instinct (which the first jobs
//!   e2e showed defeats promotion and loses output);
//! - by promotion: a foreground command still running at the check-in
//!   threshold gets DETACHED (headless runs only — interactive runs keep
//!   the human continue/kill prompt).
//!
//! The model then paces its own monitoring loop:
//!
//!   jobs(action='wait', id=N, secs=60, check='kubectl get pods -n app')
//!
//! `wait` blocks up to `secs` (returns early on exit), then reports NEW
//! output since the last look plus — when `check` is given — the output of
//! that probe command. The probe is the skill-guided progress hook: domain
//! guidance ("monitor the deploy with X") maps 1:1 onto the `check` param.
//! Deliberately NO mechanical output-idle detection: k8s deploys sit
//! silent for minutes while healthy (image pulls) — cluster-state probes
//! are the real signal, and only the model/skill knows which command that
//! is. Design: session 2026-07-14 (pkg-mcp e2e post-mortem).

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use serde_json::Value;

use super::ToolResult;
use super::permissions::{Action, PermissionManager};
use super::shell::{self, RunningShellCommand};
use crate::config::Config;

/// Cap a single `wait` so one tool round stays bounded.
pub const MAX_WAIT_SECS: u64 = 300;
/// Default `wait` window when `secs` is omitted.
pub const DEFAULT_WAIT_SECS: u64 = 60;
/// Timeout for a `check` probe command — probes are meant to be quick
/// status reads, not work.
const CHECK_TIMEOUT_SECS: u64 = 60;
/// Max chars of new job output / probe output echoed per call.
const OUTPUT_TAIL_CHARS: usize = 4000;

struct Job {
    command: String,
    started: Instant,
    running: RunningShellCommand,
    stdout_seen: u64,
    stderr_seen: u64,
}

/// Registry of live background jobs. One per agent session.
#[derive(Default)]
pub struct JobRegistry {
    jobs: Mutex<HashMap<u64, Job>>,
    next_id: AtomicU64,
}

impl JobRegistry {
    /// Register a detached command; returns the job id and the output
    /// produced so far (tail-capped), for the promotion message.
    pub fn register(&self, command: &str, running: RunningShellCommand) -> (u64, String) {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed) + 1;
        let mut job = Job {
            command: command.to_string(),
            started: Instant::now(),
            running,
            stdout_seen: 0,
            stderr_seen: 0,
        };
        let so_far = read_new_output(&mut job);
        self.jobs.lock().insert(id, job);
        (id, so_far)
    }

    pub fn is_empty(&self) -> bool {
        self.jobs.lock().is_empty()
    }
}

/// Read output appended since the last look, advancing the per-job offsets.
fn read_new_output(job: &mut Job) -> String {
    let stdout = read_from(&job.running.stdout_path, &mut job.stdout_seen);
    let stderr = read_from(&job.running.stderr_path, &mut job.stderr_seen);
    let mut out = String::new();
    if !stdout.is_empty() {
        out.push_str(&stdout);
    }
    if !stderr.is_empty() {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str("[stderr]\n");
        out.push_str(&stderr);
    }
    // Tail-cap: recent output matters most for progress judgment.
    if out.chars().count() > OUTPUT_TAIL_CHARS {
        let tail: String = out
            .chars()
            .rev()
            .take(OUTPUT_TAIL_CHARS)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        out = format!("[...tail of new output]\n{tail}");
    }
    out
}

fn read_from(path: &Path, seen: &mut u64) -> String {
    let Ok(mut f) = std::fs::File::open(path) else {
        return String::new();
    };
    if f.seek(SeekFrom::Start(*seen)).is_err() {
        return String::new();
    }
    let mut buf = Vec::new();
    if f.read_to_end(&mut buf).is_err() {
        return String::new();
    }
    *seen += buf.len() as u64;
    String::from_utf8_lossy(&buf).into_owned()
}

/// Execute a `jobs` tool call. Owns the wait loop; polls `cancelled` so
/// Ctrl+C interrupts a wait without killing the job.
pub async fn execute(
    args: &Value,
    config: &Config,
    perms: &PermissionManager,
    registry: &JobRegistry,
    cancelled: Option<&AtomicBool>,
) -> ToolResult {
    let action = args["action"].as_str().unwrap_or("");
    match action {
        "status" => status(args, config, registry),
        "wait" => wait(args, config, perms, registry, cancelled).await,
        "kill" => kill(args, config, registry),
        // `action` must be the literal verb, not the command text. Models that
        // get this wrong re-send the command in `action` and only add `command`
        // alongside it, so echo the exact corrected call rather than naming the
        // verbs — `json!` keeps the example valid when the command contains
        // quotes or shell metacharacters.
        other => ToolResult::err(match other {
            "" => "shell: 'action' is required: run|wait|status|kill. \
                   Run a command: {\"action\":\"run\",\"command\":\"ls -la\"}"
                .to_string(),
            cmd => format!(
                "shell: 'action' must be literally run|wait|status|kill, not the command text. \
                 Retry as: {}",
                serde_json::json!({"action": "run", "command": cmd})
            ),
        }),
    }
}

fn require_id(args: &Value, registry: &JobRegistry) -> Result<u64, ToolResult> {
    match args["id"].as_u64() {
        Some(id) => Ok(id),
        None => {
            // Single-job convenience: with exactly one live job, id is implied.
            let jobs = registry.jobs.lock();
            match jobs.len() {
                1 => Ok(*jobs.keys().next().expect("len checked")),
                0 => Err(ToolResult::err(
                    "jobs: no background jobs exist. Start one with \
                     shell(action='run', command=..., background=true); long-running \
                     foreground commands are also promoted to jobs automatically."
                        .into(),
                )),
                n => Err(ToolResult::err(format!(
                    "jobs: 'id' is required ({n} jobs live). Use jobs(action='status') to list."
                ))),
            }
        }
    }
}

/// Start a shell command directly as a background job (the model passed
/// `background=true`): the sanctioned replacement for self-backgrounding
/// with `&`, which loses output capture and orphans the process. Returns
/// the model-facing start message.
pub fn start_background(args: &Value, config: &Config, registry: &JobRegistry) -> ToolResult {
    let raw = match super::args::require_str(args, "command") {
        Ok(c) => c.to_string(),
        Err(e) => return ToolResult::err(e),
    };
    // background=true + a trailing '&' double-backgrounds: the sh wrapper
    // exits instantly and the real process escapes the registry as an
    // untracked orphan (jobs e2e iteration 3: the "job" reported FINISHED
    // while the deploy kept running unkillable). Models append '&' from
    // habit despite guidance — strip it mechanically.
    let mut command = raw.trim().to_string();
    let mut stripped_amp = false;
    while command.ends_with('&') {
        command.pop();
        command = command.trim_end().to_string();
        stripped_amp = true;
    }
    if command.is_empty() {
        return ToolResult::err("background start failed: empty command".into());
    }
    let mut start_args = args.clone();
    start_args["command"] = Value::String(command.clone());
    let running = match shell::start(&start_args, config) {
        Ok(r) => r,
        Err(e) => return ToolResult::err(format!("background start failed: {e}")),
    };
    let (id, so_far) = registry.register(&command, running);
    let mut msg = format!("[shell: started as background job {id}]\n  $ {command}\n");
    if stripped_amp {
        msg.push_str(
            "(note: trailing '&' removed — background=true already runs it in the background)\n",
        );
    }
    if !so_far.is_empty() {
        msg.push_str("Output so far:\n");
        msg.push_str(&so_far);
        if !so_far.ends_with('\n') {
            msg.push('\n');
        }
    }
    msg.push_str(&format!(
        "It runs in the background. Wait and monitor with \
         shell(action='wait', id={id}, secs=60, check='<status command>') — use the \
         progress-check command your task guidance recommends. \
         shell(action='kill', id={id}) stops it."
    ));
    ToolResult::ok(msg)
}

fn finish_line(id: u64, command: &str, result: &ToolResult) -> String {
    // Distinct FAILED banner: a finished job is removed from the registry on
    // first observation, so this one line is the only place its exit status
    // ever surfaces — and the agent loop's failed-job bookkeeping keys off it.
    let verdict = if result.success { "FINISHED" } else { "FAILED" };
    format!(
        "[job {id} {verdict}]  $ {command}\n{}",
        result.content.as_str()
    )
}

fn status(args: &Value, config: &Config, registry: &JobRegistry) -> ToolResult {
    // Specific id → detailed status for that job; no id → list all.
    if let Some(id) = args["id"].as_u64() {
        return status_one(id, config, registry);
    }
    let ids: Vec<u64> = {
        let jobs = registry.jobs.lock();
        if jobs.is_empty() {
            return ToolResult::ok("No background jobs.".into());
        }
        jobs.keys().copied().collect()
    };
    let mut out = String::new();
    let mut any_failed = false;
    for id in ids {
        let r = status_one(id, config, registry);
        any_failed |= !r.success;
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&r.content);
    }
    // A FAILED child must not be laundered into an ok aggregate — the agent
    // loop keys failure recording off the result's success flag.
    if any_failed {
        ToolResult::err(out)
    } else {
        ToolResult::ok(out)
    }
}

fn status_one(id: u64, config: &Config, registry: &JobRegistry) -> ToolResult {
    let mut jobs = registry.jobs.lock();
    let Some(job) = jobs.get_mut(&id) else {
        return ToolResult::err(format!("jobs: no job with id {id}."));
    };
    match job.running.child.try_wait() {
        Ok(Some(_)) => {
            let job = jobs.remove(&id).expect("checked above");
            let result = shell::render_finished_result(job.running, config);
            let content = finish_line(id, &job.command, &result);
            if result.success {
                ToolResult::ok(content)
            } else {
                ToolResult::err(content)
            }
        }
        Ok(None) => {
            let new = read_new_output(job);
            let elapsed = job.started.elapsed().as_secs();
            let mut content = format!(
                "[job {id} running, {elapsed}s since promotion]  $ {}\n",
                job.command
            );
            if new.is_empty() {
                content.push_str("(no new output since last check)");
            } else {
                content.push_str("New output:\n");
                content.push_str(&new);
            }
            ToolResult::ok(content)
        }
        Err(e) => ToolResult::err(format!("jobs: cannot poll job {id}: {e}")),
    }
}

async fn wait(
    args: &Value,
    config: &Config,
    perms: &PermissionManager,
    registry: &JobRegistry,
    cancelled: Option<&AtomicBool>,
) -> ToolResult {
    let id = match require_id(args, registry) {
        Ok(id) => id,
        Err(r) => return r,
    };
    let secs = args["secs"]
        .as_u64()
        .unwrap_or(DEFAULT_WAIT_SECS)
        .min(MAX_WAIT_SECS);
    let deadline = Instant::now() + Duration::from_secs(secs);

    // Poll without holding the lock across sleeps.
    let finished = loop {
        {
            let mut jobs = registry.jobs.lock();
            let Some(job) = jobs.get_mut(&id) else {
                return ToolResult::err(format!("jobs: no job with id {id}."));
            };
            match job.running.child.try_wait() {
                Ok(Some(_)) => break true,
                Ok(None) => {}
                Err(e) => return ToolResult::err(format!("jobs: cannot poll job {id}: {e}")),
            }
        }
        if Instant::now() >= deadline {
            break false;
        }
        if cancelled.is_some_and(|c| c.load(Ordering::Relaxed)) {
            return ToolResult::err(format!(
                "jobs: wait interrupted by user (job {id} keeps running)."
            ));
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    };

    let mut job_failed = false;
    let mut content = if finished {
        let job = {
            let mut jobs = registry.jobs.lock();
            jobs.remove(&id).expect("present in loop above")
        };
        let result = shell::render_finished_result(job.running, config);
        job_failed = !result.success;
        finish_line(id, &job.command, &result)
    } else {
        let mut jobs = registry.jobs.lock();
        let job = jobs.get_mut(&id).expect("present in loop above");
        let new = read_new_output(job);
        let elapsed = job.started.elapsed().as_secs();
        let mut c = format!(
            "[job {id} STILL RUNNING after waiting {secs}s ({elapsed}s total)]  $ {}\n",
            job.command
        );
        if new.is_empty() {
            c.push_str("(no new output)");
        } else {
            c.push_str("New output:\n");
            c.push_str(&new);
        }
        c
    };

    // Optional probe: run AFTER the wait window so every monitoring cycle
    // is paced (wait-then-look, never a tight loop).
    if let Some(check) = args["check"].as_str().filter(|c| !c.trim().is_empty()) {
        if let Err(e) = perms.check(&Action::Shell(check.to_string())) {
            content.push_str(&format!("\n[check denied: {e}]"));
        } else {
            let check_args = serde_json::json!({
                "command": check,
                "timeout": CHECK_TIMEOUT_SECS,
            });
            match shell::execute(&check_args, config).await {
                Ok(r) => {
                    content.push_str(&format!("\nCheck `{check}`:\n{}", r.content));
                }
                Err(e) => content.push_str(&format!("\n[check failed to run: {e}]")),
            }
        }
    }

    // Mirror status_one: a job that exited non-zero is an err result, even
    // when observed through wait — the flow the start banner recommends.
    if job_failed {
        ToolResult::err(content)
    } else {
        ToolResult::ok(content)
    }
}

fn kill(args: &Value, _config: &Config, registry: &JobRegistry) -> ToolResult {
    let id = match require_id(args, registry) {
        Ok(id) => id,
        Err(r) => return r,
    };
    let Some(mut job) = registry.jobs.lock().remove(&id) else {
        return ToolResult::err(format!("jobs: no job with id {id}."));
    };
    // Grab any final output before the kill discards the temp files.
    let last = read_new_output(&mut job);
    let elapsed = job.started.elapsed().as_secs();
    let _ = shell::kill(job.running, elapsed);
    let mut content = format!("[job {id} killed after {elapsed}s]  $ {}", job.command);
    if !last.is_empty() {
        content.push_str("\nFinal output:\n");
        content.push_str(&last);
    }
    ToolResult::ok(content)
}

/// The tool-result message injected when a foreground command is promoted.
pub fn promotion_message(id: u64, command: &str, timeout_secs: u64, output_so_far: &str) -> String {
    let mut msg = format!(
        "[shell: still running after {timeout_secs}s — promoted to background job {id}]\n  $ {command}\n"
    );
    if output_so_far.is_empty() {
        msg.push_str("(no output yet)\n");
    } else {
        msg.push_str("Output so far:\n");
        msg.push_str(output_so_far);
        if !output_so_far.ends_with('\n') {
            msg.push('\n');
        }
    }
    msg.push_str(&format!(
        "The command keeps running. Do NOT re-run it. \
         Wait and monitor with shell(action='wait', id={id}, secs=60, check='<status command>') — \
         use the progress-check command your task guidance recommends (e.g. kubectl get pods). \
         shell(action='kill', id={id}) stops it."
    ));
    msg
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_config(dir: &std::path::Path) -> Config {
        let mut cfg = Config::default();
        cfg.project_root = dir.to_path_buf();
        cfg
    }

    fn start_job(cfg: &Config, cmd: &str) -> RunningShellCommand {
        shell::start(&serde_json::json!({ "command": cmd }), cfg).unwrap()
    }

    #[tokio::test]
    async fn wait_returns_final_result_when_job_exits() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = scratch_config(tmp.path());
        let registry = JobRegistry::default();
        let running = start_job(&cfg, "echo done-marker");
        let (id, _) = registry.register("echo done-marker", running);

        let perms = PermissionManager::headless(&cfg);
        let args = serde_json::json!({ "action": "wait", "id": id, "secs": 5 });
        let r = execute(&args, &cfg, &perms, &registry, None).await;
        assert!(r.success, "{}", r.content);
        assert!(r.content.contains("FINISHED"), "{}", r.content);
        assert!(r.content.contains("done-marker"), "{}", r.content);
        assert!(registry.is_empty(), "finished job must be removed");
    }

    #[tokio::test]
    async fn wait_on_failed_job_returns_err_with_failed_banner() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = scratch_config(tmp.path());
        let registry = JobRegistry::default();
        let running = start_job(&cfg, "echo boom; exit 3");
        let (id, _) = registry.register("echo boom; exit 3", running);

        let perms = PermissionManager::headless(&cfg);
        let args = serde_json::json!({ "action": "wait", "id": id, "secs": 5 });
        let r = execute(&args, &cfg, &perms, &registry, None).await;
        assert!(
            !r.success,
            "failed job observed via wait must be an err result: {}",
            r.content
        );
        assert!(r.content.contains("FAILED]"), "{}", r.content);
        assert!(r.content.contains("boom"), "{}", r.content);
        assert!(registry.is_empty(), "finished job must be removed");
    }

    #[tokio::test]
    async fn aggregate_status_errs_when_a_job_failed() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = scratch_config(tmp.path());
        let registry = JobRegistry::default();
        let running = start_job(&cfg, "exit 7");
        registry.register("exit 7", running);

        let perms = PermissionManager::headless(&cfg);
        let args = serde_json::json!({ "action": "status" });
        let mut observed = None;
        for _ in 0..50 {
            let r = execute(&args, &cfg, &perms, &registry, None).await;
            if r.content.contains("FAILED]") {
                observed = Some(r);
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        let r = observed.expect("job never reported FAILED via aggregate status");
        assert!(
            !r.success,
            "aggregate status with a failed child must be an err result: {}",
            r.content
        );
    }

    #[tokio::test]
    async fn wait_times_out_and_runs_check_probe() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = scratch_config(tmp.path());
        let registry = JobRegistry::default();
        let running = start_job(&cfg, "echo progress-line; sleep 30");
        let (id, _) = registry.register("sleep-job", running);

        let perms = PermissionManager::headless(&cfg);
        let args = serde_json::json!({
            "action": "wait", "id": id, "secs": 1, "check": "echo probe-result"
        });
        let r = execute(&args, &cfg, &perms, &registry, None).await;
        assert!(r.success, "{}", r.content);
        assert!(r.content.contains("STILL RUNNING"), "{}", r.content);
        assert!(r.content.contains("progress-line"), "{}", r.content);
        assert!(r.content.contains("probe-result"), "{}", r.content);
        assert!(!registry.is_empty(), "running job must stay registered");

        // cleanup
        let _ = execute(
            &serde_json::json!({ "action": "kill", "id": id }),
            &cfg,
            &perms,
            &registry,
            None,
        )
        .await;
    }

    #[tokio::test]
    async fn status_reports_only_new_output() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = scratch_config(tmp.path());
        let registry = JobRegistry::default();
        let running = start_job(&cfg, "echo first; sleep 0.5; echo second; sleep 30");
        let (id, _) = registry.register("staged-output", running);
        let perms = PermissionManager::headless(&cfg);

        tokio::time::sleep(Duration::from_millis(200)).await;
        let r1 = execute(
            &serde_json::json!({ "action": "status", "id": id }),
            &cfg,
            &perms,
            &registry,
            None,
        )
        .await;
        assert!(r1.content.contains("first"), "{}", r1.content);

        tokio::time::sleep(Duration::from_millis(600)).await;
        let r2 = execute(
            &serde_json::json!({ "action": "status", "id": id }),
            &cfg,
            &perms,
            &registry,
            None,
        )
        .await;
        assert!(r2.content.contains("second"), "{}", r2.content);
        assert!(
            !r2.content.contains("first"),
            "already-seen output must not repeat: {}",
            r2.content
        );

        let _ = execute(
            &serde_json::json!({ "action": "kill", "id": id }),
            &cfg,
            &perms,
            &registry,
            None,
        )
        .await;
    }

    #[tokio::test]
    async fn kill_terminates_and_removes() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = scratch_config(tmp.path());
        let registry = JobRegistry::default();
        let running = start_job(&cfg, "sleep 300");
        let pid = running.child.id();
        let (id, _) = registry.register("sleeper", running);
        let perms = PermissionManager::headless(&cfg);

        let r = execute(
            &serde_json::json!({ "action": "kill", "id": id }),
            &cfg,
            &perms,
            &registry,
            None,
        )
        .await;
        assert!(r.success, "{}", r.content);
        assert!(registry.is_empty());
        #[cfg(unix)]
        {
            std::thread::sleep(Duration::from_millis(300));
            let alive = unsafe { libc::kill(pid as i32, 0) };
            assert_eq!(alive, -1, "job process must be dead");
        }
    }

    #[tokio::test]
    async fn single_live_job_implies_id() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = scratch_config(tmp.path());
        let registry = JobRegistry::default();
        let running = start_job(&cfg, "sleep 30");
        let (_id, _) = registry.register("solo", running);
        let perms = PermissionManager::headless(&cfg);

        let r = execute(
            &serde_json::json!({ "action": "wait", "secs": 1 }),
            &cfg,
            &perms,
            &registry,
            None,
        )
        .await;
        assert!(r.success, "id should be inferred: {}", r.content);
        let _ = execute(
            &serde_json::json!({ "action": "kill" }),
            &cfg,
            &perms,
            &registry,
            None,
        )
        .await;
    }

    #[tokio::test]
    async fn start_background_registers_and_teaches_management() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = scratch_config(tmp.path());
        let registry = JobRegistry::default();

        let r = start_background(
            &serde_json::json!({ "command": "echo bg-out; sleep 0.3", "background": true }),
            &cfg,
            &registry,
        );
        assert!(r.success, "{}", r.content);
        assert!(
            r.content.contains("started as background job 1"),
            "{}",
            r.content
        );
        assert!(
            r.content.contains("shell(action='wait', id=1"),
            "{}",
            r.content
        );
        assert!(!registry.is_empty());

        // The job runs to completion and reports via wait.
        let perms = PermissionManager::headless(&cfg);
        let w = execute(
            &serde_json::json!({ "action": "wait", "id": 1, "secs": 5 }),
            &cfg,
            &perms,
            &registry,
            None,
        )
        .await;
        assert!(w.content.contains("FINISHED"), "{}", w.content);
        assert!(w.content.contains("bg-out"), "{}", w.content);
    }

    #[tokio::test]
    async fn trailing_ampersand_is_stripped_so_the_job_stays_tracked() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = scratch_config(tmp.path());
        let registry = JobRegistry::default();

        let r = start_background(
            &serde_json::json!({ "command": "sleep 30 &", "background": true }),
            &cfg,
            &registry,
        );
        assert!(r.success, "{}", r.content);
        assert!(r.content.contains("trailing '&' removed"), "{}", r.content);

        // With the '&' the sh wrapper would exit instantly and status would
        // report FINISHED; stripped, the job is genuinely still running.
        let perms = PermissionManager::headless(&cfg);
        let s = execute(
            &serde_json::json!({ "action": "status", "id": 1 }),
            &cfg,
            &perms,
            &registry,
            None,
        )
        .await;
        assert!(s.content.contains("running"), "{}", s.content);
        let _ = execute(
            &serde_json::json!({ "action": "kill", "id": 1 }),
            &cfg,
            &perms,
            &registry,
            None,
        )
        .await;
    }

    #[tokio::test]
    async fn empty_registry_error_teaches_background_flag() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = scratch_config(tmp.path());
        let registry = JobRegistry::default();
        let perms = PermissionManager::headless(&cfg);

        let r = execute(
            &serde_json::json!({ "action": "wait", "secs": 1 }),
            &cfg,
            &perms,
            &registry,
            None,
        )
        .await;
        assert!(!r.success);
        assert!(
            r.content.contains("no background jobs exist"),
            "{}",
            r.content
        );
        assert!(r.content.contains("background=true"), "{}", r.content);
    }

    #[test]
    fn promotion_message_teaches_the_monitoring_loop() {
        let msg = promotion_message(3, "pack package deploy x", 60, "some output");
        assert!(msg.contains("job 3"));
        assert!(msg.contains("Do NOT re-run"));
        assert!(msg.contains("shell(action='wait', id=3"));
        assert!(msg.contains("check="));
    }
}
