//! Runtime execution helpers: dedicated LLM worker and shared tool worker pool.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc as std_mpsc};
use std::thread;

use crate::config::ModelRole;
use crate::llm::{ChatRequest, ChatResponse, ModelRouter};
use crate::tools::ToolResult;
use crate::{config::Config, tools::shell};
use tokio::sync::{mpsc, oneshot};

#[derive(Debug)]
pub enum LlmWorkerEvent {
    Token(String),
    Completed(Result<ChatResponse, String>),
}

struct LlmJob {
    role: ModelRole,
    request: ChatRequest,
    stream: bool,
    cancelled: Arc<AtomicBool>,
    events: mpsc::UnboundedSender<LlmWorkerEvent>,
}

pub struct LlmWorkerHandle {
    submit_tx: std_mpsc::Sender<LlmJob>,
}

impl LlmWorkerHandle {
    pub fn new(router: Arc<ModelRouter>) -> Self {
        let (submit_tx, submit_rx) = std_mpsc::channel::<LlmJob>();
        thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("failed to build llm worker runtime");
            while let Ok(job) = submit_rx.recv() {
                let router = router.clone();
                let events = job.events.clone();
                let cancelled = job.cancelled.clone();
                let role = job.role;
                let request = job.request.clone();
                let stream = job.stream;
                let result = runtime.block_on(async move {
                    if stream {
                        router
                            .chat_stream(
                                role,
                                &request,
                                |token| {
                                    let _ = events.send(LlmWorkerEvent::Token(token.to_string()));
                                },
                                &cancelled,
                            )
                            .await
                            .map_err(|e| e.to_string())
                    } else {
                        router.chat(role, &request).await.map_err(|e| e.to_string())
                    }
                });
                let _ = job.events.send(LlmWorkerEvent::Completed(result));
            }
        });
        Self { submit_tx }
    }

    pub fn submit(
        &self,
        role: ModelRole,
        request: ChatRequest,
        cancelled: Arc<AtomicBool>,
    ) -> mpsc::UnboundedReceiver<LlmWorkerEvent> {
        let (events_tx, events_rx) = mpsc::unbounded_channel();
        let job = LlmJob {
            role,
            request,
            stream: true,
            cancelled,
            events: events_tx,
        };
        let _ = self.submit_tx.send(job);
        events_rx
    }

    pub fn submit_non_streaming(
        &self,
        role: ModelRole,
        request: ChatRequest,
    ) -> mpsc::UnboundedReceiver<LlmWorkerEvent> {
        let (events_tx, events_rx) = mpsc::unbounded_channel();
        let job = LlmJob {
            role,
            request,
            stream: false,
            cancelled: Arc::new(AtomicBool::new(false)),
            events: events_tx,
        };
        let _ = self.submit_tx.send(job);
        events_rx
    }
}

type ToolJob = Box<dyn FnOnce() -> Result<ToolResult, String> + Send + 'static>;

struct ToolTask {
    _id: u64,
    job: ToolJob,
    result_tx: oneshot::Sender<Result<ToolResult, String>>,
}

#[derive(Debug, Clone, Copy)]
pub enum ShellControl {
    Continue,
    Kill,
}

#[derive(Debug)]
pub enum ShellWorkerEvent {
    TimedOut { command: String, timeout_secs: u64 },
    Completed(Result<ToolResult, String>),
}

pub struct ShellJobHandle {
    pub events_rx: mpsc::UnboundedReceiver<ShellWorkerEvent>,
    control_tx: std_mpsc::Sender<ShellControl>,
}

impl ShellJobHandle {
    pub fn send_control(&self, control: ShellControl) -> Result<(), String> {
        self.control_tx
            .send(control)
            .map_err(|_| "Shell worker dropped control channel".to_string())
    }
}

pub struct ToolWorkerPool {
    submit_tx: std_mpsc::Sender<ToolTask>,
    next_id: AtomicU64,
    _threads: Vec<thread::JoinHandle<()>>,
}

impl ToolWorkerPool {
    pub fn new(size: usize) -> Self {
        let (submit_tx, submit_rx) = std_mpsc::channel::<ToolTask>();
        let submit_rx = Arc::new(Mutex::new(submit_rx));
        let mut threads = Vec::new();
        for _ in 0..size.max(1) {
            let rx = submit_rx.clone();
            threads.push(thread::spawn(move || {
                while let Ok(task) = rx.lock().expect("tool worker rx poisoned").recv() {
                    let result = (task.job)();
                    let _ = task.result_tx.send(result);
                }
            }));
        }
        Self {
            submit_tx,
            next_id: AtomicU64::new(1),
            _threads: threads,
        }
    }

    pub fn submit<F>(&self, job: F) -> oneshot::Receiver<Result<ToolResult, String>>
    where
        F: FnOnce() -> Result<ToolResult, String> + Send + 'static,
    {
        let (result_tx, result_rx) = oneshot::channel();
        let task = ToolTask {
            _id: self.next_id.fetch_add(1, Ordering::Relaxed),
            job: Box::new(job),
            result_tx,
        };
        let _ = self.submit_tx.send(task);
        result_rx
    }

    pub fn submit_shell(
        &self,
        args: serde_json::Value,
        config: Config,
        cancelled: Arc<AtomicBool>,
    ) -> ShellJobHandle {
        let (events_tx, events_rx) = mpsc::unbounded_channel();
        let (control_tx, control_rx) = std_mpsc::channel::<ShellControl>();
        let (result_tx, _result_rx) = oneshot::channel();
        let task = ToolTask {
            _id: self.next_id.fetch_add(1, Ordering::Relaxed),
            job: Box::new(move || {
                let timeout_secs = args["timeout"]
                    .as_u64()
                    .unwrap_or(config.shell.default_timeout_secs);
                let command = args["command"].as_str().unwrap_or("").to_string();
                let mut running = shell::start(&args, &config).map_err(|e| e.to_string())?;

                loop {
                    match shell::wait(running, timeout_secs, &config, Some(cancelled.as_ref())) {
                        Ok(shell::ShellWaitOutcome::Completed(result)) => {
                            let _ = events_tx.send(ShellWorkerEvent::Completed(Ok(result)));
                            return Ok(ToolResult::ok(String::new()));
                        }
                        Ok(shell::ShellWaitOutcome::TimedOut(timed_out)) => {
                            running = timed_out;
                            let _ = events_tx.send(ShellWorkerEvent::TimedOut {
                                command: command.clone(),
                                timeout_secs,
                            });
                            match control_rx.recv() {
                                Ok(ShellControl::Continue) => {}
                                Ok(ShellControl::Kill) => {
                                    let result = shell::kill(running, timeout_secs);
                                    let _ = events_tx.send(ShellWorkerEvent::Completed(Ok(result)));
                                    return Ok(ToolResult::ok(String::new()));
                                }
                                Err(_) => {
                                    let result = shell::interrupt(running);
                                    let _ = events_tx.send(ShellWorkerEvent::Completed(Ok(result)));
                                    return Ok(ToolResult::ok(String::new()));
                                }
                            }
                        }
                        Ok(shell::ShellWaitOutcome::Interrupted(running)) => {
                            let result = shell::interrupt(running);
                            let _ = events_tx.send(ShellWorkerEvent::Completed(Ok(result)));
                            return Ok(ToolResult::ok(String::new()));
                        }
                        Err(e) => {
                            let _ = events_tx.send(ShellWorkerEvent::Completed(Err(format!(
                                "Shell error: {e}"
                            ))));
                            return Ok(ToolResult::ok(String::new()));
                        }
                    }
                }
            }),
            result_tx,
        };
        let _ = self.submit_tx.send(task);
        ShellJobHandle {
            events_rx,
            control_tx,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    fn test_config() -> (TempDir, Config) {
        let tmp = TempDir::new().unwrap();
        let mut config = Config::default();
        config.project_root = tmp.path().to_path_buf();
        std::fs::create_dir_all(config.miniswe_dir()).unwrap();
        (tmp, config)
    }

    #[tokio::test]
    async fn shell_job_times_out_then_kills() {
        let (_tmp, mut config) = test_config();
        config.shell.default_timeout_secs = 1;
        let pool = ToolWorkerPool::new(1);
        let mut job = pool.submit_shell(
            json!({"action":"shell","command":"sleep 5","timeout":1}),
            config,
            Arc::new(AtomicBool::new(false)),
        );

        match job.events_rx.recv().await {
            Some(ShellWorkerEvent::TimedOut { timeout_secs, .. }) => assert_eq!(timeout_secs, 1),
            other => panic!("expected timeout event, got {other:?}"),
        }

        job.send_control(ShellControl::Kill).unwrap();
        match job.events_rx.recv().await {
            Some(ShellWorkerEvent::Completed(Ok(result))) => {
                assert!(!result.success);
                assert!(result.content.contains("was killed by user"));
            }
            other => panic!("expected completed kill result, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn shell_job_times_out_then_continues() {
        let (_tmp, mut config) = test_config();
        config.shell.default_timeout_secs = 1;
        let pool = ToolWorkerPool::new(1);
        let mut job = pool.submit_shell(
            json!({"action":"shell","command":"sleep 2; printf ok","timeout":1}),
            config,
            Arc::new(AtomicBool::new(false)),
        );

        match job.events_rx.recv().await {
            Some(ShellWorkerEvent::TimedOut { timeout_secs, .. }) => assert_eq!(timeout_secs, 1),
            other => panic!("expected timeout event, got {other:?}"),
        }

        job.send_control(ShellControl::Continue).unwrap();
        loop {
            match job.events_rx.recv().await {
                Some(ShellWorkerEvent::TimedOut { .. }) => {
                    job.send_control(ShellControl::Continue).unwrap();
                }
                Some(ShellWorkerEvent::Completed(Ok(result))) => {
                    assert!(result.success);
                    assert!(result.content.contains("[shell: exit 0]"));
                    assert!(result.content.contains("ok"));
                    break;
                }
                other => panic!("expected continued timeout/completed result, got {other:?}"),
            }
        }
    }
}
