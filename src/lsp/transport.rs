//! JSON-RPC transport over stdio with Content-Length framing.

use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::process::ChildStdin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::mpsc as std_mpsc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use serde_json::Value;
use tokio::sync::oneshot;

/// State of a `$/progress` token tracked by [`LspTransport`]. Tokens
/// are removed when their `end` message arrives — `is_idle` then reads
/// "no in-flight progress" as "the server has nothing left to do".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProgressKind {
    Begin,
    Report,
}

/// Max messages queued for the stdin writer thread before senders start
/// waiting (and, after `write_timeout`, erroring out).
const WRITE_QUEUE_CAP: usize = 64;

/// JSON-RPC transport for LSP communication.
pub struct LspTransport {
    /// Queue feeding the dedicated stdin writer thread. Bounded: when the
    /// server stops draining its stdin pipe (the 08-22/08-24 bench wedge),
    /// the queue fills and `write_message` errors out after `write_timeout`
    /// instead of blocking forever in a sync `write_all` that no tokio
    /// timer can preempt.
    write_tx: std_mpsc::SyncSender<Vec<u8>>,
    /// How long `write_message` waits for queue space before declaring the
    /// server wedged. 5s in production, short in tests.
    write_timeout: Duration,
    pub(crate) pending: DashMap<i64, oneshot::Sender<Value>>,
    pub(crate) diagnostics: DashMap<String, Vec<lsp_types::Diagnostic>>,
    /// In-flight `$/progress` tokens. Inserted on `begin`, refreshed on
    /// `report`, removed on `end`. Emptiness == server is idle.
    pub(crate) progress: DashMap<String, ProgressKind>,
    next_id: AtomicI64,
    /// Shared with the stdin writer thread (which sets it when the pipe
    /// breaks), hence `Arc` rather than a bare field.
    pub(crate) crashed: Arc<AtomicBool>,
}

impl LspTransport {
    pub fn new(stdin: ChildStdin) -> Self {
        Self::with_write_timeout(stdin, Duration::from_secs(5))
    }

    /// Like [`Self::new`] but with an explicit stdin-write deadline —
    /// tests use a short one so exercising the wedged-server path does
    /// not cost 5 wall-clock seconds.
    pub(crate) fn with_write_timeout(stdin: ChildStdin, write_timeout: Duration) -> Self {
        // All stdin writes happen on this dedicated thread. A wedged
        // server (alive but not reading its pipe) blocks the thread in
        // `write_all` once the pipe buffer fills — but that only ever
        // stalls this thread, never an agent thread: senders enqueue
        // with a deadline and fail fast. Killing the server closes the
        // pipe's read end, the blocked write returns EPIPE, and the
        // thread exits.
        let (write_tx, write_rx) = std_mpsc::sync_channel::<Vec<u8>>(WRITE_QUEUE_CAP);
        let crashed = Arc::new(AtomicBool::new(false));
        let crashed_writer = Arc::clone(&crashed);
        std::thread::spawn(move || {
            let mut writer = BufWriter::new(stdin);
            while let Ok(buf) = write_rx.recv() {
                if writer
                    .write_all(&buf)
                    .and_then(|()| writer.flush())
                    .is_err()
                {
                    // Pipe broken — the server process is gone. Senders
                    // see a disconnected queue from now on.
                    crashed_writer.store(true, Ordering::Relaxed);
                    return;
                }
            }
        });
        Self {
            write_tx,
            write_timeout,
            pending: DashMap::new(),
            diagnostics: DashMap::new(),
            progress: DashMap::new(),
            next_id: AtomicI64::new(1),
            crashed,
        }
    }

    /// Returns `true` if the server has no in-flight `$/progress` tokens.
    /// A `true` result means it's safe to read diagnostics: any analysis
    /// the server was running has ended (its progress token was closed).
    pub(crate) fn is_idle(&self) -> bool {
        self.progress.is_empty()
    }

    /// Apply a parsed `$/progress` notification to the in-flight token
    /// map. Pulled out of `reader_loop` so it can be unit-tested with
    /// synthetic JSON.
    pub(crate) fn apply_progress(&self, params: &Value) {
        apply_progress_to_map(&self.progress, params);
    }

    /// Send a JSON-RPC request. Returns a receiver for the response.
    pub fn send_request(
        &self,
        method: &str,
        params: Value,
    ) -> anyhow::Result<oneshot::Receiver<Value>> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let msg = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        self.write_message(&msg)?;

        let (tx, rx) = oneshot::channel();
        self.pending.insert(id, tx);
        Ok(rx)
    }

    /// Send a JSON-RPC notification (no response expected).
    pub fn send_notification(&self, method: &str, params: Value) -> anyhow::Result<()> {
        let msg = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        self.write_message(&msg)
    }

    /// Frame a message and enqueue it for the stdin writer thread.
    ///
    /// Never blocks indefinitely: if the queue stays full for
    /// `write_timeout`, the server is not draining its pipe — a busy
    /// rust-analyzer still reads stdin promptly, so that is wedge
    /// evidence, not load. The error wording ("failed to write") is
    /// deliberately what `LspClient::infra_class` classifies as a HARD
    /// infra failure, routing callers into the restart path.
    fn write_message(&self, msg: &Value) -> anyhow::Result<()> {
        let body = serde_json::to_string(msg)?;
        let mut framed = format!("Content-Length: {}\r\n\r\n", body.len()).into_bytes();
        framed.extend_from_slice(body.as_bytes());

        let deadline = Instant::now() + self.write_timeout;
        let mut msg_bytes = framed;
        loop {
            match self.write_tx.try_send(msg_bytes) {
                Ok(()) => return Ok(()),
                Err(std_mpsc::TrySendError::Full(again)) => {
                    if Instant::now() >= deadline {
                        anyhow::bail!(
                            "failed to write to lsp stdin: write queue not drained for {:?} \
                             (server wedged — alive but not reading its pipe)",
                            self.write_timeout
                        );
                    }
                    msg_bytes = again;
                    // Bounded busy-wait (<= write_timeout total). Callers
                    // may sit on an async runtime, but a short bounded
                    // sleep beats wiring async plumbing through every
                    // notification site.
                    std::thread::sleep(Duration::from_millis(20));
                }
                Err(std_mpsc::TrySendError::Disconnected(_)) => {
                    anyhow::bail!(
                        "failed to write to lsp stdin: writer thread exited (server process gone)"
                    );
                }
            }
        }
    }

    /// Run the reader loop on stdout. Call from a blocking thread.
    /// Dispatches responses to pending requests and caches diagnostic notifications.
    pub fn reader_loop(transport: Arc<LspTransport>, stdout: std::process::ChildStdout) {
        let mut reader = BufReader::new(stdout);

        loop {
            // Parse Content-Length header
            let content_length = match read_content_length(&mut reader) {
                Some(len) => len,
                None => {
                    // EOF or malformed — server died
                    transport.crashed.store(true, Ordering::Relaxed);
                    break;
                }
            };

            // Read body
            let mut body = vec![0u8; content_length];
            if reader.read_exact(&mut body).is_err() {
                transport.crashed.store(true, Ordering::Relaxed);
                break;
            }

            let msg: Value = match serde_json::from_slice(&body) {
                Ok(v) => v,
                Err(_) => continue,
            };

            // Dispatch: response (has "id") or notification (no "id")
            if let Some(id) = msg.get("id").and_then(|v| v.as_i64()) {
                // Response — find pending request
                if let Some((_, tx)) = transport.pending.remove(&id) {
                    let _ = tx.send(msg);
                }
            } else if let Some(method) = msg.get("method").and_then(|v| v.as_str()) {
                match method {
                    "textDocument/publishDiagnostics" => {
                        if let Some(params) = msg.get("params")
                            && let Ok(diag_params) = serde_json::from_value::<
                                lsp_types::PublishDiagnosticsParams,
                            >(params.clone())
                        {
                            let uri = diag_params.uri.as_str().to_string();
                            transport.diagnostics.insert(uri, diag_params.diagnostics);
                        }
                    }
                    "$/progress" => {
                        if let Some(params) = msg.get("params") {
                            transport.apply_progress(params);
                        }
                    }
                    _ => {} // ignore other notifications
                }
            }
        }

        // Server gone — wake all pending requests
        transport.pending.retain(|_, _| false);
    }
}

/// Apply a parsed `$/progress` notification to a token map. Standalone
/// so unit tests can drive it with a fresh `DashMap` and synthetic JSON
/// — no real LSP transport / `ChildStdin` required.
pub(crate) fn apply_progress_to_map(map: &DashMap<String, ProgressKind>, params: &Value) {
    let Some((token, update)) = parse_progress_params(params) else {
        return;
    };
    match update {
        ProgressUpdate::Begin => {
            map.insert(token, ProgressKind::Begin);
        }
        ProgressUpdate::Report => {
            map.insert(token, ProgressKind::Report);
        }
        ProgressUpdate::End => {
            map.remove(&token);
        }
    }
}

/// Result of parsing a single `$/progress` notification.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ProgressUpdate {
    Begin,
    Report,
    End,
}

/// Parse the `params` field of a `$/progress` notification into
/// `(token, update_kind)`. Tokens may be either strings or integers in
/// the LSP wire format; we always return them as a string for indexing.
/// Returns `None` if the message doesn't carry the expected shape — we'd
/// rather drop a malformed update than panic the reader loop.
pub(crate) fn parse_progress_params(params: &Value) -> Option<(String, ProgressUpdate)> {
    let token = match params.get("token") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Number(n)) => n.to_string(),
        _ => return None,
    };
    let kind = params
        .get("value")
        .and_then(|v| v.get("kind"))
        .and_then(|k| k.as_str())?;
    let update = match kind {
        "begin" => ProgressUpdate::Begin,
        "report" => ProgressUpdate::Report,
        "end" => ProgressUpdate::End,
        _ => return None,
    };
    Some((token, update))
}

/// Parse Content-Length header from reader. Returns None on EOF.
fn read_content_length(reader: &mut BufReader<std::process::ChildStdout>) -> Option<usize> {
    let mut content_length: Option<usize> = None;

    loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => return None, // EOF
            Ok(_) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    // Empty line = end of headers
                    return content_length;
                }
                if let Some(value) = trimmed.strip_prefix("Content-Length: ") {
                    content_length = value.parse().ok();
                }
            }
            Err(_) => return None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_progress_params_handles_string_token_and_all_kinds() {
        let begin = json!({
            "token": "rustAnalyzer/Indexing",
            "value": { "kind": "begin", "title": "Indexing" }
        });
        let report = json!({
            "token": "rustAnalyzer/Indexing",
            "value": { "kind": "report", "message": "12/100" }
        });
        let end = json!({
            "token": "rustAnalyzer/Indexing",
            "value": { "kind": "end" }
        });

        assert_eq!(
            parse_progress_params(&begin),
            Some(("rustAnalyzer/Indexing".to_string(), ProgressUpdate::Begin))
        );
        assert_eq!(
            parse_progress_params(&report),
            Some(("rustAnalyzer/Indexing".to_string(), ProgressUpdate::Report))
        );
        assert_eq!(
            parse_progress_params(&end),
            Some(("rustAnalyzer/Indexing".to_string(), ProgressUpdate::End))
        );
    }

    #[test]
    fn parse_progress_params_handles_numeric_token() {
        // LSP allows progressTokens to be integers too. We stringify
        // them so the same DashMap can hold both kinds.
        let msg = json!({
            "token": 42,
            "value": { "kind": "begin" }
        });
        assert_eq!(
            parse_progress_params(&msg),
            Some(("42".to_string(), ProgressUpdate::Begin))
        );
    }

    #[test]
    fn parse_progress_params_rejects_malformed_messages() {
        // Missing token
        assert_eq!(
            parse_progress_params(&json!({ "value": { "kind": "begin" } })),
            None
        );
        // Missing value
        assert_eq!(parse_progress_params(&json!({ "token": "x" })), None);
        // Unknown kind
        assert_eq!(
            parse_progress_params(&json!({
                "token": "x",
                "value": { "kind": "rumour" }
            })),
            None
        );
        // Token of unsupported type (bool)
        assert_eq!(
            parse_progress_params(&json!({
                "token": true,
                "value": { "kind": "end" }
            })),
            None
        );
    }

    #[test]
    fn apply_progress_to_map_tracks_lifecycle() {
        let map = DashMap::new();

        // Begin → token in flight, not idle.
        apply_progress_to_map(
            &map,
            &json!({
                "token": "rustAnalyzer/Indexing",
                "value": { "kind": "begin" }
            }),
        );
        assert_eq!(map.len(), 1);
        assert_eq!(
            *map.get("rustAnalyzer/Indexing").unwrap().value(),
            ProgressKind::Begin
        );

        // Report → still in flight, state updated.
        apply_progress_to_map(
            &map,
            &json!({
                "token": "rustAnalyzer/Indexing",
                "value": { "kind": "report", "message": "halfway" }
            }),
        );
        assert_eq!(map.len(), 1);
        assert_eq!(
            *map.get("rustAnalyzer/Indexing").unwrap().value(),
            ProgressKind::Report
        );

        // End → token cleared, idle again.
        apply_progress_to_map(
            &map,
            &json!({
                "token": "rustAnalyzer/Indexing",
                "value": { "kind": "end" }
            }),
        );
        assert!(map.is_empty());
    }

    #[test]
    fn apply_progress_to_map_handles_concurrent_tokens() {
        let map = DashMap::new();

        // Two analyses begin in parallel.
        for token in ["rustAnalyzer/Indexing", "rustAnalyzer/Flycheck"] {
            apply_progress_to_map(
                &map,
                &json!({
                    "token": token,
                    "value": { "kind": "begin" }
                }),
            );
        }
        assert_eq!(map.len(), 2, "two distinct tokens should both be tracked");

        // First one ends, server still busy.
        apply_progress_to_map(
            &map,
            &json!({
                "token": "rustAnalyzer/Indexing",
                "value": { "kind": "end" }
            }),
        );
        assert_eq!(map.len(), 1);
        assert!(map.contains_key("rustAnalyzer/Flycheck"));

        // Second one ends, idle.
        apply_progress_to_map(
            &map,
            &json!({
                "token": "rustAnalyzer/Flycheck",
                "value": { "kind": "end" }
            }),
        );
        assert!(map.is_empty());
    }

    #[test]
    fn apply_progress_to_map_silently_ignores_garbage() {
        let map = DashMap::new();
        apply_progress_to_map(&map, &json!({ "garbage": true }));
        apply_progress_to_map(&map, &json!(null));
        apply_progress_to_map(&map, &json!("not even an object"));
        assert!(
            map.is_empty(),
            "garbage messages should not insert anything"
        );
    }

    #[test]
    fn end_for_unknown_token_is_a_noop() {
        let map = DashMap::new();
        // End without prior begin — shouldn't panic, shouldn't insert.
        apply_progress_to_map(
            &map,
            &json!({
                "token": "ghost",
                "value": { "kind": "end" }
            }),
        );
        assert!(map.is_empty());
    }

    #[test]
    fn write_message_errors_instead_of_blocking_when_stdin_not_drained() {
        // A process that never reads its stdin — stand-in for the wedged
        // rust-analyzer that voided two bench runs (08-22, 08-24). With
        // the old direct `write_all`, this test would hang forever once
        // the pipe buffer filled; now the bounded queue must surface a
        // HARD-classified error instead.
        let mut child = std::process::Command::new("sleep")
            .arg("30")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawn sleep");
        let stdin = child.stdin.take().expect("stdin");
        let t = LspTransport::with_write_timeout(stdin, Duration::from_millis(200));

        // Saturate pipe buffer (~64KB) + writer queue (WRITE_QUEUE_CAP
        // slots): 200 x 32KB is far beyond both.
        let blob = "x".repeat(32 * 1024);
        let mut got_err = None;
        for _ in 0..200 {
            if let Err(e) = t.send_notification("test/blob", json!({ "data": &blob })) {
                got_err = Some(format!("{e:#}"));
                break;
            }
        }
        let err = got_err.expect("writes kept succeeding against a non-draining pipe");
        assert!(
            err.contains("failed to write to lsp stdin"),
            "error must carry the HARD infra-failure wording, got: {err}"
        );
        let _ = child.kill();
        let _ = child.wait();
    }
}
