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

/// Grace period after spawn during which an empty progress map does NOT
/// count as settled for servers that have never reported progress or
/// server status. Between `initialize` and the first `$/progress begin`
/// the map is empty while the server is actually ramping up its initial
/// index — on a loaded CI runner that window is wide enough for
/// `get_diagnostics` to read a premature empty publish and report
/// "confirmed clean" (the 2026-09-02 `lsp_diagnostics_on_type_error` CI
/// failure). Servers that never emit either signal (e.g.
/// yaml-language-server) pay at most this once, right after spawn.
const SETTLE_WARMUP: Duration = Duration::from_secs(10);

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
    /// Whether at least one well-formed `$/progress` update has arrived.
    /// Until then an empty `progress` map is ambiguous: "nothing to do"
    /// vs "hasn't announced its initial indexing yet".
    saw_progress: AtomicBool,
    /// Latest `quiescent` flag from rust-analyzer's
    /// `experimental/serverStatus` notification (sent only because we
    /// advertise `serverStatusNotification` in the client capabilities).
    /// Meaningless until `saw_server_status` is set.
    quiescent: AtomicBool,
    /// Whether the server has ever sent `experimental/serverStatus`.
    /// When it has, `quiescent` is the authoritative settled signal and
    /// the progress heuristic is demoted to a belt-and-braces AND.
    saw_server_status: AtomicBool,
    /// When this transport was created — anchors [`SETTLE_WARMUP`].
    spawned_at: Instant,
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
            saw_progress: AtomicBool::new(false),
            quiescent: AtomicBool::new(false),
            saw_server_status: AtomicBool::new(false),
            spawned_at: Instant::now(),
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

    /// Returns `true` once the server has genuinely finished its initial
    /// analysis — the signal that makes an *empty* diagnostics publish
    /// trustworthy.
    ///
    /// - If the server sends `experimental/serverStatus` (rust-analyzer
    ///   does, because we advertise `serverStatusNotification`), that
    ///   `quiescent` flag is authoritative — ANDed with `is_idle` so an
    ///   in-flight flycheck run still counts as busy.
    /// - Otherwise fall back to the progress heuristic, hardened against
    ///   the warm-up race: an empty progress map only counts once we've
    ///   seen at least one progress update, or [`SETTLE_WARMUP`] has
    ///   elapsed since spawn (for servers that never report progress).
    pub(crate) fn is_settled(&self) -> bool {
        if self.saw_server_status.load(Ordering::Acquire) {
            return self.quiescent.load(Ordering::Acquire) && self.is_idle();
        }
        self.is_idle()
            && (self.saw_progress.load(Ordering::Relaxed)
                || self.spawned_at.elapsed() >= SETTLE_WARMUP)
    }

    /// Apply a parsed `$/progress` notification to the in-flight token
    /// map. Pulled out of `reader_loop` so it can be unit-tested with
    /// synthetic JSON.
    pub(crate) fn apply_progress(&self, params: &Value) {
        if parse_progress_params(params).is_some() {
            self.saw_progress.store(true, Ordering::Relaxed);
        }
        apply_progress_to_map(&self.progress, params);
    }

    /// Apply an `experimental/serverStatus` notification (rust-analyzer:
    /// `{"health":"ok","quiescent":bool,"message":..}`). Pulled out of
    /// `reader_loop` for unit tests.
    pub(crate) fn apply_server_status(&self, params: &Value) {
        if let Some(quiescent) = params.get("quiescent").and_then(Value::as_bool) {
            self.quiescent.store(quiescent, Ordering::Release);
            self.saw_server_status.store(true, Ordering::Release);
        }
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

            // Dispatch on shape, method first: a message WITH "method" is
            // server-initiated (a request when it also has "id", else a
            // notification); only a message with "id" and no "method" is a
            // response to one of ours. Testing "id" first would misroute
            // server-initiated requests (workspace/configuration,
            // window/workDoneProgress/create, ...) into `pending` — their
            // ids come from the server's own counter and can collide with
            // ours, resolving an unrelated in-flight request with garbage.
            if let Some(method) = msg.get("method").and_then(|v| v.as_str()) {
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
                    "experimental/serverStatus" => {
                        if let Some(params) = msg.get("params") {
                            transport.apply_server_status(params);
                        }
                    }
                    "window/workDoneProgress/create" => {
                        // The server asks to open a progress token (sent
                        // because we advertise `window.workDoneProgress`).
                        // Acknowledge so it doesn't accumulate pending
                        // requests; the updates arrive as `$/progress`.
                        if let Some(id) = msg.get("id") {
                            let _ = transport.write_message(&serde_json::json!({
                                "jsonrpc": "2.0",
                                "id": id,
                                "result": null,
                            }));
                        }
                    }
                    _ => {} // other notifications/requests ignored
                }
            } else if let Some(id) = msg.get("id").and_then(|v| v.as_i64()) {
                // Response — find pending request
                if let Some((_, tx)) = transport.pending.remove(&id) {
                    let _ = tx.send(msg);
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

    /// Spawn a dummy child so a real `LspTransport` can be constructed
    /// for driving `is_settled` with synthetic notifications.
    fn dummy_transport() -> (std::process::Child, LspTransport) {
        let mut child = std::process::Command::new("sleep")
            .arg("30")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawn sleep");
        let stdin = child.stdin.take().expect("stdin");
        let t = LspTransport::with_write_timeout(stdin, Duration::from_millis(200));
        (child, t)
    }

    #[test]
    fn fresh_transport_is_not_settled_despite_empty_progress() {
        // The CI warm-up race: between `initialize` and the server's
        // first `$/progress begin` the progress map is empty, but that
        // must NOT read as "analysis finished".
        let (mut child, t) = dummy_transport();
        assert!(t.is_idle(), "no progress tokens yet");
        assert!(
            !t.is_settled(),
            "empty progress map within the warm-up window must not count as settled"
        );
        let _ = child.kill();
        let _ = child.wait();
    }

    #[test]
    fn progress_end_settles_without_server_status() {
        // Servers without serverStatus (gopls, pyright, …): once we have
        // seen real progress traffic, empty-again means settled.
        let (mut child, t) = dummy_transport();
        t.apply_progress(&json!({
            "token": "indexing",
            "value": { "kind": "begin" }
        }));
        assert!(!t.is_settled(), "in-flight progress token = busy");
        t.apply_progress(&json!({
            "token": "indexing",
            "value": { "kind": "end" }
        }));
        assert!(
            t.is_settled(),
            "progress seen and drained should count as settled"
        );
        let _ = child.kill();
        let _ = child.wait();
    }

    #[test]
    fn server_status_quiescent_is_authoritative() {
        let (mut child, t) = dummy_transport();

        // rust-analyzer's first status: still analyzing.
        t.apply_server_status(&json!({
            "health": "ok", "quiescent": false, "message": null
        }));
        assert!(!t.is_settled(), "quiescent=false must gate settled");

        // Analysis done — settled even though no progress was ever seen
        // and the warm-up window hasn't elapsed.
        t.apply_server_status(&json!({
            "health": "ok", "quiescent": true, "message": null
        }));
        assert!(t.is_settled(), "quiescent=true means settled");

        // Belt and braces: an in-flight progress token (e.g. flycheck)
        // still counts as busy even while quiescent.
        t.apply_progress(&json!({
            "token": "rustAnalyzer/Flycheck",
            "value": { "kind": "begin" }
        }));
        assert!(!t.is_settled(), "quiescent + in-flight progress = busy");
        t.apply_progress(&json!({
            "token": "rustAnalyzer/Flycheck",
            "value": { "kind": "end" }
        }));
        assert!(t.is_settled());

        let _ = child.kill();
        let _ = child.wait();
    }

    #[test]
    fn reader_loop_routes_server_requests_by_method_not_id() {
        // A server→client REQUEST (has BOTH "method" and "id") whose id
        // collides with one of our in-flight client requests. Before the
        // method-first dispatch fix, reader_loop matched on "id" alone and
        // resolved our pending request with the server's request — this
        // pins down that it must be routed as a request instead: the
        // `window/workDoneProgress/create` gets a null-result reply and
        // the pending entry stays untouched.
        let out = tempfile::NamedTempFile::new().expect("tmp file");
        let out_path = out.path().to_str().expect("utf8 path").to_string();
        // Child: emit one create request and one publishDiagnostics
        // notification as LSP frames, then copy everything we send IT
        // into $OUT (so the test can observe our reply) until we exit.
        let script = format!(
            r#"m1='{{"jsonrpc":"2.0","id":1,"method":"window/workDoneProgress/create","params":{{"token":"t1"}}}}'
m2='{{"jsonrpc":"2.0","method":"textDocument/publishDiagnostics","params":{{"uri":"file:///w/src/lib.rs","diagnostics":[]}}}}'
printf 'Content-Length: %s\r\n\r\n%s' "${{#m1}}" "$m1"
printf 'Content-Length: %s\r\n\r\n%s' "${{#m2}}" "$m2"
cat > {out_path}"#
        );
        let mut child = std::process::Command::new("bash")
            .args(["-c", &script])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawn bash");
        let stdin = child.stdin.take().expect("stdin");
        let stdout = child.stdout.take().expect("stdout");

        let t = Arc::new(LspTransport::with_write_timeout(
            stdin,
            Duration::from_millis(500),
        ));
        // Simulate an in-flight client request whose id collides with the
        // server request's id.
        let (tx, mut rx) = oneshot::channel();
        t.pending.insert(1, tx);

        let reader_t = Arc::clone(&t);
        let reader = std::thread::spawn(move || LspTransport::reader_loop(reader_t, stdout));

        // Wait until our reply reached the child (it copies it to $OUT).
        let deadline = Instant::now() + Duration::from_secs(5);
        let reply = loop {
            let content = std::fs::read_to_string(out.path()).unwrap_or_default();
            if content.contains("\"id\":1") {
                break content;
            }
            assert!(
                Instant::now() < deadline,
                "no reply to window/workDoneProgress/create reached the server; got: {content:?}"
            );
            std::thread::sleep(Duration::from_millis(20));
        };
        assert!(
            reply.contains("\"result\":null"),
            "create request must be answered with a null result, got: {reply:?}"
        );
        // The colliding pending request must be untouched: still registered,
        // nothing sent on its channel.
        assert!(
            t.pending.contains_key(&1),
            "server request id must not resolve an unrelated pending client request"
        );
        assert!(
            rx.try_recv().is_err(),
            "pending channel must not have received the server's request"
        );
        // The notification path still works after the dispatch reorder.
        assert!(
            t.diagnostics.contains_key("file:///w/src/lib.rs"),
            "publishDiagnostics must still be cached"
        );

        let _ = child.kill();
        let _ = child.wait();
        let _ = reader.join();
    }

    #[test]
    fn malformed_server_status_is_ignored() {
        let (mut child, t) = dummy_transport();
        t.apply_server_status(&json!({ "health": "ok" }));
        t.apply_server_status(&json!(null));
        t.apply_server_status(&json!({ "quiescent": "yes" }));
        assert!(
            !t.is_settled(),
            "malformed status must not flip the transport into status mode"
        );
        let _ = child.kill();
        let _ = child.wait();
    }
}
