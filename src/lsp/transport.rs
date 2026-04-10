//! JSON-RPC transport over stdio with Content-Length framing.

use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::process::ChildStdin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};

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

/// JSON-RPC transport for LSP communication.
pub struct LspTransport {
    writer: std::sync::Mutex<BufWriter<ChildStdin>>,
    pub(crate) pending: DashMap<i64, oneshot::Sender<Value>>,
    pub(crate) diagnostics: DashMap<String, Vec<lsp_types::Diagnostic>>,
    /// In-flight `$/progress` tokens. Inserted on `begin`, refreshed on
    /// `report`, removed on `end`. Emptiness == server is idle.
    pub(crate) progress: DashMap<String, ProgressKind>,
    next_id: AtomicI64,
    pub(crate) crashed: AtomicBool,
}

impl LspTransport {
    pub fn new(stdin: ChildStdin) -> Self {
        Self {
            writer: std::sync::Mutex::new(BufWriter::new(stdin)),
            pending: DashMap::new(),
            diagnostics: DashMap::new(),
            progress: DashMap::new(),
            next_id: AtomicI64::new(1),
            crashed: AtomicBool::new(false),
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

    /// Write a Content-Length framed message to stdin.
    fn write_message(&self, msg: &Value) -> anyhow::Result<()> {
        let body = serde_json::to_string(msg)?;
        let header = format!("Content-Length: {}\r\n\r\n", body.len());

        let mut writer = self
            .writer
            .lock()
            .map_err(|e| anyhow::anyhow!("lock: {e}"))?;
        writer.write_all(header.as_bytes())?;
        writer.write_all(body.as_bytes())?;
        writer.flush()?;
        Ok(())
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
                        if let Some(params) = msg.get("params") {
                            if let Ok(diag_params) = serde_json::from_value::<
                                lsp_types::PublishDiagnosticsParams,
                            >(params.clone())
                            {
                                let uri = diag_params.uri.as_str().to_string();
                                transport.diagnostics.insert(uri, diag_params.diagnostics);
                            }
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
pub(crate) fn apply_progress_to_map(
    map: &DashMap<String, ProgressKind>,
    params: &Value,
) {
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
        assert!(map.is_empty(), "garbage messages should not insert anything");
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
}
