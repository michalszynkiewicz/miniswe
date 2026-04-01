//! JSON-RPC transport over stdio with Content-Length framing.

use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::process::ChildStdin;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;

use dashmap::DashMap;
use serde_json::Value;
use tokio::sync::oneshot;

/// JSON-RPC transport for LSP communication.
pub struct LspTransport {
    writer: std::sync::Mutex<BufWriter<ChildStdin>>,
    pub(crate) pending: DashMap<i64, oneshot::Sender<Value>>,
    pub(crate) diagnostics: DashMap<String, Vec<lsp_types::Diagnostic>>,
    next_id: AtomicI64,
    pub(crate) crashed: AtomicBool,
}

impl LspTransport {
    pub fn new(stdin: ChildStdin) -> Self {
        Self {
            writer: std::sync::Mutex::new(BufWriter::new(stdin)),
            pending: DashMap::new(),
            diagnostics: DashMap::new(),
            next_id: AtomicI64::new(1),
            crashed: AtomicBool::new(false),
        }
    }

    /// Send a JSON-RPC request. Returns a receiver for the response.
    pub fn send_request(&self, method: &str, params: Value) -> anyhow::Result<oneshot::Receiver<Value>> {
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

        let mut writer = self.writer.lock().map_err(|e| anyhow::anyhow!("lock: {e}"))?;
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
                            if let Ok(diag_params) = serde_json::from_value::<lsp_types::PublishDiagnosticsParams>(params.clone()) {
                                let uri = diag_params.uri.as_str().to_string();
                                transport.diagnostics.insert(uri, diag_params.diagnostics);
                            }
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
