//! LSP client for rust-analyzer integration.
//!
//! Spawns rust-analyzer, manages the LSP lifecycle, and provides
//! diagnostics + navigation queries. Falls back gracefully if
//! rust-analyzer is unavailable or crashes.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::{Child, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, Result};
use lsp_types::*;
use serde_json::Value;
use tokio::task::JoinHandle;

use crate::lsp::transport::LspTransport;

/// LSP client wrapping a rust-analyzer process.
pub struct LspClient {
    transport: Arc<LspTransport>,
    child: std::sync::Mutex<Child>,
    ready: AtomicBool,
    opened_files: std::sync::Mutex<HashSet<String>>,
    project_root: PathBuf,
    _reader_handle: JoinHandle<()>,
}

impl LspClient {
    /// Spawn rust-analyzer and initialize the LSP session.
    /// Returns immediately — initialization happens in the background.
    /// Check `is_ready()` before using query methods.
    pub async fn spawn(project_root: PathBuf) -> Result<Self> {
        use crate::lsp::servers::LspServer;

        let server =
            LspServer::detect(&project_root).context("no supported language detected for LSP")?;

        let binary_path = server
            .ensure_binary()
            .await
            .with_context(|| format!("failed to get {} binary", server.name()))?;

        // Retry up to 3 times — rust-analyzer sometimes crashes on first start
        let max_attempts = 2;
        for attempt in 1..=max_attempts {
            match Self::try_spawn(&server, &binary_path, &project_root).await {
                Ok(client) if client.is_ready() => return Ok(client),
                Ok(client) => {
                    // Spawned but init failed — kill and retry
                    if attempt < max_attempts {
                        eprintln!(
                            "[lsp] attempt {attempt}/{max_attempts} failed, retrying in 2s..."
                        );
                        // Kill the failed process
                        if let Ok(mut child) = client.child.lock() {
                            let _ = child.kill();
                            let _ = child.wait();
                        }
                        tokio::time::sleep(Duration::from_secs(2)).await;
                    } else {
                        // Last attempt — return the non-ready client (fallback to cargo check)
                        return Ok(client);
                    }
                }
                Err(e) => {
                    if attempt < max_attempts {
                        eprintln!(
                            "[lsp] attempt {attempt}/{max_attempts} spawn failed: {e}, retrying..."
                        );
                        tokio::time::sleep(Duration::from_secs(2)).await;
                    } else {
                        return Err(e);
                    }
                }
            }
        }
        unreachable!()
    }

    async fn try_spawn(
        server: &crate::lsp::servers::LspServer,
        binary_path: &Path,
        project_root: &Path,
    ) -> Result<Self> {
        let mut cmd = server.build_command(binary_path, project_root);
        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("Failed to spawn {}", binary_path.display()))?;

        let stdin = child.stdin.take().context("No stdin")?;
        let stdout = child.stdout.take().context("No stdout")?;
        let stderr = child.stderr.take();

        // Log stderr in background for debugging
        if let Some(stderr) = stderr {
            std::thread::spawn(move || {
                use std::io::BufRead;
                let reader = std::io::BufReader::new(stderr);
                for line in reader.lines().take(20) {
                    if let Ok(line) = line
                        && !line.trim().is_empty()
                    {
                        eprintln!("[lsp:stderr] {}", crate::truncate_chars(&line, 200));
                    }
                }
            });
        }

        let transport = Arc::new(LspTransport::new(stdin));

        let transport_clone = Arc::clone(&transport);
        let reader_handle = tokio::task::spawn_blocking(move || {
            LspTransport::reader_loop(transport_clone, stdout);
        });

        let client = Self {
            transport: Arc::clone(&transport),
            child: std::sync::Mutex::new(child),
            ready: AtomicBool::new(false),
            opened_files: std::sync::Mutex::new(HashSet::new()),
            project_root: project_root.to_path_buf(),
            _reader_handle: reader_handle,
        };

        match initialize(&transport, project_root).await {
            Ok(()) => {
                client.ready.store(true, Ordering::Release);
            }
            Err(e) => {
                eprintln!("[lsp] initialization failed: {e}");
            }
        }

        Ok(client)
    }

    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }

    pub fn has_crashed(&self) -> bool {
        self.transport.crashed.load(Ordering::Relaxed)
    }

    /// Wait until the LSP server has no in-flight `$/progress` work
    /// (indexing, flycheck, cargo metadata, etc.), or until `timeout`
    /// elapses. Returns `true` if the server reported idle in time,
    /// `false` if we timed out with work still in flight.
    ///
    /// Servers that don't emit progress at all (or that finished before
    /// we asked) trip this immediately. The point is to remove the race
    /// where `get_diagnostics` reads stale state because rust-analyzer
    /// hadn't finished re-analyzing yet.
    pub async fn wait_for_idle(&self, timeout: Duration) -> bool {
        let start = std::time::Instant::now();
        loop {
            if self.transport.is_idle() {
                return true;
            }
            if start.elapsed() >= timeout {
                return false;
            }
            // Poll cheaply — progress updates are rare events relative
            // to a 10–50ms tick.
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }

    /// Notify the server about a file change. Sends didOpen on first
    /// encounter, didChange on subsequent calls.
    pub fn notify_file_changed(&self, path: &Path) -> Result<()> {
        let uri = path_to_uri(path);
        let content =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;

        let mut opened = self
            .opened_files
            .lock()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        if opened.contains(&uri) {
            // didChange — full sync
            self.transport.send_notification(
                "textDocument/didChange",
                serde_json::json!({
                    "textDocument": { "uri": uri, "version": 1 },
                    "contentChanges": [{ "text": content }]
                }),
            )?;
        } else {
            // didOpen
            let lang_id = language_id(path);
            self.transport.send_notification(
                "textDocument/didOpen",
                serde_json::json!({
                    "textDocument": {
                        "uri": uri,
                        "languageId": lang_id,
                        "version": 1,
                        "text": content
                    }
                }),
            )?;
            opened.insert(uri.clone());
        }

        // Also send didSave to trigger full analysis
        self.transport.send_notification(
            "textDocument/didSave",
            serde_json::json!({
                "textDocument": { "uri": uri }
            }),
        )?;

        Ok(())
    }

    /// Get diagnostics for a file, waiting up to `timeout` for results.
    /// Returns diagnostics from the most recent publishDiagnostics notification.
    ///
    /// First clears any cached diagnostics for the file, then waits for the
    /// server to report idle via `$/progress` (so we don't read while
    /// indexing/flycheck is still running), and finally reads whatever
    /// `publishDiagnostics` left in the cache. Falls back to the older
    /// "wait until a non-empty diagnostic arrives" heuristic for servers
    /// that don't emit progress at all.
    pub async fn get_diagnostics(&self, path: &Path, timeout: Duration) -> Vec<Diagnostic> {
        let uri = path_to_uri(path);

        // Mark that we're waiting for fresh diagnostics
        self.transport.diagnostics.remove(&uri);
        // Also remove by any URI that ends with our path
        let path_str = path.to_string_lossy().to_string();
        self.transport
            .diagnostics
            .retain(|k, _| !k.ends_with(&path_str));

        let overall_start = std::time::Instant::now();

        // 1) Wait for the server to report idle. If the server emits
        //    `$/progress`, this gives us a deterministic point-in-time
        //    "the analysis pipeline is done". If the server never reports
        //    progress (e.g. typescript-language-server), `wait_for_idle`
        //    returns `true` immediately because the map is empty.
        //
        //    We give idle a generous slice (up to half the timeout) so
        //    the legacy "wait for diagnostics" loop below can still soak
        //    up late-arriving messages on servers that don't use progress.
        let idle_budget = timeout / 2;
        let _ = self.wait_for_idle(idle_budget).await;

        // 2) Read the cache. If we got something, return it.
        for entry in self.transport.diagnostics.iter() {
            let key = entry.key();
            if key == &uri || key.ends_with(&path_str) {
                return entry.value().clone();
            }
        }

        // 3) Fallback: legacy wait loop for servers that haven't emitted
        //    a `publishDiagnostics` for this file *yet*, even though
        //    they're idle. We poll for a brief window — most files
        //    return immediately on the first iteration.
        let mut saw_empty = false;
        while overall_start.elapsed() < timeout {
            for entry in self.transport.diagnostics.iter() {
                let key = entry.key();
                if key == &uri || key.ends_with(&path_str) {
                    let diags = entry.value().clone();
                    if !diags.is_empty() {
                        return diags;
                    }
                    saw_empty = true;
                }
            }

            if saw_empty && overall_start.elapsed() > Duration::from_secs(3) {
                return Vec::new();
            }

            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        Vec::new()
    }

    /// Go to definition of symbol at position.
    pub async fn goto_definition(
        &self,
        path: &Path,
        line: u32,
        character: u32,
    ) -> Result<Vec<Location>> {
        let uri = path_to_uri(path);
        let rx = self.transport.send_request(
            "textDocument/definition",
            serde_json::json!({
                "textDocument": { "uri": uri },
                "position": { "line": line, "character": character }
            }),
        )?;

        let response = tokio::time::timeout(Duration::from_secs(10), rx)
            .await
            .context("definition request timed out")?
            .context("channel closed")?;

        parse_locations(&response)
    }

    /// Find all references to symbol at position.
    pub async fn find_references(
        &self,
        path: &Path,
        line: u32,
        character: u32,
    ) -> Result<Vec<Location>> {
        let uri = path_to_uri(path);
        let rx = self.transport.send_request(
            "textDocument/references",
            serde_json::json!({
                "textDocument": { "uri": uri },
                "position": { "line": line, "character": character },
                "context": { "includeDeclaration": true }
            }),
        )?;

        let response = tokio::time::timeout(Duration::from_secs(10), rx)
            .await
            .context("references request timed out")?
            .context("channel closed")?;

        parse_locations(&response)
    }

    /// Get a snapshot of all current diagnostics across all files.
    pub fn diagnostics_snapshot(&self) -> Vec<(String, Vec<Diagnostic>)> {
        self.transport
            .diagnostics
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }

    /// Shut down the LSP server gracefully.
    pub async fn shutdown(self) {
        // Send shutdown request
        if let Ok(rx) = self.transport.send_request("shutdown", Value::Null) {
            let _ = tokio::time::timeout(Duration::from_secs(5), rx).await;
        }

        // Send exit notification
        let _ = self.transport.send_notification("exit", Value::Null);

        // Wait briefly for process to exit
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Force kill if still running
        if let Ok(mut child) = self.child.lock() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// Send initialize request and wait for response.
async fn initialize(transport: &LspTransport, project_root: &Path) -> Result<()> {
    let root_uri = path_to_uri(project_root);

    let rx = transport.send_request(
        "initialize",
        serde_json::json!({
            "processId": std::process::id(),
            "clientInfo": { "name": "miniswe", "version": "0.1.0" },
            "rootUri": root_uri,
            "capabilities": {
                "textDocument": {
                    "publishDiagnostics": {
                        "relatedInformation": false
                    },
                    "definition": { "dynamicRegistration": false },
                    "references": { "dynamicRegistration": false },
                    "synchronization": {
                        "didSave": true,
                        "willSave": false,
                        "willSaveWaitUntil": false
                    }
                }
            },
            "workspaceFolders": [{
                "uri": root_uri,
                "name": project_root.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("project")
            }]
        }),
    )?;

    // Wait for initialize response (up to 30s)
    let _response = tokio::time::timeout(Duration::from_secs(30), rx)
        .await
        .context("initialize timed out")?
        .context("channel closed")?;

    // Send initialized notification
    transport.send_notification("initialized", serde_json::json!({}))?;

    Ok(())
}

/// Convert a file path to a file:// URI.
fn path_to_uri(path: &Path) -> String {
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_default().join(path)
    };
    format!("file://{}", abs.display())
}

/// Convert a file:// URI back to a path.
pub fn uri_to_path(uri: &lsp_types::Uri) -> Option<PathBuf> {
    let s = uri.as_str();
    s.strip_prefix("file://").map(PathBuf::from)
}

/// Detect language ID from file extension.
fn language_id(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("rs") => "rust",
        Some("py") => "python",
        Some("ts") => "typescript",
        Some("tsx") => "typescriptreact",
        Some("js") => "javascript",
        Some("go") => "go",
        Some("java") => "java",
        Some("kt") | Some("kts") => "kotlin",
        _ => "plaintext",
    }
}

/// Parse Location or Location[] from a textDocument/definition or references response.
fn parse_locations(response: &Value) -> Result<Vec<Location>> {
    let result = response.get("result").unwrap_or(&Value::Null);

    if result.is_null() {
        return Ok(Vec::new());
    }

    // Can be a single Location, an array of Location, or an array of LocationLink
    if result.is_array() {
        let arr = result.as_array().unwrap();
        let mut locations = Vec::new();
        for item in arr {
            if let Ok(loc) = serde_json::from_value::<Location>(item.clone()) {
                locations.push(loc);
            } else if let Ok(link) = serde_json::from_value::<LocationLink>(item.clone()) {
                locations.push(Location {
                    uri: link.target_uri,
                    range: link.target_selection_range,
                });
            }
        }
        Ok(locations)
    } else if let Ok(loc) = serde_json::from_value::<Location>(result.clone()) {
        Ok(vec![loc])
    } else {
        Ok(Vec::new())
    }
}
