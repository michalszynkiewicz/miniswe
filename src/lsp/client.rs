//! LSP client for rust-analyzer integration.
//!
//! Spawns rust-analyzer, manages the LSP lifecycle, and provides
//! diagnostics + navigation queries. Falls back gracefully if
//! rust-analyzer is unavailable or crashes.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::{Child, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::Duration;

use anyhow::{Context, Result};
use lsp_types::*;
use serde_json::Value;
use std::thread::JoinHandle;

use crate::lsp::transport::LspTransport;

/// How a request failure implicates the server itself.
#[derive(Clone, Copy)]
enum InfraFailure {
    /// Transport/process provably dead (channel closed, broken pipe).
    Hard,
    /// Request deadline elapsed — could be a wedge OR a busy indexer.
    Soft,
}

/// LSP client wrapping a rust-analyzer process.
pub struct LspClient {
    transport: parking_lot::RwLock<Arc<LspTransport>>,
    child: parking_lot::Mutex<Child>,
    ready: AtomicBool,
    opened_files: parking_lot::Mutex<HashSet<String>>,
    project_root: PathBuf,
    server: crate::lsp::servers::LspServer,
    binary_path: PathBuf,
    /// Wedged-server restarts performed this session (capped — each costs
    /// a full re-index).
    restarts: AtomicU32,
    /// Single-flight guard: concurrent failing requests must not each kill
    /// and respawn the server (the second kill would murder the first's
    /// fresh spawn and burn the whole restart budget on one incident).
    restart_lock: tokio::sync::Mutex<()>,
    /// Bumped on every successful restart; waiters that queued behind a
    /// restart use it to detect the work is already done.
    generation: AtomicU32,
    reader_handle: parking_lot::Mutex<JoinHandle<()>>,
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
                        {
                            let mut child = client.child.lock();
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

    /// Spawn the server process and wire up transport + reader. Shared by
    /// initial spawn and wedged-server restart.
    fn spawn_process(
        server: &crate::lsp::servers::LspServer,
        binary_path: &Path,
        project_root: &Path,
    ) -> Result<(Child, Arc<LspTransport>, JoinHandle<()>)> {
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

        // Log stderr in background for debugging. The drainer keeps
        // running until rust-analyzer exits and closes the pipe; on
        // Drop we kill the child, which ends the loop. (Earlier this
        // capped at 20 lines, which made it impossible to see what
        // rust-analyzer was doing past the initial config dump —
        // exactly the data we needed to diagnose the CI failures of
        // `lsp_auto_check_integration`.)
        if let Some(stderr) = stderr {
            std::thread::spawn(move || {
                use std::io::BufRead;
                let reader = std::io::BufReader::new(stderr);
                for line in reader.lines() {
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
        // Plain OS thread, NOT tokio spawn_blocking: a wedged-server
        // restart runs inside the per-tool-call current_thread runtime,
        // and dropping such a runtime WAITS for its in-flight blocking
        // tasks. A reader spawned there would pin the tool worker thread
        // forever once the tool returned (the reader only exits when the
        // server dies) — the silent half of the 08-22/08-24 harness
        // hang. An OS thread outlives any runtime and exits on stdout
        // EOF when the server is killed.
        let reader_handle = std::thread::spawn(move || {
            LspTransport::reader_loop(transport_clone, stdout);
        });

        Ok((child, transport, reader_handle))
    }

    async fn try_spawn(
        server: &crate::lsp::servers::LspServer,
        binary_path: &Path,
        project_root: &Path,
    ) -> Result<Self> {
        let (child, transport, reader_handle) =
            Self::spawn_process(server, binary_path, project_root)?;

        let client = Self {
            transport: parking_lot::RwLock::new(Arc::clone(&transport)),
            child: parking_lot::Mutex::new(child),
            ready: AtomicBool::new(false),
            opened_files: parking_lot::Mutex::new(HashSet::new()),
            project_root: project_root.to_path_buf(),
            server: *server,
            binary_path: binary_path.to_path_buf(),
            restarts: AtomicU32::new(0),
            restart_lock: tokio::sync::Mutex::new(()),
            generation: AtomicU32::new(0),
            reader_handle: parking_lot::Mutex::new(reader_handle),
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

    fn tx(&self) -> Arc<LspTransport> {
        self.transport.read().clone()
    }

    /// Wedged-server restarts performed so far this session.
    pub fn restart_count(&self) -> u32 {
        self.restarts.load(Ordering::Relaxed)
    }

    /// Kill the underlying server process without the client's knowledge —
    /// test/diagnostic hook simulating a wedged or crashed server.
    pub fn kill_server_for_test(&self) {
        let mut child = self.child.lock();
        let _ = child.kill();
        let _ = child.wait();
    }

    /// Restart a wedged server: kill + respawn + re-initialize + replay the
    /// opened files, so the caller can re-issue the failed request against
    /// an indexed workspace. Capped per session (each restart costs a full
    /// re-index). Returns true when the new server is ready.
    async fn try_restart_once(&self, failure: InfraFailure) -> bool {
        const MAX_RESTARTS: u32 = 2;
        // Single-flight: whoever loses the race waits, then discovers the
        // restart already happened (generation bump) and just retries.
        let seen_gen = self.generation.load(Ordering::Acquire);
        let _guard = self.restart_lock.lock().await;
        if self.generation.load(Ordering::Acquire) != seen_gen {
            return self.is_ready();
        }
        // Busy vs wedged: a SOFT failure (request deadline) on a live,
        // uncrashed server that is actively reporting progress means it's
        // indexing a big workspace, not dead — restarting would trade a
        // nearly-warm index for a cold start and likely time out again.
        // Only hard evidence (process exit, transport crash, dead channel)
        // or timeout-while-idle (a contradiction: nothing in flight yet
        // unresponsive = dead worker) justifies the restart.
        let process_exited = {
            let mut child = self.child.lock();
            matches!(child.try_wait(), Ok(Some(_)))
        };
        if matches!(failure, InfraFailure::Soft)
            && !process_exited
            && !self.tx().crashed.load(Ordering::Relaxed)
            && !self.wait_for_idle(Duration::from_secs(20)).await
        {
            eprintln!(
                "[lsp] request timed out but {} is busy (indexing?) — not restarting",
                self.server.name()
            );
            return false;
        }
        let n = self.restarts.fetch_add(1, Ordering::SeqCst);
        if n >= MAX_RESTARTS {
            return false;
        }
        eprintln!(
            "[lsp] request failed on a wedged {} — restarting it ({}/{MAX_RESTARTS})",
            self.server.name(),
            n + 1
        );
        {
            let mut child = self.child.lock();
            let _ = child.kill();
            let _ = child.wait();
        }
        let (child, transport, reader_handle) =
            match Self::spawn_process(&self.server, &self.binary_path, &self.project_root) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("[lsp] restart spawn failed: {e:#}");
                    self.ready.store(false, Ordering::Release);
                    return false;
                }
            };
        if let Err(e) = initialize(&transport, &self.project_root).await {
            eprintln!("[lsp] restart initialization failed: {e:#}");
            self.ready.store(false, Ordering::Release);
            return false;
        }
        *self.child.lock() = child;
        *self.transport.write() = transport;
        {
            let mut handle = self.reader_handle.lock();
            // The old reader thread exits on its own: killing the old
            // child closed its stdout. Dropping the handle detaches it.
            drop(std::mem::replace(&mut *handle, reader_handle));
        }
        self.ready.store(true, Ordering::Release);
        // Replay opened files so queries hit an indexed workspace again.
        let uris: Vec<String> = {
            let mut opened = self.opened_files.lock();
            opened.drain().collect()
        };
        for uri in uris {
            if let Ok(parsed) = uri.parse::<lsp_types::Uri>()
                && let Some(path) = uri_to_path(&parsed)
            {
                // Files deleted since they were opened (bench reverts do
                // this) simply drop out of the replay.
                if !path.exists() {
                    continue;
                }
                if self.notify_file_changed(&path).is_err() {
                    // The NEW server is already refusing writes — each
                    // further attempt costs up to the stdin-write
                    // deadline, so don't grind through the rest of the
                    // list. The retried request surfaces the failure.
                    break;
                }
            }
        }
        let _ = self.wait_for_idle(Duration::from_secs(60)).await;
        self.generation.fetch_add(1, Ordering::Release);
        true
    }

    pub fn has_crashed(&self) -> bool {
        self.tx().crashed.load(Ordering::Relaxed)
    }

    /// Wait until the LSP server has *settled* — finished its in-flight
    /// analysis (indexing, flycheck, cargo metadata, etc.) — or until
    /// `timeout` elapses. Returns `true` if the server settled in time,
    /// `false` if we timed out with work still in flight.
    ///
    /// "Settled" is stronger than progress-map-empty: for rust-analyzer
    /// it means `experimental/serverStatus` reported `quiescent`, and for
    /// progress-only servers an empty map counts only after real progress
    /// traffic (or a short post-spawn warm-up for servers that emit
    /// neither). This closes the race where `get_diagnostics` read the
    /// gap between `initialize` and the first `$/progress begin` as
    /// "analysis finished".
    pub async fn wait_for_idle(&self, timeout: Duration) -> bool {
        let start = std::time::Instant::now();
        loop {
            if self.tx().is_settled() {
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

        let mut opened = self.opened_files.lock();
        if opened.contains(&uri) {
            // didChange — full sync
            self.tx().send_notification(
                "textDocument/didChange",
                serde_json::json!({
                    "textDocument": { "uri": uri, "version": 1 },
                    "contentChanges": [{ "text": content }]
                }),
            )?;
        } else {
            // didOpen
            let lang_id = language_id(path);
            self.tx().send_notification(
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
        self.tx().send_notification(
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
    /// See [`Self::get_diagnostics_with_status`] for the wait model.
    pub async fn get_diagnostics(&self, path: &Path, timeout: Duration) -> Vec<Diagnostic> {
        self.get_diagnostics_with_status(path, timeout).await.0
    }

    /// Like [`Self::get_diagnostics`], but also reports whether the result is
    /// *confirmed* — the server actually settled — versus *unconfirmed* (we
    /// timed out before analysis finished). An empty `Vec` with
    /// `confirmed == false` means "unknown / still pending", NOT "clean":
    /// callers must not claim the file is OK on an unconfirmed-empty result.
    ///
    /// The wait model rests on two measured facts about rust-analyzer:
    ///
    /// - Servers do NOT re-publish unchanged diagnostics. rust-analyzer
    ///   publishes natively once per didOpen, and after that only when a
    ///   didSave-triggered flycheck (`cargo check`) produces or clears
    ///   something. Waiting for a publish that will never come is what
    ///   used to eat the entire timeout on every clean file.
    /// - So the clean signal is *absence of publishes while settled*: once
    ///   the server has been continuously settled (quiescent AND no
    ///   in-flight `$/progress` token) for a short grace, any analysis our
    ///   didSave triggered has started, run, and ended without publishing
    ///   for this file — empty means clean. The grace only has to cover
    ///   the didSave → flycheck-token-open dispatch gap (73ms measured
    ///   locally; the token is visible because we advertise
    ///   `window.workDoneProgress`), not the analysis itself.
    ///
    /// A *non-empty* publish short-circuits immediately: the cache is
    /// cleared for this file on entry, so any errors read back arrived
    /// after this call began and reflect the file's current contents.
    /// Fresh *empty* publishes get no shortcut — on a loaded runner the
    /// didOpen-time publish can be a pre-inference empty (the 2026-09-02
    /// CI failure), so empties are only ever confirmed by the grace.
    pub async fn get_diagnostics_with_status(
        &self,
        path: &Path,
        timeout: Duration,
    ) -> (Vec<Diagnostic>, bool) {
        let uri = path_to_uri(path);

        // Clear cached entries for this file (by exact URI and by path
        // suffix) so anything read back below post-dates this call — a
        // publish from before the caller's edit can neither acquit nor
        // indict the current contents.
        self.tx().diagnostics.remove(&uri);
        let path_str = path.to_string_lossy().to_string();
        self.tx().diagnostics.retain(|k, _| !k.ends_with(&path_str));

        // How long the server must hold "settled" before empty counts as
        // clean. Must exceed the didSave → flycheck-begin dispatch gap
        // (73ms measured on a fast machine) with plenty of slack for
        // loaded CI runners; the expensive parts (indexing, cargo check)
        // hold progress tokens and so extend the wait on their own.
        const SETTLED_GRACE: Duration = Duration::from_secs(2);

        let overall_start = std::time::Instant::now();
        let mut settled_since: Option<std::time::Instant> = None;
        loop {
            for entry in self.tx().diagnostics.iter() {
                let key = entry.key();
                if (key == &uri || key.ends_with(&path_str)) && !entry.value().is_empty() {
                    return (entry.value().clone(), true);
                }
            }

            if self.tx().is_settled() {
                let since = *settled_since.get_or_insert_with(std::time::Instant::now);
                if since.elapsed() >= SETTLED_GRACE {
                    return (Vec::new(), true);
                }
            } else {
                settled_since = None;
            }

            if overall_start.elapsed() >= timeout {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        // Timed out without holding settled for the grace. Trust the
        // instantaneous settled signal for the confirmed flag: settled
        // right now means empty is still the best answer; otherwise it's
        // unknown/pending.
        (Vec::new(), self.tx().is_settled())
    }

    /// Go to definition of symbol at position.
    async fn goto_definition_raw(
        &self,
        path: &Path,
        line: u32,
        character: u32,
    ) -> Result<Vec<Location>> {
        let uri = path_to_uri(path);
        let rx = self.tx().send_request(
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
    async fn find_references_raw(
        &self,
        path: &Path,
        line: u32,
        character: u32,
    ) -> Result<Vec<Location>> {
        let uri = path_to_uri(path);
        let rx = self.tx().send_request(
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

    /// List all symbols defined in `path`.
    ///
    /// Returns a flat normalised list of `(name, kind, name_range, full_range)`
    /// tuples regardless of which response shape the server emits — newer
    /// servers return `DocumentSymbol[]` (hierarchical) and older ones return
    /// `SymbolInformation[]` (flat). For nested symbols (methods inside impls)
    /// the children are flattened in too.
    async fn document_symbol_raw(&self, path: &Path) -> Result<Vec<DocumentSymbolEntry>> {
        let uri = path_to_uri(path);
        let rx = self.tx().send_request(
            "textDocument/documentSymbol",
            serde_json::json!({
                "textDocument": { "uri": uri }
            }),
        )?;

        let response = tokio::time::timeout(Duration::from_secs(15), rx)
            .await
            .context("documentSymbol request timed out")?
            .context("channel closed")?;

        let result = response.get("result").cloned().unwrap_or(Value::Null);
        if result.is_null() {
            return Ok(Vec::new());
        }

        // Try hierarchical first, fall back to flat.
        if let Ok(hier) = serde_json::from_value::<Vec<DocumentSymbol>>(result.clone()) {
            let mut out = Vec::new();
            for sym in hier {
                flatten_document_symbol(sym, &mut out);
            }
            return Ok(out);
        }
        if let Ok(flat) = serde_json::from_value::<Vec<SymbolInformation>>(result) {
            return Ok(flat
                .into_iter()
                .map(|s| DocumentSymbolEntry {
                    name: s.name,
                    kind: s.kind,
                    name_range: s.location.range,
                    full_range: s.location.range,
                })
                .collect());
        }
        Ok(Vec::new())
    }

    /// Search the entire workspace for symbols matching `query`.
    /// Used as a "did you mean" fallback when a per-file lookup misses.
    async fn workspace_symbol_raw(&self, query: &str) -> Result<Vec<WorkspaceSymbolEntry>> {
        let rx = self
            .tx()
            .send_request("workspace/symbol", serde_json::json!({ "query": query }))?;
        let response = tokio::time::timeout(Duration::from_secs(15), rx)
            .await
            .context("workspaceSymbol request timed out")?
            .context("channel closed")?;
        let result = response.get("result").cloned().unwrap_or(Value::Null);
        if result.is_null() {
            return Ok(Vec::new());
        }
        // Newer servers may return `WorkspaceSymbol[]` with a `location` that's
        // a `OneOf<Location, {uri}>`; older return `SymbolInformation[]`.
        if let Ok(flat) = serde_json::from_value::<Vec<SymbolInformation>>(result) {
            return Ok(flat
                .into_iter()
                .filter_map(|s| {
                    Some(WorkspaceSymbolEntry {
                        name: s.name,
                        kind: s.kind,
                        path: uri_to_path(&s.location.uri)?,
                        line: s.location.range.start.line,
                    })
                })
                .collect());
        }
        Ok(Vec::new())
    }

    /// Request a workspace-wide rename of the symbol at `(line, character)`
    /// in `path` to `new_name`. Returns the `WorkspaceEdit` the server
    /// produced; the caller is responsible for applying it.
    ///
    /// `textDocument/rename` is part of the standard LSP and is supported
    /// by every server miniswe ships against (rust-analyzer, gopls,
    /// ts-language-server, pyright, clangd, jdtls). When it isn't supported
    /// the server returns null, which surfaces here as `Ok(None)`.
    async fn rename_raw(
        &self,
        path: &Path,
        line: u32,
        character: u32,
        new_name: &str,
    ) -> Result<Option<WorkspaceEdit>> {
        let uri = path_to_uri(path);
        let rx = self.tx().send_request(
            "textDocument/rename",
            serde_json::json!({
                "textDocument": { "uri": uri },
                "position": { "line": line, "character": character },
                "newName": new_name,
            }),
        )?;

        let response = tokio::time::timeout(Duration::from_secs(30), rx)
            .await
            .context("rename request timed out")?
            .context("channel closed")?;

        let result = response.get("result").cloned().unwrap_or(Value::Null);
        if result.is_null() {
            return Ok(None);
        }
        let edit: WorkspaceEdit =
            serde_json::from_value(result).context("parse rename WorkspaceEdit response")?;
        Ok(Some(edit))
    }

    /// Classify failures caused by a dead/wedged server (vs. bad requests).
    /// Hard = the transport/process is provably gone; Soft = a request
    /// deadline elapsed, which a busy-but-healthy server can also cause.
    fn infra_class(e: &anyhow::Error) -> Option<InfraFailure> {
        let s = format!("{e:#}").to_lowercase();
        // "failed to write" also catches the transport's stdin-write
        // deadline ("failed to write to lsp stdin: ...") — a server that
        // stops draining its pipe is wedged, not busy.
        if s.contains("channel closed")
            || s.contains("broken pipe")
            || s.contains("failed to write")
        {
            return Some(InfraFailure::Hard);
        }
        if s.contains("timed out") {
            return Some(InfraFailure::Soft);
        }
        None
    }

    /// Restart gate used by the request wrappers.
    async fn restart_for(&self, e: &anyhow::Error) -> bool {
        match Self::infra_class(e) {
            Some(class) => self.try_restart_once(class).await,
            None => false,
        }
    }

    pub async fn goto_definition(
        &self,
        path: &Path,
        line: u32,
        character: u32,
    ) -> Result<Vec<Location>> {
        match self.goto_definition_raw(path, line, character).await {
            Err(e) if self.restart_for(&e).await => {
                self.goto_definition_raw(path, line, character).await
            }
            r => r,
        }
    }

    pub async fn find_references(
        &self,
        path: &Path,
        line: u32,
        character: u32,
    ) -> Result<Vec<Location>> {
        match self.find_references_raw(path, line, character).await {
            Err(e) if self.restart_for(&e).await => {
                self.find_references_raw(path, line, character).await
            }
            r => r,
        }
    }

    pub async fn document_symbol(&self, path: &Path) -> Result<Vec<DocumentSymbolEntry>> {
        match self.document_symbol_raw(path).await {
            Err(e) if self.restart_for(&e).await => self.document_symbol_raw(path).await,
            r => r,
        }
    }

    pub async fn workspace_symbol(&self, query: &str) -> Result<Vec<WorkspaceSymbolEntry>> {
        match self.workspace_symbol_raw(query).await {
            Err(e) if self.restart_for(&e).await => self.workspace_symbol_raw(query).await,
            r => r,
        }
    }

    pub async fn rename(
        &self,
        path: &Path,
        line: u32,
        character: u32,
        new_name: &str,
    ) -> Result<Option<WorkspaceEdit>> {
        match self.rename_raw(path, line, character, new_name).await {
            Err(e) if self.restart_for(&e).await => {
                self.rename_raw(path, line, character, new_name).await
            }
            r => r,
        }
    }

    /// Get a snapshot of all current diagnostics across all files.
    pub fn diagnostics_snapshot(&self) -> Vec<(String, Vec<Diagnostic>)> {
        self.tx()
            .diagnostics
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }

    /// Shut down the LSP server gracefully.
    pub async fn shutdown(self) {
        // Send shutdown request
        if let Ok(rx) = self.tx().send_request("shutdown", Value::Null) {
            let _ = tokio::time::timeout(Duration::from_secs(5), rx).await;
        }

        // Send exit notification
        let _ = self.tx().send_notification("exit", Value::Null);

        // Wait briefly for process to exit
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Force kill if still running
        let mut child = self.child.lock();
        let _ = child.kill();
        let _ = child.wait();
    }
}

impl Drop for LspClient {
    /// Last-resort cleanup: if `shutdown()` wasn't awaited (typically
    /// because a caller panicked or `?`-returned), kill the rust-analyzer
    /// child so it doesn't outlive us.
    ///
    /// Without this, the reader task spawned in `try_spawn` stays blocked
    /// on the server's stdout pipe forever — which keeps the tokio runtime
    /// alive and hangs the whole process on exit. On CI this manifested
    /// as a test-binary panic followed by 28 minutes of silence until the
    /// workflow timeout.
    ///
    /// Idempotent: calling `kill` on an already-reaped child is a no-op
    /// we swallow, so the graceful `shutdown()` path is unaffected.
    fn drop(&mut self) {
        let mut child = self.child.lock();
        let _ = child.kill();
        let _ = child.wait();
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
                "workspace": {
                    "workspaceEdit": {
                        "documentChanges": true,
                        "resourceOperations": ["create", "rename", "delete"],
                        "failureHandling": "abort"
                    }
                },
                "textDocument": {
                    "publishDiagnostics": {
                        "relatedInformation": false
                    },
                    "definition": { "dynamicRegistration": false },
                    "references": { "dynamicRegistration": false },
                    "rename": {
                        "dynamicRegistration": false,
                        "prepareSupport": false
                    },
                    "documentSymbol": {
                        "dynamicRegistration": false,
                        "hierarchicalDocumentSymbolSupport": true
                    },
                    "synchronization": {
                        "didSave": true,
                        "willSave": false,
                        "willSaveWaitUntil": false
                    }
                },
                // Without this, spec-compliant servers send NO `$/progress`
                // at all — the transport's progress-token map stays empty
                // and `is_settled()` degrades to serverStatus-only, which
                // is blind to flycheck: rust-analyzer stays `quiescent`
                // while `cargo check` runs.
                "window": {
                    "workDoneProgress": true
                },
                // Opt into rust-analyzer's `experimental/serverStatus`
                // notifications — its `quiescent` flag is the only
                // reliable "initial analysis actually finished" signal
                // (progress-map emptiness is racy during warm-up).
                "experimental": {
                    "serverStatusNotification": true
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

/// One symbol from `textDocument/documentSymbol`, normalised across the
/// hierarchical and flat response shapes.
#[derive(Debug, Clone)]
pub struct DocumentSymbolEntry {
    pub name: String,
    pub kind: SymbolKind,
    /// The range of just the symbol's name (selection_range in hierarchical
    /// shape, or the whole range when only flat data is available).
    pub name_range: Range,
    /// The range covering the entire definition (signature + body for
    /// functions, brace-enclosed body for types, etc.).
    pub full_range: Range,
}

/// One match from `workspace/symbol`.
#[derive(Debug, Clone)]
pub struct WorkspaceSymbolEntry {
    pub name: String,
    pub kind: SymbolKind,
    pub path: PathBuf,
    pub line: u32,
}

fn flatten_document_symbol(sym: DocumentSymbol, out: &mut Vec<DocumentSymbolEntry>) {
    out.push(DocumentSymbolEntry {
        name: sym.name,
        kind: sym.kind,
        name_range: sym.selection_range,
        full_range: sym.range,
    });
    if let Some(children) = sym.children {
        for child in children {
            flatten_document_symbol(child, out);
        }
    }
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
