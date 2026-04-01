# LSP (rust-analyzer) Integration Design

## Context

The model's auto-check runs `cargo check` after every file write (2-5s). This is slow — the model can make 3 bad edits before seeing the first error. rust-analyzer provides ~200ms diagnostics plus navigation tools (goto-definition, find-references) that eliminate blind exploration.

## Two Integration Modes

1. **Automatic** — replace `cargo check` in `auto_check()` with LSP diagnostics after writes
2. **Tools** — expose `goto_definition` and `find_references` for model-invoked navigation

## File Structure

```
src/lsp/
  mod.rs         — LspClient public API, re-exports
  client.rs      — spawn, initialize, document sync, queries, shutdown
  transport.rs   — Content-Length JSON-RPC framing, background reader
```

## Key Structs

### LspClient

```rust
pub struct LspClient {
    transport: Arc<LspTransport>,
    child: Mutex<Child>,
    ready: AtomicBool,                     // false until init handshake done
    crashed: AtomicBool,                   // true if rust-analyzer dies
    opened_files: Mutex<HashSet<String>>,  // track didOpen state
    project_root: PathBuf,
    _reader_handle: JoinHandle<()>,
}
```

### LspTransport

```rust
pub struct LspTransport {
    writer: Mutex<BufWriter<ChildStdin>>,
    pending: DashMap<i64, oneshot::Sender<Value>>,     // request/response correlation
    diagnostics: DashMap<String, Vec<Diagnostic>>,      // URI → latest diagnostics
    next_id: AtomicI64,
}
```

Background reader thread reads Content-Length framed messages from stdout:
- Response (has "id") → dispatch via oneshot channel
- `publishDiagnostics` notification → store in diagnostics DashMap
- EOF → set crashed flag

## Lifecycle

```
Session start → LspClient::spawn()
  ├─ Spawns rust-analyzer --stdio
  ├─ Starts background reader thread
  ├─ Sends initialize request
  ├─ Waits for response (30s timeout)
  ├─ Sends initialized notification
  └─ Sets ready = true

Agent loop (per round)
  ├─ edit/write_file → auto_check()
  │   ├─ LSP ready? → didChange → wait 2s for diagnostics → inject
  │   └─ LSP not ready? → cargo check (existing fallback)
  └─ goto_definition / find_references → LSP request → format result

Session end → LspClient::shutdown()
  ├─ Sends shutdown + exit
  └─ Kills process if needed
```

## Integration Points

### 1. `execute_tool` — add LSP parameter

```rust
pub async fn execute_tool(name, args, config, perms, lsp: Option<&LspClient>)
```

Update call sites: `run.rs:361`, `repl.rs:568`

### 2. `auto_check` — try LSP first, fallback to cargo check

```rust
async fn auto_check(path, config, result, lsp: Option<&LspClient>) {
    if let Some(lsp) = lsp && lsp.is_ready() && !lsp.has_crashed() {
        // send didChange, wait for publishDiagnostics (2s timeout)
        // if received: format + append to result, return
    }
    // existing cargo check logic (unchanged)
}
```

### 3. New tools

- `goto_definition(path, line, column)` → definition location + source context
- `find_references(path, line, column)` → list of reference locations

### 4. Config

```toml
[lsp]
enabled = true
diagnostic_timeout_ms = 2000
```

### 5. Dependencies

```toml
lsp-types = "0.97"
dashmap = "6"
```

## Graceful Degradation

- rust-analyzer not installed → log warning, use cargo check
- rust-analyzer crashes mid-session → set crashed flag, fallback to cargo check
- LSP not ready yet (startup) → cargo check until ready flag set
- Diagnostic timeout → cargo check for that one call

## Benchmark

Add `lsp` to provider ablation list. Toggle via `[lsp] enabled = false` in config.
