# Remaining fail points

Post-hardening review, 2026-04-16. Lists issues NOT yet fixed — everything
from the earlier audit that was already addressed (shell metachar, symlink
jail, mutex poisoning, Drop guards, Ctrl+C, TUI channel) is excluded.

---

## High

### Non-atomic config writes
- **Where:** `src/config/mod.rs:465` (`fs::write`)
- **Risk:** crash mid-write truncates the global config file. Same
  pattern in `src/context/compressor.rs:379` (session archive).
- **Fix:** write to temp file, then `fs::rename` (atomic on POSIX).

### MCP client creates a new BufReader per request
- **Where:** `src/mcp/client.rs:174`
- **What:** each `request()` call wraps `child.stdout` in a fresh
  `BufReader`, discarding the old one's buffer state. If a prior
  response left partial data in the buffer, the next reader starts
  from a clean state but the stream position has moved → parse errors
  or hangs.
- **Fix:** create the `BufReader` once in `McpClient::connect` and
  store it as a field.

### Silent JSON arg fallbacks in tool dispatch
- **Where:** `src/tools/edit_orchestration.rs:35`,
  `src/tools/shell.rs:45-47`
- **What:** `args["path"].as_str().unwrap_or("")` treats missing keys
  as empty strings. For path this means "current directory" (silently
  wrong). For timeout this means "default" (silently different from
  what the model requested). `as_u64()` on non-number JSON silently
  returns `None`.
- **Fix:** validate required args upfront and return a clear error for
  missing / wrong-typed keys.

---

## Medium

### Streaming tool-call index defaults to 0
- **Where:** `src/llm/mod.rs:249`
- **What:** `tc_delta.index.unwrap_or(0)` on a malformed SSE chunk
  defaults the tool-call index to 0. If the LLM server omits the
  index field, the delta is silently merged into tool call 0 —
  potentially corrupting a legitimate call.
- **Fix:** log at debug if `index` is missing; skip the delta rather
  than guessing.

### Silent file-write failures in fallback paths
- **Where:** `src/mcp/registry.rs:78` (MCP tool cache),
  `src/context/compressor.rs:337` (archive read),
  `src/tools/edit_orchestration.rs:287-289` (LSP regression revert)
- **What:** `let _ = fs::write(...)` and `unwrap_or_default()` on
  reads swallow I/O errors. If a revert after an LSP regression fails,
  the file stays broken and the tool reports success.
- **Fix:** log warnings on write failures. Distinguish "file missing"
  (OK) from "permission denied" (warn).

---

## Low

### Crossterm event poll errors silently ignored
- **Where:** `src/tui/event.rs:56,111`
- **What:** `event::poll(...).unwrap_or(false)` — if terminal state
  corrupts (e.g. broken pipe), the key reader silently stops receiving
  events. User sees a frozen TUI with no error message.
- **Fix:** log the poll error; optionally set a "terminal broken" flag
  that the main loop can display.

### Non-streaming LLM error messages lack HTTP context
- **Where:** `src/llm/mod.rs:160`
- **What:** parse failures on the non-streaming response path show
  "parse LLM response" without the HTTP status or response body.
  Impossible to debug a misconfigured endpoint.
- **Fix:** include the status code and first ~200 chars of the body in
  the error.

### MCP response logging
- **Where:** `src/mcp/client.rs:198-199`
- **What:** empty / unparseable lines from MCP servers are silently
  skipped. No way to tell if the server is sending garbage or is
  simply quiet.
- **Fix:** log at debug for skipped lines.

---

## Already addressed (for reference)

| ID | Issue | Status |
|---|---|---|
| H1 | Shell allowlist metachar bypass | Fixed + 7 tests |
| H2 | Symlink jail escape | Fixed + 4 tests |
| H3 | TUI corruption on broken channel | Fixed + 1 test |
| M1 | Ctrl+C busy-loop | Fixed |
| M2 | RunningShellCommand Drop | Fixed + 1 test |
| M3 | PID-reuse race | Fixed + 1 test |
| M6 | Mutex poisoning cascade | Fixed (parking_lot) |
| — | LspClient Drop | Fixed |
| — | Native search (no rg/grep shell-out) | Fixed + 6 tests |
