# Demo video scenario: miniswe

A ~5-minute walkthrough showing miniswe solving a real coding task on a
local LLM, running entirely on a single machine with no cloud API calls.

---

## Setup (show briefly, ~30s)

**What to show:**
- Terminal: `./start-devstral-small-2.sh` launching llama.cpp
- The GPU activity (nvidia-smi or a system monitor widget showing VRAM usage)
- Quick shot of the config: `cat ~/.miniswe/config.toml` — endpoint is localhost

**Narration:** "miniswe is a coding agent that runs entirely on your local
hardware. No API keys, no cloud. I'm running Devstral Small 2 quantized
on a single RTX 3090. Let's give it a real task."

---

## Scene 1: Start the agent (~30s)

```
$ cd ~/projects/sample-web-api
$ miniswe
```

**What to show:**
- The TUI boots: header, model info, LSP starting
- The prompt appears at the bottom: `you>`
- Point out: tree-sitter indexing the project, LSP connecting

**Narration:** "miniswe indexes the project with tree-sitter, starts
rust-analyzer for real-time diagnostics, and builds a repo map using
PageRank. The model sees the structure before I even type."

---

## Scene 2: Give it a task (~60s)

```
you> add a /health endpoint that returns the server uptime as JSON
```

**What to show:**
- The spinner while the model thinks
- Streaming tokens appearing in the output pane
- The model calling tools:
  - `file(search "fn main")` → finds the entry point
  - `file(read "src/main.rs")` → reads the server setup
  - `replace_range src/main.rs L42-42` → adds the health route
  - `insert_at src/handlers.rs @L0` → creates the handler function
  - `file(shell "cargo check")` → verifies it compiles

**Narration:** "It searches the codebase, reads the relevant files, and
makes targeted edits using replace_range — fast-mode edits that land
instantly without a second LLM call. After each edit it runs cargo check
to verify the project still compiles."

---

## Scene 3: LSP catches an error (~45s)

**What to show:**
- The model makes an edit that has a type error
- `[lsp] 1 error: mismatched types` appears in the tool result
- The model reads the error, understands it, and fixes it in the next edit
- `[lsp] 0 errors` after the fix

**Narration:** "rust-analyzer runs after every edit. Here the model made a
type error — it immediately sees the LSP diagnostic and corrects it in
the next turn. No manual intervention needed."

---

## Scene 4: Plan and track progress (~45s)

```
you> now add rate limiting to all endpoints, with configurable limits
```

**What to show:**
- The model creates a plan: `plan(set, steps=[...])`
- Plan appears with checkboxes:
  ```
  - [ ] Add rate-limiter dependency to Cargo.toml [compile]
  - [ ] Create rate_limiter.rs module [compile]
  - [ ] Wire middleware into the router [compile]
  - [ ] Add config parsing for rate limits [compile]
  ```
- The model works through steps, checking each off
- `plan(check, step=1)` → compile gate runs → passes

**Narration:** "For bigger tasks, the model creates a structured plan
with compile gates. Each step must pass cargo check before it can be
marked done. This prevents the model from accumulating broken state
across many edits."

---

## Scene 5: Permission system (~30s)

**What to show:**
- The model tries to run `cargo test`
- Permission prompt appears in the TUI:
  ```
  Allow shell command?
    $ cargo test
    [y]es / [n]o / [a]lways for this command:
  ```
- User types `a` → auto-approves cargo for the session
- Tests run and pass

**Narration:** "Every shell command requires explicit approval. I can
approve once for the session, or case-by-case. Web searches and MCP
tools go through the same permission system. The model can't exfiltrate
code or run destructive commands without my consent."

---

## Scene 6: Web search (optional, ~30s)

**What to show:**
- The model encounters an unfamiliar API
- `web(search "axum rate limiting middleware")` → permission prompt
- User approves
- Search results appear, model uses the information

**Narration:** "When the model needs external knowledge, it can search
the web — with your permission. Results are injected into context so
the model can use real documentation, not hallucinated APIs."

---

## Closing (~30s)

**What to show:**
- `git diff` showing all the changes
- Final `cargo test` passing
- The MINISWE_DEBUG=1 output showing token usage and KV cache stats

**Narration:** "All of this ran on a single GPU, with 50K context, no
cloud calls. The edits are real, the tests pass, and the whole session
used about 30K tokens. miniswe — a coding agent that respects your
hardware and your privacy."

---

## Production notes

- **Screen recording:** use a tool that captures the terminal at native
  resolution (e.g. `asciinema` for the terminal, OBS for the GPU overlay)
- **Font size:** 16pt+ for readability in the video
- **Speed:** real-time for the first tool call, then 2x for repetitive
  edits, back to real-time for the error/fix sequence
- **Project:** use a small but real Rust web API (axum or actix-web,
  ~200 lines). Pre-create it so the demo starts from working code.
- **Fallback:** if the model makes an unexpected mistake, that's fine —
  showing error recovery is more convincing than a perfect run
- **Length target:** 4-5 minutes edited, 8-10 minutes raw
