# miniswe

A lightweight CLI coding agent for local LLMs.

Optimized for quantized small LLMs running locally via llama.cpp, Ollama, or vLLM. Works with any OpenAI-compatible API.

> **Tested with:** Devstral Small 2 (24B, Q4_K_XL and Q6_K) and Gemma-family 26B MoE (`unsloth/gemma-4-26B-A4B-it-GGUF`) via llama.cpp on RTX 3090.
> Other models (Qwen, Llama, etc.) should work via the OpenAI-compatible API but aren't validated yet — chat-template differences may require adjustments.

## What miniswe does

miniswe is a terminal coding agent built around one principle: **give the model the right tools and let it decide what context it needs.**

Instead of stuffing the whole repo into the system prompt, miniswe exposes a compact set of tools that the model calls on demand:

- **Pull-based context** — the model fetches a repo map, project profile, or architecture notes when it needs them.
- **AST-aware code intelligence** — tree-sitter parsing across 19 languages with a PageRank-based dependency graph for the repo map.
- **LSP diagnostics + navigation** — auto-downloads the right language server (rust-analyzer, pyright, gopls, etc.) for ~200 ms diagnostics and `goto_definition` / `find_references`.
- **Line-level edits with per-edit feedback** — `replace_range` and `insert_at` produce AST + LSP feedback and a revision table; regressions get rolled back with `revert`.
- **Unified compression** — once the conversation exceeds the token budget, older rounds are LLM-summarized and archived to `.miniswe/session_archive.md`.
- **MCP support** — connect any MCP server via `.mcp.json` (Claude Code compatible). *MCP integration is not yet tested end-to-end — expect rough edges.*
- **Permission model** — path jailing, shell approval, per-query web access, MCP approval.

## Prerequisites

- **Rust 1.85+** (edition 2024). Install via [rustup](https://rustup.rs/).
- **An OpenAI-compatible LLM endpoint.** The provided `start-*.sh` scripts use [llama.cpp](https://github.com/ggerganov/llama.cpp) — install `llama-server` from your package manager or build from source.
- **Hugging Face CLI** (`pip install -U "huggingface_hub[cli]"`) if you plan to use the bundled `start-*.sh` scripts to download models.
- **GPU with ≥16 GB VRAM** recommended for the models listed above. CPU-only works but is slow.
- Linux or macOS. Windows is untested.

## Run

```bash
# 1. Build
git clone https://github.com/michalszynkiewicz/miniswe.git
cd miniswe
cargo install --path .      # installs into ~/.cargo/bin

# 2. Start a local LLM server (pick one)
./start-devstral-small-2.sh       # Devstral Small 2 (24B)
./start-gemma4.sh                 # Gemma 26B MoE

# 3. In your project
cd /path/to/your/project
miniswe init

# 4. Use it
miniswe                                      # interactive REPL
miniswe "fix the bug in auth.rs"             # single-shot
miniswe -y "add error handling to main.rs"   # headless (auto-approve)
miniswe plan "how should I refactor auth?"   # plan mode (read-only)
```

The start scripts print the `hf download` command you need on first run.

### CLI commands

| Command | Description |
|---------|-------------|
| `miniswe` | Interactive REPL mode |
| `miniswe "message"` | Single-shot agent execution |
| `miniswe -y "message"` | Headless mode (auto-approve all permissions) |
| `miniswe init` | Initialize project (index, profile, graph) |
| `miniswe info` | Show project info and index stats |
| `miniswe config` | Show current configuration |
| `miniswe plan "question"` | Plan-only mode (no edits) |

### REPL keybindings

| Key | Action |
|-----|--------|
| `/new` | Clear history + scratchpad + plan |
| `/clear` | Clear conversation history only |
| `/help` | Show available commands |
| `Ctrl+O` | Toggle detail viewer for the latest tool result |
| `Ctrl+C` | Interrupt current LLM generation |
| `Ctrl+D` / `quit` | Exit |
| `Up` / `Down` | Scroll output when input is empty; otherwise input history |
| `PgUp` / `PgDn` | Scroll output faster |
| `Ctrl+Home` / `Ctrl+End` | Jump to top or bottom of output |

## Built-in tools

miniswe groups related actions under a single tool to keep the schema small. The model sees four grouped tools plus a handful of focused editing primitives (and `mcp_use` when MCP is configured).

### `file` — File I/O, search, and shell

| Action | Purpose |
|--------|---------|
| `read` | Read file contents with line numbers (auto-truncated to budget) |
| `delete` | Delete an existing file |
| `search` | Local ripgrep search — `query` (plain text) or `pattern` (regex) |
| `shell` | Execute shell commands (output saved to file if large, subject to approval) |
| `revert` | Revert files to a previous round via shadow-git snapshots |

### `code` — LSP and project intelligence

| Action | Purpose |
|--------|---------|
| `goto_definition` | Jump to a symbol's definition with source context |
| `find_references` | Find all references to a symbol |
| `diagnostics` | Compiler/linter errors (LSP-accelerated when available) |
| `repo_map` | PageRank-scored code structure overview |
| `project_info` | Project profile, guide, and lessons |
| `architecture_notes` | Architecture decisions from `.ai/README.md` |

### `web` — Search and fetch

| Action | Purpose |
|--------|---------|
| `search` | Web search via Serper when configured; otherwise falls back to GitHub repo search |
| `fetch` | Fetch a URL as markdown (large pages saved to file with preview) |

### `plan` — Structured planning

| Action | Purpose |
|--------|---------|
| `set` | Create a plan with compile-gated steps |
| `check` | Mark step done (runs compile gate if enabled) |
| `refine` | Replace one step with a more detailed flat sequence |
| `show` | View current plan |
| `scratchpad` | Save working notes (agent's memory) |

### Editing primitives

Every edit returns per-edit AST + LSP feedback plus a revision table, so the model can see immediately if it broke something and roll back.

| Tool | Purpose |
|------|---------|
| `replace_range` | Replace lines `[start..=end]` with `content`. Empty content deletes the range. |
| `insert_at` | Insert `content` after `after_line` (`0` = top of file; `<last line>` = append). |
| `write_file` | Create or overwrite a file; omit `content` to create an empty file. |
| `revert` | Restore a file to a prior revision. The undone revs stay visible as tombstones. |
| `show_rev` | Inspect a specific revision: operation, arguments, payload. |
| `check` | Run the project's compiler (`cargo` / `tsc` / `go vet` / `mvn` / `gradle`) for a deeper build check. |

### MCP

| Tool | Purpose |
|------|---------|
| `mcp_use` | Call any tool on a connected MCP server (only present when MCP is configured). |

> MCP support is not yet tested end-to-end. The wiring is in place but expect rough edges until we validate it against real servers.

### Smart mode (opt-in)

By default miniswe exposes the line-level primitives above. Smart mode replaces `replace_range` / `insert_at` / `revert` / `show_rev` / `check` with a single `edit_file` tool that runs an inner-model planner: you describe the change in natural language, and a planner produces and validates the patch for you. Trade-offs:

- **When it helps:** structural changes that span multiple regions of a file, where crafting precise line ranges is tedious.
- **Cost:** an extra LLM call per edit, and quantized small models tend to get stuck on it (they retry the same edit obsessively) — which is why it's opt-in.

Enable it in `.miniswe/config.toml`:

```toml
[tools]
edit_mode = "smart"    # default is "fast"
```

### Language server auto-setup

Run once; servers download to `~/.miniswe/lsp-servers/`:

| Language | Server | Detection |
|----------|--------|-----------|
| Rust | rust-analyzer | `Cargo.toml` |
| TypeScript/JS | typescript-language-server | `tsconfig.json` |
| Python | pyright | `pyproject.toml` |
| Go | gopls | `go.mod` |
| Java | jdtls | `pom.xml` / `build.gradle` |
| C/C++ | clangd | `CMakeLists.txt` |

## Configuration

`.miniswe/config.toml` in your project:

```toml
[model]
provider = "llama-cpp"
endpoint = "http://localhost:8464"
model = "devstral-small-2"
context_window = 60000
temperature = 0.15
max_output_tokens = 4096

[context]
repo_map_budget = 5000
max_rounds = 80
pause_after_rounds = 50

[lsp]
enabled = true
diagnostic_timeout_ms = 2000

[tools]
web_tools = true        # web(action='search'), web(action='fetch')
plan = true             # plan tool group
edit_mode = "fast"      # "fast" (default) or "smart" — see above

[web]
search_backend = "serper"
fetch_backend = "jina"
```

For general web search, put your Serper key in `~/.miniswe/serper.key`. Without it, `web(search)` falls back to GitHub repository search only.

## Contributing

```bash
cargo build                     # debug build
cargo build --release           # release build (default languages)
cargo test                      # run the test suite
cargo clippy                    # lint
cargo test --test e2e_lsp       # LSP integration tests (needs rust-analyzer)
cargo test --test e2e_snapshots # Snapshot/revert tests
```

### Tree-sitter language features

**Default (tier 1):** Rust, Python, JavaScript, TypeScript, Go

**Opt-in (tier 2):** Java, C, C++, Ruby, PHP, C#, Kotlin, Swift, Scala, Zig, Elixir, Haskell, Lua

```bash
cargo build --release --features all-languages
```

### Benchmarking

Docker-isolated run against a reference task (needs an LLM at `localhost:8464`):

```bash
./scripts/run-benchmark-docker.sh --timeout 2400 --max-rounds 80
```

### Test coverage notes

All tool actions have e2e tests except those requiring external services:
- `web(search)`, `web(fetch)` — need a Serper API key or network
- `mcp_use` — needs a running MCP server

LSP actions are tested in `e2e_lsp.rs`; revert is tested in `e2e_snapshots.rs`.

## License

MIT
