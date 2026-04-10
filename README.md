# miniswe

A lightweight CLI coding agent for local LLMs.

Optimized for quantized small LLMs running locally via llama.cpp, Ollama, or vLLM. Works with any OpenAI-compatible API.

> **Tested with:** Devstral Small 2 (24B, Q4_K_XL and Q6_K) via llama.cpp on RTX 3090.
> Multi-model support (Qwen, Llama, etc.) is planned but not yet validated — different models have different chat template constraints that may require adjustments to message formatting and role handling.

## Quick Start

```bash
# Build (default: Rust, Python, JS, TS, Go tree-sitter grammars)
cargo build --release

# Start the LLM server
./start-devstral-small-2.sh

# In another terminal: initialize in your project
cd /path/to/your/project
miniswe init

# Run interactively
miniswe

# Run with a single message
miniswe "fix the bug in auth.rs"

# Headless mode (auto-approve all permissions, for CI/scripts)
miniswe -y "add error handling to main.rs"

# Plan mode (read-only exploration)
miniswe plan "how should I refactor the auth module?"
```

## Architecture

miniswe operates on one principle: **give the model the right tools and let it decide what context it needs.**

### Core Components

- **Compact Tool Surface** — grouped tools (`file`, `code`, `web`, `plan`) plus focused top-level editors (`edit_file`, `write_file`) and optional `mcp_use`.
- **Pull-based Context** — Context actions (`code(action='repo_map')`, `code(action='project_info')`, `code(action='architecture_notes')`) let the model fetch project knowledge on demand instead of injecting everything into the system prompt.
- **Unified Compression** — Single-pass timeline compression. When conversation exceeds the token budget, older messages are LLM-summarized into a narrative and archived to `.miniswe/session_archive.md`.
- **Knowledge Engine** — Tree-sitter AST parsing (19 languages), PageRank-based dependency graph, doc-header extraction for file summaries, incremental re-indexing after edits.
- **LSP Integration** — Auto-downloads rust-analyzer (or other language servers). Provides ~200ms diagnostics after edits (vs 2-5s cargo check) plus `code(action='goto_definition')` and `code(action='find_references')`.
- **edit_file Tool** — LLM-powered atomic patching. Describe a change and it applies validated edits across one file, with pre-planned literal replacements for obvious repeated text edits, scoped smart edits for structural changes, split fallback for broad patches, and optional LSP validation.
- **Smart Edit** — 3-layer fuzzy matching (exact trim, indentation-preserving, line-similarity), bracket balance detection, edit failure tracking (forces write after 2 failures).
- **Tool System** — grouped tools plus `edit_file`, `write_file`, and optional MCP tools. Path jailing, shell approval, and per-query web access control.
- **LLM Interface** — OpenAI-compatible API with streaming, tool call parsing, multi-model routing (plan/code/fast roles).

### Context Budget (dynamic, based on context_window)

| Zone | Budget | Content |
|------|--------|---------|
| Work zone | ~42% | System prompt, tool schemas, current round |
| Raw history | 1/4 (25%) | Recent rounds in full |
| Compressed summary | 1/6 (17%) | LLM narrative of older rounds |
| Output headroom | 1/6 (17%) | Reserved for model response |

Per-result budget: `context_window / 10` (~6000 chars at 60K). Large results (web fetch, shell) saved to file with preview + pointer.

## Commands

| Command | Description |
|---------|-------------|
| `miniswe` | Interactive REPL mode |
| `miniswe "message"` | Single-shot agent execution |
| `miniswe -y "message"` | Headless mode (auto-approve all permissions) |
| `miniswe init` | Initialize project (index, profile, graph) |
| `miniswe info` | Show project info and index stats |
| `miniswe config` | Show current configuration |
| `miniswe plan "question"` | Plan-only mode (no edits) |

### REPL Commands

| Command | Description |
|---------|-------------|
| `/new` | Clear history + scratchpad + plan (fresh start) |
| `/clear` | Clear conversation history only |
| `/help` | Show available commands |
| `quit` | Exit |
| `Ctrl+O` | Toggle detail viewer for the latest tool result |
| `Ctrl+C` | Interrupt current LLM generation |
| `Ctrl+D` | Exit |
| `Up` / `Down` | Scroll output when input is empty; otherwise navigate input history |
| `PgUp` / `PgDn` | Scroll output faster |
| `Ctrl+Home` / `Ctrl+End` | Jump to top or bottom of output |

## Tools

Tools are grouped where that keeps the surface small. File editing uses focused top-level tools.

### `file` — File I/O and shell

| Action | Purpose |
|--------|---------|
| `read` | Read file contents with line numbers (auto-truncated to budget) |
| `delete` | Delete an existing file |
| `replace` | Replace text in a file (fuzzy matching fallback, `all=true` for every occurrence) |
| `search` | Local ripgrep search inside the project — `query` (plain text) or `pattern` (regex) |
| `shell` | Execute shell commands (output saved to file if large) |
| `revert` | Revert files to a previous round via shadow git snapshots |

### `code` — LSP and project intelligence

| Action | Purpose |
|--------|---------|
| `goto_definition` | Jump to symbol definition with source context |
| `find_references` | Find all references to a symbol |
| `diagnostics` | Get compiler/linter errors (LSP-accelerated if available) |
| `repo_map` | PageRank-scored code structure overview |
| `project_info` | Project profile, guide, lessons |
| `architecture_notes` | Architecture decisions from `.ai/README.md` |

### `web` — Search and fetch

| Action | Purpose |
|--------|---------|
| `search` | General web search via Serper when configured; otherwise falls back to GitHub repository search only |
| `fetch` | Fetch URL as markdown (large pages saved to file with preview) |

### `plan` — Structured planning

| Action | Purpose |
|--------|---------|
| `set` | Create a plan with compile-gated steps |
| `check` | Mark step done (runs compile gate if enabled) |
| `refine` | Replace one step with a more detailed flat sequence of steps |
| `show` | View current plan |
| `scratchpad` | Save working notes (agent's memory) |

### Top-level tools

| Tool | Purpose |
|------|---------|
| `edit_file` | LLM-powered code transformation — describe a change, it gets applied atomically |
| `write_file` | Create or overwrite a file; omit `content` only to create a new empty file |
| `mcp_use` | Call any tool on a connected MCP server |

`edit_file` accepts `path` and `task`, plus optional `lsp_validation`.

For broad edits such as updating repeated call sites, `edit_file` can pre-plan smaller steps before patching. The plan may use deterministic literal replacements for exact text changes and scoped LLM patching for ambiguous or structural regions. All edits are validated atomically before the file is finalized.

| `lsp_validation` | Behavior |
|------------------|----------|
| `auto` | Default. Use LSP diagnostics if available; skip if unavailable. |
| `require` | Require LSP diagnostics and reject patches that worsen file errors. |
| `off` | Skip LSP diagnostics, useful for unsupported text/config files. |

`write_file` accepts `path` and optional `content`.
If `content` is omitted, it creates a new empty file.
Do not use it for partial edits to existing files.

## LSP Support

Auto-detects project language and downloads the right LSP server:

| Language | Server | Detection |
|----------|--------|-----------|
| Rust | rust-analyzer | Cargo.toml |
| TypeScript/JS | typescript-language-server | tsconfig.json |
| Python | pyright | pyproject.toml |
| Go | gopls | go.mod |
| Java | jdtls | pom.xml / build.gradle |
| C/C++ | clangd | CMakeLists.txt |

Servers are downloaded to `~/.miniswe/lsp-servers/` on first use. No manual installation needed.

## Configuration

### `.miniswe/config.toml`

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

[web]
search_backend = "serper"
fetch_backend = "jina"
```

To enable general web search, put your Serper key in:

```text
~/.miniswe/serper.key
```

Without that key, `web(search)` falls back to GitHub repository search only.

## Benchmarking

Docker-isolated benchmark against a reference task:

```bash
# Run benchmark (needs LLM at localhost:8464)
./scripts/run-benchmark-docker.sh --timeout 2400 --max-rounds 80
```

## Tree-sitter Language Support

**Default (Tier 1):** Rust, Python, JavaScript, TypeScript, Go

**Opt-in (Tier 2):** Java, C, C++, Ruby, PHP, C#, Kotlin, Swift, Scala, Zig, Elixir, Haskell, Lua

```bash
# Build with all languages
cargo build --release --features all-languages
```

## Development

```bash
cargo build                     # debug build
cargo test                      # run 140+ tests
cargo clippy                    # lint
cargo test --test e2e_lsp       # LSP integration tests (needs rust-analyzer)
cargo test --test e2e_snapshots # Snapshot/revert tests
```

### Test coverage

All tool actions have e2e tests except those requiring external services:
- `web(action='search')`, `web(action='fetch')` — need Serper API key or network
- `mcp_use` — needs a running MCP server

LSP actions (`goto_definition`, `find_references`) are tested in `e2e_lsp.rs`.
Revert is tested in `e2e_snapshots.rs`.

## License

MIT
