# miniswe

A lightweight CLI coding agent for local LLMs.

Optimized for quantized small LLMs (Devstral Small 2, Qwen 2.5 Coder, etc.) running locally via llama.cpp, Ollama, or vLLM. Works with any OpenAI-compatible API.

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

- **Pull-based Context** — Context tools (`get_repo_map`, `get_project_info`, `get_architecture_notes`) let the model fetch project knowledge on demand instead of injecting everything into the system prompt.
- **Unified Compression** — Single-pass timeline compression. When conversation exceeds the token budget, older messages are LLM-summarized into a narrative and archived to `.miniswe/session_archive.md`.
- **Knowledge Engine** — Tree-sitter AST parsing (19 languages), PageRank-based dependency graph, doc-header extraction for file summaries, incremental re-indexing after edits.
- **LSP Integration** — Auto-downloads rust-analyzer (or other language servers). Provides ~200ms diagnostics after edits (vs 2-5s cargo check) plus `goto_definition` and `find_references` tools.
- **Transform Tool** — LLM-powered multi-site code transformation. Pattern mode (same change to every occurrence) and block mode (structural change on a line range). Auto-reverts on compile failure.
- **Smart Edit** — Whitespace-normalized fallback, function signature change detection (warns to update call sites), edit failure tracking (forces write_file after 2 failures).
- **Tool System** — 18+ built-in tools + unlimited MCP tools. Path jailing, shell approval, per-query web access control.
- **LLM Interface** — OpenAI-compatible API with streaming, tool call parsing, multi-model routing (plan/code/fast roles).

### Context Budget (dynamic, based on context_window)

| Zone | Budget | Content |
|------|--------|---------|
| Work zone | ~42% | System prompt, tool schemas, current round |
| Raw history | 1/4 (25%) | Recent rounds in full |
| Compressed summary | 1/6 (17%) | LLM narrative of older rounds |
| Output headroom | 1/6 (17%) | Reserved for model response |

Per-result budget: `context_window / 10` (~3200 chars at 32K). Large results (web_fetch, shell) saved to file with preview + pointer.

## Commands

| Command | Description |
|---------|-------------|
| `miniswe` | Interactive REPL mode (Ctrl+R history search) |
| `miniswe "message"` | Single-shot agent execution |
| `miniswe -y "message"` | Headless mode (auto-approve all permissions) |
| `miniswe init` | Initialize project (index, profile, graph) |
| `miniswe info` | Show project info and index stats |
| `miniswe config` | Show current configuration |
| `miniswe plan "question"` | Plan-only mode (no edits) |
| `miniswe docs add <url>` | Cache documentation for offline use |
| `miniswe docs list` | List cached docs |

### REPL Commands

| Command | Description |
|---------|-------------|
| `/new` | Clear history + scratchpad + plan (fresh start) |
| `/clear` | Clear conversation history only |
| `/help` | Show available commands |
| `quit` | Exit |
| `Ctrl+C` | Interrupt current LLM generation |
| `Ctrl+R` | Search input history |

## Tools

### Core (always available)

| Tool | Purpose |
|------|---------|
| `read_file` | Read file contents with line numbers (auto-truncated to budget) |
| `read_symbol` | Look up a function/class/type by name via tree-sitter index |
| `search` | ripgrep search — `query` (plain text) or `pattern` (regex) |
| `edit` | Replace text in a file (whitespace-normalized fallback, signature change detection) |
| `write_file` | Create or rewrite files (preferred for files under 200 lines) |
| `transform` | LLM-powered multi-site transformation — pattern mode or block mode |
| `shell` | Execute shell commands (output saved to file if large) |
| `task_update` | Update the task scratchpad (agent's memory) |
| `diagnostics` | Get compiler/linter errors (LSP-accelerated if available) |

### Context (pull-based, toggleable via `[tools]` config)

| Tool | Purpose |
|------|---------|
| `get_repo_map` | PageRank-scored code structure overview |
| `get_project_info` | Project profile, guide, lessons |
| `get_architecture_notes` | Architecture decisions from `.ai/README.md` |

### LSP (when language server is available)

| Tool | Purpose |
|------|---------|
| `goto_definition` | Jump to symbol definition with source context |
| `find_references` | Find all references to a symbol |

### Web (toggleable via `[tools]` config)

| Tool | Purpose |
|------|---------|
| `web_search` | Web search via Serper (shows query, asks permission) |
| `web_fetch` | Fetch URL as markdown (large pages saved to file with preview) |
| `docs_lookup` | Search local documentation cache |
| `mcp_use` | Call any tool on a connected MCP server |

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
context_window = 32000
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
context_tools = true    # get_repo_map, get_project_info, get_architecture_notes
lsp_tools = true        # goto_definition, find_references
transform = true        # LLM-powered multi-site transformation
web_tools = true        # web_search, web_fetch, docs_lookup

[web]
search_backend = "serper"
fetch_backend = "jina"
```

## Benchmarking

Provider/tool benchmarks with Docker isolation:

```bash
# Compare baseline (all tools) vs core-only
./scripts/run-benchmark-docker.sh --timeout 2400 --max-rounds 80

# Local (faster, less isolated)
./scripts/bench-task-B-max-rounds-flag.sh --timeout 1800
```

See [docs/errors.md](docs/errors.md) for benchmark analysis and tool effectiveness data.

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
cargo test                      # run 160+ tests
cargo clippy                    # lint
cargo test --test e2e_lsp       # LSP integration tests (needs rust-analyzer)
cargo test --test e2e_snapshots # Snapshot/revert tests
```

### Test coverage

16/19 tools have e2e tests. Not tested (require network/external services):
- `web_search`, `web_fetch` — need Serper API key or network
- `mcp_use` — needs a running MCP server

LSP tools (`goto_definition`, `find_references`) are tested in `e2e_lsp.rs`.
Revert is tested in `e2e_snapshots.rs`.

## License

MIT
