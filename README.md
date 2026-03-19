# miniswe

A lightweight CLI coding agent for local LLMs.

Optimized for quantized small LLMs (Devstral Small 2 Q4_K_XL, Qwen 2.5 Coder, etc.) running locally via llama.cpp, Ollama, or vLLM. Works with any OpenAI-compatible API.

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

miniswe operates on one principle: **assemble exactly the right context for each step — never dump everything in and hope.**

### Core Components

- **Context Assembler** — Per-turn context building within a strict token budget. Compressed system prompt, profile, repo map, scratchpad, history with observation masking.
- **Knowledge Engine** — Tree-sitter AST parsing (19 languages), PageRank-based dependency graph, doc-header extraction for file summaries, incremental re-indexing after edits.
- **Compression Pipeline** — Deterministic compression: structured context format, stdlib import elision, history-as-diffs, observation masking.
- **Tool System** — 11 built-in tools + unlimited MCP tools via lazy-loading bridge. Path jailing, shell approval, per-query web access control.
- **LLM Interface** — OpenAI-compatible API client with streaming, tool call parsing, and Ctrl+C interruption.
- **MCP Support** — Standard `.mcp.json` config (Claude Code compatible). Lazy-loading: only one-line summaries in context (~10 tokens/server), full schemas resolved at execution time.

### Token Budget (50K window, Devstral Small 2 Q4_K_XL on RTX 3090)

| Component | Tokens | % |
|-----------|--------|---|
| System prompt (compressed) | 1,200 | 2.4% |
| Project profile (compressed) | 350 | 0.7% |
| Repo map (PageRank-ranked) | 5,000 | 10% |
| MCP summaries | ~50 | 0.1% |
| Scratchpad | 1,500 | 3% |
| Conversation history (compressed) | 6,000 | 12% |
| **Available for output** | **~24,000** | **48%** |

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
| `Ctrl+D` | Exit |

## Tools

### Built-in (11 tools)

| Tool | Purpose |
|------|---------|
| `read_file` | Read file contents with line numbers (comments preserved, stdlib imports elided) |
| `read_symbol` | Look up a function/class/type by name via index coordinates |
| `search` | ripgrep-based code search |
| `write_file` | Create or rewrite files (primary editing tool — writes complete file content) |
| `shell` | Execute shell commands (30s default timeout, permission required) |
| `task_update` | Update the task scratchpad (agent's memory) |
| `diagnostics` | Get compiler/linter errors |
| `web_search` | Web search via Serper or GitHub (shows query, asks permission) |
| `web_fetch` | Fetch URL as clean markdown via Jina Reader (shows URL, asks permission) |
| `docs_lookup` | Search local documentation cache (no network, always allowed) |
| `mcp_use` | Call any tool on a connected MCP server |

### Web Search

Web search uses Serper (Google results) when an API key is available, falling back to GitHub repository search (no key needed, uses `gh` token if available for higher rate limits).

```bash
# Option 1: Serper key file (recommended — free 2,500 queries/month at serper.dev)
mkdir -p ~/.miniswe
echo "your-serper-key" > ~/.miniswe/serper.key

# Option 2: environment variable
export SERPER_API_KEY="your-key"

# Option 3: no key — falls back to GitHub repo search (10-30 req/min)
```

### MCP Tools (unlimited, via `.mcp.json`)

miniswe supports the standard `.mcp.json` configuration (same format as Claude Code):

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": { "GITHUB_TOKEN": "ghp_..." }
    }
  }
}
```

On startup, miniswe connects to MCP servers, fetches tool schemas, and caches them. Only a one-line summary per server goes into LLM context (~10 tokens each). Full schemas are resolved at execution time.

## Permission System

All file access is jailed to the project root. Destructive actions require user approval:

| Action | Permission |
|--------|-----------|
| Read/write files in project | Always allowed (path-jailed) |
| Absolute paths or `../` traversal | Blocked |
| Shell commands (cargo, git, ls, etc.) | Auto-approved (allowlist) |
| Shell commands (other) | Prompted per command |
| Web search | Prompted (shows query being sent) |
| Web fetch | Prompted (shows URL and proxy info) |
| MCP tool calls | Prompted per server |
| `-y` flag | Auto-approves everything |

See [docs/safe-headless-execution.md](docs/safe-headless-execution.md) for running safely in CI/Docker.

## Tree-sitter Language Support

AST-based symbol extraction with tree-sitter. Each language is a feature flag:

**Default (Tier 1):** Rust, Python, JavaScript, TypeScript, Go

**Opt-in (Tier 2):** Java, C, C++, Ruby, PHP, C#, Kotlin, Swift, Scala, Zig, Elixir, Haskell, Lua

```bash
# Build with specific languages
cargo build --release --features "lang-rust,lang-java,lang-cpp"

# Build with all languages
cargo build --release --features all-languages

# Build without tree-sitter (regex fallback only)
cargo build --release --no-default-features
```

## Configuration

### `.miniswe/config.toml`

```toml
[model]
provider = "llama-cpp"          # or "ollama", "vllm", "openai-compatible"
endpoint = "http://localhost:8464"
model = "devstral-small-2"
context_window = 50000
temperature = 0.15
max_output_tokens = 16384

[context]
repo_map_budget = 5000
snippet_budget = 12000
history_turns = 5
history_budget = 6000
scratchpad_budget = 1500
max_rounds = 100                # hard limit on tool call rounds
pause_after_rounds = 50         # ask user to confirm continuation

[hardware]
vram_gb = 24
vram_reserve_gb = 3             # reserved for OS/display (usable: 21GB)
ram_budget_gb = 80

[web]
search_backend = "serper"       # or "github", "searxng"
search_api_key = ""             # or use ~/.miniswe/serper.key
fetch_backend = "jina"          # or "local"
```

A config for RTX 4050 laptops (6GB VRAM) is included at [4050-config.toml](4050-config.toml).

## LLM Server Setup

### llama.cpp (recommended)

Use the included start script:

```bash
# Download the model (first time)
mkdir -p models
hf download unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF \
  --include 'Devstral-Small-2-24B-Instruct-2512-UD-Q4_K_XL.gguf' \
  --local-dir models/

# Start the server
./start-devstral-small-2.sh
```

Or manually:

```bash
llama-server \
  --model Devstral-Small-2-24B-UD-Q4_K_XL.gguf \
  --ctx-size 50000 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --n-gpu-layers 99 \
  --flash-attn on \
  --threads 8 \
  --port 8464 --metrics
```

### Ollama

```bash
export OLLAMA_CONTEXT_LENGTH=50000
export OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_FLASH_ATTENTION=1
ollama serve

# In .miniswe/config.toml:
# provider = "ollama"
# endpoint = "http://localhost:11434"
```

## Debugging

```bash
# Show role sequence, token usage, and KV cache stats before each LLM call
MINISWE_DEBUG=1 miniswe
```

## Development

```bash
cargo build                     # debug build
cargo test                      # run tests
cargo clippy                    # lint
RUST_LOG=debug cargo run -- init  # with tracing
```

## Design Document

See [design.md](design.md) for the full architecture specification, context budget analysis, and rationale behind every design decision.

## License

MIT
