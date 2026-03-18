# miniswe

A lightweight CLI coding agent for local LLMs.

Optimized for quantized small LLMs (Devstral Small 2 Q4_K_M, Qwen 2.5 Coder, etc.) running locally via llama.cpp, Ollama, or vLLM. Works with any OpenAI-compatible API.

## Quick Start

```bash
# Build (default: Rust, Python, JS, TS, Go tree-sitter grammars)
cargo build --release

# Build with all 19 languages
cargo build --release --features all-languages

# Initialize in your project
cd /path/to/your/project
miniswe init

# Run interactively
miniswe

# Run with a single message
miniswe "fix the bug in auth.rs"

# Plan mode (read-only exploration)
miniswe plan "how should I refactor the auth module?"
```

## Architecture

miniswe operates on one principle: **assemble exactly the right context for each step — never dump everything in and hope.**

### Core Components

- **Context Assembler** — Per-turn context building within a strict token budget. Compressed system prompt, profile, repo map, scratchpad, history with observation masking.
- **Knowledge Engine** — Tree-sitter AST parsing (19 languages), PageRank-based dependency graph, doc-header extraction for file summaries.
- **Compression Pipeline** — 5-layer deterministic compression: code format stripping, structured context format, import elision, history-as-diffs, observation masking. ~1.6x effective context multiplier.
- **Tool System** — 12 built-in tools + unlimited MCP tools via lazy-loading bridge.
- **LLM Interface** — OpenAI-compatible API client with streaming and tool call parsing.
- **MCP Support** — Standard `.mcp.json` config (Claude Code compatible). Lazy-loading: only one-line summaries in context (~10 tokens/server), full schemas resolved at execution time.

### Token Budget (50K window, Devstral Small 2 Q4_K_M on RTX 3090)

| Component | Tokens | % |
|-----------|--------|---|
| System prompt (compressed) | 1,200 | 2.4% |
| Project profile (compressed) | 350 | 0.7% |
| Repo map (PageRank-ranked) | 5,000 | 10% |
| MCP summaries | ~50 | 0.1% |
| Scratchpad | 1,500 | 3% |
| Retrieved snippets | 12,000 | 24% |
| Conversation history (compressed) | 6,000 | 12% |
| **Available for output** | **~24,000** | **48%** |

## Commands

| Command | Description |
|---------|-------------|
| `miniswe` | Interactive REPL mode |
| `miniswe "message"` | Single-shot agent execution |
| `miniswe init` | Initialize project (index, profile, graph) |
| `miniswe info` | Show project info and index stats |
| `miniswe config` | Show current configuration |
| `miniswe plan "question"` | Plan-only mode (no edits) |
| `miniswe docs add <url>` | Cache documentation for offline use |
| `miniswe docs list` | List cached docs |

## Tools

### Built-in (12 tools)

| Tool | Purpose |
|------|---------|
| `read_file` | Read file contents with line numbers |
| `read_symbol` | Look up a specific function/class/type by name |
| `search` | ripgrep-based code search |
| `edit` | Search-and-replace editing (best for large files) |
| `write_file` | Whole-file rewrite (preferred for files <200 lines, more reliable for quantized models) |
| `shell` | Execute shell commands |
| `task_update` | Update the task scratchpad (agent's memory) |
| `diagnostics` | Get compiler/linter errors |
| `web_search` | DuckDuckGo web search |
| `web_fetch` | Fetch URL as clean markdown (via Jina Reader) |
| `docs_lookup` | Search local documentation cache |
| `mcp_use` | Call any tool on a connected MCP server |

### MCP Tools (unlimited, via `.mcp.json`)

miniswe supports the standard `.mcp.json` configuration (same format as Claude Code):

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": { "GITHUB_TOKEN": "ghp_..." }
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"]
    }
  }
}
```

**Lazy-loading approach:** On startup, miniswe connects to MCP servers, fetches tool schemas, and caches them to `.miniswe/mcp/`. Only a one-line summary per server goes into the LLM context (~10 tokens each). Full schemas are resolved on the Rust side at execution time — zero context waste.

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
endpoint = "http://localhost:8080"
model = "devstral-small-2"
context_window = 50000          # conservative for RTX 3090 + display
temperature = 0.15
max_output_tokens = 16384

[context]
repo_map_budget = 5000
snippet_budget = 12000
history_turns = 5
history_budget = 6000
scratchpad_budget = 1500

[hardware]
vram_gb = 24
vram_reserve_gb = 3             # reserved for OS/display (usable: 21GB)
ram_budget_gb = 80

[web]
search_backend = "duckduckgo"   # or "searxng"
fetch_backend = "jina"          # or "local"
```

## Project Directory Structure

```
.miniswe/
├── config.toml           # Model, context, and web settings
├── profile.md            # Auto-generated project overview
├── guide.md              # Your custom instructions (<500 tokens)
├── lessons.md            # Accumulated tips from past sessions
├── scratchpad.md         # Current task state (auto-managed)
├── plan.md               # Active plan (auto-managed)
├── index/
│   ├── symbols.json      # Extracted symbol index (tree-sitter)
│   ├── graph.json        # Dependency graph + PageRank scores
│   ├── summaries.json    # Doc-header file summaries
│   └── file_tree.txt     # Project file listing
├── mcp/                  # Cached MCP server schemas
├── snippets/             # Pre-chunked code (future)
├── sessions/             # Session logs (future)
└── docs/                 # Cached documentation (llms.txt)

.mcp.json                 # MCP server configuration (Claude Code format)
```

**Git-committed:** `profile.md`, `guide.md`, `lessons.md`
**Git-ignored:** Everything else (index, mcp, snippets, sessions, scratchpad, plan)

## LLM Server Setup

### llama.cpp (recommended for RTX 3090)

```bash
llama-server \
  --model Devstral-Small-2-24B-UD-Q4_K_M.gguf \
  --ctx-size 50000 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --n-gpu-layers 99 \
  --flash-attn \
  --threads 8 \
  --port 8080
```

### Ollama

```bash
ollama serve
# In .miniswe/config.toml:
# [model]
# provider = "ollama"
# endpoint = "http://localhost:11434"
# model = "devstral"
```

## Development

```bash
# Build
cargo build

# Run tests
cargo test

# Run with tracing
RUST_LOG=debug cargo run -- init

# Check lints
cargo clippy
```

## Design Document

See [design.md](design.md) for the full architecture specification, context budget analysis, and rationale behind every design decision.

## License

MIT
