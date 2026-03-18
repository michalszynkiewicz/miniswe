# minime

A context-frugal CLI coding agent designed for 64K context windows on consumer hardware.

Optimized for small LLMs (Devstral Small 2, Qwen 2.5 Coder, etc.) running locally via llama.cpp, Ollama, or vLLM. Works with any OpenAI-compatible API.

## Quick Start

```bash
# Build
cargo build --release

# Initialize in your project
cd /path/to/your/project
minime init

# Run interactively
minime

# Run with a single message
minime "fix the bug in auth.rs"

# Plan mode (read-only exploration)
minime plan "how should I refactor the auth module?"
```

## Architecture

minime operates on one principle: **assemble exactly the right context for each step — never dump everything in and hope.**

### Core Components

- **Context Assembler** — Per-turn context building within a strict token budget. Each turn gets fresh context assembled from project profile, repo map, scratchpad, code snippets, conversation history, and lessons.
- **Knowledge Engine** — Offline indexing that scans source files, extracts symbols (functions, structs, types), and builds a file tree. Phase 2 will add tree-sitter AST parsing and PageRank-based dependency graphs.
- **Tool System** — 10 tools for code navigation, editing, search, shell execution, web access, and state management.
- **LLM Interface** — OpenAI-compatible API client with streaming support for llama.cpp, Ollama, vLLM, and cloud providers.
- **TUI** — Terminal output with colored status, tool call tracing, and streaming token display.

### Token Budget (64K window)

| Component | Tokens | % |
|-----------|--------|---|
| System prompt | 2,000 | 3.1% |
| Project profile | 800 | 1.3% |
| Repo map slice | 5,000 | 7.8% |
| Scratchpad | 1,500 | 2.3% |
| Retrieved snippets | 12,000 | 18.8% |
| Conversation history | 6,000 | 9.4% |
| **Available for output** | **~34,000** | **53%** |

## Commands

| Command | Description |
|---------|-------------|
| `minime` | Interactive REPL mode |
| `minime "message"` | Single-shot agent execution |
| `minime init` | Initialize project knowledge base |
| `minime info` | Show project info and index stats |
| `minime config` | Show current configuration |
| `minime plan "question"` | Plan-only mode (no edits) |
| `minime docs add <url>` | Cache documentation for offline use |
| `minime docs list` | List cached docs |
| `minime docs refresh` | Re-fetch cached docs |

## Tools

The agent has access to 10 tools:

| Tool | Purpose |
|------|---------|
| `read_file` | Read file contents with line numbers |
| `read_symbol` | Look up a specific function/class/type by name |
| `search` | ripgrep-based code search |
| `edit` | Search-and-replace file editing |
| `shell` | Execute shell commands |
| `task_update` | Update the task scratchpad (agent's memory) |
| `diagnostics` | Get compiler/linter errors |
| `web_search` | DuckDuckGo web search |
| `web_fetch` | Fetch URL as clean markdown (via Jina Reader) |
| `docs_lookup` | Search local documentation cache |

## Configuration

Configuration lives in `.minime/config.toml`:

```toml
[model]
provider = "llama-cpp"          # or "ollama", "vllm", "openai-compatible"
endpoint = "http://localhost:8080"
model = "devstral-small-2"
context_window = 65536
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
ram_budget_gb = 80

[web]
search_backend = "duckduckgo"   # or "searxng"
fetch_backend = "jina"          # or "local"
```

## Project Directory Structure

```
.minime/
├── config.toml           # Model, context, and web settings
├── profile.md            # Auto-generated project overview
├── guide.md              # Your custom instructions (<500 tokens)
├── lessons.md            # Accumulated tips from past sessions
├── scratchpad.md         # Current task state (auto-managed)
├── plan.md               # Active plan (auto-managed)
├── index/
│   ├── symbols.json      # Extracted symbol index
│   ├── summaries.json    # One-line file summaries
│   └── file_tree.txt     # Project file listing
├── snippets/             # Pre-chunked code (future)
├── sessions/             # Session logs (future)
└── docs/                 # Cached documentation
```

**Git-committed:** `profile.md`, `guide.md`, `lessons.md`
**Git-ignored:** Everything else (index, snippets, sessions, scratchpad, plan)

## LLM Server Setup

### llama.cpp (recommended)

```bash
llama-server \
  --model Devstral-Small-2-24B-UD-Q4_K_XL.gguf \
  --ctx-size 65536 \
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
# In .minime/config.toml:
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
