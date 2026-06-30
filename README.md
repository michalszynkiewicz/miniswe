# miniswe

A lightweight CLI coding agent for local LLMs.

Optimized for quantized small LLMs running locally via llama.cpp, Ollama, or vLLM. Works with any OpenAI-compatible API.

> Background & design notes: https://michalszynkiewicz.dev/blog/miniswe/

> **Tested with:** Devstral Small 2 (24B) and Gemma-family 26B MoE (`unsloth/gemma-4-26B-A4B-it-GGUF`) via llama.cpp on RTX 3090. Other models (Qwen, Llama, etc.) should work via the OpenAI-compatible API but aren't validated — chat-template differences may need adjustments.

## What it does

miniswe is built around one principle: **give the model the right tools and let it decide what context it needs.** Instead of stuffing the repo into the prompt, it exposes a compact set of tools the model calls on demand:

- **Pull-based context** — the model fetches a repo map, project profile, or architecture notes when it needs them.
- **AST-aware code intelligence** — tree-sitter parsing (19 languages) with a PageRank dependency graph for the repo map.
- **LSP diagnostics + navigation** — auto-downloads the right language server (rust-analyzer, pyright, gopls, …) for fast diagnostics and `goto_definition` / `find_references`.
- **Line-level edits with per-edit feedback** — edits produce AST + LSP feedback and a revision table; regressions roll back with `revert`.
- **Context compression** — when the conversation grows past budget, older turns are LLM-summarized and archived to `.miniswe/session_archive.md`.
- **MCP support** — connect any MCP server via `.mcp.json` (Claude Code compatible). *Not yet tested end-to-end.*
- **Permission model** — path jailing, shell approval, per-query web access, MCP approval.

## Install

Grab a prebuilt binary, or build from source.

**Option A — download a release.** Go to the [releases page](https://github.com/michalszynkiewicz/miniswe/releases), download the tarball for your platform (linux x86_64 musl, macOS x86_64, or macOS arm64), then:

```bash
tar xzf miniswe-*.tar.gz
# move the `miniswe` binary to a directory on your PATH, or add its location to PATH
```

**Option B — build from source** (needs Rust 1.85+, edition 2024):

```bash
git clone https://github.com/michalszynkiewicz/miniswe.git
cd miniswe
cargo install --path .            # installs into ~/.cargo/bin
```

Release binaries already bundle all languages; source-build is only needed for development or unsupported platforms.

## Run

You need an OpenAI-compatible LLM endpoint. The bundled `start-*.sh` scripts launch [llama.cpp](https://github.com/ggerganov/llama.cpp) (install `llama-server`; they print the `hf download` command on first run):

```bash
./start-devstral-small-2.sh       # Devstral Small 2 (24B)
./start-gemma4.sh                 # Gemma 26B MoE
```

Then, in your project:

```bash
miniswe init                                 # index, profile, graph
miniswe                                       # interactive REPL
miniswe "fix the bug in auth.rs"             # single-shot
miniswe -y "add error handling to main.rs"   # headless (auto-approve)
miniswe plan "how should I refactor auth?"   # plan mode (read-only)
```

Other commands: `miniswe info` (project/index stats), `miniswe config` (show current configuration).

In the REPL, `/new` clears history+scratchpad+plan, `/clear` clears history, `/help` lists commands, `Ctrl+O` toggles the detail viewer, `Ctrl+C` interrupts generation, `Ctrl+D` exits.

## Tools

The model sees four grouped tools plus a few editing primitives (and `mcp_use` when MCP is configured):

- **`file`** — `read`, `delete`, `search` (ripgrep), `shell`, `revert`.
- **`code`** — `goto_definition`, `find_references`, `diagnostics`, `repo_map`, `project_info`, `architecture_notes`.
- **`web`** — `search` (Serper, falls back to GitHub repo search), `fetch` (URL → markdown).
- **`plan`** — `set`, `check`, `refine`, `show`, `scratchpad`.
- **Editing primitives** — `replace_range`, `insert_at`, `write_file`, `revert`, `show_rev`, `check` (run the project compiler). Each edit returns AST + LSP feedback and a revision table so the model can see breakage and roll back.

## Configuration

`miniswe init` writes `.miniswe/config.toml`; `miniswe config` shows the live values. Key sections: `[model]` (provider, endpoint, model, context window, temperature), `[context]` (repo map budget, round limits), `[tools]` (web, plan), `[lsp]`, `[web]` (search/fetch backends).

For web search, put a Serper key in `~/.miniswe/serper.key`; without it, `web(search)` falls back to GitHub repository search.

## Languages

Default (tier 1): Rust, Python, JavaScript, TypeScript, Go, YAML. Opt-in (tier 2, build with `--features all-languages`): Java, C, C++, Ruby, PHP, C#, Kotlin, Swift, Scala, Zig, Elixir, Haskell, Lua. Language servers auto-download to `~/.miniswe/lsp-servers/` on first use.

## Contributing

```bash
cargo build --release           # default languages
cargo build --release --features all-languages
cargo test                      # test suite (e2e_lsp needs rust-analyzer)
cargo clippy
```

## License

MIT
