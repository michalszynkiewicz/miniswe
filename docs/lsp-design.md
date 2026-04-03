# LSP Integration (Implemented)

## Overview

rust-analyzer (or other language servers) provides fast diagnostics and code navigation. Integrated in `src/lsp/`.

## Architecture

```
src/lsp/
  mod.rs         — Re-exports
  client.rs      — LspClient: spawn, retry, init, diagnostics, queries, shutdown
  transport.rs   — Content-Length JSON-RPC framing, background reader
  servers.rs     — Multi-language server detection, auto-download, verification
```

## Supported Languages

| Language | Server | Detection | Install method |
|---|---|---|---|
| Rust | rust-analyzer | Cargo.toml | GitHub release or rustup |
| TypeScript/JS | typescript-language-server | tsconfig.json/package.json | npm install |
| Python | pyright | pyproject.toml/setup.py | npm install |
| Go | gopls | go.mod | go install |
| C/C++ | clangd | CMakeLists.txt | GitHub release |
| Java | jdtls | pom.xml/build.gradle | GitHub release |

## Key Features

- **Auto-download**: if not in PATH (or PATH binary doesn't work, e.g. rustup proxy), downloads to `~/.miniswe/lsp-servers/`
- **Binary verification**: runs `--version` to verify the binary works before using it
- **Retry logic**: 3 attempts with 2s delays if server crashes on startup
- **Stderr logging**: captures first 20 lines of server stderr for debugging
- **Graceful degradation**: falls back to cargo check/mvn compile if LSP unavailable

## Integration Points

1. **Auto-check** (`src/tools/mod.rs`): after edit/write_file, sends `didChange` to LSP, waits for diagnostics (~200ms vs 2-5s for cargo check)
2. **Tools**: `goto_definition(path, line, column)` and `find_references(path, line, column)` — registered when LSP is available and `config.tools.lsp_tools = true`
3. **Diagnostics tool**: tries LSP first, falls back to cargo check

## Config

```toml
[lsp]
enabled = true
diagnostic_timeout_ms = 2000

[tools]
lsp_tools = true
```
