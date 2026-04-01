# miniswe User Guide

## Quick Start

1. `miniswe init` — initialize a project (creates `.miniswe/` with index)
2. `miniswe "your task"` — single-shot mode (runs task, exits)
3. `miniswe` — interactive REPL mode
4. `miniswe plan "your task"` — plan-only mode (read-only, no edits)

## Configuration

Global config: `~/.miniswe/config.toml` (API keys, model, hardware settings).
Per-project overrides: `.miniswe/config.toml` (optional, overrides global).

Key settings:
- `model.endpoint` — LLM API URL (e.g., `http://localhost:8464`)
- `model.model` — model name (e.g., `devstral-small-2`)
- `model.context_window` — context size in tokens (default: 50000)
- `web.search_api_key` — Serper API key for web search
- `logging.level` — log verbosity: `info`, `debug` (default), `trace`
- `logging.enabled` — write session logs to `.miniswe/logs/` (default: true)

## Sessions and Continuity

Each run is a fresh session. The agent does not automatically remember previous sessions.

To maintain continuity across sessions:
- **Scratchpad** (`.miniswe/scratchpad.md`): the agent writes its current task and plan here via the `task_update` tool. It's cleared at session start. To continue work, tell the agent what was done before or paste the previous scratchpad content.
- **AI notes** (`.ai/README.md`): the agent updates this with architecture decisions and key changes after completing tasks. This persists across sessions and is automatically loaded into context.
- **Lessons** (`.miniswe/lessons.md`): accumulated tips and gotchas. Add entries manually or let the agent learn from mistakes. Keyword-matched and injected into context when relevant.
- **User guide** (`.miniswe/guide.md`): project-specific instructions for the agent (coding style, conventions, things to avoid). Always loaded into context.

To continue work from a previous session, you can:
1. Simply describe what needs to happen next — the agent reads `.ai/README.md` for context
2. Copy the previous scratchpad and paste it as your message
3. Say "continue the work on X" — the agent will explore the codebase to orient itself

## REPL Commands

- `/clear` — clear conversation history
- `/new` — clear history + scratchpad + plan (fresh start)
- `/help` — show available commands
- `quit` or `exit` or Ctrl+D — exit

## Keyboard Shortcuts (REPL)

- **Ctrl+O** — toggle detail viewer (full content of last tool result)
- **Ctrl+C** — interrupt current LLM generation
- **Ctrl+D** — quit
- **↑/↓** — navigate input history
- **PgUp/PgDn** — scroll output
- **Home/End** — move cursor in input
- **Ctrl+Home/End** — scroll to top/bottom of output

## Tools Available to the Agent

- `read_file(path, start?, end?)` — read a file with line numbers
- `read_symbol(name, follow_deps?)` — look up a symbol by name from the index
- `search(query, scope?, max?)` — ripgrep-based code search
- `edit(path, old, new)` — search-and-replace edit (for targeted fixes)
- `write_file(path, content)` — create or rewrite a file
- `shell(command, timeout?)` — execute a shell command
- `task_update(content)` — write scratchpad (must have ## Current Task and ## Plan)
- `diagnostics()` — run linter/type checker
- `web_search(query)` — search the web
- `web_fetch(url)` — fetch a URL as markdown
- `docs_lookup(library, topic?)` — search cached documentation

## Permission System

- File access is jailed to the project root (current directory). No absolute paths, no `../` escapes.
- Shell commands: safe commands (cargo, go, npm, git status, ls, grep, etc.) are auto-approved. Dangerous commands (rm -rf /, mkfs, etc.) are always blocked. Others prompt for approval.
- Web access: prompts on first use per session. Can approve all web access for the session.
- In headless mode (`--yes` flag): all permissions auto-approved (except blocklisted commands).

## Project Index

`miniswe init` scans the project with tree-sitter to build a symbol index:
- Symbols (functions, structs, classes, etc.) with line numbers
- File summaries (one-liner per file)
- Dependency graph with PageRank scores
- Personalized repo map (injected into context, ranked by relevance to current task)

The index is stored in `.miniswe/index/` and incrementally updated when the agent edits files. Run `miniswe init` again to do a full re-index.

## Logging

Session logs are written to `.miniswe/logs/` with timestamps. Configure via `logging.level`:
- `info` — tool calls and outcomes only
- `debug` — full LLM interactions, tool arguments and results
- `trace` — everything including context assembly stats and masking decisions

## Tips

- Keep files under 200 lines for best results with small models
- The agent auto-runs `cargo check` (or equivalent) after editing source files
- Use plan mode (`miniswe plan "task"`) to explore and plan without making changes
- The repo map is personalized per task — keywords from your message boost relevant files
- If the agent seems lost, check `.miniswe/logs/` for the session log
