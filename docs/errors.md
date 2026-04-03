# Benchmark Analysis & Known Issues

## Latest Results (docker_20260403_214456)

Task: Add --system-prompt-override CLI flag

| Variant | Rounds | Attempts | Time | Result |
|---|---|---|---|---|
| baseline (all tools) | 46 | 1 | 974s | 6/6 PASS |
| no_extra_tools (core only) | 79 | 3 | 1271s | 6/6 PASS |

**Key finding**: `transform` tool saved 33 rounds and 2 retries by updating 12 call sites in one call.

## Tool Effectiveness

| Tool | Impact | Notes |
|---|---|---|
| **transform** (pattern mode) | HIGH | 12 call sites in one call. Game changer. |
| **transform** (block mode) | NEEDS TESTING | Added after model misused pattern mode for structural changes |
| **whitespace edit fallback** | HIGH | Eliminates #1 cause of edit failures |
| **stall detector** | MEDIUM | Breaks read loops after 20 calls without edits |
| **edit failure tracking** | MEDIUM | Forces write_file after 2 edit failures |
| **actionable error hints** | MEDIUM | "expected 6 args found 5" → "search for callers" |
| **LLM summarization** | LOW (this task) | Only triggers on long sessions (50+ rounds) |
| **get_repo_map** | NOT USED | Too small a project — needs stork (250 files) |
| **get_project_info** | NOT USED | Empty profile/guide for fresh projects |
| **goto_definition/find_references** | NOT USED | LSP was failing; now fixed |
| **scratchpad** | NOT USED (baseline) | Model doesn't use proactively |

## Bugs Fixed During Benchmarking

| Bug | Impact | Fix |
|---|---|---|
| UTF-8 truncation panics (6 locations) | CRITICAL | `truncate_chars()` everywhere |
| tool_msg_idx mismatch in masking | CRITICAL | Match by tool_call_id |
| Archive re-archives every round | HIGH | `archived_indices` HashSet |
| `set -e` kills Docker retry loop | HIGH | Remove `set -e` from container script |
| Stale target/ between variants | HIGH | `rm -rf target/` + Docker isolation |
| Empty repo map (LFS .gitignore) | HIGH | Fix LFS pointer + `miniswe init` fail-fast |
| miniswe init panic (UTF-8 in indexer) | HIGH | `truncate_chars()` in indexer |
| LSP not in Docker (rustup proxy) | MEDIUM | `verify_binary()` + auto-download fallback |
| Smoke test off-by-one | MEDIUM | Better round counting |
| Loop detection hangs forever | MEDIUM | `consecutive_loops >= 3` → break |
| LLM summarization 400 error | MEDIUM | Cap prompt to 1/3 context, chunking |
| `grep -c` returns 1 with `set -e` | LOW | `|| true` + `${var:-0}` defaults |

## Known Remaining Issues

1. **Unified compression not implemented** — tool masking and history compression are separate systems
2. **REPL masking uses positional index** — should match by tool_call_id like run.rs
3. **No timeout on LLM summarization calls** — could block if LLM hangs
4. **transform auto-revert runs cargo check synchronously** — blocks the agent loop for 2-5s
5. **Scratchpad underused** — model rarely calls task_update proactively
6. **MCP client blocking read_line** — can hang if server stops responding
