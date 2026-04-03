# Benchmark Analysis & Known Issues

## Run: docker_20260403_235511 (pre-unified-compressor build)

### Results
| Variant | Rounds | Attempts | Time | Result |
|---|---|---|---|---|
| baseline | 74 | 2 | 2420s | 5/6 (test:FAIL) |

### What happened

**Attempt 1 (compile:FAIL):**
- Model edited all the right files (cli/mod.rs, main.rs, run.rs, repl.rs, context/mod.rs)
- Hit **32K context overflow** (32485 tokens) — the old masking didn't account for tool definition tokens
- Edits were incomplete when overflow killed the round

**Attempt 2 (5/6, test:FAIL):**
- Model used `transform` extensively (9 calls!) to fix test call sites
- BUT it called transform per-test-string instead of once for all: `transform(find: "context::assemble(&config, \"test\"")`, `transform(find: "context::assemble(&config, \"hello\"")`, etc.
- Each call updated 1-4 occurrences — correct but inefficient (9 calls instead of 1)
- The correct call would be: `transform(find: "context::assemble(", instruction: "add None as 6th argument")`
- Final test:FAIL was `unexpected closing delimiter: }` in e2e_context.rs — transform mangled the syntax on one of the 9 passes

**Root causes:**
1. **Context overflow (fixed)**: unified compressor with tool_def_tokens accounting prevents this
2. **Transform called too specifically**: model searched for each unique call pattern instead of the common prefix. The transform description could hint: "use the shortest unique pattern that covers all occurrences"
3. **Transform cumulative errors**: 9 sequential transforms on one file — each pass changes the file, and later passes may conflict with earlier changes. One pass left a stray `}`

### Are our fixes sufficient?

| Fix | Addresses | Sufficient? |
|---|---|---|
| Unified compressor + token accounting | Context overflow | YES — won't hit 32K anymore |
| Remove double compression | Data loss | YES |
| Transform exists | Call site updates | PARTIALLY — model calls it per-pattern instead of once |
| Transform auto-revert | Syntax errors | NO — only runs cargo check at end, doesn't catch mid-sequence errors |

### Remaining improvement needed
- Transform should hint "use shortest common pattern" in the description
- Transform on sequential calls should check compilation between each pass (not just at end)
- Or: model should use `transform(find: "context::assemble(", ...)` not `transform(find: "context::assemble(&config, \"test\"", ...)`

---


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
