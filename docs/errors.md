# Benchmark Analysis: baseline vs no_extra_tools

## Run: docker_20260403_214456
Task: Add --system-prompt-override CLI flag

## Results

| Variant | Rounds | Attempts | Time | Result |
|---|---|---|---|---|
| baseline (all tools) | 46 | 1 | 974s | 6/6 PASS |
| no_extra_tools (core only) | 79 | 3 | 1271s | 6/6 PASS |

## What made the difference: `transform` tool

### Baseline (1 attempt, 46 rounds)
The model used `transform` once:
```
transform(find: "context::assemble(&config,", instruction: "Add None as last argument")
→ Transforming 12 occurrences in tests/e2e_context.rs
```
This updated all 12 call sites in one tool call. The model also wrote 3 tests for the new feature. Everything compiled and tests passed on first try.

### no_extra_tools (3 attempts, 79 rounds)

**Attempt 1 (compile:FAIL)**: Model tried `sed` to update call sites but the loop detector caught repeated sed attempts. The sed commands failed because the regex was wrong for multi-line calls. Attempt ended with compile errors (function signature mismatch).

**Attempt 2 (test:FAIL)**: Model fixed the compile error (updated main.rs call sites) but the test file still had old call sites. Used `edit` on tests/e2e_context.rs but only fixed some occurrences.

**Attempt 3 (6/6 PASS)**: Model finally used `task_update` to plan, ran cargo test, read the test file, and fixed the remaining call sites with targeted edits.

## Tool effectiveness

| Tool | Used in baseline? | Impact |
|---|---|---|
| **transform** | YES (1 call) | **HIGH** — Saved 33 rounds and 2 retries. THE key differentiator. |
| **get_repo_map** | NO | Not used this run. Model navigated via search instead. |
| **get_project_info** | NO | Not used. |
| **get_architecture_notes** | NO | Not used. |
| **goto_definition** | NO | LSP failed to start (channel closed). |
| **find_references** | NO | LSP failed to start. |
| **task_update** | NO (baseline), YES (no_extra attempt 3) | Only used as a recovery tool when stuck. |

## Key findings

1. **transform is the killer feature.** One call replaced 12 occurrences across a 439-line file. Without it, the model spent 2 extra attempts trying sed, python scripts, and manual edits.

2. **context tools (repo_map, project_info) weren't used.** For this task on this codebase (~20 files), the model navigated fine with `search` and `read_file`. These tools would matter more on larger codebases (250+ files).

3. **LSP failed in Docker.** rust-analyzer crashed on startup (channel closed). Even with retry logic, it couldn't initialize. The stderr logging was added but not in this build. Need to investigate.

4. **Scratchpad wasn't used by baseline.** Only used by no_extra_tools on attempt 3 as a recovery plan. The model should use it proactively, not as a last resort.

5. **No masking/summarization was triggered.** 46 rounds at 32K context didn't exceed the token budget. The summarization infrastructure worked but wasn't needed for this task size.

6. **no_extra_tools tried sed and python as workarounds.** Without transform, the model improvised with shell commands — but regex escaping in sed is exactly the kind of thing small models get wrong. The loop detector caught repeated failed attempts.

## Recommendations

### Keep (high impact)
- **transform tool** — proven game-changer
- **Whitespace-normalized edit fallback** — prevents common edit failures
- **Stall detector** — breaks read loops
- **Actionable error hints** — helps model recover from compile errors
- **Edit failure tracking** (force write_file after 2 failures)

### Investigate (potentially high impact, not tested yet)
- **LSP** — fix Docker startup, could help on larger codebases
- **get_repo_map** — needs larger codebase (stork) to show value
- **LLM summarization** — needs longer sessions (50+ rounds) to trigger

### Consider dropping (low impact this run)
- **get_project_info** — profile/guide/lessons are empty/template for fresh projects
- **get_architecture_notes** — .ai/README.md doesn't exist initially

### Fix needed
- **LSP in Docker** — rust-analyzer crashes on startup. Need stderr logs to diagnose.
- **Scratchpad usage** — model doesn't use task_update proactively. Consider stronger nudge or auto-update after edits.
