# Fix Agent Read-Loops and Improve Compilation Fix Cycles

## Problem

The agent loops when fixing compilation errors: it edits a file, cargo check fails, it tries to re-read the file it just edited (because the old read_file result was masked), the edit tool fails (stale `old` content), repeat. The safety valve (`recent_calls` loop detector in repl.rs:434-453) catches this but the underlying cause is information loss.

## Change 1: Include updated file context in edit/write_file results

**Files:** `src/tools/edit.rs`, `src/tools/write_file.rs`

**edit.rs:** Already shows ~3 lines of context around the change (lines 89-101). Expand to ±10 lines. This gives the model enough surrounding code to attempt the next edit without re-reading.

**write_file.rs:** Currently returns only `"✓ Wrote src/foo.rs (150 lines, 4200 chars)"`. After a successful write, also return the last 30 lines of the file (or the full file if under 30 lines). This is cheap — the content is already in memory from the write. Format:

```
✓ Wrote src/foo.rs (150 lines, 4200 chars, +5 lines)
[tail]
 141│    }
 142│}
 ...
```

This lets the model see the end of what it wrote and verify it's correct, without a separate read_file call.

## Change 2: Include source context around cargo check errors

**Files:** `src/tools/mod.rs` (the `auto_cargo_check` function, lines 144-187)

When `auto_cargo_check` finds errors, parse each error location (regex: `([\w/]+\.rs):(\d+)`) and include ±5 lines of the source file around each error location. The file was just written, so read it from disk. Cap at 3 error locations to avoid context explosion.

Before (current):
```
[cargo check]
error[E0308]: mismatched types
  --> src/foo.rs:42:5
```

After:
```
[cargo check]
error[E0308]: mismatched types --> src/foo.rs:42
  40│ fn process(input: &str) -> Result<Response> {
  41│     let parsed = parse(input);
  42│     parsed.into()  // ← error here
  43│ }
```

This eliminates the need to call `read_file` after a compilation error — the model sees exactly what's wrong and where.

## Change 3: Track reads and inject "files in context" note

**Files:** `src/cli/commands/repl.rs`

Add a `Vec<(String, usize, usize)>` (path, round, line_count) to `run_agent_loop` tracking successful `read_file` and `read_symbol` calls. Before each LLM call, inject a short note into the last message or as an appended user message:

```
[files read this session: src/foo.rs (round 2, 150L), src/bar.rs (round 3, 80L), Symbol:Router (round 1)]
```

Cost: ~30-50 tokens. Tells the model "you already have this info, work from what you learned." When the actual content gets masked in older rounds, this reminder prevents unnecessary re-reads.

Implementation: after each `read_file`/`read_symbol` tool result in the tool execution loop (repl.rs ~464-503), push to the tracking vec. Before the LLM call (repl.rs ~340), append the summary to the messages — either as a small addendum to the system context or injected into the messages list.

## Change 4: Smarter observation masking — keep reads longer

**Files:** `src/cli/commands/repl.rs` (the `mask_old_tool_results` function, lines 508-533)

Current logic: mask all tool results older than `MASK_AFTER_RESULTS` (6) uniformly. Change to mask by tool type:

- `read_file`, `read_symbol`: keep the **last 3** in full, mask older ones
- `write_file`, `edit`: mask after **2** (the confirmation is low-value once processed)
- `shell`, `diagnostics`: mask after **2**
- `search`, `web_search`, `web_fetch`: mask after **1** (one-shot info)

In `mask_old_tool_results`, instead of using a flat index, iterate `tool_result_log` and count per-tool-type. Only mask a result when its type-specific count exceeds the threshold.

## Change 5: Generalize auto-check to other languages

**Files:** `src/tools/mod.rs`

Replace `auto_cargo_check` with `auto_check(path, config, result)` that dispatches by file extension/project type:

- `.rs` + `Cargo.toml` exists → `cargo check --message-format=short` (existing)
- `.ts`/`.tsx` + `tsconfig.json` exists → `npx tsc --noEmit 2>&1 | head -30`
- `.go` + `go.mod` exists → `go vet ./... 2>&1 | head -30`
- `.py` + (`pyproject.toml` or `setup.py`) → `python -m py_compile {path} 2>&1` (single-file check, fast)
- `.java` + `pom.xml` → `mvn compile -q 2>&1 | tail -30`; or `build.gradle` → `./gradlew compileJava -q 2>&1 | tail -30`
- `.c`/`.cpp` + `Makefile` → `make -n 2>&1 | head -30` (dry-run to check for errors)

Each returns errors in the same format: `[<tool> check]\n<error lines>` or `[<tool> check] OK`.

The error-location parsing from Change 2 should also be generalized — each language has a slightly different `file:line` format but they all follow `filename:line` or `filename(line,col)` patterns.

## E2E Test: edit-check-fix cycle without read loops

**File:** `tests/e2e_agent.rs` (or new `tests/e2e_fix_loop.rs`)

This test simulates the exact failure scenario: the model writes broken code, gets a compilation error, and needs to fix it without re-reading the file.

### Test: `edit_fix_cycle_no_reread`

```
Scenario:
1. Create a Rust project in temp dir with a Cargo.toml and src/main.rs containing valid code
2. Mock LLM call 1: returns edit tool call that introduces a type error
   (e.g., changes `let x: u32 = 42;` to `let x: u32 = "hello";`)
3. Execute the edit → auto_cargo_check fires → result contains error + source context
4. Mock LLM call 2: returns edit tool call that fixes the error
   (based on seeing the error context in the previous result)
5. Execute the edit → auto_cargo_check fires → result contains "OK"
6. Assert: the conversation never contained a read_file tool call

Setup:
- Use wiremock with `.up_to_n_times(1)` per mock to sequence the two LLM responses
- The first mock returns an edit that breaks compilation
- The second mock returns an edit that fixes it
- Run the tool execution manually (not the full agent loop — just the tool calls + result assembly)

Assertions:
- After step 3: result.content contains "[cargo check]" AND contains the error line AND contains source context lines (the ±5 lines around the error)
- After step 5: result.content contains "[cargo check] OK"
- No read_file tool was ever called (the model had enough context from edit results + error context to fix without re-reading)
```

### Test: `write_file_includes_tail`

```
Scenario:
1. Create test project
2. Call write_file with 50 lines of content
3. Assert result.content contains "[tail]" section
4. Assert the tail contains the last ~30 lines of the written content
```

### Test: `observation_masking_keeps_reads_longer`

```
Scenario:
1. Create tool_result_log with 10 entries:
   [read_file, write_file, shell, read_file, write_file, shell, read_file, write_file, shell, read_file]
2. Call mask_old_tool_results
3. Assert: the last 3 read_file entries are NOT masked (still have full content)
4. Assert: write_file and shell entries older than 2 are masked to summaries
```

### Test: `auto_check_includes_source_context`

```
Scenario:
1. Create a Rust project with Cargo.toml and src/main.rs
2. Write a .rs file with a deliberate type error at a known line
3. Call auto_cargo_check (or the new auto_check)
4. Assert: result contains the error message
5. Assert: result contains source lines around the error (numbered, ±5 lines)
6. Assert: the source lines include the line number mentioned in the error
```

## Priority

1. Change 2 (error source context) — directly fixes the loop trigger, testable immediately
2. Change 1 (expanded edit/write results) — reduces re-reads, simple
3. Change 4 (smarter masking) — keeps reads available longer
4. Change 3 (files-in-context tracking) — safety net for masked reads
5. Change 5 (multi-language auto-check) — generalizes beyond Rust
