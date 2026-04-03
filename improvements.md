# Improvements

## Original ideas

* IDE like features: rename method throughout the codebase
  - STATUS: Partially done. `transform` pattern mode works per-file. Need a `refactor` tool that wraps search + transform across all matching files automatically.
  - IMPL: `refactor(find: "old_name", replace: "new_name")` → searches all files, runs transform on each. ~20 lines of code wrapping existing tools.

* add a parameter to a method throughout the codebase
  - STATUS: Done for single files via `transform(find: "assemble(", instruction: "add None as last arg")`. Same gap as rename — needs cross-file wrapper.

* wrap in a block? so that it can insert for/if, etc? Not sure it makes sense, is language specific
  - STATUS: Done via `transform` block mode: `transform(start_line: 10, end_line: 20, instruction: "wrap in if let Some(x) = override")`. LLM handles language-specific syntax.

* yaml tools and bash tools
  - Not needed as separate tools. read_file/edit/write_file work on any text format. The model handles YAML/bash syntax through its training.

## Improvements from benchmark analysis

### High priority (proven impact)

* **Cross-file refactor tool** — wraps search + transform. Model calls `refactor(find: "fn assemble(", instruction: "add system_prompt_override: Option<&str> as last parameter")` and it updates every file in the project. Currently the model has to search, then transform each file individually.

* **Context overflow prevention** — the unified compressor triggers when raw history exceeds budget, but doesn't account for tool definition tokens (~5-10K). Need to subtract tool schema overhead from the budget. The 32K overflow error in benchmarks was caused by this.

* **Scratchpad auto-update** — the model rarely uses task_update proactively. After each successful edit cycle (edit + compile pass), automatically append to scratchpad: "✓ edited [file] — [what changed]". The model always has an up-to-date checklist without having to remember to call task_update.

* **Faster inference** — test Q4_K_XL with all the tooling improvements. The tools (transform, whitespace fallback, error hints) reduce the reasoning burden, so a smaller/faster model might work just as well. ~40% faster per round.

### Medium priority (likely helpful, not yet proven)

* **Compile-before-continue** — after edit/write_file, if auto-check finds errors, inject them AND block the model from moving to the next file until errors are fixed. Currently the model sees the error but often ignores it and moves on, accumulating errors.

* **Call site finder** — when a function signature changes, automatically search for all call sites and list them in the tool result. The model already gets "IMPORTANT: search for callers" but a small model sometimes ignores this. Making it automatic removes the choice.

* **Token counting in prompts** — show the model how much context is left: "[Context: 18K/32K tokens used]". Helps it decide whether to read another file or start editing.

* **Diff preview** — before write_file on a file that exists, show the model a diff preview of what will change. Catches accidental truncation ("// rest unchanged") before it happens.

* **Smarter stall detection** — current detector counts tool calls without edits. Better: detect file re-reads (same file read 3+ times) and inject "You already read this file. Contents: [summary]" instead of letting the read proceed.

### Low priority (nice to have)

* **Session continuation** — `--continue` mode uses dumb compress_history. Should use the unified compressor's session_archive.md for rich context.

* **Multi-model routing** — use a fast small model (7B) for transform chunks and summarization, keep the main model (24B) for reasoning. Currently both use the same model via ModelRole::Fast which is just the default.

* **Git-aware edits** — before write_file, stash current changes. If the edit breaks compilation, `git stash pop` to restore. Cleaner than the current "model manually does git checkout".

* **Test generation** — after implementing a feature, auto-detect test files and suggest/generate tests. The model often skips testing or writes broken tests.

* **Progress reporting** — after each round, log: "Round N: [tool calls made] [files changed] [errors]" in a structured format. Helps with debugging and benchmark analysis.

## Architecture improvements

* **Extract masking/compression to shared module** — run.rs and repl.rs duplicate masking logic. The unified compressor is only in run.rs. REPL still uses old per-type-count masking. Should share the compressor.

* **Tool result budget enforcement** — tool_output_budget_chars() is checked in each tool individually. Should be enforced in execute_tool() centrally, with store-and-preview as the default for any oversized result.

* **Async LSP diagnostics** — currently auto_check blocks while waiting for LSP response (up to 2s). Should be async with a background notification handler that updates results as they arrive.
