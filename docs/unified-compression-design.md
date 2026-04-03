# Context Compression (Implemented)

## Current State

### Implemented: Token-budget tool result masking

In `src/cli/commands/run.rs` `mask_old_tool_results()`:

- **Token budget**: `context_window / 2` for tool results
- **Newest-first**: walks backwards, keeps newest results full
- **LLM summarization**: batches masked results into chunks, sends to LLM for summarization
- **Chunking**: splits across multiple LLM calls if batch exceeds 1/3 context window
- **Heuristic fallback**: rich summaries (function signatures, struct defs) if LLM fails
- **Archive**: full content saved to `.miniswe/tool_history.md`, pointer in context
- **Dedup**: `archived_indices` HashSet prevents re-archiving same results
- **tool_call_id matching**: matches messages by ID, not positional counting

### Implemented: Dynamic tool output limits

`config.tool_output_budget_chars() = context_window / 10`

Large results from web_fetch, shell, docs_lookup get saved to file with preview + pointer:
- web_fetch → `.miniswe/web_cache/<url>.md`
- shell → `.miniswe/shell_output/cmd_<time>.txt`
- docs_lookup → points to `.miniswe/docs/<lib>.txt`

### Implemented: Unified timeline compression

`src/context/compressor.rs` — replaces separate tool masking + history compression.

**Budget split** (fractions of context_window):
- Output headroom: 1/6 (~17%)
- Compressed summary: 1/6 (~17%)
- Raw history: 1/4 (~25%)
- Work zone: rest (~42%)

**Trigger**: when non-system message tokens exceed 1/4 context_window.

**Flow**:
1. Count tokens in conversation (excluding system prompt)
2. If over budget, find split point to keep newest within 1/4
3. Send older messages to LLM for narrative summarization
4. Archive full content to `.miniswe/session_archive.md`
5. Replace compressed messages with `[Session summary]` + assistant acknowledgment
6. Model can `read_file(".miniswe/session_archive.md")` for details

**Replaces**: `mask_old_tool_results()` (removed, 240 lines) and `compress_history()` (still exists for `--continue` sessions but unified compressor handles in-session compression).
