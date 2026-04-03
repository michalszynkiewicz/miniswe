# Context Compression (Partially Implemented)

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

### Not yet implemented: Unified timeline compression

Design for replacing separate tool masking + history compression with a single timeline-aware compressor.

**Budget split** (fractions of context_window):
- Output headroom: 1/6 (~17%)
- Compressed summary: 1/6 (~17%)
- Raw history: 1/4 (~25%)
- Work zone: rest (~42%)

**Triggers**:
1. Raw history > 1/4 context → compress oldest half into summary via LLM
2. Summary > 1/6 context → archive oldest summary to file, re-summarize

**Key idea**: one summary block that captures decisions, discoveries, failures, and pointers to archived content. The model calls `read_file(".miniswe/session_archive.md")` for details.

This replaces both `compress_history()` (currently dumb first-line truncation) and `mask_old_tool_results()` (currently separate system).
