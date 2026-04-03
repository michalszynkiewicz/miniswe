# Unified Context Compression Design

## Problem

Two separate compression systems (tool result masking + history compression) treat the conversation as disconnected fragments. The model experiences it as one narrative — the compressor should too.

## Design: Single-pass timeline compression

Instead of separate tool/history compression, one unified compressor that sees the entire message stream and produces:

1. **A narrative summary** of older rounds (decisions, discoveries, failures)
2. **Pointers to archived content** for full retrieval via `read_file`
3. **Raw recent rounds** kept in full

## Context Budget (fractions of context_window)

```
┌─────────────────────────────────────┐
│ Work zone        (rest, ~42%)       │  system prompt + tool schemas + 
│                                     │  current round content
├─────────────────────────────────────┤
│ Raw history      (1/4, ~25%)        │  last N rounds in full
├─────────────────────────────────────┤
│ Compressed summary (1/6, ~17%)      │  LLM-written narrative of old rounds
├─────────────────────────────────────┤
│ Output headroom  (1/6, ~17%)        │  reserved for model response
└─────────────────────────────────────┘
```

For 32K context: work=~13K, raw=~8K, summary=~5K, output=~5K.

## Triggers

1. **Raw → Summary**: when raw history tokens > context_window/4, compress
   the oldest half of raw into the summary zone via LLM call.
2. **Summary → File**: when summary tokens > context_window/6, archive the
   oldest summary to `.miniswe/session_archive.md` and re-summarize.

## Compression Flow

```
Before compression:
  [system] [summary_of_1-20] [raw_round_21] [raw_round_22] ... [raw_round_40] [current]
                                                                    ↑ exceeds 1/4

After compression:
  [system] [summary_of_1-30] [raw_round_31] ... [raw_round_40] [current]
                ↑ LLM summarized rounds 21-30 into the existing summary
```

## LLM Summarization Prompt

```
Summarize this segment of a coding session. Include:
- What was attempted and why
- Key discoveries (file locations, function signatures, patterns)
- What succeeded and what failed
- Current state and next steps

Conversation segment:
[round 21-30 messages here, including tool calls and results]

Previous summary (incorporate into yours):
[existing summary of rounds 1-20]
```

## Archive Format (.miniswe/session_archive.md)

```markdown
# Session Archive

## Rounds 1-20 (compressed at round 30)
Explored codebase for CLI flag implementation. Found Cli struct in
src/cli/mod.rs:8, max_rounds config in src/config/mod.rs:256.
Agent loop in src/cli/commands/run.rs:171, repl loop in repl.rs:41.

## Full reads (retrievable via read_file)
- src/cli/mod.rs: 59 lines, Cli struct with message/continue/yes fields
- src/main.rs: 49 lines, dispatch to run/repl based on cli.command
- src/config/mod.rs: 383 lines, Config struct with ContextConfig.max_rounds
```

## What Replaces

- `compress_history()` in context/mod.rs — replaced entirely
- `mask_old_tool_results()` in run.rs/repl.rs — merged into unified compressor
- `summarize_tool_result()` in compress.rs — used as fallback only
- `tool_result_log` tracking — replaced by timeline-aware compression

## Implementation Plan

1. Create `src/context/compressor.rs` with `UnifiedCompressor` struct
2. It holds: raw messages, summary text, archive path, token budgets
3. After each round, call `compressor.maybe_compress(router)` which:
   - Counts raw history tokens
   - If over budget: batches oldest raw messages + current summary
   - Sends to LLM for summarization
   - Replaces old messages with new summary
   - Archives full content to file
4. `assemble()` uses the compressor's output instead of `compress_history()`
5. Remove separate `mask_old_tool_results` — the compressor handles it
6. Keep heuristic fallback for when LLM summarization fails
