# North Mini Code 1.0 (Cohere, 30B-A3B MoE) — first runs, 2026-08-23

## Setup

- Weights: `unsloth/North-Mini-Code-1.0-GGUF` `North-Mini-Code-1.0-UD-Q4_K_M.gguf`
  (19.20 GB, imatrix). GGUF header (parsed 2026-08-23): arch `cohere2moe`,
  49 layers (1 leading dense; sliding-window pattern 13 global + 36 SWA-4096),
  128 experts / 8 active (sigmoid gating, no shared expert, expert FF 768),
  32 heads / 4 KV heads x 128, vocab 262144, ctx 500k, rope theta 50000 (no
  scaling), BOS 2 / EOS 255001 (`<|END_OF_TURN_TOKEN|>`), tokenizer pre
  `tiny_aya`. Embedded chat template (12478 chars) == the HF repo's
  `chat_template.jinja`: Cohere `<|START_THINKING|>…<|END_THINKING|>` /
  `<|START_ACTION|>[{…}]<|END_ACTION|>` format; thinking is gated by the
  `reasoning` kwarg (default **true**) or `reasoning_effort: "none"`; there is
  NO `enable_thinking` variable, so miniswe's per-request kwarg is inert.
- Launcher: `start-north-mini-code.sh` — 60k ctx, q8_0 KV, `--n-cpu-moe 10`
  (experts 0.34 GB/MoE layer; ~19.2 GB on disk needs ~3.4 GB off the card to
  stay inside the 20 GB budget), temp 1.0 / top-p 0.95 per the card,
  `--chat-template-kwargs '{"reasoning":false}'` server default = instruct arm
  (the server merges request kwargs over defaults key-by-key, so miniswe's
  `enable_thinking` doesn't clobber it). `MINISWE_REASONING=true` for a
  thinking arm (untested; card says the model wants prior reasoning passed
  back, which miniswe never does).
- Fit: _pending_
- Probes: _pending_


## Server + probes (20:02)

- `start-north-mini-code.sh` (UD-Q4_K_M, ctx 60k, KV q8_0, `--n-cpu-moe 10`, `reasoning=false`): model loaded from page cache in 3 s; card at 20.25 GB used (≈17 GB for the model over the ~3.3 GB baseline from other processes). Decode ~98 tok/s, prefill ~550 tok/s.
- Probe 1 (tool call): `finish=tool_calls`, empty content, `read_file` parsed with arguments, no reasoning_content. Probe 2 (tool-result round trip): clean one-paragraph answer.
- `reasoning=false` renders the generation prompt as `<|CHATBOT_TOKEN|><|START_THINKING|><|END_THINKING|>`; `true` leaves `<|START_THINKING|>` open. miniswe's `enable_thinking` kwarg is inert (merged key-by-key, `reasoning` stays false from the launcher).
- **Quirk: without a system message the model leaks** — 6/6 at temp 1.0 and 6/6 at 0.2, content = `answer<|END_THINKING|><|START_TEXT|>answer` (bare text, then it closes a thinking block it was never in, then repeats). With any system turn present: 0/6 leaks (3 questions × with/without tools, temp 0.2). The only template difference is the extra `<|SYSTEM_TOKEN|>` turn after the built-in preamble. Irrelevant for the bench (miniswe always sends a system prompt); would matter for a bare curl/REPL user. The template also renders a `# Available Tools` block with `[]` when no tools are given.

## Run 1 — `docker_20260823_200353` (20:04 → 21:01)

**0/6 — compile FAIL, 3407 s timeout, 457 rounds / 390 requests / 274 tool calls. Attempt 2 skipped (no time left).**

- Minute 0–2: `code(repo_map)`, `plan(set)`, one `replace_range` on `src/cli/mod.rs` L16-22 → rev_1 broken AST (29 errors); `revert` to rev_0 (its only revert, and a correct call); second `replace_range` L12-22 → rev_2 broken again (22 errors). Both edits are the **dropped-header bug**: the replacement block re-emits the fields around the new flag but swallows `#[derive(Subcommand, Debug)] pub enum Command {` below them (the diff-echo in the tool result shows both `-` lines, and the result ends with "If this is not exactly the edit you intended, call revert", `[ast] broken: 29:5`, the revisions table). Final diff: the flag is declared, the enum header is gone, `error: unexpected closing delimiter`.
- Minute 2–57: **`file(read src/cli/mod.rs)` 235 times** (258 reads of 274 calls), every response a bare tool call with no text. The whole escalation ladder fired and failed: 115 `[loop]` notes, "same read 3 times in a row" nudge, cold prompt eval (`cache_prompt=false`), then **57 forced compactions** ("nudge failed, forcing compaction next round") — each one 18.0k → 15.7k tokens, i.e. there was nothing to compact; the loop is not context-driven, which is counter-evidence to the warm-cache probe where forced compaction broke the read loop 8/8.
- Why nothing else fired: the auto-revert cascade needs 3 broken-AST edits (there were 2); the post-revert "smallest edit, braces balanced" hint only triggers after a revert (the model never reverted rev_2); the judge only runs on a done-gate block and `done` was never called; 22 errors above baseline for 55 minutes never triggers anything by itself.
- Model-side: ~98 tok/s, no parse errors, no thinking leakage, tool calls well-formed. It simply has no move after a broken edit except re-reading the file. Same shape as the uds read-loop (38–51 reads) but never escaping.

**Gap (cross-model, new — call it gap 10):** the read-loop ladder has no terminal step. After nudge → cold eval → forced compaction all fail on the same loop key, the harness "continues" indefinitely (57×). Candidates, in order of cheapness: (a) after K (=2?) failed forced compactions on one key with the tree AST-broken, auto-revert to the last `ast=ok` revision and inject the post-revert hint (the 12/12 tier-1 winner) — this turns the loop into the state the hint was proven on; (b) treat "AST broken + N rounds without an edit" as a gate-block equivalent and fire the judge (SCRAP/REWIND would both work here: rev_0 is clean). Neither applied; tier-1 replay material: llm_dumps req-040…req-440 of this run.

## Run 2 (instruct) — killed at 42 min per user

`docker_20260823_214[0-5]*` arm, launched as the second instruct run. At 42 min: **169 reads, ZERO edits on a clean tree** — the read loop this time without even a broken edit to explain it. User called it ("is this mini? if so, kill it and run it as thinking"); killed, and the second North data point became the thinking arm instead.

## Run 3 — thinking (`docker_20260823_215011`, 21:50 → 22:47)

**5/6 — smoke PASS, only `cargo test` failing, at timeout (3424 s, 571 rounds / 408 requests). One missed call site away from 6/6.**

- Server restarted with `MINISWE_REASONING=true` (template kwarg `reasoning=true`, leaves `<|START_THINKING|>` open). Pre-launch probes: think 171–1677 chars (modest, no budget risk), `finish=tool_calls`, correct-looking `replace_range` on a synthetic edit.
- Tool profile is a different species from instruct: **74 replace_range / 33 revert / 27 shell / 226 file / 11 plan / 8 refactor / 2 insert_at / 2 check** (instruct r1: 1 edit, 235 reads of one file; r2: 0 edits). Edits span the actual wiring: `src/cli/mod.rs`, `src/cli/commands/{run,repl}.rs`, `src/context/mod.rs`, `src/main.rs`, `tests/e2e_context.rs` — it widened `context::assemble` to 6 args and went through the test crate updating call sites.
- Failure mode: **pure clock death mid-fix, no pathology.** Only 2 `[loop]` notes all run (benign 3× reads, self-broken); 0 gate blocks, 0 debugger/judge fires, 0 forced compactions; `done` never called. Last edit at 20:46:28 UTC (~83 s before timeout) fixed the L418 call site; the very next tool result — `plan(check)` compile gate — told it the one remaining error (`tests/e2e_context.rs:434:25: E0061, 5 args vs 6`) and the clock expired before one more edit. `cargo test` fails only because that one call site keeps the test crate from compiling; lib compile/build/smoke all PASS.
- Pace: ~8.4 s/round average (408 requests / 3424 s) — thinking did not blow the round budget; reasoning arrives as `reasoning_content` and is discarded, so context stayed small (masking compressed 0 results all run).
- **Verdict:** Cohere's "works best with thinking enabled" is real and binary here — same weights, same harness: instruct 0/6 + 0/6-shaped-kill, thinking 5/6-at-timeout with clean convergent behavior. The honest scoreboard framing is "0/6 thinking-off, 5/6 thinking-on (timeout, one call site short)". A ~10% faster machine or a 3600 s timeout plausibly makes this 6/6.


## DROPPED from the roster — 2026-08-24 (user decision)

Why:
- **Instruct mode is pathological, not just weak**: 0/6 (broken edit → 235-read loop, 55 min) and a second run killed at 42 min with 169 reads / zero edits on a *clean* tree — it loops even without a broken edit to explain it. Clean tool calls, ~98 tok/s; the model simply has no next move. Not a harness or quant artifact we can fix on our side.
- **Thinking mode is mandatory and still only mid-tier**: 5/6 at timeout, one call site short, converging 4-6× slower than gemma (279-550s) and Laguna (566-792s, 3/3 first-try) on the same task, with 33 reverts against 74 replace_range edits. Best case at a longer timeout is "a slower gemma".
- **Strictly dominated**: gemma, Laguna, and Devstral all beat both of its modes on this bench; Cohere's own numbers put it below Qwen3.6 35B-A3B (SWE-bench Verified 67.6 vs 73.4); community reports say Q4 is marginal for tool-call loops (Q5/Q6 advised) — and a bigger quant doesn't fit our card comfortably alongside the 60k ctx.
- **Its unique contribution is already banked**: the cleanest gap-10 read-loop specimen we have. Findings here, replay material in `benchmark_results/docker_20260823_200353*/00_baseline/llm_dumps` (req-040…req-440) — none of that needs the model in rotation.

Launcher `start-north-mini-code.sh` stays on disk for archaeology; GGUF can be deleted if disk is needed.
