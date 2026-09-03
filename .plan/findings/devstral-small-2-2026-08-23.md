# Devstral Small 2 — re-baseline (2026-08-23)

Last benched 2026-05-15..19 on a binary 51 commits behind (pre diff-echo /
lazy compaction / debugger stack): ~5/6 with smoke FAIL, walls 683–951 s.

## Does it fit at 60k context?

**Yes.** `MINISWE_CTX_SIZE=60000 ./start-devstral-small-2.sh`
(UD-Q4_K_XL 14.5 GB weights, q8_0 KV, ngl 99, temp 0.15) → llama-server
**19,090 MiB** on the card, 22.5/24.6 GB total with the ~3.4 GB of other
processes. Just under the 20 GB budget; no CPU offload needed. The launcher's
50k default can move to 60k (not changed tonight — script edits need
approval). The Q6_K (19.3 GB) will not fit at any useful context.

Bench: `THINKING=false CTX_WINDOW=60000 ./scripts/run-benchmark-docker.sh
--timeout 3400 --max-rounds 600 --model devstral-small-2`, config verified
(`compaction = "lazy"`, `context_window = 60000`, `thinking = false`,
`temperature = 0.2`). Server restarted between runs. Same docker-script flag
set as the gemma/nemotron runs (auto_revert off, reactive_debugger off,
gate_context_reset on).

## Results

| Run | Dir | PASS/6 | Wall | Rounds | Compactions |
|---|---|---|---|---|---|
| 1 | `docker_20260823_032502` | **6** (attempt 2; attempt 1 = 5, smoke FAIL) | 3169 s | 267 + 26 | 5 |
| 2 | `docker_20260823_041911` | **6** (attempt 1) | 1051 s | 113 | 2 |

Both diffs touch the same six files (`cli/mod.rs`, `cli/commands/run.rs`,
`cli/commands/repl.rs`, `context/mod.rs`, `main.rs`, `tests/e2e_context.rs`),
321 / 379 patch lines — it threads the parameter properly through the test
crate instead of leaving it uncompiled (what nemotron never managed).

### Run 1 — 50 minutes of attempt 1

- **"Wired to nothing", knowingly.** Attempt 1 ended `status=ok` at round
  267 with `cargo test` green, but both call sites read
  `None,  // No system prompt override in REPL yet` / `... in run command
  yet`. The model even narrated it: "build succeeds with just warnings about
  unused variables, which is expected". Textbook plan degradation — the
  feature step collapsed into "compiles + tests pass". The bench's attempt-2
  feedback ("SMOKE TEST FAILED … Invocation: …") was enough: 26 rounds,
  both `None`s → `system_prompt_override`, PONG_42.
- **Churn:** 76 `replace_range`, **67 `revert`** (40 on
  `tests/e2e_context.rs`, 20 on `run.rs`), 18 `refactor add_param` calls —
  the same `add_param` on `run.rs` was re-issued 4× in 3 min (loop detector
  fired 4×) and again 3× on `repl.rs`. Only 4 `cargo test` calls.
- 5 compactions (every ~6–12 min once the context filled).

### Run 2 — clean

113 rounds, 17.5 min, one attempt: 45 `replace_range`, 33 `file`, 6 `revert`,
2 `refactor`, 9 `cargo test` calls, 0 loop detections, 2 compactions.

## Reading

- Devstral on this binary: **2/2 finish-line 6/6, 1/2 first-attempt** —
  same shape as tonight's gemma (2/3 finish, 1/3 first-attempt) and it
  clears the smoke check that it could not in May (diff-echo + attempt
  feedback are the plausible reasons; not isolated).
- Variance is huge (1051 s vs 3169 s) and entirely in edit mechanics: run 1
  is the revert/refactor thrash pattern, run 2 is the same plan executed
  without it.
- Candidate harness follow-ups it exposes: (1) the done-gate should include
  the smoke/feature check, not only build — run 1 declared done with an
  explicit `None` placeholder and would have been caught by any "is the new
  value used?" check; (2) `add_param` re-issue loop (same args 4× in 3 min)
  should escalate rather than just log `[loop]`.

## Run 3 — fixed script (2026-08-23 11:50, `docker_20260823_114957`)

Fixed script (`2fdc48c` code-default flags, `88ba610` gate v3), fresh
llama-devstral at 60k, same instruct/lazy arm as runs 1–2.

| | |
|---|---|
| Result | **0/6 at the timeout** (3406s, 320 rounds) — tree left mid-edit with a broken `src/context/mod.rs`, so even `cargo check` failed |
| Tool calls | 212: 83 `replace_range`, 62 `revert`, 46 `plan`, 20 `file`, 1 `code`; 4 compactions |
| Files touched | `cli/mod.rs`, `context/mod.rs` only — never reached `main.rs` / `run.rs` |
| Gate / debugger | 0 done-gate blocks (never claimed done), 0 debugger fires, 0 auto-revert fires |

**What happened.** Flag added cleanly (rev_1, 09:51). From rev_20 (09:58, ~9 min in)
to rev_80 (10:45) the run is one byte-identical edit: `replace_range
src/context/mod.rs L333-335` (+6 -3) that drops a closing brace (`unclosed
delimiter` at EOF), followed by `revert … rev_0`, then `plan check step 2`
("already checked"), then the same edit again — a **period-3 loop** repeated
~20×, interleaved early on with a second pair (L282-289). The model narrates
it every time: "I keep making syntax errors because I'm trying to modify too
much at once … let me take a completely different approach" — and re-issues
the identical call.

**Why the harness didn't stop it (three separate gaps):**
1. The streak detectors are period-1 (3 identical in a row) and period-2
   (A,B,A,B,A,B); this cycle is A,B,C so both are structurally blind to it.
   `spiral_reset` (3+ reverts of one file per turn) would have tripped on the
   first minute of it but is off by default.
2. The window detector *did* see it — `[loop] '… L333-335' recurred 4x in the
   window — forcing a cold prompt eval` fired **16 times** (6× the edit, 5× the
   revert, 4× the plan check, 1× L282-289) — but its only action is a cold KV
   re-eval, which a temp-0.2 model reproduces straight through. Detection
   without an escalation.
3. The auto-revert cascade counts *consecutive* broken-AST edits; the model's
   own `revert` between attempts resets it, so 20 broken edits never summed to 3.

**Stale LSP after revert.** Every `revert → rev_0` result reads `[ast] ok`
then `[lsp file] 1 error(s) L385:3: this file contains an unclosed delimiter`
(or `L341:9 unexpected closing delimiter`) — a diagnostic from the broken
state, on a file that is byte-identical to the compiling baseline. The model
is told its clean tree still has a syntax error; it is hard to imagine that
*not* feeding the "I keep making syntax errors" loop. Suggest: drop (or mark
stale) LSP diagnostics in the revert result until the LSP has re-analyzed the
restored content, or print the AST verdict only.

Run-to-run: runs 1–2 (script-default flags) 6,6; run 3 (code-default flags) 0.
The debugger flag is irrelevant here (it never had a chance to fire); this is
the edit-mechanics variance already noted for runs 1–2, at its worst.

## Run 4 — fixed script (2026-08-23 12:47, `docker_20260823_124659`)

| | |
|---|---|
| Result | **4/6** (test + smoke FAIL), **hit the 600-round cap** (601 rounds) at 3381s |
| Tool calls | 153: 67 `replace_range`, 43 `file`, 15 `revert`, 14 `refactor`, 7 `plan`, 3 `code`, 2 `shell`, 2 `insert_at`; 4 compactions |
| Diff | `cli/mod.rs`, `context/mod.rs`, `run.rs`, `repl.rs`, `tests/e2e_context.rs` — `main.rs` never touched, so the flag is parsed but not threaded (smoke: "Hello! I'm ready to assist you"); test crate left with 3× E0061 |
| Harness | 1 plan-check debugger fire, 1 `recurred 4x` cold-eval, 0 done-gate blocks (never claimed done), 14 "history pruned after refactor validator failure" |

**Where the 600 rounds went: 436 of them were a harness spin, not model
turns.** `[error:llm] tool call JSON truncated (max_tokens) — injecting hint
and continuing` fired 436×, in two bursts of ~4 ms per round:

- 11:09:18–19 UTC: **266 rounds in 1 second** (rounds 107→372)
- 11:38:47 UTC: **170 rounds in <1 second** (rounds 386→556)

The chain, verified from `llm_dumps` + the server log:
1. The model emits a `refactor` call whose `position` argument is an 11k-char
   code blob (it pastes the whole function body). Generation runs until the
   **context ceiling** (server: prompt 57,086 + 3,074 generated = 60,160,
   `truncated = 1`), taking 5m43s (round 104) and ~9 min (round 385).
2. llama-server streams the partial deltas, then ends with HTTP 500 *Failed
   to parse tool call arguments as JSON … missing closing quote*. miniswe's
   streaming path has already assembled the partial call: it dispatches it
   (tool result: `Invalid JSON in tool arguments: EOF while parsing a string
   at line 1 column 11209`) and **stores the assistant message with the
   unparseable `arguments` in history** (dump `…-000114.json`, msg 100). The
   comment at `run.rs` ~L1387 ("the server dropped the assistant turn, no
   tool_call_id was issued") is not what happens on the streaming path.
3. That poisoned assistant message survives `force_compress` (it sits in the
   verbatim tail: dump `…-000200.json` msg 28, still invalid).
4. Every later request fails in the server's chat template *before any
   generation* (server log: 268 exceptions, 0 `launch_slot` between them)
   with the same column-11210 parse error; `is_truncated_tool_call_error`
   matches it, the loop pushes another hint and `continue`s — one round per
   ~4 ms — until a later `force_compress` happens to drop the message.

Fix candidates: (a) on `is_truncated_tool_call_error`, scrub any assistant
`tool_calls` whose `arguments` fail `serde_json::from_str` from both
`messages` and `conversation_history` (replace with plain content, or drop the
call + its result pair) before continuing; (b) never persist a tool call with
unparseable arguments in the first place; (c) a spin guard — N consecutive
LLM errors with no completion in between should escalate (force compress →
abort), and a round should not count when no completion was received.
Cheap mechanical guard on top: cap a single `refactor` argument at a few
hundred chars — `position` is meant to be an anchor, not a body.

**Second Devstral-specific template failure — CORRECTED.** The 7× server
500 `Jinja Exception: After the optional system message, conversation roles
must alternate user and assistant roles except for tool calls and results`
were first blamed on the 14× "agent history pruned … after refactor
validator failure". Replaying the dumped requests against a live Devstral
(`replay-dump.py`, stream=false, max_tokens=1) refuted that: the post-prune
shape (dump `000391`, prune + user hint, with the "Understood." bridge that
`sanitize_messages` inserts) renders fine. The request that actually fails
is dump `000037` — the **reactive debugger's final-report nudge**: its
investigation loop sanitizes before every request, but the no-tools
"write your final report now" request built after the loop was sent raw,
ending `tool → user`. Devstral's template counts only `user` messages and
tool-call-free `assistant` messages for alternation, so `tool → user` is
`user → user` → 500 ×7 retries (63 s) and the diagnosis was lost. Inserting
`{"role":"assistant","content":"Understood."}` before the final user message
in the same dump → 200. Fix: `1879294`.

Run 4's model-side story is the same as run 3's: `assemble`'s signature was
hand-edited (E0061 6-vs-5 at the callsites from ~min 10), then the model
alternated between fixing callsites and re-editing the signature.

### Fixes applied (2026-08-23 afternoon, one commit each)

| Gap | Commit | What |
|---|---|---|
| period-3 edit/revert loop (run 3) | `29577c4` | cycle detector generalized to periods ≤ 4; the window detector escalates a recurring edit past the cold eval |
| stale LSP parse diagnostics after `revert` (run 3) | `e6025f8` | parse errors that contradict a clean tree-sitter parse are dropped and reported as stale |
| truncated tool call persisted → 436-round 500 spin (run 4) | `3fc77fa` | `sanitize_truncated_tool_calls` stubs unparseable arguments before the message is persisted (tool loop answers the stub with "not executed, re-issue smaller"); streaming assembler aborts `refactor`/`file`/`code` calls past 4096 chars of arguments; consecutive parse/cap failures are counted — scrub history at 2, abandon the turn at 4 |
| debugger final-report 500 (run 4) | `1879294` | `sanitize_messages` on the final-report request + regression test mirroring the template's alternation rule |

Gemma-4 MoE smoke on the new binary (`docker_20260823_160906`, instruct,
lazy, 60k, code-default flags): **6/6 on attempt 1, 298s, 187 rounds**, 94
tool calls (45 `file`, 26 `replace_range`, 13 `revert`, 6 `plan`), 0 loop
detector / debugger / gate events; the stale-LSP suppression fired on the
`src/config/mod.rs` reverts ("ignored 1 stale parse error(s) … the file
parses now"). No regression.

## Run 5 — harness fixes (2026-08-23 16:16, `docker_20260823_161629`)

Fixed script + the four fixes above, fresh llama-devstral at 60k, same
instruct/lazy arm.

| | |
|---|---|
| Result | **6/6 on attempt 1**, 1021s, 196 rounds (cf. run 2's 1051s) |
| Tool calls | 117: 33 `replace_range`, 30 `file`, 14 `plan`, 13 `shell`, 9 `revert`, 6 `insert_at`, 3 `refactor`, 3 `write_file`, 3 `code`, 3 `check`; 1 context-overflow compaction |
| Diff | the full six files incl. `main.rs` and `tests/e2e_context.rs` |
| Harness | 1 loop note (`find_references` ×3), stale-LSP suppression ×6 on reverts, **0 tool-call parse errors, 0 debugger 500s**, 0 gate blocks (first claim accepted) |

None of the run-3/4 pathologies recurred, so this is "no regression and the
traps are armed" rather than a live demonstration of each trap; the stale-LSP
drop is the only fix that visibly fired.

## Run 6 — harness fixes, second run (2026-08-23 16:49, `docker_20260823_164918`)

Same config as run 5 (fixed script + fixes, fresh llama-devstral at 60k).

| | |
|---|---|
| Result | **6/6 on attempt 1**, 2212s, 276 rounds |
| Tool calls | 156: 46 `file`, 44 `replace_range`, 20 `shell`, 16 `refactor`, 13 `plan`, 12 `revert`, 3 `insert_at`, 2 `write_file`; 2 context-overflow compactions |
| Harness | 7 loop notes (all the same call, below), **0 tool-call parse errors, 0 debugger 500s**, 0 gate blocks, stale-LSP/cycle/truncation traps never fired |

Twice as slow as run 5 for one reason: from 14:53 to 15:00 UTC the model
re-issued an **identical, malformed `refactor(add_param)`** call — the
`position` string ran on into
`,,system_prompt_override: Option<&str>,callsite_fill_in":"None"} wait, let
me fix the format:[TOOL_CALLS]refactor[ARGS]{` (Mistral's native call syntax
leaking into a JSON string value; the JSON itself parsed, so none of the
truncated-call machinery applied). The validator rejected it 14×, the period-1
loop detector intercepted it 7× more (21 attempts total), and the `[loop]
repeated 3x` note changed nothing — the model only escaped at 15:00 by
switching to `replace_range` and threading the callsites by hand (14
`replace_range` edits to `tests/e2e_context.rs`).

**Gap:** a period-1 identical *failing* call is handled by a note that the
model demonstrably ignores. Candidates: after the 2nd identical failure,
drop the failing call + its results from history and re-inject the
validator's example line; after the 3rd, force a compaction (the read-loop
fix that worked in the warm-cache probe) or abort the tool for the turn.
Not applied — parked behind the Laguna run.

Under the fixes Devstral is now 2/2 at 6/6 (1021s, 2212s), n=2.

### Devstral under the fixed script: 0, 4 vs 6, 6 under the old flags

Not a debugger/auto-revert effect — neither fired in run 3, and run 4's one
debugger fire was the plan-check one. Both new runs died on edit mechanics
(run 3: one broken edit re-issued 20×; run 4: two context-ceiling
generations + the truncation spin). The flag difference that *could* matter
is `gate_context_reset` (old script: on; code default: off), but it never had
a chance to fire either (0 gate blocks in both new runs). So: n=4, Devstral
is 2/4 on this task, and the two failures are harness-visible pathologies
with concrete fixes above.
