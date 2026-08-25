# Nemotron 3.5 Lightning — lazy-compaction rerun (2026-08-23)

Question from the plan: "rerun to check if it is better with lazy compaction".
Earlier 08-20..22 runs used `unified` compaction (4, 0, 5, 2 instruct; 5 hollow
thinking) — all at the 57-min timeout.

Common setup: `start-nemotron35-30b.sh` (patched `<think></think>\n` template,
`--n-cpu-moe 99` = all experts on CPU by design, q4_0 KV, ctx 60000),
`./scripts/run-benchmark-docker.sh --timeout 3400 --max-rounds 600
--model nemotron-3.5-lightning`, `COMPACTION=lazy` (script default),
config verified per run (`compaction = "lazy"`, `context_window = 60000`).
Same flag confound as the gemma runs (auto_revert off, reactive_debugger off,
gate_context_reset on).

## Run 1 — instruct (`THINKING=false`), `docker_20260823_012527`

**4/6, timeout** (3427 s, 478 rounds, 1 attempt — the timeout reaped it).
compile/build/help/parse PASS, **test FAIL, smoke FAIL**.

- **The flag was wired but not used.** Smoke answered `Pong! 🏓` (default
  prompt) instead of `PONG_42`: the override reached `Config` but was never
  threaded into the prompt that the model answers with. Same "flag exists,
  wired to nothing" shape as the 08-22 thinking run.
- **Test suite broken by the signature change:** `assemble()` got a 6th
  parameter and the `e2e_context` test crate was never updated —
  `error[E0061]: this function takes 6 arguments but 5 were supplied`
  (14 errors) → `cargo test` fails to compile.
- **Error churn instead of progress:** the same E0061 appeared **190 times**
  across the run's build output, `E0609: no field system_prompt_override on
  Config` 50 times. 125 `shell` calls, 55 `replace_range`, 14 `revert`,
  5 `refactor`, 18 `plan` rewrites (one plan grew to 17 steps).
- **Compaction:** 4 lazy compactions (23:37 / 23:52 / 00:02 / 00:11 UTC),
  each triggered by "context window exceeded" — i.e. the history hit 60k and
  was summarized, the model kept going. No runaway-summary blowup (the
  summarizer guards held). Lazy vs unified changed nothing visible in the
  outcome: the run died of the same edit-churn the unified runs died of.
- **No whitespace floods** (0 flood events) — the template fix holds in
  instruct mode; **no git resets** this time.
- **Speed:** ~7 s/round average; the 5 longest stalls were 112–135 s, all
  `[llm:request]` prefill after a compaction (cold cache on the CPU-resident
  experts).

Verdict: **not better with lazy compaction.** The bottleneck is edit
mechanics + never running the full test suite, not context management.

## Run 2 — thinking (`THINKING=true`, temp 0.6), `docker_20260823_022349`

**6/6 — but at the timeout** (3460 s; bench reports rounds=619 summing all
session logs, the agent loop itself ran 193 rounds).

Timeline (UTC, container clock; CEST = +2h):
- 00:24 start. 00:32–00:34 three `git checkout -- src/context/mod.rs`
  self-reverts while fighting the `assemble()` edit (model-initiated, but
  targeted, not whole-tree).
- **00:48 (round 110): "The implementation is fully complete and verified"** —
  diff: `src/cli/mod.rs`, `src/config/mod.rs`, `src/context/mod.rs`,
  `src/main.rs`, 81 lines; `cargo test` green (7 `cargo test` invocations
  over the run), self-smoke produced `PONG_42`.
- 00:48–01:16: 83 more rounds. 18 separate "complete" messages, each paired
  with a tool call (`plan check step 25` → "No plan exists", `cargo build`,
  `echo "Implementation complete"`), so the loop never got a tool-free turn.
  One compaction at 00:50 ("context window exceeded").
- 01:16:16 last request; 01:20:19 `LLM request timed out after 30s` (idle
  timeout after 243 s of silence — prefill of a 60k context on CPU-resident
  experts), `[end] 193 rounds, status=error`. Bench timeout fired, checks ran
  on the tree: compile/build/help/parse/test/smoke all PASS. Smoke
  `[user] ping → PONG_42` took 75 s (cold prefill).

Tool mix: 107 shell, 23 plan, 23 file, 15 replace_range, 7 revert,
7 insert_at, 5 code, 2 refactor. Zero whitespace floods.

### Reading

- The thinking arm solved the task in ~24 min — first time nemotron has
  produced a passing tree at all. n=1 and the temperature moved with it
  (0.6 vs 0.2); needs a second run (not done tonight — the plan said
  "thinking only if instruct fails", which it did, once).
- The remaining 30 min were pure "done-but-can't-stop". Harness fix
  candidates: (a) when the assistant text contains a completion claim AND the
  attached tool call is a no-op (`echo`, `plan check` with no plan, repeated
  `cargo build` with no changes), run the done-gate instead of looping;
  (b) a round-budget after the first completion claim. Both are cheap and
  would have turned this into a 6/6 @ ~1500 s.
- Instruct lost to the same two things every nemotron run loses to: a
  signature change it could not propagate to the test crate, and a
  "wired to nothing" override.

## Run 3 — instruct, fixed script + harness fixes (`docker_20260823_190216`, 19:03)

The "cross it out" run: `THINKING=false`, lazy, 60k, temp 0.2, code-default
flags (script fix `2fdc48c`), plus the 08-23 fixes (period-3/4 loop cycles,
stale LSP suppression, truncated-call stub, debugger report sanitize).

- **5/6 at the timeout** (3426 s, 946 rounds / 364 requests). compile, build,
  help, parse, test PASS; smoke FAIL — `Pong! I'm here and ready to help`, the
  override is wired to nothing.
- The entire diff is 5 lines in `src/cli/mod.rs`: the `-s/--system-prompt-override`
  field on `Cli`. No call site, no assembly change.
- Timeline: edits 17:03–17:15 UTC (18 `file`, 11 `replace_range`, 5 `refactor`
  calls, 10 `add_param` attempts that never landed). From 17:15 to the timeout
  at 17:59 — 45 minutes — zero edits: 86× `cargo test` (26 passed each time),
  153× `echo` (71× `=== COMPLETE ===` with a hand-written summary), 42×
  `git diff --stat`, 56× `grep`, 28× `./target/debug/miniswe "…"` (which hang
  on the LLM and time out). 275 of 355 tool calls were `shell`.
- Harness view: 0 gate blocks, 0 judge fires, 0 auto-reverts, 0 compactions.
  `done` was never called, so nothing that checks the tree ever ran. `[loop]`
  fired 7× ("echo … recurred 4x" six times) and changed nothing.
- Same pathology as run 2's last 30 min ("done but can't stop"), but here the
  tree is hollow, so the "completion claim + no-op tool call ⇒ run the
  done-gate" rule would have produced a gate BLOCK (smoke wired to nothing) and
  a judge fire at ~26 min instead of a 5/6-at-timeout. That rule is still the
  one harness change nemotron points at; it is a cross-model gap (Glimmer's
  "Understood." turns are the other face of it).

### Verdict

Six instruct runs (4, 0, 5, 2, 4, 5), every one at the 57-min timeout, none
6/6; one thinking run 6/6 only because the bench scored the tree after the
agent failed to stop. Crossed out.
