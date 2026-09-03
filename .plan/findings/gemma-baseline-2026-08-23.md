# Gemma 4 26B-A4B instruct baseline — 2026-08-23 overnight

Command: `THINKING=false ./scripts/run-benchmark-docker.sh --timeout 3400 --max-rounds 600 --model gemma4`
Config (verified in each run's `config.toml`): `compaction = "lazy"`, `thinking = false`,
`context_window = 60000`, `max_output_tokens = 8000`, `stream_idle_timeout_secs = 30`, temp 0.2.
Launcher `start-gemma4.sh`: ctx 60000, `--reasoning-budget 2000`, KV **q8_0** (new default; llama-server
17,586 MiB) for runs 1–2, **q4_0** (17,218 MiB) for run 3.

| Run | KV | Result | Wall | Rounds | Attempts | Dir |
|---|---|---|---|---|---|---|
| 1 | q8_0 | **6/6** | 1090s | 330 (201 main) | 1 | `docker_20260822_235433` |
| 2 | q8_0 | **5/6** | 3403s (timeout) | 849 | 2 (5/6, then 2/6) | `docker_20260823_001405` |
| 3 | q4_0 | **6/6** | 677s | 206 (93 + 34) | 2 (5/6, then 6/6) | `docker_20260823_011216` |

References (07-14, `compaction_20260714_*/lazy/run1`): 6/6 in 299s (77 rounds, 1 attempt), 481s (54 rounds,
1 attempt), 1336s (3 attempts). So first-attempt success was 2/3 then, 1/3 tonight; final score 3/3 then, 2/3 tonight.

## Is it as successful as before? Mostly — but with caveats

1. **Per-round speed is unchanged** (~5 s/round in run 1, same as 07-14). Run 1 was slow only because it
   needed 201 rounds (65 `replace_range` + 17 `revert` + 6 plan rewrites + 3 "context window exceeded →
   compacted" events) versus 72–53 rounds for the references. Edit churn, not inference.
2. **KV q8_0 is not the cause.** The q4_0 run (3) failed attempt 1 the same way (5/6, `cargo test` FAIL with
   smoke PASS) and the q8_0 run 2 attempt 1 reached smoke PASS in 6 min — the fastest smoke of the night.
3. **The recurring 5/6 is a "tests" gap, not a feature gap.** In both run 2 and run 3 the feature worked
   (smoke PASS) but `cargo test` failed:
   - run 2: the model edited `tests/e2e_context.rs` to call `context::assemble(…, None, None, Some("override"))`
     (7 args) while `assemble` takes 6 (it wired the override elsewhere). It saw `cargo test` exit 101, then
     re-applied the *same* broken `replace_range` 11× (rev_2…rev_12, two reverts in between) and declared done.
     The done-gate only checks `cargo build` + the TOKEN_XYZ smoke, so the broken test suite passed the gate.
   - run 3: `basic_assembly_includes_system_prompt` regressed (`messages[0].role` became `user` — the system
     message was dropped when no override is given). Attempt 2 fixed it in 34 rounds / ~1 min.
   The bench's attempt-2 feedback ("TESTS FAILED TO COMPILE …") repairs this when the model can read it;
   a gate that also runs `cargo test` (or at least `cargo test --no-run`) would catch it inside attempt 1.
4. **Config confound vs 07-14.** The docker script defaults to `auto_revert_ast_cascade=false`,
   `reactive_debugger=false`, `gate_context_reset=true`; the 07-14 references ran the code defaults
   (`true`, `true`, `false`). All 08-20..23 docker runs share tonight's flags, so tonight's runs are comparable
   to each other and to the nemotron/gemma-thinking runs, but not cleanly to 07-14. Not fixed tonight (no
   script changes without approval); the knobs exist: `AUTO_REVERT=true REACTIVE_DEBUGGER=true GATE_CONTEXT_RESET=false`.

## Run 2 attempt 2: read-loop + a NEW harness hang (the real failure)

Attempt 2 started at 00:20 from attempt 1's tree. Timeline (UTC in the log = CEST −2h):
- Model tried `refactor add_param` with the schema's literal placeholder `new_param: Option<String>` as the
  parameter (contract-example leak — see memory "contract examples dominate"), applied it to `assemble` and
  16 callsites, then `drop_param`'d it, leaving a stale 3-arg call at `src/context/mod.rs:341`.
- 00:26–00:48: **read loop** — 248 of 318 tool calls were `file(read)` of the same three ranges
  (`mod.rs` L280-353, `compress.rs` L330-400), `cargo check` FAILED 5× on that one error; the loop detector
  printed "repeated 3x" every ~20 s but never escalated. (Known pathology: memory "read-loop warm-cache probe".)
- 00:48: `cargo check` OK at last. Plan refined, model moved on to `compress_history`.
- 00:52:46: `refactor add_param(compress_history, used_tokens: usize)`: the log shows
  `change_signature ask_rewrite … parsed OLD=90 NEW=114`, stderr shows
  `[lsp] request failed on a wedged rust-analyzer — restarting it (1/2)` and a fresh rust-analyzer spawned.
- 00:52:51 → 01:11 (bench kill): **no further log line, no LLM request** (llama-server idle the whole time),
  miniswe at 0% CPU with all 46 threads sleeping. Every LSP await I could find is bounded (`initialize` 30 s,
  requests 10 s, `wait_for_idle` ≤ 60 s, `ensure_ready` 60 s), so the wedge is somewhere in the
  restart → retry path inside `change_signature`. Could not attach a debugger (container process runs as root).
- Attempt 2 scored 2/6 (tree mid-refactor), attempt 3 had −2 s left → final 5/6 (best attempt).

This is the second harness-hang class after the LLM-request hang fixed in `1a7eed9`. Suggested follow-ups:
a tool-level deadline around `refactor` (like the 600 s LLM request deadline), and a reproduction via
`tests/` with a rust-analyzer that stops answering mid-session.

## Flag-matched re-run (code defaults) — the confound resolved

The three baseline runs above used the docker script's flag defaults
(`auto_revert_ast_cascade=false`, `reactive_debugger=false`,
`gate_context_reset=true`), which are NOT the code defaults the 07-14
references ran with. After the planned runs finished I re-ran gemma instruct
three times with the code defaults
(`AUTO_REVERT=true REACTIVE_DEBUGGER=true GATE_CONTEXT_RESET=false`,
everything else identical: THINKING=false, lazy compaction, 60k, q8_0 KV,
fresh server each run).

| Run | Dir | PASS/6 | Wall | Rounds | Gate blocks | Debugger fires | Notes |
|---|---|---|---|---|---|---|---|
| 1 | `docker_20260823_064417` | **6** (attempt 1) | 550s | 111 | 2 | 2 (1 judge REWIND) | 40 `replace_range`, 10 reverts, 2 loop fires, 1 overflow compaction; the gate caught the unconsumed override, the debugger REWIND repaired `repl.rs`, third claim passed |
| 2 | `docker_20260823_065509` | **6** (attempt 1) | 279s | 57 | 0 | 0 | 17 `replace_range`, 3 reverts, 0 loops, 0 compactions — cleanest run of the night |
| 3 | `docker_20260823_070105` | **6** (attempt 2) | 1411s | 165+46 | 3 | 3 (2 judge REWINDs, both `repl.rs` → rev_2) | attempt 1: 57 `replace_range`, 20 reverts, 3 loops, 3 overflow compactions, ended with a **non-compiling tree** (see below); attempt 2: 46 rounds, fixed the 2 remaining E0061s, 6/6 |

Flag-matched triple: **6, 6, 6 @ 550 / 279 / 1411 s, first-attempt 2/3** —
the same shape as the 07-14 references (6, 6, 6 @ 299 / 481 / 1336 s,
first-attempt 2/3). The docker-flag triple was 6, 5, 6 @ 1090 / 3403 / 677 s,
first-attempt 1/3. So:

- **"Is gemma as successful as before?" — yes, once the flags match.** The
  apparent regression in the first triple is the flag confound (plus run 2's
  LSP hang). n=3 each, so treat the wall-time gap as suggestive, not proven.
- Which flag carries it: `auto_revert_ast_cascade` never fired in any of the
  three runs (0 `[auto-revert]` messages). The reactive debugger fired in 2/3
  runs and its judge chose REWIND 3 times, each restoring `repl.rs` to a
  parsing state — that is the mechanism that turned run 1's gate blocks into
  a first-attempt pass. `gate_context_reset` (on in the docker defaults) was
  never reached in the first triple either, so the difference is most likely
  "debugger on" alone.
- **New harness finding — the done-gate has a finite budget, then waves
  everything through.** `validation.max_retries = 3` (bench config): the
  gate only runs while `validation_blocks < max_retries`. Run 3 attempt 1
  was blocked 3 times, then its 4th done claim at round 165 — issued right
  after the model itself reverted `repl.rs` to rev_2, leaving 2 E0061s in
  `main.rs` — **exited with `✓ Done` and no check at all** (`build:SKIP`,
  `help:FAIL` from the bench). The harness printed "Completed after 2
  blocked verification(s); model's reasons recorded in the log" — the
  "reasons" are just the model's ordinary done-text, it never claimed the
  check was wrong. In the bench, attempt 2 rescues this; in real use the
  user gets a broken tree labelled done. Suggestion (not applied): after the
  budget is spent, run the check one last time and report "finished with
  check FAILING: …" instead of `✓ Done`, or make the final exit require a
  compiling tree.
