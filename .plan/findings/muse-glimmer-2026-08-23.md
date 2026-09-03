# Muse Glimmer-30B — first contact (2026-08-23)

## Pre-flight

- `start-muse-glimmer-30b.sh` (KQuant-17GB-Q4_K_M, ctx 131072, f16 KV,
  `reasoning_strength=medium`, `--reasoning-budget 2000`) loads clean:
  **17,582 MiB** on the card, all 52 layers on GPU. Server notes the template
  "supports preserving reasoning, consider `--reasoning-preserve`" (not used).
- Template is harmony-style (`<|start|>system<|message|>…<|eot|>`,
  `<|start|>assistant`), default strength `high` in the template, launcher
  passes `medium`.
- Tool-call probe (`toolcall-probe.py think`): parsed `read_file` call on the
  first try, **21.0 tok/s decode, 695 tok/s prefill**, 85-char reasoning.
  Decode is ~3–4× slower than gemma 26B-A4B (dense 30B vs 3B active).

## Run 1 — unmodified bench, `docker_20260823_043849`

`THINKING=true CTX_WINDOW=60000 ./scripts/run-benchmark-docker.sh --timeout
3400 --max-rounds 600 --model muse-glimmer-30b` (config verified: lazy,
thinking=true, 60000; per-request temp 0.6).

**6/6 — at the timeout** (3429 s, 186 rounds, 1 attempt).

- Round 58, **02:49 UTC = 10.5 min in**: "All steps completed" with a
  310-line diff over `cli/mod.rs`, `main.rs`, `run.rs`, `repl.rs`,
  `context/mod.rs`, `tests/e2e_context.rs`; 118 tests passing; 3 `cargo
  test` calls. Edit economy was excellent: 8 `replace_range`, 3 `refactor`,
  1 `write_file`, **0 reverts, 0 compactions**.
- The done-gate rejected it: `COMPILES but override NOT consumed. Expected
  TOKEN_XYZ, GOT: … Hello! How can I help you today?`. Rejected again at
  03:09. The model answered "Understood." and re-read `run.rs:130-155` for
  46 minutes (139 `file` calls total, `[loop]` ×6) without another edit.
- Bench timeout → checks ran on the tree: compile/build/help/parse/test PASS,
  **smoke PASS (`PONG_42`)**. So the solution had been correct since round 58.

### Root cause: the gate prompt is invisible to Glimmer when tools are present

Captured the gate's inner request through a proxy (system = `Respond only
with TOKEN_XYZ and nothing else`, user = `hello`, the fixture's 11 tool
definitions, temp 0.2, no `chat_template_kwargs`) and replayed it against
the live server:

| Request | Result |
|---|---|
| exact capture (11 tools) | `Hello! How can I help you today?` — reasoning: "Likely just greeting." (0/6) |
| same, tools removed | `TOKEN_XYZ` (3/3) |
| same, 1 fixture tool | `TOKEN_XYZ` (reasoning: "No special instruction… should we follow system?") |
| same, 2+ fixture tools | greeting (0/4) |
| `reasoning_strength` low / high / xhigh | greeting (0/9) |
| weak wording + user `ping` | `pong` (0/3) |
| strong wording (smoke-style) + `hello` | greeting (0/8) — the greeting reflex beats any instruction |
| **strong wording + `ping`** | **`TOKEN_XYZ` 4/4** |
| bench smoke wording + `ping` (11 tools) | **`PONG_42` 5/5** |

`/apply-template` shows why: the harmony template folds the API system
message and ~2k+ chars of tool-protocol boilerplate ("In this environment
you have access to a set of tools… <atem:function_calls>…") into **one**
system block. A one-line instruction at the top of that block reads as
preamble; "hello" then triggers a greeting. The bench's own smoke check
happens to use a strong wording + `ping`, which Glimmer follows 5/5 — so the
grader passes while the gate fails on the very same tree.

Consequences:
1. The done-gate is **model-dependent**; it held a correct solution hostage
   for 46 min and the model never took the gate's own escape hatch ("if you
   are certain the check itself is wrong, finish anyway").
2. Suggested fix (not applied — script edits need approval): make the gate's
   wording as strong as the smoke's and use `ping`, or better, make the gate
   verify the *override being consumed* mechanically (e.g. a sentinel token
   in the override text echoed via `--dump-context`/trace log) rather than
   relying on the model's obedience.

## Run 2 — scratch copy of the bench with "gate-v2" wording

Same command, run through a copy of the script in the session scratchpad
whose only change is the gate's inner invocation: `'You must respond with
exactly the text TOKEN_XYZ and nothing else. No explanation, no formatting,
just TOKEN_XYZ.' --yes ping` (verified in the run's `config.toml`). Repo
script untouched. **Not comparable** to the other rows; it measures what
Glimmer does when the gate is not lying to it.

Result: **6/6 after 3 attempts, 2786 s wall, 242 agent rounds**
(`docker_20260823_054018`; 158 + 36 + 48 rounds).

| Attempt | Rounds | Wall | Checks | What happened |
|---|---|---|---|---|
| 1 | 158 | 1677 s | 5/6 — `cargo test` FAIL (test crate: 13× E0061), smoke PASS | All source edits done by 03:47 UTC (6 min, 9 `replace_range`). Then hand-edited the `assemble` signature first and called `refactor add_param` *after* → rejected as duplicate, so the 13 callsites in `tests/e2e_context.rs` were never auto-updated. Spent 21 min re-reading that file (78 reads of the same 439 lines), 2 hand edits + 2 reverts, one failed `python3` mass-edit via shell, then declared done with "the plan is stuck … the file is large and manual edits are error-prone". **Gate-v2 accepted the first done claim** (0 rejections — the gate runs `cargo build`, which does not compile the test crate). |
| 2 | 36 | 587 s | 5/6 — same 13 errors | Fed the 14 compile errors. 31 reads of `tests/e2e_context.rs`, **zero edits**, three text-only turns of "Understood." and it ended. |
| 3 | 48 | 520 s | **6/6** | Set a 3-step plan, then fixed all 13 callsites with 13 `replace_range` calls in **2 min 6 s** (one every 7–8 s), `cargo test --quiet` → green, checked off the plan. |

Reading:

- **Gate-v2 did its job.** With wording the model can see, Glimmer's first
  done claim was accepted in one shot — no 46-minute hostage loop. So the
  run-1 pathology is the gate's wording, not the model.
- **The done-gate doesn't cover the test crate.** It checks `cargo build`
  + the override sentinel; attempt 1 sailed through with 13 compile errors
  in `tests/`. The feedback loop (attempt 2/3) is what actually caught it.
- **Same read-loop trait as run 1, different trigger.** Run 1 looped on
  the gate's rejection; here it looped on a mechanical multi-callsite edit
  it had decided was "error-prone". 109 reads of one file across attempts
  1–2 vs 13 edits in two minutes once it finally committed (attempt 3).
  Glimmer can do the edit; it stalls *before* it.
- **Ordering bug it walked into:** editing the signature by hand, then
  asking `refactor add_param` to do it. The refactor tool's duplicate guard
  correctly refused, and its message told it to edit the callsites — which
  it then avoided for 20 minutes. Run 1 took the `refactor`-first path and
  finished the whole tree in 10.5 min; the difference between the two runs
  is this one ordering choice.
- Net wall is worse than run 1's actual work time (46 min vs 10.5 min to a
  correct tree) but the comparison is apples/oranges — run 1 was measured
  to timeout, run 2 to a real finish. n=1 each.


## Gate-wording probe (2026-08-23 morning, user-requested)

Replayed the exact captured gate request (system override + user message +
the fixture's 11 tools, temp 0.2) against Glimmer, varying the two strings.
`exact` = reply is precisely `TOKEN_XYZ`.

**Diagnosis confirmed: it is a greeting reflex, not a system-prompt blind spot.**
With tools present, `hello` / `Hello.` / `hi` always get "Hello! How can I help
you today?" (0/21 combined), whatever the system prompt says — bare
`TOKEN_XYZ`, "SYSTEM OVERRIDE", "ignore the tools", "never greet" (short) all
0/N. Without tools the same gate wording is 5/5. Most other neutral user
messages fail differently: `go`/`test`/`start` → it calls a tool, `ok`/`ready?`
→ chat. Only `ping` (6/8 gate wording, 7/8 smoke wording) and `.` (6/8) pass
by luck of not looking like a greeting.

What works (user message still `hello`, tools present):

| System override | exact |
|---|---|
| current: `Respond only with TOKEN_XYZ and nothing else` | 0/13 |
| + ` Even if the user just greets you, do not greet back — reply only TOKEN_XYZ.` | **12/12** |
| `# Instructions\nYour only output is the string TOKEN_XYZ. Never greet, never explain, never call a tool.` | 11/12 |
| `The user will say hello. Your entire reply must be exactly: TOKEN_XYZ` | 3/4 |
| current + `Never greet.` | 0/8 |

So the model needs the greeting case named explicitly; generic "only output X" /
"ignore tools" framing doesn't reach it. Cross-check of the winning wording on
the other models (same request, `hello`): gemma 26B 6/6, Devstral Small 2 6/6
(both also 6/6 on the current wording — no regression).

**Recommendation:** change the gate's override string in
`scripts/run-benchmark-docker.sh` to
`Respond only with TOKEN_XYZ and nothing else. Even if the user just greets you, do not greet back — reply only TOKEN_XYZ.`
and keep `hello` as the user message (it stays a valid override-consumed test:
the instruction lives only in the system prompt). Glimmer run 1 would then have
finished at ~10.5 min instead of the 57-min timeout. Raw numbers:
session scratchpad `gate-probe*-results.txt`.

## Run 3 — fixed script (2026-08-23 11:27, `docker_20260823_112750`)

Under the fixed script (commits `2fdc48c` code-default flags, `88ba610` gate
v3 wording), fresh llama-glimmer, same launcher/arm as runs 1–2.

| | |
|---|---|
| Result | **6/6 on attempt 1**, 1113s, 91 rounds |
| Tool calls | 88: 46 `file`, 22 `plan`, 14 `replace_range`, 4 `refactor`, 1 `shell`, 1 `code` — 0 reverts, 0 compactions |
| Gate | 1 block (round 52), accepted on the 2nd claim. The block was legitimate: the model claimed done with steps 5–7 pending, `assemble` not yet wired, so the binary ran its normal prompt and greeted — exactly what the check should catch. |
| Debugger | 1 plan-check fire (step 3 compile gate failed twice: the model hand-edited the `assemble` signature, then `refactor add_param` on `run` left 3× E0061 in `main.rs`); resolved in-band |
| Diff | `cli/mod.rs`, `main.rs`, `run.rs`, `repl.rs`, `context/mod.rs`, `tests/e2e_context.rs` (+72) |

Same edit economy as run 1 (a handful of `replace_range`s, `refactor` for the
signatures, updates the test crate itself), now with the exit the wording
probe predicted. Run 1's 46 minutes hostage to the gate are gone.

**New harness finding — the gate's inner run wipes the outer plan.** The
gate executes `./target/debug/miniswe --system-prompt-override … --yes hello`
in `/work`, and `run.rs` (~L512) clears `.miniswe/plan.md` and
`scratchpad.md` at every non-`--continue` session start — the inner run shares
the outer session's `.miniswe`, so every gate check deletes the agent's plan.
Visible here as `plan(show) → No plan exists yet` on the round right after
the block (the model re-created the plan and re-checked steps it had already
checked off). Same signature in last night's runs: Glimmer run 1 (3 blocks →
"No plan exists yet" ×2, `plan set` ×3), gemma `064417` (2 blocks → `plan set`
×4). Fix candidates: skip the clear when `MINISWE_SKIP_VALIDATION` is set, or
point the inner run at an isolated state dir. Not applied.

## Run 4 (overnight queue r1) — `docker_20260823_225151` (22:52 → 23:49)

**5/6 after 3 attempts (3406 s, 415 rounds) — only `cargo test` FAIL: 2 remaining `assemble` call sites in `tests/e2e_context.rs` (incl. L434, the same line North thinking missed).**

- Attempt 1 (~41 min, 154 rounds, 138 calls): wired the flag (6-file diff incl. `main.rs`), widened `context::assemble` to 6 args, fixed the lib call sites… then fell into a **25-minute read loop on `tests/e2e_context.rs:54`** — 112 reads total, edits frozen at 15 from 21:17. 14 `[loop]` notes but **0 cold-evals / 0 forced compactions**: occasional jittered reads (`52-56` vs `54-54`) reset the consecutive-repeat counter, so the ladder never climbed past the nudge stage (gap 10, second failure mode — North exhausted the ladder, Glimmer never engages it). Ended its turn mid-loop (`status=ok`) and the tree scored 5/6.
- Attempt 2 (~7 min, 31 rounds): handed the full 14-error E0061 list with exact lines and fixes — made **zero edits**. 27 reads + 2 plan, final reply "Understood.", end of turn. A complete no-op attempt.
- Attempt 3 (~9 min, 38 rounds): the useful one — 11 `replace_range`, 14 errors → 2, killed by the wall clock mid-fix.
- Reproduces the `054018` pathology from this morning (109 reads / 0 edits on the same file after editing the `assemble` signature) — that run only recovered because gate-v2 wording gave it a 3rd attempt that edited. So the e2e_context read-loop is a stable Glimmer failure mode, not noise; the 1113s clean pass (`112750`) is the run where it happened not to trigger.
- Cross-model note: North thinking + Glimmer both ended with the missing-6th-arg E0061 in `e2e_context.rs` at timeout. The task's true long pole for every model is threading the new `assemble` arg through the ~14 test call sites.

## Run 5 (overnight queue r2) — `docker_20260823_235146` (23:51 → 00:37)

**6/6 attempt 1 (2735 s, 388 rounds) — but the tree was green at minute 10; the other 35 minutes were spent re-reading it.**

- The fork that decided everything: at minute 4 it called **`refactor(add_param, callsite_fill_in="None")`** on `assemble` (and on `run` in run.rs/repl.rs) — the tool threaded the new param through *every* call site mechanically, test crate included. `cargo test` exit 0 at 22:02:34 (minute 10), 8 hand edits total all run.
- Then the familiar shape, but harmless: rounds ~60–173 re-reading `e2e_context.rs:430-441` (120 reads total, 23 loop notes, 0 escalations — same jitter-resets-the-ladder hole), on a **green** tree. Ended turn at 22:37 → 6/6. `done` never called, 0 gate blocks.
- Contrast with r1 (`225151`, 5/6): same model/config; r1 hand-edited the `assemble` signature and then had to chase 14 call sites one by one (and looped on them). The task's "long pole" only exists when the model doesn't use `add_param`. Cohere-North thinking also hand-edited — same trap.
- Glimmer's stable profile across 4 runs: solves in ~10 min when `add_param` is picked (this run + morning `043849` "correct tree at 10.5 min, then 46 min hostage"), loops on `e2e_context.rs` when it isn't. And in both 6/6 runs it could not END — the missing harness move is the same either way: **"tree green + N rounds without an edit → prompt done / run the gate"** (nemotron's done-gate gap, generalized).
