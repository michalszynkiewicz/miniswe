# Success validation (the "done-gate") — design

*Date: 2026-06-14. Status: design / plan — no code yet.*

## 1. The problem (proven, not hypothesized)

On the canonical bench task (`--system-prompt-override`), models commonly stall at **5/6 with `smoke:FAIL`**: the change compiles, builds, and tests pass, but the feature doesn't actually work at runtime. The deciding failure is **value plumbed but never consumed**.

Concrete evidence — comparing two runs' final diffs:

- **6/6 run (`docker_20260614_002220`)** — the winning attempt *consumes* the override:
  ```rust
  let mut system_context = if let Some(override_prompt) = system_prompt_override {
      override_prompt.to_string()          // value is USED
  ...
  if system_prompt_override.is_none() { /* run providers only if no override */ }
  ```
- **5/6 run (`docker_20260614_223936`)** — the override is only *plumbed*: the CLI field + the `assemble(…, system_prompt_override: Option<&str>)` parameter exist, but there is **no `if let Some` consumption** anywhere. The value arrives at `assemble` and is dropped.

**The only difference between 6/6 and 5/6 is whether the model happened to write the consumption block.** It is a per-attempt coin flip; best-of-N occasionally catches a lucky consuming attempt. The 6/6s were luck, not a capability gain.

### Why the model exits believing it's done
Every signal it has is green:
- plan steps checked off (self-reported),
- `compile` / `build` / `test` pass,
- LSP clean.

Nothing in the agent's feedback reveals the override is ignored at runtime. So the model exits **justifiably** — it has no way to know.

### Where the gap is in code
`src/cli/commands/run.rs:516-545` — when the model returns no tool calls, the loop nudges **once** only if `has_unchecked_steps || !plan_exists`; otherwise it `break`s. So the case **"plan fully checked + build green + value not consumed"** hits an **unguarded `break`** with zero behavioral validation. The existing `PREMATURE_EXIT_NUDGE` validates against the **plan (self-reported)**, never against **runtime behavior**.

### Diagnosis
This is primarily a **feedback-masking** failure, not a capability ceiling: the model is *demonstrably capable* of the consumption logic (it wrote it in the 6/6 run) — it just isn't told when it didn't. That makes it the best-case target for a behavioral validation gate.

## 2. Goal & principle

**Before accepting "done," verify the change actually changed runtime behavior; if it didn't, block the exit and tell the model specifically what's wrong.** Convert the coin flip into reliability by turning the invisible runtime failure into an actionable signal.

Design tenets:
- **Behavioral, not structural** — "the feature works at runtime," not "tests pass."
- **Deterministic where possible** — the harness runs the check; don't rely on the model to test itself (it won't, reliably).
- **Cheap & in-band** — reuse the warm KV prefix; avoid extra LLM calls on the hot path.
- **Task-agnostic mechanism, config-driven specificity** — never hard-code the task into agent code (anti-overfit).

## 3. Design

### 3.1 Hook point
Wrap the unguarded `break` at `run.rs:545`. New order when the model stops emitting tool calls and the plan is complete:

1. If a **behavioral check** is available and hasn't passed this turn → run it.
2. **FAIL** → inject a concrete verdict as an in-band user message (warm prefix, one suffix-prefill), reset the exit flag, `continue`. Example verdict:
   > `[Verification failed: you set --system-prompt-override but runtime output is unchanged — the value reaches assemble() but is never used to replace system_context. Wire the consumption, then finish.]`
3. **PASS** (or no check derivable) → allow the existing exit.

This is the reflexion/block-the-exit mechanism; it reuses the premature-exit plumbing rather than adding a new control path.

### 3.2 Source of the check (tiered)
There is no universal behavioral check, so resolve it in priority order:

1. **User-stated** in the request ("…and `miniswe --foo` should print X") → use directly.
2. **Propose-then-confirm (interactive REPL only)** → the model proposes a concrete check ("I'll verify by running … expecting …"); user confirms/edits. One cheap round; the model does the thinking (which forces it to make the definition-of-done explicit).
3. **Auto-derived (headless / `-y` / bench)** → run the project's test suite, or a configured `behavioral_check` command, or a derived run (invoke the binary, diff observable output). **No user available**, so this path must exist or the gate is a no-op headless.

### 3.3 The red→green guard (non-negotiable)
"Write the check first" only helps if it actually discriminates. Require:
1. Author/identify the check.
2. Run on the **pre-change** code → **must FAIL** (red). If it passes at baseline, it isn't testing the new behavior → reject/revise.
3. Implement.
4. Run at "done" → **must PASS** (green), else block the exit (§3.1).

Without step 2, a vacuous check rubber-stamps a broken change.

### 3.4 `done_when` as a first-class plan field
Extend the `plan` tool so the definition-of-done is explicit, mirroring the existing per-step `compile: bool` precedent (`src/tools/plan/`, `Step`). Add a plan-level (or final-step) `done_when` carrying the behavioral check. The loop enforces it at the gate. This makes "how do we know it worked" part of the plan, not an afterthought.

### 3.5 Scope gate
Only fire for **behavioral change-requests**. Reuse the existing explore/coding router to skip `explain` / `read` / rename / doc-only / trivial edits — don't nag where there's no runtime behavior to verify.

### 3.6 Headless vs interactive
- **Headless / bench:** auto-derive (tests / configured cmd / binary-run-and-diff) + red→green guard, no prompts.
- **Interactive REPL:** propose-then-confirm UX on top; surfaces the proposed `done_when` for a quick confirm/edit.

## 4. What it explicitly catches
The plumbed-but-not-consumed case from §1: a behavioral check ("set the override, run, assert output changed") **fails** on the 5/6 diff (output unchanged) and **passes** on the 6/6 diff. The gate converts the unlucky attempt into "told it's not wired → writes the `if let Some` → passes."

## 5. Anti-overfitting
- Keep the gate's logic **task-agnostic** (run a check, compare, block-or-allow). Push all task-specificity into **config/derivation**, never into agent code.
- The general form is "feature works end-to-end (or project tests green)," not a bespoke `PONG_42` grep.
- **Validate on ≥2 distinct tasks** (e.g., the self-task + stork) — overfitting shows up as "helps task A, flat/negative on task B."

## 6. Measurement plan
The current metric hides everything: best-of-N pass/fail can't see the per-attempt success probability `p` (that's what made the lucky 6/6s look like wins).

- Track **per-attempt `smoke` pass rate at N≥5 single-attempt runs**, not best-of-3.
- **A/B: gate-off vs gate-on**, same model build, same reasoning state (see §7), N≥5. Success = higher per-attempt `p` (fewer plumb-only exits).
- Baseline to beat: the current coin-flip rate (unknown precisely; measure it first).

## 6.5 Model agency over the gate (escape hatches)

The gate exists *because the model is unreliable at self-assessing "done"* — so it must **not** get a "dismiss this gate because I'm sure I'm done" button (that hands the override to the exact bias we're guarding against). But it must not silently trap the model on a wrong-but-runnable check either. The balance:

- **Bounded, never infinite.** The gate blocks at most `validation.max_retries` times (default 3), then accepts the exit. The model is never stuck.
- **Broken checks never block.** A check that can't spawn or times out returns `Skipped` — it degrades to the prior behavior, never traps.
- **The model has an auditable voice (implemented).** Each time the gate blocks, the model's no-tool-call **completion rationale is captured and logged**, and the corrective message explicitly invites: *"fix it, OR — only if you're certain the check is wrong — finish and state why; it will be recorded."* Disputes **count toward the cap** (bounded) and are surfaced at exit. This is a recorded voice, **not a free pass**.
- **No deliberate single-shot bypass.** The model cannot wave off a passing-baseline check on attempt 1; it must either fix the change or exhaust the (small) retry budget.

What's still **deferred to phase 2** for "the check is genuinely wrong":
- **Red→green guard** — require the check to fail on the pre-change baseline, so a vacuous or mis-targeted check is rejected *before* it can ever block.
- **Model-refinable `done_when`** — when the check is one the *model authored* (propose-then-confirm), "I think the check is wrong" has a legitimate channel: `plan(action='refine')` to correct it. For human/bench-configured checks, the human owns correctness, not the model.

## 7. Relationship to other work
- **Reasoning (issue #39):** orthogonal to this gate — build the gate regardless. But pin reasoning to a known state (`--reasoning-budget 0` or on) before measuring, so the gate A/B has one variable. A reasoning model *given* "output unchanged" feedback may fix it better — possible synergy to test later.
- **Priority:** this gate is the data-justified #1 functional fix. Bench instrumentation (N≥5 + token capture) and a headroom task (stork) are prerequisites for measuring it.

## 8. Build order (phased)
1. **✅ DONE — config-driven gate + bounded retries + auditable model voice.** `[validation] command` (default empty = no-op → zero regression), run at the `run.rs` exit; non-zero blocks completion up to `max_retries`; the model's rationale on each block is logged; broken/timed-out checks `Skip` (never trap). Implemented in `src/cli/commands/agent/validation.rs` + the exit hook in `run.rs`; unit-tested; full suite green. **Not yet wired into the bench harness** (see below).
2. **Red→green guard** — require the check to fail on the pre-change baseline before it's allowed to block (rejects vacuous/mis-targeted checks).
3. **`done_when` plan field** — makes the definition-of-done explicit and gives the loop a concrete, model-refinable thing to enforce.
4. **Interactive propose-then-confirm** — REPL UX; validate on interactive traces (not bench pass-rate).

### Wiring the bench (to measure phase 1)
The gate only fires when `[validation] command` is set, so `run-benchmark-docker.sh generate_config` must emit one to exercise it. **Anti-overfit:** use a behavioral check that verifies the *capability* (override actually changes output) with a **different sentinel than the grader's `PONG_42`**, so the grader stays held-out — e.g. build, run with a distinct override string, assert the output contains it. A/B gate-off vs gate-on at N≥5 single-attempt (§6).

## 9. Risks & open questions
- **Model authors a bad check** — red→green catches vacuous checks, but not narrowly-right-but-incomplete ones; the win is bounded by the model's ability to *specify* behavior (weaker-correlated with implementing it, but related).
- **Not every task has a derivable behavioral check** — gate must degrade to a no-op gracefully (don't block forever).
- **Wall-clock cost** — baseline-run + done-run add rounds; on the wall-clock-bound bench prefer the cheapest deterministic check (project tests / single binary run), not a model-authored test it must debug.
- **Defining "done" for non-behavioral tasks** — scope gate (§3.5) must exclude them, or they stall.

## 10. References
- `src/cli/commands/agent/validation.rs` — the behavioral-check runner (`run_behavioral_check`, `CheckOutcome`).
- `src/cli/commands/run.rs` — the exit hook (the formerly-unguarded `break`): runs the gate, blocks up to `max_retries`, records the model's rationale.
- `src/config/mod.rs` — `ValidationConfig` (`[validation] command / timeout_secs / max_retries`; default command empty = disabled).
- `src/cli/commands/agent/hints.rs` — `PREMATURE_EXIT_NUDGE` (plan-based, not behavioral).
- `src/tools/plan/actions.rs`, `src/tools/plan/step.rs` — plan steps + `compile` flag precedent for `done_when`.
- `scripts/run-benchmark-docker.sh:388-401` — the reference behavioral check (set override, run, grep `PONG_42`).
- Evidence diffs: `benchmark_results/docker_20260614_002220_*/00_baseline/diff.patch` (consumed) vs `…_223936_*/00_baseline/diff.patch` (plumbed-only).
