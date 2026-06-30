# Plan: incorporating Aider techniques into miniswe

Goal: borrow Aider's data-backed techniques for reliable edits from **small local
models** (Gemma 4, Qwen3-Coder), *without* (for now) changing how edits are emitted.

**Explicitly OUT of scope for this plan** (deferred by decision, 2026-06-19):
- Switching edits from JSON tool-call args to plain-text edit blocks
  (Aider's whole/SEARCH-REPLACE/udiff response formats). This is the biggest
  rewrite and the most likely single win, but it's parked until later.

Everything below is independent of that change — it improves the *tool-calling*
agent we already have.

## Why these (the evidence)

- Our own week of benches: interventions targeting **diagnosis/context** (reactive
  debugger, gate_context_reset, spiral_reset) didn't move the bench; the one
  targeting **edit-execution mechanics** (auto-revert AST cascade) clearly did
  (ON 5.8 vs OFF 3.75). The bottleneck is mechanical edit execution.
- Session dissections: gemma builds the *correct* structure but botches the
  mechanical thread — passes `None` at the value-carrying callsite, or churns
  `refactor add_param` (add/drop/add) until test callsites explode to 8–12 args.
- Aider's measured findings reinforce: precise failed-edit re-prompts, bounded
  reflection, lean context, and lint-feedback loops are what make weak models
  reliable. (Aider keeps *context* lean but *format rules* strict — matching our
  `ceremony=strict` bench win.)

All items validate on the **seed-patch repro harness**
(`scripts/seeds/system-prompt-override-halfwired.patch` + `SEED_PATCH=…`), which
reliably forces the threading/refactor failure these target.

---

## Phase 1 — Tool-result feedback (highest leverage, lowest risk)

The model only learns from what our tool results say. Today they often say the
wrong thing (e.g. refactor's old `✓ COMPLETE — all callsites consistent`, which
told the agent it was done when the value wasn't threaded).

### 1a. Honest + instructive refactor results  *(started)*
- DONE (uncommitted): `add_param` success no longer claims "COMPLETE"; it names the
  placeholder and says value-carrying callsites still need editing.
  (`src/tools/refactor/add_param.rs`; A/B-gated by `MINISWE_ADDPARAM_LEGACY_MSG`.)
- NEXT: after `add_param` with a placeholder, **list the stubbed callsites** and,
  if feasible, flag which still pass the placeholder for the new param — turn the
  hidden threading work into a visible checklist. This is the deeper fix behind the
  wording tweak: the refactor tool currently *hides* the per-callsite decision the
  model most needs to make (Aider has no refactor engine precisely so the model
  must look at each site).

### 1b. Anti-churn / "already applied" feedback  (Aider's editblock re-prompts)
Port Aider's failed-edit message patterns to our edit tools
(`src/tools/refactor/*`, fast-mode `replace_range`/`insert_at`):
- "This SEARCH/range failed to match — here are the actual current lines."
- "The REPLACE lines are **already** in {path}" → catch no-ops instead of letting
  the model re-apply.
- On a multi-callsite op: "the other N callsites applied successfully — **don't
  re-send them**; only fix the ones that failed." This directly targets the
  add/drop/add churn that blew up test callsites in the seeded A/B.

**Effort:** S–M. **Risk:** low (messaging only). **Validate:** seeded A/B, track
churn (callsite arg-count drift) before/after.

---

## Phase 2 — Bounded reflection loop (rethink the done-gate grind)

Aider: a single capped budget (`max_reflections = 3`) shared across edit-fix,
lint-fix, and test-fix; then **stop**. Our behavioral done-gate instead *blocks
completion and keeps grinding in-context*, which our own data flagged as
net-harmful on hard misses (qwen ground 121 rounds → still failed; a fresh attempt
fixed it in 53).

### 2a. Introduce an explicit bounded reflection budget
- Unify the gate's retries + auto_check feedback under one capped counter that
  feeds the failure back **once per failure** and then stops, instead of grinding.
- A/B this against the current grind on the seed harness. If "feedback + cap +
  stop" matches or beats the grind, it's simpler and avoids the whole
  context-reset machinery we built to paper over the grind.

### 2b. Make `auto_check` (LSP) feedback as instructive as Aider's lint loop
- We already run LSP diagnostics after edits (parity with Aider's auto-lint).
  Ensure its failure text is specific and routed through the Phase-2a budget.

**Effort:** M. **Risk:** medium (touches the loop in `run.rs`). **Validate:**
seeded A/B (grind vs bounded), plus the existing from-scratch bench for regression.

---

## Phase 3 — Lean the context (carefully)

Aider keeps the working context lean: ~1k-token repo map + files the user
explicitly `/add`s; everything else is map-only. Our `assemble()` injects several
context providers by default.

- **Do NOT cut format rules** — `ceremony=strict` won the bench. This is purely
  about trimming bulk *content* volume.
- Audit provider token cost (profile/guide/lessons/project_notes/repo_map) and
  A/B trimming the heaviest against current, to free budget for the actual edit on
  small context windows.
- Confirm our repo map is signature-level and budgeted (~1k) — Aider parity.

**Effort:** S (audit) + M (A/B). **Risk:** low–medium. **Validate:** bench at
reduced context budget.

---

## Phase 4 — Measurement (supports Phases 1–3)

Turn the seed harness into a richer signal than `x/6`. Add per-run metrics to the
bench parse:
- did it thread the value (smoke) **and** keep callsites consistent (no arg drift)?
- # edits applied first-try vs # that needed a reflection
- # reflections used / did it churn (repeated add/drop on the same symbol)?

**Effort:** S. **Risk:** none. **Validate:** itself.

---

## Deferred / separate decisions (NOT in this plan)

- **Plain-text edit format** instead of JSON tool-call args — parked (the big one).
- **Context-match edits** (drop line numbers; address by surrounding context like
  Aider's udiff) — a different axis from the text-format change and doable inside
  JSON tools, but it's a substantial edit-primitive rework. Flag for a later,
  separate decision; the Phase-1 feedback may reduce its urgency.
- **Architect/editor split** — we already have it (smart-mode `edit_file` inner
  LLM + plan/code/fast roles); lean in only if Phases 1–2 plateau.

## Suggested order
1 (1a→1b) → 4 → 2 → 3. Phase 1 is cheap, on-mechanism, and Phase 4 makes the rest
measurable. Phase 2 is the higher-risk/higher-reward structural change; do it once
the feedback (Phase 1) and metrics (Phase 4) are in place.
