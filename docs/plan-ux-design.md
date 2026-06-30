# Plan UX Design — review, live progress, stop, revive

Status: design (not implemented). Counterpart to
[`plan-system-design.md`](plan-system-design.md), which covers the
*model-facing* plan tool (set / check / refine / show, write-gating,
per-step compression). This doc covers the *user-facing* surface: how a
human watches and steers a plan while the agent runs it.

## Motivation

Today the structured plan lives entirely inside the model's context. The
user only sees it if they go looking (e.g. a `/plan` command) and there is
no live signal that the agent is making — or failing to make — progress.
For small local models, runs are long (hundreds of rounds) and failure
modes are quiet (the brace cascade, premature "done", chasing its tail).
The user wants to:

1. **Review the plan** the model committed to — without asking for it.
2. **See progress** live as steps get checked off.
3. **Stop** a run that's going wrong, and decide what to do with the
   half-finished plan.
4. **Revive** a stopped/canceled plan later instead of starting cold.

## Principles

- **Zero-friction visibility.** The plan appears the moment the model
  calls `plan(action='set')` — no `/plan` needed. The persistent panel is
  the default view, not an opt-in.
- **The panel is a *projection*, never a second source of truth.** It
  renders the same plan state the model already maintains (see
  plan-system-design.md §Context Injection). Stopping/reviving manipulates
  that one state; we never fork plan state between "what the model thinks"
  and "what the UI shows".
- **Reviving is deterministic and explicit.** Restoring a stashed plan
  must not silently re-inject stale plan text into the model's context and
  hope it picks up. It re-seats the exact plan + completed-step markers and
  resumes at the first unchecked step.
- **One stash slot.** A single "previous plan" — restoring or starting a
  new plan overwrites it. No history stack, no GC story, no ambiguity about
  *which* previous plan `/resume` targets.

## 1. Live progress panel

A persistent panel (TUI region; in plain-CLI mode, a compact re-rendered
block) shown whenever a plan exists for the session.

```
┌ Plan ─────────────────────────────────────── round 47 · 3/6 ─┐
│ Task: Add --system-prompt-override flag                       │
│  ✓ 1  Add flag to Cli struct            (round 5)             │
│  ✓ 2  Wire through main.rs dispatch      (round 8)            │
│  ✓ 3  Update assemble() in context/mod   (round 15)           │
│  ▸ 4  Update run.rs signature + callsite  ← current           │
│    5  Update repl.rs signature + callsite                     │
│    6  Update test call sites                                  │
└──────────────────────────────────────────────────────────────┘
```

- **Source of state.** Driven by the same plan store the model reads/writes
  via the plan tool. Each `plan(action='set'|'check'|'refine')` emits an
  event the panel consumes; no separate bookkeeping.
- **Current-step marker (`▸`).** First unchecked step. Matches what the
  model sees as "Current step" in its context.
- **Header counters.** `round N` (stall signal — if it climbs while
  `k/total` doesn't, the run is stuck) and `done/total`.
- **No plan yet.** During the read-only exploration phase the panel shows
  `Plan: (exploring — no plan set yet)` so the user knows the model is
  legitimately pre-plan, not hung.

### Eventing

The plan tool already mutates a single in-session plan structure. Add a
lightweight observer channel: every mutation pushes a `PlanEvent`
(`Set{task, steps}`, `Checked{step, round}`, `Refined{steps}`,
`Stalled{rounds_since_progress}`). The panel renderer subscribes. This
keeps the agent loop ignorant of the UI (it just mutates plan state) and
lets headless/bench runs ignore the channel entirely.

## 2. Stop & decide

The user interrupts (Ctrl-C / a stop key). Instead of killing the process
outright, surface a decision prompt:

```
Stopped at step 4/6. What now?
  [r] resume now        — continue from step 4
  [s] stash & exit      — save plan+progress as the previous plan, quit
  [d] discard & exit    — drop the plan, quit
  [k] keep editing      — drop the plan, stay in session (manual mode)
```

- **Stash** writes the current plan + completed-step markers + the
  current-step pointer into the single stash slot, then exits.
- **Discard** clears plan state and exits — no stash.
- A second Ctrl-C at the prompt is a hard kill (no stash) — the usual
  escape hatch.
- **Crash / hard kill** never silently strands work: the stash slot is
  also written opportunistically on each `plan(action='check')` (cheap,
  it's the same serialized blob), so a killed run leaves a restorable
  stash at the last checkpoint even if the decision prompt never showed.

## 3. Revive a stashed plan

`/resume` (no args) restores the single stashed plan.

```
$ miniswe
Found a stashed plan from 2026-06-17 14:02 (3/6 done):
  "Add --system-prompt-override flag"
Resume it?  [y] resume   [n] start fresh (keeps the stash)   [x] discard stash
```

What "resume" does — **deterministically**, not by prose-injection:

1. Re-seat the plan structure (task + steps + which are checked + the
   completed-round annotations) into the live plan store.
2. Set the current-step pointer to the first unchecked step.
3. Inject **one** synthetic context block — the same `[PLAN — Step k of N]`
   block the model would normally see (plan-system-design.md §Context
   Injection) — so the model resumes mid-plan exactly as if it had just
   checked off step k-1. We do **not** dump the full historical transcript
   or the raw stashed markdown as a user turn; only the structured
   current-step view re-enters context.
4. The panel renders immediately from the re-seated state.

Because the completed steps are *marked done* (not replayed), the model
does not re-do steps 1–3. It must, however, re-establish any in-memory
facts it needs for step 4 by reading the relevant files — which is correct,
since "files which were analyzed might've changed" (the same reason we
don't trust per-round recall; see the compaction discussion). Resume gives
the model a clean **structural** restart point, not a fake memory of work
it can no longer verify.

### Why a single slot

A stack of stashed plans raises questions we don't want to answer for a
local-first tool: which one does `/resume` mean, when are old ones GC'd,
how are they disambiguated in the UI. One slot makes `/resume`
unambiguous and the mental model trivial: *there is at most one paused
plan; resuming or starting a new plan replaces it.* Starting a brand-new
plan while a stash exists prompts once
(`A stashed plan exists — overwrite it? [y/N]`).

## State & storage

- **Live plan state:** in-session, as today (the plan tool's structure).
- **Stash slot:** one serialized file under `.miniswe/` (e.g.
  `.miniswe/stashed-plan.json`) holding `{task, steps[], checked[],
  current_step, stashed_at, round}`. Session-scoped project dir, so it
  travels with the working copy. Overwritten on stash; deleted on
  discard/consume.
- **No transcript in the stash.** Only the structured plan + progress.
  Resume rebuilds context from the structured view, per §3.

## Out of scope (future)

- Multiple named stashes / a plan history browser.
- Editing the plan from the panel (today only the model mutates it via the
  plan tool; a user-edit path would need a write channel into plan state
  and conflict rules with the model's own `refine`).
- Cross-session stash sync / sharing.

## Open questions

- **Stall threshold for the header `Stalled` event** — reuse the existing
  stall detector's edit-progress window, or a plan-specific
  rounds-since-last-`check` counter? Leaning toward the latter: "rounds
  since the last checked step" is the metric the panel header already
  implies, and it's exactly the signal that catches "chasing its tail".
- **Plain-CLI rendering cadence** — re-print the panel on every plan event
  only, or also throttle-refresh the round counter? Event-only keeps logs
  readable; a live counter needs cursor control that fights piped output.
