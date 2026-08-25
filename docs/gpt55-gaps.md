# GPT-5.5 Project Gaps

Date: 2026-07-06

This is a quick pass over the repo, with emphasis on context assembly,
compaction, plan state, and scratchpad state.

## Highest-value gaps

### 1. Plan is injected twice

`src/context/providers.rs` has `PlanProvider`, which contributes `[PLAN]` when
`context.providers.plan` is enabled. `src/context/mod.rs::assemble` then calls
`tools::plan::load_plan()` and appends another `[PLAN]` block unconditionally
when a plan exists.

Impact:

- Every request with a plan spends tokens on duplicate state.
- If one path is later changed to a focused/current-step view and the other is
  not, the model can receive two conflicting plan views.
- This makes any plan-placement experiment noisy because it is not moving one
  source of truth.

Fix:

- Keep exactly one plan injection path.
- Prefer making plan state a named provider, then remove the manual append in
  `assemble()`.
- Add a context assembly test that counts `[PLAN]` exactly once when a plan
  exists.

### 2. The injected plan/scratchpad can be stale inside a run

The docs say `assemble()` rebuilds context every tool round, but `run.rs`
assembles once before the loop and then mutates `messages`. It only rebuilds
`messages[0]` when `plan_exists` flips, so `plan(action='set')` refreshes the
system prompt, but `plan(action='check')`, `plan(action='refine')`, and
`plan(action='scratchpad')` do not refresh the injected provider content.

Impact:

- After a plan check, the system prompt can still show the step as unchecked.
- After a refine, the system prompt can show the old step list.
- After a scratchpad update, the injected scratchpad can remain stale until a
  fresh assembly path runs.
- The tool result gives some immediate feedback, but it is not a replacement for
  the current persisted state.

Fix:

- Treat plan and scratchpad as dynamic current state and refresh them before
  each LLM request, or at least after every plan/scratchpad mutating action.
- Add tests for `set`, `check`, `refine`, and `scratchpad` that assert the next
  request sees the updated state.

### 3. Compaction budget does not account for the system/current-state block

`src/context/compressor.rs::budgets` subtracts tool definitions and output
headroom, then gives raw history `available / 3`. It does not subtract the
system prompt, which can include profile, guide, project notes, lessons, MCP,
scratchpad, usage guide, and duplicated plan.

Impact:

- The compressor can declare history under budget while the actual request is
  over budget.
- Moving plan/scratchpad near the end improves recency, but does not fix total
  request budgeting unless the new current-state block is counted.
- Larger guide/scratchpad/project-note files can silently steal the work zone.

Fix:

- Make the budget calculation operate on full request cost:
  `system/current state + tool defs + history + output headroom`.
- Cap or summarize high-variance providers, especially scratchpad, guide, and
  project notes.
- Emit metrics for total request tokens, not only non-system history tokens.

### 4. Single-shot and REPL compaction paths diverge

`run.rs` uses `context::compressor::maybe_compress()` inside the tool loop.
`repl.rs` still calls its own `mask_old_tool_results()` inside the loop and
only runs `maybe_compress()` after the turn.

Impact:

- Interactive behavior is not testing the same compaction strategy as headless
  runs.
- Docs say unified compression replaced the old tool masking path, but the REPL
  still has a live local implementation.
- Compaction regressions can reproduce in one mode and not the other.

Fix:

- Move REPL onto the same `maybe_compress()` path as `run.rs`, or explicitly
  document the divergence and test both modes.
- Delete the local REPL masking implementation once unified compression owns the
  behavior.

### 5. Plan-system design doc is ahead of implementation

`docs/plan-system-design.md` describes a focused current-step plan view,
compression on every `plan(action='check')`, and step-boundary snapshots.
Current code stores full markdown in `.miniswe/plan.md`, injects the full plan,
and does not run step-scoped compression from the plan tool.

Impact:

- The design doc is useful, but it reads like implemented behavior.
- Full-plan injection gets progressively more expensive and less current-step
  focused.
- Step completion does not create the clean "summarize completed step, move on"
  context boundary that the design relies on.

Fix:

- Either mark the design as aspirational, or implement the focused plan view and
  step-check compression path.
- Add plan-view tests: current step, completed summary, and full plan only on
  `plan(action='show')`.

### 6. Scratchpad is persistent but unbounded

`task_update` validates that scratchpad content contains `## Current Task` and
`## Plan`, but there is no size budget, summarization, or structural merge.
Every update rewrites the whole file and provider injection includes it in full.

Impact:

- A model can accidentally turn scratchpad into another large transcript.
- Because scratchpad is compaction-surviving, bloat here directly reduces every
  future request's work zone.
- Moving it to a current-state tail block makes this more visible, but also
  makes bloat more recency-dominant.

Fix:

- Add a scratchpad budget.
- Encourage or enforce sections like `Current task`, `Decisions`, `Files`,
  `Still need`, and trim older detail.
- Consider a tool mode that patches named sections instead of replacing the full
  file.

### 7. Assembly token estimates undercount history

The compressor has `msg_token_cost()` that counts tool-call argument bytes.
`assemble()` still estimates conversation history from message content only.

Impact:

- Context logs can under-report real request cost when assistant tool-call args
  are large.
- This makes overflow diagnosis harder, especially for edit calls with large
  `replace_range` payloads.

Fix:

- Reuse one message-cost function for assembly logging and compression.
- Include current-state/system tokens separately in logs.

## Compaction strategy: should plan and scratchpad move out of system?

Mostly yes, but the useful version is slightly different from "place them at
the end from when they were updated."

Plan and scratchpad are mutable task state, not static behavioral instructions.
Keeping them in the system prompt makes them compaction-proof, but it also puts
them at the oldest/front-most part of the request and hides stale-state bugs.
For small local models, the current next step and scratchpad notes are exactly
the kind of information that benefits from recency.

Recommended shape:

1. Keep the system prompt for stable rules only: role, tool contract, safety,
   path rules, and editing policy.
2. Build one regenerated `[CURRENT STATE]` block for every request from disk:
   current plan view, scratchpad summary, maybe active validation status.
3. Place that block as late as role hygiene allows, ideally immediately before
   the user task at user-turn boundaries.
4. Do not rely on historical "plan was updated here" messages as the canonical
   state. Historical messages can be compressed, masked, contradicted by later
   updates, or simply become too far back.
5. Count the current-state block in the real request budget.

There is a role-order caveat. During a tool loop, the request often ends in a
`tool` message and the next valid model response should be `assistant`. Appending
a synthetic `user` message after every tool result may force bridge messages or
confuse strict chat templates. So the implementation should be deliberate:

- At user-turn boundaries, fold `[CURRENT STATE]` into the final user message or
  inject it immediately before that user message.
- Immediately after `plan` or `scratchpad` tools mutate state, include a compact
  current-state view in that tool's result, so the update is naturally recent.
- For every LLM request, refresh `messages[0]` or another valid injected state
  carrier so the state cannot go stale. If the only universally safe carrier is
  `messages[0]`, still remove duplication and budget it; recency can then be
  improved where role order permits.

Bottom line: moving plan/scratchpad out of the system prompt is a good direction
if it becomes a regenerated current-state block with budget accounting. Moving
them only to a historical "where they were updated" location is not enough and
can be worse than the current system injection because the canonical state would
again be subject to drift and compaction.

## Suggested near-term order

1. Remove duplicate plan injection.
2. Add tests proving plan/scratchpad updates are visible in the next request.
3. Introduce a single `CurrentState` assembler with token accounting.
4. Use that assembler from both `run.rs` and `repl.rs`.
5. Unify REPL and headless compaction paths.
