# Structured Plan System Design

## Overview

The model works through tasks in a structured way: create a plan, break into steps, execute one step at a time, verify, checkpoint, compress, move on.

## Lifecycle

```
1. TASK RECEIVED
   ↓
2. EXPLORATION PHASE (read-only tools: search, read_file, get_repo_map)
   Model explores until it understands the scope
   ↓
3. PLAN CREATION (enforced — model must call plan(action='set') before any writes)
   Produces:
     ## Task: Add --system-prompt-override flag
     - [ ] Step 1: Add flag to Cli struct (cli/mod.rs)
     - [ ] Step 2: Wire through main.rs dispatch
     - [ ] Step 3: Update assemble() in context/mod.rs
     - [ ] Step 4: Update run.rs signature and call site
     - [ ] Step 5: Update repl.rs signature and call site
     - [ ] Step 6: Update test call sites (tests/e2e_context.rs)
   ↓
4. STEP EXECUTION (one step at a time)
   Model sees ONLY the current step in context:
     Current step: Step 3 — Update assemble() in context/mod.rs
     - [ ] Add system_prompt_override: Option<&str> parameter
     - [ ] Add if/else: use override when set, default otherwise
     Previous: Steps 1-2 completed ✓
   ↓
5. STEP COMPLETION
   Model calls plan(action='check', step=3)
   → Snapshot taken (revert point)
   → Step's messages compressed into outcome summary
   → Full plan shown briefly so model picks up next step
   → Context freed for the next step
   ↓
6. NEXT STEP (back to 4)
   ↓
7. ALL STEPS DONE → verify (diagnostics, tests)
```

## Context Injection (what the model sees each round)

### During a step:
```
[PLAN — Step 3 of 6: Update assemble() in context/mod.rs]
- [ ] Add system_prompt_override: Option<&str> parameter
- [ ] Add if/else: use override when set, default otherwise
[Completed: Step 1 (round 5), Step 2 (round 8)]
```

### After checking off a step:
```
[PLAN — Updated]
## Task: Add --system-prompt-override flag
- [x] (round 5) Step 1: Add flag to Cli struct ✓
- [x] (round 8) Step 2: Wire through main.rs ✓
- [x] (round 15) Step 3: Update assemble() ✓
- [ ] Step 4: Update run.rs ← NEXT
- [ ] Step 5: Update repl.rs
- [ ] Step 6: Update test call sites
```

### On plan(action='show'):
Full plan with all details.

## Plan Tool API

```
plan(action='set', content='...')     → Create the plan (enforced before writes)
plan(action='check', step=N)          → Mark step done, trigger snapshot+compression
plan(action='detail', step=N, content='...')  → Add sub-steps to a step
plan(action='show')                   → Show full plan
```

## Enforcement

Before allowing edit/write_file/replace_all/fix_file, check if a plan exists:
```rust
if config.tools.plan && !plan_exists(&config) {
    if is_write_tool(&tc.function.name) {
        return ToolResult::err(
            "Create a plan first: use plan(action='set') with your approach."
        );
    }
}
```

Read-only tools always allowed — the model needs them to plan.

## Compression on Step Completion

When `plan(action='check')` is called:

1. **Snapshot**: `snapshots.begin_round(round)` — revert point at step boundary
2. **Compress**: run `maybe_compress` immediately, targeting messages since the last step check. Summary is outcome-focused:
   ```
   Step 3: Updated assemble() — added system_prompt_override: Option<&str> as 6th param,
   added if/else in assemble() at line 296. context/mod.rs now 385 lines.
   ```
3. **Free context**: old step's raw messages are gone, replaced by the summary
4. **Show plan**: briefly inject the full plan so model sees what's next

## Revert Points

Each step completion creates a revert point:
```
Round 5:  Step 1 done → snapshot "step_1_done"
Round 8:  Step 2 done → snapshot "step_2_done"
Round 15: Step 3 done → snapshot "step_3_done"
```

If Step 4 goes wrong, the model calls:
```
revert(to_round=15)  → back to state after Step 3
```

The plan itself tracks which round each step completed on, so the model knows exactly where to revert to.

## Write Tool Gating

```rust
fn is_write_tool(name: &str) -> bool {
    matches!(name, "edit" | "write_file" | "replace_all" | "fix_file" | "shell")
}
```

Shell is gated too — `sed`, `mv`, etc. are writes.

Exception: `task_update` is always allowed (scratchpad is freeform notes, not gated).

## Implementation Steps

1. Modify `plan.rs`:
   - Add `plan_exists()` function
   - Add `action='detail'` for sub-steps
   - On `action='check'`: trigger snapshot + compression
   - `load_plan_context()`: return only current step + completed summary

2. Modify `run.rs`:
   - Before write tools: check plan exists
   - After step check: force compress + show full plan
   - Pass snapshot manager to plan tool

3. Modify `compressor.rs`:
   - Add `compress_step()`: compress messages since last step boundary
   - Step summaries are outcome-focused (what changed, not how)

4. Modify context injection in `mod.rs`:
   - `load_plan()` returns focused view (current step only)
   - After step check: briefly returns full plan

## Not in scope (future)

- Auto-detecting when a step is done (model must explicitly check it off)
- Parallel step execution
- Plan revision (model can set a new plan, replacing the old one)
- Sub-step tracking (steps have detail text but no separate checkboxes)
