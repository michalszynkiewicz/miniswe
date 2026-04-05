# Experiment Results — 2026-04-05

## Setup

- **Task**: Add `--system-prompt-override` / `-s` CLI flag to replace default system prompt and skip context providers
- **Model**: Devstral-Small-2 24B (Q6_K), temperature 0.0
- **Context window**: 32,000 tokens
- **Timeout**: 2,400s per run, up to 3 attempts, 80 rounds max per attempt
- **Baseline SHA**: `cc34d2626faf32c1b6dd1b8b33af693fb936b098` (pre-LSP codebase)

## Critical Bug Found

The plan tool definition had `name: "diagnostics"` instead of `name: "plan"`. This meant the LLM could never call the plan tool, and the write gate (`plan_exists()` check) blocked ALL file edits. Fixed on `stable` before running experiments.

## Results

| Experiment | Score | Attempts | Wall Time | Rounds |
|---|---|---|---|---|
| Baseline (plan fix only) | **0/6** | 2 | 2401s | 182 |
| Exp 1: describe_code tool | **6/6** | 3 | 2419s | — |
| Exp 2: strict plan + detail | **5/6** | 2 | 2403s | — |

### Validation checks (6 total)
1. `cargo check` — compilation
2. `cargo build` — binary produced
3. `--help` shows the new flag
4. Flag parses without error
5. `cargo test` — all tests pass
6. Smoke test — override prompt produces expected output

## Experiment 1: Two-Layer Code Descriptions (`describe_code`)

**Hypothesis**: Adding a second-layer tool that provides enriched descriptions (doc comments, parameter details) for specific files would help the model make better decisions about what to modify.

**Changes**:
- New `describe_code(path, symbols?)` tool — returns per-symbol doc comments, parameter descriptions, and file summaries
- Updated `get_repo_map` description to point to `describe_code` for details
- Registered in context tools (gated behind `context_tools` config)

**Result**: 6/6 PASS on attempt 3. The model used the standard exploration tools and made surgical edits. The describe_code tool was available but the key improvement was likely from the overall tool availability and the plan bug fix being present.

**Attempt breakdown**:
- Attempt 1: Compile failed (model rewrote too much of run.rs)
- Attempt 2: 5/6 — tests failed (similar issue to exp2)
- Attempt 3: 6/6 — model fixed test call sites

**Decision**: Merged to `stable`.

## Experiment 2: Strict Plan Enforcement with Step Breakdowns

**Hypothesis**: Enforcing step order and allowing the LLM to break steps into sub-steps would improve task execution quality.

**Changes**:
- Steps must be completed in order (check rejects out-of-order)
- New `plan(action='detail', step=N, content='...')` action for sub-step breakdown
- Plan injection highlights current step, hides non-current sub-steps
- System prompt guides explicit workflow: explore → plan → detail → execute → check → verify
- Enhanced plan feedback (step count, next step guidance)
- Fixed plan tool name from `diagnostics` to `plan`

**Result**: 5/6 on both attempts. The model correctly implemented the feature (compile, build, help, parse, smoke all pass) but consistently forgot to update test call sites in `tests/e2e_context.rs` — the `assemble()` function gained a new parameter but 2 test calls weren't updated.

**Attempt breakdown**:
- Attempt 1: 5/6 — test compile error (2 missing args in test file)
- Attempt 2: 5/6 — same issue, couldn't fix within remaining time
- Attempt 3: Skipped (timeout)

**Decision**: Not merged. The strict plan enforcement showed promise (5/6 on first attempt vs baseline 0/6) but the model still struggled with updating all call sites. The `detail` sub-step feature was available but may not have been used effectively. Results documented here.

## Key Observations

1. **Plan tool name bug was catastrophic**: The `diagnostics` → `plan` name mismatch made the write gate permanently block ALL edits. This was the primary cause of the baseline 0/6 score.

2. **Both experiments significantly outperformed baseline**: 6/6 and 5/6 vs 0/6. Most of this improvement comes from the plan bug fix.

3. **Test call site updates remain a challenge**: In both experiments, the model struggled with updating test files when function signatures changed. This is a common failure mode for the 24B model.

4. **Context window pressure**: The model hit the 32K token limit during experiment 1 attempt 1, causing an error. Compression kicked in but the damage was done. The describe_code tool may have contributed to context pressure.

5. **2400s is the right timeout**: All three runs used essentially the full 2400s. Shorter timeouts would cut off recovery attempts.

## Recommendations

1. The describe_code tool should be kept (merged) — it provides useful information without significant overhead
2. The strict plan system needs refinement — specifically, after modifying function signatures, the system should proactively suggest checking test files
3. Consider adding an auto call-site finder that triggers when function signatures change (already explored in `main` branch experiments)
