# Experiment Results — 2026-05-08 to 2026-05-11

## Task

Add a CLI flag `--system-prompt-override` (short: `-s`) that takes a string and
replaces the default system prompt with the provided text. When set, skip all
context providers and just use the override text as the system message. Must
work in both single-shot and interactive modes.

Multi-file change: `src/cli/mod.rs` (add field), `src/cli/commands/run.rs` +
`src/cli/commands/repl.rs` (thread param), `src/context/mod.rs` (assemble takes
override and short-circuits). 6 bench checks: `compile`, `build`, `help`,
`parse`, `test`, `smoke` (end-to-end runtime check).

## Bench config

- Baseline SHA: `cc34d2626faf32c1b6dd1b8b33af693fb936b098`
- `context_window = 60000`, `max_output_tokens = 4096` (restored from 16384
  after data showed the smaller input window starved the model)
- `repo_map_budget = 5000`, `snippet_budget = 12000`
- `history_turns = 5`, `history_budget = 6000`
- 2400s per attempt, up to 3 attempts with `--continue` retry
- Per-model context tightening for Devstral (20K) **caused 0/6 regression** —
  reverted in commit `38e9a2d`.

## Results — non-Chinese models

| Model | Score | Wall | Attempts | Notes |
|---|---|---|---|---|
| **Devstral Small 2 24B (UD-Q4)** | **6/6** | 1344s | 1 | `change_signature` hidden via probe-only Devstral detection. 0 leaks. Self-planned at round 12. |
| **Gemma 4 26B-A4B (UD-Q4_K_M)** | **6/6** | 2407s | 2 | Slower convergence. Self-planned at round 6. Never used `change_signature` even when available — preference for `replace_range`. |
| **GPT-OSS 20B (F16)** | 5/6 | 2400s+ | 1 (rest skipped) | `smoke:FAIL`. Required *both* nudges (round 12 + round 20) to plan. Wired CLI flag correctly but didn't thread override into `assemble()`. |
| **Mistral Small 4 119B MoE** | (running) | — | — | First test of the Devstral successor. CPU-offload experts; ~15-30 tok/s expected. |

## Failure modes catalogued

### 1. `[TOOL_CALLS]` parser leak (Devstral, Mistral-template-specific)
- Symptom: chat-template control tokens leak into tool-call argument JSON
- Cause: KV-cache state contamination across requests — **not** bytes-deterministic
- Reproduction: verbatim replay of a leaking request showed 0/10 leaks; fresh
  state always produces clean output
- Fix: `cache_prompt: false` one-shot retry on detection (`has_tool_call_leak`
  in `src/llm/mod.rs`). 5-line landing; defensible across versions.

### 2. `change_signature` schema confusion (Devstral)
- Symptom: model packs the entire signature + extra keys into `position` field
- Cause: bytes-deterministic — verbatim replay shows 6/8 attempts mangle the
  same way at the same context size
- Fix: hide `change_signature` from Devstral. Probe-only detection so non-
  Devstral models (Gemma, GPT-OSS) still see it.

### 3. Schema-runtime mismatch on edit tools (GPT-OSS especially)
- Symptom: model calls `edit_file` → runtime returns "Create a plan first" →
  model doesn't update its mental model → retries the same call → repeat
- Worst observed: GPT-OSS attempt produced **0 edits** in a full 2400s window
- Fix: `visible_tool_defs` filters write tools from the tool list pre-plan, so
  the model never sees what it can't use. System prompt's WORKFLOW line
  explicitly tells the model `plan(action='set')` unlocks them.
- Secondary fix: graduated nudges at round 12 (gentle) and round 20+ (urgent),
  both plan-aware in their content.

### 4. Context starvation (Devstral, regression)
- Symptom: per-model context tightening to 20K for Devstral (to dodge
  `[TOOL_CALLS]` leaks above ~78KB) caused 0/6 regressions
- Root cause: effective input budget shrunk to ~3.6K tokens, model couldn't
  hold multi-file state in history
- Fix: removed Devstral tightening branch. All models use 60K input window.
- Lesson: aggressively patching a rare failure (leaks) can cause a worse
  systematic failure (starvation).

### 5. Per-attempt dump clobbering
- Symptom: bench retry attempts shared `/output/llm_dumps` mount; each
  container's process started its counter at `req-000000` and overwrote prior
  attempts' data. Most-diagnostic data was always lost.
- Fix: dump filenames now use `req-{secs}-{pid}-{n:06}.json`. Cross-attempt
  preservation; chronological sort.

### 6. Plan-tool thrashing
- Symptom: Gemma's earlier runs called `plan(action='set')` 4-5 times per
  session, resetting the plan instead of refining
- Improvement: after adding the WORKFLOW preview that previews unlocked tools,
  Gemma's pattern shifted to balanced set/refine/check (1 set + 3 refine + 6
  check on the 6/6 run vs 4-5 sets and 0 refines before).

## Infrastructure changes that helped (commits on `skills-fixes`)

- `38e9a2d` — bench: model tagging, LLM dumps, **drop Devstral context tightening**
- `3dac143` — bench: revert `max_output_tokens` to 4096
- `2f2d1de` — agent: plan-first tool gating + probe-based Devstral detection
  - `visible_tool_defs` helper (hide edit tools pre-plan)
  - `retry-on-leak` (KV-cache-state-aware retry)
  - Probe-only `is_devstral_family` (no false positives on bench config alias)
  - Dump session prefix (no overwrite across attempts)
  - `change_signature` moved to top of tool list; `rename` moved last
  - System prompt WORKFLOW preview (adapted per model — omits
    `change_signature` for Devstral)
  - Plan tool description: `set (UNLOCKS edit tools …)`

Uncommitted at end of session: round-12 / round-20 plan-aware nudges (in
`src/cli/commands/run.rs`). Validated by GPT-OSS run going from 0 edits → 5/6.

## Open questions / non-findings

- **Why doesn't Gemma use `change_signature`?** Probed Gemma directly with a
  "plan a change for another agent given these tools" prompt — Gemma plans
  `edit_file` for every step, including ones that are textbook
  `change_signature` use cases. **Genuine preference, not inertia.** Tool
  position #1, descriptions, system prompt hints — none flipped Gemma's
  preference. Likely a training-data bias toward edit-by-patch over
  refactor-by-symbol-tool.
- **5/6 ceiling on `smoke:FAIL`** — Devstral, GPT-OSS, and some Gemma runs all
  trip on the same end-to-end runtime check: feature wired through CLI but
  override value not actually replacing the prompt at runtime. Multi-step
  value-threading. Could be model capability, or could be the agent's tool
  feedback masking the real issue. Open.
- **`change_signature` for the Devstral family** — Mistral Small 4 absorbs
  Devstral and is in-flight as of writing. If the new template/post-training
  doesn't have the `[TOOL_CALLS]` leak or schema-confusion behaviors, we can
  drop the Devstral-only hides and re-test.

## Cross-model patterns

- **All non-Chinese open coders in the 20-30B tier hit the same 5-6/6 ceiling**
  on this multi-file task. The ceiling is real and reflects the model's
  ability to thread a new value through several call sites correctly while
  keeping all existing tests passing.
- **Plan-first gating + graduated nudges** is the difference between
  "GPT-OSS produces 0 edits" and "GPT-OSS produces 5/6". Cheap, works across
  models. Worth keeping.
- **Probe-based model family detection** (vs config-string OR-fallback) is
  essential for clean cross-model comparison — the bench config hard-codes
  `model = "devstral-small-2"` regardless of which server is running, so
  config-string substring matches false-positive Devstral on every other
  model.

## Hardware

- RTX 3090 (24 GB VRAM, ~21 GB usable)
- 128 GB RAM
- AMD Ryzen 9 9950X3D (16C/32T)
- Server: `llama-server` from llama.cpp
