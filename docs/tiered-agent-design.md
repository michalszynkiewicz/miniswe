# miniswe agent design — evidence-distilled (local models)

Status: design, no code yet. Supersedes all earlier drafts of this file.
Date: 2026-05-18. Basis: a multi-arm probe (`scripts/simple-tier-probe.py`,
`scripts/simple-tier-probe-rich.py`) on a multi-file value-threading task
with a real `cargo check` + behavioral oracle, run across
gemma-4-26B-A4B, devstral-small-2 (Mistral-family), qwen3-coder-next-80B.
All numbers below are from the **error-audited, sanitized** harness
(`sanitize_messages` ported in — see §Corrections).

## ⚠ REAL-BENCH VERDICT (2026-05-18, supersedes everything below)

The probe-driven "de-ceremonialize / lean is better" thesis in this
doc was **refuted by the real docker bench.** Decisive A/B on the
Qwen3-Coder-Next control (reliably 6/6 historically), same HEAD, same
harness, only `tools.ceremony` varied:

| ceremony | real-bench result |
|---|---|
| **`strict`** (plan-gate enforcement + per-step compile-gated check-off) | **6/6, smoke:PASS** ✅ |
| `off` (lean: no plan machinery) | 5/6, **smoke:FAIL** |
| `advise` (lean + advice to decompose, plan optional) | 4/6, test+smoke FAIL |

`smoke` = the feature actually works at runtime — the only check that
matters. Only **enforced** plan-first ceremony delivers it. Advice is
not enough; LSP/compile feedback (present in *all* modes) is not
enough; aider was *not* a reliable counterexample (0–6/6 on this task,
mostly 3/6, its best unscoped run still smoke:FAIL). The synthetic
probe could not measure real multi-step value-threading; the bench
can, and it says the enforcement is load-bearing.

**Shipped decision: default `tools.ceremony = "strict"`** (committed —
`95e62da`). `off`/`advise` remain opt-in flags, documented here as
real-bench-refuted, not recommended. The de-ceremonialization work
(commits `e54588b`, `a6a9790`) is preserved behind the flag, *not*
the default. Everything in the TL;DR below is the original
probe-driven reasoning, retained for the record but **wrong as a
production direction.**

## TL;DR / decision (PROBE-DRIVEN — refuted above, kept for record)

The highest-leverage change is **not** building an aider-style text
tier. It is **de-ceremonializing the existing tool-call agent**:

1. **Keep a real OpenAI tool-call channel as the primary path.** It is
   the universal winner for capable models and the *only* viable path
   for Mistral-family models.
2. **Delete the ceremony that degrades it**: the plan-gate, the
   enumerable `PLAN CHECK <step>` verb, instructed-"STOP" scaffolding,
   and the grouped `refactor` action/`position`/`callsite_fill_in` DSL.
3. **Flatten the tool surface** to flat single-purpose tools.
4. **Enforce strict message hygiene** (`sanitize_messages`) — it is
   load-bearing, not a nicety (see §Corrections).
5. **One minimal prompt for all models.** A per-family structured
   prompt was hypothesized and then **tested in clean isolation and
   refuted** (Devstral, sanitized, no-gate, tool-calls: structured H
   0/6 = minimal G 0/6). Prompt shape has no effect; do not add
   per-family prompt complexity.

A no-tool request/response "simple tier" (arm A) is viable and *fast*
for capable models but is **not** pursued as the default: it only
matches (does not beat) a clean tool-call path on capable models, and
it *fails* the model that struggles most (Devstral 1/6). Fix the
tool-call path; do not build a parallel text agent.

## Clean evidence

Reframed/no-gate prompt, sanitized harness, N=6, 15 rounds.
**F** = tool-shaped surface delivered as free-text (verb grammar).
**G** = same surface via real OpenAI tool-calls. Arm **A** (no tool
abstraction at all — pure `READ:`→wait / SEARCH-REPLACE) and **E**
(flat tools, *structured/gated* prompt) shown for reference.

| | Gemma | Devstral (Mistral) | Qwen-80B |
|---|---|---|---|
| **A** no-tool text | 6/6 | 1/6 | 6/6 |
| **F** tool-shaped *free-text* | **0/6** | **0/6** | **0/6** |
| **G** tool-calls (loose prompt) | **6/6** | 2/6 | **6/6** |
| **E** tool-calls (structured/gated prompt) | (budget-bound) | **4/6** | 5/6 |

All cells above are error-audited clean (ERR=0). Earlier contaminated
cells (Devstral C, Devstral F-unsanitized) are discarded, not shown.

### Confirmed mechanisms

1. **Tool-shaped surface as free-text fails universally.** F = 0/6 on
   *every* model, clean, even after the prompt fix and message
   sanitization. Tool-trained models cannot drive a tool-shaped
   surface without a real tool-call channel: they leak chat-template
   control tokens, fake `<|tool_call>` JSON, spray multiple
   actions/turn, or never converge. The one-action stop boundary must
   be **structural** (the tool-call closes the turn), never instructed
   — Gemma provably ignores "emit one action then STOP" in text.
2. **Real tool-calls are the robust winner for capable models** —
   Gemma 6/6, Qwen 6/6, clean and fast.
3. **The plan-gate + enumerable `PLAN CHECK <step>` + instructed stop
   is a degeneration generator.** In text mode it produced an
   unbounded `PLAN CHECK 1, 2, … 273` repetition to the token cap
   (~70s/round of pure garbage). Removing the gate and the enumerable
   verb (soft, post-completion check only) eliminates it.
4. **`sanitize_messages` is mandatory and load-bearing.** Devstral's
   strict Mistral jinja template hard-`raise_exception`s on consecutive
   user messages (confirmed from the server's own 500 body: *"roles
   must alternate user and assistant…"*). Merging consecutive same-role
   messages fixes it. It is **not only** a crash fix: it also moved
   Gemma G from 4/6 → 6/6 (cleaner conversation → better convergence).
5. **Flat tools ≥ grouped/DSL refactor.** Weak models ignore the
   bespoke `refactor(add_param, position:"after:x", callsite_fill_in)`
   tool entirely and succeed via plain primitives; the DSL only adds a
   documented deterministic Devstral failure surface.
6. **Model-dependence is narrower than first thought:** capable models
   (Gemma, Qwen) are excellent with *either* no-tool text *or* real
   tool-calls. Mistral-family (Devstral) needs real tool-calls (text
   unusable) but is **prompt-shape-insensitive** — clean isolation
   (sanitized, no-gate, only the prompt varied) gave structured 0/6 =
   minimal 0/6. Devstral is simply weak-at-convergence on this task via
   no-gate tool-calls (~0–2/6 across clean runs) regardless of prompt;
   it makes the right edits (uses the atomic param tool) but doesn't
   land a fully-correct solution within budget. The universal loser is
   tool-shaped-free-text, for everyone. **Open residual (real-bench,
   not a probe):** the one earlier Devstral 4/6 was *gate-on +
   unsanitized*; whether the gate itself helps Devstral-with-tools is
   untested-clean and confounded — the gate-drop still stands on the
   strong Gemma-collapse evidence + simplicity, but flag this for the
   real bench.

## Corrections log (claims revised during the investigation)

Honest record — several earlier conclusions were over-stated and
corrected as methodology tightened (the user caught several):

- **"Modality lock is universal; remove tool-calling for all models."**
  *Refuted.* It conflated arm B's *mixed*-modality failure with
  tool-calls per se. Real tool-calls (D/E/G) are the winner.
- **"Devstral C 0/6 even with a tolerant parser → not a matcher
  problem."** *Invalid* — that C run was 5/6 server-500-contaminated.
- **"Devstral can't act in free-text (tool-call-native, confirmed)."**
  Originally asserted from a *crashed* run. Re-run clean: Devstral F is
  a genuine 0/6 (edits but never converges) — the conclusion survives
  but only after de-contamination; the original basis was invalid.
- **"Gemma prefers no-tool text; tool-calls only 4/6."** *Corrected* —
  with a clean (sanitized) harness Gemma G = 6/6, equal to arm A. The
  4/6 was harness-induced degradation, not a Gemma property.
- **"Mistral-family wants a structured/directive prompt."** *Refuted*
  by a dedicated clean isolation (Devstral, sanitized, no-gate,
  tool-calls, only the prompt varied): structured 0/6 = minimal 0/6.
  The earlier E 4/6 vs G 2/6 was a gate+harness confound, not prompt
  shape. → dropped per-family prompt from the design.
- **"Rust crate" hardcoded in probe prompts** — overfitting; fixed to
  language-agnostic (was a shared constant, didn't bias comparisons).

Methodological takeaways now baked in: always error-audit before
concluding; never compare across harness versions; confirm root causes
from logs, not inference.

## The design (what to change in miniswe)

miniswe already uses tool-calls — so this is mostly *subtraction*.

1. **Config:** `agent.ceremony = "off"` (default) vs `"strict"`
   (legacy plan-gate, opt-in escape hatch only). No model-family
   branching — prompt shape is the same for all (per-family refuted).
2. **Remove the plan gate** (`run.rs` write-gating, `plan.rs`
   gate-exists checks) and the enumerable `PLAN CHECK <step>` /
   plan-unlock prompt language in `context/mod.rs`. Keep an optional
   *soft* post-completion progress note only.
3. **Flatten the tool surface** (`definitions.rs`): replace grouped
   action-dispatch (`file`/`code`/`plan` with conditional-required
   params) and the `refactor` DSL with flat single-purpose tools
   (`read_file`, `search`, `shell`, `replace_range`, `insert_at`,
   `write_file`, and at most a flat `add_param`/`rename`).
4. **Keep & assert `sanitize_messages`** on every request path
   (`context/mod.rs` already has it — make it non-optional, add a
   debug assert that no two consecutive same-role messages are sent).
5. **One minimal system prompt** (`context/build_system_prompt`):
   short, example-driven, no plan-gate / no `PLAN CHECK` language. No
   per-family branching (clean isolation: structured = minimal for
   Devstral). Same prompt for every model.
6. **Structural stop only:** rely on tool-call turn termination; delete
   instructed-"STOP/emit one"-style prompt text (it doesn't hold).

Net code direction: delete the plan gate + its ~5 nudge epicycles,
delete the refactor DSL, flatten tools, harden message sanitization,
collapse to one minimal prompt. Almost entirely removal — no new
per-family machinery.

## Not doing (and why)

- **No aider-style no-tool text tier as default.** Arm A is 6/6 fast
  for Gemma/Qwen but only *matches* clean tool-calls there and is 1/6
  for Devstral. A parallel text agent adds surface and helps no model
  that the deceremonialized tool path doesn't already serve.
- **No auto-escalation machinery (v1).** It existed to rescue a simple
  tier we are no longer defaulting to. Revisit only if a deceremonical-
  ized tool agent underperforms on the real bench.
- **No more synthetic probes.** The greeter task has given all it can;
  further arms are epicycles.

## Validation (the only experiment that matters next)

Implement the subtractions behind `agent.tier`, then run the **real**
`scripts/run-agent-comparison.sh` (real task + validation suite) vs
aider, on Gemma + Qwen + Devstral. Gate: deceremonialized tool agent
≥ current miniswe and ≥ aider on all three. That is the next evidence
— not another probe.

## Caveats

- One synthetic-but-representative task; N=6; 15-round budget. Effect
  patterns are large and consistent (F = 0/6 ×3; G = 6/6 for
  Gemma/Qwen). Absolute rates are budget-sensitive; *contrasts* are
  robust. Final decisions go through the real bench.
- Devstral arm A (1/6) logged ERR=0 but the probe harness lacked
  Mistral hygiene for much of the investigation; treat Devstral
  text-mode numbers as "weak, directionally trustworthy" not exact.
- Probe artifacts archived: `docs/simple-tier-probe*-*.jsonl`,
  `docs/simple-tier-probeFG-san-*.jsonl`, scripts in `scripts/`.
