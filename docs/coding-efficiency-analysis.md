# Coding-efficiency analysis & improvement plan

*Date: 2026-06-13. Status: **analysis only — no code changed, no benchmarks executed in producing this.***

**Provenance.** This was produced by a multi-agent read of the production code (10 subsystem
readers + a token/round-budget tracer), a synthesis pass that ranked 43 ideas, and 9 adversarial
critics that vetted the top ideas against this repo's git history and `docs/experiment-results-*`.
No benchmark was run; every empirical claim is the critics' reading of **existing** records
(`docs/experiment-results-20260511.md`, git log), not fresh runs. Treat the ranked ideas as
hypotheses, each with a concrete A/B to run before trusting it.

---

## 1. What the production code does

Per user message, `run.rs:285` runs a round loop:

1. `maybe_compress` the history,
2. call the LLM with the *currently-visible* tool set (under `ceremony=Strict` the write tools
   stay hidden until `plan(action='set')`),
3. parse tool-calls, repairing XML / leaked-arg corruption in `llm/mod.rs`,
4. dispatch on a thread pool,
5. append per-edit feedback (tree-sitter AST + rust-analyzer LSP diagnostics), a `[round N/max]`
   footer, and strict-only plan nudges, then feed results back.

Recovery machinery: a consecutive-identical loop detector (trips at 3; one recovery then abort),
graduated nudges (round-12 no-plan, round-20 stall, premature-exit), and prunable-failure history
rewrites.

Context is **pull-based**: repo map / profile / notes are tools the model calls — they are **not**
auto-injected. Only `plan`, `scratchpad`, `mcp`, `usage_guide` are on by default
(`config/mod.rs:222-236`; `repo_map: false`).

Defaults (`config/mod.rs`): `ceremony=Strict`, `edit_mode=Fast` (blind line primitives), grouped
tools incl. the `refactor` DSL, `llm_concurrency=1`, `context_window=50K`.

---

## 2. The most important framing (read first)

**The recurring failure is end-to-end wiring, and its cause is officially open.**

On the canonical multi-file task, 20–30B open models *commonly* stall at 5/6 — usually on the
behavioral `smoke` check (value wired through the CLI but not actually threaded into runtime),
though an earlier era stalled on `test:FAIL` (call-sites not updated). **Full 6/6 runs do exist**
(Devstral and Gemma each reached 6/6 in `experiment-results-20260511.md:30-31`), so this is a
*common ceiling on one task*, not a hard wall.

Crucially, the record leaves the **cause open**:

> "5/6 ceiling on `smoke:FAIL` … Multi-step value-threading. Could be model capability, or could be
> **the agent's tool feedback masking the real issue.** Open." — `experiment-results-20260511.md:111-115`

So do **not** conclude "mechanic fixes can't help." If the cause is the agent hiding the failure
(theme #2 below), then agent-side fixes are exactly the lever. The correct first move is **Tier-0
instrumentation that distinguishes the two causes** before betting on either.

---

## 3. What's off — six structural themes

1. **Duplication tax (per round).** `refactor` guidance ships **3×** (tool schema ~530 tok +
   `edit_contract` blurb + routing table); edit examples duplicate the tool schemas. Cutting
   *duplication* (not substance) is the one simplification the bench hasn't already refuted — it
   refuted cutting substance 4-for-4.
2. **Silent-success / false-OK.** The agent tells a small model "all good" when it isn't:
   fast-mode edits that break the build return `success` (`replace_range.rs`); LSP returns
   `[lsp] OK` on a 2 s timeout *before* checkOnSave finishes (`client.rs:259` discards the
   `wait_for_idle` bool); `finish_reason` is hardcoded `"stop"` so truncation is invisible
   (`llm/mod.rs:444`). A small model can't infer failure from a green light.
3. **Remove the choice, hand data.** Small models reliably *ignore* advisory prose ("search for
   callers", "set a plan") but reliably *act* on concrete data and automatic mechanisms.
4. **Grouped-tool boundary leaks.** `replace_range`/`insert_at` are top-level, not `file` actions
   (causing misroute-then-recover, proven by the dedicated recovery hint at `dispatch.rs:137`);
   EXPLORE "read-only" mode still leaves `file(shell/delete/revert)` reachable (filtered by name,
   not enforced at dispatch).
5. **Measurement blind spots.** The trusted metric can't see what's being optimized — token usage
   isn't captured, N=1 single-task can't tell a real +1 from noise, the smoke (the only
   value-threading proof) is gated *last*, and the REPL router is structurally un-benchable.
6. **Frozen pull-context.** The repo map and PageRank graph are built once at `init` and never
   refresh, so when the model *pulls* the map mid-task it gets a stale snapshot that omits the file
   it just created, and gives new files score `0.001` (`repo_map.rs:53`).

---

## 4. Where tokens & rounds actually go (corrected)

**Per-round fixed cost** is **~3K tool schema + ~0.5–0.9K system prompt** — *not* +5K repo map.
(The tracer initially assumed `repo_map` was embedded; it's `false` by default — verified at
`config/mod.rs:230`.) Inside the schema, `refactor` (~530 tok), `plan` (~428), and `spawn_agents`
(~177) dominate.

**Round drivers** (where the 50-round budget burns):

- ~6–12 mandatory pre-edit exploration rounds before `plan(set)` unlocks editing (strict);
- "create a plan first" gate-retry loops (GPT-OSS produced *0 edits* in a whole window from this);
- a forced plan-update round before every compression;
- one-tool-call-per-round serialization;
- silent tool-call truncation at `max_output_tokens=4096`.

---

## 5. The plan — tiered, with critic verdicts

### Tier 0 — Fix the bench so you can validate anything (do first)

Every critic's "cheapest decisive experiment" needs instrumentation you don't have.

| Idea | What | Why |
|---|---|---|
| `real-token-accounting-in-bench` | Snapshot llama-server `/metrics` before/after each attempt in `run-benchmark-docker.sh` | Today rounds/wall only — no token win is measurable |
| `bench-n-runs-statistical` | N≥5 fresh-container runs; report mean smoke-rate + variance, not best-of-3 | N=1 can't distinguish +1 check from noise |
| `ceremony-refactor-toggle-matrix` | Emit `[tools].ceremony/.flat` variants in the harness | Ablate the exact knobs your docs argue about, instead of offline probes the docs say mispredict the bench |

### Tier 1 — Verified safe wins (low effort, low risk, do now)

| Idea | Verdict | Notes |
|---|---|---|
| `compressor-consistent-token-measure` | **Promising** | Real bug: trigger counts tool-call args, the keep/split (`compressor.rs:96,111`) doesn't → "successful" compression overshoots budget. Reshape: fix `needs_compression` too; **drop** the "fewer truncated_tool_call retries" claim (output-side, wrong mechanism). |
| `gate-spawn-agents-on-concurrency` | **Promising** | `spawn_agents` ships unconditionally but is **never invoked in any stored bench run**; ~177 tok/round + most complex tool shape, useless at `concurrency=1`. Gate behind `>1`. Frame as hygiene (flat pass rate, fewer tokens). |
| `lsp-honest-pending-vs-OK` (half of #7) | **Needs-scoping → ship this half** | High value *on this bench specifically*: target is the 32K-LOC miniswe repo, where flycheck exceeds the 2 s budget, so the false-`OK` is the *common* case. Thread the discarded `wait_for_idle` bool out, emit `[lsp] pending` vs `[lsp] OK`. Your old cargo-check design did this (`suggestions.md:51`); the LSP migration dropped it. **Promoted — see §2: directly tests the "feedback masking" hypothesis.** |
| `dedup-refactor-and-edit-guidance` | (not vetted) | Theme #1; cut the 3× duplication, keep substance once. The *safe* simplification. |
| `wire-or-delete-context-budget-knobs` | (not vetted) | `history_budget`/`scratchpad_budget` etc. referenced only by display code, never by `src/context`. Dead config — wire or delete. |
| `search-output-budget` | **Needs-scoping** | Real: `search.rs:93` is the one result path with no byte/line cap. But ~0 bench impact (Rust repo has no minified lines). Merge as **hygiene via a unit test**, not as a coding win. |

### Tier 2 — Worth trying, but MEASURE FIRST (good mechanism, unproven benefit)

| Idea | Verdict | The catch |
|---|---|---|
| `is_idle-ignore-flycheck` (other half of #7) | reshape | Make `is_idle` skip the Flycheck progress token so fast in-memory diagnostics return without blocking on cargo check. **Don't** blanket-disable checkOnSave un-A/B'd. |
| `finish-reason-truncation-signal` | **Needs-scoping** | Most truncations already cut mid-JSON → already recovered at `run.rs:566`. Real net-new value is cross-server hint wording + truncated *plain-text* turns. Bench at the real 4096 regime first; drop the dead Usage clause. |
| `fast-mode-auto-revert-ast-break` | **Speculative** | Ceiling is AST-clean, so this can't touch it; hard-revert risks breaking legit transient-broken multi-edits. Reshape to a **soft nudge**; first instrument how often models actually linger on AST-broken files. |
| `replace-range-optional-anchor` | (not vetted) | Optional `expect` first-line check before splicing (smart mode + refactor already anchor; blind `replace_range` doesn't). Plausible edit-reliability win. |
| `rebuild-pagerank-and-reanchor-repo-map` | (not vetted) | Fixes theme #6 (stale pulled map). Note: `shrink-repo-map-after-first-edit` is **moot by default** (map isn't re-shipped unless `repo_map=true`). |

### Tier 3 — The actual frontier (end-to-end wiring)

- `smoke-equivalent-early-behavioral-check` — run the PONG_42 smoke as soon as `build:PASS` and
  inject the result mid-loop, so the model learns its value-threading failed **while it still has
  rounds to fix it**. **Promoted — see §2:** if the ceiling cause is feedback-masking, this (and
  honest-LSP) are the most direct tests of it. Treat as a research bet, not a quick win.

### Explicitly DROP / deprioritize (refuted by your own history)

- `compress-budget-subtract-system-prompt` → **LikelyRefuted.** Premise (5–10K system prompt) is
  false — context is pull-based, repo map isn't embedded by default. Subtracting it *shrinks*
  `raw_budget` and risks re-creating the documented Devstral starvation 0/6
  (`experiment-results-20260511.md:62`). At most add a cheap pre-send
  `system+tools+history ≤ window` assertion that logs if it ever fires (prediction: it won't).
- `auto-list-callsites-on-signature-change` → **Redundant.** Built and refuted twice
  (`ee20308`, `f244ba7`); auto_check *already* prints broken file:lines. Only salvage: raise the
  cap-of-3 in `extract_error_locations` (`cargo_check.rs:101`) to ~15, then confirm it *still*
  doesn't flip `test:FAIL` (which would re-confirm the round-budget root cause).
- `plumb-min-p-sampling` → **LikelyRefuted.** Your catalogued tool-call failures are KV-cache
  contamination (already retried via `cache_prompt:false`), byte-deterministic schema confusion
  (re-sampling won't help), or truncation — none are tail-sampling. Project stance is stock
  sampling. If curious, offline-replay `llm_dumps` min_p=0 vs 0.05; zero prod risk.

---

## 6. Context v2 (compaction strategy)

Discussion outcome on whether to replace lazy bulk-compaction with per-step summarization +
pointers:

- **The real compression defect is the measurement inconsistency (Tier 1 #1), not the
  lazy-vs-eager philosophy.** Fix that first.
- **Eager-author / batched-apply.** Author the per-step recap fresh (fold a one-line `RECAP:` into
  the model's own turn, or a tiny Fast-model call) but only *swap it into the prefix in batches* at
  the budget crossing — preserves summary quality without busting the llama.cpp KV-cache every turn
  (per-turn prefix mutation = full prefill every turn on local inference).
- **Re-pull, don't replay.** For recall: durable records (decisions, the action log) can be served
  verbatim; file/search/diagnostic snapshots must **not** be replayed (the agent mutates those
  files, so cached bytes go stale). Store a breadcrumb keyed by **round number** (already on every
  result via `[round N/max]`) and re-fetch live; an optional content-hash is a *validity check*, not
  the address. `individually-age-tool-results` (#37) is the dead `compress_history` (`mod.rs:556`)
  resurrected to do exactly this. The archive currently truncates contents (`compressor.rs:382`) so
  it isn't lossless anyway — archive the **log**, re-pull the **content**.
- **Drop the system-prompt-budget worry** — pull-based context means the system message is
  ~0.5–1.5K, not the 5–10K that would justify it.

---

## 7. Caveats

- Analysis-stage hypotheses only; no code changed, no benches run here.
- The token argument depends on `repo_map=false` (verified). If a bench config flips it on, redo
  the per-round math and the Tier-2 repo-map items get more important.
- The ceiling's cause is **open** in your own notes (capability vs feedback-masking) — which is the
  argument for Tier-0 instrumentation before betting on fixes.

---

## Appendix — full ranked idea list (43)

Ranked by the synthesizer as impact-for-small-model-coding ÷ effort. The top 9 carry an adversarial
critic verdict; the rest are unvetted. `file:line` citations are the agents' and should be
spot-checked before acting.

### 1. Use one consistent per-message token measure (incl. tool-call args) for both compression trigger and split
`compressor-consistent-token-measure` · impact **High** · effort S · risk Low · critic verdict: **Promising**

**What:** Build a single msg_tokens[] that includes content + tool_call argument tokens, and use it for both the trigger (total_tokens) and the keep/compress split (kept_tokens). Today compressor.rs:91-95 adds tool-call args to the trigger total but msg_tokens[] (line 96) excludes them, and the split at line 111 uses msg_tokens — so compression fires but keeps MORE raw history than budgeted.

**Why:** Coding histories are dominated by large write_file/replace_range tool-call args; under-measuring them is exactly why the model still rides the context ceiling after a 'successful' compression pass and loses tool-call fidelity / gets server-truncated. Confirmed bug: line 92 increments total_tokens with tc args, line 96 pushes only `tokens` to msg_tokens, line 111 sums msg_tokens for the split. Pure correctness fix to the live compression path.

**Validate:** Run the multi-file docker bench; instrument the '[compressor] summarized N messages' line and assert post-compression total lands at-or-below raw_budget in /output dumps. Expect fewer subsequent compressions per attempt and fewer truncated_tool_call retries (run.rs:449). Pass rate flat-or-up.

### 2. Subtract the actual assembled system-prompt tokens (repo map + plan + lessons) from the compression budget
`compress-budget-subtract-system-prompt` · impact **High** · effort S · risk Low · critic verdict: **LikelyRefuted**

**What:** Pass the system-message token count (already computed as AssembledContext.token_estimate) into maybe_compress so available = context_window − tool_def_tokens − system_tokens − window/6. Today the compressor loop skips role==system (pushes 0, compressor.rs:83-84) so it never accounts for the 5-10K-token system message.

**Why:** available only subtracts tool defs + output headroom, never the system message carrying the 5K repo map + profile + plan + lessons. raw_budget=available/3 then lets raw history grow on a window that already silently lost 5-10K. Actual send (system + tool_defs + summary + raw) can exceed context_window → server-side truncation drops the tail (latest tool_call) or starves earlier multi-file state. This is the SAME class as the improvements.md:24 overflow that caused a real 0/6 (experiment-results-20260511.md:62-69); the tool-def term was fixed but the larger system-prompt term was left out.

**Validate:** A/B the docker bench at context_window=50K/60K: compare pass rate, round count, and count of '32K/overflow' or truncated_tool_call retries in /output. Expect fewer overflow/starvation failures.

### 3. Capture finish_reason from the stream and surface truncation cross-server
`finish-reason-truncation-signal` · impact **High** · effort S · risk Low · critic verdict: **Needs-scoping**

**What:** Read StreamChoice.finish_reason from the last SSE delta (and Usage if present) instead of hardcoding 'stop' at llm/mod.rs:444. When finish_reason=='length', inject the existing truncated_tool_call_hint (hints.rs:109) regardless of server, decoupling truncation recovery from the brittle llama.cpp English substring (TRUNCATED_TOOL_CALL_MARKER, mod.rs:605).

**Why:** Confirmed: finish_reason is hardcoded Some('stop') and StreamChoice.finish_reason is never read. The truncation-recovery path in run.rs:449 keys ONLY off the llama.cpp 500 marker, so on vLLM/Ollama (both named targets) or any non-500 truncation (truncated content, truncated-but-valid-JSON args), the model gets a silently-cut response treated as complete — it loops or gives up with a half-written edit. finish_reason is the one reliable cross-server signal the model overran max_output_tokens (bench runs 4096, making this frequent on large write_file payloads).

**Validate:** Set max_output_tokens artificially low (e.g. 1024) in config.toml to force truncation; A/B before/after. Compare pass rate and wasted rounds; diff llm_dumps for length-finish events.

### 4. Apply tool_output_budget_chars + per-line cap to search results
`search-output-budget` · impact **High** · effort S · risk Low · critic verdict: **Needs-scoping**

**What:** In search.rs, cap total output at tool_output_budget_chars() and each hit line at MAX_LINE_CHARS (reuse crate::truncate_chars), with a '[N more matches — refine query]' note when capped. Confirmed: search.rs:93-101 applies ONLY a max_results cap — no byte budget, no per-line cap, unlike every other result tool.

**Why:** search is the one unbounded result path. A regex over minified JS/JSON/lockfiles/generated code returns up to 20 full lines verbatim, each potentially thousands of chars — far past the ~5000-char per-result budget on a small-context model. One oversized search evicts the working context and pushes toward the documented 32K overflow (improvements.md:24); small models can't recover context once it's masked/compressed, so every subsequent round silently degrades. The fix mirrors read_file.rs:22 / shell.rs:180 which already do this.

**Validate:** Run the bench on a repo with minified/long-line fixtures; compare peak tokens/round and overflow incidence with/without the cap on search-heavy tasks. Pass rate flat-or-up.

### 5. Auto-revert fast-mode edits that introduce an AST break (mirror smart-mode for the DEFAULT path)
`fast-mode-auto-revert-ast-break` · impact **High** · effort M · risk Med · critic verdict: **Speculative**

**What:** When replace_range/insert_at turns baseline ast_ok=true into ast_ok=false (a clean structural break), automatically restore the prior revision, return success=false, and tell the model 'edit reverted, AST broke at L:C — re-issue with correct syntax/range'. Gate ONLY on the unambiguous AST break (tree-sitter, no LSP wait); keep LSP-error-only regressions informational.

**Why:** Fast mode is the DEFAULT (confirmed edit_mode=Fast), yet ALL failure-recovery sophistication (auto_check, auto-revert, regression gating) lives in the bypassed smart path. A broken replace_range returns ToolResult::ok with '[ast] broken' buried in a feedback table the small model must parse, recognize, then choose to revert with the right rev number — the most fragile step in the default loop, and the experiment logs show models do it unreliably (they keep editing on broken code). Auto-revert on AST break turns an N-round corruption-notice-revert cycle into a self-correcting single round; the AST result is already computed and the revert content is already in the revision store. Only auto-revert when the SAME call caused the break (baseline ast_ok=true→false) to avoid clobbering edits on already-broken files.

**Validate:** run-benchmark-docker-fast.sh A/B: compare pass rate and tool-rounds-per-task. Hypothesis: fewer rounds, equal-or-higher pass.

### 6. Auto-enumerate callsites in the tool result when a signature edit breaks the build
`auto-list-callsites-on-signature-change` · impact **High** · effort M · risk Med · critic verdict: **Redundant**

**What:** When replace_range/insert_at/refactor changes a function signature (or auto_check sees expected/argument/found errors, edit_orchestration.rs:464), run the existing sites.rs find_callsites for the changed symbol and list concrete file:line callsites inline in the result. Turn the existing advisory 'search for ALL callers' hint into data.

**Why:** The smoke ceiling's recurring root cause is un-threaded values / unupdated callsites. The canonical multi-round failure: model edits the definition, the build breaks at every callsite, and it hunts each one by hand. Small models reliably IGNORE the textual nudge (experiment-results-20260511.md:31 — Gemma never used change_signature, preferred replace_range) but reliably ACT on a concrete file:line checklist — their strong mode. Converts the longest round-burning sequence into a bounded list of edit targets, directly attacking the documented universal 5/6 smoke ceiling.

**Validate:** run-benchmark-docker.sh + fast variant on test:PASS / smoke:PASS rate; check whether the 'arguments but' test-callsite failure drops. Cross-check on stork for cross-file generalization; measure extra tool-result tokens with a token meter. Cap find_references latency with the existing wait_for_idle.

### 7. Disable rust-analyzer checkOnSave + never report '[lsp] OK' on timeout-without-idle
`lsp-disable-checkonsave-and-honest-timeout` · impact **High** · effort M · risk Med · critic verdict: **Needs-scoping**

**What:** Add initializationOptions to the initialize payload (confirmed NONE at client.rs:518) setting checkOnSave=false so per-edit diagnostics come from sub-second incremental type analysis, and run a real cargo check only on demand. Also: distinguish 'idle reached, 0 errors' from 'timed out, unknown' (wait_for_idle returns a bool, discarded at client.rs:259) and surface the latter as '[lsp] pending', not '[lsp] OK'.

**Why:** Confirmed: initialize sends no initializationOptions and didSave:true, so default checkOnSave spawns a real cargo check on every edit and wait_for_idle blocks on it — the advertised '~200ms vs cargo check' (lsp-design.md:38) doesn't exist in the configured state. Worse, diagnostic_timeout_ms=2000 caps the wait below typical compile time, so get_diagnostics returns empty and feedback says '[lsp] OK' / '0 errors' while the build is actually broken — a FALSE-OK that defeats the recover-from-mistakes loop (the model trusts a clean bill it never got and ships a broken edit). Disabling checkOnSave makes feedback genuinely sub-second; the pending/OK distinction removes the dangerous false signal. The harness's final cargo check still gates correctness, so a regression would surface.

**Validate:** A/B docker bench: default checkOnSave vs checkOnSave=false. Compare pass rate, mean rounds, wall-clock/task. Separately count tasks where final cargo check fails despite all per-edit feedback saying OK (false-OK proxy) before/after the pending fix.

### 8. Plumb min_p (+ optional top_p/stop) into ModelConfig and the request body
`plumb-min-p-sampling` · impact **High** · effort S · risk Med · critic verdict: **LikelyRefuted**

**What:** Add optional sampling fields to ModelConfig (default None = unchanged) and inject them alongside temperature in chat_with_cancel (only temperature+max_tokens reach the body today, mod.rs:159). Set per-model defaults (e.g. min_p=0.05) in the bench config.

**Why:** For small quantized models, tool-call JSON validity is extremely sensitive to tail sampling: a single low-probability token in an arguments string breaks parsing and wastes a round. min_p/tighter top_p truncates that tail and is the highest-leverage decode-level knob for cutting malformed tool calls — and it isn't even plumbed. `stop` in Xml/Auto mode caps post-tool-call prose (output tokens + second-block corruption). Cheaper and lower-risk than grammar-constrained decoding while attacking the same failure class.

**Validate:** A/B the bench with min_p=0 vs 0.05 (and top_p 1.0 vs 0.9) via config.toml. Measure malformed-tool-call rate (count 'Invalid JSON in tool arguments' + truncation markers), pass rate, tokens/round. Watch pass rate (not just validity) for over-truncation hurting reasoning.

### 9. Gate spawn_agents behind llm_concurrency > 1
`gate-spawn-agents-on-concurrency` · impact **Med** · effort S · risk Low · critic verdict: **Promising**

**What:** Only push spawn_agents_tool_definition (confirmed unconditional at run.rs:105) when config.runtime.llm_concurrency > 1. Default single-GPU setups (llm_concurrency=1, confirmed config:507) stop seeing it.

**Why:** On the target hardware llm_concurrency=1 serializes all LLM calls, so spawn_agents' core promise (concurrency) is unavailable by default — yet its ~140-token nested-array schema (the most complex shape, exactly what small quantized models format worst) ships every round and presents a mis-selectable tool whose benefit is zero on the default deployment. Also resolves the 'Emit ONE tool call per response' vs spawn_agents contradiction. No spawn_agents A/B exists in the docs, so gating is pure low-risk overhead removal.

**Validate:** Standard bench with default config before/after. Expect tokens/round to drop ~140 and pass rate flat-or-better; confirm no task regresses (any task that needed spawn_agents would only have helped at concurrency>1).

### 10. De-duplicate refactor/edit guidance: cut triplicated prose, keep substance once
`dedup-refactor-and-edit-guidance` · impact **Med** · effort S · risk Med

**What:** Refactor guidance appears 3x per round — routing table (mod.rs:243), refactor_blurb (mod.rs:117), and the refactor tool schema description (~530 tok, definitions.rs:40). Keep the routing table; slim the refactor schema to mechanics only (actions + name-resolution); drop the redundant blurb. Likewise collapse the edit_contract's inline JSON examples (mod.rs:163-186) to ONE canonical example, relying on the tool schemas for the rest.

**Why:** ~400-900 tok of triplicated/duplicated instruction re-sent EVERY round pushes real code away from the high-attention prompt edges and risks small models over-weighting repeated emphasis or copying the prose example's literal placeholders (start:10,end:15) instead of computing from file state. Crucially this cuts DUPLICATION of identical content, not SUBSTANCE — the bench refuted cutting substance 4-for-4 (off/advise/flat all regressed) but never tested cutting duplication, making this the one plausibly-safe simplification.

**Validate:** A/B strict+grouped bench (Qwen 6/6 control + Gemma + Devstral) HEAD vs de-duplicated prompt+schema. Watch refactor-adoption, smoke:PASS, tool-malformation rate, tokens/round. Treat any Qwen 6/6→<6/6 as a HARD FAIL and revert (substance was load-bearing).

### 11. Re-rank/refresh the repo map as the model edits (rebuild PageRank on reindex; re-anchor on touched files)
`rebuild-pagerank-and-reanchor-repo-map` · impact **Med** · effort M · risk Low

**What:** In reindex_file (walker.rs:149) and reindex_project_incremental, recompute and re-save the dependency graph after updating symbols (today only init.rs:88 ever builds it). Cheaper variant: lazily rebuild in render() when index is newer than graph.json and give new files a median score instead of 0.001 (repo_map.rs:53). Optionally re-seed personalization from the paths the model actually opened/edited.

**Why:** Confirmed: the PageRank graph is frozen at init; reindex refreshes symbols but never the graph. So a file the model just created (src/foo.rs) gets the 0.001 fallback and sinks to the bottom of the map or gets budget-truncated out — the file it most needs to re-see is ranked last. Note references is dropped to empty on load (mod.rs:116), so a rebuild must re-extract references. Keeps the map's ORDER correct for an iterating small model, removing the biggest 'map lies about what's important' failure.

**Validate:** A/B graph-rebuild-on-reindex on the multi-file self-task (spans 4+ files); measure pass rate and rounds spent re-discovering just-created files. Best validated together with the stork large-repo task (idea reactivate-stork-bench).

### 12. Count blocked/rejected tool calls toward the stall + budget counters
`count-blocked-calls-toward-stall` · impact **Med** · effort S · risk Low

**What:** Increment calls_since_last_edit (and feed the loop key) in the invalid-JSON (run.rs:579), plan-only-block (659), no-plan-write-block (673), and repeated-read (610) continue paths — not just the normal dispatch path (1003). Move the increment to the top of the per-tool loop.

**Why:** The most common strict-mode failure — repeatedly attempting writes before a plan exists (GPT-OSS produced 0 edits in a full window from this) — never trips the stall warning because rejections bypass the counter. Counting them makes the urgent 'set a plan now / edit tools are hidden' warning actually fire for the model that most needs it, instead of grinding rejections to max_rounds.

**Validate:** A/B on strict (default) with GPT-OSS/Gemma (the plan-ignorers); measure pass rate and rounds-to-first-edit. Earlier stall warning should cut wasted rounds.

### 13. Add an alternation/window loop detector alongside the consecutive one
`windowed-loop-detector` · impact **Med** · effort S · risk Med

**What:** Keep a small ring buffer (last ~6) of loop_call_keys; trip a softer hint when the SET of distinct keys in the window is <=2 (covers A-B-A-B, A-A-B-A-B), in addition to the existing >=3-consecutive rule. Reuse loop_call_key/canonical_json for keying consistency.

**Why:** Confirmed: same_call_streak only increments on exact consecutive match (run.rs:587-592); any different call resets it. Small models loop by 'trying the other file/line then coming back' (read A→read B→read A→B, or replace_range L5→L9→L5) far more than by emitting the identical call 3x — these never trip the detector and burn the whole budget with no recovery signal. improvements.md:40 already flagged 'detect file re-reads' as the open half of this. Med risk: false-positive nudges could hurt legitimate two-file editing.

**Validate:** A/B Devstral+Gemma+Qwen with/without the windowed detector; primary pass rate, secondary avg rounds and tool_errors. Watch rounds-to-first-edit for false-positive damage to legitimate multi-file work.

### 14. Centralize over-budget store-and-preview in execute_tool
`centralize-output-truncation` · impact **Med** · effort M · risk Low

**What:** Wrap the ToolResult from dispatch.rs:execute_tool: if content exceeds tool_output_budget_chars(), spill full content to .miniswe/tool_output/ and replace with preview + read-back pointer (the pattern shell.rs:194 / web.rs:324 already use). Keep per-tool truncation as a fast path. Superset of the search-specific cap; also bounds find_references/goto which currently emit one uncapped line per ref.

**Why:** Budget is re-implemented per tool and thus easy to forget (search.rs is the live proof; find_references/goto are unbounded too). A single gate at the dispatch boundary makes over-budget results structurally impossible regardless of which tool produced them — exactly what improvements.md:66 asks for. The read-back pointer lets the model pull the rest deliberately instead of drowning. Closes the bug class rather than one instance.

**Validate:** A/B with the central gate on/off; assert no result message exceeds budget (log assertion) and compare rounds-to-completion + token totals on tasks triggering large search/diagnostics output.

### 15. Surface a behavioral/runtime self-test EARLY, not just at grading
`smoke-equivalent-early-behavioral-check` · impact **Med** · effort M · risk Med

**What:** Give the agent a cheap way to verify end-to-end behavior before exit — have the harness run the PONG_42 smoke as soon as build:PASS and inject the result mid-loop, and/or nudge the agent to run its own behavioral check. The model learns its value-threading failed while it still has rounds and context to fix it.

**Why:** The documented universal ceiling for all 20-30B models is smoke:FAIL — they wire + compile + pass their own weak tests, then exit believing they're done; LSP/compile feedback confirms the type-checks, actively MASKING the semantic gap. A small model cannot infer 'the value never reaches assemble()' from a green compile — it needs a behavioral signal. Surfacing it early converts an unrecoverable end-of-run failure into a normal fix-on-feedback round (which small models handle well). Pairs with auto-list-callsites which attacks the same un-threaded-value root cause.

**Validate:** run-benchmark-docker A/B: baseline vs build that runs smoke after first build:PASS and injects the result. Compare smoke:PASS across N>=5/model on Devstral+Gemma+Qwen. Reuse the exact smoke invocation at run-benchmark-docker.sh:391-396.

### 16. Make EXPLORE genuinely read-only at the dispatch layer, not by name-filtering
`explore-mode-runtime-readonly-guard` · impact **Med** · effort M · risk Low

**What:** Thread a read_only bool into dispatch.rs and hard-reject file(action in {shell,delete,revert}) and any write sub-action when set (or drop those actions from the file schema for EXPLORE turns). Fix the unit test (repl.rs:1770) to model file sub-actions, not standalone tools.

**Why:** Confirmed design flaw: read_only_tool_defs filters by TOP-LEVEL tool name, but writes are grouped sub-actions — file(action='shell') can run sed -i/rm/git checkout, plus action='delete'/'revert'. The file tool MUST survive (it carries read/search), so all three mutating actions remain offered, and dispatch.rs has zero read-only awareness. A small model that ignores the soft [INVESTIGATION MODE] prompt can mutate the repo in a turn the user was told is 'no edits'. The passing unit test gives false confidence by modeling revert/delete as standalone tools. A runtime guard is the only thing making the read-only promise true.

**Validate:** Unit test: assert dispatch rejects file(action=shell/delete/revert) under read_only. If an explore bench variant exists (idea explore-mode-bench-flag), assert git diff is empty across its runs.

### 17. Optional OLD-anchor (expect param) on replace_range, verified before splice
`replace-range-optional-anchor` · impact **Med** · effort M · risk Med

**What:** Add an optional `expect` param: the model passes the first 1-2 lines it believes sit at `start`; replace_range verifies them (whitespace-tolerant) before splicing. On mismatch, search ±5 lines and return 'L{start} is {actual}; did you mean L{found}?' WITHOUT writing (or auto-correct to the found line). Confirmed: replace_range.rs:71-119 validates only that start/end are in-range — no content check, unlike smart REPLACE_AT (byte-exact OLD) and refactor (prefill-anchored).

**Why:** Small models routinely produce line numbers off by a few — read_file's compression gaps (reading.rs) shift the model's mental line count. With no anchor, an off-by-3 replace_range silently overwrites real code, and (per fast-mode-auto-revert) the call still returns success. The optional anchor catches the dominant blind-edit failure at zero round cost — the wrong-line edit never lands. Optional + tolerant so models that get line numbers right pay nothing, keeping the primitive simple under ceremony.

**Validate:** A/B run-benchmark-docker-fast.sh; measure wrong-line recovery sequences in traces (revert-after-ast-broken). Make param optional+tolerant so a mis-filled anchor doesn't block a correct edit.

### 18. Make loop recovery forgiving: reset/decrement the recovery budget after productive edits
`loop-recovery-reset-on-progress` · impact **Med** · effort S · risk Low

**What:** Reset loop_recoveries when a successful file write lands (next to the trackers reset at run.rs:987-990), and raise the kill threshold from 2 distinct loops to ~3. Today the second loop anywhere in the turn sets had_error and kills the whole attempt (run.rs:632-637), even after real edits in between.

**Why:** A model that loops once early (on file shape), recovers, edits successfully, then loops again 30 rounds later (on a compile error) is treated identically to one spiraling — the second loop throws away the whole in-progress attempt and its partial-edit momentum. Tying the budget to progress converts would-be-aborted attempts into completions; the max_rounds cap still bounds runaway.

**Validate:** A/B with the reset; key metric pass rate and attempts-per-task (fewer hard aborts should lower avg_attempts in the score formula). Confirm avg_rounds doesn't blow up.

### 19. Lift XML content tool-calls when all JSON tool_calls are unparseable (+ tolerant truncated-XML parse)
`salvage-xml-on-broken-json-calls` · impact **Med** · effort M · risk Med

**What:** In normalize_xml_tool_calls Auto branch (mod.rs:672), change the short-circuit from 'has any tool_calls' to 'has any tool_call whose args parse OR was repaired'; if every JSON call is broken and content has valid XML, lift the XML instead of discarding it. Also generalize extract_parameters_tolerant to the pass-2 content lift so a clean-but-truncated XML block (no close tag) is best-effort lifted with a truncation warning rather than dropped silently.

**Why:** Salvages turns where a small model dual-emits a malformed JSON call AND a correct XML block (common for Qwen/Hermes-style models) — today repair_leaked_args returns None (args aren't valid JSON) and the XML lift is short-circuited, so run.rs:566 reports 'Invalid JSON' and discards usable output: a guaranteed wasted round. The tolerant-truncation variant recovers the very common max_tokens-cut clean XML block (parse() bails without a close tag) that currently produces a silent empty turn. Strengthens the validated JSON-tool-call path; XML handling is pure salvage.

**Validate:** Unit-test the dual-emit and truncated-XML shapes, then run the Qwen3-Coder-Next bench (the dominant XML-leak case) before/after. Compare pass rate and 'Invalid JSON in tool arguments' / empty-turn counts. Prefer a truncation hint over auto-executing when the close tag is missing.

### 20. Truncate the echoed raw args on JSON-parse failure and route it through the prunable de-prime path
`deprime-json-parse-failures` · impact **Med** · effort M · risk Med

**What:** At run.rs:566, cap the echoed Raw args to first+last ~200 chars (keep the full parser error {e}) and route a pure JSON-parse failure into the same prunable-failure rewind (run.rs:546-561) so the malformed assistant turn is replaced by one terse corrective instead of left in history.

**Why:** On a parse failure the tool_result re-injects the entire multi-KB malformed args blob; small models PRIME on the recent malformed shape and reproduce it — the exact priming-chain the prunable-failure rewind was built to break, but that rewind only fires for validator failures, not JSON-parse failures. So the worst tool-call corruptions currently get the LEAST de-priming. Keep the parser error full so the real syntax issue isn't hidden.

**Validate:** Bench tasks with large edits (big args blobs). Compare tokens/round and consecutive-malformed-call streaks before/after.

### 21. Enrich diagnostic feedback with the offending source line (+ re-enable relatedInformation)
`enrich-diagnostics-with-source-line` · impact **Med** · effort S · risk Low

**What:** In render_file_diagnostics (feedback.rs:151) and the orchestration error report, append the actual source line (reuse read_source_context as goto_definition already does). Re-enable relatedInformation in initialize (confirmed disabled at client.rs:533) and include the diagnostic code/quick-fix hint when present.

**Why:** Diagnostics today are bare 'L42:9: mismatched types: expected String, found &str' with no code context. A tiny model often can't map that back to the actual code without re-reading the file (another round); rust-analyzer carries relatedInformation + a code/suggestion that's currently dropped. More grounding tokens at the exact failure point reduce mis-edits and re-read rounds. Keep the existing top-N cap so context doesn't bloat.

**Validate:** A/B mean rounds-to-fix after an LSP regression, and pass rate. Token cost rises slightly per diagnostic; verify net rounds drop.

### 22. Name-based fallback for goto_definition/find_references when the column misses
`lsp-name-fallback-for-nav` · impact **Med** · effort M · risk Low

**What:** When goto/find returns empty at (line,column), reuse the existing workspace_symbol/document_symbol path (already used by refactor in sites.rs) to resolve the symbol by name on that line and retry, returning the best hit with a 'resolved by name' note. Relax the tool description so the model knows column can be approximate.

**Why:** Confirmed: nav tools require exact (line,column) with no fallback (code_intel.rs:98-119), yet the codebase already built a name-based resolver and uses it for refactor target resolution. Small models reliably know the line and symbol name but not the exact 1-based column; a one-off column yields 'No definition found' and the model wastes a round re-guessing coordinates or falling back to grep. A name fallback turns the dead end into a hit.

**Validate:** Hard to isolate end-to-end (nav is opportunistic); add a targeted probe with off-by-one columns and measure success-found rate, plus watch overall bench rounds for regressions.

### 23. One diagnostics query per edit; share the result across validate + auto_check
`single-diagnostics-query-per-edit` · impact **Med** · effort M · risk Med

**What:** Capture the pre-edit baseline once at dispatch (capture_edit_baseline exists, edit_file/mod.rs:1080) and thread the single post-edit get_diagnostics result through to auto_check instead of re-querying. Confirmed: smart path queries diagnostics twice (mod.rs:1088 baseline + 1103 candidate) and auto_check does ANOTHER notify+get on the same file — 2-3 full checkOnSave analyses per single edit.

**Why:** Each get_diagnostics triggers a checkOnSave compile and wait_for_idle blocks on it; doing it 2-3x per edit is the single biggest per-round latency contributor in smart mode. Per-round latency is a bench metric and bounds how many rounds fit in the wall-clock timeout. Best stacked on lsp-disable-checkonsave (which makes each query cheap); together they cut LSP-induced latency substantially. (Lower standalone priority since edit_mode=Fast is the default path, but smart mode is reachable and edit_file's inner repair loop amplifies it.)

**Validate:** Bench mean wall-clock/round before/after (tokens unchanged); assert identical diagnostic content in unit tests so feedback quality holds. Pass rate flat-or-up.

### 24. Collapse the double build_feedback into one LSP poll per fast edit
`collapse-double-build-feedback` · impact **Med** · effort S · risk Low

**What:** Record the revision first using stats from a single build_feedback (or have build_feedback append the just-recorded row without a second poll). Confirmed: replace_range.rs:126-165 and insert_at.rs:93-128 call build_feedback twice, each doing collect_file_diagnostics + collect_project_error_count.

**Why:** Every successful fast-mode line edit pays ~2x LSP latency (the first pass only feeds revision-table stats capturable from one poll). With dozens of edits per task that's seconds of wall-clock per task on the DEFAULT path, adding nothing the model sees. Compounds with the checkOnSave cost; pure latency win, behaviorally invisible. Best paired with lsp-disable-checkonsave.

**Validate:** Measure wall-clock/task on run-benchmark-docker-fast.sh before/after; pass rate must be unchanged. Existing feedback.rs render tests confirm the table still shows the new row.

### 25. Wire history_budget/scratchpad_budget into behavior, or delete them
`wire-or-delete-context-budget-knobs` · impact **Med** · effort S · risk Low

**What:** Either make maybe_compress read config.context.history_budget as raw_budget (and tool-output budgets from config) instead of hardcoded window fractions (1/6, 1/3, 1/4), or remove the four dead fields (snippet_budget/history_turns/history_budget/scratchpad_budget) and their config/info display lines. Confirmed: referenced only by config.rs/info.rs display code, never by src/context.

**Why:** The four fields are printed by `config`/`info` as if they control context but assemble/maybe_compress never read them. An operator hitting overflow on a small model who lowers history_budget gets no effect and no error — they can't tune their way out of the documented starvation without editing source. Wiring them turns a hidden constant into a real A/B knob the bench can sweep; deleting clarifies that compression is window-fraction-based. Misleading surface costs debugging time on the exact failure the bench cares about.

**Validate:** Sweep history_budget on the multi-file bench (window/4 vs /3 vs /2) and chart pass rate vs rounds — turns a hidden constant into an A/B knob without per-run code changes.

### 26. Wire real token accounting into the default harness via /metrics snapshots
`real-token-accounting-in-bench` · impact **Med** · effort S · risk Low

**What:** Snapshot llama-server /metrics (prompt+completion tokens) before/after each attempt in run-benchmark-docker.sh and write tokens_in/out to the summary (the comparison-harness design already specifies this; port it to the single-task runner). Today run-benchmark-docker.sh extracts NO tokens, only rounds/attempts/wall.

**Why:** Token cost is an EXPLICIT efficiency goal but is essentially unmeasured, so every token-cheapening idea here (dedup-guidance, gate-spawn-agents, search-output-budget, repo-map shrink) is currently invisible and unjustifiable on the trusted metric. A small model is latency-bound: fewer prompt tokens/round = faster rounds = more rounds within the timeout = higher completion. This is a prerequisite that unlocks validating most of the token-saving ideas above.

**Validate:** Add the snapshot, re-run an existing A/B (repo_map on vs off) and confirm you can see the token delta alongside unchanged pass rate — proving the meter works without altering the agent.

### 27. Make the bench statistically honest: N>=5 runs per config, report distribution not best-of-3
`bench-n-runs-statistical` · impact **Med** · effort S · risk Low

**What:** Wrap run-benchmark-docker.sh in a loop running the same config N times in fresh containers; report mean smoke-pass-rate + variance instead of one best-of-attempts number (lift average_metrics/print_summary from bench-common.sh into the docker runner).

**Why:** Every empirical conclusion rests on best-of-3 on ONE task with binary 0-6 outcomes, and the docs show high variance (Gemma strict+flat = {0/6,3/6,3/6}). A single 6/6-vs-5/6 result is one check flipping — within noise for a stochastic 24B. Without N you can't distinguish a real +1 from a coin flip, so good ideas get rejected and bad ones shipped. This is a prerequisite that protects every other A/B in this list (and could retroactively re-test whether the strict-is-default decision survives at N).

**Validate:** Self-validating: re-run strict-vs-off at N=8 and check whether the 6/6-vs-5/6 gap survives. If distributions overlap, the harness itself flags that past decisions need more N.

### 28. Reactivate the stork large-repo Java task as a second standing benchmark
`reactivate-stork-bench` · impact **Med** · effort M · risk Med

**What:** Port bench-task-stork-weighted-rr.sh (250-file smallrye-stork, mvn compile + structural checks) onto the current docker harness with the validated 60K/4K/strict config and the [tools] knobs, and run it alongside the self-task.

**Why:** The self-task is small enough to brute-force by grepping 'system_prompt', so the agent's navigation machinery (repo map, PageRank, find_references) — a large fraction of the codebase aimed at helping small models scope big repos — is NEVER stressed by the default bench. A second real multi-module task turns single-task overfitting into a 2-point signal and is the only way to validate the discovery-tooling ideas (rebuild-pagerank, auto-list-callsites cross-file). If a decision that helped the self-task hurts stork, that's a finding the single-task bench cannot produce.

**Validate:** It IS the bench addition. Establish a baseline at N>=5, then re-run shipped decisions (strict vs off, repo_map on/off, refactor visibility) against it.

### 29. Add ceremony/flat/refactor-visibility toggles to the docker harness
`ceremony-refactor-toggle-matrix-in-bench` · impact **Med** · effort S · risk Low

**What:** Extend run-benchmark-docker.sh generate_config to emit [tools].ceremony and [tools].flat, and add variants (strict+grouped, strict+grouped-without-refactor, strict+flat) so the harness ablates the exact knobs the empirical program argues about — instead of relying on offline probes the docs say mispredict the bench.

**Why:** The refactor DSL is the tool built for cross-file value-threading (the smoke-ceiling root cause) yet is hidden for Devstral (schema confusion) and ignored by Gemma. Today the load-bearing 'rich grouped surface incl refactor is essential' claim can only be re-tested via probes the team distrusts (tiered-agent-design.md:48). A toggle matrix lets it be re-verified directly on the trusted metric — and tests the auto-list-callsites idea against whether the DSL itself helps.

**Validate:** Self-contained: variants run in the existing harness with different config emission. Compare smoke-pass across strict+grouped / strict+grouped-no-refactor / strict+flat at N>=5 per model.

### 30. Adopt --continue in the docker retry path so the agent keeps plan/scratchpad/trajectory
`bench-continue-on-retry` · impact **Med** · effort S · risk Med

**What:** Pass --continue on attempts 2+ in run-benchmark-docker.sh (as bench-common.sh:321-326 already does) and stop wiping logs/state between attempts; keep the compiler-error retry message but let the model resume with prior plan/scratchpad intact.

**Why:** Each retry restarts miniswe cold with only a bash-composed error string, discarding 12+ rounds of codebase understanding the model spent its budget building (experiment-results-20260405.md:66 shows it then timing out re-deriving from scratch). For a small model, re-establishing context is the expensive part; preserving plan+scratchpad lets attempt 2 spend its budget on the actual fix — directly improving the documented weakest link (test call-site fixes not completed within remaining time). Note: also makes the docker harness match the non-docker one, removing a silent divergence.

**Validate:** A/B with/without --continue at N>=5 on Devstral (times out on retries). Compare attempt-2 pass-rate and wall time; use bench-peek.sh trajectory to catch the risk that stale plan/scratchpad causes thrash.

### 31. Skip the relocation YES/NO round-trip for a unique byte-exact relocated match
`skip-relocation-confirm-on-unique-exact` · impact **Low** · effort S · risk Low

**What:** In try_relocate_and_replace (apply.rs:357-401), when exactly ONE byte-exact candidate exists and differs from the declared scope, auto-apply and NOTE the relocation instead of calling request_relocation_confirmation. Keep confirmation only for whitespace/fuzzy matches or multiple candidates.

**Why:** Whitespace drift / stale line numbers in a model-emitted OLD block are common; the rescue is good for correctness but a unique byte-exact relocated match is unambiguous, so the YES/NO inner-model call adds latency without information. Saves one inner-model round on the common 'correct OLD, stale line number' case. Smart-mode only, so lower priority given the Fast default.

**Validate:** Run run-benchmark-docker.sh (smart mode); compare 'literal:relocate_confirm' log occurrences and pass rate. Pass rate should hold; confirm calls drop for byte-exact cases.

### 32. Short-circuit the smart-mode repair loop on a byte-identical repeated failure
`shortcircuit-smart-repair-on-identical-failure` · impact **Low** · effort S · risk Med

**What:** Detect when an attempt produces the SAME failed step + same failure_reason + same error count as the prior RepairContext (mod.rs:692) and short-circuit to FAILED after 2 identical rounds instead of burning the full 4+2 budget.

**Why:** A stuck small model re-emits the same broken plan 4-6 times; short-circuiting on byte-identical repeats frees the outer agent to try another approach sooner, saving tokens/latency exactly where it's already failing. Short-circuit ONLY on byte-identical repeats (not merely non-improving) so the existing improving-but-slow extra-attempt credit still protects converging runs. Smart-mode only.

**Validate:** A/B run-benchmark-docker.sh: total inner-model calls per task and pass rate.

### 33. Add raw:true to read_file (skip compression) + make start/end ranges strict
`read-file-raw-option-and-strict-ranges` · impact **Low** · effort S · risk Low

**What:** Add an optional `raw`/`verbatim` bool to file(action='read') that bypasses compress_for_reading and emits every source line with the gutter (default stays compressed). Separately, replace read_file's inline as_u64().unwrap_or(default) (read_file.rs:59-68) with opt_u64 so a mistyped range gives a correctable error instead of silently reading the whole file. Also fix the truncation-count to report visible (Some) remaining lines, not raw source positions.

**Why:** compress_for_reading drops std imports and collapses blank lines, showing gaps — so a model issuing a line-anchored Fast edit against a line it never saw, or assuming contiguous numbering across a gap, mis-targets (a recipe for edit-fail-retry loops). raw gives byte-exact source before an edit. Strict ranges: a mistyped start_line:'40' silently reads the whole file (the over-large read ranges were meant to avoid) with no signal to fix it, unlike every other numeric arg. Low individually; bundle as a combined read-path A/B.

**Validate:** Compare edit-failure/old_string-match-failure counts and pass rate on edit-heavy tasks with raw available vs not; grep logs for non-integer range args and measure full-file-read incidence before/after.

### 34. Stop models misrouting replace_range/insert_at through the file group
`fix-route-edit-primitives` · impact **Low** · effort S · risk Med

**What:** Either (a) expose replace_range/insert_at as file actions to match the grouping mental model, or (b, safer) add the dispatch recovery hint proactively to the prompt's tool contract: 'replace_range/insert_at are TOP-LEVEL tools — call them directly, not via file(action=...)'. Confirmed: dispatch.rs:137-149 has a dedicated recovery error proving models make this mistake.

**Why:** The grouping teaches 'file operations go through file(action=...)' but the structurally-similar edit primitives are top-level — an inconsistent boundary a small model generalizing 'edits go under file' will trip, costing a wasted round + a confused-state history entry each time. The hand-written recovery only fires AFTER the failed call; prevention is free. Option (b) is prompt-only (doesn't touch the validated tool surface) so it must clear the same smoke bar but carries less surface risk than (a).

**Validate:** Grep bench transcripts for 'is a top-level tool, not a file action' to baseline frequency, then A/B option (b); measure rounds-to-completion and that error's drop. Hold Qwen 6/6 as the revert gate.

### 35. Move config/setup prose out of always-present schemas into help actions
`relocate-config-prose-to-help` · impact **Low** · effort S · risk Low

**What:** Trim the web tool description (definitions.rs:128, ~68 tok of Serper-key setup) to a one-liner pointing at web(action='help'), which already carries the setup text; audit other schemas (code group provider listings) for similar dead-weight config advice.

**Why:** Setup advice the model can't act on mid-task ('put your key in ~/.miniswe/serper.key') is dead weight in every round across the 12-tool list. web is rarely on the coding critical path, so moving it to on-demand help keeps it discoverable while reclaiming per-round attention budget. Token saving is deterministic (count the assembled JSON), so it can be validated without a model.

**Validate:** Token-count assembled tool JSON before/after to confirm the saving, then run the bench to confirm pass rate flat. Regression risk minimal (help text unchanged, still reachable).

### 36. Shrink/refresh the repo map once a plan exists or the first edit lands
`shrink-repo-map-after-first-edit` · impact **Med** · effort M · risk Med

**What:** Drop the repo map to a small budget (~1500 tok) or refresh it filtered to only touched files once a plan exists / first edit lands, instead of re-shipping the full 5000-token frozen block every round. A/B-able by gating RepoMapProvider budget on plan_exists.

**Why:** The repo map is the single largest per-round block (~35-45%, 5000 tok) and is frozen in messages[0] for the whole turn (only rebuilt on plan flip under strict). It is task-agnostic once the model has read the files it needs — every one of 50-100 rounds re-pays 5K tokens for an overview the model no longer reads. Potential saving ~3.5K tok x every round after the first few. Med risk: removing orientation context could regress tasks that re-consult the map; pairs naturally with rebuild-pagerank (refresh-not-just-shrink) and needs real-token-accounting to measure the win. NOTE: cutting context aggressively was refuted before (Devstral context-tightening caused 0/6), so this must be smoke-gated carefully.

**Validate:** A/B repo_map_budget 5000 vs 1500-after-plan on Devstral+Gemma+Qwen at N>=5; measure pass rate, tokens/round (needs the token meter), and rounds. Treat any smoke regression as a revert (the context-starvation refutation is the cautionary precedent).

### 37. Individually mask/age old tool results instead of waiting for the 13K global threshold
`individually-age-tool-results` · impact **Med** · effort M · risk Med

**What:** Replace old read/search/diagnostic results with a one-line pointer (path + 'see archive') after they scroll past the last 2-3 rounds, rather than waiting for maybe_compress to fire at raw_budget ~12.9K. The dead compress_history (mod.rs:556) was meant to do exactly this.

**Why:** Below ~13K tokens nothing is compressed, so a 1250-tok file read or cargo dump sits in full for the rest of the turn and re-ships every round — on a 20-round task that's ~20 x 1250 tok of stale output. Aging individual results saves the most on long (20+ round) tasks, which are exactly the ones at risk of running out of window. Med risk: aggressive masking could drop content the model still needs (the REPL's two-system masking already shows the hazard), so it must keep a read-back pointer and be smoke-gated.

**Validate:** A/B on long multi-file tasks; instrument peak tokens and count re-reads of already-seen files. Needs the token meter. Watch pass rate for over-aggressive masking.

### 38. Fire the per-edit plan nudges once (or decay them) instead of on every edit
`decay-per-edit-nudge-tax` · impact **Low** · effort S · risk Low

**What:** PLAN_PROGRESS_NUDGE (~47 tok) is appended to EVERY successful edit and persisted to history (run.rs:994), accumulating ~470 tok over a 10-edit task that re-ships every round; CHECKPOINT (~68 tok) every 5 edits. Gate them behind 'fire once' / decay, and skip pushing hints the model can't act on (loop hint before an abort at run.rs:613; the wrap-up warning the model has no budget to honor).

**Why:** Strict-only ceremony with a measurable, compounding token cost and no per-edit benefit (the value-threading carry comes from step-decomposition, not the nudges per MEMORY/tiered-agent-design). Removing dead pre-abort hints also stops polluting the final compressed context with instructions the model never sees, and avoids framing the last productive rounds as 'wrap up' while real steps remain.

**Validate:** A/B with fire-once nudges on the strict bench; measure tokens/round (needs token meter) and pass rate — should be flat-or-up since the nudge content is preserved on first fire.

### 39. Make the REPL router cheap for CODING + feed it bounded history
`router-cheap-prefilter-and-history` · impact **Low** · effort S · risk Med

**What:** Add a fast local pre-filter: if the message contains an imperative/edit verb (add/fix/implement/refactor/change/write/remove/rename) skip the LLM classifier and go straight to CODING; only call the LLM for the ambiguous remainder. Also feed a bounded tail of conversation_history into the classifier so corrections ('actually, just explain it') route against real context (today it's history-blind, repl.rs:71).

**Why:** Every REPL turn pays a full extra prefill (a ~200-tok classifier prompt sharing no KV-cache prefix with the turn) before the real turn starts — paid 100% of the time to occasionally save work, and on thinking-mode models it silently always falls back to CODING (pure overhead). A keyword pre-filter removes the cost for the unambiguous majority while keeping the LLM tiebreaker and the fail-safe asymmetry. Low impact because the bench can't see the REPL at all (run.rs has no router), so this is UX-only until an explore flag exists on run.rs.

**Validate:** Reuse repl-router-classifier-probe.py: confirm the pre-filter never produces a dangerous CODING->EXPLORE on the probe's CODING cases and measure how many cases skip the LLM. End-to-end only measurable if the explore-mode run.rs flag is added.

### 40. Give subagents the parent's loop detection + premature-exit handling + explicit empty-output signal
`harden-subagents` · impact **Low** · effort M · risk Med

**What:** Factor loop-detection + premature-exit out of run.rs into a shared helper and call it from run_single_subagent (subagent.rs:119, currently 30-round cap with NO loop detection or nudges). Have format_outputs emit '[no output — agent stopped at round R without producing a result]' when content is empty.

**Why:** spawn_agents amplifies small-model failure: subagents loop unchecked for 30 rounds and return empty strings the parent can't distinguish from success, treating a silent spinout as a completed subtask. Hardening + surfacing failure lets the parent retry/fall back. Low priority on the DEFAULT deployment because spawn_agents should be gated off entirely at llm_concurrency=1 (see gate-spawn-agents-on-concurrency) — only matters once concurrency>1 is in use.

**Validate:** A/B the shared-helper refactor for regressions on the normal suite first (it touches the hot loop); primary metric pass rate on any decomposition-prone task.

### 41. Unify result-header framing across tools
`unify-result-headers` · impact **Low** · effort S · risk Med

**What:** Standardize the bracketed header shape (lowercase tag + key facts, e.g. '[tool key=val]') across read_file/search/shell/web/code_intel — web search emits uppercase [SEARCH:...] while code search emits lowercase [search ...].

**Why:** Small quantized models pattern-match aggressively on result framing; inconsistent casing/shape makes it harder to reliably locate the file:line payload to anchor the next edit and inflates few-shot drift. BUT this is cosmetic, and the empirical culture cuts hard against unvalidated formatting changes (strict>lean and strict+flat-refuted were both surprises) — header churn can also hurt, so it MUST be smoke-tested and reverted if neutral. Lowest priority: speculative, must clear the same 6/6 bar as any ceremony tweak.

**Validate:** A/B the full bench; watch pass rate and revert if neutral/negative.

### 42. Add grammar-constrained / json_schema tool-call mode behind a config flag
`grammar-constrained-tool-calls` · impact **Med** · effort L · risk High

**What:** When targeting llama.cpp/vLLM, optionally send a GBNF grammar or response_format json_schema derived from the active tool definitions so the server constrains generation to valid tool-call JSON. Gate by provider + a tool_call_format::Grammar variant; keep Auto as fallback.

**Why:** Makes malformed tool-call JSON structurally impossible at decode time — the single biggest small-model failure class this codebase currently repairs after the fact (normalize_xml_tool_calls, repair_leaked_args). Could cut format-error wasted rounds to ~0 for compliant servers. High risk/effort: grammar can constrain the model into degenerate/empty calls, some servers ignore/misimplement it, and it must be verified per-server. The docs already settled that real tool-calls beat XML, so this strengthens the winning path. Lower-rank than plumb-min-p, which attacks the same class far cheaper — try min_p first.

**Validate:** A/B grammar-on vs current Auto per model. Measure malformed-call rate (target ~0), pass rate, tokens. Verify per-server; keep Auto fallback.

### 43. Expose an EXPLORE-equivalent flag on run.rs so the router thesis becomes benchable
`explore-mode-bench-flag` · impact **Low** · effort M · risk Low

**What:** Expose the EXPLORE config (plan=false, ceremony=Off, read-only tools, investigation directive) as a one-shot mode/flag on run::run, and add a bench variant running the read/explain subset through it. Today the router lives only in repl::run and is structurally invisible to the bench (which calls miniswe --yes -> run.rs).

**Why:** The routing/read-only thesis is currently untestable on the metric the project trusts — it was validated only by a standalone classifier-accuracy probe, never end-to-end, and CANNOT be with the current entry-point split. A flag lets you measure whether read-only+no-ceremony actually lowers tokens/rounds for investigation-shaped tasks vs the full coding path, turning an unvalidated feature into an empirical one (and is a prerequisite for the explore-mode-runtime-readonly-guard's git-diff-empty assertion). Low coding-impact directly; its value is unblocking measurement.

**Validate:** Add a --mode=explore bench column and run read/explain tasks both ways; compare tokens/round and answer quality. CODING tasks bypass the flag and must be unchanged.

