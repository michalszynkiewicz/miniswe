# Final round — tally (updated 2026-08-25 15:40)

## Per-run protocol
1. `docker ps --format '{{.Names}}' | grep -iE "llama|cuda" | xargs -r docker rm -f`
2. `./start-<model>.sh` via run_in_background
3. `until curl -sf http://localhost:8464/health >/dev/null 2>&1; do sleep 5; done`
4. `[THINKING=true] ./scripts/run-benchmark-docker.sh --timeout 3400 --max-rounds 600`
5. verify `<rundir>/00_baseline/config.toml`
6. AFTER the run, BEFORE killing: idle speed probe (2x) on /completion
NEVER edit run-benchmark-docker.sh while a run is in flight — bash re-reads the
file by byte offset; run 8 died with "syntax error near unexpected token" at the
results table for exactly this reason (the benchmark itself had already finished).

## Binary generations (READ BEFORE COMPARING WALL TIMES)
- **G0** — before the LSP / stale-anchor / lockout fixes.
- **G1** — those three fixes in (rows marked POST-FIX below).
- **G2** — G1 + `fcd8378`, the 512-token-cliff fix. Wall times are NOT comparable
  across generations for any run whose plan block crossed ~1778 B; see the
  contamination scan below.

## Results
| # | Model | Mode | Result | Wall | Rounds | Decode | defect | verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | Gemma 4 26B | instruct | 6/6 a1 | 419s | 166 | 94.2 | none | KEEP |
| 2 | Gemma 4 26B | instruct | **6/6 a1** | **278s** | **78** | **83.6/87.7** | none (POST-FIX binary) | REPLACED 12:20 — was 5/6 3402s |
| 3 | Gemma 4 26B | thinking | 6/6 a1 | 456s | 119 | 94.2 | none | KEEP |
| 4 | Gemma 4 26B | thinking | 6/6 a1 | 625s | 124 | 86.0 | none | KEEP |
| 5 | Laguna XS | instruct | 6/6 a1 | 1599s | 253 | 130.6 | none (slow — KV-slot eviction, root-caused below) | KEEP |
| 6 | Laguna XS | instruct | **6/6 a1** | **1550s** | **331** | **130.4/129.8** | none (POST-FIX) | REPLACED 12:51 — was 4/6 timeout 3423s |
| 7 | Laguna XS | thinking | **6/6 a1** | **516s** | **127** | **130.9/130.5** | none (POST-FIX) | REPLACED 13:00 — was 6/6 1671s w/ stale anchor + sed |
| 8 | Laguna XS | thinking | **6/6 a1** | **481s** | **167** | **130.3/130.4** | none (POST-FIX, zero sed) | REPLACED 13:14 — was 6/6 a2 1887s w/ PARTIAL 15/16 + sed |
| 9 | Muse Glimmer 30B | thinking | 6/6 a1 | 657s | 77 calls | 21.5/21.1 | none | KEEP |
| 10 | Muse Glimmer 30B | thinking | 6/6 a1 | 571s | 67 calls | 20.1/19.3 | none | KEEP |
| 11 | Laguna XS | instruct | **6/6 a1** | **514s** | **375** | **130.1/129.0** | none (**G2**, cliff-fix verification) | KEEP |

### Run 11 — the cliff fix, verified live (15:24-15:34)
Same model/arm/config as rows 5 and 6. Defect scan clean. Decode 130.1/129.0 =
normal band, so no hardware confound.

| run | main calls | sub-role | median block | rounds >cliff | wall | result |
|---|---|---|---|---|---|---|
| 10:05 (G1) | 124 | 9  | 1936 B | **103** | 3423s | 4/6 |
| 15:24 (G2) | 142 | 1  | **2270 B** | **120** | **514s** | **6/6 a1** |

Run 11 is the *same regime as the 3423s failure, only worse* — bigger plan, more
over-cliff rounds — and it paid 2 full re-prefills instead of 103 (109,005
prefill tokens over 145 events, matching the 08-24 FAST run's 109,104 while
doing 5x the rounds). Block parked behind the tail in 117 of 121 block-bearing
rounds; only 4 relocations.

**Do not over-claim it against rows 5/6.** Those had median blocks of 647 B and
1602 B (max 1756 — never crossed), i.e. **0** over-cliff rounds; their cost was
sub-role KV eviction (25 and 47 sub-role calls vs 1 here). The fix kills the
catastrophic tail; it does not explain the whole 1550s -> 514s gap. Sub-role
eviction remains unaddressed.

## Queue after the reruns (all on the fixed binary)
- ~~Laguna XS instruct rerun~~ DONE
- ~~Laguna XS thinking rerun x2~~ DONE (516s, 481s)
- ~~Muse Glimmer 30B x2~~ DONE (657s, 571s, both 6/6 a1)
- Devstral Small 2 x2  <-- NEXT. Run 1 ABORTED 13:44 (started 13:40, ~20 calls in) —
  GPU paused at user's request. Dir docker_20260825_133833_...devstral is a PARTIAL,
  NOT a result: delete or ignore it. Restart this run from scratch when benching resumes.
- gpt-oss-20b x2, high effort only (add --chat-template-kwargs '{"reasoning_effort":"high"}' to start-gpt-oss-20b.sh, thinking=false)
- Mistral Small 4 x2
- Laguna S 2.1 x2 (last — 73 GB, slowest)

## Contamination scan (CORRECTED 15:5x — the cliff is PER-MODEL)

**The 512-token cliff is not a llama.cpp constant — it is the model's own
`attention.sliding_window`.** Laguna XS's metadata says exactly 512, which is
why the probe matched to the token. Read straight from the GGUF files:

| model | `attention.sliding_window` | cliff @3.47 B/tok |
|---|---|---|
| gpt-oss-20b | **128** | ~444 B |
| Laguna XS / Laguna S | **512** | ~1776 B |
| Gemma 4 26B / 31B | 1024 | ~3553 B |
| Muse Glimmer 30B | 2048 | ~7106 B |
| North Mini Code | 4096 | ~14212 B |
| Devstral (mistral3), Mistral Small 4, Nemotron 3.5, Qwen3.8 | *(absent — full attention)* | no cliff |

Re-scanned all 223 runs against each model's own window
(`scratchpad/contam2.py`). **Only 5 runs ever crossed, and all 5 are Laguna XS:**

| run | rounds | median B | over cliff |
|---|---|---|---|
| 20260825_152440 (run 11, G2 — fix absorbed them) | 142 | 2270 | 120 |
| 20260825_100518 (the 4/6 @ 3423s) | 124 | 1936 | 103 |
| 20260824_162908 |  50 | 1821 | 27 |
| 20260824_005124 | 103 | 1740 | 22 |
| 20260823_172958 | 131 | 1610 | 13 |

An earlier version of this table applied Laguna's threshold to every model and
wrongly flagged two gemma runs (`20260825_080756`, `20260824_165637`) and a
Nemotron run. **Those are clean** — gemma's window is 2x wider than the blocks
involved, and Nemotron has no window at all. No gemma/Glimmer/Devstral/Nemotron/
North result needs a rerun on cliff grounds.

**Rebench rule (revised):** nothing needs redoing. The only run the bug plausibly
broke was the 4/6 @ 3423s, already replaced by run 11.

## OPEN — `STATE_REWIND_BUDGET` is hardcoded for Laguna, and gpt-oss is in the queue
`fcd8378` uses a fixed 1500 B budget, correct for Laguna's 512-token window.
**gpt-oss-20b's window is 128 tokens (~444 B)**, so blocks between ~444 B and
1500 B would still relocate every round and blow its cache — the fix would not
engage where it is needed most. Our blocks run 500-2300 B.

`/props` does not expose the window, so it cannot be discovered at runtime; it
needs to be configured. Proposal: a `context.state_rewind_budget` setting
(bytes, default 1500 = today's behaviour), set per model in the launcher/arm
settings, with a sentinel meaning "full attention -> never sticky, always
re-anchor" for Devstral / Mistral Small 4 / Nemotron / Qwen. That also *raises*
the budget for gemma (~3200 B) and Glimmer, removing stickiness those models
never needed. **Decide this before the gpt-oss runs.**

## End of round
- posts/benchmarking-small-models.md: per-model recent-run bands (gemma 5/6s INCLUDED), fill Decode col
- docs/model-scoreboard.md: add rows
- Say explicitly that runs 1/3/4/5 used the pre-fix binary; defensible because the
  three fixes only alter failure paths (no LSP / stale anchor / lockout), none of
  which those runs entered.

## Launch commands (copy-paste, one run at a time)
```bash
# 1. free the GPU
docker ps --format '{{.Names}}' | grep -iE "llama|cuda" | xargs -r docker rm -f
# 2. server (background). setsid = survives a Claude Code restart
setsid nohup ./start-<model>.sh > /tmp/llama.log 2>&1 &
until curl -sf http://localhost:8464/health >/dev/null 2>&1; do sleep 5; done
# 3. bench  (prefix THINKING=true for thinking arms)
setsid nohup ./scripts/run-benchmark-docker.sh --timeout 3400 --max-rounds 600 \
    > /tmp/bench.log 2>&1 &
# 4. verify: grep -E "thinking|compaction|reactive_debugger|stuck_check" \
#       "$(ls -dt benchmark_results/docker_* | head -1)"/00_baseline/config.toml
# 5. after it lands, BEFORE killing the server — idle speed probe x2:
curl -s http://localhost:8464/completion -d '{"prompt":"Write a detailed explanation of how a hash map works, covering buckets, collisions, and resizing.","n_predict":200,"temperature":0.2}' \
  | python3 -c "import sys,json;t=json.load(sys.stdin)['timings'];print(f\"decode {t['predicted_per_second']:.1f} prefill {t['prompt_per_second']:.1f}\")"
```

## Per-run health check (did it hit a known defect?)
```bash
R=$(ls -dt benchmark_results/docker_* | head -1)/00_baseline
grep -a "DEGRADED" "$R"/stderr_attempt1.txt          # fix 1: no LSP
grep -aho 'add_param PARTIAL[^"]*' "$R"/llm_dumps/*.json | sort -u   # fix 2: stale anchor
grep -aho 'already has a parameter named[^"]*' "$R"/llm_dumps/*.json # fix 3: lockout
grep -l '"sed ' "$R"/llm_dumps/*.json | wc -l        # the tell for all three
```

## Run 7 rerun (post-fix): 6/6 attempt 1, 516s, 127 calls, decode 130.9/130.5
Defect scan clean: no DEGRADED, no `add_param PARTIAL`, no lockout. `add_param` never
attempted; the 7 `sed -i` calls were narrow single-line substitutions, not bulk regex.

## Why the 08-25 Laguna runs were slow (investigated, root-caused)
SUPERSEDED/EXTENDED 08-25 13:5x by .plan/prefill-slowdown-causes.md — second
confirmed driver found (forced cold prefill from the tool-call-leak retry,
7/2/7 in slow runs vs 0/0/0 in fast), plus the refuted list. Read that first.
NOT hardware, NOT context length, NOT the harness config (configs byte-identical
apart from a cosmetic `model =` label). Server-side prefill throughput is constant
across both eras (~1000-1270 tok/s).

Cause: **llama-server runs with a single KV slot** (no `--parallel` in
start-laguna-xs.sh, so `-np 1`). Every miniswe *sub-role* LLM call — the refactor
"edit-applier" (`You apply a single localized code edit...`) and the action
summarizer — carries a different system prompt, so it evicts the main
conversation's KV cache. Returning to the main loop re-prefills the WHOLE context.

Evidence (llama-server `print_timing`):
| run | sub-role calls | full re-prefills >15k tok | prefill tokens | prefill s | wall |
|---|---|---|---|---|---|
| 08-24 16:50 FAST | 0   | 1  (40535 tok, 33s)      | 109,104 |  96s |  304s |
| 08-25 09:36 SLOW | 47  | 9  (22k-45k tok, 14-38s) | 588,514 | 531s | 1599s |
| 08-25 11:35 SLOW | 139 | 7  (22k-52k tok, 18-50s) | 444,551 | 436s | 1887s |
Every full re-prefill is immediately preceded by a short-prompt request (91-9214
tok) — the sub-role call that evicted the slot. Cost per eviction ~35s at 45k ctx.
Consecutive sub-role calls in one burst cost only one re-prefill on return, so the
number of *bursts* drives wall time, not the number of sub-role calls.

Why now and not on 08-24: the fast 08-24 runs hand-edited and never invoked
`refactor add_param` at all (0 sub-role calls). Today's fixes make add_param
complete instead of refusing -> more edit-applier bursts -> more evictions.

Ruled out with data: context length (12:51 ran 3.1s median at 25k+ tok on the same
server), client-side prefix breaks (97% message-prefix reuse on the slow calls),
tool-list churn (1 change/run), cargo time (all builds <10s), LSP project
diagnostics (4-10s each, only 7-22 per run).

Possible mitigation (UNTESTED): give sub-roles their own KV slot. `-np 2` splits
n_ctx evenly per slot in llama.cpp, which won't fit 60k x2 here; `--kv-unified` on
build 10524 may allow slots to share one cache. Needs verification before use --
it would change the perf baseline for every model, so do NOT change it mid-round.
