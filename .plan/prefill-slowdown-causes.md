# Why some runs are 3-5x slower: causes for full-context re-prefill

Investigated 2026-08-25 while benches were paused. All offline — llm_dumps +
preserved llama-server logs. No GPU used.

## 0. The one number that explains wall time

Wall time tracks **prompt-eval tokens**, nothing else:

| run (start)  | wall  | server busy | idle% | prefill tokens | tasks >15k prefill | cancelled     | cache_prompt=false |
|--------------|-------|-------------|-------|----------------|--------------------|---------------|--------------------|
| 08-24 16:29  |  298s |  148s       |  50%  |     80,863     | 1                  | 0             | 0 |
| 08-24 16:38  |  422s |  240s       |  43%  |     96,730     | 1                  | 1             | 0 |
| 08-24 16:50  |  304s |  181s       |  41%  |    109,104     | 1                  | 0             | 1 |
| 08-25 09:36  | 1599s |  776s       |  51%  |    588,514     | 9                  | 18            | 7 |
| 08-25 11:06  | 1671s | 1075s       |  36%  |    444,551     | 7                  | 3             | 3 |
| 08-25 10:05  | 3423s | 2019s       |  41%  |  2,321,086     | 69                 | 41            | 5 |

(Logs are named for when the server was *killed*; mapping verified by task count
vs dump count: 139->162, 360->364, 133->175.)

Two things fall out immediately:

* **Idle fraction is a constant ~36-51% in fast AND slow runs.** Client-side
  harness overhead is NOT the variable. Do not chase it.
* **The unit of waste is one full-context re-prefill.** On Laguna XS at ~44k
  tokens that is 36-39s of pure prefill for a 64-token answer. Fast runs pay it
  once (the initial prompt). Slow runs pay it 7-69 times.

Client-side signature of the wasted call (09:36 run, calls 89-109): gap 35-38s,
output 20-52 tokens, prompt 98% byte-identical to the previous one, no sub-role
call in between. Steps are discrete: 2s -> 10s -> 36s while the prompt grows by
only 150 tokens. That is not compute scaling; that is a cache hit turning into a
cache miss.

## 1. CORRECTED — cause A is the loop-breaker, NOT the tool-call-leak retry

**Earlier writeups of this file said `has_tool_call_leak()` at `src/llm/mod.rs:254`.
That was wrong.** `has_tool_call_leak` never fired in any of these runs. Grepping
stderr, all 7 events are:

```
[loop] forcing a cold prompt eval (cache_prompt=false) to break the KV-cache loop
```

i.e. the agent's own loop detector in `src/cli/commands/run.rs:1341`. Two triggers
set `force_cold_prefill_next_round`:

* **soft** (`run.rs:2145`) — same `call_key` >= `COLD_PREFILL_FREQ` (4) times in a
  12-entry window. Fired **1 of 7** times.
* **hard** (`run.rs:2170`) — `same_call_streak >= 3 || cycle.is_some()`. Fired
  **6 of 7** times. Its comment claims it is "the cheapest, most reliable breaker
  (the loop is q4-KV-cache-induced; a fresh eval proceeds)".

### It does not work: 7 fires, 0 breaks

```
  #    clock    gap  cold  produced
 88 09:42:06   10.5        plan {"action":"check","step":16}
 89 09:42:16   34.9  COLD  plan {"action":"show"}
 90 09:42:51   37.2        plan {"action":"check","step":16}
104 09:50:31   35.4  COLD  plan {"action":"check","step":16}
107 09:52:19   36.2  COLD  plan {"action":"check","step":16}
118 09:54:24   38.1  COLD  plan {"action":"check","step":16}
130 09:56:44   37.9  COLD  plan {"action":"check","step":16}
133 09:58:38   38.7  COLD  plan {"action":"check","step":16}
136 10:00:35   39.3  COLD  plan {"action":"check","step":16}
```

Six of seven produced the identical call on the very next request; #89's break
reverted at #90. It also can never escalate: the promotion to
`force_compact_next_round` is gated on `key_is_file_edit(&call_key)`, and `plan`
is not a file edit, so it can fire indefinitely at ~37s a shot.

### And it is not the cost driver

The non-COLD calls in the same stretch cost the same 36-39s (#90 = 37.2s with no
cache bust). The 7/2/7-vs-0/0/0 separation is real correlation but wrong
causation: cold fires and slowness are both downstream of the model being stuck
at 44k ctx, not of each other. Note the perverse detail that
`COLD_PREFILL_IDLE_SECS = 60` widens the stream idle window *only* when
`cache_prompt=false` is set, so the cold calls are the only ones protected from
section 3's cancellation.

Real cost of this mechanism: 7 rounds x ~1-2s of marginal prefill, plus 7 wasted
rounds. Worth removing on effectiveness grounds (0/7), not on speed grounds.

## 1b. CONFIRMED (probe, 2026-08-25 14:2x) — any large rollback = FULL re-prefill

Controlled probe against a live Laguna XS (`scratchpad/kvprobe.py`), 54k-token
prompt:

| request                                   | prefilled   |
|-------------------------------------------|-------------|
| pure extension (+1 line)                  | **22 tok** / 0.10s |
| identical re-send                         | **1 tok** / 0.02s  |
| one word changed at 98% through           | **53,891 tok** / 42.0s |
| one word changed at 50% through           | **53,891 tok** / 44.9s |

The server's own decision, from the log:

```
f_keep = 1.000  ->        22 tokens prefilled
f_keep = 0.999  ->         1 token
f_keep = 0.977  ->    53,891 tokens   (FULL)
f_keep = 0.490  ->    53,891 tokens   (FULL)
```

It reports a keep fraction and then does not honour it beyond a small tail. This
is the signature of sliding-window attention (start-laguna-xs.sh: "10 global
layers cache full context") — positions older than the window cannot be rolled
back to, so llama.cpp clears the sequence.

**`f16` KV is byte-for-byte identical to `q8_0` here (probe re-run with
`MINISWE_KV_TYPE=f16`): KV quantization is REFUTED as the cause.**

### What makes the prompt diverge at all

The `[CURRENT STATE]` / `[PLAN]` block is appended to the *newest* tool message
each round and stripped from the previous one when it moves forward. It is a
**sliding block that mutates an already-cached message**, so the prompt is almost
never a pure extension:

```
run                         main calls   pure ext   diverged
08-25 09:36 Laguna slow           92         25        66  (72.5%)
08-25 12:51 Laguna fast          102         13        88  (87.1%)
08-24 16:38 Laguna fast           93         61        31  (33.7%)   <- no plan block
```

Divergence is universal in fast and slow runs alike, so it is a **constant tax
whose price scales with context**, not a slow-run-only defect. Measured rollback
distance in the 09:36 run: 25 rounds pure, 61 rounds 257-1024 tok, 5 rounds
1025-4096 tok, none beyond. The 61 rolled back fine (median 809 tok prefilled).

**Still unresolved:** in the real run, 16 tasks did a full prefill from zero
without `cache_prompt=false` and without a rollback >4096 tok. The controlled
cliff measurement (`scratchpad/cliff.py`) is meant to pin the rollback threshold
that explains them.

## 2. CONFIRMED cause B — single KV slot, evicted by sub-role calls

`start-laguna-xs.sh` passes no `--parallel`, so `-np 1`, one slot,
`kv_unified=false`. Every sub-role call (refactor edit-applier, action
summarizer, `stuck_check`) carries a different system prompt with a small
(4-17k) prompt, takes the slot, and overwrites the main conversation's KV.
Returning to the main loop re-prefills the whole thing.

This dominates where cause A does not: the 10:05 run has **69** >15k prefills but
only **5** forced cold prefills — the other ~64 are evictions. (Established
earlier from the server logs: every full re-prefill is immediately preceded by a
short-prompt request.)

Consecutive sub-role calls in one burst cost only one re-prefill on return, so
the number of *bursts* drives wall time, not the number of sub-role calls.

Why 08-24 was fast and 08-25 slow: the fast 08-24 runs hand-edited and never
invoked `refactor add_param` at all (0 edit-applier calls). The 08-25 refactor
fixes make add_param complete instead of refusing -> more edit-applier bursts.

## 3. RESOLVED — the "0-token tasks" are client cancellations

They are not 0-token. They are requests **cancelled mid-prefill by the client**:

```
19.55.956  launch_slot_: task 17721 | processing task
19.59.912  prompt processing, n_tokens = 6144, progress = 0.14   <- prefilling from ZERO
20.25.958  W srv stop: cancel task, id_task = 17721              <- exactly 30.002s after launch
20.27.202  release: task 17721 | n_tokens = 38912
20.27.203  get_availabl: f_sim_best = 0.890, f_keep = 1.000
20.33.650  task 17742 | prompt eval = 5221 ms / 4831 tokens      <- retry resumes from partial KV
```

`srv stop: cancel task` lands at 30.002s after launch, every time — that is
`stream_idle_timeout_secs = 30` firing against a prefill that needs 33-39s.
Prefill emits no stream bytes, so the idle timer cannot distinguish "slow" from
"dead". The counts match the mystery exactly: **18 / 3 / 41 cancels vs the
18 / 3 / 41 unexplained tasks**. No `--verbose` run needed.

The partial KV survives the cancel, so the retry finishes in 5-9s and the
*marginal* cost is only ~1-2s. But it burns one of `max_retries = 3` per
occurrence and is a latent spiral: a prefill needing >60s would cancel twice.

Correction to the previous note here: the claim of "0 idle timeouts" was based on
grepping the harness stderr, which does not log this path. The server does.

## 4. Still-theoretical causes, not yet excluded

Ordered by how cheaply they can be tested.

1. **llama.cpp RAM prompt cache thrash.** Build 10524 keeps evicted prompt
   states in host RAM and restores by LCP similarity. Two live conversations
   (main ~44k, stuck_check ~5-7k) alternating on one slot may exceed the
   `--cache-ram` budget, so restore fails and it re-prefills instead. Test: set
   `--cache-ram` explicitly large and count >15k prefills.
2. **CPU contention on the `--n-cpu-moe 6` layers.** Six layers of experts run on
   CPU with `--threads 16`, inside the same host as a container running cargo
   builds and rust-analyzer. This would lower prefill *throughput* (tok/s), not
   the *count* of prefills — so it is a multiplier on causes A/B, not a trigger.
   Test: compare `prompt eval` tok/s during a cargo build vs idle.
3. **Chat-template non-determinism shifting the prefix.** Templates that keep
   tool definitions or reasoning blocks only in the last turn rewrite the middle
   of the prompt each round. Our client-side prefix is 98% stable, but that is
   measured pre-template. Test: hit `/apply-template` with two consecutive dumps
   and diff the rendered strings.
4. **Page-cache eviction of the mmapped GGUF.** cargo writes GBs to target/,
   which can evict the model's host-resident pages, stalling the CPU-offloaded
   layers. Test: watch `nvidia-smi` + `vmstat` during a slow phase.
5. **GPU power/thermal throttle** at the 200W cap. Would be gradual, not the
   observed 2s->10s->36s step. Low prior. Test: log `clocks_throttle_reasons`.
6. **Context shift at n_ctx.** Real risk at `CTX_SIZE=60000` but not the cause
   here — `max_tokens` is a constant 8000 and the *fast* runs peaked higher
   (49.7k prompt, 2.3k headroom) than the slow 09:36 run (37.8k) and stayed fast.

## 5. Refuted, with the evidence

* **Context length per se** — 12:51 ran 2.6s median at 30-40k; 09:36 ran 35.8s
  median at the same bucket on the same server.
* **Client-side prefix breaks** — 98% byte-level prefix reuse on all 24 slow
  calls; system-prompt hash unchanged; tool-list hash unchanged.
* **n_ctx overflow / context shift** — see 4.6.
* ~~Client timeout + retry spiral~~ — WRONG, see section 3: 18/3/41 server-side
  cancellations at exactly 30.0s. The harness stderr does not log this path.
* **Client-side dead time** — idle% is ~36-51% in fast and slow runs alike.
* **Hardware** — idle decode ~130 tok/s in both eras; prefill throughput a
  constant ~1000-1270 tok/s.
* **Output volume** — the slow calls emit 20-64 tokens.
* **Agent thrash alone** — the 09:36 model does loop on `plan check step 16`, but
  the same call costs 2s early and 36s late. The loop wastes rounds; it does not
  make a round slow.

## 6. What to do when the GPU is free

Cheapest first, each one run:

1. `--parallel 2 --kv-unified` so sub-roles get their own slot without splitting
   n_ctx. Directly targets cause B. **Changes the perf baseline for every model —
   do not do it mid-round.**
2. Route sub-role calls (stuck_check, summarizer, edit-applier) to a second
   llama-server on another port, or make them reuse the main system prompt.
   Same target, no server-flag risk.
3. For cause A: on leak, retry with a *trimmed* prompt rather than
   `cache_prompt=false`, or accept the leak and repair it client-side
   (`tool_call_repair.rs` already exists). 37s to re-read 44k tokens to fix a
   64-token answer is a bad trade.
4. ~~One `--verbose` run to identify the 0-token tasks.~~ DONE — see section 3.
5. Raise `stream_idle_timeout_secs` (or scale the idle window with prompt size)
   so a 33-39s prefill is not cancelled at 30.0s.
6. Stop the `[CURRENT STATE]` block from sliding: if history were append-only the
   prompt would be a pure extension and prefill would drop to ~20 tok/round.
