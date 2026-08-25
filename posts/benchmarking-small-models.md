# Benchmarking small local models as coding agents

After [making miniswe 2.5× faster](https://michalszynkiewicz.dev/blog/miniswe-2_5x-faster/), I had something more valuable than a faster tool: a benchmark I trusted. So I pointed it at every interesting small model that fits on my RTX 3090 (power-capped to 200 W).

Six models went in. Two came out with a 100% record. Two got dropped entirely. And one of them turned out to have a bug in the vendor's own chat template.

## Methodology

The task is real work on a real codebase: add a `--system-prompt-override` CLI flag to a pinned version of miniswe itself. That means a clap flag, threading the value through a few layers of calls, and updating every call site — including the test crate.

Six checks decide the score:

* the code compiles
* the binary builds
* the flag shows up in `--help`
* the flag parses
* the tests pass
* **smoke**: the binary must actually answer through the new flag

Smoke is the only check that proves the feature works end to end. A 5/6 with smoke failing usually means "the flag exists but is wired to nothing".

The rules: 57-minute timeout, up to three attempts (each attempt gets a fresh context; the working tree carries over). Everything runs headless in Docker. llama-server is restarted between runs — a long-running server is an uncontrolled variable. The GPU is a 24 GB RTX 3090 held at a 200 W power cap for every run, so the tok/s numbers below are lower than an uncapped card would give, but they are directly comparable to each other. And at this model size, never trust a single run: LLM non-determinism is brutal, so I run each configuration several times.

One caveat: miniswe itself evolved during these weeks — every model failure taught the harness something (more on that at the end). The headline table below is the final validation round: all models, same day, same shipped defaults, fresh server per run.

## The contestants

| Model | Architecture | On the card | Decode (tok/s) |
|---|---|---|---|
| Gemma 4 26B A4B | MoE | fully on GPU | — |
| Devstral Small 2 | 24B dense | 19.1 GB, 60k ctx | — |
| Laguna XS 2.1 | 33B-A3B MoE | 17.2 GB (some experts on CPU) | 129 |
| Muse Glimmer 30B | dense, always-thinking | 17.6 GB | 21 |
| North Mini Code 1.0 | ~30B MoE | ~17 GB | ~98 |
| Nemotron 3.5 Lightning | 30B | — | — |

All at Q4 quants, mostly with quantized KV cache. Gemma is the reference model — the one the previous post's numbers were tuned on.

**Sampling.** Every instruct run is at temperature 0.2. Thinking runs are at 0.6 — reasoning traces degenerate at code-task temperatures, so miniswe raises the temperature whenever it enables thinking. That coupling matters for reading this post: a thinking-vs-instruct comparison below is a *two*-variable change, not one, and I have not separated them. Muse Glimmer can't disable thinking at all, so all of its runs are at 0.6 while the models it sits next to in the table are at 0.2.

## Results

The final round:

| Model | Result | Wall time |
|---|---|---|
| Laguna XS 2.1 | 6/6, 6/6 — both first attempt | 422s, 304s |
| Gemma 4 26B | 5/6 after 3 attempts | 2850s |
| Muse Glimmer 30B | 6/6, first attempt | 1277s |
| Devstral Small 2 | 6/6, first attempt | 806s |

And what the full history behind that snapshot says:

**Gemma 4 26B — the reference.** Typical run: 6/6 first try in 5–8 minutes, historical band 279–1411s. The 5/6 above is its one recent blemish, and an instructive one: the production code was correct, but it corrupted its own test file with a non-idempotent `sed` and never recovered. Mostly-6/6 record, fastest converger of the field.

**Laguna XS 2.1 — the surprise.** Five runs, five 6/6, all on the first attempt — the only model besides gemma with a clean first-try record, and its last two runs (422s, 304s) are gemma-class fast. Distinct personality: it leans on shell (grep, sed, python heredocs) over the structured edit tools, writes 23-step plans, and checks every box at the end. 129 tok/s from a 33B MoE with a slice of experts on the CPU.

**Devstral Small 2 — solid.** Finishes 6/6 routinely; best run 806s, older band 1021–2463s. Its early runs stumbled on edit mechanics (one brace-dropping edit re-issued twenty times) — those runs are precisely where several harness guards came from.

**Muse Glimmer 30B — best edit economy, slowest clock.** Thinking can't be disabled, and at 21 tok/s reasoning is a tax: runs are 751–1277s even when clean. But it makes the fewest, most surgical edits of any model here — one run finished the feature with 8 edits and zero reverts. It was also the model most punished by harness gaps: before the stuck-detection nudge it would sit in half-hour read loops (3406s, 2735s); after, three consecutive clean passes.

**North Mini Code 1.0 — dropped.** Cohere's "works best with thinking enabled" is binary here (with the temperature caveat above — the thinking run is also the 0.6 run): instruct mode scored 0/6 (55 minutes re-reading one file, 235 reads, zero edits); thinking mode reached 5/6 at the timeout, one call site short. Even its good mode is 4–6× slower to converge than gemma or Laguna. Not worth the VRAM.

**Nemotron 3.5 Lightning — dropped, with a story.** Six instruct runs, never a single 6/6, never finished inside 57 minutes (4, 0, 5, 2...). Thinking mode — again, also a temperature change — came closer: it produced nemotron's only 6/6 tree, complete at minute 24 — and then spent the remaining half hour announcing it was done while issuing one more tool call, never ending its turn; the bench scored the finished tree at the timeout. In both modes it kept flooding thousands of blank lines mid-edit, burning its whole output budget. I chased temperature, repetition penalty, grammars — all refuted. The root cause: NVIDIA's own chat template prefills `<think></think>` with no trailing newline in non-thinking mode, and the model desperately wants that newline. Adding it: zero floods on exact replays — but the fix only exists for instruct mode; one thinking run drowned in a 13,005-line flood. As far as I can tell, this isn't publicly reported.

**The verdict.** Gemma 4 26B and Laguna XS 2.1 are the top tier — fast, reliable, first-try finishers. Devstral is a dependable third. Glimmer works if you can pay the thinking tax. The rest of the small-model field, at least on this task, isn't there.

## What you actually find when you benchmark models

You think you're testing models. Mostly, you're fuzzing your own harness. Nearly every guard in miniswe today is named after a failure some model produced:

* Glimmer's read loops → a stuck-detection nudge (its runs went from ~3000s to ~1000s)
* Devstral's re-issued broken edit → longer-period loop detection
* Devstral's truncated tool calls spinning 436 rounds → argument caps and call stubbing
* a wedged rust-analyzer silently freezing runs for 40 minutes → bounded LSP writes and a hard tool deadline
* Nemotron's summarizer hallucinating 30k-token changelogs during compaction → output caps and a reject-if-larger guard
* an LLM request that hung for 47 minutes → real request deadlines
* gemma's sed corruption → the next batch: diff-echo for shell edits, and refusing "done" while the model's own last test run is red

Every model brings a new way to break the harness. That's the real reason to keep adding them.

Next on the bench: Qwen3.8-27B — weights downloaded, launcher written. If you try miniswe with a model I haven't, I'd love to hear how it went.
