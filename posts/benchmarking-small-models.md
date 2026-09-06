# Miniswe on current local models

It's been 1.5 years since I bought a used RTX 3090 with the intent of running local LLMs.
Roughly a year later, I started `miniswe` - I wanted to figure out whether I could build some smartness around a small model to let it handle real tasks.

My first test model was Devstral Small 2. Dense, rather slow but fitting my GPU.
A month or so later Google introduced Gemma 4, with a 26B MoE model that is crazy fast compared to Devstral on my hardware.

At first it didn't seem as smart as Devstral on my benchmarks, but when I added a few workarounds to `miniswe`, it started working nearly as well as Devstral on my benchmarks.

A few weeks ago I had a bit of time and decided to check what's new on the market for models that I could run locally. It turned out a lot has changed.
I knew I had to benchmark the models to see what I could use as Gemma's replacement.

After a bit of research I decided to bench models from the following families: Gemma, Devstral, Laguna, Muse Glimmer, North Mini, Nemotron and GPT OSS.

The most notable absentee from the benchmark is Qwen3.8. I'd expect strong results from it, but for some of my use cases its origin rules it out.

## Hardware

I'm running the models on:
* RTX 3090 capped to 200 W - an old 24 GB VRAM card
* Ryzen 9950X3D + 128 GB of RAM

## Methodology

The task is real work on a real codebase: add a `--system-prompt-override` CLI flag to a pinned, old version of miniswe itself. That means a clap flag, threading the value through a few layers of calls, and updating every callsite.
The project is not huge, but it is not that easy to work with: a couple of 500–900 line files, Rust code, a few layers of prod code to change, and fourteen callsites in the test crate alone.

Six checks decide the score:

* the code compiles
* the binary builds
* the flag shows up in `--help`
* the flag parses
* the tests pass
* **smoke**: the binary must actually answer through the new flag

Smoke is the only check that proves the feature works end to end. A 5/6 with smoke failing usually means "the flag exists but is wired to nothing".

The rules: 57-minute timeout, up to three attempts (each attempt gets a fresh context; the working tree carries over). Everything runs headless in Docker. llama-server is restarted between runs — a long-running server is an uncontrolled variable.
At this model size, LLM non-determinism is brutal, so I run each configuration several times.

## Contestants

| Model | Parameters | CPU offload | Thinking | Quant |
|---|---|---|---|---|
| Gemma 4 26B | 26B-A4B MoE | none | both arms | UD-Q4_K_M / q8_0 KV |
| Devstral Small 2 | 24B dense | none | instruct only — no thinking mode | UD-Q4_K_XL / q8_0 KV |
| Laguna XS 2.1 | 33B-A3B MoE | experts of 6 layers out of 39 | both arms | IQ4_XS / q8_0 KV |
| Laguna S 2.1 | 118B-A8B MoE | experts of 40 layers out of 48, 68 GB of RAM | instruct only — thinking arm not run | UD-Q4_K_XL / q8_0 KV |
| Muse Glimmer | 30B dense | none | thinking only — can't be turned off | Q4_K_M / f16 KV |
| North Mini Code 1.0 | 30B-A3B MoE | experts of 10 layers out of 48 | both arms | UD-Q4_K_M / q8_0 KV |
| Nemotron 3.5 Lightning | 30B-A3B MoE, hybrid Mamba-Transformer | every expert | both arms | UD-Q4_K_XL / q4_0 KV |
| GPT OSS | 20B-A3.6B MoE | none | always reasons, at `reasoning_effort: high` | MXFP4 native / q4_0 KV |


**Sampling.** Every instruct (non-thinking) run is at temperature 0.2. Thinking runs are at 0.6, because 0.2 is too low for reasoning: long chains of thought start repeating themselves and circling the same idea. So miniswe raises the temperature whenever it enables thinking. That coupling matters for reading this post: a thinking-vs-instruct comparison below is a *two*-variable change, not one. Muse Glimmer can't disable thinking at all, so all of its runs are at 0.6 while the models it sits next to in the table are at 0.2.

## Results

The final round: 23 runs, seven models, 16:07 to 00:40.

| Model | Arm | Attempt 1 | Final | Wall times |
|---|---|---|---|---|
| Laguna XS 2.1 | thinking | 6/6, 6/6, 5/6 | 6/6, 6/6, 6/6 | 465s, 525s, 753s |
| Muse Glimmer | thinking | 6/6, 6/6, 6/6 | 6/6, 6/6, 6/6 | 543s, 552s, 633s |
| Laguna XS 2.1 | instruct | 6/6, 6/6, 6/6 | 6/6, 6/6, 6/6 | 529s, 597s, 2046s |
| Gemma 4 26B | instruct | 6/6, 5/6, 5/6 | 6/6, 6/6, 6/6 | 355s, 641s, 864s |
| Laguna S 2.1 | instruct | 6/6, 6/6, 6/6 | 6/6, 6/6, 6/6 | 690s, 898s, 983s |
| Gemma 4 26B | thinking | 6/6, 6/6, 6/6 | 6/6, 6/6, 6/6 | 741s, 1133s, 1145s |
| Devstral Small 2 | instruct | 6/6, 6/6, **0/6** | 6/6, 6/6, **0/6** | 1149s, 3071s, 3402s |
| GPT OSS | reasoning: high | 3/6 | 5/6 | 3413s |
| North Mini Code 1.0 | thinking | 0/6 | 0/6 | 3409s |

Twenty of the twenty-three runs scored 6/6, seventeen of them on the first attempt. All three failures ran out the full 57-minute clock: devstral and North both ended on `compile:FAIL`, and GPT OSS got the flag wired but failed smoke.

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
