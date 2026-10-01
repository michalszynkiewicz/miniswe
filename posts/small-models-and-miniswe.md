---
title: "Miniswe on current local models"
description: "How do recent locally runnable models work with Miniswe"
date: 2026-09-11
---

# Miniswe on current local models

It's been 1.5 years since I bought a used RTX 3090 with the intent of running local LLMs.
Roughly a year later, I started `miniswe` - I wanted to figure out whether I could build some smartness around a small model to let it handle real tasks.

My first test model was Devstral Small 2. Dense, rather slow but fitting my GPU.
A month or so later Google introduced Gemma 4, with a 26B MoE model that is crazy fast compared to Devstral on my hardware.

At first it didn't seem as smart as Devstral on my benchmarks, but when I added a few workarounds to `miniswe`, it started working nearly as well as Devstral.

A few weeks ago I had a bit of time and decided to check what's new on the market for models that I could run locally. It turned out a lot had changed.
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

### Temperature

Temperature controls how much the model gambles on its next token. Low, and it takes the likeliest one every time; high, and it reaches for less obvious choices — useful for prose, dangerous for code that has to compile.

The instruct (non-thinking) runs use a temperature of 0.2. Thinking runs use 0.6, because 0.2 is too low for reasoning: long chains of thought start repeating themselves and circling the same idea. So miniswe raises the temperature whenever it enables thinking. That coupling matters for reading this post: a thinking-vs-instruct comparison below is a *two*-variable change, not one.

## Contestants

| Model | Parameters | CPU offload | Thinking | Quant |
|---|---|---|---|---|
| Gemma 4 26B | 26B-A4B MoE | none | both arms | UD-Q4_K_M / q8_0 KV |
| Devstral Small 2 | 24B dense | none | instruct only — no thinking mode | UD-Q4_K_XL / q8_0 KV |
| Laguna XS 2.1 | 33B-A3B MoE | experts of 6 layers out of 39 | both arms | IQ4_XS / q8_0 KV |
| Laguna S 2.1 | 118B-A8B MoE | experts of 40 layers out of 48, 68 GB of RAM | instruct only — thinking arm not run | UD-Q4_K_XL / q8_0 KV |
| Muse Glimmer | 30B dense | none | thinking only — no usable instruct mode | Q4_K_M / f16 KV |
| North Mini Code 1.0 | 30B-A3B MoE | experts of 10 layers out of 48 | both arms | UD-Q4_K_M / q8_0 KV |
| Nemotron 3.5 Lightning | 30B-A3B MoE, hybrid Mamba-Transformer | every expert | both arms | UD-Q4_K_XL / q4_0 KV |
| GPT OSS | 20B-A3.6B MoE | none | always reasons, at `reasoning_effort: high` | MXFP4 native / q4_0 KV |

## Results

The final round: 26 runs, seven models.

| Model | Arm | Attempt 1 | Final | Wall times | Decode |
|---|---|---|---|---|---|
| Muse Glimmer | thinking | 6/6, 6/6, 6/6 | 6/6, 6/6, 6/6 | 543s, 552s, 633s | 20 tok/s |
| Muse Glimmer + DFlash | thinking | 6/6, 6/6, 6/6 | 6/6, 6/6, 6/6 | 308s, 378s, 703s | 75 tok/s |
| Laguna XS 2.1 | thinking | 6/6, 6/6, 5/6 | 6/6, 6/6, 6/6 | 465s, 525s, 753s | 68 tok/s |
| Laguna XS 2.1 | instruct | 6/6, 6/6, 6/6 | 6/6, 6/6, 6/6 | 529s, 597s, 2046s | 63 tok/s |
| Gemma 4 26B | instruct | 6/6, 5/6, 5/6 | 6/6, 6/6, 6/6 | 355s, 641s, 864s | 74 tok/s |
| Laguna S 2.1 | instruct | 6/6, 6/6, 6/6 | 6/6, 6/6, 6/6 | 690s, 898s, 983s | 19 tok/s |
| Gemma 4 26B | thinking | 6/6, 6/6, 6/6 | 6/6, 6/6, 6/6 | 741s, 1133s, 1145s | 74 tok/s |
| Devstral Small 2 | instruct | 6/6, 6/6, **0/6** | 6/6, 6/6, **0/6** | 1149s, 3071s, 3402s | 15 tok/s |
| GPT OSS | reasoning: high | 3/6 | 5/6 | 3413s | 65 tok/s |
| North Mini Code 1.0 | thinking | 0/6 | 0/6 | 3409s | 58 tok/s |

Twenty-three of the twenty-six runs scored 6/6, twenty of them on the first attempt. All three failures ran out the full 57-minute clock: devstral and North both ended on `compile:FAIL`, and GPT OSS got the flag wired but failed smoke. Decode rates are measured mid-run under real load, not from an idle probe: total tokens generated over total generation time.

The first finding is clear: I need to figure out a tougher benchmark soon :)

### Detailed results - small and fast models

#### THE WINNER
**Muse Glimmer 30B — the beast.** Glimmer makes the fewest edits and the fewest reverts of anything in the field: one run built the whole feature in nine edits — eight range replacements and one atomic refactor — and never reverted once. Across all the runs of the round it reverted nothing at all, while gemma reverted between 3 and 17 times per run and devstral up to 49. When I did the first set of benchmarks I thought: it's awesome it is so accurate, it makes up for how slow it is - it did only 20 tokens per second. And even at that speed its average and median time were faster than Gemma!

Imagine my surprise when I realized I'm not running Glimmer the way it's intended to be run. 
Muse Glimmer comes with a helper model - based on DFlash - see more in the [announcement](https://research.meta.ai/blog/introducing-muse-glimmer-open-agentic-model). 

The Glimmer+DFlash combo makes a ton of difference in speed: decode speed went from 20 tokens per second to 75 - the fastest in our fleet! Out of three runs with DFlash, only one exceeded 400s on my hardware, but there was a reason for it. It chose a more complex path - it added integration tests for the newly added functionality.

I think I have my new go-to model for local inference!

#### THE GOOD

Two other models stood out as effective and efficient. They required some fixes in `miniswe`, but with those in place solved the task nearly every time.

**Laguna XS 2.1 — smart, fast, stubborn.** It refuses to use provided tools, leans heavily on shell (grep, sed), yet it does it quite well. The thinking variant needed a correction attempt once. Excluding Glimmer+DFlash, it had the lowest median time and second lowest average time. The instruct variant got straight 6/6s but went in circles on one of the attempts - which cost a lot in wall time.
With a small CPU offload and MoE architecture it's one of the fast ones in terms of token throughput. It needed one fix to be efficient. It has a 512-token sliding attention window. miniswe keeps the current plan at the tail of the context, and rewriting it there forced a costly re-prefill every round, so those rewrites had to be limited.

**Gemma 4 26B — \*fastest, but needs specific workarounds.** In the instruct variant, it often needs a correction attempt. The thinking variant was much slower but scored straight 6/6 on the first attempts. Until I added DFlash to Glimmer, it was the fastest token-per-second-wise. Needs many more rounds to succeed than Laguna S or Muse Glimmer but makes it up with its sheer speed. Its failures tend to be self-inflicted rather than confused: in one run the production code was already correct and it corrupted its own test file with a non-idempotent `sed`. It leaked thinking blocks into the output, sometimes went into change->revert spiral. But with fixes on the harness (miniswe) level, it did a really good job.

#### THE BAD
**GPT OSS 20B — dropped.** In the proper benchmark, it wasn't able to solve the task. In pre-benchmark runs, it failed every time too. Never actually pushed the flag through all the layers. Not smart enough for the task with `miniswe` as the harness.

**North Mini Code 1.0 — dropped.** 0/6 on timeout in the benchmark run. All the previous runs failed too. Enabling thinking helps but is not sufficient. Even its good mode is several times slower to converge than gemma or Laguna.

### Detailed results - heavyweights

#### SLOW IS SMOOTH, SMOOTH IS FAST
**Laguna S 2.1 — the colossus.** The benchmark runs, and all the pre-benchmark runs, scored straight 6/6, even with thinking disabled, 690–983s even though on my hardware it only achieves 19 tok/s. It is also the most expensive thing here by a wide margin: 118B-A8B with the experts of 40 of its 48 layers spilled into 68 GB of system RAM.

#### THE TURTLE

**Devstral Small 2.** Devstral is effective but isn't efficient. Its only failure was hitting the time limit. It's demanding on the harness, too: loop detection over longer periods, and hard caps on runaway tool arguments, both exist because of devstral.

### Skipped from the benchmark
**Mistral Small 4 119B — the elephant.** It's large, it's clumsy. On one pre-benchmark attempt it scored 3/6, and 0/6 and a few more. It started doing something, wasn't able to converge in a reasonable time. I left it out of the benchmark.

**Nemotron 3.5 Lightning — dropped, with a story.** Six instruct runs, never a single 6/6, never a finish inside 57 minutes. Its one complete tree came out of thinking mode, done at minute 24 — after which it spent the remaining half hour announcing it was finished while issuing one more tool call, never ending its turn, and the bench scored the tree at the timeout. Underneath that sits a real bug that I can't find reported anywhere: NVIDIA's own chat template prefills `<think></think>` with no trailing newline in non-thinking mode, and the model wants that newline badly enough to flood thousands of blank lines mid-edit until its output budget is gone. Patch the template to add it and the floods stop dead. The patch only covers instruct mode, though — in thinking mode the flood still wins, and one run drowned in 13,005 blank lines.

**The verdict.** Both Laguna 2.1 models are strong, but Muse Glimmer turned out to be the strongest in terms of efficiency + effectiveness.

# Takeaways
* I need a more difficult benchmark
* Even though the space of open-weight small models is not as dynamic as the large models, there's a lot happening in it
* Muse Glimmer 30B + DFlash seems to be the best thing I can reasonably run on my hardware, with Laguna 2.1 models a close second
