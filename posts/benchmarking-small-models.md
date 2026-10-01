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
| Muse Glimmer | 30B dense | none | thinking only — no usable instruct mode | Q4_K_M / f16 KV |
| North Mini Code 1.0 | 30B-A3B MoE | experts of 10 layers out of 48 | both arms | UD-Q4_K_M / q8_0 KV |
| Nemotron 3.5 Lightning | 30B-A3B MoE, hybrid Mamba-Transformer | every expert | both arms | UD-Q4_K_XL / q4_0 KV |
| GPT OSS | 20B-A3.6B MoE | none | always reasons, at `reasoning_effort: high` | MXFP4 native / q4_0 KV |


**Sampling.** Every instruct (non-thinking) run is at temperature 0.2. Thinking runs are at 0.6, because 0.2 is too low for reasoning: long chains of thought start repeating themselves and circling the same idea. So miniswe raises the temperature whenever it enables thinking. That coupling matters for reading this post: a thinking-vs-instruct comparison below is a *two*-variable change, not one. Muse Glimmer is the awkward one — it is a reasoning model, and the handful of runs where I switched thinking off and dropped to 0.2 were the worst it produced. So every Glimmer number here is a 0.6 thinking run standing next to models at 0.2.

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

That snapshot is one good night. The full history — more than a hundred scored runs between August 22nd and September 5th, all on the same harness lineage — is less flattering and more useful.

| Model | Arm | Runs | Finished 6/6 | 6/6 on attempt 1 | Median clean run | Decode |
|---|---|---|---|---|---|---|
| Laguna S 2.1 | instruct | 8 | 8 | 8 | 856s | 19 tok/s |
| Laguna XS 2.1 | instruct | 17 | 17 | 17 | 637s | 70 tok/s |
| Laguna XS 2.1 | thinking | 7 | 7 | 5 | 525s | 75 tok/s |
| Gemma 4 26B | instruct | 24 | 19 | 10 | 653s | 83 tok/s |
| Gemma 4 26B | thinking | 8 | 8 | 7 | 799s | 80 tok/s |
| Muse Glimmer | thinking | 13 | 12 | 11 | 775s | 20 tok/s |
| Devstral Small 2 | instruct | 17 | 13 | 10 | 1384s | 22 → 13 tok/s |
| GPT OSS | reasoning: high | 3 | 0 | 0 | — | 74 tok/s |
| North Mini Code 1.0 | both | 5 | 0 | 0 | — | 59 tok/s |
| Nemotron 3.5 Lightning | both | 4 | 1 | 1 | — | — |
| Mistral Small 4 119B | instruct | 4 | 0 | 0 | — | — |

Decode rates are measured mid-run under real load, not from an idle probe. North and Nemotron were dropped early, so their rows count runs inside this window, not their lifetime totals.

Laguna XS ran eighteen times; the eighteenth is excluded here — 57 minutes lost to a bug in miniswe rather than anything the model did, described below. The rest of the table is uncorrected, and the models did not all face the same harness: Laguna S, GPT OSS and Mistral only ever ran on the current one, while more than half of gemma's runs predate the fixes.

The most important thing here isn't in any row. Twenty-seven of those runs scored below 6/6 — and twenty-six of them took at least 47 minutes to get there. Exactly one model, exactly once, failed fast. Everything else failed by running out of clock: still editing, still re-reading, still convinced it was one fix away. A small model that can't do the task doesn't stop. It grinds until you stop it. That, not the score column, is what you have to budget for.

**Laguna XS 2.1 — the surprise, and it held up.** Twenty-five runs across both arms, twenty-four of them 6/6, twenty-two on the first attempt. Its one sub-6/6 (4/6 at 3423s) wasn't the model — miniswe was rebuilding the prompt in a way that made llama.cpp re-read the whole context every round, and the run spent its 57 minutes on that. Fixed since; with it gone, Laguna XS has yet to lose a run on its own merits. It also owns the fastest run in the field at 298s. Distinct personality: it leans on shell (grep, sed, heredocs) over the structured edit tools, writes 23-step plans, and checks every box at the end. 70 tok/s from a 33B MoE with a slice of experts on the CPU.

**Gemma 4 26B — fastest, but it wants the retries.** Median clean run 653s and a floor of 278s: the quickest converger here. The catch is the attempt column. Ten of twenty-four instruct runs reached 6/6 on the first try; the rest needed a second or third pass. The score recovers, the clock doesn't — that's exactly what the 355s / 641s / 864s spread in the round above is showing you. Turning thinking on fixes precisely that (seven of eight first-try, eight of eight finished) for about 150 seconds of median wall. Its failures tend to be self-inflicted rather than confused: in one run the production code was already correct and it corrupted its own test file with a non-idempotent `sed`.

**Laguna S 2.1 — the only perfect record.** Eight runs, eight 6/6, every one on the first attempt, 690–983s. Nothing else in the field has a band that tight. It is also the most expensive thing here by a wide margin: 118B-A8B with the experts of 40 of its 48 layers spilled into 68 GB of system RAM, decoding at 19 tok/s. What that buys is variance reduction, not capability — its little sibling reaches the same answer, usually sooner.

**Muse Glimmer 30B — the surgeon.** Glimmer makes the fewest edits and the fewest reverts of anything in the field: one run built the whole feature in 8 edits with zero reverts. That matters because it is slow. At a flat 20 tok/s it generates a quarter as fast as gemma, and it still turns in a median clean run of 775s — ahead of devstral, within striking distance of gemma's 653s. It buys the difference back by getting it right the first time: eleven of its thirteen runs hit 6/6 on attempt one. It is, however, the model that most needs a harness watching it. Left alone it will settle into a read loop and stay there until the clock runs out; the stuck-detection nudge is what makes the numbers above possible.

**Devstral Small 2 — the slow one.** Devstral finishes: 13 runs out of 17. But its median clean run is 1384s against gemma's 653s and Laguna XS's 637s, and its slowest *successful* run still took 3169s. It's also the only model whose throughput collapses over a run — 22 tok/s on a fresh context, 13 tok/s by the end of a long one — which is a feedback loop rather than a constant, since the longer it takes the slower it gets. All four of its failures ran out the clock. It's demanding on the harness, too: loop detection over longer periods, and hard caps on runaway tool arguments, both exist because of devstral.

**GPT OSS 20B — the same failure, three times.** Three runs, no 6/6, and every one landed in the identical place: 5/6 with `smoke:FAIL`. The flag parses. The help text lists it. The build is green and the tests pass. The binary ignores the override. It builds the entire CLI surface, wires it to nothing, and reports success — which makes it the most interesting failure in the set, because it's the one a human reviewer skimming the diff would also wave through.

**North Mini Code 1.0 — dropped.** Five runs across both arms, never a 6/6. Cohere's "works best with thinking enabled" is real but not sufficient (with the temperature caveat above — the thinking run is also the 0.6 run): instruct mode scored 0/6 by re-reading one file for 55 minutes, 235 reads and zero edits; its best thinking run reached 5/6 at the timeout, one call site short; its most recent one was 0/6 on `compile:FAIL`. Even its good mode is several times slower to converge than gemma or Laguna. Not worth the VRAM.

**Nemotron 3.5 Lightning — dropped, with a story.** Six instruct runs, never a single 6/6, never a finish inside 57 minutes. Its one complete tree came out of thinking mode, done at minute 24 — after which it spent the remaining half hour announcing it was finished while issuing one more tool call, never ending its turn, and the bench scored the tree at the timeout. Underneath that sits a real bug that I can't find reported anywhere: NVIDIA's own chat template prefills `<think></think>` with no trailing newline in non-thinking mode, and the model wants that newline badly enough to flood thousands of blank lines mid-edit until its output budget is gone. Patch the template to add it and the floods stop dead. The patch only covers instruct mode, though — in thinking mode the flood still wins, and one run drowned in 13,005 blank lines.

**Mistral Small 4 119B — the control group.** Four runs, four 0/6, `compile:FAIL` every time. One of them I gave a three-hour clock instead of 57 minutes, purely to rule out the deadline. It wasn't the deadline. A 119B model that can't get the tree to compile, sitting next to a 33B that finishes twenty-four times out of twenty-five, is the cleanest single result in this whole exercise: on this task, parameter count is not the variable.

**The verdict.** Laguna XS 2.1 is the one I'd actually run — 24 of 25, the fastest floor in the field, and it fits in the card. Laguna S 2.1 is the same answer with better variance and ten times the hardware behind it. Gemma 4 26B is just as quick and lands just as often, but budget for a second attempt unless you turn thinking on. Glimmer trades throughput for precision and gets away with it. Devstral finishes, slowly. The rest didn't get there.

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

Next on the bench: Qwen3.8-27B. If you try miniswe with a model I haven't, I'd love to hear how it went.
