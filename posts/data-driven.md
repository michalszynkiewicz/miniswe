# Making miniswe 2.5× faster, the data-driven way

I built miniswe. I was pretty happy with it, until I benchmarked it against OpenCode. It lost badly: 2.5× slower, and it failed runs that OpenCode passed. This is the story of closing that gap. Spoiler: most of the clever ideas didn't survive the benchmarks.

## miniswe
As I shared [before](https://michalszynkiewicz.dev/blog/miniswe/), I used the power that LLMs gave me to create miniswe - an AI coding agent/CLI. 
I asked Claude to generate me a relatively demanding benchmark script, and started tweaking and adding ideas and validating it with the benchmark.
Sort of data-driven. But without too much oversight. I had an idea, asked Claude to code it, played with it a bit, asked Claude to fix stuff, when I felt something may help, I asked Claude to A/B test it, incorporated what made sense.


## Reality check
Here I was fixing some bugs, adding some features. At some point I reached a state when miniswe mostly succeeded on my benchmark job. I assumed it's the model's weakness that it doesn't succeed 100%, and decided it's time to compare miniswe with some "real" tool. A colleague recently mentioned OpenCode, and I went with it. And the results were brutal.

LLMs are non-deterministic, so I always do at least 3 runs. On my benchmark job, OpenCode passed all six checks in every run, with an average runtime below 500s. miniswe fully succeeded only once, and each run was significantly longer:

| Tool | Check successes | Run times (s) |
|---|---|---|
| **miniswe** | 5/6, 5/6, 6/6 | 1278, 1571, 784 — avg ~1211 |
| **OpenCode** | 6/6, 6/6, 6/6 | 533, 493, 419 — avg ~482 |

## Denial... and don't trust your LLM
How could my fancy tool, with a repo map, LSP feedback everywhere, such a smart context compaction strategy, rigid planning, even an LSP-based refactor - every single thing helping the model in my tests, be so much worse than something else?

### Context handling
I started comparing what's going on during the execution. Compaction strategy was the first thing that Claude flagged.

LLMs use the same context for input and output. And some models are reasoning, so sometimes it's input, reasoning output (which is dropped) and proper output. In the normal work with an LLM, the input context is not only the current prompt but all the things that happened in the conversation (inputs and outputs, and a special first line of the context - the system prompt).
miniswe's conversation compaction is quite smart. It keeps a part of the context raw, and uses LLM-based compaction of older messages, with anchors to full history which is stored on disk.

I tried to be smart with when it fires too. I authoritatively decided that I want that part of context for history, and this for output.
And this is precisely what I did wrong. OpenCode's lazy context compaction - firing only on errors, when the context size was exceeded, was much more efficient.
So I changed what miniswe did, adopted the lazy approach. And reran the benchmarks.

But barely anything improved.

### Various tweaks
I kept digging. Another difference between OpenCode and miniswe was llama-server connection handling. I had been very generous, 120s on a hanging request - which surprisingly happened quite often - was a considerable time-eater, but the gap barely moved.

Another gap was the fact that OpenCode's execution, even though it didn't provide LSP feedback after edits, kept the source code closer to a working version during the whole execution. The reason was simple: it asked the model to run the tests. I adopted this pattern too. Again, it helped, but wasn't even close to closing the gap.

### The smoking gun
At this point, my frustration was quite high. Thankfully, Claude finally read through the execution log from benchmark runs thoroughly. And it found out, to my surprise, that one of the mechanisms miniswe had was broken. `replace_range` - one of the few tools that allow for direct code manipulation - failed ~40% of the time! The model tried to rewrite too large chunks of files, usually dropping lines it shouldn't have touched. Given that a single benchmark run has multiple `replace_range` calls, it always caused trouble.

I had a few ideas for how to fix it, one of them being capping the max lines to replace. This backfired, benchmarks showed runs that attempted perfect replaces but failed due to the cap. Committed to making data-driven decisions, I had to revert it.

So I let `replace_range` execute normally, then showed the resulting diff to the model and instructed it to repair or revert unintended changes. Across ten consecutive benchmark runs, no run failed or spent significant time fixing these kinds of issues again!

| Tool | Check successes | Run times (s) |
|---|---|---|
| **miniswe** | 6/6, 6/6, 6/6, 6/6, 6/6, 6/6 | 682, 550, 377, 518, 291, 356 — avg ~462 |
| **OpenCode** | 6/6, 6/6, 6/6, 6/6, 6/6, 6/6 | 533, 493, 419, 1085, 335, 947 — avg ~635 |

Note that I reran OpenCode three more times and combined all six runs — the new ones had a bit less luck, averaging it up to 635s. That's the nature of running such work on small models. That's LLM non-determinism. But the difference between miniswe results is obvious.

## The lessons

### Don't trust the LLM
Claude and Codex analyzed the logs multiple times. Yet they didn't flag the problem with `replace_range` before - and it was there - lying in the logs for many, many runs.
Models are lazy. They will find something small and fixate on it instead of looking at the big picture. Even the largest and smartest models. Keep pushing, and measuring. LLMs make experimenting cheap - there's no excuse to stop halfway.
They are a great tool but not an oracle. Make them show the data, verify their conclusions.

### Wording is not enough
Along the way, I tried multiple tweaks in miniswe messages - e.g. keep `replace_range` changes minimal. Out of seven plausible attempts, none worked. What helped were mechanical, benchmark-proven changes. Harness that keeps the model on the right path.

## The final scorecard

With everything committed — lazy compaction, the diff-echo guard, the stuck-check nudge, the LSP-wedge fix — I ran one last validation round: fresh llama-server per run, shipped defaults, same task and time budget, across every model I keep on the bench.

| Model | Result | Wall time | Notes |
|---|---|---|---|
| Laguna XS 2.1 (run 1) | 6/6, attempt 1 | 422s | done at ~minute 4, then re-ran tests on a green tree |
| Laguna XS 2.1 (run 2) | 6/6, attempt 1 | 304s | same shape; its fastest run yet |
| Gemma 4 26B | 5/6, 3 attempts | 2850s | see below |
| Muse Glimmer 30B (thinking) | 6/6, attempt 1 | 1277s | one stuck-nudge fired, run wrapped 4 min later |
| Devstral Small 2 | 6/6, attempt 1 | 806s | fastest Devstral run to date (previous best: 1021s) |

Glimmer is the clearest before/after: pre-fix it went 5/6 at 3406s and 6/6 at 2735s, wasting half an hour per run re-reading one test file; with the stuck-check nudge it's now three-for-three clean (751s, 800s, 1277s). Devstral, which had never finished under 1000s, came in at 806s.

And gemma — the reference model this whole post is built on — delivered a parting reminder of everything above. It wired the feature correctly in three minutes. Then it spent half an hour in a revert loop, corrupted its own test file with a non-idempotent `sed` it ran three times (each pass appending one more argument to every callsite), correctly diagnosed the damage on the next attempt — and "verified" its non-fix with `cargo check`, which never compiles the test crate. Five out of six, entirely self-inflicted. Right diagnosis, failed execution: the same edit-mechanics bottleneck `replace_range` had, wearing a different hat. That's the nature of benchmarking small models: every run you add is another chance to learn a new way the harness can be betrayed — and the next mechanical guard to build.

## Benchmark setup
The tests were executed on an RTX 3090 power-capped to 200 W, using llama-server, with Gemma 4 26B A4B (MoE), Q4_K_M. The cap costs some absolute throughput, but it is held constant across every run and every tool, so comparisons are unaffected.
Both OpenCode and miniswe were executed headless, running in a Docker container. 
llama-server was restarted between the runs.

The benchmark is a prompt to change a pinned version of miniswe by adding a prompt override parameter. The tools are given three attempts to achieve the task. After each attempt, the benchmark harness checks that:
* the code compiles
* the binary builds
* the tests pass
* the new parameter is available in help
* the parameter is accepted
* the parameter works as intended
