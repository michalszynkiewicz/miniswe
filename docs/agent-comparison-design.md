# Agent comparison harness: miniswe vs aider

## Goal

For each local model M in `{qwen3-coder-next, devstral-small-2, gemma-4-26B-A4B}`,
run the same SWE task through **miniswe** and through **aider**, against the
same local llama-server, with the same validation suite, and emit a single
unified results table.

We want to answer:

- Does miniswe's tool surface (plan, refactor, spawn_agents, etc.) outperform
  aider's simpler search/replace patch model on local LLMs?
- Does that gap depend on the model? (Hypothesis: agents with richer tool
  surfaces help weaker models more than stronger ones — strong models can
  recover from a thinner agent.)

## Non-goals

- Comparing against frontier API models (out of scope — local only).
- Multi-task benchmarks. One representative task is enough for v1; more once
  the harness is proven.
- Tuning either agent. Both run with stock defaults and recommended sampling.

## Scope: which models

Start with three, picked to span the strength axis:

| Model | Size | Local strength tier |
|---|---|---|
| Devstral Small 2 | 24B dense | low-mid (baseline coder) |
| Gemma 4 26B-A4B | 26B MoE / 4B active | mid (generalist) |
| Qwen3-Coder-Next | 80B MoE / 3B active | high (current best local coder for this rig) |

Each model already has a `start-*.sh` launcher pointing at the host's
llama-server CUDA Docker image. Re-use as-is.

## Methodology

### Shared validation suite

Whatever the agent produces — a `git diff`, modified files, anything —
gets graded the same way. The existing 6-check suite from
`scripts/run-benchmark-docker.sh` is already agent-agnostic; we lift it
out into a shared function.

Checks:

1. `cargo check` passes
2. `cargo build` passes
3. `miniswe --help` shows a `--*prompt*` flag
4. The flag parses without error
5. `cargo test` passes
6. Smoke test: the binary, invoked with the new flag and prompt
   `"respond with PONG_42"`, returns `PONG_42`

Score is sum of passes (0–6). Wall time and token counts captured per
attempt.

### Task

Same task as recent miniswe runs:

> Add a CLI flag `--system-prompt-override` (short: `-s`) that takes a
> string and replaces the default system prompt with the provided text.
> When this flag is set, skip all context providers and just use the
> override text as the system message. Make sure it works for both
> single-shot and interactive modes.

Pinned baseline SHA: `cc34d2626faf32c1b6dd1b8b33af693fb936b098` (current
`BASELINE_SHA` in the bench script).

### Per-agent harness

Each agent runs inside its own Docker container against the same
`--network=host` so they share the same llama-server endpoint.

#### miniswe (existing)

Uses `scripts/run-benchmark-docker.sh` as-is. Already does:
- Fresh checkout at pinned SHA
- Generates `config.toml`
- Runs `miniswe --yes "${TASK}"` with timeout
- Captures stdout/stderr, diff, LLM dumps
- Runs validation, writes pass/fail to `container.log`

#### aider (new)

New script `scripts/run-aider-bench.sh`, mirroring the structure of the
miniswe one. Differences:

- Uses a separate Docker image with `aider` installed (`pip install aider-chat`)
  on top of the same base (Rust toolchain + llama-server CLI tools).
- Aider invocation:
  ```
  aider --yes \
        --model openai/local \
        --openai-api-base http://localhost:8464/v1 \
        --openai-api-key sk-nokey \
        --message "${TASK}" \
        --auto-test \
        --test-cmd "cargo check"
  ```
  - `--yes` auto-confirms edits
  - `--auto-test` re-runs `cargo check` after each edit batch (analogous
    to miniswe's compile gate)
  - `--message` runs in non-interactive one-shot mode
- After aider exits, run the same 6-check validation on the resulting
  working tree.
- Capture aider's full stdout for token-usage parsing.

### Per-model orchestrator

New `scripts/run-agent-comparison.sh`:

```
for each model in MODELS:
    1. ./start-${model}.sh &       # background; wraps `docker run`
    2. wait_for_server               # poll /v1/models until 200
    3. ./scripts/run-benchmark-docker.sh --model ${model} ...
    4. ./scripts/run-aider-bench.sh  --model ${model} ...
    5. docker stop llama-server-*    # kill the server container
    6. wait_for_server_down          # confirm port 8464 free
done
print_summary
```

Constraints:

- Server start/stop must be reliable. Use the container name set by
  `LLAMA_CONTAINER_NAME` so we can `docker stop` it deterministically.
- Server must be fully loaded before benches start. Poll `/v1/models` and
  also do a tiny `/v1/chat/completions` ping to verify weights are ready.
- Aider and miniswe runs are sequential (not parallel) — both need
  exclusive access to the GPU.

### Results layout

```
benchmark_results/comparison_<timestamp>/
├── qwen3-coder-next/
│   ├── miniswe/    # same structure as today's miniswe bench results
│   └── aider/      # symmetrical layout for aider
├── devstral-small-2/
│   ├── miniswe/
│   └── aider/
├── gemma-4-26B-A4B/
│   ├── miniswe/
│   └── aider/
└── summary.tsv     # one row per (model, agent), columns below
```

`summary.tsv` columns:

```
model    agent    attempts    final_score    rounds    wall_s    tokens_in    tokens_out    diff_lines    tool_errors
```

`tokens_in/out` parsed from llama-server's `/metrics` endpoint snapshots
taken at the start and end of each agent run.

## Pitfalls

1. **Aider's tool surface is different.** It uses search/replace patches
   it generates, not OpenAI tool calls. So we're partly measuring
   *editing-via-text* vs *editing-via-tools*. That's fine — that's exactly
   the question — but the framing in the writeup should be honest about
   it.

2. **`--auto-test` cost.** Aider re-runs `cargo check` after each edit
   batch. On a cold target dir this is slow. Either accept the cost or
   pre-warm `target/` once before each attempt.

3. **Server warm-up bias.** First request after starting the server is
   slow (model paging in). To avoid favoring whichever agent runs second,
   add a single warm-up call between server-start and the first bench.

4. **Aider's silent token usage.** Aider doesn't emit a structured
   token-usage trailer by default. Pull from llama-server's `/metrics`
   endpoint as ground truth for both agents.

5. **Time budget.** Per model:
   - server load: ~30–60s
   - miniswe bench: up to 3 × 3400s = ~2.8h
   - aider bench: similar timeout
   - server unload: ~10s
   Three models × ~6h ≈ **18h of wall time** for a full comparison.
   Reasonable for an overnight run; not for an interactive iteration.
   Make timeouts a CLI flag so you can do a quick "shape check" run in 30
   minutes.

## What we ship

1. `docs/agent-comparison-design.md` — this file.
2. `scripts/Dockerfile.aider` — aider container image.
3. `scripts/run-aider-bench.sh` — aider equivalent of
   `run-benchmark-docker.sh`, same validation suite.
4. `scripts/run-agent-comparison.sh` — top-level orchestrator that loops
   over models, starts/stops the server, runs both agents, aggregates
   results.
5. Minor refactor: extract the 6-check validation into a shared shell
   function (or shared inline script blob) so miniswe and aider scripts
   stay in sync.
