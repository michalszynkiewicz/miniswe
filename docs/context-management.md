# How miniswe manages context

miniswe runs a single LLM agent loop against a fixed context window (60K tokens
in the benchmarks). Every tool round it **re-assembles** the prompt from scratch
and, when the conversation outgrows its budget, **compacts** the history. This
doc describes both halves: how a turn's context is built, and how it's kept
within budget over a long session.

The two moving parts:

- **Assembly** (`src/context/mod.rs::assemble`) — builds the per-turn prompt:
  a system message (instructions + injected context blocks) followed by the
  conversation history and the current user message.
- **Compaction** (`src/context/compressor.rs`) — when raw history exceeds its
  token budget, reduces it via the configured strategy.

---

## 1. Per-turn assembly

`assemble()` rebuilds the message list every round. It never mutates history in
place; it composes a fresh `Vec<Message>`:

1. **System prompt** (`build_system_prompt`) — who the agent is, the tool-routing
   table (which intent maps to which edit tool), the one-tool-call-per-response
   contract, and the edit contract. Its shape varies with `ceremony` mode and
   whether a plan exists (pre-plan "explore→plan" vs post-plan "you are editing
   now").
2. **Project root** — the absolute path; all tool paths are relative to it.
3. **Context providers** (see §3) — each enabled provider may append a tagged
   block (`[PLAN]`, `[REPO MAP]`, `[LESSONS]`, …) to the system message.
4. **Active plan** — `.miniswe/plan.md`, if present, appended as `[PLAN]`.
5. **Conversation history** — all non-system messages so far, added verbatim
   (compaction has already run on them in the loop; see §4).
6. **Current user message** — the task, or the latest follow-up.

The result is one `system` message carrying the static "frame," then the rolling
`user`/`assistant`/`tool` history. Token estimate is rough: **~4 chars/token**
(`estimate_tokens`).

---

## 2. The token budget

All budgeting derives from `context_window` minus fixed overhead. From
`compressor::budgets()`:

```
available    = context_window − tool_definition_tokens − context_window/6   (output headroom)
raw_budget   = available / 3      # max tokens of un-compacted recent history
summary_budget = available / 4    # target size of an LLM summary
```

At a 60K window with ~3K of tool definitions: `available ≈ 47K`,
**`raw_budget ≈ 15.6K`**. When the non-system history exceeds `raw_budget`,
compaction fires. The same `raw_budget` is the trigger for *every* strategy, so
they're comparable — only the action taken differs.

Tool outputs are budgeted separately: a single `file read` / `shell` result is
capped at `tool_output_budget_chars = context_window / 10` (≈6K chars / ~75
lines at 60K), so one read can't blow the window on its own.

---

## 3. Context providers

Providers are pluggable blocks injected into the system prompt
(`src/context/providers.rs`, toggled under `[context.providers]`). Each reads a
source and contributes a tagged section, or nothing. Order is fixed by
`default_providers()`.

| provider | source | header | default | notes |
|---|---|---|---|---|
| `profile` | `.miniswe/profile.md` | — | **off** | language/structure/deps overview (compressed) |
| `guide` | `.miniswe/guide.md` | `[GUIDE]` | **off** | project-specific instructions; skipped if it's the placeholder |
| `project_notes` | `.ai/README.md` | `[PROJECT NOTES]` | **off** | architecture notes, truncated to the output budget |
| `plan` | `.miniswe/plan.md` | `[PLAN]` | **on** | the active step plan |
| `lessons` | `.miniswe/lessons.md` | `[LESSONS]` | **off** | accumulated tips; keyword-filtered (full if <2000 chars) |
| `repo_map` | index + dep graph | `[REPO MAP]` | **off** | PageRank-scored, task-personalized map; on-demand via `code(action='repo_map')` |
| `mcp` | live MCP summary | `[MCP SERVERS]` | **on** | one line per connected MCP server |
| `scratchpad` | `.miniswe/scratchpad.md` | `[SCRATCHPAD]` | **on** | the agent's persistent task notes |
| `usage_guide` | embedded | `[USAGE GUIDE]` | **on** | only when the user asks a meta-question about miniswe |
| `plan_mode` | runtime flag | `[MODE:PLAN]` | **on** | read-only marker in plan mode |

> **Gotcha (learned the hard way):** the `off`-by-default providers
> (`profile`/`guide`/`project_notes`/`lessons`) are still reachable on demand via
> tools, but they are *not* auto-injected. A bench config that set only
> `repo_map = false` and relied on defaults silently dropped all four — and that
> alone took gemma-4 from **6/6 → 5/6** on the flag task (it lost the codebase
> orientation + lessons that help thread a change end-to-end). The standard bench
> (`run-benchmark-docker.sh`) turns them all **on**.

---

## 4. Conversation compaction

When non-system history exceeds `raw_budget`, `maybe_compress()` runs the
configured strategy (`[context] compaction`). All four keep the most-recent
turns raw and act only on the older prefix; they differ in *what* replaces it.

| strategy | action when over budget |
|---|---|
| **`unified`** *(default)* | miniswe's production approach: LLM rolling summary anchored on the plan, **keep recent raw**, full pre-compaction text archived to `.miniswe/session_archive.md` with a pointer in the summary |
| `sliding_window` | pure truncation: drop the oldest turns, keep recent within budget. No summary, no LLM, no archive — just a one-line "older turns dropped" marker |
| `rolling_summary` | textbook rolling LLM summary: summarize old → running summary, keep recent raw. No plan-anchor, no disk archive, neutral prompt |
| `observation_masking` | keep the full action trajectory; replace old tool **observations** (results) with a placeholder, keeping the last 3 raw. No LLM call |

### The `unified` (production) path, in detail

1. **Plan anchor.** If the plan tool is on, the *first* time history goes over
   budget the compressor doesn't summarize — it injects a "update your plan
   before I compress" nudge and returns. The refreshed plan then anchors the
   summary on the next pass.
2. **Split.** Walk from the newest message back, keeping turns until `raw_budget`
   is reached; everything older is the compaction set.
3. **Summarize.** Send the old turns to the **Fast** model role with a structured
   "one line per file changed, end with *Still need:*" prompt
   (`SummaryStyle::Structured`), carrying any prior summary forward.
4. **Archive.** Append the *full, untruncated* old turns to
   `.miniswe/session_archive.md` — the lossless on-disk record behind the lossy
   in-context summary.
5. **Replace.** Swap the old prefix for one `[Your earlier work in this session]`
   message containing the summary + a pointer to the archive, then re-append the
   recent raw turns.

The three baselines exist mainly for benchmarking miniswe's approach against the
canonical alternatives; each is implemented at its honest textbook strength.

### Instrumentation

Every compaction event emits one stderr line for measurement:

```
[compaction] strategy=unified before_tokens=17122 after_tokens=12891 elided_tokens=4231 msgs_before=10 msgs_after=8
```

---

## 5. Message sanitation

Before each LLM call, `sanitize_messages()` enforces the role alternation that
strict chat templates (Mistral/Devstral-style) hard-require: one leading system
message, alternating `user`/`assistant`, `tool` only after an assistant
tool-call, consecutive same-role messages merged, dangling tool-calls dropped.
Violating this returns HTTP 500 from some servers, so it's load-bearing, not
cosmetic.

---

## 6. Context-level recovery (the `[tools]` knobs)

Beyond rolling compaction, miniswe has opt-in mechanisms that manipulate context
when the agent gets *stuck*, gated on the **behavioral done-gate** — a configured
check (`[validation] command`) that runs the change end-to-end before accepting
completion (e.g. "does the new flag actually change runtime behavior?"). When it
fails, the agent is told to keep working rather than finish.

| knob | default | what it does on repeated gate blocks |
|---|---|---|
| `gate_context_reset` | off (code) / **on** (bench) | after N blocks, **drop the polluted history and re-assemble a clean context** — the in-session equivalent of a fresh best-of-3 attempt (files persist on disk). The evidence: clean reset beats in-context grinding (a qwen run ground 121 rounds → fail vs. a fresh attempt 53 rounds → 6/6). |
| `reactive_debugger` | off | hand the specific failure to a fresh-context, read-only debugger sub-agent; its diagnosis is fed back for the main agent to apply |
| `spiral_reset` | off | detect a revert-loop (same file reverted 3+×/turn) and inject a reset + forced replan |
| `auto_revert_ast_cascade` | off | on 3+ consecutive broken-AST edits, force-revert to the last clean revision |

These are *recovery* mechanisms, distinct from rolling compaction, but they live
in the same space (they manage what the model sees). For a clean compaction A/B
they're held constant across arms.

---

## 7. Config reference

```toml
[model]
context_window = 60000        # the hard ceiling everything budgets against
max_output_tokens = 8000

[context]
compaction = "unified"        # unified | sliding_window | rolling_summary | observation_masking
repo_map_budget = 5000
max_rounds = 50

[context.providers]           # see §3 for per-provider defaults
profile = true
guide = true
project_notes = true
lessons = true
repo_map = false

[tools]
gate_context_reset = true     # §6 recovery knobs
reactive_debugger = false
spiral_reset = false
auto_revert_ast_cascade = false
```

## Source map

| concern | file |
|---|---|
| per-turn assembly, system prompt, sanitation | `src/context/mod.rs` |
| compaction strategies + budgets + instrumentation | `src/context/compressor.rs` |
| context providers | `src/context/providers.rs` |
| done-gate, gate-reset, debugger | `src/cli/commands/run.rs` |
| tool-output budget | `src/config/mod.rs` (`tool_output_budget_chars`) |
