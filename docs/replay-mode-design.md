# Replay mode — resume a real run from its "before first fix" state

Goal: instead of a synthetic seed, drop the agent into the **exact** state a real
run was in right before its first fix attempt — same working tree **and** same
conversation context — and see whether it recovers. This is the authentic
version of the seed experiment (no inflated 5/6 floor, no idealized one-link gap).

## Status

- **Fixture extraction: DONE + tested** (`scripts/replay/extract-fixture.py`).
  Given a run's `00_baseline` dir it writes a fixture dir:
  - `context.json` — the exact LLM request at the first done-gate rejection:
    `{messages, tools, model, temperature}`. `messages` is the full
    system+history (with the real observation-masking applied) — i.e. the
    context the agent carried into its first fix.
  - `tree/` — the working tree at that round, exported from miniswe's
    `shadow-git` (which commits "round N — before round N" every round).
  - `manifest.json` — source run, dump, shadow commit, round, previews.
  Verified on `docker_20260617_170328`: clean original-task context (no reset
  compaction), real done-gate signature, 66-file half-built tree.

  Key decisions baked into the extractor:
  - "before first fix" = first message containing the done-gate signature
    *"A check that exercises the change end-to-end"* (NOT the looser
    "do NOT finish yet", which also appears in refactor/tool rejections).
  - Skip `gate_context_reset` runs' compacted "[Your earlier work…] Still need:"
    contexts — those aren't a natural pre-fix history. Prefer reset-OFF runs.
  - Map dump → shadow round by **counting assistant turns** in the captured
    history (rounds == main-agent turns), not by file mtime (volume/clock skew
    makes mtimes unreliable). Try round, round-1, round+1.
  - `git archive` with `filter.lfs.smudge=cat` etc. to bypass the inherited LFS
    filter.

- **Replay runner: TO BUILD** (needs the LLM server to validate end-to-end).

## Replay runner — design

A Docker-isolated harness (sibling of `run-benchmark-docker.sh`) that:
1. Copies `fixture/tree/` into `/work` (the exact half-built code state).
2. `git init` + baseline commit (so diff capture + shadow-git work as usual).
3. Runs miniswe **seeded with the captured conversation** instead of a fresh
   task, then applies the same 6-check validation + best-of-N as the other benches.

The one miniswe-core addition needed: a way to **start the agent loop from a
given conversation history** rather than a fresh `assemble()`. Options:

- **(A) `--replay-context <context.json>` flag (preferred).** `run()` loads the
  `messages` array as the initial conversation, uses the trailing user (gate)
  message as the turn's driver, and enters the loop. The captured `system`
  message is used **verbatim** → faithful "context we had then". The done-gate
  participates normally so we observe whether it recovers.
  - Touch points: `src/cli/mod.rs` (flag), `src/cli/commands/run.rs` (when set,
    skip the fresh `context::assemble` and instead seed `messages` from the
    file; map the JSON roles → `llm::Message`). The dumps are miniswe's own
    serialization, so the mapping is round-trippable (watch tool_call ids +
    tool-result pairing).
- **(B) session-restore.** Translate `context.json` into a `.miniswe` session
  file and use `--continue`. Avoids a new flag but couples to the session
  format; (A) is cleaner and more explicit.

## Two experiments the replay enables

1. **Faithful resume** — verbatim captured context + tree → does it recover?
   Compare to what the *original* run did from here (its final score).
2. **Clean-context resume** — same tree, but re-`assemble()` a fresh context
   (drop the failure-primed history) instead of the captured one. This is the
   direct test of the "clean restart beats in-context grind" thesis
   ([[gate-context-reset]]) on a *real* stuck state.

## Caveats

- Determinism: the model is sampled (temp>0), so a single replay is one draw;
  run N for a distribution, like the other benches.
- The captured history already reflects observation-masking — that's authentic;
  don't re-mask.
- Pick the fixture run deliberately: a smoke-fail (threading-miss) stuck state
  vs a compile-broken one are different scenarios; the extractor currently takes
  the *first* done-gate block — add a `--failure-kind` filter later if needed.
