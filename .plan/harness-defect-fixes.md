# Harness defect fixes — found during the 2026-08-25 validation round

Three issues, found by analysing why 3 of 7 runs reached for `sed -i`.
Fix / test / commit ONE AT A TIME. Never push. No co-committer line.

---

## Issue 1 — LSP can vanish silently (rust-analyzer download is a SPOF)
Evidence: run 6 (docker_20260825_100518) is the ONLY run of 38 where the
startup download failed. `refactor add_param` then returned
"requires LSP support (no LSP client available)" and the model fell back to
`sed -i` across 14 test callsites, read-looped, and timed out at 4/6.
Nothing in the results directory records that the toolset was degraded.

- [ ] 1a. Surface LSP status in the run results so a degraded run can never be
      silently averaged in (record status file / marker, or abort the bench run).
      RANKED FIRST — this is the one that would have saved an hour today.
- [ ] 1b. `scripts/Dockerfile.benchmark`: BAKE the same GitHub artifact into
      /root/.miniswe/lsp-servers/ at image-build time so ensure_binary() returns
      at the cache check and no run touches the network. NOT `rustup component
      add rust-analyzer` -- that installs a DIFFERENT binary (toolchain 1.89 vs
      GitHub releases/latest) and would silently change behaviour on every
      remaining run. Same artifact = no confound.
- [ ] 1c. `src/lsp/servers/download.rs::download_rust_analyzer`: retry with
      backoff + request timeout. `src/cli/commands/run.rs:549`: `{e}` -> `{e:#}`
      so the actual cause is visible (we still don't know if run 6 was a 429,
      DNS, or a bad gzip). Backoff is REQUIRED not optional: the URL is
      unauthenticated `releases/latest/download/...` hit 20+ times a day from
      one IP, so a tight retry could make rate-limiting worse.

## Issue 2 — add_param callsite anchors go stale -> PARTIAL
Evidence: 2 independent reproductions.
  gemma 165637: "repl.rs:210 validation failed: OLD line 1 doesn't match
                 source at line 210 (anchor 209)"  -> PARTIAL 15/16
  laguna 110624: "OLD line 2 doesn't match source at line 283 (anchor 281)"
                 -> signature rewrite failed, then PARTIAL 15/16
Hypothesis to CONFIRM before fixing: callsite positions are resolved BEFORE the
signature rewrite, so edits to the same file shift the lines out from under the
anchors. In 165637 an earlier successful `add_param run@repl.rs` shifted repl.rs
before `add_param assemble` tried to patch its callsite at line 210.
Leaves the project uncompilable: signature rewritten, 15/16 callsites done.

- [ ] 2. Re-resolve callsite positions against current file content immediately
      before each callsite edit (or validate by content near the anchor).
      NOTE: model_edit.rs:582,598 document a deliberate "we don't search" policy
      — respect it; prefer re-resolution over fuzzy search.

## Issue 3 — duplicate guard locks the tool out in exactly the broken state
Evidence: after the PARTIAL above, the model reverted repl.rs, then:
  add_param run      -> "✗ already has a parameter named system_prompt_override"
  add_param assemble -> "✗ already has a parameter named system_prompt_override"
`add_param.rs:224` checks only `has_param(signature)`. It assumes
param-exists => callsites-consistent. After a PARTIAL or a revert that is false,
so the only bulk tool refuses precisely when it is needed, across 14 callsites.
Its advice ("EDIT the specific callsite") is what sends the model to sed.

- [ ] 3. Make the guard state-aware: refuse only when signature AND callsites
      agree. When they disagree, re-sync the stale callsites instead of refusing.

---

## Round bookkeeping (do AFTER all three are committed)
Runs invalidated by these defects, to rerun ONCE each and replace the result:
  - run 2  gemma instruct  5/6 3402s  — add_param locked out -> sed
  - run 6  laguna instruct 4/6 3423s  — no LSP -> sed
  - run 7  laguna thinking 6/6 1671s/755rds — stale anchor -> sed (slow *because* of the failure)
Keep as-is (clean add_param, no sed): runs 1, 3, 4, 5.
  - run 5 (laguna instruct 6/6 1599s) was CLEAN but slow for unclear reasons; flag, do not rerun yet.

BINARY CONSISTENCY: runs 1-8 used the pre-fix binary. The fixes only alter
failure paths, so clean runs are unaffected by construction — that is the
argument for keeping them. State this explicitly in the post.

---

## STATUS 2026-08-25 12:16 — all three fixed, tested, committed

| fix | commit | files |
|---|---|---|
| 1 LSP single point of failure | `739f89a` | `src/lsp/servers/download.rs`, `src/cli/commands/run.rs`, `scripts/Dockerfile.benchmark`, 3 hunks of `scripts/run-benchmark-docker.sh` |
| 2 stale callsite anchors | `5b699b4` | `src/tools/refactor/sites.rs` (+10 tests), `drop_param.rs` |
| 3 duplicate-guard lockout | `3db1b84` | `src/tools/refactor/add_param.rs`, `ast_span.rs` (+8 tests) |

`cargo fmt` / `cargo clippy --all-targets` (clean) / `cargo test` (581 lib + all
integration suites, 0 failures) run before each commit. Nothing pushed.

Image rebuilt and verified: rust-analyzer 0.3.3025 baked at
`/root/.miniswe/lsp-servers/rust-analyzer`, and the image's `miniswe` binary
carries both new strings (`LSP: DEGRADED`, the re-sync message).

The pre-existing env-knob hunks of `run-benchmark-docker.sh` (THINKING /
CTX_WINDOW / MAX_OUTPUT_TOKENS / STREAM_IDLE_SECS / COMPACTION) stay
uncommitted, as do `docs/TODO.md`, `start-gemma4.sh`, `start-gemma4-31b.sh`.

### Run 8 (Laguna XS thinking #2) — hit defect 2 as well
6/6 on attempt 2, 1887s, 864 rounds, decode 120.5/117.1 tok/s. Attempt 1 failed
smoke after `add_param` returned PARTIAL 15/16 — the one failure was
`src/cli/commands/run.rs:131: OLD line 2 doesn't match source at line 132
(anchor 130)`, i.e. the reported line sits one ABOVE the real call opener
(`let assembled = context::assemble(`). 251 dumps mention `sed`. Result is not
comparable; rerun.
