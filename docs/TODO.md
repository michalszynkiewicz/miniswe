# TODO

## Docs subsystem

### No `docs read` CLI command for humans

The CLI has `add`, `list`, `refresh` but no way to read cached docs without
manually `cat`-ing files from `~/.miniswe/docs/`.

**Fix:** Add a `docs read <name>` subcommand that prints cached content to
stdout (with optional `--topic` flag to filter sections, reusing
`extract_relevant_sections`).

### Filename matching is fragile

`docs add` derives the filename from the URL's last path segment
(e.g. `https://docs.rs/tokio/latest/tokio/` -> `tokio`). The LLM's
`docs_lookup` then does a case-insensitive substring match on filenames.
This breaks when the slug doesn't match the library name (e.g.
`hooks.html` for React docs).

**Fix:** Store a sidecar metadata file (`docs/_index.json`) mapping each
cached file to its source URL, library name, and fetch timestamp. Use the
library field for `docs_lookup` matching instead of the filename. This also
enables `docs refresh`.

### Raw HTML is stored, wasting LLM context tokens

`docs add` saves the HTTP response body as-is. For most web pages this is
full HTML with tags, scripts, and nav chrome — all noise for the LLM.

**Fix:** Run fetched content through an HTML-to-markdown converter (e.g.
`htmd` or `html2text` crate) before saving. Fall back to raw storage if
conversion fails or if the content is already plain text / markdown.

### `docs refresh` is unimplemented

The original URLs aren't stored, so there's nothing to re-fetch.

**Fix:** Depends on the metadata index above. Once `_index.json` tracks
source URLs, `refresh` iterates the index and re-fetches + reconverts each
entry.

## Helm / YAML support

### Tree-sitter YAML check flags every Helm template as `[ast] broken`

`parse_check` (`src/tools/fast/ast.rs`, `lang-yaml` default feature) runs the
plain YAML grammar on anything ending in `.yaml`/`.yml`. Inline expressions
(`name: {{ include "x.fullname" . }}`) happen to parse as flow mappings, but
block directives on their own line — `{{- if … }}`, `{{- range … }}`,
`{{- toYaml … | nindent 4 }}`, `{{- end }}` — are a `1:1: syntax error`
(verified 2026-08-23 against `chart/templates/svc.yaml` snippets). Nearly every
real `templates/*.yaml` has those. Because `parse_check_supported` also
answers true for YAML, the harness TRUSTS the verdict: each such edit is
recorded `ast_ok = false`, the auto-revert cascade
(`src/tools/fast/auto_revert.rs`) force-reverts after 3 trailing broken
revisions, and the post-revert "smallest edit, braces balanced" hint then
steers the model in circles — on edits that were correct.

**Fix:** treat YAML under a `templates/` directory, `.tpl` files, or content
containing `{{` as *unsupported* (no AST verdict; `parse_check_supported`
→ false) instead of running the YAML grammar. Plain `values.yaml`, `pack.yaml`,
`Chart.yaml`, `pkg-bundle.yaml` keep the syntax + duplicate-key check. Unit
tests for both sides. Cheap, model-agnostic, and the highest-value item here.

### No LSP for Helm charts (helm-ls)

`LspServer::detect` (`src/lsp/servers.rs`) knows rust-analyzer,
typescript-language-server, pyright, gopls, clangd, jdtls — a PKG package repo
(no Cargo.toml/package.json/go.mod) gets NO LSP at all, and `language_id()`
(`src/lsp/client.rs`) maps `.yaml` to `plaintext`.

`helm-ls` (mrjosh/helm-ls, v0.5.4 2025-11, single Go binary, the default in
Neovim/Mason and the VS Code Helm extension) is the only real Helm server:
tree-sitter-go-template parsing, hover/completion/go-to-definition for
`.Values.*`, `.Chart`, `.Release`, `include`/`template` → `_helpers.tpl`,
values.yaml ↔ templates references (0.5.0+), diagnostics from `helm lint`
(needs `helm` on PATH) and — optionally — yaml-language-server run over a
templated copy of the file. It is chart-scoped (`templates/**`, `*.tpl`,
`values*.yaml`); it is not a general YAML server, and it does not flag an
undefined `.Values.foo`. Priority: Helm over plain YAML (decided 2026-08-23);
yaml-language-server (schema validation for pack.yaml etc.) is a later opt-in.

**Plan:**
- `LspServer::HelmLs`: detect AFTER all existing language checks (single
  server per project — a Rust repo with a chart keeps rust-analyzer) when
  `Chart.yaml` exists at the root or 1–2 levels down (`chart/Chart.yaml`,
  `charts/*/Chart.yaml`, the PKG `**/chart/Chart.yaml` layout).
- Binary: GitHub release asset `helm_ls_{linux,darwin}_{amd64,arm64}`
  (`servers/download.rs`, same shape as rust-analyzer/clangd), cached as
  `~/.miniswe/lsp-servers/helm_ls`; PATH lookup first like the others; start
  with `serve`, verify with `version`.
- `language_id`: `helm` for `templates/**/*.yaml|*.tpl`, `yaml` for other
  `.yaml/.yml` (helm-ls handles `values*.yaml` under that id).
- Verify live before calling it done: spawn against a real chart
  (`/home/michal/dev/tmp/go-rest-fixed/Chart.yaml` or a scratch chart), break
  a template, confirm diagnostics arrive through `get_diagnostics_with_status`
  within `lsp.diagnostic_timeout_ms`. Check whether helm-ls publishes on
  `didChange` or only `didOpen`/`didSave` — if the latter, the
  `[lsp file] pending — diagnostics didn't settle` line would fire on every
  Helm edit and needs handling (not a false green).
- Land the AST fix first; it matters regardless of the LSP.

## Cleanup: features proven not helpful

Analysis 2026-08-23, grounded in the bench record (docs/tiered-agent-design.md,
docs/context-management.md, docs/model-scoreboard.md) plus a log-mining pass over
the 33 August docker runs (`benchmark_results/docker_202608*`). Every gate-side
flag is implemented twice (run.rs and repl.rs), so each knob removed is ~2 sites
plus config plumbing. Nothing here has been changed yet — this is the worklist.

### A. Refuted — delete

| Knob / module | Evidence | Footprint |
|---|---|---|
| `flat = true` | strict+flat: Gemma {0,3,3}/6, Devstral 3/6 vs grouped 6/6 | definitions.rs:367, context/mod.rs:109,511, run.rs:485-494, add_param.rs:53 |
| `ceremony = off \| advise` | off 5/6 with smoke FAIL, advise 4/6, strict 6/6 (Qwen, n=6) | context/mod.rs:231,631-693 + run.rs/repl.rs/debugger.rs; keep `strict` as the only mode and drop the enum |
| `gate_context_reset` | A/B OFF 6.0 vs ON 5.67 and 1.6x slower; 21/33 August runs had it silently ON via the docker-script confound (fixed 2fdc48c) | run.rs:1968-1989 + repl.rs mirror, spiral.rs `GATE_RESET_AFTER_BLOCKS`; docs/context-management.md:103-106,160 still says "on (bench)" — fix the doc when removing |
| `CompactionStrategy::TieredSmart` | 5.3, inert-to-harmful vs tiered 5.5 | compressor.rs (tiered branch arg), config enum variant |

### B. Superseded — delete after tagging

All of these were built before the judge stack (`debugger_judge` + `debugger_judge_rewind`,
default ON) and the loop-detector fixes (period-2/3/4 cycles, stale LSP, gap 7/8). Deep-dive
per item, including what would still be worth trying:

**`gate_restart`** — "restart from clean tree after N blocks". The judge is a strict superset:
August verdicts SCRAP 4 / REWIND 5 / CONTINUE 1, i.e. the judge already does restart (SCRAP)
*and* the single-file rewind that restart lacks, with a diagnosis attached. Nothing worth
trying; delete (run.rs:1759-1797 + repl.rs mirror).

**`gate_replan`** — goal re-anchor on the first non-compile block (run.rs:1802-1830). The judge
prompt already carries the GOAL verbatim (debugger.rs:490, "judge whether the work is on-path
for the GOAL") and its CONTINUE verdict emits an anchored PLAN; the judge fires at block 2
(`DEBUGGER_TRIGGER_BLOCKS = 2`), replan at block 1 and only for test-fails. The only
differential is "one block earlier". The paused A/B fixture still exists
(`benchmark_results/_fixtures/run2-depoisoned`, baseline 0/15 smoke). Worth trying ONLY as
the paused experiment, and only if plan degradation is seen again in a judge-on run; none of
the August failures were plan-degradation (they were edit-mechanics grinds, see revert_to_green
below). Recommendation: delete, keep the fixture.

**`revert_to_green`** — `REVERT_TO_GREEN_BLOCKS = 6` (run.rs:925) is a plain consecutive
red-round counter (run.rs:1153-1170): any 6 rounds with LSP errors above baseline reverts the
tree to `last_green_round`. Measured above-baseline streaks in August: failing runs 167
(nemotron 012527), 379 (devstral 124659), 253 (devstral 114957), 55 (gemma 001405), 49
(nemotron 215504) — but PASSING runs 33 (devstral 164918, 6/6) and 14 (laguna 172958, 6/6).
The feature requires ~15-35 red rounds of correct mid-feature work, so threshold 6 would have
destroyed both passing runs. As designed it is refuted without a bench. What IS real: the
three 167-379 grinds are the one failure class the judge never sees — the model never claims
done, so the gate never blocks and the judge never fires. Worth trying: a NON-CONVERGENCE
trigger instead of a streak — e.g. "error count not strictly decreasing over the last K
edit-rounds" or "same LSP error signature for K rounds" (this overlaps gap 9, the identical
failing call x21). Threshold must clear 33 on passing runs (>=40 separates the n=7 sample but
is overfit); validate as tier-1 replay on the nemotron 012527 / devstral 124659 moments
before any bench. Delete the streak implementation; file the trigger idea under gap 9.

**`spiral_reset`** — `SPIRAL_REVERT_THRESHOLD = 3` reverts + reset message (spiral.rs, 173
lines; run.rs:716/3108-3128 + repl.rs). Revert-loop spirals existed mainly before the gap-7/8
fixes: Devstral 114957 had 5 loop-notes / 62 reverts, 032502 2 / 67; post-fix runs show 0
notes. The loop detector already emits `[loop] 'to rev_N' recurred 4x` for exactly this
pattern, so the trigger is absorbed; the only unique ingredient left is the reset MESSAGE
framing (names what failed + forces a replan + concrete next step), which probed 8/8. Worth
trying: reuse that framing as the loop-detector hint text for `rev_N` keys (no flag, no
module), tier-1 replay on the 114957 / 032502 moments. Then delete spiral.rs and the flag.

**Compaction arms `sliding_window`, `rolling_summary`, `observation_masking`, `tiered_rolling`**
— gemma matrix (n=3): unified 6.0/1372s, sliding 5.7/3078s, rolling 5.3, masking 5.3 (thrash;
0/6 collapse with the cascade pre-8651650), tiered 5.5, tiered_rolling 5.5 at 4x less churn,
lazy 3x 6/6 and now default. Matrix v2 (compaction_20260704_164112) never finished. Keep
`unified` (the fallback), `tiered` (the keeper) and `lazy`; keep `mask_old_observations`
(compressor.rs:909) because tiered uses it; delete compact_sliding_window (827),
compact_rolling_summary (745), compact_observation_masking (946), TieredRolling. Tag the
pre-removal commit (the blog post in posts/ cites these arms) so the numbers stay
reproducible. Worth trying — the one live idea from the matrix: observation masking as tier-1
INSIDE lazy (reactive trigger: mask first, LLM-summarize only if still over budget), which
is the tiered-compaction memory note; needs a >3-run A/B against plain lazy.

### C. Abandoned — decision, not experiment

**`edit_mode = smart`** — the edit_file inner planner (edit_file/mod.rs 3575 lines +
apply.rs 856 + parse.rs 657, ~5.1k lines, 47 refs in 11 files). Fast mode has been the
default since fc80dbb (2026-04) and every win since (cascade, revert-hint, diff-echo, judge
rewind) is fast-mode-only and gated on `edit_mode == Fast`. No bench has run smart in 4
months. Nothing to try — there is no hypothesis under which re-benching it beats investing
in fast-mode edit mechanics (docs/aider-features-plan.md). Blocker: edit_file/mod.rs and
apply.rs carry uncommitted hunks; resolve those first, then remove. refactor/model_edit.rs
stays (add_param/drop_param use it in fast mode).

### D. Keep (earning their place)

auto_revert_ast_cascade, reactive_debugger + debugger_multifire, debugger_judge +
debugger_judge_rewind, plan_gate_debugger, compaction lazy/unified/tiered, skill router +
skill_step_injection (in flight), loop detector, long-running jobs.

### Order of work

1. Tag current HEAD (blog reproducibility), then A (mechanical, ~4 knobs).
2. B: gate_restart, gate_replan, revert_to_green streak — pure deletions.
3. B: spiral_reset — lift the message framing into the loop-detector hint first (tier-1
   replay), then delete the module.
4. B: compaction arms — after confirming nothing in-flight references them.
5. C: after the edit_file hunks land.
Estimated A+B ≈ 600-900 lines and 8 config knobs; C ≈ 5k lines.

## Bench harness: derive the "Model:" header from the server, not a constant

`scripts/run-benchmark-docker.sh:74` hardcodes `MODEL="devstral-small-2"`, so the run
header prints devstral for every model (spotted 2026-08-25 on a gemma run whose header
said devstral). The results *directory* is already correct: `MODEL_TAG` (line ~41)
probes `${LLAMA_ENDPOINT}/v1/models` and self-labels the dir.

Fix: derive the header from the same probe instead of the constant — reuse `MODEL_TAG`
(or the raw `/v1/models` id) for the `Model:` line at line 117, keeping the constant only
as the fallback when the server isn't reachable. Cosmetic today, but the header is what
gets pasted into findings/scoreboard rows, so a wrong label there is a real
misattribution risk.
