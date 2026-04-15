# Refactoring TODO: splitting large modules

Audit run 2026-04-15 against the `ci` branch. Totals: ~24.5k lines across
67 Rust files. The tree has four files over 800 lines and pervasive
REPL↔run-loop duplication — both of which make edits hazardous (easy to
miss a call site, easy to regress on drift).

This is a plan, not a commitment. Split where the seams are already
visible; don't invent abstractions for hypothetical future needs.

---

## Module-layout convention: pattern B

**All new modules and splits use pattern B: `foo.rs` as the module root,
`foo/submodule.rs` for submodules.** Do **not** create new `foo/mod.rs`
files.

```
# Pattern B (preferred — used for all new work)
src/knowledge/indexer.rs          # module root
src/knowledge/indexer/walker.rs   # submodule
src/knowledge/indexer/summary.rs

# Pattern A (legacy — 13 existing files, see migration section below)
src/tools/mod.rs
src/tools/read_file.rs
```

Pattern B is the Rust 2018+ default. It avoids the "15 tabs all named
`mod.rs`" IDE problem and makes `git grep foo` / "go to file" land in
the right place. The stdlib still has `mod.rs` in some places for
historical reasons, but most crates written post-2018 use pattern B
(tokio, reqwest, ripgrep, clap, serde).

**Existing 13 `mod.rs` files stay as-is during these refactorings** —
converting them is its own separate task (see
[Cleanup: migrate legacy mod.rs files](#cleanup-migrate-legacy-modrs-files)
at the bottom).

---

## Priority 1 — REPL ↔ run-loop duplication

`src/cli/commands/repl.rs` (1840 lines) and `src/cli/commands/run.rs`
(1052 lines) both re-implement:

| Helper | repl.rs | run.rs |
|--------|---------|--------|
| `loop_detected_hint` | 49 | 49 |
| `truncated_tool_call_hint` | 60 | 60 |
| `summarize_args` | 1185 | 876 |
| `loop_call_key` | 1282 | 963 |
| `canonical_json` | 1403 | 967 |
| `permission_action` | 1286 | *(likely also present)* |

Plus similar masking logic (`mask_old_tool_results` at repl.rs:1091 is
documented as "same logic as run.rs"). Two copies of this is how the
*actual* bugs get introduced — a fix lands in one and not the other.

Propose a new shared module at `src/cli/commands/agent.rs` with
submodules under `src/cli/commands/agent/`:

```
src/cli/commands/agent.rs               # re-exports + the shared `AgentState` if any
src/cli/commands/agent/hints.rs         # loop_detected_hint, truncated_tool_call_hint
src/cli/commands/agent/display.rs       # summarize_args
src/cli/commands/agent/loop_detector.rs # loop_call_key, canonical_json
src/cli/commands/agent/permissions.rs   # permission_action
src/cli/commands/agent/masking.rs       # the shared observation-masking routine
```

`repl.rs` and `run.rs` become thin shells over this module. Aim to shrink
both below 800 lines.

This is the highest *correctness* risk on the list — a drift bug waiting
to happen.

---

## Priority 2 — `src/knowledge/indexer.rs` (841 lines)

Cleanest seam in the tree: six separate symbol extractors sharing a
walker. Split by language.

```
src/knowledge/indexer.rs            # module root (walker dispatch, re-exports)
src/knowledge/indexer/walker.rs     # index_project, reindex_file, file_mtime, audit_file_sizes
src/knowledge/indexer/end_line.rs   # count_braces_outside_strings, compute_end_lines
src/knowledge/indexer/summary.rs    # generate_summary, extract_doc_header, truncate_summary
src/knowledge/indexer/lang.rs       # lang sub-module root (extract_symbols dispatch)
src/knowledge/indexer/lang/common.rs # extract_name_after
src/knowledge/indexer/lang/rust.rs   # extract_rust_symbols
src/knowledge/indexer/lang/python.rs # extract_python_symbols
src/knowledge/indexer/lang/js_ts.rs  # extract_js_ts_symbols
src/knowledge/indexer/lang/go.rs     # extract_go_symbols
```

Bonus: makes the regex-based extractors easy to eventually replace with
tree-sitter parsers one language at a time (compare `ts_extract.rs`,
which already does this for some).

---

## Priority 3 — `src/context/compress.rs` (762 lines)

Five distinct operations share a file:

```
src/context/compress.rs                # module root + re-exports
src/context/compress/format.rs         # strip_code_format, strip_inline_comment
src/context/compress/reading.rs        # compress_for_reading, detect_license_header, is_stdlib_import
src/context/compress/imports.rs        # elide_std_imports
src/context/compress/profile.rs        # compress_profile
src/context/compress/tool_result.rs    # summarize_tool_result, extract_symbol_names_from_content
```

Each is ~100–250 lines with no cross-dependencies beyond primitives.

---

## Priority 4 — `src/tools/plan.rs` (605 lines)

Already split conceptually (step types, architecture hints, validation,
each action handler).

```
src/tools/plan.rs                # module root: plan_exists, load_plan, failure_hint, re-exports
src/tools/plan/step.rs           # Step struct + parsing / serialization
src/tools/plan/hints.rs          # ARCHITECTURE_REVIEW_HINT_* constants
src/tools/plan/validate.rs       # validate_steps
src/tools/plan/actions.rs        # the `execute` action dispatch (split further if it grows)
```

---

## Priority 5 — `src/lsp/servers.rs` (696 lines)

Mixes server enum, verification, platform detection, HTTP downloads, and
package-manager installers:

```
src/lsp/servers.rs               # LspServer enum + ensure_binary, lsp_cache_dir
src/lsp/servers/platform.rs      # platform_triple, find_in_path, has_c_sources
src/lsp/servers/verify.rs        # VerifyResult, verify_binary_verbose
src/lsp/servers/download.rs      # HTTP downloaders (rust-analyzer, gopls, etc.)
src/lsp/servers/install.rs       # npm_install, go_install
```

---

## Priority 6 (optional) — `src/tools/fast/feedback.rs` (639 lines)

Cohesive single-purpose module (render edit feedback). Only worth
splitting if it grows more responsibilities. Suggested shape if we do:

```
src/tools/fast/feedback.rs                 # EditFeedback + build_feedback
src/tools/fast/feedback/diagnostics.rs     # render_file_diagnostics
src/tools/fast/feedback/revisions.rs       # render_revision_table, render_live_row,
                                           # render_tombstone_*, outcome_tag,
                                           # truncate_preview, render_label
```

---

## Parked — `src/tools/edit_file/` (3557 + 852 lines)

`edit_file/mod.rs` (3557 lines) and `edit_file/apply.rs` (852 lines) are
the two largest files in the tree, but splitting them is deferred until
the surrounding modules are cleaner. When we revisit, the target shape is
below — note the layout also converts the current `edit_file/mod.rs` →
`src/tools/edit_file.rs` (pattern B).

### `edit_file/mod.rs` split

| New file | Contents |
|----------|----------|
| `src/tools/edit_file.rs` | `pub async fn execute` entry + public re-exports + `SplitResult` |
| `src/tools/edit_file/types.rs` | `ValidationError`, `LspErrorLocation`, `LspRegression`, `RepairContext`, `PatchResponse`, `PreplanOutcome`, `PreplanResult`, `PlannedExecutionFailure`, `LspValidationMode`, `InspectionCommand`, `PreplanWindowResponse`, `InspectionCounters` |
| `src/tools/edit_file/execute_preplan.rs` | `execute_preplanned_steps` |
| `src/tools/edit_file/execute_plan.rs` | `execute_planned_steps`, `execute_smart_step` |
| `src/tools/edit_file/validation.rs` | `validate_candidate_for_write`, `validate_candidate_with_lsp`, `build_lsp_regression`, `diagnostics_for_current_file`, `error_diagnostics`, `validate_candidate`, `validate_steps_in_file`, `gate_truncation` |
| `src/tools/edit_file/repair.rs` | `build_retry_feedback`, `is_signature_mismatch_error`, `format_repair_context`, `format_prior_applied`, `format_prior_failed`, `format_repair_steps_for_window`, `format_lsp_regression_for_planner` |
| `src/tools/edit_file/preplan_parser.rs` | `parse_preplan_window_response`, `extract_line_ranges_from_note`, `has_case_insensitive_prefix`, `strip_case_insensitive_prefix`, `execute_inspection_commands`, `search_in_file`, `read_in_file`, `render_numbered_slice`, `extend_unique_notes`, `append_inspection_result` |
| `src/tools/edit_file/llm_requests.rs` | `request_patch`, `request_patch_for_region`, `request_preplan_steps` |
| `src/tools/edit_file/windows.rs` | `build_windows` and friends |

### `edit_file/apply.rs` split

```
src/tools/edit_file/apply.rs           # apply_patch_dry_run, apply_literal_replace_in_scope,
                                       # apply_resolved_patch
src/tools/edit_file/apply/similarity.rs # ws_squash, edit_distance, line_similarity
src/tools/edit_file/apply/resolve.rs    # ResolvedOp, ResolvedKind, resolve_ops,
                                        # resolve_old_anchor, find_exact_block_matches,
                                        # find_trimmed_block_matches
src/tools/edit_file/apply/preview.rs    # preview_anchor, preview_block, format_line_list,
                                        # op_label, display_span
src/tools/edit_file/apply/validate.rs   # reject_overlapping_spans, validate_insert_line
```

---

## What NOT to split

These are large but cohesive and touching them would pessimise code
navigation:

- `src/tools/edit_file/parse.rs` (657 lines) — one parser for one format.
- `src/llm/mod.rs` (535 lines) — streaming client, splitting would fragment the SSE handling.
- `src/tui/ui.rs` (536 lines) — one render function plus helpers, hard to cut.
- `src/config/mod.rs` (513 lines) — one schema.
- `src/tools/permissions.rs` (510 lines) — one policy engine.

Re-evaluate if any crosses ~800 lines.

---

## Cleanup: migrate legacy `mod.rs` files

Separate task, lower priority, do after the priority-1–6 splits above so
we're not fighting interleaving. Thirteen existing files to migrate:

```
src/cli/commands/mod.rs          → src/cli/commands.rs
src/cli/mod.rs                   → src/cli.rs
src/config/mod.rs                → src/config.rs
src/context/mod.rs               → src/context.rs
src/knowledge/mod.rs             → src/knowledge.rs
src/llm/mod.rs                   → src/llm.rs
src/lsp/mod.rs                   → src/lsp.rs
src/mcp/mod.rs                   → src/mcp.rs
src/runtime/mod.rs               → src/runtime.rs
src/tools/edit_file/mod.rs       → src/tools/edit_file.rs  (handled by the parked edit_file split)
src/tools/fast/mod.rs            → src/tools/fast.rs
src/tools/mod.rs                 → src/tools.rs
src/tui/mod.rs                   → src/tui.rs
```

Each is a pure `git mv` with no content change. Do it as one PR (or a
small series grouped by top-level directory) once the split-out work
above is landed so we don't have ongoing edits conflicting with the
renames. `git log --follow` handles the rename transparently.

Verify each move with:
- `cargo test`
- `cargo clippy --all-targets -- -D warnings`
- Check no `include_str!` / `include_bytes!` inside these files use
  relative paths that break on rename (none today, per a quick grep).

---

## Sequencing

Ordered by value-per-risk, not LOC:

1. **REPL ↔ run-loop dedup** (P1). Highest bug-avoidance value —
   duplicated helpers are already a live drift source.
2. **`indexer.rs` by language** (P2). Cleanest seam, easy win, enables
   incremental tree-sitter migration.
3. **`compress.rs`** (P3). Small blast radius, good follow-up.
4. **`plan.rs`** (P4). Medium lift, clear seams.
5. **`lsp/servers.rs`** (P5). Medium lift.
6. **`feedback.rs`** (P6, optional). Only if it grows.
7. **Legacy `mod.rs` migration** (cleanup). After the above so the
   renames don't conflict with in-flight splits.
8. **`edit_file/` split** (parked). Revisit when the surrounding modules
   are clean.

Each split should be its own PR with:
- `cargo test` green before and after (no semantic change)
- `cargo clippy --all-targets -- -D warnings` green (CI already enforces)
- One commit per file move so `git log --follow` still works
- Pattern B for every new file (never `mod.rs`)

No behaviour changes inside split PRs — pure code motion. Improvements to
the split-out code belong in follow-up PRs so review stays tractable.
