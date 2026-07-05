# Tree-sitter migration plan (refactor tool's span extraction)

Status: **planning only, not started**. Follow-up to the hand-rolled
lexer-aware scanner in `src/tools/refactor/sites.rs` (`signature_old_block`,
`callsite_old_block`, `balanced_parens`) — see git history around
2026-07-05 for the bug trail (parens-in-strings, non-call references, Go
receivers, Go backtick strings) that motivated this. That scanner is now
fully tested (40 tests, all passing) and live-validated, but the repeated
whack-a-mole pattern is itself the argument for replacing the *mechanism*,
not just patching it further.

## Why

We keep discovering new syntax our hand-rolled scanner doesn't understand
(raw strings, template literals, receiver clauses...). A real parser
eliminates the whole bug class instead of patching instances of it.
Precedent: ast-grep and difftastic both use exactly this approach.

## Scope

Replace the *internals* of two functions; keep their signatures and
fallback contract (`Option<String>`, `None` on any uncertainty → caller
falls back to the model-transcribes-OLD path) unchanged, so nothing
downstream (`model_edit.rs`, `add_param.rs`, `drop_param.rs`) needs to
change:

- `signature_old_block(source, from_line) -> Option<String>`
- `callsite_old_block(source, line_0, column_0) -> Option<String>`

Out of scope: `rename` (pure LSP, no text extraction involved).
Optional/lower-priority: `signature_has_param`'s naive comma-split param
check could also move to tree-sitter for the same robustness reasons, but
it's false-negative-only today (safe), so it can be a fast-follow rather
than blocking this migration.

## Versions to use

Confirmed available and current on crates.io as of this research
(2026-07-05) — pin close to these, verify nothing newer breaks at
implementation time:

| crate | version | notes |
|---|---|---|
| `tree-sitter` | `0.26` | core; only a dev-dependency for grammar crates themselves |
| `tree-sitter-rust` | `0.24` | |
| `tree-sitter-python` | `0.25` | |
| `tree-sitter-go` | `0.25` | |
| `tree-sitter-typescript` | `0.23` | exposes both TSX and plain TS grammars — use the plain TS one (`LANGUAGE_TYPESCRIPT`), confirm exact export name at implementation time |
| `tree-sitter-javascript` | `0.25` | |
| `tree-sitter-java` | `0.23` | |

Grammar crates depend on the ABI-stable `tree-sitter-language` crate, not
the full `tree-sitter` crate, so they don't lag/break against core version
bumps — this was the main version-skew risk and it's designed away.

Build-time cost: grammars ship generated C source compiled via `cc` in
their own `build.rs` (static link, no runtime dependency for end users).
The project *already* has `cc` as a transitive dependency and a C compiler
in both the dev environment and the benchmark Docker image, so this isn't
a new category of requirement — just more of it.

## What to change

1. **`Cargo.toml`**: add the 6 grammar crates + `tree-sitter` above.
2. **New submodule** `src/tools/refactor/ast_span.rs` (logic lives here,
   not in `mod.rs`, per house rules):
   - `enum Lang { Rust, Python, Go, TypeScript, JavaScript, Java }`
   - `fn detect_lang(path: &Path) -> Option<Lang>` — by extension
     (`.rs`, `.py`, `.go`, `.ts`/`.tsx`, `.js`/`.jsx`/`.mjs`, `.java`).
   - `fn ts_language(lang: Lang) -> tree_sitter::Language`
   - Per-language node-type tables (**verify these exact names against a
     real parse at implementation time** — the list below is from research,
     not yet confirmed against actual grammar output):
     - call node: `call_expression` (Rust/Go/JS/TS), `call` (Python),
       `method_invocation` (Java)
     - definition node: `function_item` (Rust), `function_definition`
       (Python), `function_declaration` (Go/JS/TS — also
       `method_definition` for JS/TS class methods, needs checking),
       `method_declaration`/`constructor_declaration` (Java)
   - `fn node_span_at(source: &str, lang: Lang, byte_offset: usize, node_types: &[&str], field: &str) -> Option<(usize, usize)>`:
     parse source, `root.descendant_for_byte_range(byte_offset, byte_offset)`,
     walk `.parent()` until a node whose `.kind()` is in `node_types`,
     return that node's `child_by_field_name(field)` byte range (or the
     whole node's range if the field lookup fails — needs a decision at
     implementation time on which is more correct for the "OLD block"
     convention, which currently includes trailing content like `) -> T {`
     on the signature's last line).
3. **Rewrite `signature_old_block`/`callsite_old_block`** in `sites.rs` to:
   detect language from the file extension (need the path threaded in —
   currently these take `source: &str` only, so their call sites in
   `add_param.rs`/`drop_param.rs` need to also pass `path`), call
   `node_span_at`, convert the resulting byte range to a line range
   (count `\n` before each offset), and return the same whole-line-joined
   `String` as today. Fall back to `None` (safe, existing contract) if
   `detect_lang` fails, parsing fails, or no matching ancestor node is
   found.
4. **Delete** `balanced_parens`, `scan_balanced_parens_from`,
   `skip_generic_args`, and the `ScanState` enum entirely once the new
   path is validated — no dead code, no dual-path hedge (per house rules:
   don't keep both a new and old implementation "just in case").
   `signature_has_param` still needs *something* for its param-name
   check; either keep its current naive-but-safe comma-split (documented
   false-negative-only) or upgrade it to walk the parameter-list node's
   children — decide at implementation time, not blocking.

## How to organize this

One self-contained commit/PR, not a gradual dual-path rollout — the
existing `Option<None>` fallback already makes this safe to land in one
shot, and half-finished hedges are exactly what CLAUDE.md tells us to
avoid. Suggested order within that one piece of work:
1. Add dependencies, get one language (Rust) working end-to-end against
   the existing Rust test cases first — fastest feedback loop, and it's
   the language this codebase itself is written in.
2. Add the other 4 languages once the Rust path's shape is proven.
3. Port all existing tests, add new ones (see below), delete old scanner.
4. Re-run the live gemma replay test as the final acceptance gate.

## How to test it

- **Port every existing test as-is**: all 27 tests in
  `signature_old_block_tests`, `callsite_old_block_tests`, and
  `tricky_callsite_tests` must still pass against the tree-sitter-backed
  implementation, unchanged (same inputs, same expected outputs). Any of
  these that only passed "by luck" in the hand-rolled version should now
  pass for the right reason.
- **New cases the hand-rolled version couldn't fully guarantee**: the
  specific residual risk noted 2026-07-05 — a Rust `r#"..."#` raw string
  containing an *odd* number of internal quote characters positioned
  asymmetrically relative to a paren. Also: Python decorators above a
  `def`, TypeScript function overload signatures, Java method annotations
  (`@Override`), Go interface method declarations, deeply nested generics.
- **Differential check against `syn`**: for a handful of real function
  definitions/call sites pulled from miniswe's own `src/` tree (a real,
  large, honestly-formatted Rust corpus we already have on disk), confirm
  the extracted span is exactly what a human would select — spot-check by
  eye is probably enough for the plan; a stronger version would feed the
  computed OLD block back through `syn::parse_str` on a wrapped fragment
  to confirm it's syntactically well-formed, for a batch of real
  callsites/signatures rather than hand-picked ones.
- **Live re-validation**: re-run `tests/e2e_refactor_replay.rs` (the 3
  historical bench failures, replayed against real gemma) against the new
  implementation as the final go/no-go check — it must still show all 3
  signature rewrites succeeding (2 of 3 fully compiling; the 3rd's
  remaining failure is the historical agent's own bad `callsite_fill_in`
  choice, unrelated to span extraction, and isn't expected to change).
- **Full suite**: `cargo test`, `cargo clippy --all-targets`, `cargo fmt`
  must all stay clean, per CLAUDE.md, before considering this done.
