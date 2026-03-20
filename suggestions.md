# miniswe: Rust-Specific Improvements

## Problem
miniswe struggles when working on Rust projects (specifically: building a REST framework library). Three root causes: (1) the repo map discards `impl` block information that is critical for understanding Rust code, (2) there is no automatic compiler feedback after file edits, (3) the `edit` tool is hidden but `write_file` fails in practice because devstral-small-2 refuses to output complete files — it writes partial content and stops.

## Change 1: Enhance tree-sitter extraction to capture impl blocks with their methods

**Files:** `src/knowledge/ts_extract.rs`

- The current extraction uses `tree-sitter-tags` which captures methods inside impl blocks as standalone `function` definitions, losing the crucial `impl Trait for Type` grouping context
- Add a new field to `Symbol`: `parent_impl: Option<String>` — stores the impl header signature (e.g., `"impl<T: Clone> Service<Request> for Router<T>"`) for methods that belong to an impl block
- After the existing `tree_sitter_tags` extraction pass (which already gets functions/methods), add a **second pass** using raw `tree-sitter` parsing for `.rs` files only:
  - Parse the file with `tree_sitter::Parser` using `tree_sitter_rust::LANGUAGE`
  - Walk the tree looking for `impl_item` nodes
  - For each `impl_item`: extract the full source text from the start of the node to the opening `{` — this is the impl header signature (e.g., `impl<T> Handler for MyRouter<T> where T: Clone`)
  - For each `function_item` child inside the impl's `declaration_list`: match it by line number to the already-extracted symbols and set their `parent_impl` field to the impl header
  - Also extract `impl_item` nodes themselves as symbols with `kind: "impl"`, using the impl header as the signature. The `name` should be the type being implemented (e.g., `Router` from `impl Handler for Router`)
- For trait definitions (`trait_item`): extract associated types (`type_alias` children) and add them as symbols with `kind: "type"` and `parent_impl` set to the trait name
- This second pass is additive — it enriches the existing extraction, doesn't replace it

## Change 2: Render impl blocks as grouping headers in the repo map

**Files:** `src/knowledge/repo_map.rs`, `src/knowledge/mod.rs` (Symbol struct)

- Add `parent_impl: Option<String>` to the `Symbol` struct in `mod.rs` (with `#[serde(default)]` for backward compat)
- In `render()` and `find_tier1_cutoff()`: **remove** the `.filter(|s| s.kind != "impl")` filters (lines 77-79 and line 153)
- Change the Tier 1 rendering to group methods under their impl headers:
  - Sort symbols within a file: impl symbols first (sorted by line), then methods grouped under their parent_impl, then standalone symbols
  - When rendering Tier 1, if a symbol has `parent_impl = Some(header)` and this header hasn't been printed yet for this file, print `│ {header}` as a grouping line first, then indent the method: `│   {signature}`
  - Standalone impl symbols (kind == "impl") that have no children can be rendered as a single line: `│ impl Trait for Type` (no indented children)
- Example desired output for a Rust file:
  ```
  src/router.rs:
  │ pub struct Router<T>
  │ impl<T: Clone> Router<T>
  │   pub fn new() -> Self
  │   pub fn add_route(&mut self, path: &str, handler: impl Handler)
  │ impl<T: Clone> Service<Request> for Router<T>
  │   fn call(&self, req: Request) -> Response
  ```
- For Tier 0 (names only): include impl headers in the comma-separated list, e.g., `impl Service for Router, new, add_route, call`

## Change 3: Auto-run `cargo check` after Rust file writes

**Files:** `src/tools/mod.rs`, `src/tools/write_file.rs`

- In `execute_tool` for the `"write_file"` and `"edit"` branches: after a successful write to a `.rs` file, automatically run `cargo check --message-format=short 2>&1 | head -30` (same command as the `diagnostics` tool but capped tighter)
- Append the compiler output to the tool result content, prefixed with `\n[cargo check]\n`
- Only do this for `.rs` files (check `path.ends_with(".rs")`)
- If `cargo check` produces no errors, append `\n[cargo check] OK\n` — this is cheap confirmation that prevents the model from wasting a turn calling diagnostics manually
- Keep the timeout short (15 seconds) so it doesn't block the agent loop; if it times out, append `\n[cargo check] timed out\n` and move on
- This saves one full LLM turn per edit cycle (the model no longer needs to decide to call diagnostics — it gets feedback immediately)

## Change 4: Un-hide the edit tool with better guidance for Rust

**Files:** `src/tools/definitions.rs`

- The `edit` tool is currently hidden (comment at line 80-83) because small models abuse it with single-line changes
- Add it back to the tool definitions list but with a Rust-aware description:
  ```
  "Replace a section in a file. `old` must match exactly and be unique in the file. 
   Include 3+ surrounding lines for a unique match. Preferred over write_file for 
   targeted changes to large files (>100 lines). For new files or rewrites, use write_file."
  ```
- This gives the model a path for targeted edits (adding a `Clone` derive, fixing a lifetime, adding an import) without requiring full-file rewrites that it can't/won't complete
- In the system prompt (`src/context/mod.rs`, `build_system_prompt()`), update rule 2 to:
  ```
  2.edit(path,old,new) for targeted fixes;write_file for new files/rewrites
  ```

## Change 5: Add a default Rust lesson

**Files:** `.miniswe/lessons.md` (template), `src/cli/commands/init.rs` (if lessons template is generated there)

- When the project is detected as Rust (Cargo.toml present), seed `.miniswe/lessons.md` with:
  ```markdown
  ## Rust
  - Always return Result<T, E> in library code, never unwrap
  - Prefer impl Trait over dyn Trait in function arguments
  - Use thiserror for library error types, anyhow for applications
  - When borrow checker rejects code: try Clone, Arc, or restructure ownership before adding lifetime parameters
  - Run cargo check after every edit — read the compiler errors carefully, they contain the fix
  - Keep files under 200 lines; split into modules early
  - For async traits: use the async-trait crate or return Pin<Box<dyn Future>>
  - When writing trait impls: read the trait definition first (read_symbol) to understand required methods and associated types
  ```
- This gets loaded into context via the existing keyword-matching in `load_relevant_lessons` whenever the user message contains "rust", "impl", "trait", "borrow", "lifetime", etc.

## Change 6: Cargo workspace awareness in project profile

**Files:** `src/knowledge/profile.rs`

- During `miniswe init`, if `Cargo.toml` contains `[workspace]`: parse the workspace members list and include a "Workspace" section in the generated profile showing the crate dependency DAG
- Example addition to profile:
  ```
  ## Workspace
  - core/ (no deps) — base types, error definitions
  - routing/ (depends: core) — route matching, handler traits
  - middleware/ (depends: core, routing) — middleware chain
  - server/ (depends: core, routing, middleware) — HTTP server lifecycle
  ```
- Parse this from the individual crate `Cargo.toml` files: read `[dependencies]` and check which are path dependencies pointing to sibling crates
- This tells the model which crate to edit first when changes span crates (always edit deps before dependents)

## Priority order
1. Change 3 (auto cargo check) — highest impact, simplest change, ~20 lines
2. Change 4 (un-hide edit tool) — fixes the practical "model won't write full files" problem
3. Change 1+2 (impl block extraction + repo map) — biggest architectural change, biggest payoff for framework-level Rust
4. Change 5 (Rust lessons) — easy, helps immediately
5. Change 6 (workspace awareness) — nice to have for multi-crate projects
