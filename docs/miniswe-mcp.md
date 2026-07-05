# miniswe-mcp

`src/bin/miniswe-mcp/` is a second binary built on top of the `miniswe` library
crate (the crate has always been split as `src/lib.rs` + `src/main.rs`, so
adding a second `[[bin]]` needed no restructuring). It exposes a slice of
miniswe's own tooling as an MCP (Model Context Protocol) server over stdio,
so any MCP-capable coding agent — Claude Code in particular — can call it
directly instead of paying for the same work in its own, much more expensive,
tokens.

## Why this exists

Claude Code burns a large share of its tokens on two patterns miniswe already
solves cheaply and deterministically:

- **Exploration**: "where is X defined / used" resolved via several rounds of
  grep → read → reason, instead of one LSP query.
- **Multi-file mechanical edits**: adding/removing a function parameter means
  hand-editing every callsite one at a time, instead of one atomic call.

Both are already implemented in miniswe as plain, standalone async functions
(`tools::execute_tool`, `tools::execute_refactor_tool`) with no dependency on
the interactive agent loop's worker pool — so wrapping them in an MCP server
was mostly plumbing, not new logic. One thing was deliberately **not** ported:
the debugger-as-judge sub-agent. Its value in miniswe comes from pairing a
structured diagnosis step with a *weak* local model; Claude Code's own model
already reasons well about build/test failures, so that trade doesn't carry
over — and it's also the one piece deeply coupled to miniswe's own
worker-pool/LLM-worker internals, unlike the code-intel/refactor functions.

## Tools exposed

Both are pulled **verbatim** from `tools::definitions` — no schema is
redefined for MCP:

- `code` (`tools::definitions::tool_definitions`) — `goto_definition`,
  `find_references`, `diagnostics`, `repo_map`, `project_info`,
  `architecture_notes`.
- `add_function_param`, `drop_function_param`, `rename_symbol`
  (`tools::definitions::flat_refactor_tool_definitions`) — the flat,
  single-purpose refactor tools already used by `tools.flat` mode. Their
  arguments are normalized via `flat_to_refactor_args` into the grouped shape
  `execute_refactor_tool` consumes.

Both dispatch into the **same** production entry points the interactive
`miniswe` agent loop uses (`execute_tool`, `execute_refactor_tool`) — there is
no parallel implementation to keep in sync.

`add_function_param`/`drop_function_param` do call out to an LLM (via
`ModelRouter`) to rewrite the signature and each callsite — LSP is only used
to *locate* the definition and callsites. Point the server at the same local
endpoint miniswe itself uses for this to work; `rename_symbol` and `code` are
pure LSP/index reads and need no LLM.

## Telemetry

`src/bin/miniswe-mcp/telemetry.rs` appends one JSONL line per tool call, plus
`start`/`stop` lifecycle lines, to:

```
~/.miniswe/projects/<slug>/mcp.log
```

`<slug>` is the project's absolute path with `/` replaced by `-`. This lives
under the home directory rather than the project's own `.miniswe/` so that one
`miniswe-mcp` entry in a global MCP config can serve any number of different
project directories, with each project's usage still easy to find without
needing a separate lookup.

Each `tool_call` line carries `{tool, outcome: "ok"|"error", duration_ms,
detail}` (`detail` is a truncated error excerpt, present only on failures);
each `stop` line carries a `summary` of that session's per-tool `{ok, error}`
counts, so a quick tally doesn't require aggregating the whole file. There is
no dedicated "read my telemetry" tool yet — see the note in the last section.

## Error rendering: `ToolDetail`

The interesting design problem this server ran into: `execute_tool` and
`execute_refactor_tool` are **shared** with the interactive `miniswe` agent
loop, and several of their error/validation messages reference miniswe's own
tool names — `file(action='search')`, `file(action="revert", ...)`, or a
copy-pasteable example in the *grouped* `refactor(action="add_param", ...)`
syntax. None of those tools exist in an MCP client's tool list. Left as-is, a
validation error would point Claude Code at a tool call that doesn't exist —
wasting a turn at best, producing a hallucinated retry at worst.

The fix is **not** string-sanitizing the returned text after the fact — that
would be fragile (matching against exact substrings of prose that can change)
and, for `add_param`/`drop_param` specifically, actively risky: some of that
wording is probe-validated (see the comment in
`src/tools/refactor/add_param.rs` above the "Next: the placeholder
callsites..." message — a specific replay experiment measured it as the only
wording that produced follow-up callsite edits, and the comment explicitly
says not to reword it without re-probing). Editing that string at all, even
just to split it, was out of the question.

Instead, `ToolResult` (`src/tools/mod.rs`) gained a second, purely additive
field:

```rust
pub struct ToolResult {
    pub content: String,           // unchanged — still miniswe's own message
    pub success: bool,
    pub detail: Option<ToolDetail>,  // new — structured facts, no wording
}

pub enum ToolDetail {
    LspUnavailable,
    InvalidArgs { action, missing, bad_type, unknown },
    PartialSignatureChange { action, total, succeeded, callsite_failures, callsite_report },
}
```

The producing code (`code_intel.rs`, `refactor/validation.rs`,
`refactor/add_param.rs`, `refactor/drop_param.rs`) attaches a `ToolDetail`
alongside the existing message **without changing that message at all** —
`content` is byte-for-byte what miniswe's own agent loop has always seen, so
the native path has zero behavior change (confirmed by the full test suite,
including the LSP-backed `e2e_refactor.rs` cases that exercise these exact
code paths).

`miniswe-mcp`'s `errors.rs` is the other consumer: when `ToolResult.detail` is
present, it renders **its own** message from the structured facts — using its
actual tool names (`add_function_param(...)` instead of
`refactor(action="add_param", ...)`), and its actual recovery story (no
revert tool is wired into this server, so the guidance points at `git diff`/
`git checkout` instead of `file(action="revert", ...)`). When `detail` is
absent, it falls back to `content` as-is (most tool outcomes need no
consumer-specific rendering at all — e.g. `rename`'s errors already say
`code(find_references)`, which is valid for both consumers since both expose
the same `code` tool).

This generalizes: any future consumer with yet another tool surface adds its
own renderer over the same `ToolDetail` facts, and any future situation that
needs consumer-specific wording gets its own `ToolDetail` variant — the
"common part" is the facts, "the message" is always the consumer's job.

### Known gap

`flat_to_refactor_args` fills every grouped string field via
`.unwrap_or("")`, so a flat-tool call missing `function`/`param`/`call_value`
never reaches `validate()` as a *missing* key — it arrives as an empty
string, passes the "is this a string" check, and fails later with a more
confusing "no function named `` defined" error instead of a clean
"missing required parameter" one. Pre-existing behavior, not introduced by
this work; worth a follow-up if it shows up in practice.

### Not yet done

There's no `mcp_stats`-style tool that reads `mcp.log` and answers "how's
this server doing" from inside a conversation — right now that requires
looking at the file directly. Follow-up if/when needed.
