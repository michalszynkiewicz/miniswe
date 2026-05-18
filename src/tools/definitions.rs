//! Tool definitions in OpenAI function-calling format.
//!
//! Tools are grouped into 6 top-level tools to reduce the function list
//! size for small models. Each grouped tool uses an `action` parameter
//! to dispatch to sub-tools. Use `action="help"` to list available actions.

use crate::config::EditMode;
use crate::llm::{FunctionDefinition, ToolDefinition};
use serde_json::json;

/// Return all tool definitions for the LLM.
///
/// `edit_mode` controls which edit tool the descriptions point at: Smart mode
/// references `edit_file` (the LLM-driven planner); Fast mode references the
/// `replace_range` / `insert_at` primitives. Pointing the model at a tool that
/// isn't in its list wastes rounds, so the strings have to match the actual
/// surface — see `cli/commands/{run,repl}.rs` for how the list is filtered.
pub fn tool_definitions(edit_mode: EditMode) -> Vec<ToolDefinition> {
    let (file_edit_hint, write_file_description) = match edit_mode {
        EditMode::Smart => (
            "For code edits use edit_file; for full-file overwrites use write_file.",
            "Create or overwrite a file. Provide complete file contents in `content`, or omit `content` to create a new empty file. Do not use for partial edits to existing code; use edit_file instead.",
        ),
        EditMode::Fast => (
            "For surgical line-precise code edits use replace_range or insert_at; for structural rewrites (e.g. wrapping a block in if-let) use edit_file; for full-file overwrites use write_file.",
            "Create a NEW file. Provide complete contents in `content`. WARNING: regenerating existing files often drops imports/braces/struct fields. For editing existing files, use replace_range/insert_at/refactor instead.",
        ),
    };
    vec![
        // ── refactor: top of list — verb name matches the model's mental
        //    model for "I need to modify a function signature." Empirically
        //    Gemma successfully used a tool named `refactor` (Apr 30 fast
        //    6/6 runs, one call rewrote 16 callsites across 4 files); after
        //    we renamed it to `change_signature` Gemma stopped reaching for
        //    it. Restored the broader name + `callsite_fill_in` field.
        ToolDefinition {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "refactor".into(),
                description: "ATOMIC multi-file refactor — updates the definition AND every callsite \
                    in one call. Use this WHENEVER the task is:\n\
                    • adding a parameter (e.g. 'add a flag', 'add a context arg', 'extend signature with X')\n\
                    • removing a parameter\n\
                    • renaming a function, method, type, or variable across the codebase\n\
                    \n\
                    DO NOT enumerate or edit callsites yourself with replace_range/insert_at for these \
                    tasks — that's manual, error-prone, and the exact thing this tool exists to avoid. \
                    One refactor call is faster and atomic.\n\
                    \n\
                    Target is resolved by `name` via LSP, so exact line/column isn't needed. \
                    Actions: add_param, drop_param, rename. Use action='help' for parameter details \
                    and worked examples.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "One of: add_param, drop_param, rename, help"
                        },
                        "path": { "type": "string", "description": "File containing the function DEFINITION or a reference to the symbol (relative to project root). If you only have a callsite, use code(goto_definition) first." },
                        "name": { "type": "string", "description": "Function/symbol name. Resolved via LSP — works even if your line number is approximate." },
                        "line": { "type": "integer", "description": "1-based line hint. For add_param/drop_param: optional, disambiguates when multiple methods share `name`. For rename: required — line where `name` appears." },
                        "new_param": { "type": "string", "description": "For add_param: the new parameter declaration as it should appear in the signature, e.g. 'system_prompt_override: Option<&str>'" },
                        "position": { "type": "string", "description": "For add_param: where to insert. Either 'start' or 'after:<existing_param_name>'. Use after:<last_param> to append at the end." },
                        "callsite_fill_in": { "type": "string", "description": "For add_param: the literal expression to insert at every existing callsite, e.g. 'None'" },
                        "param": { "type": "string", "description": "For drop_param: the name of the parameter to remove" },
                        "new_name": { "type": "string", "description": "For rename: the new symbol name" }
                    },
                    "required": ["action"]
                }),
            },
        },
        // ── file: core file I/O and shell ─────────────────────────────
        ToolDefinition {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "file".into(),
                description: format!("File operations: read, search, shell, delete, revert. Use action='help' for details. {file_edit_hint}"),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "One of: read, delete, search, shell, revert, help"
                        },
                        "path": { "type": "string", "description": "File path (required for read/delete/revert)" },
                        "start_line": { "type": "integer", "description": "Start line for action='read'" },
                        "end_line": { "type": "integer", "description": "End line for action='read'" },
                        "query": { "type": "string", "description": "Search text (for search)" },
                        "pattern": { "type": "string", "description": "Regex pattern (for search)" },
                        "scope": { "type": "string", "description": "Search scope (for search)" },
                        "max_results": { "type": "integer", "description": "Max results (for search)" },
                        "command": { "type": "string", "description": "Shell command (for shell)" },
                        "timeout": { "type": "integer", "description": "Timeout in seconds (for shell)" },
                        "to_round": { "type": "integer", "description": "Round to revert to (for revert)" }
                    },
                    "required": ["action"]
                }),
            },
        },
        // ── code: LSP + project intelligence ──────────────────────────
        ToolDefinition {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "code".into(),
                description: "Code intelligence: goto_definition, find_references, diagnostics, repo_map, project_info, architecture_notes. Use action='help' for details.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "One of: goto_definition, find_references, diagnostics, repo_map, project_info, architecture_notes, help"
                        },
                        "path": { "type": "string", "description": "File path (for goto_definition, find_references, diagnostics)" },
                        "line": { "type": "integer", "description": "Line number (for goto_definition, find_references)" },
                        "column": { "type": "integer", "description": "Column number (for goto_definition, find_references)" },
                        "keywords": { "type": "string", "description": "Keywords to focus repo map on (for repo_map)" }
                    },
                    "required": ["action"]
                }),
            },
        },
        // ── web: search + fetch ───────────────────────────────────────
        ToolDefinition {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "web".into(),
                description: "Web access: search the web or fetch a URL. General web search uses Serper when configured. If no Serper key is configured, web(search) falls back to GitHub repository search only. Recommended setup: put your Serper key in ~/.miniswe/serper.key. Use action='help' for details.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "One of: search, fetch, help"
                        },
                        "query": { "type": "string", "description": "Search query (for search)" },
                        "url": { "type": "string", "description": "URL to fetch (for fetch)" },
                        "max_results": { "type": "integer", "description": "Max results (for search)" },
                        "selector": { "type": "string", "description": "CSS selector (for fetch)" }
                    },
                    "required": ["action"]
                }),
            },
        },
        // ── plan: structured planning + scratchpad ────────────────────
        ToolDefinition {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "plan".into(),
                description: "Plan and track work. Actions: \
                    'set' (create plan from `steps` or `content` — UNLOCKS the edit tools refactor/edit_file/write_file/replace_range/insert_at), \
                    'check' REQUIRES `step` (1-indexed step number; marks done with compile gate), \
                    'refine' REQUIRES `step` AND `substeps` (replaces target step with substeps), \
                    'show' (view plan), \
                    'scratchpad' (save notes), \
                    'help' (full details). \
                    Call 'set' ONCE early in the session before editing.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "One of: set, check, refine, show, scratchpad, help"
                        },
                        "content": {
                            "type": "string",
                            "description": "For set: plan in markdown. For scratchpad: notes content."
                        },
                        "steps": {
                            "type": "array",
                            "description": "For action='set': structured step list. Each item is {step: string (the action text), compile?: boolean (default true)}.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "step": { "type": "string", "description": "What to do in this step." },
                                    "compile": { "type": "boolean", "description": "Whether the project should compile after this step. Default true." }
                                },
                                "required": ["step"]
                            }
                        },
                        "step": { "type": "integer", "description": "REQUIRED when action is 'check' or 'refine'. 1-indexed step number from the current plan." },
                        "substeps": {
                            "type": "array",
                            "description": "REQUIRED when action is 'refine'. Array of substep objects that replace the target step. Same item shape as the 'steps' array.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "step": { "type": "string", "description": "What to do in this substep." },
                                    "compile": { "type": "boolean", "description": "Whether the project should compile after this substep. Default true." }
                                },
                                "required": ["step"]
                            }
                        }
                    },
                    "required": ["action"]
                }),
            },
        },
        // ── edit_file: LLM-powered code transformation ────────────────
        ToolDefinition {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "edit_file".into(),
                description: "Apply a semantic code change across one file using an atomic patch. Best for multi-line edits, repeated call-site updates, changed function callers, or 5+ similar edits that need per-site reasoning. Supports optional LSP validation.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path relative to project root"
                        },
                        "task": {
                            "type": "string",
                            "description": "What to change and why. Be specific: include types, parameter names, values."
                        },
                        "lsp_validation": {
                            "type": "string",
                            "enum": ["auto", "require", "off"],
                            "description": "Optional validation policy: auto (default) uses LSP if available, require fails if LSP is unavailable or diagnostics worsen, off skips LSP validation."
                        }
                    },
                    "required": ["path", "task"]
                }),
            },
        },
        ToolDefinition {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "write_file".into(),
                description: write_file_description.to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path relative to project root"
                        },
                        "content": {
                            "type": "string",
                            "description": "Complete replacement file contents. Omit only to create a new empty file."
                        }
                    },
                    "required": ["path"]
                }),
            },
        },
    ]
}

/// Return the fast-mode tool definitions (`replace_range`, `insert_at`,
/// `revert`, `show_rev`, `check`). Fed to the router when
/// `tools.edit_mode = "fast"`, replacing `edit_file`. See
/// `docs/fast-mode-design.md`.
pub fn fast_mode_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        // The "smallest range" line is the model-facing fix for a Mistral 4
        // failure mode: when given a wide range it tries to reproduce the
        // surrounding unchanged lines from memory in `content`, drops bits,
        // and silently deletes parts of the file. Keeping the range tight
        // sidesteps that — content only needs to cover what's actually
        // changing.
        ToolDefinition {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "replace_range".into(),
                description: "Replace lines [start..=end] (1-based, inclusive) with `content`. Empty content deletes. Keep the range TIGHT — anything in the range that isn't in `content` is gone. To ADD new lines, use insert_at. For signature changes / renames, use refactor. Per-edit AST+LSP feedback comes back in the response; if you see a regression, call `revert`.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "File path relative to project root" },
                        "start": { "type": "integer", "description": "First line to replace (1-based, inclusive)" },
                        "end": { "type": "integer", "description": "Last line to replace (1-based, inclusive)" },
                        "content": { "type": "string", "description": "Replacement text. Empty string deletes [start..=end]." }
                    },
                    "required": ["path", "start", "end", "content"]
                }),
            },
        },
        ToolDefinition {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "insert_at".into(),
                description: "Insert `content` after line `after_line` (1-based). Use after_line=0 to insert at the top of the file, after_line=<last line> to append. Use `replace_range` when you need to replace or delete existing lines. For adding a parameter to a function (which also requires inserting an argument at every callsite), use `refactor(add_param)` instead — one atomic call beats hand-editing each callsite.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "File path relative to project root" },
                        "after_line": { "type": "integer", "description": "Line to insert after (0 = top of file, N = after current line N)" },
                        "content": { "type": "string", "description": "Text to insert. Cannot be empty." }
                    },
                    "required": ["path", "after_line", "content"]
                }),
            },
        },
        ToolDefinition {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "revert".into(),
                description: "Restore `path` to a named prior revision. Pick the rev number from the revision table attached to every edit's feedback. Reverting to rev_N marks rev_{N+1}.. as tombstones (`[reverted]` rows that stay in the table so you can see what you undid — use `show_rev` for details). The next edit gets a fresh monotonic number, never a recycled one. Tombstones cannot be the target of another revert.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "File path relative to project root" },
                        "rev": { "type": "integer", "description": "Revision number to restore (0 = original / pristine). Must be a live rev, not a tombstone." }
                    },
                    "required": ["path", "rev"]
                }),
            },
        },
        ToolDefinition {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "show_rev".into(),
                description: "Show the full stored details for a specific revision of `path`: operation, arguments, outcome, and the verbatim payload (new_text / inserted text) capped at 2 KB. Works for both live revs and tombstones — useful for deciding whether your next edit would be a byte-identical replay of an edit you already reverted.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "File path relative to project root" },
                        "rev": { "type": "integer", "description": "Revision number to inspect (0 = pristine)" }
                    },
                    "required": ["path", "rev"]
                }),
            },
        },
        ToolDefinition {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "check".into(),
                description: "Run the project's compiler (cargo / tsc / go vet / mvn / gradle) and report errors. Per-edit feedback already shows LSP state; reach for this when you want a deeper, synchronous confirmation that the whole project builds.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {}
                }),
            },
        },
    ]
}

/// Flat single-purpose refactor tools (`tools.flat`). Replace the
/// grouped `refactor{action,position,callsite_fill_in}` — no DSL, each
/// tool one intent with self-evident, all-required params. `after` is
/// the only optional field; omitted = append at end (footgun-free).
pub fn flat_refactor_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "add_function_param".into(),
                description: "Add a parameter to a function and update its definition AND every \
                    callsite in ONE atomic call (resolved by name via LSP). Use for 'add a flag', \
                    'thread a context arg', etc. Do not hand-edit callsites for this."
                    .into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "File containing the function DEFINITION (relative to project root)." },
                        "function": { "type": "string", "description": "Function/method name. Resolved via LSP." },
                        "param": { "type": "string", "description": "Full new parameter declaration as it appears in the signature, e.g. 'shout: bool'." },
                        "call_value": { "type": "string", "description": "Literal expression to pass at every existing callsite, e.g. 'false' or an in-scope variable name." },
                        "after": { "type": "string", "description": "Optional: existing parameter name to insert after. Omit to append at the end (the common case)." },
                        "line": { "type": "integer", "description": "Optional 1-based line hint to disambiguate overloaded names." }
                    },
                    "required": ["path", "function", "param", "call_value"]
                }),
            },
        },
        ToolDefinition {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "drop_function_param".into(),
                description: "Remove a parameter from a function and update its definition AND \
                    every callsite in ONE atomic call (resolved by name via LSP)."
                    .into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "File containing the function DEFINITION." },
                        "function": { "type": "string", "description": "Function/method name." },
                        "param": { "type": "string", "description": "Name of the parameter to remove." },
                        "line": { "type": "integer", "description": "Optional 1-based line hint." }
                    },
                    "required": ["path", "function", "param"]
                }),
            },
        },
        ToolDefinition {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "rename_symbol".into(),
                description: "Rename a function/method/type/variable across the codebase \
                    (definition + all references) in ONE atomic call."
                    .into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "A file where the symbol appears (relative to project root)." },
                        "line": { "type": "integer", "description": "1-based line where `name` appears." },
                        "name": { "type": "string", "description": "Current symbol name." },
                        "new_name": { "type": "string", "description": "New symbol name." }
                    },
                    "required": ["path", "line", "name", "new_name"]
                }),
            },
        },
    ]
}

/// Normalize a flat refactor tool call into the legacy grouped
/// `refactor` args shape that `execute_refactor_tool` already consumes.
/// Returns `None` if `name` is not a flat refactor tool.
pub fn flat_to_refactor_args(name: &str, a: &serde_json::Value) -> Option<serde_json::Value> {
    let s = |k: &str| a.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
    match name {
        "add_function_param" => {
            let after = s("after");
            let position = if after.is_empty() {
                "end".to_string()
            } else {
                format!("after:{after}")
            };
            let mut out = json!({
                "action": "add_param",
                "path": s("path"),
                "name": s("function"),
                "new_param": s("param"),
                "callsite_fill_in": s("call_value"),
                "position": position,
            });
            if let Some(l) = a.get("line") {
                out["line"] = l.clone();
            }
            Some(out)
        }
        "drop_function_param" => {
            let mut out = json!({
                "action": "drop_param",
                "path": s("path"),
                "name": s("function"),
                "param": s("param"),
            });
            if let Some(l) = a.get("line") {
                out["line"] = l.clone();
            }
            Some(out)
        }
        "rename_symbol" => Some(json!({
            "action": "rename",
            "path": s("path"),
            "line": a.get("line").cloned().unwrap_or(json!(0)),
            "name": s("name"),
            "new_name": s("new_name"),
        })),
        _ => None,
    }
}

/// Return the mcp_use tool definition (only added when MCP servers are configured).
pub fn mcp_tool_definition() -> ToolDefinition {
    ToolDefinition {
        r#type: "function".into(),
        function: FunctionDefinition {
            name: "mcp_use".into(),
            description: "Call a tool on a connected MCP server.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "server": {
                        "type": "string",
                        "description": "MCP server name (from [MCP:name] in context)"
                    },
                    "tool": {
                        "type": "string",
                        "description": "Tool name on that server"
                    },
                    "arguments": {
                        "type": "object",
                        "description": "Arguments to pass to the tool"
                    }
                },
                "required": ["server", "tool"]
            }),
        },
    }
}

pub fn spawn_agents_tool_definition() -> ToolDefinition {
    ToolDefinition {
        r#type: "function".into(),
        function: FunctionDefinition {
            name: "spawn_agents".into(),
            description: "Run multiple independent agent sub-tasks concurrently. \
                Each agent gets its own prompt and runs to completion with full tool access. \
                All agents share the same LLM concurrency limit. \
                Returns all agents' outputs when every agent has finished."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "agents": {
                        "type": "array",
                        "description": "List of agents to run",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {
                                    "type": "string",
                                    "description": "Short label for this agent (shown in output)"
                                },
                                "prompt": {
                                    "type": "string",
                                    "description": "Full task prompt for this agent"
                                }
                            },
                            "required": ["label", "prompt"]
                        }
                    }
                },
                "required": ["agents"]
            }),
        },
    }
}

/// Help text for the `file` tool group.
pub fn file_help(edit_mode: EditMode) -> &'static str {
    match edit_mode {
        EditMode::Smart => {
            "\
Available actions for `file`:

- read: Read a file. Params: path (required), start_line, end_line
- delete: Delete an existing file. Params: path (required)
  Example: {\"action\":\"delete\",\"path\":\"src/bin/old.rs\"}
- search: Search codebase. Params: query (literal text, no regex) OR pattern (regex); exactly one required. Also: scope, max_results
- shell: Run a command. Params: command (required), timeout
  Example: {\"action\":\"shell\",\"command\":\"cargo check\",\"timeout\":30}
- revert: Revert files to a previous round. Params: to_round, path (both optional)

For text edits use edit_file (semantic, planner-driven). For full-file overwrites \
use write_file. There is no longer a deterministic search-and-replace action."
        }
        EditMode::Fast => {
            "\
Available actions for `file`:

- read: Read a file. Params: path (required), start_line, end_line
- delete: Delete an existing file. Params: path (required)
  Example: {\"action\":\"delete\",\"path\":\"src/bin/old.rs\"}
- search: Search codebase. Params: query (literal text, no regex) OR pattern (regex); exactly one required. Also: scope, max_results
- shell: Run a command. Params: command (required), timeout
  Example: {\"action\":\"shell\",\"command\":\"cargo check\",\"timeout\":30}
- revert: Revert files to a previous round. Params: to_round, path (both optional)

For partial edits use replace_range or insert_at; if an edit regresses, \
revert that revision instead of layering more edits. For full-file overwrites \
use write_file."
        }
    }
}

/// Help text for the `code` tool group.
pub fn code_help() -> &'static str {
    "\
Available actions for `code`:

- goto_definition: Jump to definition. Params: path, line, column (all required)
- find_references: Find all references. Params: path, line, column (all required)
- diagnostics: Get compiler errors/warnings. Params: path (optional)
- repo_map: Get project structure with signatures. Params: keywords (optional)
- project_info: Get project metadata, guidelines, lessons. No params.
- architecture_notes: Get architecture overview from .ai/README.md. No params."
}

/// Help text for the `web` tool group.
pub fn web_help() -> &'static str {
    "\
Available actions for `web`:

- search: Search the web. Params: query (required), max_results
  General web search uses Serper when configured.
  Recommended setup: put the raw key in `~/.miniswe/serper.key`.
  Without a Serper key, `web(search)` falls back to GitHub repository search only.
- fetch: Fetch a URL as markdown. Params: url (required), selector"
}

/// Help text for the `plan` tool.
pub fn plan_help() -> &'static str {
    "\
Available actions for `plan`:

- set: Create a plan. Params: steps (array) or content (markdown)
  Each step has: description, compile (bool, default true), reason (if compile=false)
- check: Mark step done. Params: step (number). Runs compile gate if compile=true.
- refine: Split a step into substeps. Params: step (number), substeps (array)
- show: View current plan. No params.
- scratchpad: Save working notes. Params: content (required, must have ## Current Task and ## Plan sections)"
}

#[cfg(test)]
mod flat_tests {
    use super::*;

    #[test]
    fn add_function_param_maps_to_add_param_end_by_default() {
        let a =
            json!({"path":"src/lib.rs","function":"assemble","param":"x: u32","call_value":"0"});
        let out = flat_to_refactor_args("add_function_param", &a).unwrap();
        assert_eq!(out["action"], "add_param");
        assert_eq!(out["name"], "assemble");
        assert_eq!(out["new_param"], "x: u32");
        assert_eq!(out["callsite_fill_in"], "0");
        assert_eq!(out["position"], "end");
    }

    #[test]
    fn add_function_param_after_maps_to_after_anchor() {
        let a = json!({"path":"s","function":"f","param":"p: P","call_value":"v","after":"b","line":42});
        let out = flat_to_refactor_args("add_function_param", &a).unwrap();
        assert_eq!(out["position"], "after:b");
        assert_eq!(out["line"], 42);
    }

    #[test]
    fn drop_and_rename_map_correctly() {
        let d = flat_to_refactor_args(
            "drop_function_param",
            &json!({"path":"s","function":"f","param":"p"}),
        )
        .unwrap();
        assert_eq!(d["action"], "drop_param");
        assert_eq!(d["param"], "p");
        let r = flat_to_refactor_args(
            "rename_symbol",
            &json!({"path":"s","line":7,"name":"old","new_name":"new"}),
        )
        .unwrap();
        assert_eq!(r["action"], "rename");
        assert_eq!(r["new_name"], "new");
        assert_eq!(r["line"], 7);
    }

    #[test]
    fn non_flat_name_returns_none() {
        assert!(flat_to_refactor_args("refactor", &json!({})).is_none());
        assert_eq!(flat_refactor_tool_definitions().len(), 3);
    }
}
