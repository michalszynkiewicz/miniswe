//! Tool definitions in OpenAI function-calling format.
//!
//! Tools are grouped into 6 top-level tools to reduce the function list
//! size for small models. Each grouped tool uses an `action` parameter
//! to dispatch to sub-tools. Use `action="help"` to list available actions.

use crate::llm::{FunctionDefinition, ToolDefinition};
use serde_json::json;

/// Return all tool definitions for the LLM.
pub fn tool_definitions() -> Vec<ToolDefinition> {
    vec![
        // ── file: core file I/O and shell ─────────────────────────────
        ToolDefinition {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "file".into(),
                description: "File operations: read, search, shell, delete, revert. Use action='help' for details. For code edits use edit_file; for full-file overwrites use write_file.".into(),
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
                description: "Plan and track work. Actions: set (create plan), check (mark step done with compile gate), refine (split step), show (view plan), scratchpad (save notes). Use action='help' for details.".into(),
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
                            "description": "For set: structured step list",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "description": { "type": "string" },
                                    "compile": { "type": "boolean" },
                                    "reason": { "type": "string" }
                                },
                                "required": ["description"]
                            }
                        },
                        "step": { "type": "integer", "description": "Step number (for check/refine)" },
                        "substeps": {
                            "type": "array",
                            "description": "For refine: substeps to replace target step",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "description": { "type": "string" },
                                    "compile": { "type": "boolean" },
                                    "reason": { "type": "string" }
                                },
                                "required": ["description"]
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
                description: "Create or overwrite a file. Provide complete file contents in `content`, or omit `content` to create a new empty file. Do not use for partial edits to existing code; use edit_file instead.".into(),
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
/// `revert`, `check`). Fed to the router when `tools.edit_mode = "fast"`,
/// replacing `edit_file`. See `docs/fast-mode-design.md`.
pub fn fast_mode_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "replace_range".into(),
                description: "Replace lines [start..=end] (1-based, inclusive) with `content`. Empty content deletes the range. After each call you receive per-edit AST + LSP feedback and the file's revision table; if you see a regression, call `revert` with the prior rev number.".into(),
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
                description: "Insert `content` after line `after_line` (1-based). Use after_line=0 to insert at the top of the file, after_line=<last line> to append. Use `replace_range` when you need to replace or delete existing lines.".into(),
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
                description: "Restore `path` to a named prior revision. Pick the rev number from the revision table attached to every edit's feedback. Linear history: reverting to rev_N truncates rev_{N+1}.. and the next edit becomes rev_{N+1}.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "File path relative to project root" },
                        "rev": { "type": "integer", "description": "Revision number to restore (0 = original / pristine)" }
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

/// Help text for the `file` tool group.
pub fn file_help() -> &'static str {
    "\
Available actions for `file`:

- read: Read a file. Params: path (required), start_line, end_line
- delete: Delete an existing file. Params: path (required)
  Example: {\"action\":\"delete\",\"path\":\"src/bin/old.rs\"}
- search: Search codebase. Params: query or pattern (one required), scope, max_results
- shell: Run a command. Params: command (required), timeout
  Example: {\"action\":\"shell\",\"command\":\"cargo check\",\"timeout\":30}
- revert: Revert files to a previous round. Params: to_round, path (both optional)

For text edits use edit_file (semantic, planner-driven). For full-file overwrites \
use write_file. There is no longer a deterministic search-and-replace action."
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
