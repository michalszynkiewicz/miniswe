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
                description: "File operations: read, write, replace, search, shell, revert. Use action='help' for details.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "One of: read, write, replace, search, shell, revert, help"
                        },
                        "path": { "type": "string", "description": "File path (for read/write/replace/revert)" },
                        "content": { "type": "string", "description": "File content (for write)" },
                        "old": { "type": "string", "description": "Text to find (for replace)" },
                        "new": { "type": "string", "description": "Replacement text (for replace)" },
                        "all": { "type": "boolean", "description": "Replace all occurrences (for replace)" },
                        "start_line": { "type": "integer", "description": "Start line for read" },
                        "end_line": { "type": "integer", "description": "End line for read" },
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
                description: "Web access: search the web or fetch a URL. Use action='help' for details.".into(),
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
        // ── fix_file: LLM-powered code transformation ─────────────────
        ToolDefinition {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "fix_file".into(),
                description: "Describe a change and it gets applied across the file. Provide specific details (types, parameter names, values).".into(),
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
                        }
                    },
                    "required": ["path", "task"]
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
- write: Create or overwrite a file. Params: path (required), content (required)
- replace: Replace text. Params: path (required), old (required), new (required), all (optional bool)
  Default replaces one unique match. Set all=true for every occurrence.
- search: Search codebase. Params: query or pattern (one required), scope, max_results
- shell: Run a command. Params: command (required), timeout
- revert: Revert files to a previous round. Params: to_round, path (both optional)"
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
