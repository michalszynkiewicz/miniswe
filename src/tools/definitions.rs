//! Tool definitions in OpenAI function-calling format.

use crate::llm::{FunctionDefinition, ToolDefinition};
use serde_json::json;

/// Return all tool definitions for the LLM.
pub fn tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "read_file".into(),
                description: "Read a file or line range with line numbers.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path relative to project root"
                        },
                        "start_line": {
                            "type": "integer",
                            "description": "Start line number (1-indexed, optional)"
                        },
                        "end_line": {
                            "type": "integer",
                            "description": "End line number (inclusive, optional)"
                        }
                    },
                    "required": ["path"]
                }),
            },
        },
        ToolDefinition {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "read_symbol".into(),
                description: "Read a function, class, or type definition by name.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Symbol name to look up"
                        },
                        "follow_deps": {
                            "type": "boolean",
                            "description": "Also include type definitions this symbol depends on"
                        }
                    },
                    "required": ["name"]
                }),
            },
        },
        ToolDefinition {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "search".into(),
                description: "Search the codebase for text or patterns. Returns matching lines with file:line context.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Plain text to search for (e.g. 'fn assemble' or 'max_rounds'). No escaping needed."
                        },
                        "pattern": {
                            "type": "string",
                            "description": "Regex pattern for advanced searches (e.g. 'fn\\s+run\\b'). Use query for simple text searches."
                        },
                        "scope": {
                            "type": "string",
                            "description": "Search scope: 'project' (default), a directory path, or 'symbols'"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum results to return (default: 20)"
                        }
                    }
                }),
            },
        },
        ToolDefinition {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "edit".into(),
                description: "Replace one occurrence of 'old' with 'new' in a file. The 'old' text must match exactly and uniquely.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path relative to project root"
                        },
                        "old": {
                            "type": "string",
                            "description": "Exact text to find (include 3+ lines of context for unique match)"
                        },
                        "new": {
                            "type": "string",
                            "description": "Replacement text"
                        }
                    },
                    "required": ["path", "old", "new"]
                }),
            },
        },
        ToolDefinition {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "write_file".into(),
                description: "Create or overwrite a file with complete content.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path relative to project root"
                        },
                        "content": {
                            "type": "string",
                            "description": "Complete file content to write"
                        }
                    },
                    "required": ["path", "content"]
                }),
            },
        },
        ToolDefinition {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "shell".into(),
                description: "Execute a shell command and return stdout/stderr.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Shell command to execute"
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds (default: 60)"
                        }
                    },
                    "required": ["command"]
                }),
            },
        },
        ToolDefinition {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "task_update".into(),
                description: "Save notes and progress to the scratchpad. Persists across rounds.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Full scratchpad content in markdown format"
                        }
                    },
                    "required": ["content"]
                }),
            },
        },
        ToolDefinition {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "plan".into(),
                description: "Manage a structured plan. Actions: 'set' (create plan), 'check' (mark step done), 'show' (view plan).".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "One of: 'set' (create/replace plan), 'check' (mark a step complete), 'show' (view current plan)"
                        },
                        "content": {
                            "type": "string",
                            "description": "For action='set': the plan in markdown with '- [ ] step' checkboxes"
                        },
                        "step": {
                            "type": "integer",
                            "description": "For action='check': which step number to mark complete (1-indexed)"
                        }
                    },
                    "required": ["action"]
                }),
            },
        },
        ToolDefinition {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "diagnostics".into(),
                description: "Get compiler errors and warnings for the project.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path (optional, defaults to project-wide)"
                        }
                    }
                }),
            },
        },
        ToolDefinition {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "web_search".into(),
                description: "Search the web for documentation or solutions.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum results (default: 5)"
                        }
                    },
                    "required": ["query"]
                }),
            },
        },
        ToolDefinition {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "web_fetch".into(),
                description: "Fetch a URL and extract content as markdown.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL to fetch"
                        },
                        "selector": {
                            "type": "string",
                            "description": "CSS selector to narrow extraction (optional)"
                        }
                    },
                    "required": ["url"]
                }),
            },
        },
        ToolDefinition {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "docs_lookup".into(),
                description: "Search cached local documentation for a library.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "library": {
                            "type": "string",
                            "description": "Library name (e.g., 'prisma', 'next')"
                        },
                        "topic": {
                            "type": "string",
                            "description": "Specific topic to search for (optional)"
                        }
                    },
                    "required": ["library"]
                }),
            },
        },
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
                            "description": "What to change and why. Be specific: include types, parameter names, values. E.g. 'Add system_prompt_override: Option<&str> as the 6th parameter to every call to context::assemble(), passing None for now.'"
                        }
                    },
                    "required": ["path", "task"]
                }),
            },
        },
        ToolDefinition {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "replace_all".into(),
                description: "Replace ALL occurrences of 'old' with 'new' in a file.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path relative to project root"
                        },
                        "old": {
                            "type": "string",
                            "description": "Exact text to find (every occurrence will be replaced)"
                        },
                        "new": {
                            "type": "string",
                            "description": "Replacement text"
                        }
                    },
                    "required": ["path", "old", "new"]
                }),
            },
        },
        ToolDefinition {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "revert".into(),
                description: "Revert files to a previous round's state. Each round is automatically snapshotted.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "to_round": {
                            "type": "integer",
                            "description": "Round number to revert to (0 = session start). Omit to revert everything to session start."
                        },
                        "path": {
                            "type": "string",
                            "description": "Revert only this file (optional — omit to revert all files)"
                        }
                    }
                }),
            },
        },
    ]
}

/// Return LSP tool definitions (only added when LSP is available).
pub fn lsp_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "goto_definition".into(),
                description: "Jump to a symbol's definition. Returns file, line, and source context.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path relative to project root"
                        },
                        "line": {
                            "type": "integer",
                            "description": "Line number (1-indexed)"
                        },
                        "column": {
                            "type": "integer",
                            "description": "Column number (1-indexed)"
                        }
                    },
                    "required": ["path", "line", "column"]
                }),
            },
        },
        ToolDefinition {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "find_references".into(),
                description: "Find all references to a symbol. Returns file:line locations.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path relative to project root"
                        },
                        "line": {
                            "type": "integer",
                            "description": "Line number (1-indexed)"
                        },
                        "column": {
                            "type": "integer",
                            "description": "Column number (1-indexed)"
                        }
                    },
                    "required": ["path", "line", "column"]
                }),
            },
        },
    ]
}

/// Context tools — pull-based access to project knowledge.
pub fn context_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "get_repo_map".into(),
                description: "Get the project's code structure: files ranked by importance with function/type signatures. Use describe_code for details on specific files.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "keywords": {
                            "type": "string",
                            "description": "Space-separated keywords to focus the map on (e.g. 'config cli run'). Optional — omit for full overview."
                        }
                    }
                }),
            },
        },
        ToolDefinition {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "describe_code".into(),
                description: "Get detailed descriptions for a file's functions and types: doc comments, parameter details, what each symbol does. Use after get_repo_map to understand specific files before reading them.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path relative to project root"
                        },
                        "symbols": {
                            "type": "string",
                            "description": "Comma-separated symbol names to describe (optional — omit for all symbols in the file)"
                        }
                    },
                    "required": ["path"]
                }),
            },
        },
        ToolDefinition {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "get_project_info".into(),
                description: "Get project metadata: language, build system, entry points, guidelines.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {}
                }),
            },
        },
        ToolDefinition {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "get_architecture_notes".into(),
                description: "Get architecture overview and key decisions from .ai/README.md.".into(),
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
