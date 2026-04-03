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
                description: "Read a file or line range. Auto-truncates to 4K tokens with line numbers.".into(),
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
                description: "Read the source code of a specific function, class, or type by name. Uses tree-sitter for precision.".into(),
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
                description: "Search the codebase. Returns matching lines with file:line context. USE THIS after changing a function name or signature to find all call sites that need updating.".into(),
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
                description: "Replace text in a file. RULES: (1) 'old' must match file content EXACTLY including whitespace (2) include 3+ unchanged lines for unique match (3) ONLY use for single targeted fix in files >200 lines (4) if edit fails, switch to write_file. AFTER editing a function signature: use search() to find all callers and update them.".into(),
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
                description: "Write complete file contents. PREFERRED for most changes. Use for: new files, ANY file under 200 lines, multiple changes to one file, or when edit fails. Always include the COMPLETE file content — do not omit sections with comments like '// rest unchanged'.".into(),
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
                description: "Execute a shell command and return stdout/stderr. Output tail-truncated to 3K tokens.".into(),
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
                description: "Rewrite the task scratchpad with current state. Must contain ## Current Task and ## Plan sections.".into(),
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
                name: "diagnostics".into(),
                description: "Get LSP/linter errors and warnings for a file or the whole project.".into(),
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
                description: "Search the web. Returns title+URL+snippet for top results.".into(),
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
                description: "Fetch a URL and extract main content as clean markdown. Truncated to 4K tokens.".into(),
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
                description: "Search local llms.txt documentation cache for a library's API info.".into(),
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
    ]
}

/// Return LSP tool definitions (only added when LSP is available).
pub fn lsp_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "goto_definition".into(),
                description: "Jump to the definition of a symbol at a given file location. Returns the definition's file, line, and surrounding source code.".into(),
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
                description: "Find all references to the symbol at a given file location. Returns a list of file:line locations with the referencing code.".into(),
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
                description: "Get the project's code structure overview — files ranked by importance with function/type signatures. Use this when you need to understand the codebase layout before making changes. Optionally filter by keywords to focus on relevant areas.".into(),
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
                name: "get_project_info".into(),
                description: "Get project metadata: language, build system, entry points, coding guidelines, and accumulated tips. Useful at the start of a task to understand the project.".into(),
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
                description: "Get architecture overview and key decisions from previous sessions (.ai/README.md). Useful when you need to understand design decisions or overall system structure.".into(),
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
            description: "Call a tool on an MCP server. Use the [MCP:name] entries in context to see available servers and tools.".into(),
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
