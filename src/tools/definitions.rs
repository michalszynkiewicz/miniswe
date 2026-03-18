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
                description: "Search the codebase using ripgrep. Returns matching lines with file:line context.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search pattern (regex supported)"
                        },
                        "scope": {
                            "type": "string",
                            "description": "Search scope: 'project' (default), a directory path, or 'symbols'"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum results to return (default: 20)"
                        }
                    },
                    "required": ["query"]
                }),
            },
        },
        ToolDefinition {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "edit".into(),
                description: "Replace old text with new text in a file. Best for surgical changes to large files. For small files (<200 lines), prefer write_file.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path relative to project root"
                        },
                        "old": {
                            "type": "string",
                            "description": "Exact text to find and replace"
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
                description: "Write complete file contents. More reliable than edit for small files (<200 lines). Preferred for creating new files or rewriting existing ones.".into(),
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
                description: "Search the web using DuckDuckGo. Returns title+URL+snippet for top results.".into(),
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
