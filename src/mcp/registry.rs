//! MCP tool registry — lazy-loading bridge between MCP servers and the agent.
//!
//! On startup, connects to configured MCP servers, fetches tool schemas,
//! and caches them. Provides one-line summaries for LLM context and
//! executes tool calls on demand.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

use super::client::{McpClient, McpToolDef};
use super::config::McpConfig;

/// Cached info about an MCP server and its tools.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerInfo {
    /// Server name (key from .mcp.json)
    pub name: String,
    /// One-line summary for LLM context
    pub summary: String,
    /// Tools available on this server
    pub tools: Vec<McpToolInfo>,
}

/// Cached info about a single MCP tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolInfo {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// Registry of all MCP servers and their tools.
/// Holds live connections to servers for tool execution.
pub struct McpRegistry {
    /// Cached server info (persisted to .minime/mcp/)
    pub servers: Vec<McpServerInfo>,
    /// Live client connections (not serialized)
    clients: HashMap<String, McpClient>,
}

impl McpRegistry {
    /// Initialize the registry: connect to all configured MCP servers,
    /// fetch their tool schemas, and cache them.
    pub fn connect(config: &McpConfig, cache_dir: &Path) -> Result<Self> {
        let mut servers = Vec::new();
        let mut clients = HashMap::new();

        std::fs::create_dir_all(cache_dir)?;

        for (name, server_config) in &config.servers {
            match McpClient::connect(name, server_config) {
                Ok(mut client) => {
                    match client.list_tools() {
                        Ok(tools) => {
                            let tool_infos: Vec<McpToolInfo> = tools
                                .iter()
                                .map(|t| McpToolInfo {
                                    name: t.name.clone(),
                                    description: t.description.clone(),
                                    input_schema: t.input_schema.clone(),
                                })
                                .collect();

                            let summary = build_summary(name, &tool_infos);

                            let info = McpServerInfo {
                                name: name.clone(),
                                summary,
                                tools: tool_infos,
                            };

                            // Cache to disk
                            let cache_path = cache_dir.join(format!("{name}.json"));
                            if let Ok(json) = serde_json::to_string_pretty(&info) {
                                let _ = std::fs::write(&cache_path, json);
                            }

                            servers.push(info);
                            clients.insert(name.clone(), client);

                            tracing::info!("MCP server '{name}': {} tools available", tools.len());
                        }
                        Err(e) => {
                            tracing::warn!("MCP server '{name}': failed to list tools: {e}");
                            // Still keep the client in case it recovers
                            clients.insert(name.clone(), client);
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("MCP server '{name}': failed to connect: {e}");

                    // Try loading from cache as fallback
                    let cache_path = cache_dir.join(format!("{name}.json"));
                    if let Ok(cached) = std::fs::read_to_string(&cache_path) {
                        if let Ok(info) = serde_json::from_str::<McpServerInfo>(&cached) {
                            tracing::info!(
                                "MCP server '{name}': using cached schema ({} tools)",
                                info.tools.len()
                            );
                            servers.push(info);
                        }
                    }
                }
            }
        }

        Ok(Self { servers, clients })
    }

    /// Load registry from cache only (no connections).
    /// Used when MCP servers aren't running but we still want summaries.
    pub fn load_cached(cache_dir: &Path) -> Self {
        let mut servers = Vec::new();

        if let Ok(entries) = std::fs::read_dir(cache_dir) {
            for entry in entries.flatten() {
                if entry.path().extension().and_then(|e| e.to_str()) == Some("json") {
                    if let Ok(content) = std::fs::read_to_string(entry.path()) {
                        if let Ok(info) = serde_json::from_str::<McpServerInfo>(&content) {
                            servers.push(info);
                        }
                    }
                }
            }
        }

        Self {
            servers,
            clients: HashMap::new(),
        }
    }

    /// Generate the context string for LLM injection.
    /// One line per server with tool count and names.
    pub fn context_summary(&self) -> Option<String> {
        if self.servers.is_empty() {
            return None;
        }

        let mut lines = Vec::new();
        for server in &self.servers {
            lines.push(server.summary.clone());
        }

        Some(lines.join("\n"))
    }

    /// Call a tool on an MCP server.
    pub fn call_tool(
        &mut self,
        server_name: &str,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<String> {
        let client = self
            .clients
            .get_mut(server_name)
            .ok_or_else(|| {
                anyhow::anyhow!("MCP server '{server_name}' not connected. Available: {}",
                    self.servers.iter().map(|s| s.name.as_str()).collect::<Vec<_>>().join(", "))
            })?;

        client.call_tool(tool_name, arguments)
    }

    /// Get the full schema for a specific tool (for validation/display).
    pub fn get_tool_schema(
        &self,
        server_name: &str,
        tool_name: &str,
    ) -> Option<&McpToolInfo> {
        self.servers
            .iter()
            .find(|s| s.name == server_name)?
            .tools
            .iter()
            .find(|t| t.name == tool_name)
    }

    /// Check if any servers are available.
    pub fn has_servers(&self) -> bool {
        !self.servers.is_empty()
    }

    /// Total number of MCP tools available.
    pub fn tool_count(&self) -> usize {
        self.servers.iter().map(|s| s.tools.len()).sum()
    }
}

/// Build a one-line summary for LLM context.
/// Example: "[MCP:github] 12 tools: create_issue, list_prs, review, ..."
fn build_summary(name: &str, tools: &[McpToolInfo]) -> String {
    let tool_names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    let preview = if tool_names.len() <= 5 {
        tool_names.join(", ")
    } else {
        format!(
            "{}, ... (+{} more)",
            tool_names[..4].join(", "),
            tool_names.len() - 4
        )
    };

    format!("[MCP:{}] {} tools: {}", name, tools.len(), preview)
}
