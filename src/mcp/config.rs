//! `.mcp.json` configuration parsing.
//!
//! Compatible with Claude Code's `.mcp.json` format:
//! ```json
//! {
//!   "mcpServers": {
//!     "server-name": {
//!       "command": "npx",
//!       "args": ["-y", "some-mcp-server"],
//!       "env": { "API_KEY": "..." },
//!       "timeout": 60000
//!     }
//!   }
//! }
//! ```

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Top-level `.mcp.json` structure.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct McpConfig {
    #[serde(rename = "mcpServers", default)]
    pub servers: HashMap<String, McpServerConfig>,
}

/// Configuration for a single MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    /// Command to launch the server (stdio transport)
    pub command: String,
    /// Arguments to pass to the command
    #[serde(default)]
    pub args: Vec<String>,
    /// Environment variables for the server process
    #[serde(default)]
    pub env: HashMap<String, String>,
    /// Timeout in milliseconds (default: 60000)
    #[serde(default = "default_timeout")]
    pub timeout: u64,
}

fn default_timeout() -> u64 {
    60000
}

impl McpConfig {
    /// Load `.mcp.json` from the project root.
    /// Returns an empty config if the file doesn't exist.
    pub fn load(project_root: &Path) -> Result<Self> {
        let path = project_root.join(".mcp.json");
        if !path.exists() {
            return Ok(Self::default());
        }

        let contents = std::fs::read_to_string(&path)
            .with_context(|| format!("Failed to read {}", path.display()))?;
        let config: McpConfig = serde_json::from_str(&contents)
            .with_context(|| format!("Failed to parse {}", path.display()))?;

        Ok(config)
    }

    /// Check if any MCP servers are configured.
    pub fn has_servers(&self) -> bool {
        !self.servers.is_empty()
    }
}
