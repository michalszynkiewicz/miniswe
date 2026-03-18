//! MCP JSON-RPC client over stdio transport.
//!
//! Spawns an MCP server as a child process and communicates via
//! JSON-RPC 2.0 over stdin/stdout.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::config::McpServerConfig;

static REQUEST_ID: AtomicU64 = AtomicU64::new(1);

/// A connected MCP server process.
pub struct McpClient {
    child: Child,
    server_name: String,
}

/// An MCP tool definition as returned by tools/list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolDef {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(rename = "inputSchema", default)]
    pub input_schema: Value,
}

/// JSON-RPC request.
#[derive(Serialize)]
struct JsonRpcRequest {
    jsonrpc: &'static str,
    id: u64,
    method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<Value>,
}

/// JSON-RPC response.
#[derive(Deserialize)]
struct JsonRpcResponse {
    id: Option<u64>,
    result: Option<Value>,
    error: Option<JsonRpcError>,
}

#[derive(Deserialize)]
struct JsonRpcError {
    code: i64,
    message: String,
}

impl McpClient {
    /// Spawn an MCP server process and perform the initialize handshake.
    pub fn connect(name: &str, config: &McpServerConfig) -> Result<Self> {
        let mut cmd = Command::new(&config.command);
        cmd.args(&config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());

        // Set environment variables
        for (k, v) in &config.env {
            cmd.env(k, v);
        }

        let child = cmd
            .spawn()
            .with_context(|| format!("Failed to start MCP server '{name}': {}", config.command))?;

        let mut client = Self {
            child,
            server_name: name.to_string(),
        };

        // Send initialize request
        let init_result = client.request(
            "initialize",
            Some(serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "minime",
                    "version": env!("CARGO_PKG_VERSION")
                }
            })),
        )?;

        // Send initialized notification (no id, no response expected)
        client.notify("notifications/initialized", None)?;

        Ok(client)
    }

    /// List available tools from the server.
    pub fn list_tools(&mut self) -> Result<Vec<McpToolDef>> {
        let result = self.request("tools/list", None)?;

        let tools: Vec<McpToolDef> = result
            .get("tools")
            .and_then(|t| serde_json::from_value(t.clone()).ok())
            .unwrap_or_default();

        Ok(tools)
    }

    /// Call a tool on the server.
    pub fn call_tool(&mut self, tool_name: &str, arguments: Value) -> Result<String> {
        let result = self.request(
            "tools/call",
            Some(serde_json::json!({
                "name": tool_name,
                "arguments": arguments
            })),
        )?;

        // Extract text content from the response
        if let Some(content) = result.get("content").and_then(|c| c.as_array()) {
            let texts: Vec<&str> = content
                .iter()
                .filter_map(|item| {
                    if item.get("type").and_then(|t| t.as_str()) == Some("text") {
                        item.get("text").and_then(|t| t.as_str())
                    } else {
                        None
                    }
                })
                .collect();
            Ok(texts.join("\n"))
        } else {
            Ok(serde_json::to_string_pretty(&result)?)
        }
    }

    /// Send a JSON-RPC request and wait for the response.
    fn request(&mut self, method: &str, params: Option<Value>) -> Result<Value> {
        let id = REQUEST_ID.fetch_add(1, Ordering::Relaxed);

        let request = JsonRpcRequest {
            jsonrpc: "2.0",
            id,
            method: method.to_string(),
            params,
        };

        let mut request_json = serde_json::to_string(&request)?;
        request_json.push('\n');

        // Write to stdin
        let stdin = self
            .child
            .stdin
            .as_mut()
            .context("MCP server stdin not available")?;
        stdin.write_all(request_json.as_bytes())?;
        stdin.flush()?;

        // Read response from stdout
        let stdout = self
            .child
            .stdout
            .as_mut()
            .context("MCP server stdout not available")?;
        let mut reader = BufReader::new(stdout);

        // Read lines until we get a JSON-RPC response with our id
        // (skip notifications which have no id)
        loop {
            let mut line = String::new();
            let bytes_read = reader.read_line(&mut line)?;
            if bytes_read == 0 {
                bail!(
                    "MCP server '{}' closed connection unexpectedly",
                    self.server_name
                );
            }

            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            // Try to parse as JSON-RPC response
            if let Ok(response) = serde_json::from_str::<JsonRpcResponse>(line) {
                // Skip notifications (no id)
                if response.id.is_none() {
                    continue;
                }

                if let Some(error) = response.error {
                    bail!(
                        "MCP server '{}' error ({}): {}",
                        self.server_name,
                        error.code,
                        error.message
                    );
                }

                return Ok(response.result.unwrap_or(Value::Null));
            }
            // If it's not valid JSON-RPC, skip it (could be a log line)
        }
    }

    /// Send a JSON-RPC notification (no response expected).
    fn notify(&mut self, method: &str, params: Option<Value>) -> Result<()> {
        let notification = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params.unwrap_or(Value::Object(Default::default()))
        });

        let mut json = serde_json::to_string(&notification)?;
        json.push('\n');

        let stdin = self
            .child
            .stdin
            .as_mut()
            .context("MCP server stdin not available")?;
        stdin.write_all(json.as_bytes())?;
        stdin.flush()?;

        Ok(())
    }

    /// Shut down the server gracefully.
    pub fn shutdown(&mut self) {
        // Try to kill the child process
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for McpClient {
    fn drop(&mut self) {
        self.shutdown();
    }
}
