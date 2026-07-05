//! JSON-RPC 2.0 framing for the MCP stdio transport: one message per line.
//!
//! Mirrors the wire format already spoken by `miniswe::mcp::client::McpClient`
//! (which connects to MCP servers as a client) — this module is the server
//! side of that same protocol.

use std::io::Write;

use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Deserialize)]
pub struct Request {
    #[serde(default)]
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

impl Request {
    /// Notifications (e.g. `notifications/initialized`) omit `id` or send it
    /// as `null` — both mean "no response expected".
    pub fn is_notification(&self) -> bool {
        matches!(self.id, None | Some(Value::Null))
    }
}

pub fn write_response(out: &mut impl Write, id: Value, result: Value) -> std::io::Result<()> {
    let msg = serde_json::json!({"jsonrpc": "2.0", "id": id, "result": result});
    writeln!(out, "{msg}")?;
    out.flush()
}

pub fn write_error(
    out: &mut impl Write,
    id: Value,
    code: i64,
    message: impl Into<String>,
) -> std::io::Result<()> {
    let msg = serde_json::json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {"code": code, "message": message.into()},
    });
    writeln!(out, "{msg}")?;
    out.flush()
}
