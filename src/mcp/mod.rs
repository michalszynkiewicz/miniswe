//! MCP (Model Context Protocol) client for minime.
//!
//! Connects to MCP servers configured in `.mcp.json`, fetches tool schemas,
//! and provides a lazy-loading bridge: only one-line summaries go into LLM
//! context, full schemas are resolved at execution time.

pub mod client;
pub mod config;
pub mod registry;

pub use config::McpConfig;
pub use registry::McpRegistry;
