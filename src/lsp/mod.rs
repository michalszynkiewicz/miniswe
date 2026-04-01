//! LSP integration for fast diagnostics and code navigation.
//!
//! Spawns rust-analyzer (or other LSP servers) at session start.
//! Provides two integration modes:
//! - **Automatic**: fast diagnostics after file writes (replaces cargo check)
//! - **Tools**: goto_definition and find_references for model navigation

pub mod client;
pub mod servers;
pub mod transport;

pub use client::LspClient;
pub use servers::LspServer;
