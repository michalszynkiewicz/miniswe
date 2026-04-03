//! miniswe — A lightweight CLI coding agent for local LLMs
//!
//! Library crate exposing core functionality for integration tests.

#![allow(dead_code)]

pub mod cli;
pub mod config;
pub mod context;
pub mod knowledge;
pub mod llm;
pub mod logging;
pub mod lsp;
pub mod mcp;
pub mod tools;
pub mod tui;

/// Truncate a string to at most `max_chars` characters (not bytes).
/// Appends "..." if truncated.
pub fn truncate_chars(s: &str, max_chars: usize) -> String {
    let char_count = s.chars().count();
    if char_count <= max_chars {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_chars).collect();
        format!("{truncated}...")
    }
}
