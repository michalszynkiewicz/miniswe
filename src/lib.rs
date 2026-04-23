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
pub mod runtime;
pub mod skills;
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

/// Write `contents` to `path` via a same-directory temp file + `rename`,
/// so a crash mid-write leaves either the old file or the new one —
/// never a truncated / half-written file.
pub fn atomic_write(path: &std::path::Path, contents: &[u8]) -> std::io::Result<()> {
    let parent = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    let file_name = path.file_name().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "path has no file name")
    })?;
    let tmp = parent.join(format!(
        ".{}.tmp.{}",
        file_name.to_string_lossy(),
        std::process::id()
    ));
    std::fs::write(&tmp, contents)?;
    std::fs::rename(&tmp, path)
}
