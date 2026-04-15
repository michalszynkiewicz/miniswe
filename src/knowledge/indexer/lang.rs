//! Per-language regex-based symbol extractors. Used as a bootstrap when
//! `ts_extract` (tree-sitter) doesn't support the language.

mod common;
mod go;
mod js_ts;
mod python;
mod rust;

use crate::knowledge::Symbol;

/// Extract symbols from source code using regex patterns.
/// This is a bootstrap implementation — tree-sitter replaces it where
/// available (see `crate::knowledge::ts_extract`).
pub fn extract_symbols(file: &str, content: &str, ext: &str) -> Vec<Symbol> {
    let mut symbols = Vec::new();

    match ext {
        "rs" => rust::extract(file, content, &mut symbols),
        "py" => python::extract(file, content, &mut symbols),
        "js" | "ts" | "tsx" | "jsx" => js_ts::extract(file, content, &mut symbols),
        "go" => go::extract(file, content, &mut symbols),
        _ => {} // Unsupported language for now
    }

    symbols
}
