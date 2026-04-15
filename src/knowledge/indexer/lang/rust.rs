//! Regex-based symbol extraction for Rust source files.

use crate::knowledge::Symbol;

use super::common::extract_name_after;

pub fn extract(file: &str, content: &str, symbols: &mut Vec<Symbol>) {
    for (line_num, line) in content.lines().enumerate() {
        let trimmed = line.trim();

        // Functions
        if trimmed.starts_with("pub fn ")
            || trimmed.starts_with("fn ")
            || trimmed.starts_with("pub async fn ")
            || trimmed.starts_with("async fn ")
            || trimmed.starts_with("pub(crate) fn ")
            || trimmed.starts_with("pub(super) fn ")
        {
            if let Some(name) = extract_name_after(trimmed, "fn ") {
                symbols.push(Symbol {
                    name,
                    file: file.into(),
                    line: line_num + 1,
                    kind: "function".into(),
                    signature: trimmed.trim_end_matches('{').trim().to_string(),
                    end_line: 0,
                    deps: Vec::new(),
                    parent_impl: None,
                });
            }
        }
        // Structs
        else if trimmed.starts_with("pub struct ")
            || trimmed.starts_with("struct ")
            || trimmed.starts_with("pub(crate) struct ")
        {
            if let Some(name) = extract_name_after(trimmed, "struct ") {
                symbols.push(Symbol {
                    name,
                    file: file.into(),
                    line: line_num + 1,
                    kind: "struct".into(),
                    signature: trimmed.trim_end_matches('{').trim().to_string(),
                    end_line: 0,
                    deps: Vec::new(),
                    parent_impl: None,
                });
            }
        }
        // Enums
        else if trimmed.starts_with("pub enum ") || trimmed.starts_with("enum ") {
            if let Some(name) = extract_name_after(trimmed, "enum ") {
                symbols.push(Symbol {
                    name,
                    file: file.into(),
                    line: line_num + 1,
                    kind: "enum".into(),
                    signature: trimmed.trim_end_matches('{').trim().to_string(),
                    end_line: 0,
                    deps: Vec::new(),
                    parent_impl: None,
                });
            }
        }
        // Traits
        else if trimmed.starts_with("pub trait ") || trimmed.starts_with("trait ") {
            if let Some(name) = extract_name_after(trimmed, "trait ") {
                symbols.push(Symbol {
                    name,
                    file: file.into(),
                    line: line_num + 1,
                    kind: "trait".into(),
                    signature: trimmed.trim_end_matches('{').trim().to_string(),
                    end_line: 0,
                    deps: Vec::new(),
                    parent_impl: None,
                });
            }
        }
        // Impl blocks
        else if trimmed.starts_with("impl ")
            && let Some(name) = extract_name_after(trimmed, "impl ")
        {
            symbols.push(Symbol {
                name: format!("impl {name}"),
                file: file.into(),
                line: line_num + 1,
                kind: "impl".into(),
                signature: trimmed.trim_end_matches('{').trim().to_string(),
                end_line: 0,
                deps: Vec::new(),
                parent_impl: None,
            });
        }
    }
}
