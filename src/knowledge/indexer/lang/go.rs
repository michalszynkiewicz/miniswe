//! Regex-based symbol extraction for Go source files.

use crate::knowledge::Symbol;

use super::common::extract_name_after;

pub fn extract(file: &str, content: &str, symbols: &mut Vec<Symbol>) {
    for (line_num, line) in content.lines().enumerate() {
        let trimmed = line.trim();

        if trimmed.starts_with("func ") {
            if let Some(name) = extract_name_after(trimmed, "func ") {
                // Skip method receivers: func (r *Receiver) Method(...)
                let actual_name = if name.starts_with('(') {
                    // Method with receiver
                    if let Some(after_paren) = trimmed.split(") ").nth(1) {
                        extract_name_after(after_paren, "").unwrap_or(name)
                    } else {
                        name
                    }
                } else {
                    name
                };
                symbols.push(Symbol {
                    name: actual_name,
                    file: file.into(),
                    line: line_num + 1,
                    kind: "function".into(),
                    signature: trimmed.trim_end_matches('{').trim().to_string(),
                    end_line: 0,
                    deps: Vec::new(),
                    parent_impl: None,
                });
            }
        } else if trimmed.starts_with("type ")
            && let Some(name) = extract_name_after(trimmed, "type ")
        {
            let kind = if trimmed.contains("struct") {
                "struct"
            } else if trimmed.contains("interface") {
                "interface"
            } else {
                "type"
            };
            symbols.push(Symbol {
                name,
                file: file.into(),
                line: line_num + 1,
                kind: kind.into(),
                signature: trimmed.trim_end_matches('{').trim().to_string(),
                end_line: 0,
                deps: Vec::new(),
                parent_impl: None,
            });
        }
    }
}
