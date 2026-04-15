//! Regex-based symbol extraction for JavaScript / TypeScript source files.

use crate::knowledge::Symbol;

use super::common::extract_name_after;

pub fn extract(file: &str, content: &str, symbols: &mut Vec<Symbol>) {
    for (line_num, line) in content.lines().enumerate() {
        let trimmed = line.trim();

        // export function / export async function / function
        if trimmed.contains("function ") {
            let keyword = "function ";
            if let Some(name) = extract_name_after(trimmed, keyword) {
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
        // export class / class
        else if trimmed.contains("class ") && !trimmed.starts_with("//") {
            if let Some(name) = extract_name_after(trimmed, "class ") {
                symbols.push(Symbol {
                    name,
                    file: file.into(),
                    line: line_num + 1,
                    kind: "class".into(),
                    signature: trimmed.trim_end_matches('{').trim().to_string(),
                    end_line: 0,
                    deps: Vec::new(),
                    parent_impl: None,
                });
            }
        }
        // interface
        else if trimmed.contains("interface ") && !trimmed.starts_with("//") {
            if let Some(name) = extract_name_after(trimmed, "interface ") {
                symbols.push(Symbol {
                    name,
                    file: file.into(),
                    line: line_num + 1,
                    kind: "interface".into(),
                    signature: trimmed.trim_end_matches('{').trim().to_string(),
                    end_line: 0,
                    deps: Vec::new(),
                    parent_impl: None,
                });
            }
        }
        // type alias
        else if (trimmed.starts_with("export type ") || trimmed.starts_with("type "))
            && let Some(name) = extract_name_after(trimmed, "type ")
        {
            symbols.push(Symbol {
                name,
                file: file.into(),
                line: line_num + 1,
                kind: "type".into(),
                signature: trimmed.to_string(),
                end_line: 0,
                deps: Vec::new(),
                parent_impl: None,
            });
        }
    }
}
