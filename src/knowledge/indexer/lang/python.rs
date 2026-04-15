//! Regex-based symbol extraction for Python source files.

use crate::knowledge::Symbol;

use super::common::extract_name_after;

pub fn extract(file: &str, content: &str, symbols: &mut Vec<Symbol>) {
    for (line_num, line) in content.lines().enumerate() {
        let trimmed = line.trim();

        if trimmed.starts_with("def ") || trimmed.starts_with("async def ") {
            // `extract_name_after` searches for "def " in either form.
            let keyword = "def ";
            if let Some(name) = extract_name_after(trimmed, keyword) {
                symbols.push(Symbol {
                    name,
                    file: file.into(),
                    line: line_num + 1,
                    kind: "function".into(),
                    signature: trimmed.trim_end_matches(':').to_string(),
                    end_line: 0,
                    deps: Vec::new(),
                    parent_impl: None,
                });
            }
        } else if trimmed.starts_with("class ")
            && let Some(name) = extract_name_after(trimmed, "class ")
        {
            symbols.push(Symbol {
                name,
                file: file.into(),
                line: line_num + 1,
                kind: "class".into(),
                signature: trimmed.trim_end_matches(':').to_string(),
                end_line: 0,
                deps: Vec::new(),
                parent_impl: None,
            });
        }
    }
}
