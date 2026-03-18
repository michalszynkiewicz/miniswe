//! File indexer — scans the project and extracts symbols.
//!
//! Phase 1: Uses regex-based symbol extraction as a bootstrap.
//! Phase 2 (future): Full tree-sitter parsing for precise AST-based extraction.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use ignore::WalkBuilder;

use super::{ProjectIndex, Symbol};

/// Known source file extensions.
const SOURCE_EXTENSIONS: &[&str] = &[
    "rs", "py", "js", "ts", "tsx", "jsx", "go", "java", "c", "cpp", "h", "hpp", "rb", "php",
    "swift", "kt", "scala", "zig", "hs", "ml", "ex", "exs", "clj",
];

/// Index a project directory.
pub fn index_project(root: &Path) -> Result<ProjectIndex> {
    let mut index = ProjectIndex::default();
    let mut file_count = 0;

    // Walk the directory tree, respecting .gitignore
    let walker = WalkBuilder::new(root)
        .hidden(true) // skip hidden files
        .git_ignore(true) // respect .gitignore
        .git_global(true)
        .build();

    for entry in walker {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        let path = entry.path();

        // Skip directories and non-source files
        if !path.is_file() {
            continue;
        }

        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        // Record all files in the tree
        if let Ok(rel) = path.strip_prefix(root) {
            let rel_str = rel.to_string_lossy().to_string();

            // Skip .minime directory itself
            if rel_str.starts_with(".minime") {
                continue;
            }

            index.file_tree.push(rel_str.clone());

            // Only extract symbols from source files
            if SOURCE_EXTENSIONS.contains(&ext) {
                file_count += 1;
                if let Ok(content) = std::fs::read_to_string(path) {
                    let symbols = extract_symbols(&rel_str, &content, ext);
                    for sym in &symbols {
                        index
                            .symbols
                            .entry(sym.name.clone())
                            .or_default()
                            .push(sym.clone());
                    }
                    index.total_symbols += symbols.len();

                    // Generate a one-line summary
                    let summary = generate_summary(&content, &symbols);
                    index.summaries.insert(rel_str, summary);
                }
            }
        }
    }

    index.total_files = file_count;
    index.file_tree.sort();

    Ok(index)
}

/// Extract symbols from source code using regex patterns.
/// This is a bootstrap implementation — tree-sitter will replace this.
fn extract_symbols(file: &str, content: &str, ext: &str) -> Vec<Symbol> {
    let mut symbols = Vec::new();

    match ext {
        "rs" => extract_rust_symbols(file, content, &mut symbols),
        "py" => extract_python_symbols(file, content, &mut symbols),
        "js" | "ts" | "tsx" | "jsx" => extract_js_ts_symbols(file, content, &mut symbols),
        "go" => extract_go_symbols(file, content, &mut symbols),
        _ => {} // Unsupported language for now
    }

    symbols
}

fn extract_rust_symbols(file: &str, content: &str, symbols: &mut Vec<Symbol>) {
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
                    deps: Vec::new(),
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
                    deps: Vec::new(),
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
                    deps: Vec::new(),
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
                    deps: Vec::new(),
                });
            }
        }
        // Impl blocks
        else if trimmed.starts_with("impl ") {
            if let Some(name) = extract_name_after(trimmed, "impl ") {
                symbols.push(Symbol {
                    name: format!("impl {name}"),
                    file: file.into(),
                    line: line_num + 1,
                    kind: "impl".into(),
                    signature: trimmed.trim_end_matches('{').trim().to_string(),
                    deps: Vec::new(),
                });
            }
        }
    }
}

fn extract_python_symbols(file: &str, content: &str, symbols: &mut Vec<Symbol>) {
    for (line_num, line) in content.lines().enumerate() {
        let trimmed = line.trim();

        if trimmed.starts_with("def ") || trimmed.starts_with("async def ") {
            let keyword = if trimmed.starts_with("async") {
                "def "
            } else {
                "def "
            };
            if let Some(name) = extract_name_after(trimmed, keyword) {
                symbols.push(Symbol {
                    name,
                    file: file.into(),
                    line: line_num + 1,
                    kind: "function".into(),
                    signature: trimmed.trim_end_matches(':').to_string(),
                    deps: Vec::new(),
                });
            }
        } else if trimmed.starts_with("class ") {
            if let Some(name) = extract_name_after(trimmed, "class ") {
                symbols.push(Symbol {
                    name,
                    file: file.into(),
                    line: line_num + 1,
                    kind: "class".into(),
                    signature: trimmed.trim_end_matches(':').to_string(),
                    deps: Vec::new(),
                });
            }
        }
    }
}

fn extract_js_ts_symbols(file: &str, content: &str, symbols: &mut Vec<Symbol>) {
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
                    deps: Vec::new(),
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
                    deps: Vec::new(),
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
                    deps: Vec::new(),
                });
            }
        }
        // type alias
        else if trimmed.starts_with("export type ") || trimmed.starts_with("type ") {
            if let Some(name) = extract_name_after(trimmed, "type ") {
                symbols.push(Symbol {
                    name,
                    file: file.into(),
                    line: line_num + 1,
                    kind: "type".into(),
                    signature: trimmed.to_string(),
                    deps: Vec::new(),
                });
            }
        }
    }
}

fn extract_go_symbols(file: &str, content: &str, symbols: &mut Vec<Symbol>) {
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
                    deps: Vec::new(),
                });
            }
        } else if trimmed.starts_with("type ") {
            if let Some(name) = extract_name_after(trimmed, "type ") {
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
                    deps: Vec::new(),
                });
            }
        }
    }
}

/// Extract an identifier name after a keyword.
fn extract_name_after(line: &str, keyword: &str) -> Option<String> {
    let after = if keyword.is_empty() {
        line
    } else {
        line.split(keyword).nth(1)?
    };
    let name: String = after
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

/// Generate a one-line summary of a file based on its symbols.
fn generate_summary(content: &str, symbols: &[Symbol]) -> String {
    if symbols.is_empty() {
        let line_count = content.lines().count();
        return format!("{line_count} lines, no exported symbols");
    }

    let kinds: Vec<&str> = symbols.iter().map(|s| s.kind.as_str()).collect();
    let names: Vec<&str> = symbols
        .iter()
        .take(5)
        .map(|s| s.name.as_str())
        .collect();

    let kind_summary = {
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for k in &kinds {
            *counts.entry(k).or_default() += 1;
        }
        let mut parts: Vec<String> = counts
            .iter()
            .map(|(k, v)| format!("{v} {k}{}", if *v > 1 { "s" } else { "" }))
            .collect();
        parts.sort();
        parts.join(", ")
    };

    format!(
        "{kind_summary}: {}{}",
        names.join(", "),
        if symbols.len() > 5 {
            format!(", ... ({} total)", symbols.len())
        } else {
            String::new()
        }
    )
}
