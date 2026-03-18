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

                    // Generate a one-line summary (prefers doc headers)
                    let summary = generate_summary(&content, &symbols, ext);
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

/// Generate a one-line summary of a file.
///
/// Prefers the module-level doc comment (//!, """, /**, etc.) if present —
/// these are high-signal, human-written descriptions of the file's purpose.
/// Falls back to a symbol-based summary.
fn generate_summary(content: &str, symbols: &[Symbol], ext: &str) -> String {
    // Try to extract a module-level doc header first
    if let Some(doc) = extract_doc_header(content, ext) {
        return doc;
    }

    // Fallback: symbol-based summary
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

/// Extract the module-level doc comment from a source file.
///
/// Supports:
/// - Rust: `//!` lines at the top
/// - Python: module-level `"""..."""` docstring
/// - JS/TS: `/** ... */` JSDoc at the top
/// - Go/Java/C/C++: `// ...` or `/* ... */` block at line 1
///
/// Returns the first meaningful line (stripped of comment markers),
/// truncated to ~100 chars.
fn extract_doc_header(content: &str, ext: &str) -> Option<String> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return None;
    }

    match ext {
        "rs" => {
            // Rust: //! doc comments
            let mut doc_lines = Vec::new();
            for line in &lines {
                let trimmed = line.trim();
                if trimmed.starts_with("//!") {
                    let text = trimmed.trim_start_matches("//!").trim();
                    if !text.is_empty() {
                        doc_lines.push(text);
                    }
                } else if trimmed.is_empty() && doc_lines.is_empty() {
                    continue; // skip leading blank lines
                } else {
                    break;
                }
            }
            if doc_lines.is_empty() {
                return None;
            }
            // Take first line as primary, append second if short
            let mut summary = doc_lines[0].to_string();
            if summary.len() < 50 && doc_lines.len() > 1 {
                summary.push_str(" — ");
                summary.push_str(doc_lines[1]);
            }
            Some(truncate_summary(&summary))
        }
        "py" => {
            // Python: module docstring (""" at start)
            for line in &lines {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if trimmed.starts_with("\"\"\"") || trimmed.starts_with("'''") {
                    let quote = &trimmed[..3];
                    // Single-line docstring: """text"""
                    if trimmed.len() > 6 && trimmed.ends_with(quote) {
                        let text = &trimmed[3..trimmed.len() - 3];
                        return Some(truncate_summary(text));
                    }
                    // Multi-line: take first content line
                    let text = trimmed[3..].trim();
                    if !text.is_empty() {
                        return Some(truncate_summary(text));
                    }
                    // Content is on next line
                    for next_line in lines.iter().skip(1) {
                        let t = next_line.trim();
                        if !t.is_empty() && !t.starts_with(quote) {
                            return Some(truncate_summary(t));
                        }
                        if t.starts_with(quote) {
                            break;
                        }
                    }
                    return None;
                }
                // If first non-empty line isn't a docstring, no module doc
                if !trimmed.starts_with('#') {
                    return None;
                }
            }
            None
        }
        "js" | "ts" | "tsx" | "jsx" | "java" | "go" | "c" | "cpp" | "h" | "hpp" => {
            // JSDoc / block comment at top: /** ... */ or /* ... */
            // Also: // line comments at the very top
            for line in &lines {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                // /** JSDoc style */
                if trimmed.starts_with("/**") || trimmed.starts_with("/*") {
                    let after = trimmed
                        .trim_start_matches("/**")
                        .trim_start_matches("/*")
                        .trim_start_matches('*')
                        .trim();
                    if !after.is_empty() && !after.starts_with("*/") {
                        return Some(truncate_summary(after));
                    }
                    // Multi-line: check next lines
                    for next_line in lines.iter().skip(1) {
                        let t = next_line.trim();
                        if t.starts_with("*/") {
                            break;
                        }
                        let text = t.trim_start_matches('*').trim();
                        if !text.is_empty() && !text.starts_with('@') {
                            return Some(truncate_summary(text));
                        }
                    }
                    return None;
                }
                // // line comment at top
                if trimmed.starts_with("//") {
                    let text = trimmed.trim_start_matches("//").trim();
                    if !text.is_empty() {
                        return Some(truncate_summary(text));
                    }
                }
                // package/import — no doc comment
                return None;
            }
            None
        }
        _ => None,
    }
}

fn truncate_summary(s: &str) -> String {
    if s.len() <= 100 {
        s.to_string()
    } else {
        format!("{}...", &s[..97])
    }
}

/// Check file sizes and return warnings for large files.
pub fn audit_file_sizes(root: &Path, max_lines: usize) -> Vec<(String, usize)> {
    let mut large_files = Vec::new();

    let walker = WalkBuilder::new(root)
        .hidden(true)
        .git_ignore(true)
        .git_global(true)
        .build();

    for entry in walker {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        if !SOURCE_EXTENSIONS.contains(&ext) {
            continue;
        }

        if let Ok(rel) = path.strip_prefix(root) {
            let rel_str = rel.to_string_lossy().to_string();
            if rel_str.starts_with(".minime") {
                continue;
            }

            if let Ok(content) = std::fs::read_to_string(path) {
                let line_count = content.lines().count();
                if line_count > max_lines {
                    large_files.push((rel_str, line_count));
                }
            }
        }
    }

    large_files.sort_by(|a, b| b.1.cmp(&a.1));
    large_files
}
