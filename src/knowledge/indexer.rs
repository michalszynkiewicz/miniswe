//! File indexer — scans the project and extracts symbols.
//!
//! Phase 1: Uses regex-based symbol extraction as a bootstrap.
//! Phase 2 (future): Full tree-sitter parsing for precise AST-based extraction.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use ignore::WalkBuilder;

use super::ts_extract;
use super::{ProjectIndex, Symbol};

/// Known source file extensions.
const SOURCE_EXTENSIONS: &[&str] = &[
    "rs", "py", "js", "ts", "tsx", "jsx", "go", "java", "c", "cpp", "h", "hpp", "rb", "php",
    "swift", "kt", "scala", "zig", "hs", "ml", "ex", "exs", "clj",
];

/// Get file mtime as seconds since epoch.
fn file_mtime(path: &Path) -> u64 {
    path.metadata()
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Index a project directory. If `previous` is provided, only re-indexes
/// files whose mtime has changed (incremental mode).
pub fn index_project(root: &Path, previous: Option<&ProjectIndex>) -> Result<ProjectIndex> {
    let mut index = ProjectIndex::default();
    let mut file_count = 0;
    let mut reused = 0;

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

        if let Ok(rel) = path.strip_prefix(root) {
            let rel_str = rel.to_string_lossy().to_string();

            if rel_str.starts_with(".miniswe") {
                continue;
            }

            index.file_tree.push(rel_str.clone());

            if SOURCE_EXTENSIONS.contains(&ext) {
                file_count += 1;
                let mtime = file_mtime(path);

                // Incremental: reuse previous index if file hasn't changed
                if let Some(prev) = previous {
                    if !prev.is_stale(&rel_str, mtime) {
                        // Copy symbols from previous index
                        for (name, syms) in &prev.symbols {
                            for sym in syms {
                                if sym.file == rel_str {
                                    index
                                        .symbols
                                        .entry(name.clone())
                                        .or_default()
                                        .push(sym.clone());
                                    index.total_symbols += 1;
                                }
                            }
                        }
                        if let Some(summary) = prev.summaries.get(&rel_str) {
                            index.summaries.insert(rel_str.clone(), summary.clone());
                        }
                        index.mtimes.insert(rel_str, mtime);
                        reused += 1;
                        continue;
                    }
                }

                // (Re-)index this file
                if let Ok(content) = std::fs::read_to_string(path) {
                    let mut symbols = if let Some(ts_result) =
                        ts_extract::extract(&rel_str, &content, ext)
                    {
                        for sym_ref in &ts_result.references {
                            index
                                .references
                                .entry(rel_str.clone())
                                .or_default()
                                .push(sym_ref.name.clone());
                        }
                        ts_result.symbols
                    } else {
                        extract_symbols(&rel_str, &content, ext)
                    };

                    // Compute end_line for each symbol
                    compute_end_lines(&mut symbols, &content);

                    for sym in &symbols {
                        index
                            .symbols
                            .entry(sym.name.clone())
                            .or_default()
                            .push(sym.clone());
                    }
                    index.total_symbols += symbols.len();

                    let summary = generate_summary(&content, &symbols, ext);
                    index.summaries.insert(rel_str.clone(), summary);
                    index.mtimes.insert(rel_str, mtime);
                }
            }
        }
    }

    index.total_files = file_count;
    index.file_tree.sort();

    if reused > 0 {
        tracing::info!("Incremental index: {reused} files reused, {} re-indexed", file_count - reused);
    }

    Ok(index)
}

/// Re-index a single file in an existing index.
///
/// Removes old symbols for that file, re-extracts, recomputes end_lines,
/// updates mtime, and saves the index to disk. Takes <1ms per file.
pub fn reindex_file(
    rel_path: &str,
    abs_path: &Path,
    index: &mut ProjectIndex,
    miniswe_dir: &Path,
) {
    let ext = abs_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    if !SOURCE_EXTENSIONS.contains(&ext) {
        return;
    }

    let content = match std::fs::read_to_string(abs_path) {
        Ok(c) => c,
        Err(_) => return,
    };

    // Remove old symbols for this file
    for syms in index.symbols.values_mut() {
        syms.retain(|s| s.file != rel_path);
    }
    // Remove empty entries
    index.symbols.retain(|_, v| !v.is_empty());

    // Re-extract symbols
    let mut symbols = if let Some(ts_result) = ts_extract::extract(rel_path, &content, ext) {
        for sym_ref in &ts_result.references {
            index
                .references
                .entry(rel_path.to_string())
                .or_default()
                .push(sym_ref.name.clone());
        }
        ts_result.symbols
    } else {
        extract_symbols(rel_path, &content, ext)
    };

    compute_end_lines(&mut symbols, &content);

    // Insert new symbols
    for sym in &symbols {
        index
            .symbols
            .entry(sym.name.clone())
            .or_default()
            .push(sym.clone());
    }

    // Update summary and mtime
    let summary = generate_summary(&content, &symbols, ext);
    index.summaries.insert(rel_path.to_string(), summary);
    index.mtimes.insert(rel_path.to_string(), file_mtime(abs_path));

    // Recount
    index.total_symbols = index.symbols.values().map(|v| v.len()).sum();

    // Save to disk (best-effort, don't fail the tool call)
    let _ = index.save(miniswe_dir);
}

/// Count net braces (`{` minus `}`) on a line, skipping braces inside string
/// literals, character literals, and comments.
///
/// Handles: `"strings"`, `'c'` char literals, `//` line comments,
/// `/* block comments */`, escaped quotes (`\"`), and escaped backslashes (`\\`).
fn count_braces_outside_strings(line: &str) -> i32 {
    let chars: Vec<char> = line.chars().collect();
    let len = chars.len();
    let mut i = 0;
    let mut depth: i32 = 0;

    while i < len {
        let ch = chars[i];

        // Line comment: // — skip rest of line
        if ch == '/' && i + 1 < len && chars[i + 1] == '/' {
            break;
        }

        // Block comment: /* ... */ — skip until */
        if ch == '/' && i + 1 < len && chars[i + 1] == '*' {
            i += 2;
            while i + 1 < len {
                if chars[i] == '*' && chars[i + 1] == '/' {
                    i += 2;
                    break;
                }
                i += 1;
            }
            continue;
        }

        // String literal: "..." — skip until unescaped "
        if ch == '"' {
            i += 1;
            while i < len {
                if chars[i] == '\\' {
                    i += 2; // skip escaped char
                    continue;
                }
                if chars[i] == '"' {
                    i += 1;
                    break;
                }
                i += 1;
            }
            continue;
        }

        // Character literal: '.' or '\.' — skip
        if ch == '\'' {
            if i + 2 < len && chars[i + 1] == '\\' && i + 3 < len && chars[i + 3] == '\'' {
                i += 4; // '\x'
                continue;
            }
            if i + 2 < len && chars[i + 2] == '\'' {
                i += 3; // 'x'
                continue;
            }
            // Not a char literal (e.g., lifetime 'a) — skip the quote
            i += 1;
            continue;
        }

        if ch == '{' {
            depth += 1;
        } else if ch == '}' {
            depth -= 1;
        }

        i += 1;
    }

    depth
}

/// Compute end_line for each symbol by scanning for matching braces/dedent.
///
/// Heuristic: from the symbol's start line, track brace depth. When it
/// returns to 0 (or for Python, when indentation returns to the definition
/// level), that's the end line.
fn compute_end_lines(symbols: &mut Vec<Symbol>, content: &str) {
    let lines: Vec<&str> = content.lines().collect();
    let total = lines.len();

    // Sort by line number so we can use the next symbol as a boundary hint
    symbols.sort_by_key(|s| s.line);

    for i in 0..symbols.len() {
        let start = symbols[i].line.saturating_sub(1); // 0-indexed
        if start >= total {
            continue;
        }

        // Upper bound: next symbol's start or end of file
        let upper_bound = if i + 1 < symbols.len() {
            symbols[i + 1].line.saturating_sub(1)
        } else {
            total
        };

        let start_line = lines[start];
        let start_indent = start_line.len() - start_line.trim_start().len();

        // For brace-delimited languages: track depth
        if start_line.contains('{') || lines.get(start + 1).is_some_and(|l| l.trim() == "{") {
            let mut depth = 0;
            for j in start..upper_bound.min(total) {
                let brace_delta = count_braces_outside_strings(lines[j]);
                depth += brace_delta;
                if depth <= 0 && j > start {
                    symbols[i].end_line = j + 1;
                    break;
                }
            }
            if symbols[i].end_line == 0 {
                symbols[i].end_line = upper_bound;
            }
        } else {
            // For indent-delimited (Python): find where indent returns to start level
            for j in (start + 1)..upper_bound.min(total) {
                let line = lines[j];
                if line.trim().is_empty() {
                    continue;
                }
                let indent = line.len() - line.trim_start().len();
                if indent <= start_indent {
                    symbols[i].end_line = j; // line before the dedent
                    break;
                }
            }
            if symbols[i].end_line == 0 {
                symbols[i].end_line = upper_bound;
            }
        }
    }
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
        else if trimmed.starts_with("impl ") {
            if let Some(name) = extract_name_after(trimmed, "impl ") {
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
                    end_line: 0,
                    deps: Vec::new(),
                    parent_impl: None,
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
                    end_line: 0,
                    deps: Vec::new(),
                    parent_impl: None,
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
        else if trimmed.starts_with("export type ") || trimmed.starts_with("type ") {
            if let Some(name) = extract_name_after(trimmed, "type ") {
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
                    end_line: 0,
                    deps: Vec::new(),
                    parent_impl: None,
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
                    end_line: 0,
                    deps: Vec::new(),
                    parent_impl: None,
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
    crate::truncate_chars(s, 97)
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
            if rel_str.starts_with(".miniswe") {
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
