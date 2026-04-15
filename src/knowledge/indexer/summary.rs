//! Per-file one-line summary generation. Prefers the module-level doc
//! comment (`//!`, `"""`, `/** */`) when present and falls back to a
//! symbol-based summary.

use std::collections::HashMap;

use crate::knowledge::Symbol;

/// Generate a one-line summary of a file.
///
/// Prefers the module-level doc comment (`//!`, `"""`, `/**`, etc.) if
/// present — these are high-signal, human-written descriptions of the
/// file's purpose. Falls back to a symbol-based summary.
pub fn generate_summary(content: &str, symbols: &[Symbol], ext: &str) -> String {
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
    let names: Vec<&str> = symbols.iter().take(5).map(|s| s.name.as_str()).collect();

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
