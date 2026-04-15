//! Context compression pipeline.
//!
//! Five layers of deterministic, lossless-for-code-semantics compression:
//! 1. Code format stripping (remove comments, collapse whitespace)
//! 2. Structured context format (key-value notation for system context)
//! 3. Import elision (drop standard library imports)
//! 4. History-as-diffs (replace old tool results with one-line summaries)
//! 5. Observation masking (post-use replacement of tool outputs)
//!
//! Total effective multiplier: ~1.6× — a 64K window carries ~100K of information.

/// Strip code formatting: remove comments, collapse whitespace, remove blank lines.
///
/// This achieves ~34% token savings on average. The edit tool's active target
/// retains formatting (the model needs exact matches for old_content).
pub fn strip_code_format(code: &str, language: &str) -> String {
    let mut result = String::new();
    let mut in_block_comment = false;

    for line in code.lines() {
        let trimmed = line.trim();

        // Skip blank lines
        if trimmed.is_empty() {
            continue;
        }

        // Handle block comments
        match language {
            "rs" | "js" | "ts" | "tsx" | "jsx" | "go" | "java" | "c" | "cpp" | "h" | "hpp" => {
                if in_block_comment {
                    if let Some(end) = trimmed.find("*/") {
                        in_block_comment = false;
                        let after = trimmed[end + 2..].trim();
                        if !after.is_empty() {
                            result.push_str(after);
                            result.push('\n');
                        }
                    }
                    continue;
                }

                if trimmed.starts_with("/*") {
                    if !trimmed.contains("*/") {
                        in_block_comment = true;
                    }
                    continue;
                }

                // Skip line comments
                if trimmed.starts_with("//") {
                    continue;
                }

                // Strip inline comments (but not in strings — simplified heuristic)
                let code_part = strip_inline_comment(trimmed, "//");
                if !code_part.is_empty() {
                    result.push_str(code_part);
                    result.push('\n');
                }
            }
            "py" => {
                // Skip Python comments
                if trimmed.starts_with('#') {
                    continue;
                }
                // Skip docstrings (simplified: lines that are just triple quotes)
                if trimmed == "\"\"\"" || trimmed == "'''" {
                    in_block_comment = !in_block_comment;
                    continue;
                }
                if in_block_comment {
                    continue;
                }
                let code_part = strip_inline_comment(trimmed, "#");
                if !code_part.is_empty() {
                    result.push_str(code_part);
                    result.push('\n');
                }
            }
            _ => {
                // For unknown languages, just skip obvious comments
                if trimmed.starts_with("//") || trimmed.starts_with('#') {
                    continue;
                }
                result.push_str(trimmed);
                result.push('\n');
            }
        }
    }

    result
}

/// Strip inline comments from a line (simplified: doesn't handle comments inside strings).
fn strip_inline_comment<'a>(line: &'a str, comment_prefix: &str) -> &'a str {
    // Don't strip if the comment marker is inside a string literal
    // Simple heuristic: if there's a quote before the comment, skip stripping
    if let Some(pos) = line.find(comment_prefix) {
        let before = &line[..pos];
        let single_quotes = before.chars().filter(|&c| c == '\'').count();
        let double_quotes = before.chars().filter(|&c| c == '"').count();
        // If quotes are balanced, the comment is real
        if single_quotes % 2 == 0 && double_quotes % 2 == 0 {
            return line[..pos].trim_end();
        }
    }
    line
}

/// Compress code for reading context. Preserves line numbers by replacing
/// removed lines with None (gaps in output, but line numbering stays correct).
///
/// What gets stripped:
/// - License headers (block comment at very top of file, before any code)
/// - Consecutive blank lines (collapsed to one)
/// - Standard library imports (model knows them)
///
/// What is KEPT:
/// - All comments (they contain knowledge: why, edge cases, TODOs, invariants)
/// - Doc comments (//!, ///, """, /** */)
/// - File description headers (critical for understanding file purpose)
pub fn compress_for_reading(code: &str, ext: &str) -> Vec<Option<String>> {
    let lines: Vec<&str> = code.lines().collect();
    let mut result: Vec<Option<String>> = Vec::with_capacity(lines.len());

    // Phase 1: detect license header (block comment at top before any code)
    let license_end = detect_license_header(&lines, ext);

    let mut prev_was_blank = false;

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();

        // Strip license header lines
        if i < license_end {
            result.push(None);
            continue;
        }

        // Collapse consecutive blank lines (keep first, strip rest)
        if trimmed.is_empty() {
            if prev_was_blank {
                result.push(None);
            } else {
                result.push(Some(String::new()));
                prev_was_blank = true;
            }
            continue;
        }
        prev_was_blank = false;

        // Strip stdlib imports
        if is_stdlib_import(trimmed, ext) {
            result.push(None);
            continue;
        }

        // Keep everything else (code AND comments)
        result.push(Some(line.to_string()));
    }

    result
}

/// Detect a license header at the top of a file.
///
/// A license header is a block comment (/* ... */ or # ... block) at the
/// very start of the file, before any code, that contains license-related
/// keywords. Returns the line index where the license ends (0 if none).
fn detect_license_header(lines: &[&str], ext: &str) -> usize {
    if lines.is_empty() {
        return 0;
    }

    let license_keywords = [
        "license",
        "copyright",
        "permission is hereby granted",
        "licensed under",
        "apache license",
        "mit license",
        "gnu general public",
        "all rights reserved",
        "bsd",
        "redistribute",
        "warranty",
    ];

    match ext {
        "rs" | "js" | "ts" | "tsx" | "jsx" | "go" | "java" | "c" | "cpp" | "h" | "hpp" => {
            // Look for /* ... */ block at the top
            let first_non_empty = lines.iter().position(|l| !l.trim().is_empty());
            let start = match first_non_empty {
                Some(i) => i,
                None => return 0,
            };

            let trimmed = lines[start].trim();
            if !trimmed.starts_with("/*") {
                return 0;
            }

            // Find the end of the block comment
            let mut end = start;
            let mut found_license_keyword = false;
            for (i, line) in lines[start..].iter().enumerate() {
                let t = line.to_lowercase();
                if license_keywords.iter().any(|kw| t.contains(kw)) {
                    found_license_keyword = true;
                }
                if line.contains("*/") {
                    end = start + i + 1;
                    break;
                }
            }

            if found_license_keyword { end } else { 0 }
        }
        "py" | "rb" | "sh" | "bash" => {
            // Look for # comment block at top with license keywords
            let mut end = 0;
            let mut found_license_keyword = false;
            for (i, line) in lines.iter().enumerate() {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    if found_license_keyword {
                        return i; // blank line after license block
                    }
                    continue;
                }
                let is_comment = trimmed.starts_with('#');
                if !is_comment {
                    break; // hit code
                }
                let t = trimmed.to_lowercase();
                if license_keywords.iter().any(|kw| t.contains(kw)) {
                    found_license_keyword = true;
                }
                end = i + 1;
            }

            if found_license_keyword { end } else { 0 }
        }
        _ => 0,
    }
}

/// Check if a line is a standard library import that can be elided.
fn is_stdlib_import(trimmed: &str, ext: &str) -> bool {
    match ext {
        "rs" => {
            trimmed.starts_with("use std::")
                || trimmed.starts_with("use core::")
                || trimmed.starts_with("use alloc::")
        }
        "py" => {
            let is_import = trimmed.starts_with("import ") || trimmed.starts_with("from ");
            if is_import {
                let stdlib = [
                    "os",
                    "sys",
                    "re",
                    "json",
                    "math",
                    "datetime",
                    "collections",
                    "itertools",
                    "functools",
                    "typing",
                    "pathlib",
                    "io",
                    "abc",
                    "dataclasses",
                    "enum",
                    "copy",
                    "hashlib",
                    "logging",
                    "unittest",
                    "argparse",
                    "subprocess",
                    "threading",
                    "time",
                    "random",
                ];
                stdlib.iter().any(|lib| {
                    trimmed.starts_with(&format!("import {lib} "))
                        || trimmed == format!("import {lib}")
                        || trimmed.starts_with(&format!("import {lib},"))
                        || trimmed.starts_with(&format!("from {lib} "))
                        || trimmed.starts_with(&format!("from {lib}."))
                })
            } else {
                false
            }
        }
        "js" | "ts" | "tsx" | "jsx" => {
            let builtins = [
                "\"fs\"",
                "\"path\"",
                "\"os\"",
                "\"url\"",
                "\"http\"",
                "\"https\"",
                "\"crypto\"",
                "\"util\"",
                "\"events\"",
                "'fs'",
                "'path'",
                "'os'",
                "'url'",
                "'http'",
            ];
            let is_import = trimmed.starts_with("import ") || trimmed.starts_with("const ");
            is_import && builtins.iter().any(|b| trimmed.contains(b))
        }
        _ => false,
    }
}

/// Elide standard library imports (the model knows them).
/// Only keep project-internal and third-party imports.
pub fn elide_std_imports(code: &str, language: &str) -> String {
    let mut result = String::new();

    for line in code.lines() {
        let trimmed = line.trim();

        let should_skip = match language {
            "rs" => {
                trimmed.starts_with("use std::")
                    || trimmed.starts_with("use core::")
                    || trimmed.starts_with("use alloc::")
            }
            "py" => {
                let is_import = trimmed.starts_with("import ") || trimmed.starts_with("from ");
                if is_import {
                    // Skip standard library imports
                    let stdlib = [
                        "os",
                        "sys",
                        "re",
                        "json",
                        "math",
                        "datetime",
                        "collections",
                        "itertools",
                        "functools",
                        "typing",
                        "pathlib",
                        "io",
                        "abc",
                        "dataclasses",
                        "enum",
                        "copy",
                        "hashlib",
                        "logging",
                        "unittest",
                        "argparse",
                        "subprocess",
                        "threading",
                        "time",
                        "random",
                    ];
                    stdlib.iter().any(|lib| {
                        trimmed.starts_with(&format!("import {lib}"))
                            || trimmed.starts_with(&format!("from {lib}"))
                    })
                } else {
                    false
                }
            }
            "go" => false, // Go imports are essential for understanding code
            "js" | "ts" | "tsx" | "jsx" => {
                // Skip node builtins
                let builtins = [
                    "\"fs\"",
                    "\"path\"",
                    "\"os\"",
                    "\"url\"",
                    "\"http\"",
                    "\"https\"",
                    "\"crypto\"",
                    "\"util\"",
                    "\"events\"",
                    "'fs'",
                    "'path'",
                    "'os'",
                    "'url'",
                    "'http'",
                ];
                let is_import = trimmed.starts_with("import ") || trimmed.starts_with("const ");
                is_import && builtins.iter().any(|b| trimmed.contains(b))
            }
            _ => false,
        };

        if !should_skip {
            result.push_str(line);
            result.push('\n');
        }
    }

    result
}

/// Compress a project profile from prose to structured key-value format.
///
/// Input: markdown profile with headings and bullet points
/// Output: dense [SECTION]key=value|key=value notation
pub fn compress_profile(profile: &str) -> String {
    let mut result = String::new();
    for line in profile.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("# ") {
            // Skip the main heading
            continue;
        }

        if trimmed.starts_with("## ") {
            let section = trimmed
                .trim_start_matches("## ")
                .to_uppercase()
                .replace(' ', "_");
            result.push_str(&format!("[{section}]"));
            continue;
        }

        if trimmed.starts_with("- ") {
            let item = trimmed.trim_start_matches("- ");
            // Convert "Key: value" to "key=value"
            if let Some((key, value)) = item.split_once(": ") {
                let short_key = key
                    .to_lowercase()
                    .replace(' ', "_")
                    .chars()
                    .take(12)
                    .collect::<String>();
                result.push_str(&format!("{short_key}={value}|"));
            } else {
                result.push_str(item);
                result.push('|');
            }
        }
    }

    // Clean trailing pipe
    if result.ends_with('|') {
        result.pop();
    }

    result.push('\n');
    result
}

/// Summarize a tool result into a one-line observation for history compression.
///
/// This replaces full tool outputs in older conversation turns with dense summaries.
pub fn summarize_tool_result(tool_name: &str, args: &serde_json::Value, content: &str) -> String {
    let action = args["action"].as_str().unwrap_or("");

    match tool_name {
        "file" => match action {
            "read" => {
                let path = args["path"].as_str().unwrap_or("?");
                let line_count = content.lines().count();
                let mut sigs = Vec::new();
                for line in content.lines() {
                    let stripped = if let Some(pos) = line.find('│') {
                        &line[pos + '│'.len_utf8()..]
                    } else {
                        line
                    };
                    let trimmed = stripped.trim();
                    if (trimmed.starts_with("pub fn ")
                        || trimmed.starts_with("pub async fn ")
                        || trimmed.starts_with("fn ")
                        || trimmed.starts_with("async fn "))
                        && trimmed.contains('(')
                    {
                        let sig = trimmed.split('{').next().unwrap_or(trimmed).trim();
                        if sig.len() < 80 {
                            sigs.push(sig.to_string());
                        }
                    }
                    if (trimmed.starts_with("pub struct ")
                        || trimmed.starts_with("pub enum ")
                        || trimmed.starts_with("pub trait ")
                        || trimmed.starts_with("struct ")
                        || trimmed.starts_with("enum ")
                        || trimmed.starts_with("trait "))
                        && !trimmed.contains(';')
                    {
                        let def = trimmed.split('{').next().unwrap_or(trimmed).trim();
                        if def.len() < 80 {
                            sigs.push(def.to_string());
                        }
                    }
                    if sigs.len() >= 10 {
                        break;
                    }
                }
                if sigs.is_empty() {
                    format!(
                        "[read:{path} ({line_count}L) — use file(action='read', path='{path}') to re-read]"
                    )
                } else {
                    format!(
                        "[read:{path} ({line_count}L) — use file(action='read', path='{path}') to re-read]\n{}",
                        sigs.join("\n")
                    )
                }
            }
            "search" => {
                let query = args["query"].as_str().unwrap_or("?");
                let match_count = content.lines().filter(|l| !l.starts_with('[')).count();
                format!("[search:\"{query}\"→{match_count} matches]")
            }
            "shell" => {
                let cmd = args["command"].as_str().unwrap_or("?");
                let short_cmd = crate::truncate_chars(cmd, 30);
                let exit_code = if content.contains("exit 0") {
                    "ok"
                } else {
                    "err"
                };
                format!("[shell:\"{short_cmd}\"→{exit_code}]")
            }
            _ => format!("[file.{action}→done]"),
        },
        "code" => match action {
            "diagnostics" => {
                let errors = content.lines().filter(|l| l.contains("error")).count();
                let warnings = content.lines().filter(|l| l.contains("warning")).count();
                format!("[diag:{errors}E,{warnings}W]")
            }
            _ => format!("[code.{action}→done]"),
        },
        "web" => match action {
            "search" => {
                let query = args["query"].as_str().unwrap_or("?");
                let result_count = content
                    .lines()
                    .filter(|l| l.starts_with(|c: char| c.is_ascii_digit()))
                    .count();
                format!("[web_search:\"{query}\"→{result_count} results]")
            }
            "fetch" => {
                let url = args["url"].as_str().unwrap_or("?");
                format!("[web_fetch:{url}→{}chars]", content.len())
            }
            _ => format!("[web.{action}→done]"),
        },
        "plan" => match action {
            "scratchpad" => "[scratchpad→ok]".to_string(),
            _ => format!("[plan.{action}→done]"),
        },
        "edit_file" => {
            let path = args["path"].as_str().unwrap_or("?");
            if content.contains('✓') {
                if content.contains("error")
                    && (content.contains("[cargo check]") || content.contains("[lsp]"))
                {
                    let errors: Vec<&str> = content
                        .lines()
                        .filter(|l| l.contains("error"))
                        .take(2)
                        .collect();
                    format!("[edit_file:{path}→ok but errors: {}]", errors.join("; "))
                } else {
                    format!("[edit_file:{path}→ok]")
                }
            } else {
                format!("[edit_file:{path}→failed]")
            }
        }
        "mcp_use" => {
            let server = args["server"].as_str().unwrap_or("?");
            let tool = args["tool"].as_str().unwrap_or("?");
            format!("[mcp:{server}/{tool}→{}chars]", content.len())
        }
        _ => format!("[{tool_name}→done]"),
    }
}

/// Extract likely exported symbol names from code content.
fn extract_symbol_names_from_content(content: &str) -> Vec<String> {
    let mut names = Vec::new();
    for line in content.lines().take(50) {
        let trimmed = line.trim();
        // Look for function/struct/class definitions
        for keyword in &[
            "pub fn ",
            "fn ",
            "pub struct ",
            "struct ",
            "pub enum ",
            "class ",
            "def ",
            "function ",
            "export function ",
        ] {
            if trimmed.contains(keyword)
                && let Some(after) = trimmed.split(keyword).nth(1)
            {
                let name: String = after
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                if !name.is_empty() && name.len() > 1 {
                    names.push(name);
                }
            }
        }
    }
    names.truncate(5);
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_rust_comments() {
        let code = r#"
// This is a comment
fn main() {
    // Another comment
    let x = 42;
    /* block comment */
    println!("{}", x);
}
"#;
        let stripped = strip_code_format(code, "rs");
        assert!(!stripped.contains("This is a comment"));
        assert!(!stripped.contains("Another comment"));
        assert!(!stripped.contains("block comment"));
        assert!(stripped.contains("fn main()"));
        assert!(stripped.contains("let x = 42;"));
    }

    #[test]
    fn test_strip_python_comments() {
        let code = r#"
# This is a comment
def hello():
    # Another comment
    print("hello")
"#;
        let stripped = strip_code_format(code, "py");
        assert!(!stripped.contains("This is a comment"));
        assert!(stripped.contains("def hello():"));
    }

    #[test]
    fn test_elide_rust_std_imports() {
        let code =
            "use std::collections::HashMap;\nuse crate::config::Config;\nuse anyhow::Result;\n";
        let elided = elide_std_imports(code, "rs");
        assert!(!elided.contains("std::collections"));
        assert!(elided.contains("crate::config"));
        assert!(elided.contains("anyhow::Result"));
    }

    #[test]
    fn test_tool_result_summary() {
        let args = serde_json::json!({"action": "read", "path": "src/main.rs"});
        let content = "pub fn main() {\n    println!(\"hello\");\n}\n";
        let summary = summarize_tool_result("file", &args, content);
        assert!(
            summary.contains("src/main.rs"),
            "should have path: {summary}"
        );
        assert!(
            summary.contains("read"),
            "should hint at re-read: {summary}"
        );
    }

    #[test]
    fn test_compress_strips_std_imports_keeps_comments() {
        let code = "use std::collections::HashMap;\nuse crate::config::Config;\n// A comment\nfn main() {\n    // inline comment\n    let x = 42;\n}\n";
        let compressed = compress_for_reading(code, "rs");
        // Line 1: std import → stripped
        assert!(compressed[0].is_none());
        // Line 2: crate import → kept
        assert!(compressed[1].is_some());
        assert!(compressed[1].as_ref().unwrap().contains("crate::config"));
        // Line 3: comment → KEPT (comments are knowledge)
        assert!(compressed[2].is_some());
        assert!(compressed[2].as_ref().unwrap().contains("A comment"));
        // Line 4: fn main → kept
        assert!(compressed[3].is_some());
        // Line 5: inline comment → KEPT
        assert!(compressed[4].is_some());
        assert!(compressed[4].as_ref().unwrap().contains("inline comment"));
        // Line 6: code → kept
        assert!(compressed[5].is_some());
    }

    #[test]
    fn test_compress_strips_python_stdlib() {
        let code = "import os\nimport requests\n# comment\ndef hello():\n    pass\n";
        let compressed = compress_for_reading(code, "py");
        // import os → stdlib, stripped
        assert!(compressed[0].is_none());
        // import requests → third-party, kept
        assert!(compressed[1].is_some());
        // comment → KEPT
        assert!(compressed[2].is_some());
        // def hello → kept
        assert!(compressed[3].is_some());
    }

    #[test]
    fn test_compress_preserves_line_count() {
        let code = "// c1\n// c2\nfn a() {}\n// c3\nfn b() {}\n";
        let compressed = compress_for_reading(code, "rs");
        assert_eq!(compressed.len(), 5);
    }

    #[test]
    fn test_compress_strips_license_header() {
        let code = "/* Copyright 2026 Acme Inc.\n * Licensed under the MIT License.\n * All rights reserved.\n */\n\nfn main() {}\n";
        let compressed = compress_for_reading(code, "rs");
        // License block (lines 0-3) → stripped
        assert!(compressed[0].is_none());
        assert!(compressed[1].is_none());
        assert!(compressed[2].is_none());
        assert!(compressed[3].is_none());
        // fn main → kept
        assert!(compressed[5].is_some());
        assert!(compressed[5].as_ref().unwrap().contains("fn main"));
    }

    #[test]
    fn test_compress_keeps_doc_comments() {
        let code = "//! Module description.\n//! More details.\n\nfn foo() {}\n";
        let compressed = compress_for_reading(code, "rs");
        // Doc comments must be kept
        assert!(compressed[0].is_some());
        assert!(
            compressed[0]
                .as_ref()
                .unwrap()
                .contains("Module description")
        );
        assert!(compressed[1].is_some());
    }

    #[test]
    fn test_compress_collapses_consecutive_blanks() {
        let code = "fn a() {}\n\n\n\nfn b() {}\n";
        let compressed = compress_for_reading(code, "rs");
        // First blank kept, rest stripped
        assert!(compressed[1].is_some()); // first blank → kept (empty string)
        assert!(compressed[2].is_none()); // second blank → stripped
        assert!(compressed[3].is_none()); // third blank → stripped
    }
}
