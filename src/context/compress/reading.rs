//! Line-preserving compression for the `read_file` tool.
//!
//! Output is a `Vec<Option<String>>` with one entry per source line:
//! `Some(s)` for lines that should be shown, `None` for lines that were
//! stripped. Keeping the positions lets the tool display the same line
//! numbers as the file on disk, which matters because the model's edits
//! are expressed in terms of those numbers.
//!
//! What gets stripped:
//! - License headers (block comment at the very top of the file)
//! - Consecutive blank lines (collapsed to one)
//! - Standard library imports (the model already knows them)
//!
//! What is KEPT:
//! - All comments (they contain knowledge: why, edge cases, TODOs)
//! - Doc comments (`//!`, `///`, `"""`, `/** */`)
//! - File-description headers (critical for orientation)

/// Compress code for reading context. Preserves line numbers by replacing
/// removed lines with None (gaps in output, but line numbering stays correct).
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

#[cfg(test)]
mod tests {
    use super::*;

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
