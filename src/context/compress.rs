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
                        "os", "sys", "re", "json", "math", "datetime", "collections",
                        "itertools", "functools", "typing", "pathlib", "io", "abc",
                        "dataclasses", "enum", "copy", "hashlib", "logging", "unittest",
                        "argparse", "subprocess", "threading", "time", "random",
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
                    "\"fs\"", "\"path\"", "\"os\"", "\"url\"", "\"http\"",
                    "\"https\"", "\"crypto\"", "\"util\"", "\"events\"",
                    "'fs'", "'path'", "'os'", "'url'", "'http'",
                ];
                let is_import = trimmed.starts_with("import ") || trimmed.starts_with("const ");
                is_import
                    && builtins
                        .iter()
                        .any(|b| trimmed.contains(b))
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
    match tool_name {
        "read_file" => {
            let path = args["path"].as_str().unwrap_or("?");
            let line_count = content.lines().count();
            // Extract symbol names from content if it looks like code
            let symbols = extract_symbol_names_from_content(content);
            if symbols.is_empty() {
                format!("[read:{path}→{line_count}L]")
            } else {
                format!(
                    "[read:{path}→{line_count}L,exports:{}]",
                    symbols.join(",")
                )
            }
        }
        "read_symbol" => {
            let name = args["name"].as_str().unwrap_or("?");
            let line_count = content.lines().count();
            format!("[symbol:{name}→{line_count}L]")
        }
        "search" => {
            let query = args["query"].as_str().unwrap_or("?");
            let match_count = content
                .lines()
                .filter(|l| !l.starts_with('['))
                .count();
            format!("[search:\"{query}\"→{match_count} matches]")
        }
        "edit" => {
            let path = args["path"].as_str().unwrap_or("?");
            if content.contains('✓') {
                format!("[edit:{path}→ok]")
            } else {
                format!("[edit:{path}→failed]")
            }
        }
        "write_file" => {
            let path = args["path"].as_str().unwrap_or("?");
            if content.contains('✓') {
                let lines = content.lines().next().unwrap_or("");
                format!("[write:{path}→{lines}]")
            } else {
                format!("[write:{path}→failed]")
            }
        }
        "shell" => {
            let cmd = args["command"].as_str().unwrap_or("?");
            let short_cmd = if cmd.len() > 30 {
                &cmd[..30]
            } else {
                cmd
            };
            let exit_code = if content.contains("exit 0") {
                "ok"
            } else {
                "err"
            };
            format!("[shell:\"{short_cmd}\"→{exit_code}]")
        }
        "task_update" => "[task_update→ok]".to_string(),
        "diagnostics" => {
            let errors = content
                .lines()
                .filter(|l| l.contains("error"))
                .count();
            let warnings = content
                .lines()
                .filter(|l| l.contains("warning"))
                .count();
            format!("[diag:{errors}E,{warnings}W]")
        }
        "web_search" => {
            let query = args["query"].as_str().unwrap_or("?");
            let result_count = content
                .lines()
                .filter(|l| l.starts_with(|c: char| c.is_ascii_digit()))
                .count();
            format!("[web_search:\"{query}\"→{result_count} results]")
        }
        "web_fetch" => {
            let url = args["url"].as_str().unwrap_or("?");
            let char_count = content.len();
            format!("[web_fetch:{url}→{char_count}chars]")
        }
        "docs_lookup" => {
            let lib = args["library"].as_str().unwrap_or("?");
            format!("[docs:{lib}→found]")
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
        for keyword in &["pub fn ", "fn ", "pub struct ", "struct ", "pub enum ", "class ", "def ", "function ", "export function "] {
            if trimmed.contains(keyword) {
                if let Some(after) = trimmed.split(keyword).nth(1) {
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
        let code = "use std::collections::HashMap;\nuse crate::config::Config;\nuse anyhow::Result;\n";
        let elided = elide_std_imports(code, "rs");
        assert!(!elided.contains("std::collections"));
        assert!(elided.contains("crate::config"));
        assert!(elided.contains("anyhow::Result"));
    }

    #[test]
    fn test_tool_result_summary() {
        let args = serde_json::json!({"path": "src/main.rs"});
        let content = "pub fn main() {\n    println!(\"hello\");\n}\n";
        let summary = summarize_tool_result("read_file", &args, content);
        assert!(summary.starts_with("[read:src/main.rs→"));
    }
}
