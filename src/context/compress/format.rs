//! Aggressive code-format stripping — removes comments, collapses
//! whitespace, and drops blank lines. Used on the model's *non-active*
//! context buffer where exact round-tripping isn't needed.
//!
//! The active edit target keeps its formatting so the edit tool can find
//! exact `old_content` matches.

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
}
