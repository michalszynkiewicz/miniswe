//! Compute `end_line` for extracted symbols using brace depth (or
//! indentation, for Python) to find where a symbol's body ends.

use crate::knowledge::Symbol;

/// Count net braces (`{` minus `}`) on a line, skipping braces inside
/// string literals, character literals, and comments.
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

/// Compute end_line for each symbol by scanning for matching braces / dedent.
///
/// Heuristic: from the symbol's start line, track brace depth. When it
/// returns to 0 (or for Python, when indentation returns to the definition
/// level), that's the end line.
pub fn compute_end_lines(symbols: &mut Vec<Symbol>, content: &str) {
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

        let scan_end = upper_bound.min(total);
        // For brace-delimited languages: track depth
        if start_line.contains('{') || lines.get(start + 1).is_some_and(|l| l.trim() == "{") {
            let mut depth = 0;
            for (j, line) in lines.iter().enumerate().take(scan_end).skip(start) {
                depth += count_braces_outside_strings(line);
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
            for (j, line) in lines.iter().enumerate().take(scan_end).skip(start + 1) {
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
