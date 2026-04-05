//! describe_code tool — Layer 2 of the two-layer code understanding system.
//!
//! Layer 1 (get_repo_map) shows ranked file→signature maps.
//! Layer 2 (describe_code) returns enriched descriptions for specific files:
//! - File summary (from index)
//! - Per-symbol: doc comment, parameter descriptions, what it does
//!
//! This avoids reading entire files when the LLM just needs to understand
//! what functions do and what parameters they take.

use serde_json::Value;

use crate::config::Config;
use crate::knowledge::ProjectIndex;
use super::ToolResult;

/// Execute the describe_code tool.
pub async fn execute(args: &Value, config: &Config) -> anyhow::Result<ToolResult> {
    let path = args["path"].as_str().unwrap_or("");
    if path.is_empty() {
        return Ok(ToolResult::err("path is required".into()));
    }

    let miniswe_dir = config.miniswe_dir();
    let index = match ProjectIndex::load(&miniswe_dir) {
        Ok(idx) => idx,
        Err(_) => return Ok(ToolResult::err("No project index. Run `miniswe init` first.".into())),
    };

    // Read the source file
    let abs_path = config.project_root.join(path);
    let content = match std::fs::read_to_string(&abs_path) {
        Ok(c) => c,
        Err(_) => return Ok(ToolResult::err(format!("Cannot read file: {path}"))),
    };

    let lines: Vec<&str> = content.lines().collect();

    // Get file summary from index
    let summary = index.summaries.get(path).map(|s| s.as_str()).unwrap_or("");

    // Collect symbols for this file
    let mut file_symbols: Vec<_> = index.symbols.values()
        .flat_map(|syms| syms.iter())
        .filter(|s| s.file == path)
        .collect();
    file_symbols.sort_by_key(|s| s.line);

    if file_symbols.is_empty() {
        return Ok(ToolResult::ok(format!(
            "{path}: {summary}\nNo indexed symbols in this file."
        )));
    }

    // Optional: filter to specific symbols
    let filter_symbols: Option<Vec<&str>> = args["symbols"].as_str()
        .map(|s| s.split(',').map(|n| n.trim()).collect());

    let mut output = String::new();
    output.push_str(&format!("{path}: {summary}\n\n"));

    for sym in &file_symbols {
        // Apply symbol filter if provided
        if let Some(ref filter) = filter_symbols {
            if !filter.iter().any(|f| *f == sym.name) {
                continue;
            }
        }

        // Skip impl block markers — their methods carry the info
        if sym.kind == "impl" {
            continue;
        }

        // Extract doc comment above this symbol
        let doc = extract_doc_comment(&lines, sym.line);

        // Extract parameter descriptions from the signature
        let params = extract_param_info(&sym.signature, &sym.kind);

        // Format the description
        output.push_str(&format!("{}:{} [{}] {}\n", sym.name, sym.line, sym.kind, sym.signature));

        if let Some(doc) = &doc {
            output.push_str(&format!("  {doc}\n"));
        }

        if !params.is_empty() {
            output.push_str(&format!("  params: {params}\n"));
        }

        output.push('\n');
    }

    Ok(ToolResult::ok(output))
}

/// Extract the doc comment immediately above a symbol's start line.
///
/// Handles: `///`, `//!`, `/** */`, `# comment` (Python), `//` (Go/JS).
/// Returns the concatenated doc comment as a single string, or None.
fn extract_doc_comment(lines: &[&str], symbol_line: usize) -> Option<String> {
    if symbol_line == 0 || symbol_line > lines.len() {
        return None;
    }

    // Walk backwards from the line before the symbol
    let start_idx = symbol_line - 1; // 0-indexed line before symbol
    if start_idx == 0 {
        return None;
    }

    let mut doc_lines: Vec<String> = Vec::new();
    let mut i = start_idx.saturating_sub(1);

    loop {
        let line = lines[i].trim();

        // Rust: /// or //!
        if let Some(text) = line.strip_prefix("///") {
            doc_lines.push(text.trim().to_string());
        } else if let Some(text) = line.strip_prefix("//!") {
            doc_lines.push(text.trim().to_string());
        }
        // Python: # comment (but not #! shebang)
        else if line.starts_with('#') && !line.starts_with("#!") {
            let text = line.trim_start_matches('#').trim();
            doc_lines.push(text.to_string());
        }
        // JSDoc/C: lines starting with * inside a block comment
        else if line.starts_with('*') && !line.starts_with("*/") {
            let text = line.trim_start_matches('*').trim();
            if !text.starts_with('@') { // skip @param tags (we extract params separately)
                doc_lines.push(text.to_string());
            }
        }
        // Block comment start: /** or /*
        else if line.starts_with("/**") || line.starts_with("/*") {
            let text = line.trim_start_matches("/**").trim_start_matches("/*").trim();
            if !text.is_empty() && !text.starts_with("*/") {
                doc_lines.push(text.to_string());
            }
            break; // reached start of block comment
        }
        // Go/JS: // regular comment (only if we're still in the comment block)
        else if line.starts_with("//") {
            let text = line.trim_start_matches("//").trim();
            doc_lines.push(text.to_string());
        }
        // Blank line or attribute — stop collecting
        else if line.is_empty() || line.starts_with('@') {
            break;
        }
        // Non-comment line — stop
        else {
            break;
        }

        if i == 0 {
            break;
        }
        i -= 1;
    }

    if doc_lines.is_empty() {
        return None;
    }

    doc_lines.reverse();
    // Filter out empty lines and join
    let joined: String = doc_lines
        .into_iter()
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" ");

    if joined.is_empty() { None } else { Some(crate::truncate_chars(&joined, 200)) }
}

/// Extract parameter info from a function signature.
///
/// Returns a brief summary like "path: &str, start_line: Option<usize>, end_line: Option<usize>"
fn extract_param_info(signature: &str, kind: &str) -> String {
    if kind != "function" {
        return String::new();
    }

    // Find the parameter list between first ( and matching )
    let open = match signature.find('(') {
        Some(i) => i,
        None => return String::new(),
    };

    // Find matching close paren
    let mut depth = 0;
    let mut close = open;
    for (i, ch) in signature[open..].char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    close = open + i;
                    break;
                }
            }
            _ => {}
        }
    }

    if close <= open + 1 {
        return String::new();
    }

    let params = &signature[open + 1..close];
    // Strip &self, &mut self, self
    let cleaned: Vec<&str> = params
        .split(',')
        .map(|p| p.trim())
        .filter(|p| !p.is_empty() && *p != "&self" && *p != "&mut self" && *p != "self")
        .collect();

    if cleaned.is_empty() {
        return String::new();
    }

    cleaned.join(", ")
}
