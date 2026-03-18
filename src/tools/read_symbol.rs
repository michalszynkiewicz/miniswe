//! read_symbol tool — Look up a symbol by name in the index, then read its
//! source code from disk using stored coordinates (file + line range).
//!
//! Much more token-efficient than read_file: returns only the function/struct
//! body instead of the whole file. If follow_deps is true, also reads the
//! source of type dependencies.

use anyhow::Result;
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

use crate::config::Config;
use crate::context::compress;
use crate::knowledge::ProjectIndex;
use super::ToolResult;

/// Maximum lines to return per symbol.
const MAX_LINES_PER_SYMBOL: usize = 100;

/// Maximum total symbols to return (including deps).
const MAX_SYMBOLS: usize = 5;

pub async fn execute(args: &Value, config: &Config) -> Result<ToolResult> {
    let name = args["name"].as_str().unwrap_or("");
    let follow_deps = args["follow_deps"].as_bool().unwrap_or(false);

    if name.is_empty() {
        return Ok(ToolResult::err("Missing required parameter: name".into()));
    }

    // Load the index
    let miniswe_dir = config.miniswe_dir();
    let index = match ProjectIndex::load(&miniswe_dir) {
        Ok(idx) => idx,
        Err(_) => {
            return Ok(ToolResult::err(
                "Index not available. Run `miniswe init` first.".into(),
            ));
        }
    };

    // Look up the symbol
    let symbols = index.lookup(name);
    if symbols.is_empty() {
        return Ok(ToolResult::err(format!(
            "Symbol '{name}' not found in index. Try search(\"{name}\") for a grep-based search."
        )));
    }

    let mut output = String::new();
    let mut symbols_shown = 0;

    for sym in &symbols {
        if symbols_shown >= MAX_SYMBOLS {
            output.push_str(&format!(
                "\n... and {} more definitions. Use search for full results.\n",
                symbols.len() - MAX_SYMBOLS
            ));
            break;
        }

        let source = read_symbol_source(sym.file.as_str(), sym.line, sym.end_line, config);
        output.push_str(&format!(
            "[{} {} in {}:{}",
            sym.kind, sym.name, sym.file, sym.line
        ));
        if sym.end_line > 0 {
            output.push_str(&format!("-{}", sym.end_line));
        }
        output.push_str("]\n");
        output.push_str(&source);
        output.push('\n');
        symbols_shown += 1;
    }

    // Follow dependencies if requested
    if follow_deps && symbols_shown < MAX_SYMBOLS {
        let mut dep_names: Vec<&str> = Vec::new();
        for sym in &symbols {
            for dep in &sym.deps {
                if dep != name && !dep_names.contains(&dep.as_str()) {
                    dep_names.push(dep.as_str());
                }
            }
        }

        for dep_name in dep_names {
            if symbols_shown >= MAX_SYMBOLS {
                break;
            }

            let dep_symbols = index.lookup(dep_name);
            for dep_sym in dep_symbols.iter().take(1) {
                let source = read_symbol_source(
                    dep_sym.file.as_str(),
                    dep_sym.line,
                    dep_sym.end_line,
                    config,
                );
                output.push_str(&format!(
                    "\n[dep: {} {} in {}:{}",
                    dep_sym.kind, dep_sym.name, dep_sym.file, dep_sym.line
                ));
                if dep_sym.end_line > 0 {
                    output.push_str(&format!("-{}", dep_sym.end_line));
                }
                output.push_str("]\n");
                output.push_str(&source);
                output.push('\n');
                symbols_shown += 1;
            }
        }
    }

    Ok(ToolResult::ok(output))
}

/// Read the source code for a symbol from disk, with line numbers.
fn read_symbol_source(
    file: &str,
    start_line: usize,
    end_line: usize,
    config: &Config,
) -> String {
    let path = resolve_path(file, config);

    let content = match fs::read_to_string(&path) {
        Ok(c) => c,
        Err(_) => return format!("(failed to read {file})"),
    };

    let lines: Vec<&str> = content.lines().collect();
    let total = lines.len();

    let start = start_line.saturating_sub(1).min(total);
    let end = if end_line > 0 {
        end_line.min(total)
    } else {
        // No end_line — show up to MAX_LINES_PER_SYMBOL lines or next blank line
        let mut e = start;
        for i in start..total.min(start + MAX_LINES_PER_SYMBOL) {
            e = i + 1;
            // Stop at double blank line (likely end of definition)
            if i > start + 2
                && lines.get(i).is_some_and(|l| l.trim().is_empty())
                && lines.get(i.wrapping_sub(1)).is_some_and(|l| l.trim().is_empty())
            {
                break;
            }
        }
        e
    };

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    // Apply compression (strip license, blank runs, std imports)
    let compressed = compress::compress_for_reading(&content, ext);

    let mut result = String::new();
    for i in start..end {
        if let Some(Some(line)) = compressed.get(i) {
            result.push_str(&format!("{:>4}│{line}\n", i + 1));
        }
    }

    result
}

fn resolve_path(path_str: &str, config: &Config) -> PathBuf {
    let path = PathBuf::from(path_str);
    if path.is_absolute() {
        path
    } else {
        config.project_root.join(path)
    }
}
