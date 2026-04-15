//! Project walker: full-index and incremental per-file re-index, plus
//! the file-size audit used by the TUI warnings.

use std::path::Path;

use anyhow::Result;
use ignore::WalkBuilder;

use crate::knowledge::ts_extract;
use crate::knowledge::{ProjectIndex, Symbol};

use super::end_line::compute_end_lines;
use super::lang::extract_symbols;
use super::summary::generate_summary;

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

        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

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
                if let Some(prev) = previous
                    && !prev.is_stale(&rel_str, mtime)
                {
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

                // (Re-)index this file
                if let Ok(content) = std::fs::read_to_string(path) {
                    let mut symbols: Vec<Symbol> =
                        if let Some(ts_result) = ts_extract::extract(&rel_str, &content, ext) {
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
        tracing::info!(
            "Incremental index: {reused} files reused, {} re-indexed",
            file_count - reused
        );
    }

    Ok(index)
}

/// Re-index a single file in an existing index.
///
/// Removes old symbols for that file, re-extracts, recomputes end_lines,
/// updates mtime, and saves the index to disk. Takes <1ms per file.
pub fn reindex_file(rel_path: &str, abs_path: &Path, index: &mut ProjectIndex, miniswe_dir: &Path) {
    let ext = abs_path.extension().and_then(|e| e.to_str()).unwrap_or("");

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
    let mut symbols: Vec<Symbol> =
        if let Some(ts_result) = ts_extract::extract(rel_path, &content, ext) {
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
    index
        .mtimes
        .insert(rel_path.to_string(), file_mtime(abs_path));

    // Recount
    index.total_symbols = index.symbols.values().map(|v| v.len()).sum();

    // Save to disk (best-effort, don't fail the tool call)
    let _ = index.save(miniswe_dir);
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

        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

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
