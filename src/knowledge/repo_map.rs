//! Repo map rendering — budget-aware, tiered code structure overview.
//!
//! Renders a ranked overview of the most relevant code structure,
//! using PageRank scores to determine which files and symbols get
//! the most detail.
//!
//! Three tiers:
//! - Tier 0: Name only (1-2 tokens per symbol)
//! - Tier 1: Signature skeleton (5-15 tokens per symbol)
//! - Tier 2+: Omitted (not included)

use std::collections::HashMap;

use super::graph::DependencyGraph;
use super::{ProjectIndex, Symbol};

/// Render a repo map within the given token budget.
///
/// Returns the map as a string ready for injection into the LLM context.
///
/// `task_keywords` personalizes the ranking — files matching these keywords
/// get boosted in the PageRank scores.
pub fn render(
    index: &ProjectIndex,
    graph: &DependencyGraph,
    token_budget: usize,
    task_keywords: &[&str],
) -> String {
    if index.symbols.is_empty() {
        return String::new();
    }

    // Get personalized scores
    let scores = graph.personalized_scores(task_keywords);

    // Group symbols by file, sort files by score
    let mut file_symbols: HashMap<&str, Vec<&Symbol>> = HashMap::new();
    for symbols in index.symbols.values() {
        for sym in symbols {
            file_symbols
                .entry(sym.file.as_str())
                .or_default()
                .push(sym);
        }
    }

    // Sort files by PageRank score (descending)
    let mut ranked_files: Vec<(&str, f64)> = file_symbols
        .keys()
        .map(|&f| {
            let score = scores.get(f).copied().unwrap_or(0.001);
            (f, score)
        })
        .collect();
    ranked_files.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Sort symbols within each file by line number
    for syms in file_symbols.values_mut() {
        syms.sort_by_key(|s| s.line);
    }

    // Binary search for the right tier split to fit the budget
    // Strategy: top files get Tier 1 (full signatures), rest get Tier 0 (names only)
    let mut output = String::new();
    let mut used_tokens = 0;

    // Estimate: header per file ~5 tokens, Tier 1 sig ~10 tokens, Tier 0 name ~2 tokens
    let tier1_count = find_tier1_cutoff(&ranked_files, &file_symbols, token_budget);

    for (i, (file, _score)) in ranked_files.iter().enumerate() {
        let symbols = match file_symbols.get(file) {
            Some(s) => s,
            None => continue,
        };

        // Filter out impl blocks for cleaner output
        let display_symbols: Vec<&&Symbol> = symbols
            .iter()
            .filter(|s| s.kind != "impl")
            .collect();

        if display_symbols.is_empty() {
            continue;
        }

        let is_tier1 = i < tier1_count;

        // File header
        let header = if is_tier1 {
            format!("{file}:\n")
        } else {
            format!("{file}: (names)\n")
        };
        let header_tokens = estimate_tokens(&header);

        if used_tokens + header_tokens > token_budget {
            break;
        }

        output.push_str(&header);
        used_tokens += header_tokens;

        if is_tier1 {
            // Tier 1: full signatures
            for sym in &display_symbols {
                let line = format!("│ {}\n", sym.signature);
                let line_tokens = estimate_tokens(&line);
                if used_tokens + line_tokens > token_budget {
                    output.push_str("│ ...\n");
                    used_tokens += 2;
                    break;
                }
                output.push_str(&line);
                used_tokens += line_tokens;
            }
        } else {
            // Tier 0: names only, comma-separated
            let names: Vec<&str> = display_symbols
                .iter()
                .map(|s| s.name.as_str())
                .collect();
            let names_line = format!("│ {}\n", names.join(", "));
            let names_tokens = estimate_tokens(&names_line);
            if used_tokens + names_tokens > token_budget {
                break;
            }
            output.push_str(&names_line);
            used_tokens += names_tokens;
        }

        output.push('\n');
        used_tokens += 1;
    }

    output
}

/// Find the cutoff index: files before this index get Tier 1 (full signatures),
/// files after get Tier 0 (names only).
fn find_tier1_cutoff(
    ranked_files: &[(&str, f64)],
    file_symbols: &HashMap<&str, Vec<&Symbol>>,
    budget: usize,
) -> usize {
    let mut total = 0;

    for (i, (file, _)) in ranked_files.iter().enumerate() {
        let symbols = match file_symbols.get(file) {
            Some(s) => s,
            None => continue,
        };

        let non_impl: Vec<&&Symbol> = symbols.iter().filter(|s| s.kind != "impl").collect();

        // Estimate tokens for this file at Tier 1
        let file_header = 5; // "file.rs:\n"
        let sig_tokens: usize = non_impl
            .iter()
            .map(|s| estimate_tokens(&s.signature) + 2) // "│ " prefix + newline
            .sum();

        let tier1_cost = file_header + sig_tokens;

        // Estimate at Tier 0 (names only)
        let _tier0_cost = file_header + non_impl.len() * 2 + 3;

        // If adding this file at Tier 1 would bust the budget for remaining files
        // at Tier 0, stop promoting to Tier 1
        let remaining_tier0: usize = ranked_files
            .iter()
            .skip(i + 1)
            .filter_map(|(f, _)| {
                file_symbols.get(f).map(|s| {
                    let count = s.iter().filter(|sym| sym.kind != "impl").count();
                    5 + count * 2 + 3
                })
            })
            .sum();

        if total + tier1_cost + remaining_tier0 > budget {
            return i;
        }

        total += tier1_cost;
    }

    ranked_files.len()
}

/// Rough token estimate.
fn estimate_tokens(text: &str) -> usize {
    (text.len() / 4).max(1)
}
