//! Dependency graph with PageRank scoring.
//!
//! Builds a directed graph where:
//! - Nodes are files
//! - Edges represent cross-file symbol references (file A uses symbol defined in file B)
//!
//! PageRank scores rank files by "importance" — files that are depended on
//! by many others rank higher. Task-aware personalization boosts files
//! relevant to the current task.

use std::collections::HashMap;
use std::path::Path;

use petgraph::graph::{DiGraph, NodeIndex};
use serde::{Deserialize, Serialize};

use super::{ProjectIndex, Symbol};

/// Persisted graph data: adjacency list + PageRank scores.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DependencyGraph {
    /// File → list of files it depends on (imports/references symbols from)
    pub edges: HashMap<String, Vec<String>>,
    /// File → base PageRank score (before task personalization)
    pub scores: HashMap<String, f64>,
}

impl DependencyGraph {
    /// Build the dependency graph from a project index.
    ///
    /// Strategy: for each symbol reference found in a file, if that symbol
    /// is defined in a different file, add an edge from the referencing file
    /// to the defining file.
    pub fn build(index: &ProjectIndex) -> Self {
        // Build a map: symbol_name → defining file(s)
        let mut symbol_to_files: HashMap<&str, Vec<&str>> = HashMap::new();
        for (name, symbols) in &index.symbols {
            for sym in symbols {
                symbol_to_files
                    .entry(name.as_str())
                    .or_default()
                    .push(sym.file.as_str());
            }
        }

        // Collect all source files that have symbols
        let source_files: Vec<&str> = index
            .summaries
            .keys()
            .map(|s| s.as_str())
            .collect();

        // Build petgraph for PageRank computation
        let mut graph = DiGraph::<&str, ()>::new();
        let mut node_map: HashMap<&str, NodeIndex> = HashMap::new();

        // Add all files as nodes
        for file in &source_files {
            let idx = graph.add_node(file);
            node_map.insert(file, idx);
        }

        // Build edges from cross-file references
        let mut edges: HashMap<String, Vec<String>> = HashMap::new();

        // Prefer tree-sitter references (precise) when available
        if !index.references.is_empty() {
            // Tree-sitter path: use actual references extracted from AST
            for (ref_file, ref_names) in &index.references {
                for ref_name in ref_names {
                    // Find which file(s) define this symbol
                    if let Some(def_files) = symbol_to_files.get(ref_name.as_str()) {
                        for &def_file in def_files {
                            if def_file != ref_file.as_str() {
                                edges
                                    .entry(ref_file.clone())
                                    .or_default()
                                    .push(def_file.to_string());

                                if let (Some(&from), Some(&to)) = (
                                    node_map.get(ref_file.as_str()),
                                    node_map.get(def_file),
                                ) {
                                    graph.add_edge(from, to, ());
                                }
                            }
                        }
                    }
                }
            }
        } else {
            // Regex fallback: use signature-substring matching (heuristic)
            for (file, _summary) in &index.summaries {
                let file_symbols: Vec<&Symbol> = index
                    .symbols
                    .values()
                    .flatten()
                    .filter(|s| s.file == *file)
                    .collect();

                for (other_file, _) in &index.summaries {
                    if other_file == file {
                        continue;
                    }

                    let other_symbols: Vec<&Symbol> = index
                        .symbols
                        .values()
                        .flatten()
                        .filter(|s| s.file == *other_file)
                        .collect();

                    for my_sym in &file_symbols {
                        for other_sym in &other_symbols {
                            if other_sym.signature.contains(&my_sym.name) && my_sym.name.len() > 2
                            {
                                edges
                                    .entry(other_file.clone())
                                    .or_default()
                                    .push(file.clone());

                                if let (Some(&from), Some(&to)) = (
                                    node_map.get(other_file.as_str()),
                                    node_map.get(file.as_str()),
                                ) {
                                    graph.add_edge(from, to, ());
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }

        // Deduplicate edges
        for deps in edges.values_mut() {
            deps.sort();
            deps.dedup();
        }

        // Compute PageRank
        let scores = pagerank(&graph, &node_map, 0.85, 20);

        DependencyGraph { edges, scores }
    }

    /// Get PageRank scores personalized for a task.
    ///
    /// Boosts files that match any of the given keywords (from the user's
    /// message, scratchpad, or recent edits).
    pub fn personalized_scores(&self, keywords: &[&str]) -> HashMap<String, f64> {
        let mut scores = self.scores.clone();

        if keywords.is_empty() {
            return scores;
        }

        // Boost files whose paths or associated symbols match keywords
        for (file, score) in scores.iter_mut() {
            let file_lower = file.to_lowercase();
            for kw in keywords {
                let kw_lower = kw.to_lowercase();
                if kw_lower.len() < 3 {
                    continue;
                }
                if file_lower.contains(&kw_lower) {
                    *score *= 3.0; // 3x boost for path match
                }
            }
        }

        // Also boost files that are depended on by boosted files (1-hop)
        let boosted: Vec<(String, f64)> = scores
            .iter()
            .filter(|(_, s)| **s > 0.0)
            .map(|(f, s)| (f.clone(), *s))
            .collect();

        for (file, boost) in &boosted {
            if let Some(deps) = self.edges.get(file) {
                for dep in deps {
                    if let Some(dep_score) = scores.get_mut(dep) {
                        *dep_score += boost * 0.3; // 30% reflected boost
                    }
                }
            }
        }

        scores
    }

    /// Save the graph to `.miniswe/index/graph.json`.
    pub fn save(&self, miniswe_dir: &Path) -> anyhow::Result<()> {
        let path = miniswe_dir.join("index").join("graph.json");
        std::fs::write(path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }

    /// Load the graph from `.miniswe/index/graph.json`.
    pub fn load(miniswe_dir: &Path) -> anyhow::Result<Self> {
        let path = miniswe_dir.join("index").join("graph.json");
        if path.exists() {
            Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
        } else {
            Ok(Self::default())
        }
    }
}

/// Compute PageRank on a directed graph.
///
/// Simple iterative power method implementation.
/// - `damping`: probability of following a link (typically 0.85)
/// - `iterations`: number of power iterations
fn pagerank(
    graph: &DiGraph<&str, ()>,
    node_map: &HashMap<&str, NodeIndex>,
    damping: f64,
    iterations: usize,
) -> HashMap<String, f64> {
    let n = graph.node_count();
    if n == 0 {
        return HashMap::new();
    }

    let initial = 1.0 / n as f64;
    let mut ranks: Vec<f64> = vec![initial; n];
    let mut new_ranks: Vec<f64> = vec![0.0; n];

    // Build reverse adjacency: for each node, who points to it?
    let mut incoming: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    for edge in graph.edge_indices() {
        let Some((src, dst)) = graph.edge_endpoints(edge) else {
            continue;
        };
        let out_degree = graph
            .neighbors_directed(src, petgraph::Direction::Outgoing)
            .count();
        incoming[dst.index()].push((src.index(), out_degree));
    }

    for _ in 0..iterations {
        let teleport = (1.0 - damping) / n as f64;

        for i in 0..n {
            let mut sum = 0.0;
            for &(src, out_degree) in &incoming[i] {
                if out_degree > 0 {
                    sum += ranks[src] / out_degree as f64;
                }
            }
            new_ranks[i] = teleport + damping * sum;
        }

        std::mem::swap(&mut ranks, &mut new_ranks);
    }

    // Map back to file names
    let mut scores = HashMap::new();
    for (&file, &idx) in node_map {
        scores.insert(file.to_string(), ranks[idx.index()]);
    }

    scores
}

/// Populate `deps` fields on symbols by scanning for cross-references.
///
/// For each symbol, check if its signature contains the names of other symbols.
/// This is a heuristic — tree-sitter will make this precise in Phase 2.
pub fn populate_symbol_deps(index: &mut ProjectIndex) {
    // Collect all symbol names
    let all_names: Vec<String> = index.symbols.keys().cloned().collect();

    // For each symbol, check if its signature references other symbols
    let mut updates: Vec<(String, Vec<String>)> = Vec::new();

    for (name, symbols) in &index.symbols {
        for sym in symbols {
            let mut deps = Vec::new();
            for other_name in &all_names {
                if other_name == name || other_name.len() < 3 {
                    continue;
                }
                if sym.signature.contains(other_name.as_str()) {
                    deps.push(other_name.clone());
                }
            }
            if !deps.is_empty() {
                updates.push((format!("{}:{}", sym.file, sym.line), deps));
            }
        }
    }

    // Apply updates
    for (key, deps) in updates {
        for symbols in index.symbols.values_mut() {
            for sym in symbols.iter_mut() {
                let sym_key = format!("{}:{}", sym.file, sym.line);
                if sym_key == key {
                    sym.deps = deps.clone();
                }
            }
        }
    }
}
