//! Knowledge engine — offline indexing with tree-sitter and PageRank.
//!
//! Provides:
//! - Tree-sitter AST parsing for symbol extraction
//! - Dependency graph with PageRank scoring
//! - Pre-computed file summaries
//! - Project profile auto-generation
//!
//! This module is designed to be built incrementally. Phase 1 provides
//! basic file scanning and simple symbol extraction. Full tree-sitter
//! integration with PageRank comes in later phases.

pub mod graph;
pub mod indexer;
pub mod profile;
pub mod repo_map;

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// A symbol extracted from source code.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Symbol {
    /// Symbol name (e.g., "createSession")
    pub name: String,
    /// File path relative to project root
    pub file: String,
    /// Line number (1-indexed)
    pub line: usize,
    /// Symbol kind: function, struct, enum, trait, type, const, etc.
    pub kind: String,
    /// Signature (e.g., "pub fn createSession(user: User) -> Session")
    pub signature: String,
    /// Symbols this depends on
    pub deps: Vec<String>,
}

/// The project index — all extracted knowledge.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProjectIndex {
    /// All symbols indexed by name
    pub symbols: HashMap<String, Vec<Symbol>>,
    /// File summaries (one-line per file)
    pub summaries: HashMap<String, String>,
    /// File tree as a flat list of paths
    pub file_tree: Vec<String>,
    /// Total files indexed
    pub total_files: usize,
    /// Total symbols extracted
    pub total_symbols: usize,
}

impl ProjectIndex {
    /// Save the index to `.minime/index/`.
    pub fn save(&self, minime_dir: &Path) -> anyhow::Result<()> {
        let index_dir = minime_dir.join("index");
        std::fs::create_dir_all(&index_dir)?;

        let symbols_path = index_dir.join("symbols.json");
        let summaries_path = index_dir.join("summaries.json");
        let tree_path = index_dir.join("file_tree.txt");

        std::fs::write(symbols_path, serde_json::to_string_pretty(&self.symbols)?)?;
        std::fs::write(summaries_path, serde_json::to_string_pretty(&self.summaries)?)?;
        std::fs::write(tree_path, self.file_tree.join("\n"))?;

        Ok(())
    }

    /// Load the index from `.minime/index/`.
    pub fn load(minime_dir: &Path) -> anyhow::Result<Self> {
        let index_dir = minime_dir.join("index");

        let symbols: HashMap<String, Vec<Symbol>> = {
            let path = index_dir.join("symbols.json");
            if path.exists() {
                serde_json::from_str(&std::fs::read_to_string(path)?)?
            } else {
                HashMap::new()
            }
        };

        let summaries: HashMap<String, String> = {
            let path = index_dir.join("summaries.json");
            if path.exists() {
                serde_json::from_str(&std::fs::read_to_string(path)?)?
            } else {
                HashMap::new()
            }
        };

        let file_tree: Vec<String> = {
            let path = index_dir.join("file_tree.txt");
            if path.exists() {
                std::fs::read_to_string(path)?
                    .lines()
                    .map(|s| s.to_string())
                    .collect()
            } else {
                Vec::new()
            }
        };

        let total_symbols = symbols.values().map(|v| v.len()).sum();
        let total_files = file_tree.len();

        Ok(Self {
            symbols,
            summaries,
            file_tree,
            total_files,
            total_symbols,
        })
    }
}
