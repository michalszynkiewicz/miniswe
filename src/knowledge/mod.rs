//! Knowledge engine — offline indexing with tree-sitter and PageRank.
//!
//! Provides:
//! - Tree-sitter AST parsing for symbol extraction
//! - Dependency graph with PageRank scoring
//! - Pre-computed file summaries
//! - Project profile auto-generation
//! - Incremental re-indexing via mtime tracking

pub mod graph;
pub mod indexer;
pub mod profile;
pub mod repo_map;
pub mod ts_extract;

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
    /// Start line number (1-indexed)
    pub line: usize,
    /// End line number (1-indexed, inclusive). 0 means unknown.
    #[serde(default)]
    pub end_line: usize,
    /// Symbol kind: function, struct, enum, trait, type, const, etc.
    pub kind: String,
    /// Signature (e.g., "pub fn createSession(user: User) -> Session")
    pub signature: String,
    /// Symbols this depends on
    pub deps: Vec<String>,
    /// For methods inside impl blocks: the impl header signature
    /// e.g., "impl<T: Clone> Service<Request> for Router<T>"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_impl: Option<String>,
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
    /// Per-file references: file → list of symbol names referenced in that file
    #[serde(default)]
    pub references: HashMap<String, Vec<String>>,
    /// Per-file modification times (seconds since epoch) for incremental reindex
    #[serde(default)]
    pub mtimes: HashMap<String, u64>,
    /// Total files indexed
    pub total_files: usize,
    /// Total symbols extracted
    pub total_symbols: usize,
}

impl ProjectIndex {
    /// Save the index to `.miniswe/index/`.
    pub fn save(&self, miniswe_dir: &Path) -> anyhow::Result<()> {
        let index_dir = miniswe_dir.join("index");
        std::fs::create_dir_all(&index_dir)?;

        std::fs::write(
            index_dir.join("symbols.json"),
            serde_json::to_string_pretty(&self.symbols)?,
        )?;
        std::fs::write(
            index_dir.join("summaries.json"),
            serde_json::to_string_pretty(&self.summaries)?,
        )?;
        std::fs::write(
            index_dir.join("file_tree.txt"),
            self.file_tree.join("\n"),
        )?;
        std::fs::write(
            index_dir.join("mtimes.json"),
            serde_json::to_string_pretty(&self.mtimes)?,
        )?;

        Ok(())
    }

    /// Load the index from `.miniswe/index/`.
    pub fn load(miniswe_dir: &Path) -> anyhow::Result<Self> {
        let index_dir = miniswe_dir.join("index");

        let symbols: HashMap<String, Vec<Symbol>> = load_json(&index_dir, "symbols.json")?;
        let summaries: HashMap<String, String> = load_json(&index_dir, "summaries.json")?;
        let mtimes: HashMap<String, u64> = load_json(&index_dir, "mtimes.json")?;

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
            references: HashMap::new(),
            mtimes,
            total_files,
            total_symbols,
        })
    }

    /// Look up a symbol by name. Returns all definitions across files.
    pub fn lookup(&self, name: &str) -> Vec<&Symbol> {
        self.symbols
            .get(name)
            .map(|syms| syms.iter().collect())
            .unwrap_or_default()
    }

    /// Check if a file has been modified since it was last indexed.
    pub fn is_stale(&self, file: &str, current_mtime: u64) -> bool {
        match self.mtimes.get(file) {
            Some(&indexed_mtime) => current_mtime > indexed_mtime,
            None => true, // not indexed yet
        }
    }
}

/// Helper to load a JSON file, returning Default if it doesn't exist.
fn load_json<T: serde::de::DeserializeOwned + Default>(
    dir: &Path,
    filename: &str,
) -> anyhow::Result<T> {
    let path = dir.join(filename);
    if path.exists() {
        Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
    } else {
        Ok(T::default())
    }
}
