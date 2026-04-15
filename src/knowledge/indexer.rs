//! File indexer — scans the project and extracts symbols.
//!
//! Phase 1: regex-based symbol extraction as a bootstrap (see `lang/`).
//! Phase 2 (future): tree-sitter for precise AST-based extraction (see
//! `crate::knowledge::ts_extract`).

mod end_line;
mod lang;
mod summary;
mod walker;

pub use walker::{audit_file_sizes, index_project, reindex_file};
