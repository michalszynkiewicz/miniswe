//! Per-file revision store for fast-mode edits.
//!
//! Semantics (see `docs/fast-mode-design.md`):
//! - Revisions numbered `rev_0`, `rev_1`, … per file.
//! - `rev_0` is the pristine state (first time the agent touches the file).
//! - Linear history: reverting to `rev_N` truncates `rev_{N+1}..`.
//! - Per-file cap (default 20). When exceeded, drop oldest *but never rev_0*.
//! - Only *successful* writes create revisions.
//!
//! Storage layout (proposed): `.miniswe/revisions/<file-hash>/rev_<n>.txt`.

use anyhow::Result;
use std::path::Path;
use std::sync::Mutex;

/// One entry in a file's revision table. Produced by every successful
/// write-ish tool and rendered into the per-edit feedback.
#[derive(Debug, Clone)]
pub struct Revision {
    /// 0 for pristine, 1+ for subsequent edits.
    pub number: usize,
    /// Human-readable summary of the tool call that produced this rev.
    /// e.g. `"replace_range L42 (+1 -1)"`, `"initial"`.
    pub label: String,
    /// Lines added in this revision.
    pub added: usize,
    /// Lines removed in this revision.
    pub removed: usize,
    /// AST parse status at this revision.
    pub ast_ok: bool,
    /// LSP error count in this file at this revision.
    pub file_errors: usize,
    /// Project-wide LSP error count at this revision.
    pub project_errors: usize,
}

/// Per-session revision store. Thread-safe via an internal `Mutex` so it
/// can be shared across async tool invocations without `&mut` threading.
pub struct RevisionStore {
    inner: Mutex<Inner>,
}

struct Inner {
    /// Root under `.miniswe/revisions/` for on-disk snapshots.
    _storage_root: std::path::PathBuf,
    /// Per-file revision metadata (file path → revs). On-disk bytes live
    /// separately under `_storage_root`.
    _per_file: std::collections::HashMap<String, Vec<Revision>>,
    /// Cap per file (excluding rev_0, which is never dropped).
    _cap: usize,
}

impl RevisionStore {
    /// Create a new store rooted at `.miniswe/revisions/`. Creates the
    /// directory if absent. Default cap is 20 revs per file.
    pub fn new(_miniswe_dir: &Path) -> Result<Self> {
        todo!("RevisionStore::new — create .miniswe/revisions/, cap=20")
    }

    /// Record the pristine state of `path` as `rev_0`. Idempotent: calling
    /// twice for the same file is a no-op after the first success.
    pub fn ensure_pristine(&self, _rel_path: &str, _content: &str) -> Result<()> {
        todo!("ensure_pristine — write rev_0 if missing")
    }

    /// Record a new revision for `rel_path` with the given label and stats.
    /// Returns the rev number assigned (monotonic, starts at 1 after rev_0).
    pub fn record(
        &self,
        _rel_path: &str,
        _new_content: &str,
        _label: &str,
        _added: usize,
        _removed: usize,
        _ast_ok: bool,
        _file_errors: usize,
        _project_errors: usize,
    ) -> Result<usize> {
        todo!("record — append revision, evict oldest non-rev_0 if over cap")
    }

    /// Return the stored bytes for `rel_path` at `rev`. Returns an error if
    /// `rev` doesn't exist (already truncated or never created).
    pub fn read_content(&self, _rel_path: &str, _rev: usize) -> Result<String> {
        todo!("read_content — load rev bytes from disk")
    }

    /// Truncate history for `rel_path` to `rev` inclusive. Called after a
    /// successful `revert` so the next edit becomes `rev+1`.
    pub fn truncate_to(&self, _rel_path: &str, _rev: usize) -> Result<()> {
        todo!("truncate_to — drop rev+1..end both in-memory and on disk")
    }

    /// List current revision metadata for `rel_path` (rendered into the
    /// per-edit feedback table).
    pub fn list(&self, _rel_path: &str) -> Vec<Revision> {
        todo!("list — return in-order rev metadata")
    }

    /// Current (highest) revision number for `rel_path`, or `None` if the
    /// agent hasn't touched this file yet.
    pub fn current(&self, _rel_path: &str) -> Option<usize> {
        todo!("current — highest rev number for file")
    }
}
