//! Per-file revision store for fast-mode edits.
//!
//! Semantics (see `docs/fast-mode-design.md`):
//! - Revisions numbered `rev_0`, `rev_1`, … per file.
//! - `rev_0` is the pristine state (first time the agent touches the file).
//! - Linear history: reverting to `rev_N` truncates `rev_{N+1}..`.
//! - Per-file cap (default 20). When exceeded, drop oldest *but never rev_0*.
//! - Only *successful* writes create revisions.
//!
//! V1 storage is in-memory and session-scoped: revisions live in a
//! `Mutex<HashMap>` and disappear when the process exits. `ensure_pristine`
//! takes the content the agent sees *now* as rev_0, which means a fresh
//! session treats whatever is on disk as the new pristine state. On-disk
//! durability is a possible follow-up.

use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

/// Default per-file revision cap. rev_0 never counts toward this.
pub const DEFAULT_CAP: usize = 20;

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
    /// File content at this revision (retained in-memory for `revert`).
    pub(super) content: String,
}

/// Per-session revision store. Thread-safe via an internal `Mutex` so it
/// can be shared across async tool invocations without `&mut` threading.
pub struct RevisionStore {
    inner: Mutex<Inner>,
}

struct Inner {
    per_file: HashMap<String, Vec<Revision>>,
    cap: usize,
}

impl RevisionStore {
    /// Create a new store. `_miniswe_dir` is accepted so on-disk storage can
    /// be added later without changing call sites; v1 ignores it.
    pub fn new(_miniswe_dir: &Path) -> Result<Self> {
        Ok(Self {
            inner: Mutex::new(Inner {
                per_file: HashMap::new(),
                cap: DEFAULT_CAP,
            }),
        })
    }

    /// Create a store with an explicit cap. Used by tests to exercise
    /// eviction without having to record 20+ revs.
    pub fn with_cap(cap: usize) -> Self {
        Self {
            inner: Mutex::new(Inner {
                per_file: HashMap::new(),
                cap,
            }),
        }
    }

    /// Record the pristine state of `rel_path` as `rev_0`. Idempotent: the
    /// first call wins, subsequent calls for the same path are no-ops.
    pub fn ensure_pristine(&self, rel_path: &str, content: &str) -> Result<()> {
        let mut inner = self.inner.lock().expect("revision store mutex poisoned");
        inner
            .per_file
            .entry(rel_path.to_string())
            .or_insert_with(|| {
                vec![Revision {
                    number: 0,
                    label: "initial".into(),
                    added: 0,
                    removed: 0,
                    ast_ok: true,
                    file_errors: 0,
                    project_errors: 0,
                    content: content.to_string(),
                }]
            });
        Ok(())
    }

    /// Record a new revision for `rel_path`. Returns the rev number
    /// assigned (monotonic — the new highest number, starting from 1 after
    /// rev_0). Errors if `ensure_pristine` was never called for this file.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &self,
        rel_path: &str,
        new_content: &str,
        label: &str,
        added: usize,
        removed: usize,
        ast_ok: bool,
        file_errors: usize,
        project_errors: usize,
    ) -> Result<usize> {
        let mut inner = self.inner.lock().expect("revision store mutex poisoned");
        let cap = inner.cap;
        let revs = inner
            .per_file
            .get_mut(rel_path)
            .ok_or_else(|| anyhow!("no pristine baseline recorded for {rel_path}"))?;

        let number = revs.last().map(|r| r.number + 1).unwrap_or(1);
        revs.push(Revision {
            number,
            label: label.to_string(),
            added,
            removed,
            ast_ok,
            file_errors,
            project_errors,
            content: new_content.to_string(),
        });

        // Evict oldest non-rev_0 if over cap. cap counts non-rev_0 entries.
        while revs.len() > cap + 1 {
            // index 0 is rev_0; drop index 1 (oldest non-pristine).
            revs.remove(1);
        }

        Ok(number)
    }

    /// Return the stored content for `rel_path` at `rev`. Errors if the
    /// file is unknown or the revision was truncated / evicted.
    pub fn read_content(&self, rel_path: &str, rev: usize) -> Result<String> {
        let inner = self.inner.lock().expect("revision store mutex poisoned");
        let revs = inner
            .per_file
            .get(rel_path)
            .ok_or_else(|| anyhow!("no revisions recorded for {rel_path}"))?;
        revs.iter()
            .find(|r| r.number == rev)
            .map(|r| r.content.clone())
            .ok_or_else(|| anyhow!("rev_{rev} not found for {rel_path} (available: {})",
                revs.iter().map(|r| format!("rev_{}", r.number)).collect::<Vec<_>>().join(", ")))
    }

    /// Truncate history for `rel_path` to `rev` inclusive. Used after a
    /// successful `revert` — the next edit becomes `rev+1`.
    pub fn truncate_to(&self, rel_path: &str, rev: usize) -> Result<()> {
        let mut inner = self.inner.lock().expect("revision store mutex poisoned");
        let revs = inner
            .per_file
            .get_mut(rel_path)
            .ok_or_else(|| anyhow!("no revisions recorded for {rel_path}"))?;
        if !revs.iter().any(|r| r.number == rev) {
            return Err(anyhow!("rev_{rev} not found for {rel_path}"));
        }
        revs.retain(|r| r.number <= rev);
        Ok(())
    }

    /// Return the revision list for `rel_path` in order. Empty vec if the
    /// file hasn't been touched by the agent yet.
    pub fn list(&self, rel_path: &str) -> Vec<Revision> {
        let inner = self.inner.lock().expect("revision store mutex poisoned");
        inner
            .per_file
            .get(rel_path)
            .cloned()
            .unwrap_or_default()
    }

    /// Highest revision number for `rel_path`, or `None` if unknown.
    pub fn current(&self, rel_path: &str) -> Option<usize> {
        let inner = self.inner.lock().expect("revision store mutex poisoned");
        inner
            .per_file
            .get(rel_path)
            .and_then(|revs| revs.last().map(|r| r.number))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> RevisionStore {
        RevisionStore::new(Path::new("/tmp/_unused")).unwrap()
    }

    #[test]
    fn pristine_is_idempotent_and_sets_rev_0() {
        let s = store();
        s.ensure_pristine("a.rs", "v0").unwrap();
        s.ensure_pristine("a.rs", "SHOULD_BE_IGNORED").unwrap();
        assert_eq!(s.current("a.rs"), Some(0));
        assert_eq!(s.read_content("a.rs", 0).unwrap(), "v0");
    }

    #[test]
    fn record_assigns_monotonic_numbers_starting_at_1() {
        let s = store();
        s.ensure_pristine("a.rs", "v0").unwrap();
        let n1 = s.record("a.rs", "v1", "insert_at L1 (+1 -0)", 1, 0, true, 0, 0).unwrap();
        let n2 = s.record("a.rs", "v2", "replace_range L1 (+1 -1)", 1, 1, true, 0, 0).unwrap();
        assert_eq!((n1, n2), (1, 2));
        assert_eq!(s.current("a.rs"), Some(2));
    }

    #[test]
    fn record_without_pristine_is_an_error() {
        let s = store();
        let err = s.record("a.rs", "v1", "replace_range", 0, 0, true, 0, 0).unwrap_err();
        assert!(err.to_string().contains("pristine"));
    }

    #[test]
    fn cap_evicts_oldest_non_pristine() {
        // cap=2 → rev_0 + 2 non-pristine max. rev_0 is never dropped.
        let s = RevisionStore::with_cap(2);
        s.ensure_pristine("a.rs", "v0").unwrap();
        s.record("a.rs", "v1", "rev1", 0, 0, true, 0, 0).unwrap();
        s.record("a.rs", "v2", "rev2", 0, 0, true, 0, 0).unwrap();
        s.record("a.rs", "v3", "rev3", 0, 0, true, 0, 0).unwrap();

        let revs: Vec<usize> = s.list("a.rs").iter().map(|r| r.number).collect();
        assert_eq!(revs, vec![0, 2, 3]); // rev_0 preserved, rev_1 evicted
        assert!(s.read_content("a.rs", 1).is_err());
        assert_eq!(s.read_content("a.rs", 0).unwrap(), "v0");
    }

    #[test]
    fn truncate_drops_later_revs_keeps_target() {
        let s = store();
        s.ensure_pristine("a.rs", "v0").unwrap();
        s.record("a.rs", "v1", "r1", 0, 0, true, 0, 0).unwrap();
        s.record("a.rs", "v2", "r2", 0, 0, true, 0, 0).unwrap();
        s.record("a.rs", "v3", "r3", 0, 0, true, 0, 0).unwrap();

        s.truncate_to("a.rs", 1).unwrap();
        let revs: Vec<usize> = s.list("a.rs").iter().map(|r| r.number).collect();
        assert_eq!(revs, vec![0, 1]);
        assert!(s.read_content("a.rs", 2).is_err());
    }

    #[test]
    fn truncate_unknown_rev_errors() {
        let s = store();
        s.ensure_pristine("a.rs", "v0").unwrap();
        let err = s.truncate_to("a.rs", 99).unwrap_err();
        assert!(err.to_string().contains("rev_99"));
    }

    #[test]
    fn per_file_isolation() {
        let s = store();
        s.ensure_pristine("a.rs", "a_v0").unwrap();
        s.ensure_pristine("b.rs", "b_v0").unwrap();
        s.record("a.rs", "a_v1", "r1", 0, 0, true, 0, 0).unwrap();

        assert_eq!(s.current("a.rs"), Some(1));
        assert_eq!(s.current("b.rs"), Some(0));
        assert_eq!(s.read_content("b.rs", 0).unwrap(), "b_v0");
    }

    #[test]
    fn current_none_for_untouched_file() {
        let s = store();
        assert_eq!(s.current("untouched.rs"), None);
        assert!(s.list("untouched.rs").is_empty());
    }
}
