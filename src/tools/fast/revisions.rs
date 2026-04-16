//! Per-file revision store for fast-mode edits.
//!
//! Semantics (see `docs/fast-mode-design.md`):
//! - Revisions numbered `rev_0`, `rev_1`, … per file.
//! - `rev_0` is the pristine state (first time the agent touches the file).
//! - Linear history: reverting to `rev_N` does NOT drop `rev_{N+1}..` — it
//!   marks them as tombstones (`reverted=true`). They stay visible in the
//!   table so the model can see "I already tried this and undid it" and
//!   avoid byte-identical replays.
//! - Numbering is strictly monotonic across the full history (live +
//!   tombstones): next edit is `max(any rev) + 1`, never a recycled number.
//! - Caps are applied separately:
//!     * live chain (non-reverted, non-rev_0): `DEFAULT_LIVE_CAP` (20)
//!     * tombstones: `DEFAULT_TOMBSTONE_CAP` (20), oldest evicted first
//!     * `rev_0` never counts toward either cap and is never dropped.
//! - Only *successful* writes create revisions.
//!
//! V1 storage is in-memory and session-scoped: revisions live in a
//! `Mutex<HashMap>` and disappear when the process exits. `ensure_pristine`
//! takes the content the agent sees *now* as rev_0, which means a fresh
//! session treats whatever is on disk as the new pristine state. On-disk
//! durability is a possible follow-up.

use anyhow::{Result, anyhow};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::path::Path;

/// Default cap for live (non-reverted, non-rev_0) entries per file.
pub const DEFAULT_LIVE_CAP: usize = 20;

/// Default cap for tombstoned (reverted) entries per file.
pub const DEFAULT_TOMBSTONE_CAP: usize = 20;

/// Sanity cap on stored payload size per revision. Pathological edits
/// (e.g. a 100 KB `replace_range`) get truncated at store time with a
/// marker so `show_rev` stays bounded. Typical payloads are a few
/// hundred bytes; this is a safety net, not a common path.
pub const MAX_STORED_PAYLOAD_BYTES: usize = 16 * 1024;

/// One entry in a file's revision table. Produced by every successful
/// write-ish tool and rendered into the per-edit feedback.
#[derive(Debug, Clone)]
pub struct Revision {
    /// 0 for pristine, 1+ for subsequent edits. Strictly monotonic —
    /// reverting then replaying picks up where numbering left off.
    pub number: usize,
    /// The tool that produced this rev: `"initial"`, `"replace_range"`,
    /// `"insert_at"`, `"write_file"`.
    pub operation: String,
    /// Human-readable short summary used in the one-line table row.
    /// e.g. `"replace_range L418-419"`, `"insert_at after L12"`,
    /// `"initial"`.
    pub label: String,
    /// For `replace_range`: `(start, end)` in 1-based line numbers.
    /// `None` for other operations.
    pub range: Option<(usize, usize)>,
    /// The verbatim text the tool wrote (new_text / inserted content).
    /// `None` for `initial` and `write_file` (for write_file the full
    /// content is already in `content`, no need to duplicate).
    pub payload: Option<String>,
    /// Lines added in this revision.
    pub added: usize,
    /// Lines removed in this revision.
    pub removed: usize,
    /// AST parse status at this revision.
    pub ast_ok: bool,
    /// Short error summary (first parse error) when `ast_ok=false`.
    /// e.g. `"L441:1: syntax error"`. `None` when `ast_ok=true`.
    pub ast_error: Option<String>,
    /// LSP error count in this file at this revision.
    pub file_errors: usize,
    /// Project-wide LSP error count at this revision.
    pub project_errors: usize,
    /// Tombstone flag. Set by `mark_reverted_to`. Read-only: cannot be
    /// the target of another `revert`, does not count as "current".
    pub reverted: bool,
    /// File content at this revision (retained in-memory for `revert`).
    pub(super) content: String,
}

/// Arguments for [`RevisionStore::record`]. Grouped into a struct so the
/// call sites stay readable as fields are added.
pub struct RecordArgs<'a> {
    pub operation: &'a str,
    pub label: &'a str,
    pub range: Option<(usize, usize)>,
    pub payload: Option<String>,
    pub added: usize,
    pub removed: usize,
    pub ast_ok: bool,
    pub ast_error: Option<String>,
    pub file_errors: usize,
    pub project_errors: usize,
}

/// Per-session revision store. Thread-safe via an internal `Mutex` so it
/// can be shared across async tool invocations without `&mut` threading.
pub struct RevisionStore {
    inner: Mutex<Inner>,
}

struct Inner {
    per_file: HashMap<String, Vec<Revision>>,
    live_cap: usize,
    tombstone_cap: usize,
}

impl RevisionStore {
    /// Create a new store. `_miniswe_dir` is accepted so on-disk storage can
    /// be added later without changing call sites; v1 ignores it.
    pub fn new(_miniswe_dir: &Path) -> Result<Self> {
        Ok(Self {
            inner: Mutex::new(Inner {
                per_file: HashMap::new(),
                live_cap: DEFAULT_LIVE_CAP,
                tombstone_cap: DEFAULT_TOMBSTONE_CAP,
            }),
        })
    }

    /// Create a store with explicit caps. Used by tests to exercise
    /// eviction without having to record dozens of revs.
    pub fn with_caps(live_cap: usize, tombstone_cap: usize) -> Self {
        Self {
            inner: Mutex::new(Inner {
                per_file: HashMap::new(),
                live_cap,
                tombstone_cap,
            }),
        }
    }

    /// Convenience: store with both caps set to `cap`. Preserved for tests
    /// written against the earlier single-cap API.
    pub fn with_cap(cap: usize) -> Self {
        Self::with_caps(cap, cap)
    }

    /// Record the pristine state of `rel_path` as `rev_0`. Idempotent: the
    /// first call wins, subsequent calls for the same path are no-ops.
    pub fn ensure_pristine(&self, rel_path: &str, content: &str) -> Result<()> {
        let mut inner = self.inner.lock();
        inner
            .per_file
            .entry(rel_path.to_string())
            .or_insert_with(|| {
                vec![Revision {
                    number: 0,
                    operation: "initial".into(),
                    label: "initial".into(),
                    range: None,
                    payload: None,
                    added: 0,
                    removed: 0,
                    ast_ok: true,
                    ast_error: None,
                    file_errors: 0,
                    project_errors: 0,
                    reverted: false,
                    content: content.to_string(),
                }]
            });
        Ok(())
    }

    /// Record a new revision for `rel_path`. Returns the rev number
    /// assigned — strictly monotonic across the full history including
    /// tombstones. Errors if `ensure_pristine` was never called.
    pub fn record(&self, rel_path: &str, new_content: &str, args: RecordArgs<'_>) -> Result<usize> {
        let mut inner = self.inner.lock();
        let live_cap = inner.live_cap;
        let revs = inner
            .per_file
            .get_mut(rel_path)
            .ok_or_else(|| anyhow!("no pristine baseline recorded for {rel_path}"))?;

        // Monotonic: next number is max-so-far + 1. Tombstones count too,
        // so a reverted rev_13 will NOT be re-assigned to a new edit.
        let number = revs.iter().map(|r| r.number).max().unwrap_or(0) + 1;

        let payload = args.payload.map(cap_payload);
        revs.push(Revision {
            number,
            operation: args.operation.to_string(),
            label: args.label.to_string(),
            range: args.range,
            payload,
            added: args.added,
            removed: args.removed,
            ast_ok: args.ast_ok,
            ast_error: args.ast_error,
            file_errors: args.file_errors,
            project_errors: args.project_errors,
            reverted: false,
            content: new_content.to_string(),
        });

        evict_live_overflow(revs, live_cap);
        Ok(number)
    }

    /// Return the stored content for `rel_path` at `rev`. Errors if the
    /// file is unknown, the revision was evicted, or the revision is a
    /// tombstone (reverted). Tombstones are advisory-only; reverting back
    /// to them is not supported.
    pub fn read_content(&self, rel_path: &str, rev: usize) -> Result<String> {
        let inner = self.inner.lock();
        let revs = inner
            .per_file
            .get(rel_path)
            .ok_or_else(|| anyhow!("no revisions recorded for {rel_path}"))?;
        let found = revs.iter().find(|r| r.number == rev).ok_or_else(|| {
            anyhow!(
                "rev_{rev} not found for {rel_path} (available: {})",
                revs.iter()
                    .map(|r| format!("rev_{}", r.number))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
        if found.reverted {
            return Err(anyhow!(
                "rev_{rev} is a tombstone (previously reverted) and cannot be the \
                 target of revert. Use show_rev to inspect it, or pick a live rev."
            ));
        }
        Ok(found.content.clone())
    }

    /// Return full revision metadata for inspection (including tombstones
    /// and payload). Used by the `show_rev` tool. `None` if the file or
    /// rev is unknown.
    pub fn get(&self, rel_path: &str, rev: usize) -> Option<Revision> {
        let inner = self.inner.lock();
        inner
            .per_file
            .get(rel_path)
            .and_then(|revs| revs.iter().find(|r| r.number == rev).cloned())
    }

    /// Mark all revs strictly greater than `rev` as tombstones for
    /// `rel_path`. Called after a successful `revert`. The target rev
    /// itself stays live. Applies the tombstone cap after marking.
    pub fn mark_reverted_to(&self, rel_path: &str, rev: usize) -> Result<()> {
        let mut inner = self.inner.lock();
        let tombstone_cap = inner.tombstone_cap;
        let revs = inner
            .per_file
            .get_mut(rel_path)
            .ok_or_else(|| anyhow!("no revisions recorded for {rel_path}"))?;
        let target = revs
            .iter()
            .find(|r| r.number == rev)
            .ok_or_else(|| anyhow!("rev_{rev} not found for {rel_path}"))?;
        if target.reverted {
            return Err(anyhow!(
                "rev_{rev} is a tombstone (previously reverted); pick a live rev"
            ));
        }
        for r in revs.iter_mut() {
            if r.number > rev {
                r.reverted = true;
            }
        }
        evict_tombstone_overflow(revs, tombstone_cap);
        Ok(())
    }

    /// Return the revision list for `rel_path` in order. Empty vec if the
    /// file hasn't been touched by the agent yet. Includes tombstones.
    pub fn list(&self, rel_path: &str) -> Vec<Revision> {
        let inner = self.inner.lock();
        inner.per_file.get(rel_path).cloned().unwrap_or_default()
    }

    /// Highest *live* revision number for `rel_path`, or `None` if
    /// unknown. Tombstones are skipped.
    pub fn current(&self, rel_path: &str) -> Option<usize> {
        let inner = self.inner.lock();
        inner
            .per_file
            .get(rel_path)
            .and_then(|revs| revs.iter().rev().find(|r| !r.reverted).map(|r| r.number))
    }
}

/// Truncate a payload at `MAX_STORED_PAYLOAD_BYTES`. Appends a note so
/// `show_rev` still tells the model it was truncated. Byte-level cut
/// (respects UTF-8 boundary).
fn cap_payload(mut s: String) -> String {
    if s.len() <= MAX_STORED_PAYLOAD_BYTES {
        return s;
    }
    let original_len = s.len();
    // Walk back from the cap to the nearest char boundary.
    let mut cut = MAX_STORED_PAYLOAD_BYTES;
    while cut > 0 && !s.is_char_boundary(cut) {
        cut -= 1;
    }
    s.truncate(cut);
    s.push_str(&format!(
        "\n… (truncated at store time — original was {original_len} bytes)"
    ));
    s
}

/// Drop the oldest LIVE non-rev_0 entries until the count fits
/// `live_cap`. rev_0 and tombstones are skipped.
fn evict_live_overflow(revs: &mut Vec<Revision>, live_cap: usize) {
    loop {
        let live_count = revs.iter().filter(|r| !r.reverted && r.number != 0).count();
        if live_count <= live_cap {
            return;
        }
        // Find index of oldest non-rev_0 live entry and remove it.
        let Some(idx) = revs.iter().position(|r| !r.reverted && r.number != 0) else {
            return;
        };
        revs.remove(idx);
    }
}

/// Drop the oldest TOMBSTONE entries until the tombstone count fits
/// `tombstone_cap`. rev_0 is skipped; rev_0 is never a tombstone anyway.
fn evict_tombstone_overflow(revs: &mut Vec<Revision>, tombstone_cap: usize) {
    loop {
        let tomb_count = revs.iter().filter(|r| r.reverted).count();
        if tomb_count <= tombstone_cap {
            return;
        }
        let Some(idx) = revs.iter().position(|r| r.reverted) else {
            return;
        };
        revs.remove(idx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> RevisionStore {
        RevisionStore::new(Path::new("/tmp/_unused")).unwrap()
    }

    fn args<'a>(op: &'a str, label: &'a str) -> RecordArgs<'a> {
        RecordArgs {
            operation: op,
            label,
            range: None,
            payload: None,
            added: 0,
            removed: 0,
            ast_ok: true,
            ast_error: None,
            file_errors: 0,
            project_errors: 0,
        }
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
        let n1 = s
            .record("a.rs", "v1", args("insert_at", "insert_at L1"))
            .unwrap();
        let n2 = s
            .record("a.rs", "v2", args("replace_range", "replace_range L1-1"))
            .unwrap();
        assert_eq!((n1, n2), (1, 2));
        assert_eq!(s.current("a.rs"), Some(2));
    }

    #[test]
    fn record_without_pristine_is_an_error() {
        let s = store();
        let err = s
            .record("a.rs", "v1", args("replace_range", "rr"))
            .unwrap_err();
        assert!(err.to_string().contains("pristine"));
    }

    #[test]
    fn live_cap_evicts_oldest_non_pristine() {
        // live_cap=2 → rev_0 + 2 live non-pristine max. rev_0 is never dropped.
        let s = RevisionStore::with_caps(2, 20);
        s.ensure_pristine("a.rs", "v0").unwrap();
        s.record("a.rs", "v1", args("replace_range", "r1")).unwrap();
        s.record("a.rs", "v2", args("replace_range", "r2")).unwrap();
        s.record("a.rs", "v3", args("replace_range", "r3")).unwrap();

        let revs: Vec<usize> = s.list("a.rs").iter().map(|r| r.number).collect();
        assert_eq!(revs, vec![0, 2, 3]); // rev_0 preserved, rev_1 evicted
        assert!(s.read_content("a.rs", 1).is_err());
        assert_eq!(s.read_content("a.rs", 0).unwrap(), "v0");
    }

    #[test]
    fn mark_reverted_keeps_later_revs_as_tombstones() {
        let s = store();
        s.ensure_pristine("a.rs", "v0").unwrap();
        s.record("a.rs", "v1", args("replace_range", "r1")).unwrap();
        s.record("a.rs", "v2", args("replace_range", "r2")).unwrap();
        s.record("a.rs", "v3", args("replace_range", "r3")).unwrap();

        s.mark_reverted_to("a.rs", 1).unwrap();
        let rows: Vec<(usize, bool)> = s
            .list("a.rs")
            .iter()
            .map(|r| (r.number, r.reverted))
            .collect();
        assert_eq!(
            rows,
            vec![(0, false), (1, false), (2, true), (3, true)],
            "rev_2 and rev_3 should remain as tombstones"
        );
        // current() skips tombstones
        assert_eq!(s.current("a.rs"), Some(1));
        // read_content refuses tombstones
        assert!(s.read_content("a.rs", 2).is_err());
        // read_content on live target works
        assert_eq!(s.read_content("a.rs", 1).unwrap(), "v1");
    }

    #[test]
    fn next_record_after_revert_uses_monotonic_number() {
        let s = store();
        s.ensure_pristine("a.rs", "v0").unwrap();
        s.record("a.rs", "v1", args("replace_range", "r1")).unwrap();
        s.record("a.rs", "v2", args("replace_range", "r2")).unwrap();
        s.record("a.rs", "v3", args("replace_range", "r3")).unwrap();
        s.mark_reverted_to("a.rs", 1).unwrap();

        // Next record should be rev_4, NOT rev_2 (2 and 3 are tombstones)
        let n = s.record("a.rs", "v4", args("replace_range", "r4")).unwrap();
        assert_eq!(n, 4);
        assert_eq!(s.current("a.rs"), Some(4));

        let rows: Vec<(usize, bool)> = s
            .list("a.rs")
            .iter()
            .map(|r| (r.number, r.reverted))
            .collect();
        assert_eq!(
            rows,
            vec![(0, false), (1, false), (2, true), (3, true), (4, false)]
        );
    }

    #[test]
    fn mark_reverted_unknown_rev_errors() {
        let s = store();
        s.ensure_pristine("a.rs", "v0").unwrap();
        let err = s.mark_reverted_to("a.rs", 99).unwrap_err();
        assert!(err.to_string().contains("rev_99"));
    }

    #[test]
    fn cannot_revert_onto_tombstone() {
        let s = store();
        s.ensure_pristine("a.rs", "v0").unwrap();
        s.record("a.rs", "v1", args("replace_range", "r1")).unwrap();
        s.record("a.rs", "v2", args("replace_range", "r2")).unwrap();
        s.mark_reverted_to("a.rs", 1).unwrap(); // rev_2 is now a tombstone

        let err = s.mark_reverted_to("a.rs", 2).unwrap_err();
        assert!(
            err.to_string().contains("tombstone"),
            "error should mention tombstone: {err}"
        );
    }

    #[test]
    fn tombstone_cap_evicts_oldest() {
        // tombstone_cap=2
        let s = RevisionStore::with_caps(20, 2);
        s.ensure_pristine("a.rs", "v0").unwrap();
        for i in 0..5 {
            s.record("a.rs", &format!("v{i}"), args("replace_range", "r"))
                .unwrap();
        }
        // Revert to rev_0 → all 5 live entries become tombstones. Cap=2,
        // so only the 2 newest tombstones survive.
        s.mark_reverted_to("a.rs", 0).unwrap();
        let rows: Vec<usize> = s.list("a.rs").iter().map(|r| r.number).collect();
        assert_eq!(rows, vec![0, 4, 5]);
    }

    #[test]
    fn per_file_isolation() {
        let s = store();
        s.ensure_pristine("a.rs", "a_v0").unwrap();
        s.ensure_pristine("b.rs", "b_v0").unwrap();
        s.record("a.rs", "a_v1", args("replace_range", "r1"))
            .unwrap();

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

    #[test]
    fn payload_over_cap_is_truncated_with_marker() {
        let s = store();
        s.ensure_pristine("a.rs", "").unwrap();
        let big = "x".repeat(MAX_STORED_PAYLOAD_BYTES + 1000);
        let n = s
            .record(
                "a.rs",
                "",
                RecordArgs {
                    operation: "replace_range",
                    label: "r",
                    range: None,
                    payload: Some(big),
                    added: 0,
                    removed: 0,
                    ast_ok: true,
                    ast_error: None,
                    file_errors: 0,
                    project_errors: 0,
                },
            )
            .unwrap();
        let rev = s.get("a.rs", n).unwrap();
        let payload = rev.payload.unwrap();
        assert!(payload.len() < MAX_STORED_PAYLOAD_BYTES + 200);
        assert!(payload.contains("truncated at store time"));
    }

    #[test]
    fn get_returns_live_and_tombstoned_revs() {
        let s = store();
        s.ensure_pristine("a.rs", "v0").unwrap();
        s.record("a.rs", "v1", args("replace_range", "r1")).unwrap();
        s.record("a.rs", "v2", args("replace_range", "r2")).unwrap();
        s.mark_reverted_to("a.rs", 1).unwrap();

        let live = s.get("a.rs", 1).unwrap();
        assert!(!live.reverted);
        let tomb = s.get("a.rs", 2).unwrap();
        assert!(tomb.reverted);
    }
}
