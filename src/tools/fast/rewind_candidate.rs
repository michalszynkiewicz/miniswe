//! Mechanical detection of an abandoned-clean revision (experimental, opt-in
//! via `tools.debugger_judge_rewind`).
//!
//! `debugger_judge`'s SCRAP/CONTINUE choice is all-or-nothing: revert the
//! WHOLE tree, or revert nothing at all. But a stuck run often has just ONE
//! file that regressed from a clean-ish earlier revision into a much worse
//! one, while the rest of the tree is fine — observed on Gemma 4: a file
//! clean at rev_9 (0 errors) edited forward to rev_15 (129 errors) with 3
//! fresh-context debugger fires along the way, none of which considered
//! reverting just that file.
//!
//! A tier-1 replay probe found the judge basically never proposes this
//! itself, even when explicitly offered a third option and shown the
//! revision tables (0/24 hits, plus a rise in malformed responses) — but
//! given a MECHANICALLY computed candidate and a narrowed accept/reject
//! choice, the hit rate jumped to 13/24. This module is the mechanical half:
//! find the single best candidate, if any, so the judge only has to decide
//! whether to take it.

use super::revisions::RevisionStore;

/// Minimum error-count improvement (current vs candidate) required to
/// propose a rewind. Below this, forward-fixing is cheaper than reverting.
pub const REGRESSION_MARGIN: usize = 3;

/// Ceiling on the candidate's OWN error count — it must be near-clean
/// itself, not just "less bad than now".
pub const CANDIDATE_MAX_ERRORS: usize = 1;

/// A single-file rewind candidate: an earlier revision materially cleaner
/// than the file's current (live) state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RewindCandidate {
    pub path: String,
    pub rev: usize,
    pub file_errors_now: usize,
    pub file_errors_then: usize,
}

/// Scan every changed file's live revision chain for the single best rewind
/// candidate. A revision `N` on `path` qualifies when: `N > 0` (excludes
/// pristine — undoing every edit to a file is SCRAP's job, not a targeted
/// rewind), it parsed cleanly (`ast_ok`), it was itself near-clean
/// (`file_errors <= CANDIDATE_MAX_ERRORS`), and the file's current state has
/// regressed at least `REGRESSION_MARGIN` errors past it. Among qualifying
/// revisions on a file, the LATEST one wins (preserves the most progress);
/// among files, the one with the largest regression wins. `None` if no file
/// qualifies.
pub fn find_rewind_candidate(
    changed_files: &[String],
    revisions: &RevisionStore,
) -> Option<RewindCandidate> {
    changed_files
        .iter()
        .filter_map(|path| {
            let live: Vec<_> = revisions
                .list(path)
                .into_iter()
                .filter(|r| !r.reverted)
                .collect();
            let current = live.last()?;
            let best_rev = live.iter().rev().skip(1).find(|r| {
                r.number > 0
                    && r.ast_ok
                    && r.file_errors <= CANDIDATE_MAX_ERRORS
                    && current.file_errors >= r.file_errors + REGRESSION_MARGIN
            })?;
            Some(RewindCandidate {
                path: path.clone(),
                rev: best_rev.number,
                file_errors_now: current.file_errors,
                file_errors_then: best_rev.file_errors,
            })
        })
        .max_by_key(|c| c.file_errors_now.saturating_sub(c.file_errors_then))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::fast::revisions::RecordArgs;

    fn rec(op: &'static str, ast_ok: bool, file_errors: usize) -> RecordArgs<'static> {
        RecordArgs {
            operation: op,
            label: op,
            range: None,
            payload: None,
            added: 1,
            removed: 1,
            ast_ok,
            ast_error: if ast_ok {
                None
            } else {
                Some("err".to_string())
            },
            file_errors,
            project_errors: file_errors,
        }
    }

    #[test]
    fn finds_the_abandoned_clean_revision() {
        // rev_0 initial(0) -> rev_1 add_param(0) -> rev_2 ok(1) -> ... -> rev_5 ok(31, current)
        let store = RevisionStore::with_cap(20);
        store.ensure_pristine("run.rs", "v0").unwrap();
        store
            .record("run.rs", "v1", rec("change_signature.add_param", true, 0))
            .unwrap();
        store
            .record("run.rs", "v4", rec("replace_range", true, 1))
            .unwrap();
        store
            .record("run.rs", "v5", rec("replace_range", true, 31))
            .unwrap();

        let c = find_rewind_candidate(&["run.rs".to_string()], &store).unwrap();
        assert_eq!(c.path, "run.rs");
        assert_eq!(c.rev, 2); // the rev holding file_errors=1 (numbered 2 in this store)
        assert_eq!(c.file_errors_now, 31);
        assert_eq!(c.file_errors_then, 1);
    }

    #[test]
    fn no_candidate_when_current_is_already_best() {
        // Monotonically improving (even though still broken) — no earlier rev is better.
        let store = RevisionStore::with_cap(20);
        store.ensure_pristine("mod.rs", "v0").unwrap();
        store
            .record("mod.rs", "v1", rec("replace_range", false, 29))
            .unwrap();
        store
            .record("mod.rs", "v2", rec("replace_range", false, 51))
            .unwrap();
        store
            .record("mod.rs", "v3", rec("replace_range", false, 4))
            .unwrap();

        assert!(find_rewind_candidate(&["mod.rs".to_string()], &store).is_none());
    }

    #[test]
    fn no_candidate_when_regression_is_below_margin() {
        let store = RevisionStore::with_cap(20);
        store.ensure_pristine("f.rs", "v0").unwrap();
        store
            .record("f.rs", "v1", rec("replace_range", true, 0))
            .unwrap();
        store
            .record("f.rs", "v2", rec("replace_range", true, 2))
            .unwrap(); // delta=2 < margin=3

        assert!(find_rewind_candidate(&["f.rs".to_string()], &store).is_none());
    }

    #[test]
    fn ignores_pristine_as_a_candidate_target() {
        // Only rev_0 is clean; every real edit made things worse. Undoing
        // everything is SCRAP's job, not a targeted single-file rewind.
        let store = RevisionStore::with_cap(20);
        store.ensure_pristine("f.rs", "v0").unwrap();
        store
            .record("f.rs", "v1", rec("replace_range", false, 10))
            .unwrap();
        store
            .record("f.rs", "v2", rec("replace_range", false, 8))
            .unwrap();

        assert!(find_rewind_candidate(&["f.rs".to_string()], &store).is_none());
    }

    #[test]
    fn picks_the_larger_regression_across_files() {
        let store = RevisionStore::with_cap(20);
        store.ensure_pristine("small.rs", "v0").unwrap();
        store
            .record("small.rs", "v1", rec("replace_range", true, 0))
            .unwrap();
        store
            .record("small.rs", "v2", rec("replace_range", true, 5))
            .unwrap(); // delta=5

        store.ensure_pristine("big.rs", "v0").unwrap();
        store
            .record("big.rs", "v1", rec("replace_range", true, 1))
            .unwrap();
        store
            .record("big.rs", "v2", rec("replace_range", true, 40))
            .unwrap(); // delta=39

        let c =
            find_rewind_candidate(&["small.rs".to_string(), "big.rs".to_string()], &store).unwrap();
        assert_eq!(c.path, "big.rs");
    }

    #[test]
    fn reverted_revisions_are_excluded_from_consideration() {
        // A manually-reverted-away good rev must not surface as "current" or
        // as a candidate source — only the LIVE chain matters.
        let store = RevisionStore::with_cap(20);
        store.ensure_pristine("f.rs", "v0").unwrap();
        store
            .record("f.rs", "v1", rec("replace_range", true, 0))
            .unwrap();
        store.mark_reverted_to("f.rs", 0).unwrap();
        store
            .record("f.rs", "v2", rec("replace_range", true, 20))
            .unwrap();

        // rev_1 is tombstoned (reverted), so the only live history is
        // rev_0(0) -> rev_2(20); rev_0 is pristine and excluded => no candidate.
        assert!(find_rewind_candidate(&["f.rs".to_string()], &store).is_none());
    }
}
