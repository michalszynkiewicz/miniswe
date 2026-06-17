//! Auto-revert on AST-break cascade (experimental, opt-in via
//! `tools.auto_revert_ast_cascade`).
//!
//! Fast mode tolerates *transient* broken AST by design: a model may edit a
//! signature on one call and the body on the next, leaving the file briefly
//! unparseable in between. But small models sometimes dig deeper instead of
//! recovering — each subsequent edit keeps the AST broken, compounding the
//! damage (observed on Gemma 4: a single brace slip snowballing into 232
//! consecutive `[ast] broken` rounds with no recovery).
//!
//! This guard detects that *cascade* — `CASCADE_THRESHOLD`+ consecutive
//! broken-AST live revisions with no intervening clean parse — and forcibly
//! reverts the file to the most recent AST-clean revision, breaking the loop
//! and handing the model a known-good base to retry from. A single broken
//! edit followed by a fix never triggers it, so the legitimate two-step
//! structural edit is left alone.

use crate::config::Config;
use crate::lsp::LspClient;

use super::revisions::RevisionStore;

/// Number of consecutive broken-AST live revisions that constitutes a
/// cascade worth force-reverting. 3 keeps legitimate two-step structural
/// edits (header then body) untouched while catching the dig-deeper loop.
pub const CASCADE_THRESHOLD: usize = 3;

/// If the live chain for `path` ends in `CASCADE_THRESHOLD`+ consecutive
/// broken-AST revisions, revert the file to the most recent AST-clean
/// revision (rev_0 pristine if none) and return a message to append to the
/// tool result. Returns `None` when no cascade is detected or the revert
/// could not be performed (best-effort; on failure the state is left as-is).
pub async fn maybe_break_cascade(
    path: &str,
    config: &Config,
    lsp: Option<&LspClient>,
    revisions: &RevisionStore,
) -> Option<String> {
    let revs = revisions.list(path);
    // Live chain only, in recorded order.
    let live: Vec<_> = revs.iter().filter(|r| !r.reverted).collect();

    // Current on-disk state parses fine → nothing to break out of.
    if live.last().map(|r| r.ast_ok).unwrap_or(true) {
        return None;
    }

    // Count trailing consecutive broken-AST live revs.
    let trailing_broken = live.iter().rev().take_while(|r| !r.ast_ok).count();
    if trailing_broken < CASCADE_THRESHOLD {
        return None;
    }

    // Anchor = most recent live rev that parsed cleanly. rev_0 is always
    // ast_ok and always live, so this is guaranteed to find something.
    let anchor_num = live.iter().rev().find(|r| r.ast_ok)?.number;
    let restored = revisions.read_content(path, anchor_num).ok()?;

    let abs = config.project_root.join(path);
    if std::fs::write(&abs, &restored).is_err() {
        return None; // leave broken state on disk rather than lie about reverting
    }
    revisions.mark_reverted_to(path, anchor_num).ok()?;
    if let Some(lsp) = lsp {
        let _ = lsp.notify_file_changed(&abs);
    }
    super::super::edit_orchestration::reindex_changed_file(path, config);

    Some(format!(
        "\n\n[auto-revert] Your last {trailing_broken} edits to {path} EACH left the syntax tree broken — \
         you were digging deeper, not recovering. I reverted {path} to rev_{anchor_num} \
         (the last state that parsed cleanly). STOP patching line-by-line. \
         Re-read the relevant region with file(action='read'), then make ONE complete, \
         balanced edit: a single replace_range over the whole enclosing block (with matching \
         braces/brackets), not a sequence of partial line fixes. Do not replay the \
         byte-identical edits now shown as [reverted] in the revision table."
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::fast::RevisionStore;
    use std::path::Path;

    fn cfg(dir: &Path) -> Config {
        let mut c = Config::default();
        c.project_root = dir.to_path_buf();
        c
    }

    fn record(store: &RevisionStore, path: &str, content: &str, ast_ok: bool) {
        store
            .record(
                path,
                content,
                crate::tools::fast::RecordArgs {
                    operation: "replace_range",
                    label: "replace_range L1-1",
                    range: Some((1, 1)),
                    payload: Some(content.to_string()),
                    added: 1,
                    removed: 1,
                    ast_ok,
                    ast_error: if ast_ok {
                        None
                    } else {
                        Some("L1:1: syntax error".into())
                    },
                    file_errors: 0,
                    project_errors: 0,
                },
            )
            .unwrap();
    }

    #[tokio::test]
    async fn no_cascade_when_current_state_is_clean() {
        let tmp = tempfile::tempdir().unwrap();
        let c = cfg(tmp.path());
        std::fs::write(tmp.path().join("f.rs"), "fn ok() {}\n").unwrap();
        let store = RevisionStore::with_cap(50);
        store.ensure_pristine("f.rs", "fn ok() {}\n").unwrap();
        record(&store, "f.rs", "broken", false);
        record(&store, "f.rs", "fn ok() {}\n", true); // recovered
        assert!(
            maybe_break_cascade("f.rs", &c, None, &store)
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn no_cascade_below_threshold() {
        let tmp = tempfile::tempdir().unwrap();
        let c = cfg(tmp.path());
        std::fs::write(tmp.path().join("f.rs"), "broken2").unwrap();
        let store = RevisionStore::with_cap(50);
        store.ensure_pristine("f.rs", "fn ok() {}\n").unwrap();
        record(&store, "f.rs", "broken1", false);
        record(&store, "f.rs", "broken2", false); // only 2 broken in a row
        assert!(
            maybe_break_cascade("f.rs", &c, None, &store)
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn cascade_reverts_to_last_clean_revision() {
        let tmp = tempfile::tempdir().unwrap();
        let c = cfg(tmp.path());
        let pristine = "fn ok() {}\n";
        std::fs::write(tmp.path().join("f.rs"), "broken3").unwrap();
        let store = RevisionStore::with_cap(50);
        store.ensure_pristine("f.rs", pristine).unwrap();
        record(&store, "f.rs", "broken1", false);
        record(&store, "f.rs", "broken2", false);
        record(&store, "f.rs", "broken3", false); // 3 broken in a row → cascade

        let msg = maybe_break_cascade("f.rs", &c, None, &store)
            .await
            .expect("cascade should fire");
        assert!(msg.contains("auto-revert"));
        assert!(msg.contains("rev_0"));
        // File on disk restored to pristine.
        let disk = std::fs::read_to_string(tmp.path().join("f.rs")).unwrap();
        assert_eq!(disk, pristine);
        // Broken revs are now tombstones; current live rev is rev_0.
        assert_eq!(store.current("f.rs"), Some(0));
    }

    #[tokio::test]
    async fn cascade_anchors_on_intervening_clean_rev() {
        let tmp = tempfile::tempdir().unwrap();
        let c = cfg(tmp.path());
        let good = "fn good() {}\n";
        std::fs::write(tmp.path().join("f.rs"), "broken3").unwrap();
        let store = RevisionStore::with_cap(50);
        store.ensure_pristine("f.rs", "fn ok() {}\n").unwrap();
        record(&store, "f.rs", good, true); // rev_1 clean
        record(&store, "f.rs", "broken1", false);
        record(&store, "f.rs", "broken2", false);
        record(&store, "f.rs", "broken3", false); // 3 broken → cascade

        let msg = maybe_break_cascade("f.rs", &c, None, &store)
            .await
            .expect("cascade should fire");
        assert!(msg.contains("rev_1"), "should anchor on rev_1: {msg}");
        let disk = std::fs::read_to_string(tmp.path().join("f.rs")).unwrap();
        assert_eq!(disk, good);
        assert_eq!(store.current("f.rs"), Some(1));
    }

    // Real scenario from a Gemma 4 bench run: the model manually reverts to
    // rev_0 several times (tombstoning rev_1..rev_8), then lands 3 consecutive
    // broken edits (rev_9/10/11) with no revert between. The trailing-broken
    // count must look only at the *live* chain, ignoring the tombstones, so
    // the cascade is still detected.
    #[tokio::test]
    async fn cascade_after_manual_reverts_tombstone_earlier_revs() {
        let tmp = tempfile::tempdir().unwrap();
        let c = cfg(tmp.path());
        std::fs::write(tmp.path().join("m.rs"), "broken11").unwrap();
        let store = RevisionStore::with_cap(50);
        store.ensure_pristine("m.rs", "fn ok() {}\n").unwrap();
        // 8 broken edits, each followed by a manual revert to rev_0.
        for i in 1..=8 {
            record(&store, "m.rs", &format!("broken{i}"), false);
            store.mark_reverted_to("m.rs", 0).unwrap();
        }
        // 3 consecutive broken with no revert between → cascade.
        record(&store, "m.rs", "broken9", false);
        record(&store, "m.rs", "broken10", false);
        record(&store, "m.rs", "broken11", false);
        assert!(
            maybe_break_cascade("m.rs", &c, None, &store)
                .await
                .is_some(),
            "guard must fire on 3 live consecutive broken even when earlier revs are tombstoned"
        );
    }
}
