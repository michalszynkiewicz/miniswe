//! `replace_range <path> <start> <end> <content>`
//!
//! Replaces lines `[start..=end]` (1-based, inclusive) with `content`.
//! Empty `content` deletes the range.
//!
//! No OLD-block confirmation: wrong-line edits surface as broken AST or
//! new LSP errors in the next feedback block, and the model reverts.
//!
//! Every applied edit echoes its real line diff back to the model, with a
//! revert hint, so a silently dropped line is visible immediately —
//! AST/LSP feedback can't catch drops that still compile (the recurring
//! dropped-provider-header bench bug: wide rewrites retyped from memory
//! lose lines without any error). A hard range cap was tried against the
//! same failure mode (2026-07-13) and REVERTED: it blocked a fully correct
//! 46-line rewrite 14 times in a row until the attempt died — the model
//! can't reliably do the "split your own payload" surgery the rejection
//! asks for.

use anyhow::Result;
use serde_json::Value;

use crate::config::Config;
use crate::lsp::LspClient;

use super::super::ToolResult;
use super::super::permissions::PermissionManager;
use super::feedback::build_feedback;
use super::lines::{
    join_with_trailing_nl, split_preserving_trailing_nl, split_replacement, validate_range,
};
use super::revisions::{RecordArgs, RevisionStore};

/// Line-level LCS diff of the replaced range, rendered as `-`/`+` lines
/// with unchanged lines omitted. Ranges are typically tens of lines, so
/// the quadratic table is negligible.
fn render_applied_diff(old: &[String], new: &[String]) -> String {
    let n = old.len();
    let m = new.len();
    // lcs[i][j] = LCS length of old[i..] vs new[j..]
    let mut lcs = vec![vec![0usize; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            lcs[i][j] = if old[i] == new[j] {
                lcs[i + 1][j + 1] + 1
            } else {
                lcs[i + 1][j].max(lcs[i][j + 1])
            };
        }
    }
    let (mut i, mut j) = (0, 0);
    let mut out = String::new();
    while i < n || j < m {
        if i < n && j < m && old[i] == new[j] {
            i += 1;
            j += 1;
        } else if j < m && (i == n || lcs[i][j + 1] >= lcs[i + 1][j]) {
            out.push('+');
            out.push_str(&new[j]);
            out.push('\n');
            j += 1;
        } else {
            out.push('-');
            out.push_str(&old[i]);
            out.push('\n');
            i += 1;
        }
    }
    out
}

pub async fn execute(
    args: &Value,
    config: &Config,
    perms: &PermissionManager,
    lsp: Option<&LspClient>,
    revisions: &RevisionStore,
    project_baseline_errors: usize,
) -> Result<ToolResult> {
    let path = match super::super::args::require_str(args, "path") {
        Ok(p) => p,
        Err(e) => return Ok(ToolResult::err(e)),
    };
    let start = match super::super::args::require_u64(args, "start") {
        Ok(n) => n as usize,
        Err(e) => return Ok(ToolResult::err(e)),
    };
    let end = match super::super::args::require_u64(args, "end") {
        Ok(n) => n as usize,
        Err(e) => return Ok(ToolResult::err(e)),
    };
    let content = match super::super::args::require_str(args, "content") {
        Ok(c) => c,
        Err(e) => return Ok(ToolResult::err(e)),
    };

    if let Err(e) = perms.resolve_and_check_path(path) {
        return Ok(ToolResult::err(e));
    }

    let abs_path = config.project_root.join(path);
    let original = match std::fs::read_to_string(&abs_path) {
        Ok(s) => s,
        Err(e) => {
            return Ok(ToolResult::err(format!(
                "replace_range: cannot read {path}: {e}"
            )));
        }
    };

    let (mut lines_owned, had_nl) = {
        let (lines, had_nl) = split_preserving_trailing_nl(&original);
        (
            lines.into_iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            had_nl,
        )
    };
    let line_count = lines_owned.len();

    if let Err(msg) = validate_range(start, end, line_count) {
        return Ok(ToolResult::err(format!("replace_range: {msg}")));
    }

    let replacement_lines = split_replacement(content);
    let removed = end - start + 1;
    let added = replacement_lines.len();

    // Splice [start..=end] (1-based) with replacement.
    let head: Vec<String> = lines_owned.drain(..start - 1).collect();
    let removed_lines: Vec<String> = lines_owned.drain(..removed).collect();
    let tail = std::mem::take(&mut lines_owned);

    // Diff before the splice consumes `replacement_lines`. Pure deletion
    // (empty content) diffs against nothing — split_replacement's [""]
    // placeholder would otherwise render as a spurious `+` blank.
    let empty: Vec<String> = Vec::new();
    let applied_diff = render_applied_diff(
        &removed_lines,
        if content.is_empty() {
            &empty
        } else {
            &replacement_lines
        },
    );

    let mut new_lines: Vec<String> = head;
    new_lines.extend(replacement_lines);
    new_lines.extend(tail);

    // Deletion semantics: if `content` was empty AND we removed at least
    // one line, the `split_replacement("")` → [""] placeholder would
    // leave a spurious blank row. Drop it.
    let new_lines = if content.is_empty() {
        // Replace [start..=end] with nothing — remove the placeholder
        // empty string that split_replacement inserted.
        let mut trimmed = Vec::with_capacity(new_lines.len().saturating_sub(1));
        trimmed.extend(
            new_lines
                .into_iter()
                .enumerate()
                .filter_map(|(i, l)| if i == start - 1 { None } else { Some(l) }),
        );
        trimmed
    } else {
        new_lines
    };

    // Handle edge case: we just deleted the entire file.
    let (new_lines, had_nl) = if new_lines.is_empty() {
        (vec![String::new()], false)
    } else {
        (new_lines, had_nl)
    };

    let new_content = join_with_trailing_nl(&new_lines, had_nl);

    // No-op guard (cf. Aider's "those lines are already present"): if the
    // splice changes nothing, the model is re-applying an edit already in place.
    // Say so instead of recording a duplicate revision and implying success —
    // a small model that gets "applied" here will think the fix landed when it
    // didn't, and churn.
    if new_content == original {
        return Ok(ToolResult::err(format!(
            "replace_range: lines L{start}-{end} of {path} already match the content you provided — \
             nothing changed. The edit is already in place; re-read the file and look elsewhere if \
             something still isn't right."
        )));
    }

    if let Err(e) = std::fs::write(&abs_path, &new_content) {
        return Ok(ToolResult::err(format!(
            "replace_range: write failed for {path}: {e}"
        )));
    }

    // Make sure a pristine baseline exists before we record a new rev.
    revisions.ensure_pristine(path, &original)?;

    // Build feedback (AST + LSP + delta) so we have the stats before
    // recording the revision — they go into the table row.
    let fb = build_feedback(
        path,
        &new_content,
        config,
        lsp,
        revisions,
        project_baseline_errors,
    )
    .await;

    // Record the new revision with the feedback stats.
    let rev = revisions.record(
        path,
        &new_content,
        RecordArgs {
            operation: "replace_range",
            label: &format!("replace_range L{start}-{end}"),
            range: Some((start, end)),
            payload: Some(content.to_string()),
            added,
            removed,
            ast_ok: fb.ast_ok,
            ast_error: fb.ast_error.clone(),
            file_errors: fb.file_errors,
            project_errors: fb.project_errors,
        },
    )?;

    // Re-render feedback so the revision table includes the row we just
    // added. (build_feedback before record() shows state *without* the
    // new row; after record() the revs list reflects it.)
    let fb = build_feedback(
        path,
        &new_content,
        config,
        lsp,
        revisions,
        project_baseline_errors,
    )
    .await;

    let header =
        format!("replace_range {path} L{start}-{end}: rev_{rev} applied (+{added} -{removed})");
    let mut out = String::from(&header);
    // Echo the real applied diff: a dropped line that still compiles is
    // invisible to AST/LSP feedback, so show the model exactly what it
    // removed and give it an immediate way out.
    if !applied_diff.is_empty() {
        out.push_str("\nApplied diff (every '-' line is now GONE from the file):\n");
        out.push_str(&applied_diff);
        out.push_str("If this is not exactly the edit you intended, call revert.");
    }
    out.push_str(&fb.text);
    Ok(ToolResult::ok(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::tools::permissions::PermissionManager;

    fn scratch_config(dir: &std::path::Path) -> Config {
        let mut cfg = Config::default();
        cfg.project_root = dir.to_path_buf();
        cfg
    }

    async fn run(
        args: serde_json::Value,
        cfg: &Config,
        store: &RevisionStore,
    ) -> Result<ToolResult> {
        let perms = PermissionManager::new(cfg);
        execute(&args, cfg, &perms, None, store, 0).await
    }

    #[tokio::test]
    async fn replaces_single_line() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = scratch_config(tmp.path());
        std::fs::write(tmp.path().join("f.rs"), "a\nb\nc\n").unwrap();
        let store = RevisionStore::with_cap(20);

        let r = run(
            serde_json::json!({ "path": "f.rs", "start": 2, "end": 2, "content": "B" }),
            &cfg,
            &store,
        )
        .await
        .unwrap();
        assert!(r.success, "{}", r.content);
        let disk = std::fs::read_to_string(tmp.path().join("f.rs")).unwrap();
        assert_eq!(disk, "a\nB\nc\n");
        assert_eq!(store.current("f.rs"), Some(1));
    }

    #[tokio::test]
    async fn replaces_multi_line_with_multi_line() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = scratch_config(tmp.path());
        std::fs::write(tmp.path().join("f.rs"), "1\n2\n3\n4\n").unwrap();
        let store = RevisionStore::with_cap(20);

        let r = run(
            serde_json::json!({
                "path": "f.rs", "start": 2, "end": 3, "content": "X\nY\nZ"
            }),
            &cfg,
            &store,
        )
        .await
        .unwrap();
        assert!(r.success);
        let disk = std::fs::read_to_string(tmp.path().join("f.rs")).unwrap();
        assert_eq!(disk, "1\nX\nY\nZ\n4\n");
    }

    #[tokio::test]
    async fn empty_content_deletes_range() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = scratch_config(tmp.path());
        std::fs::write(tmp.path().join("f.rs"), "1\n2\n3\n4\n").unwrap();
        let store = RevisionStore::with_cap(20);

        let r = run(
            serde_json::json!({ "path": "f.rs", "start": 2, "end": 3, "content": "" }),
            &cfg,
            &store,
        )
        .await
        .unwrap();
        assert!(r.success, "{}", r.content);
        let disk = std::fs::read_to_string(tmp.path().join("f.rs")).unwrap();
        assert_eq!(disk, "1\n4\n");
    }

    #[tokio::test]
    async fn noop_replacement_is_flagged_not_recorded() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = scratch_config(tmp.path());
        std::fs::write(tmp.path().join("f.rs"), "a\nb\nc\n").unwrap();
        let store = RevisionStore::with_cap(20);

        // Replace line 2 ("b") with "b" — a no-op.
        let r = run(
            serde_json::json!({ "path": "f.rs", "start": 2, "end": 2, "content": "b" }),
            &cfg,
            &store,
        )
        .await
        .unwrap();
        assert!(!r.success, "no-op should be flagged: {}", r.content);
        assert!(r.content.contains("already match"));
        // No revision recorded for a no-op.
        assert_eq!(store.current("f.rs"), None);
        // Disk unchanged.
        assert_eq!(
            std::fs::read_to_string(tmp.path().join("f.rs")).unwrap(),
            "a\nb\nc\n"
        );
    }

    #[tokio::test]
    async fn out_of_range_errors_and_does_not_touch_disk() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = scratch_config(tmp.path());
        std::fs::write(tmp.path().join("f.rs"), "a\nb\n").unwrap();
        let store = RevisionStore::with_cap(20);

        let r = run(
            serde_json::json!({ "path": "f.rs", "start": 5, "end": 5, "content": "X" }),
            &cfg,
            &store,
        )
        .await
        .unwrap();
        assert!(!r.success);
        let disk = std::fs::read_to_string(tmp.path().join("f.rs")).unwrap();
        assert_eq!(disk, "a\nb\n");
        assert_eq!(
            store.current("f.rs"),
            None,
            "no rev should be recorded on failure"
        );
    }

    #[tokio::test]
    async fn wide_range_is_allowed_and_diffed() {
        // The 30-line cap was tried and reverted (see module doc) — wide
        // ranges must apply, with the diff echo as the safety net.
        let tmp = tempfile::tempdir().unwrap();
        let cfg = scratch_config(tmp.path());
        let original: String = (1..=40).map(|i| format!("line{i}\n")).collect();
        std::fs::write(tmp.path().join("f.rs"), &original).unwrap();
        let store = RevisionStore::with_cap(20);

        let r = run(
            serde_json::json!({ "path": "f.rs", "start": 1, "end": 35, "content": "X" }),
            &cfg,
            &store,
        )
        .await
        .unwrap();
        assert!(r.success, "35-line range must apply: {}", r.content);
        assert!(r.content.contains("Applied diff"), "{}", r.content);
        assert!(r.content.contains("-line35"), "{}", r.content);
    }

    #[tokio::test]
    async fn result_echoes_applied_diff_with_revert_hint() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = scratch_config(tmp.path());
        std::fs::write(tmp.path().join("f.rs"), "keep1\nold\nkeep2\n").unwrap();
        let store = RevisionStore::with_cap(20);

        let r = run(
            serde_json::json!({ "path": "f.rs", "start": 1, "end": 3, "content": "keep1\nnew\nkeep2" }),
            &cfg,
            &store,
        )
        .await
        .unwrap();
        assert!(r.success, "{}", r.content);
        assert!(r.content.contains("Applied diff"), "{}", r.content);
        assert!(r.content.contains("-old"), "{}", r.content);
        assert!(r.content.contains("+new"), "{}", r.content);
        // unchanged lines inside the range are NOT echoed
        assert!(!r.content.contains("-keep1"), "{}", r.content);
        assert!(!r.content.contains("+keep2"), "{}", r.content);
        assert!(r.content.contains("call revert"), "{}", r.content);
    }

    #[tokio::test]
    async fn deletion_diff_shows_all_removed_lines() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = scratch_config(tmp.path());
        std::fs::write(tmp.path().join("f.rs"), "a\nb\nc\n").unwrap();
        let store = RevisionStore::with_cap(20);

        let r = run(
            serde_json::json!({ "path": "f.rs", "start": 2, "end": 3, "content": "" }),
            &cfg,
            &store,
        )
        .await
        .unwrap();
        assert!(r.success, "{}", r.content);
        assert!(r.content.contains("-b"), "{}", r.content);
        assert!(r.content.contains("-c"), "{}", r.content);
        // no spurious "+" blank from the deletion placeholder
        assert!(!r.content.contains("\n+\n"), "{}", r.content);
    }

    #[tokio::test]
    async fn second_edit_records_rev_2() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = scratch_config(tmp.path());
        std::fs::write(tmp.path().join("f.rs"), "a\nb\n").unwrap();
        let store = RevisionStore::with_cap(20);

        run(
            serde_json::json!({ "path": "f.rs", "start": 1, "end": 1, "content": "A" }),
            &cfg,
            &store,
        )
        .await
        .unwrap();
        run(
            serde_json::json!({ "path": "f.rs", "start": 2, "end": 2, "content": "B" }),
            &cfg,
            &store,
        )
        .await
        .unwrap();
        assert_eq!(store.current("f.rs"), Some(2));
        // rev_0 should still be the pristine original
        assert_eq!(store.read_content("f.rs", 0).unwrap(), "a\nb\n");
    }
}
