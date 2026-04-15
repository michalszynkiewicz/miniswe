//! `replace_range <path> <start> <end> <content>`
//!
//! Replaces lines `[start..=end]` (1-based, inclusive) with `content`.
//! Empty `content` deletes the range.
//!
//! No OLD-block confirmation: wrong-line edits surface as broken AST or
//! new LSP errors in the next feedback block, and the model reverts.

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

pub async fn execute(
    args: &Value,
    config: &Config,
    perms: &PermissionManager,
    lsp: Option<&LspClient>,
    revisions: &RevisionStore,
    project_baseline_errors: usize,
) -> Result<ToolResult> {
    let path = args["path"].as_str().unwrap_or("");
    let start = args["start"].as_u64().unwrap_or(0) as usize;
    let end = args["end"].as_u64().unwrap_or(0) as usize;
    let content = args["content"].as_str().unwrap_or("");

    if path.is_empty() {
        return Ok(ToolResult::err("replace_range: 'path' is required".into()));
    }
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
    let _removed_lines: Vec<String> = lines_owned.drain(..removed).collect();
    let tail = std::mem::take(&mut lines_owned);

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
