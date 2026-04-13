//! `insert_at <path> <after_line> <content>`
//!
//! Inserts `content` after line `after_line` (1-based).
//! `after_line = 0` inserts at the top of the file.
//!
//! Kept distinct from `replace_range` so tiny models don't have to encode
//! insertion as `replace_range N N "<echoed line N>\nnew"` or as
//! `start > end` slice trickery. Both are recurring cognitive traps.

use anyhow::Result;
use serde_json::Value;

use crate::config::Config;
use crate::lsp::LspClient;

use super::super::ToolResult;
use super::super::permissions::PermissionManager;
use super::feedback::build_feedback;
use super::lines::{
    join_with_trailing_nl, split_preserving_trailing_nl, split_replacement,
    validate_insertion_anchor,
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
    let after_line = args["after_line"].as_u64().unwrap_or(0) as usize;
    let content = args["content"].as_str().unwrap_or("");

    if path.is_empty() {
        return Ok(ToolResult::err("insert_at: 'path' is required".into()));
    }
    if content.is_empty() {
        return Ok(ToolResult::err(
            "insert_at: 'content' is required (empty insertion is a no-op)".into(),
        ));
    }
    if let Err(e) = perms.resolve_and_check_path(path) {
        return Ok(ToolResult::err(e));
    }

    let abs_path = config.project_root.join(path);
    let original = match std::fs::read_to_string(&abs_path) {
        Ok(s) => s,
        Err(e) => {
            return Ok(ToolResult::err(format!(
                "insert_at: cannot read {path}: {e}"
            )));
        }
    };

    let (lines, had_nl) = split_preserving_trailing_nl(&original);
    let mut lines: Vec<String> = lines.into_iter().map(|s| s.to_string()).collect();
    let line_count = lines.len();

    if let Err(msg) = validate_insertion_anchor(after_line, line_count) {
        return Ok(ToolResult::err(format!("insert_at: {msg}")));
    }

    let replacement_lines = split_replacement(content);
    let added = replacement_lines.len();

    // after_line=0 → insert at index 0 (top of file).
    // after_line=N → insert at index N (after line N in 1-based terms).
    let insert_at = after_line;
    let tail = lines.split_off(insert_at);
    let mut new_lines = lines;
    new_lines.extend(replacement_lines);
    new_lines.extend(tail);

    let new_content = join_with_trailing_nl(&new_lines, had_nl);

    if let Err(e) = std::fs::write(&abs_path, &new_content) {
        return Ok(ToolResult::err(format!(
            "insert_at: write failed for {path}: {e}"
        )));
    }

    revisions.ensure_pristine(path, &original)?;

    let fb = build_feedback(
        path,
        &new_content,
        config,
        lsp,
        revisions,
        project_baseline_errors,
    )
    .await;

    let rev = revisions.record(
        path,
        &new_content,
        RecordArgs {
            operation: "insert_at",
            label: &format!("insert_at after L{after_line}"),
            range: None,
            payload: Some(content.to_string()),
            added,
            removed: 0,
            ast_ok: fb.ast_ok,
            ast_error: fb.ast_error.clone(),
            file_errors: fb.file_errors,
            project_errors: fb.project_errors,
        },
    )?;

    let fb = build_feedback(
        path,
        &new_content,
        config,
        lsp,
        revisions,
        project_baseline_errors,
    )
    .await;

    let header = format!("insert_at {path} after L{after_line}: rev_{rev} applied (+{added} -0)");
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
    async fn insert_after_middle_line() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = scratch_config(tmp.path());
        std::fs::write(tmp.path().join("f.rs"), "a\nb\nc\n").unwrap();
        let store = RevisionStore::with_cap(20);

        let r = run(
            serde_json::json!({ "path": "f.rs", "after_line": 2, "content": "NEW" }),
            &cfg,
            &store,
        )
        .await
        .unwrap();
        assert!(r.success, "{}", r.content);
        let disk = std::fs::read_to_string(tmp.path().join("f.rs")).unwrap();
        assert_eq!(disk, "a\nb\nNEW\nc\n");
    }

    #[tokio::test]
    async fn insert_at_top_uses_after_line_zero() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = scratch_config(tmp.path());
        std::fs::write(tmp.path().join("f.rs"), "a\nb\n").unwrap();
        let store = RevisionStore::with_cap(20);

        let r = run(
            serde_json::json!({ "path": "f.rs", "after_line": 0, "content": "TOP" }),
            &cfg,
            &store,
        )
        .await
        .unwrap();
        assert!(r.success, "{}", r.content);
        let disk = std::fs::read_to_string(tmp.path().join("f.rs")).unwrap();
        assert_eq!(disk, "TOP\na\nb\n");
    }

    #[tokio::test]
    async fn insert_at_end_appends() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = scratch_config(tmp.path());
        std::fs::write(tmp.path().join("f.rs"), "a\nb\n").unwrap();
        let store = RevisionStore::with_cap(20);

        let r = run(
            serde_json::json!({ "path": "f.rs", "after_line": 2, "content": "END" }),
            &cfg,
            &store,
        )
        .await
        .unwrap();
        assert!(r.success);
        let disk = std::fs::read_to_string(tmp.path().join("f.rs")).unwrap();
        assert_eq!(disk, "a\nb\nEND\n");
    }

    #[tokio::test]
    async fn multi_line_content_inserted_verbatim() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = scratch_config(tmp.path());
        std::fs::write(tmp.path().join("f.rs"), "a\nb\n").unwrap();
        let store = RevisionStore::with_cap(20);

        let r = run(
            serde_json::json!({ "path": "f.rs", "after_line": 1, "content": "X\nY\nZ" }),
            &cfg,
            &store,
        )
        .await
        .unwrap();
        assert!(r.success);
        let disk = std::fs::read_to_string(tmp.path().join("f.rs")).unwrap();
        assert_eq!(disk, "a\nX\nY\nZ\nb\n");
    }

    #[tokio::test]
    async fn anchor_past_end_rejected_and_disk_unchanged() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = scratch_config(tmp.path());
        std::fs::write(tmp.path().join("f.rs"), "a\n").unwrap();
        let store = RevisionStore::with_cap(20);

        let r = run(
            serde_json::json!({ "path": "f.rs", "after_line": 99, "content": "X" }),
            &cfg,
            &store,
        )
        .await
        .unwrap();
        assert!(!r.success);
        let disk = std::fs::read_to_string(tmp.path().join("f.rs")).unwrap();
        assert_eq!(disk, "a\n");
        assert_eq!(store.current("f.rs"), None);
    }

    #[tokio::test]
    async fn empty_content_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = scratch_config(tmp.path());
        std::fs::write(tmp.path().join("f.rs"), "a\n").unwrap();
        let store = RevisionStore::with_cap(20);

        let r = run(
            serde_json::json!({ "path": "f.rs", "after_line": 1, "content": "" }),
            &cfg,
            &store,
        )
        .await
        .unwrap();
        assert!(!r.success);
    }
}
