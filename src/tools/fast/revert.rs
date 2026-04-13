//! `revert <path> <rev>`
//!
//! Restores `path` to a named prior revision. Always explicit — no
//! "undo last" shortcut, no step count. The model scans the revision
//! table (attached to every feedback block) and picks the row to restore.
//!
//! Linear history: reverting to `rev_N` truncates `rev_{N+1}..` and the
//! next edit becomes `rev_{N+1}`. No branching. Revert itself does not
//! create a new revision — the table just shortens.

use anyhow::Result;
use serde_json::Value;

use crate::config::Config;
use crate::lsp::LspClient;

use super::super::ToolResult;
use super::super::permissions::PermissionManager;
use super::feedback::build_feedback;
use super::revisions::RevisionStore;

pub async fn execute(
    args: &Value,
    config: &Config,
    perms: &PermissionManager,
    lsp: Option<&LspClient>,
    revisions: &RevisionStore,
    project_baseline_errors: usize,
) -> Result<ToolResult> {
    let path = args["path"].as_str().unwrap_or("");
    let rev = args["rev"].as_u64().unwrap_or(u64::MAX);

    if path.is_empty() {
        return Ok(ToolResult::err("revert: 'path' is required".into()));
    }
    if rev == u64::MAX {
        return Ok(ToolResult::err(
            "revert: 'rev' is required (the numeric revision to restore, e.g. 0 for pristine)".into(),
        ));
    }
    let rev = rev as usize;

    if let Err(e) = perms.resolve_and_check_path(path) {
        return Ok(ToolResult::err(e));
    }

    // Pull the target revision content from the store. This is the only
    // source of truth — disk may be newer/older/corrupt, we don't care.
    let target_content = match revisions.read_content(path, rev) {
        Ok(s) => s,
        Err(e) => {
            return Ok(ToolResult::err(format!("revert: {e}")));
        }
    };

    let abs_path = config.project_root.join(path);
    if let Err(e) = std::fs::write(&abs_path, &target_content) {
        return Ok(ToolResult::err(format!(
            "revert: write failed for {path}: {e}"
        )));
    }

    // Truncate history: drop rev+1..latest. The next edit becomes rev+1.
    if let Err(e) = revisions.truncate_to(path, rev) {
        return Ok(ToolResult::err(format!("revert: {e}")));
    }

    let fb = build_feedback(
        path,
        &target_content,
        config,
        lsp,
        revisions,
        project_baseline_errors,
    )
    .await;

    let header = format!("revert {path} → rev_{rev}: restored");
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
    async fn revert_to_pristine_restores_rev0_and_truncates() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = scratch_config(tmp.path());
        let p = tmp.path().join("f.rs");
        std::fs::write(&p, "a\nb\n").unwrap();
        let store = RevisionStore::with_cap(20);

        // Seed pristine and two edits
        store.ensure_pristine("f.rs", "a\nb\n").unwrap();
        store.record("f.rs", "A\nb\n", "replace_range L1", 1, 1, true, 0, 0).unwrap();
        store.record("f.rs", "A\nB\n", "replace_range L2", 1, 1, true, 0, 0).unwrap();
        std::fs::write(&p, "A\nB\n").unwrap();

        let r = run(
            serde_json::json!({ "path": "f.rs", "rev": 0 }),
            &cfg,
            &store,
        )
        .await
        .unwrap();
        assert!(r.success, "{}", r.content);
        let disk = std::fs::read_to_string(&p).unwrap();
        assert_eq!(disk, "a\nb\n", "disk should be rev_0 content");
        let nums: Vec<usize> = store.list("f.rs").iter().map(|x| x.number).collect();
        assert_eq!(nums, vec![0], "history should be truncated to rev_0");
    }

    #[tokio::test]
    async fn revert_to_middle_rev_keeps_that_rev() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = scratch_config(tmp.path());
        let p = tmp.path().join("f.rs");
        std::fs::write(&p, "a\n").unwrap();
        let store = RevisionStore::with_cap(20);

        store.ensure_pristine("f.rs", "a\n").unwrap();
        store.record("f.rs", "v1\n", "r1", 1, 1, true, 0, 0).unwrap();
        store.record("f.rs", "v2\n", "r2", 1, 1, true, 0, 0).unwrap();
        store.record("f.rs", "v3\n", "r3", 1, 1, true, 0, 0).unwrap();
        std::fs::write(&p, "v3\n").unwrap();

        let r = run(
            serde_json::json!({ "path": "f.rs", "rev": 1 }),
            &cfg,
            &store,
        )
        .await
        .unwrap();
        assert!(r.success, "{}", r.content);
        let disk = std::fs::read_to_string(&p).unwrap();
        assert_eq!(disk, "v1\n");
        let nums: Vec<usize> = store.list("f.rs").iter().map(|x| x.number).collect();
        assert_eq!(nums, vec![0, 1]);
    }

    #[tokio::test]
    async fn revert_unknown_rev_errors_and_disk_unchanged() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = scratch_config(tmp.path());
        let p = tmp.path().join("f.rs");
        std::fs::write(&p, "current\n").unwrap();
        let store = RevisionStore::with_cap(20);

        store.ensure_pristine("f.rs", "pristine\n").unwrap();

        let r = run(
            serde_json::json!({ "path": "f.rs", "rev": 99 }),
            &cfg,
            &store,
        )
        .await
        .unwrap();
        assert!(!r.success);
        let disk = std::fs::read_to_string(&p).unwrap();
        assert_eq!(disk, "current\n", "disk should be untouched on failure");
    }

    #[tokio::test]
    async fn revert_with_missing_rev_arg_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = scratch_config(tmp.path());
        let store = RevisionStore::with_cap(20);

        let r = run(
            serde_json::json!({ "path": "f.rs" }),
            &cfg,
            &store,
        )
        .await
        .unwrap();
        assert!(!r.success);
        assert!(r.content.contains("rev"));
    }

    #[tokio::test]
    async fn next_edit_after_revert_becomes_next_rev_number() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = scratch_config(tmp.path());
        let p = tmp.path().join("f.rs");
        std::fs::write(&p, "a\n").unwrap();
        let store = RevisionStore::with_cap(20);

        store.ensure_pristine("f.rs", "a\n").unwrap();
        store.record("f.rs", "b\n", "r1", 1, 1, true, 0, 0).unwrap();
        store.record("f.rs", "c\n", "r2", 1, 1, true, 0, 0).unwrap();

        let _ = run(
            serde_json::json!({ "path": "f.rs", "rev": 1 }),
            &cfg,
            &store,
        )
        .await
        .unwrap();
        // After truncating to rev_1, the next record should assign rev_2.
        let n = store.record("f.rs", "d\n", "next", 1, 1, true, 0, 0).unwrap();
        assert_eq!(n, 2);
    }
}
