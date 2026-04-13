//! E2E tests for the fast-mode dispatcher.
//!
//! Exercises `execute_fast_tool` end-to-end with a real RevisionStore and
//! permission manager — the same surface the session REPL uses when
//! `tools.edit_mode = "fast"`.

mod helpers;

use std::fs;
use std::path::PathBuf;

use miniswe::tools;
use miniswe::tools::permissions::PermissionManager;
use serde_json::json;

fn setup() -> (tempfile::TempDir, miniswe::config::Config, tools::RevisionStore) {
    let (tmp, config) = helpers::create_test_project();
    let store = tools::RevisionStore::new(&PathBuf::from("/tmp/_unused_test")).unwrap();
    (tmp, config, store)
}

#[tokio::test]
async fn replace_range_applies_and_records_revision() {
    let (_tmp, config, store) = setup();
    fs::write(helpers::project_path(&config, "f.rs"), "a\nb\nc\n").unwrap();
    let perms = PermissionManager::headless(&config);

    let r = tools::execute_fast_tool(
        "replace_range",
        &json!({ "path": "f.rs", "start": 2, "end": 2, "content": "B" }),
        &config,
        &perms,
        None,
        &store,
        0,
    )
    .await
    .unwrap();

    assert!(r.success, "{}", r.content);
    let disk = fs::read_to_string(helpers::project_path(&config, "f.rs")).unwrap();
    assert_eq!(disk, "a\nB\nc\n");
    assert_eq!(store.current("f.rs"), Some(1));
    assert!(r.content.contains("rev_1 applied"), "header should announce rev_1: {}", r.content);
    assert!(r.content.contains("[revisions] f.rs"), "feedback missing revision table: {}", r.content);
}

#[tokio::test]
async fn insert_at_top_and_append_both_work() {
    let (_tmp, config, store) = setup();
    fs::write(helpers::project_path(&config, "f.rs"), "a\nb\n").unwrap();
    let perms = PermissionManager::headless(&config);

    let top = tools::execute_fast_tool(
        "insert_at",
        &json!({ "path": "f.rs", "after_line": 0, "content": "TOP" }),
        &config,
        &perms,
        None,
        &store,
        0,
    )
    .await
    .unwrap();
    assert!(top.success, "{}", top.content);
    assert_eq!(
        fs::read_to_string(helpers::project_path(&config, "f.rs")).unwrap(),
        "TOP\na\nb\n"
    );

    let end = tools::execute_fast_tool(
        "insert_at",
        &json!({ "path": "f.rs", "after_line": 3, "content": "END" }),
        &config,
        &perms,
        None,
        &store,
        0,
    )
    .await
    .unwrap();
    assert!(end.success, "{}", end.content);
    assert_eq!(
        fs::read_to_string(helpers::project_path(&config, "f.rs")).unwrap(),
        "TOP\na\nb\nEND\n"
    );
}

#[tokio::test]
async fn revert_restores_pristine_and_truncates_history() {
    let (_tmp, config, store) = setup();
    fs::write(helpers::project_path(&config, "f.rs"), "original\n").unwrap();
    let perms = PermissionManager::headless(&config);

    // Two edits
    tools::execute_fast_tool(
        "replace_range",
        &json!({ "path": "f.rs", "start": 1, "end": 1, "content": "v1" }),
        &config,
        &perms,
        None,
        &store,
        0,
    )
    .await
    .unwrap();
    tools::execute_fast_tool(
        "replace_range",
        &json!({ "path": "f.rs", "start": 1, "end": 1, "content": "v2" }),
        &config,
        &perms,
        None,
        &store,
        0,
    )
    .await
    .unwrap();
    assert_eq!(store.current("f.rs"), Some(2));

    // Revert to rev_0 (pristine)
    let r = tools::execute_fast_tool(
        "revert",
        &json!({ "path": "f.rs", "rev": 0 }),
        &config,
        &perms,
        None,
        &store,
        0,
    )
    .await
    .unwrap();
    assert!(r.success, "{}", r.content);
    assert_eq!(
        fs::read_to_string(helpers::project_path(&config, "f.rs")).unwrap(),
        "original\n"
    );
    // History truncated to rev_0
    let nums: Vec<usize> = store.list("f.rs").iter().map(|r| r.number).collect();
    assert_eq!(nums, vec![0]);
}

#[tokio::test]
async fn check_on_empty_dir_reports_unknown_toolchain_gracefully() {
    let (_tmp, config, store) = setup();
    let perms = PermissionManager::headless(&config);

    let r = tools::execute_fast_tool(
        "check",
        &json!({}),
        &config,
        &perms,
        None,
        &store,
        0,
    )
    .await
    .unwrap();
    assert!(r.success);
    assert!(r.content.contains("no recognized project toolchain"));
}

#[tokio::test]
async fn unknown_fast_tool_bails() {
    let (_tmp, config, store) = setup();
    let perms = PermissionManager::headless(&config);

    let err = tools::execute_fast_tool(
        "nonexistent_tool",
        &json!({}),
        &config,
        &perms,
        None,
        &store,
        0,
    )
    .await
    .unwrap_err();
    assert!(err.to_string().contains("Unknown fast-mode tool"));
}

#[tokio::test]
async fn fast_definitions_cover_all_primitives() {
    let defs = tools::fast_mode_tool_definitions();
    let names: Vec<&str> = defs.iter().map(|d| d.function.name.as_str()).collect();
    assert!(names.contains(&"replace_range"));
    assert!(names.contains(&"insert_at"));
    assert!(names.contains(&"revert"));
    assert!(names.contains(&"check"));
    assert_eq!(defs.len(), 4);
}
