//! E2E tests for the snapshot/revert system.

mod helpers;

use std::fs;
use miniswe::tools::snapshots::SnapshotManager;

#[test]
fn snapshot_init_creates_shadow_git() {
    let (_tmp, config) = helpers::create_test_project();
    fs::write(helpers::project_path(&config, "test.txt"), "original").unwrap();

    let snap = SnapshotManager::init(&config.project_root).unwrap();

    assert!(config.project_root.join(".miniswe/shadow-git").exists());
    let snapshots = snap.list_snapshots().unwrap();
    assert!(snapshots.contains("round 0"), "should have initial snapshot: {snapshots}");
}

#[test]
fn revert_all_restores_to_session_start() {
    let (_tmp, config) = helpers::create_test_project();
    fs::write(helpers::project_path(&config, "test.txt"), "original").unwrap();

    let mut snap = SnapshotManager::init(&config.project_root).unwrap();

    // Modify file
    fs::write(helpers::project_path(&config, "test.txt"), "modified").unwrap();
    snap.begin_round(1).unwrap();

    // Verify modified
    assert_eq!(fs::read_to_string(helpers::project_path(&config, "test.txt")).unwrap(), "modified");

    // Revert to start
    snap.revert_all().unwrap();

    // Should be back to original
    assert_eq!(
        fs::read_to_string(helpers::project_path(&config, "test.txt")).unwrap(),
        "original"
    );
}

#[test]
fn revert_to_specific_round() {
    let (_tmp, config) = helpers::create_test_project();
    fs::write(helpers::project_path(&config, "test.txt"), "v0").unwrap();

    let mut snap = SnapshotManager::init(&config.project_root).unwrap();

    // Round 1: change to v1
    fs::write(helpers::project_path(&config, "test.txt"), "v1").unwrap();
    snap.begin_round(1).unwrap();

    // Round 2: change to v2
    fs::write(helpers::project_path(&config, "test.txt"), "v2").unwrap();
    snap.begin_round(2).unwrap();

    // Round 3: change to v3
    fs::write(helpers::project_path(&config, "test.txt"), "v3").unwrap();
    snap.begin_round(3).unwrap();

    // Revert to round 1 (should have v1)
    snap.revert_to_round(1).unwrap();
    assert_eq!(
        fs::read_to_string(helpers::project_path(&config, "test.txt")).unwrap(),
        "v1"
    );
}

#[test]
fn revert_single_file() {
    let (_tmp, config) = helpers::create_test_project();
    fs::write(helpers::project_path(&config, "a.txt"), "a_original").unwrap();
    fs::write(helpers::project_path(&config, "b.txt"), "b_original").unwrap();

    let mut snap = SnapshotManager::init(&config.project_root).unwrap();

    // Modify both
    fs::write(helpers::project_path(&config, "a.txt"), "a_modified").unwrap();
    fs::write(helpers::project_path(&config, "b.txt"), "b_modified").unwrap();
    snap.begin_round(1).unwrap();

    // Revert only a.txt to round 0
    snap.revert_file("a.txt", 0).unwrap();

    // a.txt should be original, b.txt should stay modified
    assert_eq!(fs::read_to_string(helpers::project_path(&config, "a.txt")).unwrap(), "a_original");
    assert_eq!(fs::read_to_string(helpers::project_path(&config, "b.txt")).unwrap(), "b_modified");
}

#[test]
fn revert_new_file_roundtrip() {
    let (_tmp, config) = helpers::create_test_project();

    let mut snap = SnapshotManager::init(&config.project_root).unwrap();

    // Create a new file and snapshot it
    fs::write(helpers::project_path(&config, "new_file.txt"), "new content").unwrap();
    snap.begin_round(1).unwrap();

    // Modify the new file
    fs::write(helpers::project_path(&config, "new_file.txt"), "modified content").unwrap();
    snap.begin_round(2).unwrap();

    // Revert to round 1 — should have original "new content"
    snap.revert_to_round(1).unwrap();
    assert_eq!(
        fs::read_to_string(helpers::project_path(&config, "new_file.txt")).unwrap(),
        "new content"
    );
}

#[test]
fn revert_nonexistent_round_fails() {
    let (_tmp, config) = helpers::create_test_project();

    let snap = SnapshotManager::init(&config.project_root).unwrap();

    let result = snap.revert_to_round(999);
    assert!(result.is_err());
}

#[test]
fn multiple_snapshots_listed() {
    let (_tmp, config) = helpers::create_test_project();
    fs::write(helpers::project_path(&config, "test.txt"), "v0").unwrap();

    let mut snap = SnapshotManager::init(&config.project_root).unwrap();

    fs::write(helpers::project_path(&config, "test.txt"), "v1").unwrap();
    snap.begin_round(1).unwrap();

    fs::write(helpers::project_path(&config, "test.txt"), "v2").unwrap();
    snap.begin_round(2).unwrap();

    let list = snap.list_snapshots().unwrap();
    assert!(list.contains("round 0"), "should list round 0: {list}");
    assert!(list.contains("round 1"), "should list round 1: {list}");
    assert!(list.contains("round 2"), "should list round 2: {list}");
}
