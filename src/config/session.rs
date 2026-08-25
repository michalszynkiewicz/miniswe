//! Per-session state directories under `.miniswe/sessions/<id>/`.
//!
//! `plan.md` and `scratchpad.md` are session working state, not project
//! state, but they used to live at a fixed path in `.miniswe/`. That made
//! every miniswe process in a project share one mutable copy, and since a
//! non-`--continue` session start deleted both files to clear stale state,
//! any second process silently wiped the first one's plan. The benchmark hit
//! this constantly: the task under test *is* miniswe, so when the agent
//! smoke-tested the binary it had just built inside its own workspace, the
//! child run deleted its parent's plan mid-run — which also stripped the
//! parent's edit tools, because `plan_exists` gates them.
//!
//! Giving each session its own directory removes the shared path entirely,
//! so start-up no longer needs to delete anything to be safe.

use std::path::{Path, PathBuf};

/// How many session directories to keep. Older ones are pruned at session
/// start so `.miniswe/sessions/` doesn't grow without bound.
pub const RETENTION: usize = 20;

/// Pointer file naming the most recent session, so `--continue` can find
/// the previous session's state.
const LAST_POINTER: &str = "last";

/// A fresh session id: local timestamp plus pid.
///
/// No uuid dependency needed — two live processes can't share a pid, and
/// a recycled pid can't land in the same second as its predecessor. Sorts
/// chronologically, matching the `logs/` naming convention.
pub fn new_id() -> String {
    format!(
        "{}_{}",
        chrono::Local::now().format("%Y%m%d_%H%M%S"),
        std::process::id()
    )
}

/// Record `id` as the most recent session.
pub fn record_last(sessions_dir: &Path, id: &str) {
    let _ = std::fs::create_dir_all(sessions_dir);
    let _ = std::fs::write(sessions_dir.join(LAST_POINTER), id);
}

/// The previous session's id, if one was recorded and its directory still
/// exists. `None` means `--continue` has nothing to carry forward.
pub fn last_id(sessions_dir: &Path) -> Option<String> {
    let raw = std::fs::read_to_string(sessions_dir.join(LAST_POINTER)).ok()?;
    let id = raw.trim();
    if id.is_empty() || !sessions_dir.join(id).is_dir() {
        return None;
    }
    Some(id.to_string())
}

/// Delete all but the `keep` most recent session directories, never the
/// `active` one.
///
/// Recency is directory-name order: ids are timestamp-prefixed by
/// `new_id`, so lexicographic order is chronological order — no metadata
/// syscalls, and deterministic regardless of filesystem mtime granularity.
///
/// Only directories are candidates: `sessions/` also holds the `last`
/// pointer and `repl_history.txt`, which must survive.
pub fn prune(sessions_dir: &Path, keep: usize, active: &str) {
    let Ok(entries) = std::fs::read_dir(sessions_dir) else {
        return;
    };
    let mut dirs: Vec<PathBuf> = entries
        .flatten()
        .filter(|entry| entry.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .filter(|entry| entry.file_name() != std::ffi::OsStr::new(active))
        .map(|entry| entry.path())
        .collect();

    // `keep` counts the active session, which is excluded above.
    let keep_others = keep.saturating_sub(1);
    if dirs.len() <= keep_others {
        return;
    }
    dirs.sort_by(|a, b| b.file_name().cmp(&a.file_name()));
    for path in dirs.into_iter().skip(keep_others) {
        let _ = std::fs::remove_dir_all(path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch_session(root: &Path, id: &str) -> PathBuf {
        let dir = root.join(id);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("plan.md"), "1. step\n").unwrap();
        dir
    }

    #[test]
    fn ids_are_unique_per_process_second() {
        assert!(new_id().contains(&std::process::id().to_string()));
    }

    #[test]
    fn last_id_roundtrips() {
        let tmp = tempfile::tempdir().unwrap();
        touch_session(tmp.path(), "s1");
        record_last(tmp.path(), "s1");
        assert_eq!(last_id(tmp.path()).as_deref(), Some("s1"));
    }

    #[test]
    fn last_id_none_when_directory_gone() {
        let tmp = tempfile::tempdir().unwrap();
        record_last(tmp.path(), "vanished");
        assert!(last_id(tmp.path()).is_none());
    }

    #[test]
    fn prune_keeps_newest_and_active() {
        let tmp = tempfile::tempdir().unwrap();
        let dirs: Vec<PathBuf> = (0..5)
            .map(|i| touch_session(tmp.path(), &format!("20260825_00000{i}_1")))
            .collect();
        let active = touch_session(tmp.path(), "20260101_000000_9");

        prune(tmp.path(), 3, "20260101_000000_9");

        assert!(
            active.is_dir(),
            "active session must survive even when oldest"
        );
        assert!(dirs[4].is_dir(), "newest kept");
        assert!(dirs[3].is_dir(), "second newest kept");
        assert!(!dirs[2].is_dir(), "beyond retention pruned");
        assert!(!dirs[0].is_dir(), "oldest pruned");
    }

    #[test]
    fn prune_leaves_non_directories_alone() {
        let tmp = tempfile::tempdir().unwrap();
        let history = tmp.path().join("repl_history.txt");
        std::fs::write(&history, "hi").unwrap();
        record_last(tmp.path(), "s0");
        touch_session(tmp.path(), "s0");

        prune(tmp.path(), 1, "s0");

        assert!(history.is_file(), "repl history must survive pruning");
        assert!(tmp.path().join("last").is_file(), "pointer must survive");
    }
}
