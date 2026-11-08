//! Single test on purpose: it rewires the process's fd 2, so it must not
//! share a binary with other tests.

#![cfg(unix)]

use std::io::Write;

use buzztr::cli::commands::repl::StderrRedirect;

/// Write straight to fd 2. (`eprintln!` goes through libtest's output
/// capture and would never reach the descriptor.)
fn raw_stderr(msg: &str) {
    let mut err = std::io::stderr();
    let _ = writeln!(err, "{msg}");
    let _ = err.flush();
}

#[test]
fn stderr_goes_to_file_while_guard_lives_and_not_after() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("logs/stderr.log");

    let guard = StderrRedirect::to_file(&path).unwrap();
    raw_stderr("MARKER-inside");
    drop(guard);
    let after_drop = std::fs::read_to_string(&path).unwrap();
    assert!(after_drop.contains("MARKER-inside"), "got: {after_drop:?}");

    raw_stderr("MARKER-outside");
    let later = std::fs::read_to_string(&path).unwrap();
    assert!(!later.contains("MARKER-outside"), "got: {later:?}");
}
