//! Tests for fix_file tool — edit instruction parsing and application.
//!
//! The LLM call is mocked by testing the parse/apply logic directly.
//! Full integration tests happen in benchmarks.

mod helpers;

use miniswe::tools::fix_file;

// ── Edit parsing ──────────────────────────────────────────────────

#[test]
fn parse_single_edit() {
    let input = "EDIT 42\nOLD: let x = 1;\nNEW: let x = 42;\n";
    let edits = fix_file::parse_edits(input);
    assert_eq!(edits.len(), 1);
    assert_eq!(edits[0].line, 42);
    assert_eq!(edits[0].old, "let x = 1;");
    assert_eq!(edits[0].new, "let x = 42;");
}

#[test]
fn parse_multiple_edits() {
    let input = "\
EDIT 10
OLD: foo(a, b)
NEW: foo(a, b, c)

EDIT 25
OLD: bar(x)
NEW: bar(x, y)
";
    let edits = fix_file::parse_edits(input);
    assert_eq!(edits.len(), 2);
    assert_eq!(edits[0].line, 10);
    assert_eq!(edits[1].line, 25);
}

#[test]
fn parse_no_changes() {
    let input = "NO_CHANGES\n";
    let edits = fix_file::parse_edits(input);
    assert!(edits.is_empty());
}

#[test]
fn parse_with_preamble_text() {
    // LLM sometimes adds explanation before edits
    let input = "Here are the changes:\n\nEDIT 5\nOLD: return false;\nNEW: return true;\n";
    let edits = fix_file::parse_edits(input);
    assert_eq!(edits.len(), 1);
    assert_eq!(edits[0].line, 5);
}

#[test]
fn parse_with_extra_whitespace() {
    let input = "EDIT 15\nOLD:   let y = 2;  \nNEW:   let y = 99;  \n";
    let edits = fix_file::parse_edits(input);
    assert_eq!(edits.len(), 1);
    assert_eq!(edits[0].old, "let y = 2;");
    assert_eq!(edits[0].new, "let y = 99;");
}

// ── Window building ───────────────────────────────────────────────

#[test]
fn single_window_for_small_file() {
    let windows = fix_file::build_windows(100, 800, 100);
    assert_eq!(windows, vec![(0, 100)]);
}

#[test]
fn multiple_windows_for_large_file() {
    let windows = fix_file::build_windows(2000, 800, 100);
    assert!(windows.len() >= 3, "should need 3+ windows: {:?}", windows);
    // First window starts at 0
    assert_eq!(windows[0].0, 0);
    // Last window ends at total
    assert_eq!(windows.last().unwrap().1, 2000);
    // Windows overlap
    assert!(windows[1].0 < windows[0].1, "windows should overlap");
}

#[test]
fn windows_cover_entire_file() {
    let windows = fix_file::build_windows(1500, 800, 100);
    // Every line should be in at least one window
    for line in 0..1500 {
        let covered = windows.iter().any(|(s, e)| line >= *s && line < *e);
        assert!(covered, "line {line} not covered by any window");
    }
}

// ── Missing params ────────────────────────────────────────────────

#[tokio::test]
async fn fix_file_missing_path() {
    let (_tmp, config) = helpers::create_test_project();
    let router = miniswe::llm::ModelRouter::new(&config);

    let args = serde_json::json!({"task": "do something"});
    let result = fix_file::execute(&args, &config, &router).await.unwrap();
    assert!(!result.success);
    assert!(result.content.contains("path"));
}

#[tokio::test]
async fn fix_file_missing_task() {
    let (_tmp, config) = helpers::create_test_project();
    let router = miniswe::llm::ModelRouter::new(&config);

    let args = serde_json::json!({"path": "test.rs"});
    let result = fix_file::execute(&args, &config, &router).await.unwrap();
    assert!(!result.success);
    assert!(result.content.contains("task"));
}

#[tokio::test]
async fn fix_file_not_found() {
    let (_tmp, config) = helpers::create_test_project();
    let router = miniswe::llm::ModelRouter::new(&config);

    let args = serde_json::json!({"path": "nonexistent.rs", "task": "fix it"});
    let result = fix_file::execute(&args, &config, &router).await.unwrap();
    assert!(!result.success);
    assert!(result.content.contains("not found"));
}
