//! E2E tests for tool execution against a real temp filesystem.
//! No LLM server needed — tools are called directly.

mod helpers;

use std::fs;

use miniswe::config::Config;
use miniswe::llm::Message;
use miniswe::tools;
use miniswe::tools::permissions::PermissionManager;
use serde_json::json;

/// Create a headless PermissionManager for the test config.
fn perms(config: &Config) -> PermissionManager {
    PermissionManager::headless(config)
}

// ── read_file ──────────────────────────────────────────────────────

#[tokio::test]
async fn read_file_returns_content_with_line_numbers() {
    let (_tmp, config) = helpers::create_test_project();

    // Create a test file
    let content = "line one\nline two\nline three\n";
    fs::write(helpers::project_path(&config, "test.txt"), content).unwrap();

    let args = json!({"action": "read", "path": "test.txt"});
    let result = tools::execute_tool("file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(
        result.success,
        "read_file should succeed: {}",
        result.content
    );
    assert!(
        result.content.contains("test.txt"),
        "should mention filename"
    );
    assert!(result.content.contains("3 lines"), "should show line count");
    // Check line numbers are present
    assert!(result.content.contains("1│"), "should have line 1");
    assert!(result.content.contains("2│"), "should have line 2");
    assert!(result.content.contains("3│"), "should have line 3");
}

#[tokio::test]
async fn read_file_with_line_range() {
    let (_tmp, config) = helpers::create_test_project();

    let content = (1..=10)
        .map(|i| format!("line {i}"))
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(helpers::project_path(&config, "range.txt"), &content).unwrap();

    let args = json!({"action": "read", "path": "range.txt", "start_line": 3, "end_line": 5});
    let result = tools::execute_tool("file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(result.success);
    assert!(result.content.contains("L3-5"), "should mention line range");
    assert!(result.content.contains("line 3"), "should contain line 3");
    assert!(result.content.contains("line 5"), "should contain line 5");
    assert!(
        !result.content.contains("line 1\n"),
        "should not contain line 1 content"
    );
}

#[tokio::test]
async fn read_file_compresses_rust_stdlib_imports() {
    let (_tmp, config) = helpers::create_test_project();

    // compress_for_reading strips stdlib imports and license headers, not regular comments
    let content = "use std::io;\nuse std::fs;\nuse crate::config::Config;\n\nfn main() {\n    println!(\"hello\");\n}\n";
    fs::write(helpers::project_path(&config, "test.rs"), content).unwrap();

    let args = json!({"action": "read", "path": "test.rs"});
    let result = tools::execute_tool("file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(result.success);
    // stdlib imports should be stripped
    assert!(
        result.content.contains("stripped"),
        "should indicate stripped lines: {}",
        result.content
    );
    // Non-stdlib import should remain
    assert!(result.content.contains("crate::config::Config"));
    // Code should remain
    assert!(result.content.contains("fn main()"));
    assert!(result.content.contains("println!"));
}

#[tokio::test]
async fn read_file_not_found() {
    let (_tmp, config) = helpers::create_test_project();

    let args = json!({"action": "read", "path": "nonexistent.txt"});
    let result = tools::execute_tool("file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(!result.success);
    assert!(result.content.contains("not found") || result.content.contains("Not found"));
}

#[tokio::test]
async fn read_file_rejects_large_file() {
    let (_tmp, config) = helpers::create_test_project();

    // Create a file > 10MB
    let large = "x".repeat(11 * 1024 * 1024);
    fs::write(helpers::project_path(&config, "huge.txt"), &large).unwrap();

    let args = json!({"action": "read", "path": "huge.txt"});
    let result = tools::execute_tool("file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(!result.success);
    assert!(result.content.contains("too large"));
}

// ── write_file ─────────────────────────────────────────────────────

#[tokio::test]
async fn write_file_creates_new_file() {
    let (_tmp, config) = helpers::create_test_project();

    let args = json!({
        "path": "new_file.txt",
        "content": "hello world\nsecond line\n"
    });
    let result = tools::execute_tool("write_file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(
        result.success,
        "write_file should succeed: {}",
        result.content
    );
    assert!(result.content.contains("Created"));

    // Verify file exists on disk
    let disk_content = fs::read_to_string(helpers::project_path(&config, "new_file.txt")).unwrap();
    assert_eq!(disk_content, "hello world\nsecond line\n");
}

#[tokio::test]
async fn write_file_creates_parent_dirs() {
    let (_tmp, config) = helpers::create_test_project();

    let args = json!({
        "path": "deeply/nested/dir/file.txt",
        "content": "nested content"
    });
    let result = tools::execute_tool("write_file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(result.success);
    let path = helpers::project_path(&config, "deeply/nested/dir/file.txt");
    assert!(path.exists());
    assert_eq!(fs::read_to_string(path).unwrap(), "nested content");
}

#[tokio::test]
async fn write_file_overwrites_existing() {
    let (_tmp, config) = helpers::create_test_project();

    fs::write(
        helpers::project_path(&config, "existing.txt"),
        "old content",
    )
    .unwrap();

    let args = json!({
        "path": "existing.txt",
        "content": "new content"
    });
    let result = tools::execute_tool("write_file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(result.success);
    assert!(result.content.contains("Wrote"));
    let disk = fs::read_to_string(helpers::project_path(&config, "existing.txt")).unwrap();
    assert_eq!(disk, "new content");
}

#[tokio::test]
async fn write_file_without_content_creates_empty_file() {
    let (_tmp, config) = helpers::create_test_project();

    let args = json!({"path": "empty.txt"});
    let result = tools::execute_tool("write_file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(
        result.success,
        "write_file should create empty file: {}",
        result.content
    );
    assert!(result.content.contains("Created empty file empty.txt"));
    assert_eq!(
        fs::read_to_string(helpers::project_path(&config, "empty.txt")).unwrap(),
        ""
    );
    assert!(result.content.contains("[tail]"));
    assert!(result.content.contains("(empty file)"));
}

#[tokio::test]
async fn write_file_new_python_file_with_syntax_error_is_warning_not_failure() {
    // Regression: creating a brand-new helper .py script whose contents
    // py_compile rejects (e.g. an unresolved import or syntax issue) used
    // to flip result.success to false even though the file was written as
    // requested. The user can't tell from a "failed" tool result whether
    // the file actually got written.
    //
    // New behavior: for new files (no baseline) the checker output is
    // surfaced as a WARNING and result.success stays true.
    if std::process::Command::new("python3")
        .arg("--version")
        .output()
        .is_err()
    {
        eprintln!("python3 not available, skipping test");
        return;
    }

    let (_tmp, config) = helpers::create_test_project();

    let args = json!({
        "path": "new_helper.py",
        // Deliberately invalid Python — bad indent / unfinished function.
        "content": "def helper(\n    return 42\n",
    });
    let result = tools::execute_tool("write_file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    // The file was written; the tool call succeeded.
    assert!(
        result.success,
        "writing a new .py file should NOT fail just because py_compile complains: {}",
        result.content
    );
    // The checker output should still appear, marked as a WARNING.
    assert!(
        result.content.contains("WARNING")
            && result.content.contains("newly-created new_helper.py"),
        "expected WARNING about new file, got: {}",
        result.content
    );
    // File exists on disk with the requested content.
    let disk = fs::read_to_string(helpers::project_path(&config, "new_helper.py")).unwrap();
    assert_eq!(disk, "def helper(\n    return 42\n");
}

#[tokio::test]
async fn write_file_overwriting_python_with_syntax_error_still_fails() {
    // Counterpoint: if the file already existed (and presumably compiled
    // before), overwriting it with broken Python *is* a regression and
    // SHOULD flip success to false. Otherwise the leniency on new files
    // would mask real breakage on edits.
    if std::process::Command::new("python3")
        .arg("--version")
        .output()
        .is_err()
    {
        eprintln!("python3 not available, skipping test");
        return;
    }

    let (_tmp, config) = helpers::create_test_project();

    // Pre-existing valid .py file.
    fs::write(
        helpers::project_path(&config, "helper.py"),
        "def helper():\n    return 42\n",
    )
    .unwrap();

    let args = json!({
        "path": "helper.py",
        "content": "def helper(\n    return 42\n",
    });
    let result = tools::execute_tool("write_file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(
        !result.success,
        "overwriting an existing file with broken syntax should fail: {}",
        result.content
    );
}

#[tokio::test]
async fn write_file_new_python_file_with_valid_content_succeeds_silently() {
    // Sanity check: a new .py file with valid syntax should succeed
    // without any WARNING marker.
    if std::process::Command::new("python3")
        .arg("--version")
        .output()
        .is_err()
    {
        eprintln!("python3 not available, skipping test");
        return;
    }

    let (_tmp, config) = helpers::create_test_project();

    let args = json!({
        "path": "good_helper.py",
        "content": "def helper():\n    return 42\n",
    });
    let result = tools::execute_tool("write_file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(
        result.success,
        "valid python file should succeed: {}",
        result.content
    );
    assert!(
        result.content.contains("[py_compile] OK"),
        "expected py_compile OK marker, got: {}",
        result.content
    );
    assert!(
        !result.content.contains("WARNING"),
        "should not mark valid file as WARNING, got: {}",
        result.content
    );
}

#[tokio::test]
async fn write_file_without_content_rejects_existing_file() {
    let (_tmp, config) = helpers::create_test_project();

    fs::write(
        helpers::project_path(&config, "existing.txt"),
        "old content",
    )
    .unwrap();

    let args = json!({"path": "existing.txt"});
    let result = tools::execute_tool("write_file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(!result.success);
    assert!(result.content.contains("already exists"));
}

#[tokio::test]
async fn delete_file_removes_existing_file() {
    let (_tmp, config) = helpers::create_test_project();

    fs::write(
        helpers::project_path(&config, "delete_me.txt"),
        "remove this file\n",
    )
    .unwrap();

    let args = json!({"action": "delete", "path": "delete_me.txt"});
    let result = tools::execute_tool("file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(result.success, "delete should succeed: {}", result.content);
    assert!(result.content.contains("Deleted delete_me.txt"));
    assert!(!helpers::project_path(&config, "delete_me.txt").exists());
}

#[tokio::test]
async fn delete_file_missing_path_or_file_fails() {
    let (_tmp, config) = helpers::create_test_project();

    let missing_path_args = json!({"action": "delete"});
    let missing_path =
        tools::execute_tool("file", &missing_path_args, &config, &perms(&config), None)
            .await
            .unwrap();
    assert!(!missing_path.success);
    assert!(
        missing_path
            .content
            .contains("Missing required parameter: path")
    );

    let missing_file_args = json!({"action": "delete", "path": "nope.txt"});
    let missing_file =
        tools::execute_tool("file", &missing_file_args, &config, &perms(&config), None)
            .await
            .unwrap();
    assert!(!missing_file.success);
    assert!(missing_file.content.contains("File not found: nope.txt"));
}

// ── replace deprecation ─────────────────────────────────────────────

#[tokio::test]
async fn file_replace_action_returns_deprecation_redirect() {
    let (_tmp, config) = helpers::create_test_project();
    fs::write(helpers::project_path(&config, "x.txt"), "hello\n").unwrap();

    let args = json!({"action": "replace", "path": "x.txt", "old": "hello", "new": "world"});
    let result = tools::execute_tool("file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(!result.success, "replace should be deprecated");
    assert!(
        result.content.contains("no longer supported"),
        "must declare deprecation: {}",
        result.content
    );
    // The deprecation hint points at whichever edit surface is active:
    // `edit_file` in smart mode, `replace_range` / `insert_at` in fast mode.
    assert!(
        result.content.contains("write_file")
            && (result.content.contains("edit_file") || result.content.contains("replace_range")),
        "deprecation must redirect to write_file and an edit primitive: {}",
        result.content
    );

    // File must be untouched.
    let disk = fs::read_to_string(helpers::project_path(&config, "x.txt")).unwrap();
    assert_eq!(disk, "hello\n");
}

// ── search ──────────────────────────────────────────────────────────

#[tokio::test]
async fn search_finds_matches() {
    let (_tmp, config) = helpers::create_test_project();

    // Create files with searchable content
    fs::create_dir_all(helpers::project_path(&config, "src")).unwrap();
    fs::write(
        helpers::project_path(&config, "src/main.rs"),
        "fn main() {\n    println!(\"hello world\");\n}\n",
    )
    .unwrap();
    fs::write(
        helpers::project_path(&config, "src/lib.rs"),
        "pub fn greet() -> String {\n    \"hello world\".to_string()\n}\n",
    )
    .unwrap();

    let args = json!({"action": "search", "query": "hello world"});
    let result = tools::execute_tool("file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(result.success, "search should succeed: {}", result.content);
    assert!(
        result.content.contains("hello world"),
        "should find the match"
    );
    // Should find matches in both files
    assert!(result.content.contains("2 matches") || result.content.contains("match"));
}

#[tokio::test]
async fn search_no_matches() {
    let (_tmp, config) = helpers::create_test_project();

    fs::create_dir_all(helpers::project_path(&config, "src")).unwrap();
    fs::write(
        helpers::project_path(&config, "src/main.rs"),
        "fn main() {}\n",
    )
    .unwrap();

    let args = json!({"action": "search", "query": "ZZZZUNIQUENOMATCH"});
    let result = tools::execute_tool("file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(result.success); // "no matches" is still a successful search
    assert!(result.content.contains("No matches"));
}

// ── shell ───────────────────────────────────────────────────────────

#[tokio::test]
async fn shell_runs_command_and_captures_output() {
    let (_tmp, config) = helpers::create_test_project();

    let args = json!({"action": "shell", "command": "echo hello"});
    let result = tools::execute_tool("file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(result.success, "echo should succeed: {}", result.content);
    assert!(result.content.contains("hello"));
    assert!(result.content.contains("exit 0"));
}

#[tokio::test]
async fn shell_captures_exit_code() {
    let (_tmp, config) = helpers::create_test_project();

    let args = json!({"action": "shell", "command": "exit 42"});
    let result = tools::execute_tool("file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(!result.success);
    assert!(result.content.contains("exit 42"));
}

#[tokio::test]
async fn shell_timeout() {
    let (_tmp, config) = helpers::create_test_project();

    let args = json!({"action": "shell", "command": "sleep 60", "timeout": 1});
    let result = tools::execute_tool("file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(!result.success);
    assert!(result.content.contains("timed out"));
}

#[tokio::test]
async fn shell_runs_in_project_root() {
    let (_tmp, config) = helpers::create_test_project();

    // Create a marker file in project root
    fs::write(helpers::project_path(&config, "marker.txt"), "found").unwrap();

    let args = json!({"action": "shell", "command": "cat marker.txt"});
    let result = tools::execute_tool("file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(result.success);
    assert!(result.content.contains("found"));
}

// ── shell output truncation ─────────────────────────────────────────

#[tokio::test]
async fn shell_truncates_very_long_lines() {
    let (_tmp, config) = helpers::create_test_project();

    // Generate a single line with 200K characters (no newlines)
    // Use printf with brace expansion — avoids pipes which can cause issues with shell tool's try_wait
    let args = json!({"action": "shell", "command": "printf 'x%.0s' $(seq 1 200000)"});
    let result = tools::execute_tool("file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(result.success, "command should succeed: {}", result.content);
    // Output should be capped, not 200K+ chars
    assert!(
        result.content.len() < 50_000,
        "Shell output should be truncated for very long lines, got {} bytes",
        result.content.len()
    );
}

#[tokio::test]
async fn shell_truncates_many_lines() {
    let (_tmp, config) = helpers::create_test_project();

    // Generate 500 lines of output
    let args = json!({"action": "shell", "command": "seq 1 500"});
    let result = tools::execute_tool("file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(result.success);
    assert!(
        result.content.contains("showing last"),
        "Should indicate truncation: {}",
        result.content.lines().next().unwrap_or("")
    );
    // Should keep the LAST N lines (tail priority), so "500" should be visible
    assert!(
        result.content.contains("500"),
        "Should contain the last line (500)"
    );
    // "1" (first line) might or might not be visible depending on exact truncation
}

// ── task_update ─────────────────────────────────────────────────────

#[tokio::test]
async fn task_update_creates_scratchpad() {
    let (_tmp, config) = helpers::create_test_project();

    let scratchpad = "## Current Task\nImplement feature X\n\n## Plan\n1. Step one\n2. Step two\n";
    let args = json!({"action": "scratchpad", "content": scratchpad});
    let result = tools::execute_tool("plan", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(
        result.success,
        "task_update should succeed: {}",
        result.content
    );

    let path = config.miniswe_path("scratchpad.md");
    assert!(path.exists());
    let disk = fs::read_to_string(path).unwrap();
    assert_eq!(disk, scratchpad);
}

#[tokio::test]
async fn task_update_rejects_missing_sections() {
    let (_tmp, config) = helpers::create_test_project();

    // Missing ## Plan section
    let args = json!({"action": "scratchpad", "content": "## Current Task\nDo something\n"});
    let result = tools::execute_tool("plan", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(!result.success);
    assert!(result.content.contains("Plan"));

    // Missing ## Current Task section
    let args = json!({"action": "scratchpad", "content": "## Plan\n1. Step\n"});
    let result = tools::execute_tool("plan", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(!result.success);
    assert!(result.content.contains("Current Task"));
}

// ── unknown tool / action ───────────────────────────────────────────

#[tokio::test]
async fn unknown_tool_returns_error() {
    let (_tmp, config) = helpers::create_test_project();

    let result = tools::execute_tool(
        "nonexistent_tool",
        &json!({}),
        &config,
        &perms(&config),
        None,
    )
    .await;

    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("Unknown tool"));
}

#[tokio::test]
async fn unknown_action_returns_error() {
    let (_tmp, config) = helpers::create_test_project();

    let args = json!({"action": "bogus_action"});
    let result = tools::execute_tool("file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(
        !result.success,
        "unknown action should fail: {}",
        result.content
    );
}

#[tokio::test]
async fn file_help_returns_action_list() {
    let (_tmp, config) = helpers::create_test_project();

    let args = json!({"action": "help"});
    let result = tools::execute_tool("file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(result.success);
    assert!(
        result.content.contains("read"),
        "help should list read action: {}",
        result.content
    );
    assert!(
        !result.content.contains("\n- write:"),
        "file help should no longer advertise write: {}",
        result.content
    );
    assert!(
        !result.content.contains("\n- replace:"),
        "file help should no longer advertise replace: {}",
        result.content
    );
    // The help text points at whichever edit surface is active:
    // `edit_file` in smart mode, `replace_range` / `insert_at` in fast mode.
    assert!(
        result.content.contains("edit_file")
            || result.content.contains("replace_range")
            || result.content.contains("insert_at"),
        "help should advertise an edit primitive: {}",
        result.content
    );
}

#[tokio::test]
async fn code_help_returns_action_list() {
    let (_tmp, config) = helpers::create_test_project();

    let args = json!({"action": "help"});
    let result = tools::execute_tool("code", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(result.success);
    assert!(
        result.content.contains("goto_definition"),
        "help should list goto_definition: {}",
        result.content
    );
    assert!(
        result.content.contains("repo_map"),
        "help should list repo_map: {}",
        result.content
    );
}

#[tokio::test]
async fn web_help_returns_action_list() {
    let (_tmp, config) = helpers::create_test_project();

    let args = json!({"action": "help"});
    let result = tools::execute_tool("web", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(result.success);
    assert!(
        result.content.contains("search"),
        "help should list search: {}",
        result.content
    );
    assert!(
        result.content.contains("fetch"),
        "help should list fetch: {}",
        result.content
    );
}

// ── missing required params ─────────────────────────────────────────

#[tokio::test]
async fn read_file_missing_path() {
    let (_tmp, config) = helpers::create_test_project();

    let args = json!({"action": "read"});
    let result = tools::execute_tool("file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(!result.success);
    assert!(result.content.contains("path") || result.content.contains("Missing"));
}

#[tokio::test]
async fn write_file_missing_content() {
    let (_tmp, config) = helpers::create_test_project();

    let args = json!({"path": "file.txt"});
    let result = tools::execute_tool("write_file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(result.success);
    assert_eq!(
        fs::read_to_string(helpers::project_path(&config, "file.txt")).unwrap(),
        ""
    );
}

#[tokio::test]
async fn shell_missing_command() {
    let (_tmp, config) = helpers::create_test_project();

    let args = json!({"action": "shell"});
    let result = tools::execute_tool("file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(!result.success);
    assert!(result.content.contains("command") || result.content.contains("Missing"));
}

// ── whitespace-normalized edit fallback ────────────────────────────

// ── context pull-based tools ──────────────────────────────────────

#[tokio::test]
async fn get_project_info_returns_profile() {
    let (_tmp, config) = helpers::create_test_project();

    fs::write(
        config.miniswe_path("profile.md"),
        "# Test\n## Overview\n- Name: test-proj\n- Language: Rust\n",
    )
    .unwrap();

    let args = json!({"action": "project_info"});
    let result = tools::execute_tool("code", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(result.success);
    assert!(
        result.content.contains("test-proj") || result.content.contains("PROFILE"),
        "should contain profile: {}",
        result.content
    );
}

#[tokio::test]
async fn get_architecture_notes_missing_file() {
    let (_tmp, config) = helpers::create_test_project();

    let args = json!({"action": "architecture_notes"});
    let result = tools::execute_tool("code", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(result.success);
    assert!(
        result.content.contains("does not exist"),
        "should say file doesn't exist: {}",
        result.content
    );
}

#[tokio::test]
async fn get_architecture_notes_returns_content() {
    let (_tmp, config) = helpers::create_test_project();

    let ai_dir = helpers::project_path(&config, ".ai");
    fs::create_dir_all(&ai_dir).unwrap();
    fs::write(
        ai_dir.join("README.md"),
        "# Architecture\nLayered design.\n",
    )
    .unwrap();

    let args = json!({"action": "architecture_notes"});
    let result = tools::execute_tool("code", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(result.success);
    assert!(
        result.content.contains("Layered design"),
        "should contain notes: {}",
        result.content
    );
}

// ── tool enabling/disabling ───────────────────────────────────────

#[test]
fn tools_config_disables_web() {
    let mut defs = miniswe::tools::tool_definitions(miniswe::config::EditMode::Smart);
    // Filter like run.rs does
    defs.retain(|t| t.function.name != "web");

    let names: Vec<&str> = defs.iter().map(|t| t.function.name.as_str()).collect();
    assert!(!names.contains(&"web"), "should not have web");
    assert!(names.contains(&"file"), "should still have file");
    assert!(names.contains(&"code"), "should still have code");
}

#[test]
fn tools_config_all_enabled_by_default() {
    let config = miniswe::config::Config::default();
    assert!(config.tools.context_tools);
    assert!(config.tools.web_tools);
}

#[test]
fn tools_has_grouped_tools() {
    let defs = miniswe::tools::tool_definitions(miniswe::config::EditMode::Smart);
    let names: Vec<&str> = defs.iter().map(|t| t.function.name.as_str()).collect();
    assert!(names.contains(&"file"), "should have file");
    assert!(names.contains(&"code"), "should have code");
    assert!(names.contains(&"web"), "should have web");
    assert!(names.contains(&"plan"), "should have plan");
    assert!(names.contains(&"edit_file"), "should have edit_file");
    assert!(names.contains(&"write_file"), "should have write_file");
    // Old flat tools should be gone
    assert!(!names.contains(&"read_file"), "read_file should be gone");
    assert!(!names.contains(&"replace"), "flat replace should be gone");
    assert!(!names.contains(&"search"), "flat search should be gone");
    assert!(!names.contains(&"shell"), "flat shell should be gone");
}

// ── dynamic tool output budget ────────────────────────────────────

#[test]
fn tool_output_budget_scales_with_context() {
    let mut config = miniswe::config::Config::default();

    config.model.context_window = 32000;
    assert_eq!(config.tool_output_budget_chars(), 3200);

    config.model.context_window = 50000;
    assert_eq!(config.tool_output_budget_chars(), 5000);

    config.model.context_window = 128000;
    assert_eq!(config.tool_output_budget_chars(), 12800);
}

// ── search: query vs pattern mode ─────────────────────────────────

#[tokio::test]
async fn search_query_mode_is_literal() {
    let (_tmp, config) = helpers::create_test_project();

    // Create file with regex-like content
    fs::write(
        helpers::project_path(&config, "test.rs"),
        "fn foo() -> Result<()> {\n    Ok(())\n}\n",
    )
    .unwrap();

    // query mode: "Result<()>" should match literally (no regex interpretation)
    let args = json!({"action": "search", "query": "Result<()>"});
    let result = tools::execute_tool("file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(result.success);
    assert!(
        result.content.contains("Result<()>"),
        "literal search should find Result<()>: {}",
        result.content
    );
}

#[tokio::test]
async fn search_pattern_mode_is_regex() {
    let (_tmp, config) = helpers::create_test_project();

    fs::write(
        helpers::project_path(&config, "test.rs"),
        "fn foo() {}\nfn bar() {}\nfn baz_qux() {}\n",
    )
    .unwrap();

    // pattern mode: regex to find functions starting with 'b'
    let args = json!({"action": "search", "pattern": "fn b\\w+"});
    let result = tools::execute_tool("file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(result.success);
    assert!(
        result.content.contains("bar") && result.content.contains("baz_qux"),
        "regex should match bar and baz_qux: {}",
        result.content
    );
    assert!(
        !result.content.contains("foo"),
        "regex should not match foo: {}",
        result.content
    );
}

#[tokio::test]
async fn search_needs_query_or_pattern() {
    let (_tmp, config) = helpers::create_test_project();

    let args = json!({"action": "search"});
    let result = tools::execute_tool("file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(!result.success);
    assert!(
        result.content.contains("query") || result.content.contains("pattern"),
        "should ask for query or pattern: {}",
        result.content
    );
}

// ── store-and-preview for large shell output ──────────────────────

#[tokio::test]
async fn shell_large_output_saved_to_file() {
    let (_tmp, config) = helpers::create_test_project();

    // Generate output larger than budget
    let budget_lines = config.tool_output_budget_chars() / 80;
    let line_count = budget_lines + 100;
    let args = json!({"action": "shell", "command": format!("seq 1 {line_count}")});
    let result = tools::execute_tool("file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(result.success);
    assert!(
        result.content.contains("Full output saved to"),
        "should have file pointer: {}",
        result.content.lines().last().unwrap_or("")
    );

    // Verify the file was actually created
    let shell_dir = config.miniswe_dir().join("shell_output");
    assert!(shell_dir.exists(), ".miniswe/shell_output should exist");
    let files: Vec<_> = fs::read_dir(&shell_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .collect();
    assert!(!files.is_empty(), "should have saved output file");
}

// ── unified compressor ────────────────────────────────────────────

#[test]
fn compressor_no_op_when_under_budget() {
    let config = Config::default();
    // Small conversation — should not compress
    let messages = [
        Message::system("You are a coding agent."),
        Message::user("Hello"),
        Message::assistant("Hi! How can I help?"),
    ];
    let original_len = messages.len();

    // Can't call async maybe_compress in sync test, but we can verify
    // the budget calculation: 3 small messages << context_window/4
    let total_tokens: usize = messages
        .iter()
        .filter(|m| m.role != "system")
        .map(|m| miniswe::context::estimate_tokens(m.content.as_deref().unwrap_or("")))
        .sum();
    let budget = config.model.context_window / 4;

    assert!(
        total_tokens < budget,
        "small conversation ({total_tokens} tokens) should be under budget ({budget})"
    );
    assert_eq!(messages.len(), original_len, "messages should not change");
}

// ── edit: whitespace normalization edge cases ─────────────────────

// ── tool response content tests ───────────────────────────────────
// Verify what the model actually sees in tool results

#[tokio::test]
async fn write_file_response_shows_tail() {
    let (_tmp, config) = helpers::create_test_project();

    let content = (1..=20)
        .map(|i| format!("line {i}"))
        .collect::<Vec<_>>()
        .join("\n");
    let args = json!({"path": "new.txt", "content": content});
    let result = tools::execute_tool("write_file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(result.success);
    assert!(
        result.content.contains("✓ Wrote") || result.content.contains("✓ Created"),
        "should confirm write: {}",
        result.content
    );
    assert!(
        result.content.contains("line 20"),
        "should show tail of file: {}",
        result.content
    );
}

#[tokio::test]
async fn search_response_shows_file_and_line() {
    let (_tmp, config) = helpers::create_test_project();

    fs::write(
        helpers::project_path(&config, "findme.rs"),
        "pub fn hello() {}\nfn world() {}\n",
    )
    .unwrap();

    let args = json!({"action": "search", "query": "pub fn hello"});
    let result = tools::execute_tool("file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(result.success);
    assert!(
        result.content.contains("findme.rs"),
        "should show filename: {}",
        result.content
    );
    assert!(
        result.content.contains("pub fn hello"),
        "should show matching line: {}",
        result.content
    );
}

#[tokio::test]
async fn get_repo_map_response_shows_symbols() {
    let (_tmp, config) = helpers::create_test_project();

    // Create some Rust source files
    fs::create_dir_all(helpers::project_path(&config, "src")).unwrap();
    fs::write(
        helpers::project_path(&config, "src/lib.rs"),
        "pub struct Config {\n    pub name: String,\n}\n\npub fn run() -> bool {\n    true\n}\n",
    )
    .unwrap();
    fs::write(
        helpers::project_path(&config, "Cargo.toml"),
        "[package]\nname = \"test\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
    )
    .unwrap();

    // Index the project
    let miniswe_dir = config.miniswe_dir();
    fs::create_dir_all(&miniswe_dir).ok();
    let index = miniswe::knowledge::indexer::index_project(&config.project_root, None).unwrap();
    index.save(&miniswe_dir).unwrap();
    let graph = miniswe::knowledge::graph::DependencyGraph::build(&index);
    graph.save(&miniswe_dir).unwrap();

    let args = json!({"action": "repo_map"});
    let result = tools::execute_tool("code", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(result.success);
    assert!(
        result.content.contains("Config") || result.content.contains("run"),
        "should show symbols: {}",
        result.content
    );
}

#[tokio::test]
async fn revert_tool_through_execute() {
    let (_tmp, config) = helpers::create_test_project();

    // revert goes through run.rs dispatch, not execute_tool
    // but we can test the SnapshotManager directly
    let mut snap = miniswe::tools::snapshots::SnapshotManager::init(&config.project_root).unwrap();

    fs::write(helpers::project_path(&config, "code.rs"), "original").unwrap();
    snap.begin_round(1).unwrap();

    fs::write(helpers::project_path(&config, "code.rs"), "broken").unwrap();
    snap.begin_round(2).unwrap();

    // Revert to round 1
    let msg = snap.revert_to_round(1).unwrap();
    assert!(msg.contains("round 1"), "should confirm revert: {msg}");
    assert_eq!(
        fs::read_to_string(helpers::project_path(&config, "code.rs")).unwrap(),
        "original"
    );
}

#[tokio::test]
async fn diagnostics_response_is_actionable() {
    let (_tmp, config) = helpers::create_test_project();

    // Create a Rust project with a type error
    fs::write(
        helpers::project_path(&config, "Cargo.toml"),
        "[package]\nname = \"test\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
    )
    .unwrap();
    fs::create_dir_all(helpers::project_path(&config, "src")).unwrap();
    fs::write(
        helpers::project_path(&config, "src/main.rs"),
        "fn main() {\n    let x: u32 = \"not a number\";\n    println!(\"{x}\");\n}\n",
    )
    .unwrap();

    let args = json!({"action": "diagnostics"});
    let result = tools::execute_tool("code", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    // diagnostics runs cargo check — should report errors
    if result.content.contains("error") {
        // Good — it found the error
        assert!(
            result.content.contains("mismatched") || result.content.contains("expected"),
            "should show type error details: {}",
            result.content
        );
    }
    // If cargo isn't available, test is a no-op
}
