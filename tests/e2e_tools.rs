//! E2E tests for tool execution against a real temp filesystem.
//! No LLM server needed — tools are called directly.

mod helpers;

use std::fs;

use miniswe::config::Config;
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

    let args = json!({"path": "test.txt"});
    let result = tools::execute_tool("read_file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(result.success, "read_file should succeed: {}", result.content);
    assert!(result.content.contains("test.txt"), "should mention filename");
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

    let args = json!({"path": "range.txt", "start_line": 3, "end_line": 5});
    let result = tools::execute_tool("read_file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(result.success);
    assert!(result.content.contains("L3-5"), "should mention line range");
    assert!(result.content.contains("line 3"), "should contain line 3");
    assert!(result.content.contains("line 5"), "should contain line 5");
    assert!(!result.content.contains("line 1\n"), "should not contain line 1 content");
}

#[tokio::test]
async fn read_file_compresses_rust_stdlib_imports() {
    let (_tmp, config) = helpers::create_test_project();

    // compress_for_reading strips stdlib imports and license headers, not regular comments
    let content = "use std::io;\nuse std::fs;\nuse crate::config::Config;\n\nfn main() {\n    println!(\"hello\");\n}\n";
    fs::write(helpers::project_path(&config, "test.rs"), content).unwrap();

    let args = json!({"path": "test.rs"});
    let result = tools::execute_tool("read_file", &args, &config, &perms(&config), None)
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

    let args = json!({"path": "nonexistent.txt"});
    let result = tools::execute_tool("read_file", &args, &config, &perms(&config), None)
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

    let args = json!({"path": "huge.txt"});
    let result = tools::execute_tool("read_file", &args, &config, &perms(&config), None)
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

    assert!(result.success, "write_file should succeed: {}", result.content);
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

    fs::write(helpers::project_path(&config, "existing.txt"), "old content").unwrap();

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

// ── edit ────────────────────────────────────────────────────────────

#[tokio::test]
async fn edit_performs_replacement() {
    let (_tmp, config) = helpers::create_test_project();

    let content = "fn main() {\n    let x = 1;\n    let y = 2;\n    println!(\"{}\", x + y);\n}\n";
    fs::write(helpers::project_path(&config, "edit_test.txt"), content).unwrap();

    let args = json!({
        "path": "edit_test.txt",
        "old": "    let x = 1;",
        "new": "    let x = 42;"
    });
    let result = tools::execute_tool("edit", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(result.success, "edit should succeed: {}", result.content);
    assert!(result.content.contains("1 replacement"));

    let disk = fs::read_to_string(helpers::project_path(&config, "edit_test.txt")).unwrap();
    assert!(disk.contains("let x = 42;"));
    assert!(!disk.contains("let x = 1;"));
}

#[tokio::test]
async fn edit_rejects_ambiguous_match() {
    let (_tmp, config) = helpers::create_test_project();

    let content = "foo\nbar\nfoo\nbaz\n";
    fs::write(helpers::project_path(&config, "dupe.txt"), content).unwrap();

    let args = json!({
        "path": "dupe.txt",
        "old": "foo",
        "new": "qux"
    });
    let result = tools::execute_tool("edit", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(!result.success);
    assert!(
        result.content.contains("matches 2 locations"),
        "should report ambiguity: {}",
        result.content
    );
    // Should show line numbers of matches
    assert!(result.content.contains("L1"));
    assert!(result.content.contains("L3"));
}

#[tokio::test]
async fn edit_file_not_found() {
    let (_tmp, config) = helpers::create_test_project();

    let args = json!({
        "path": "no_such_file.txt",
        "old": "foo",
        "new": "bar"
    });
    let result = tools::execute_tool("edit", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(!result.success);
    assert!(result.content.contains("not found") || result.content.contains("Not found"));
}

#[tokio::test]
async fn edit_old_not_found_in_file() {
    let (_tmp, config) = helpers::create_test_project();

    fs::write(helpers::project_path(&config, "miss.txt"), "actual content\n").unwrap();

    let args = json!({
        "path": "miss.txt",
        "old": "nonexistent text",
        "new": "replacement"
    });
    let result = tools::execute_tool("edit", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(!result.success);
    assert!(result.content.contains("not found"));
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

    let args = json!({"query": "hello world"});
    let result = tools::execute_tool("search", &args, &config, &perms(&config), None)
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
    fs::write(helpers::project_path(&config, "src/main.rs"), "fn main() {}\n").unwrap();

    let args = json!({"query": "ZZZZUNIQUENOMATCH"});
    let result = tools::execute_tool("search", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(result.success); // "no matches" is still a successful search
    assert!(result.content.contains("No matches"));
}

// ── shell ───────────────────────────────────────────────────────────

#[tokio::test]
async fn shell_runs_command_and_captures_output() {
    let (_tmp, config) = helpers::create_test_project();

    let args = json!({"command": "echo hello"});
    let result = tools::execute_tool("shell", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(result.success, "echo should succeed: {}", result.content);
    assert!(result.content.contains("hello"));
    assert!(result.content.contains("exit 0"));
}

#[tokio::test]
async fn shell_captures_exit_code() {
    let (_tmp, config) = helpers::create_test_project();

    let args = json!({"command": "exit 42"});
    let result = tools::execute_tool("shell", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(!result.success);
    assert!(result.content.contains("exit 42"));
}

#[tokio::test]
async fn shell_timeout() {
    let (_tmp, config) = helpers::create_test_project();

    let args = json!({"command": "sleep 60", "timeout": 1});
    let result = tools::execute_tool("shell", &args, &config, &perms(&config), None)
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

    let args = json!({"command": "cat marker.txt"});
    let result = tools::execute_tool("shell", &args, &config, &perms(&config), None)
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
    let args = json!({"command": "printf 'x%.0s' $(seq 1 200000)"});
    let result = tools::execute_tool("shell", &args, &config, &perms(&config), None)
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
    let args = json!({"command": "seq 1 500"});
    let result = tools::execute_tool("shell", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(result.success);
    assert!(
        result.content.contains("showing last 100"),
        "Should indicate truncation: {}",
        result.content.lines().next().unwrap_or("")
    );
    // Should keep the LAST 100 lines (tail priority), so "500" should be visible
    assert!(result.content.contains("500"), "Should contain the last line (500)");
    // "1" (first line) might or might not be visible depending on exact truncation
}

// ── task_update ─────────────────────────────────────────────────────

#[tokio::test]
async fn task_update_creates_scratchpad() {
    let (_tmp, config) = helpers::create_test_project();

    let scratchpad = "## Current Task\nImplement feature X\n\n## Plan\n1. Step one\n2. Step two\n";
    let args = json!({"content": scratchpad});
    let result = tools::execute_tool("task_update", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(result.success, "task_update should succeed: {}", result.content);

    let path = config.miniswe_path("scratchpad.md");
    assert!(path.exists());
    let disk = fs::read_to_string(path).unwrap();
    assert_eq!(disk, scratchpad);
}

#[tokio::test]
async fn task_update_rejects_missing_sections() {
    let (_tmp, config) = helpers::create_test_project();

    // Missing ## Plan section
    let args = json!({"content": "## Current Task\nDo something\n"});
    let result = tools::execute_tool("task_update", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(!result.success);
    assert!(result.content.contains("Plan"));

    // Missing ## Current Task section
    let args = json!({"content": "## Plan\n1. Step\n"});
    let result = tools::execute_tool("task_update", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(!result.success);
    assert!(result.content.contains("Current Task"));
}

// ── unknown tool ────────────────────────────────────────────────────

#[tokio::test]
async fn unknown_tool_returns_error() {
    let (_tmp, config) = helpers::create_test_project();

    let result = tools::execute_tool("nonexistent_tool", &json!({}), &config, &perms(&config), None).await;

    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("Unknown tool"));
}

// ── missing required params ─────────────────────────────────────────

#[tokio::test]
async fn read_file_missing_path() {
    let (_tmp, config) = helpers::create_test_project();

    let args = json!({});
    let result = tools::execute_tool("read_file", &args, &config, &perms(&config), None)
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

    assert!(!result.success);
    assert!(result.content.contains("content") || result.content.contains("Missing"));
}

#[tokio::test]
async fn shell_missing_command() {
    let (_tmp, config) = helpers::create_test_project();

    let args = json!({});
    let result = tools::execute_tool("shell", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(!result.success);
    assert!(result.content.contains("command") || result.content.contains("Missing"));
}
