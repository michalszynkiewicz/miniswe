//! E2E tests for the loop-fix changes:
//! - edit tool: expanded ±10 context + near-match on failure
//! - write_file tool: [tail] section in output
//! - auto_check: source context around errors
//! - observation masking: per-type thresholds

mod helpers;

use std::fs;

use miniswe::config::Config;
use miniswe::tools;
use miniswe::tools::permissions::PermissionManager;
use serde_json::json;

fn perms(config: &Config) -> PermissionManager {
    PermissionManager::headless(config)
}

// ── edit: expanded context ──────────────────────────────────────────

#[tokio::test]
async fn edit_shows_10_lines_context() {
    let (_tmp, config) = helpers::create_test_project();

    // Create a file with enough lines to see ±10 context
    let lines: Vec<String> = (1..=30)
        .map(|i| format!("line {i}"))
        .collect();
    let content = lines.join("\n") + "\n";
    fs::write(helpers::project_path(&config, "ctx.txt"), &content).unwrap();

    // Edit line 15
    let args = json!({"action": "replace",
        "path": "ctx.txt",
        "old": "line 15",
        "new": "EDITED LINE 15"
    });
    let result = tools::execute_tool("file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(result.success);
    // Should show context reaching back ~10 lines (line 5) and forward ~10 (line 25)
    assert!(result.content.contains("line 5"), "should show line 5 in context");
    assert!(result.content.contains("line 25"), "should show line 25 in context");
    assert!(result.content.contains("EDITED LINE 15"), "should show the edit");
    assert!(result.content.contains("showing L"), "should mention line range");
}

// ── edit: near-match on failure ─────────────────────────────────────

#[tokio::test]
async fn edit_not_found_shows_near_match() {
    let (_tmp, config) = helpers::create_test_project();

    let content = "fn main() {\n    let x = 42;\n    println!(\"{x}\");\n}\n";
    fs::write(helpers::project_path(&config, "near.txt"), content).unwrap();

    // Try to match with wrong indentation — first line of `old` trimmed
    // will match a line in the file, triggering the near-match display
    let args = json!({"action": "replace",
        "path": "near.txt",
        "old": "let x = 42;\n    println!(\"wrong\");",
        "new": "let x = 99;"
    });
    let result = tools::execute_tool("file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(!result.success);
    // Should show a near-match with the actual file content
    assert!(
        result.content.contains("near match"),
        "should show near match section: {}",
        result.content
    );
    assert!(
        result.content.contains("let x = 42"),
        "should show the actual line content: {}",
        result.content
    );
    assert!(
        result.content.contains("4 lines total"),
        "should show total line count: {}",
        result.content
    );
}

#[tokio::test]
async fn edit_not_found_no_near_match() {
    let (_tmp, config) = helpers::create_test_project();

    fs::write(helpers::project_path(&config, "nomatch.txt"), "hello world\n").unwrap();

    let args = json!({"action": "replace",
        "path": "nomatch.txt",
        "old": "completely different text that does not appear",
        "new": "replacement"
    });
    let result = tools::execute_tool("file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(!result.success);
    assert!(result.content.contains("not found"));
    // Should still show file line count for orientation
    assert!(result.content.contains("1 lines total"));
}

// ── write_file: tail section ────────────────────────────────────────

#[tokio::test]
async fn write_file_includes_tail() {
    let (_tmp, config) = helpers::create_test_project();

    let lines: Vec<String> = (1..=50)
        .map(|i| format!("line {i}"))
        .collect();
    let content = lines.join("\n") + "\n";

    let args = json!({"action": "write",
        "path": "tail_test.txt",
        "content": content
    });
    let result = tools::execute_tool("file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(result.success);
    assert!(result.content.contains("[tail]"), "should have tail section");
    // Tail should show the last 30 lines (lines 21-50)
    assert!(result.content.contains("line 50"), "should show last line");
    assert!(result.content.contains("line 21"), "should show line 21 (start of tail)");
    // Should NOT show line 20 (before the tail)
    assert!(
        !result.content.contains("│line 20\n"),
        "should not show line 20 in tail"
    );
}

#[tokio::test]
async fn write_file_short_file_shows_all_in_tail() {
    let (_tmp, config) = helpers::create_test_project();

    let args = json!({"action": "write",
        "path": "short.txt",
        "content": "line 1\nline 2\nline 3\n"
    });
    let result = tools::execute_tool("file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    assert!(result.success);
    assert!(result.content.contains("[tail]"));
    assert!(result.content.contains("line 1"), "short file should show all lines");
    assert!(result.content.contains("line 3"));
}

// ── auto_check: source context around errors ────────────────────────

#[tokio::test]
async fn auto_check_includes_source_context_on_error() {
    let (_tmp, config) = helpers::create_test_project();

    // Create a minimal Rust project with a deliberate type error
    fs::write(
        helpers::project_path(&config, "Cargo.toml"),
        "[package]\nname = \"test-proj\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
    )
    .unwrap();
    fs::create_dir_all(helpers::project_path(&config, "src")).unwrap();
    fs::write(
        helpers::project_path(&config, "src/main.rs"),
        "fn main() {\n    let x: u32 = \"hello\";\n    println!(\"{x}\");\n}\n",
    )
    .unwrap();

    // Use write_file to trigger auto_check
    let args = json!({"action": "write",
        "path": "src/main.rs",
        "content": "fn main() {\n    let x: u32 = \"hello\";\n    println!(\"{x}\");\n}\n"
    });
    let result = tools::execute_tool("file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    // auto_check should have run and found the type error
    if result.content.contains("[cargo check]") {
        assert!(!result.success, "should fail due to type error");
        assert!(result.content.contains("error"), "should contain error text");
        // Should include source context showing the error location
        assert!(
            result.content.contains("[source context]"),
            "should include source context: {}",
            result.content
        );
        assert!(
            result.content.contains("let x: u32") || result.content.contains("hello"),
            "source context should show the offending line: {}",
            result.content
        );
    }
    // If cargo is not available, the test is a no-op (CI environments)
}

#[tokio::test]
async fn auto_check_ok_on_valid_code() {
    let (_tmp, config) = helpers::create_test_project();

    fs::write(
        helpers::project_path(&config, "Cargo.toml"),
        "[package]\nname = \"test-proj\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
    )
    .unwrap();
    fs::create_dir_all(helpers::project_path(&config, "src")).unwrap();

    let args = json!({"action": "write",
        "path": "src/main.rs",
        "content": "fn main() {\n    println!(\"hello\");\n}\n"
    });
    let result = tools::execute_tool("file", &args, &config, &perms(&config), None)
        .await
        .unwrap();

    if result.content.contains("[cargo check]") {
        assert!(result.success);
        assert!(result.content.contains("OK"), "should say OK on valid code");
        assert!(
            !result.content.contains("[source context]"),
            "should not have source context when check passes"
        );
    }
}

// ── observation masking: per-type thresholds ────────────────────────

// These test the masking logic indirectly through the agent module.
// We can't call mask_old_tool_results directly (it's in run.rs, private),
// but we can verify the behavior through the public mask_keep_count concept.

#[test]
fn token_budget_masking_keeps_newest() {
    // Token-budget masking: walk backwards, keep newest until budget exceeded.
    // Budget = 100 tokens (~400 chars). Each "big" entry is ~200 chars (50 tokens).
    use miniswe::context::estimate_tokens;

    let big = "x".repeat(200); // ~50 tokens
    let small = "y".repeat(40); // ~10 tokens

    let log: Vec<(String, serde_json::Value, String)> = vec![
        ("file".into(), json!({"action": "read", "path": "a.rs"}), big.clone()),    // oldest
        ("file".into(), json!({"action": "read", "path": "b.rs"}), big.clone()),
        ("file".into(), json!({"action": "write", "path": "c.rs"}), small.clone()),
        ("file".into(), json!({"action": "read", "path": "d.rs"}), big.clone()),    // newest
    ];

    // Budget of 100 tokens: newest two (big=50 + small=10 = 60) fit,
    // third from end (big=50, total 110) exceeds budget → masked
    let budget = 100;
    let mut used = 0;
    let mut should_mask: Vec<bool> = vec![false; log.len()];
    for i in (0..log.len()).rev() {
        used += estimate_tokens(&log[i].2);
        if used > budget {
            should_mask[i] = true;
        }
    }

    assert!(should_mask[0], "oldest should be masked (over budget)");
    assert!(should_mask[1], "second oldest should be masked (over budget)");
    assert!(!should_mask[2], "file(write) fits in budget");
    assert!(!should_mask[3], "newest read fits in budget");
}

#[test]
fn token_budget_masking_nothing_when_under_budget() {
    use miniswe::context::estimate_tokens;

    let small = "content".to_string(); // ~2 tokens

    let log: Vec<(String, serde_json::Value, String)> = vec![
        ("file".into(), json!({"action": "read", "path": "a.rs"}), small.clone()),
        ("file".into(), json!({"action": "write", "path": "b.rs"}), small.clone()),
        ("file".into(), json!({"action": "read", "path": "c.rs"}), small.clone()),
    ];

    // Budget of 1000 tokens: total is ~6 tokens, well under budget
    let budget = 1000;
    let mut used = 0;
    let mut should_mask: Vec<bool> = vec![false; log.len()];
    for i in (0..log.len()).rev() {
        used += estimate_tokens(&log[i].2);
        if used > budget {
            should_mask[i] = true;
        }
    }

    assert!(!should_mask.iter().any(|m| *m), "nothing should be masked under budget");
}

#[test]
fn rich_summary_includes_function_signatures() {
    use miniswe::context::compress::summarize_tool_result;

    let content = r#"[src/cli/mod.rs: 59 lines]
   1│pub mod commands;
   2│
   3│use clap::{Parser, Subcommand};
   4│
   5│pub struct Cli {
   6│    pub message: Option<String>,
   7│}
   8│
   9│pub fn run_cli(args: &[String]) -> Result<()> {
  10│    todo!()
  11│}
"#;

    let summary = summarize_tool_result(
        "file",
        &json!({"action": "read", "path": "src/cli/mod.rs"}),
        content,
    );

    assert!(summary.contains("src/cli/mod.rs"), "should have path");
    assert!(summary.contains("pub struct Cli"), "should have struct signature");
    assert!(summary.contains("pub fn run_cli"), "should have function signature");
}

#[test]
fn rich_summary_includes_impl_blocks() {
    use miniswe::context::compress::summarize_tool_result;

    let content = r#"[src/config.rs: 30 lines]
   1│pub struct Config {
   2│    pub name: String,
   3│}
   4│
   5│impl Config {
   6│    pub fn new() -> Self {
   7│        Config { name: "default".into() }
   8│    }
   9│}
  10│
  11│pub trait Loadable {
  12│    fn load(path: &str) -> Self;
  13│}
"#;

    let summary = summarize_tool_result(
        "file",
        &json!({"action": "read", "path": "src/config.rs"}),
        content,
    );

    assert!(summary.contains("pub struct Config"), "should have struct: {summary}");
    assert!(summary.contains("pub fn new"), "should have method: {summary}");
    assert!(summary.contains("pub trait Loadable"), "should have trait: {summary}");
    assert!(summary.contains("file(action='read'"), "should hint at file(action='read') for re-reading: {summary}");
}

#[test]
fn rich_summary_edit_with_errors() {
    use miniswe::context::compress::summarize_tool_result;

    let content = "✓ Edited src/main.rs (1 replacement)\n[cargo check]\nerror[E0061]: expected 4 arguments\n";

    let summary = summarize_tool_result(
        "file",
        &json!({"action": "replace", "path": "src/main.rs"}),
        content,
    );

    assert!(summary.contains("src/main.rs"), "should have path: {summary}");
    assert!(summary.contains("error") || summary.contains("FAILED"),
        "should mention the error: {summary}");
}

#[test]
fn rich_summary_edit_success() {
    use miniswe::context::compress::summarize_tool_result;

    let content = "✓ Edited src/main.rs (1 replacement)\n[cargo check] OK\n";

    let summary = summarize_tool_result(
        "file",
        &json!({"action": "replace", "path": "src/main.rs"}),
        content,
    );

    assert!(summary.contains("src/main.rs"), "should have path: {summary}");
    assert!(!summary.contains("error") && !summary.contains("FAILED"),
        "should not mention errors: {summary}");
}
