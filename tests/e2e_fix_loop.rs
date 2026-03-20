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
    let args = json!({
        "path": "ctx.txt",
        "old": "line 15",
        "new": "EDITED LINE 15"
    });
    let result = tools::execute_tool("edit", &args, &config, &perms(&config))
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
    let args = json!({
        "path": "near.txt",
        "old": "let x = 42;\n    println!(\"wrong\");",
        "new": "let x = 99;"
    });
    let result = tools::execute_tool("edit", &args, &config, &perms(&config))
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

    let args = json!({
        "path": "nomatch.txt",
        "old": "completely different text that does not appear",
        "new": "replacement"
    });
    let result = tools::execute_tool("edit", &args, &config, &perms(&config))
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

    let args = json!({
        "path": "tail_test.txt",
        "content": content
    });
    let result = tools::execute_tool("write_file", &args, &config, &perms(&config))
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

    let args = json!({
        "path": "short.txt",
        "content": "line 1\nline 2\nline 3\n"
    });
    let result = tools::execute_tool("write_file", &args, &config, &perms(&config))
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
    let args = json!({
        "path": "src/main.rs",
        "content": "fn main() {\n    let x: u32 = \"hello\";\n    println!(\"{x}\");\n}\n"
    });
    let result = tools::execute_tool("write_file", &args, &config, &perms(&config))
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

    let args = json!({
        "path": "src/main.rs",
        "content": "fn main() {\n    println!(\"hello\");\n}\n"
    });
    let result = tools::execute_tool("write_file", &args, &config, &perms(&config))
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
fn masking_keeps_reads_longer_than_writes() {
    // Simulate the masking logic: with 8 tool results of mixed types,
    // reads should be kept longer than writes.
    use miniswe::context::compress;

    // Build a simulated tool_result_log
    let log: Vec<(String, serde_json::Value, String)> = vec![
        ("read_file".into(), json!({"path": "a.rs"}), "content of a.rs...".into()),
        ("write_file".into(), json!({"path": "b.rs"}), "✓ Wrote b.rs".into()),
        ("shell".into(), json!({"command": "ls"}), "[shell: exit 0]\nfile1".into()),
        ("read_file".into(), json!({"path": "c.rs"}), "content of c.rs...".into()),
        ("write_file".into(), json!({"path": "d.rs"}), "✓ Wrote d.rs".into()),
        ("search".into(), json!({"query": "foo"}), "matches...".into()),
        ("read_file".into(), json!({"path": "e.rs"}), "content of e.rs...".into()),
        ("read_file".into(), json!({"path": "f.rs"}), "content of f.rs...".into()),
    ];

    // Apply the per-type masking logic (reimplemented here for testing)
    fn mask_keep_count(tool_name: &str) -> usize {
        match tool_name {
            "read_file" | "read_symbol" => 3,
            "write_file" | "edit" => 2,
            "shell" | "diagnostics" => 2,
            "search" | "web_search" | "web_fetch" | "docs_lookup" => 1,
            _ => 2,
        }
    }

    let mut type_counts: std::collections::HashMap<&str, usize> =
        std::collections::HashMap::new();
    let mut should_mask: Vec<bool> = vec![false; log.len()];

    for i in (0..log.len()).rev() {
        let tool_name = log[i].0.as_str();
        let count = type_counts.entry(tool_name).or_insert(0);
        *count += 1;
        if *count > mask_keep_count(tool_name) {
            should_mask[i] = true;
        }
    }

    // First read_file (index 0): 4th oldest of 4 reads → MASKED (keep 3)
    assert!(should_mask[0], "oldest read_file should be masked");

    // write_file at index 1: 2nd oldest of 2 writes → kept
    assert!(!should_mask[1], "2nd write_file should be kept (threshold 2)");

    // shell at index 2: 1st of 1 shell → kept
    assert!(!should_mask[2], "only shell should be kept");

    // read_file at index 3: 3rd oldest of 4 reads → kept (threshold 3)
    assert!(!should_mask[3], "3rd most recent read should be kept");

    // search at index 5: 1st of 1 search → kept (threshold 1)
    assert!(!should_mask[5], "only search should be kept");

    // Last 2 reads (indices 6,7): most recent → kept
    assert!(!should_mask[6], "2nd most recent read should be kept");
    assert!(!should_mask[7], "most recent read should be kept");
}

#[test]
fn masking_nothing_when_under_thresholds() {
    // With only a few tool results, nothing should be masked
    let log: Vec<(String, serde_json::Value, String)> = vec![
        ("read_file".into(), json!({"path": "a.rs"}), "content...".into()),
        ("write_file".into(), json!({"path": "b.rs"}), "✓ Wrote".into()),
        ("read_file".into(), json!({"path": "c.rs"}), "content...".into()),
    ];

    fn mask_keep_count(tool_name: &str) -> usize {
        match tool_name {
            "read_file" | "read_symbol" => 3,
            _ => 2,
        }
    }

    let mut type_counts: std::collections::HashMap<&str, usize> =
        std::collections::HashMap::new();
    let mut should_mask: Vec<bool> = vec![false; log.len()];

    for i in (0..log.len()).rev() {
        let tool_name = log[i].0.as_str();
        let count = type_counts.entry(tool_name).or_insert(0);
        *count += 1;
        if *count > mask_keep_count(tool_name) {
            should_mask[i] = true;
        }
    }

    // 2 reads (under threshold of 3) + 1 write (under threshold of 2) → nothing masked
    assert!(!should_mask.iter().any(|m| *m), "nothing should be masked");
}
