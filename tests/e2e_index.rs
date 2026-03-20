//! E2E tests for the knowledge index pipeline.
//!
//! Tests the full chain: source files → indexer → symbol extraction →
//! compute_end_lines → read_symbol → repo_map rendering.
//!
//! These tests verify that the coordinates (file paths, line numbers)
//! the LLM receives are accurate.

mod helpers;

use std::fs;

use miniswe::config::Config;
use miniswe::knowledge::indexer;
use miniswe::knowledge::graph::{DependencyGraph, populate_symbol_deps};
use miniswe::knowledge::repo_map;
use miniswe::knowledge::ProjectIndex;
use miniswe::tools;
use miniswe::tools::permissions::PermissionManager;
use serde_json::json;

fn perms(config: &Config) -> PermissionManager {
    PermissionManager::headless(config)
}

// ── Symbol extraction + line numbers ────────────────────────────────

#[test]
fn index_rust_file_correct_line_numbers() {
    let (_tmp, config) = helpers::create_test_project();

    let source = "\
use std::io;

pub struct Config {
    pub name: String,
    pub value: i32,
}

impl Config {
    pub fn new(name: &str) -> Self {
        Config {
            name: name.to_string(),
            value: 0,
        }
    }

    pub fn with_value(mut self, v: i32) -> Self {
        self.value = v;
        self
    }
}

pub fn create_config() -> Config {
    Config::new(\"default\")
}
";
    fs::create_dir_all(helpers::project_path(&config, "src")).unwrap();
    fs::write(helpers::project_path(&config, "src/config.rs"), source).unwrap();

    let index = indexer::index_project(&config.project_root, None).unwrap();

    // Verify Config struct is found
    let config_syms = index.lookup("Config");
    assert!(!config_syms.is_empty(), "Should find Config symbol");

    // Tree-sitter and regex extractors should both report Rust structs as "struct"
    let config_sym = config_syms.iter().find(|s| s.kind == "struct")
        .expect("Should find Config with kind=struct (not class)");
    assert_eq!(config_sym.file, "src/config.rs");
    assert_eq!(config_sym.line, 3, "Config struct should be on line 3");
    assert!(
        config_sym.end_line >= 6,
        "Config struct should end on or after line 6, got {}",
        config_sym.end_line
    );

    // Verify create_config function
    let func_syms = index.lookup("create_config");
    assert!(!func_syms.is_empty(), "Should find create_config");
    let func = &func_syms[0];
    assert_eq!(func.line, 22, "create_config should be on line 22");

    // Verify method new is found
    let new_syms = index.lookup("new");
    let new_method = new_syms.iter().find(|s| s.file == "src/config.rs");
    assert!(new_method.is_some(), "Should find new method");
    assert_eq!(new_method.unwrap().line, 9, "new() should be on line 9");
}

#[test]
fn index_python_file_correct_line_numbers() {
    let (_tmp, config) = helpers::create_test_project();

    let source = "\
import os

class Handler:
    def __init__(self, name):
        self.name = name

    def process(self, data):
        return data.upper()

def main():
    h = Handler(\"test\")
    print(h.process(\"hello\"))
";
    fs::write(helpers::project_path(&config, "handler.py"), source).unwrap();

    let index = indexer::index_project(&config.project_root, None).unwrap();

    let handler_syms = index.lookup("Handler");
    assert!(!handler_syms.is_empty(), "Should find Handler class");
    assert_eq!(handler_syms[0].line, 3, "Handler should be on line 3");

    let main_syms = index.lookup("main");
    assert!(!main_syms.is_empty(), "Should find main function");
    assert_eq!(main_syms[0].line, 10, "main should be on line 10");
}

// ── compute_end_lines accuracy ──────────────────────────────────────

#[test]
fn end_line_correct_for_simple_rust_function() {
    let (_tmp, config) = helpers::create_test_project();

    let source = "\
fn first() {
    println!(\"hello\");
}

fn second() {
    println!(\"world\");
}
";
    fs::write(helpers::project_path(&config, "simple.rs"), source).unwrap();

    let index = indexer::index_project(&config.project_root, None).unwrap();

    let first = &index.lookup("first")[0];
    assert_eq!(first.line, 1);
    assert_eq!(first.end_line, 3, "first() should end on line 3 (closing brace)");

    let second = &index.lookup("second")[0];
    assert_eq!(second.line, 5);
    assert_eq!(second.end_line, 7, "second() should end on line 7");
}

#[test]
fn end_line_with_braces_in_string_literal() {
    let (_tmp, config) = helpers::create_test_project();

    // This is a known edge case: braces inside string literals can confuse
    // the naive brace-depth tracker in compute_end_lines.
    let source = "\
fn with_braces_in_string() {
    let s = \"}\";
    let t = \"{\";
    println!(\"{s}{t}\");
}

fn next_function() {
    println!(\"ok\");
}
";
    fs::write(helpers::project_path(&config, "braces.rs"), source).unwrap();

    let index = indexer::index_project(&config.project_root, None).unwrap();

    let sym = &index.lookup("with_braces_in_string")[0];
    assert_eq!(sym.line, 1);

    // The correct end line is 5 (the closing brace of the function).
    // compute_end_lines must not be confused by braces inside string literals.
    assert_eq!(
        sym.end_line, 5,
        "with_braces_in_string() should end at line 5, got {} — \
         brace counting must skip braces inside string literals",
        sym.end_line
    );

    // Regardless of the first function's end_line, next_function should still be found
    let next = &index.lookup("next_function")[0];
    assert_eq!(next.line, 7, "next_function should still be on line 7");
}

#[test]
fn end_line_with_nested_braces() {
    let (_tmp, config) = helpers::create_test_project();

    let source = "\
fn nested() {
    if true {
        for i in 0..10 {
            if i > 5 {
                break;
            }
        }
    }
}

fn after_nested() {
    println!(\"ok\");
}
";
    fs::write(helpers::project_path(&config, "nested.rs"), source).unwrap();

    let index = indexer::index_project(&config.project_root, None).unwrap();

    let sym = &index.lookup("nested")[0];
    assert_eq!(sym.line, 1);
    assert_eq!(sym.end_line, 9, "nested() should end at line 9, got {}", sym.end_line);

    let after = &index.lookup("after_nested")[0];
    assert_eq!(after.line, 11);
}

// ── count_braces edge cases ─────────────────────────────────────────

#[test]
fn end_line_with_raw_string() {
    let (_tmp, config) = helpers::create_test_project();

    // Raw string with inner quotes: r#"has "quotes" and }"# — the inner "
    // should NOT end the string, but naive " detection exits string mode early.
    let source = r##"fn with_raw_string() {
    let s = r#"has "quotes" and }"#;
    println!("{s}");
}

fn after_raw() {
    println!("ok");
}
"##;
    fs::write(helpers::project_path(&config, "raw.rs"), source).unwrap();

    let index = indexer::index_project(&config.project_root, None).unwrap();

    let sym = &index.lookup("with_raw_string")[0];
    assert_eq!(sym.line, 1);
    assert_eq!(
        sym.end_line, 4,
        "with_raw_string() should end at line 4, got {}",
        sym.end_line
    );
}

#[test]
fn end_line_with_block_comment() {
    let (_tmp, config) = helpers::create_test_project();

    // Block comment with unbalanced brace: /* } */
    let source = "\
fn with_block_comment() {
    /* } */
    let x = 1;
}

fn after_comment() {
    println!(\"ok\");
}
";
    fs::write(helpers::project_path(&config, "block.rs"), source).unwrap();

    let index = indexer::index_project(&config.project_root, None).unwrap();

    let sym = &index.lookup("with_block_comment")[0];
    assert_eq!(sym.line, 1);
    assert_eq!(
        sym.end_line, 4,
        "with_block_comment() should end at line 4, got {}",
        sym.end_line
    );
}

#[test]
fn end_line_with_escaped_backslash_before_quote() {
    let (_tmp, config) = helpers::create_test_project();

    // String ending with escaped backslash: "\\" followed by closing brace
    // The \\\\ in the Rust literal becomes \\ on disk, so the file contains: "\\"}
    // The " after \\ is the real end of the string (not escaped).
    let source = "fn with_escaped_backslash() {\n    if true { let s = \"\\\\\"; }\n    println!(\"ok\");\n}\n\nfn after_escaped() {\n    println!(\"ok\");\n}\n";
    fs::write(helpers::project_path(&config, "escaped.rs"), source).unwrap();

    let index = indexer::index_project(&config.project_root, None).unwrap();

    let sym = &index.lookup("with_escaped_backslash")[0];
    assert_eq!(sym.line, 1);
    assert_eq!(
        sym.end_line, 4,
        "with_escaped_backslash() should end at line 4, got {}",
        sym.end_line
    );
}

#[test]
fn end_line_with_closure() {
    let (_tmp, config) = helpers::create_test_project();

    // Closures create nested braces that must not truncate the outer function
    let source = "\
fn with_closure() {
    let items = vec![1, 2, 3];
    let doubled: Vec<i32> = items.iter().map(|x| {
        x * 2
    }).collect();
    println!(\"{doubled:?}\");
}

fn after_closure() {
    println!(\"ok\");
}
";
    fs::write(helpers::project_path(&config, "closure.rs"), source).unwrap();

    let index = indexer::index_project(&config.project_root, None).unwrap();

    let sym = &index.lookup("with_closure")[0];
    assert_eq!(sym.line, 1);
    assert_eq!(
        sym.end_line, 7,
        "with_closure() should end at line 7, got {} — \
         closures must not truncate the outer function",
        sym.end_line
    );
}

// ── read_symbol returns correct source ──────────────────────────────

#[tokio::test]
async fn read_symbol_returns_correct_source() {
    let (_tmp, config) = helpers::create_test_project();

    let source = "\
pub fn greet(name: &str) -> String {
    format!(\"Hello, {name}!\")
}

pub fn farewell(name: &str) -> String {
    format!(\"Goodbye, {name}!\")
}
";
    fs::create_dir_all(helpers::project_path(&config, "src")).unwrap();
    fs::write(helpers::project_path(&config, "src/greet.rs"), source).unwrap();

    // Index the project
    let index = indexer::index_project(&config.project_root, None).unwrap();
    index.save(&config.miniswe_dir()).unwrap();

    // Use read_symbol tool
    let args = json!({"name": "greet"});
    let result = tools::execute_tool("read_symbol", &args, &config, &perms(&config))
        .await
        .unwrap();

    assert!(result.success, "read_symbol should succeed: {}", result.content);
    assert!(result.content.contains("greet"), "should mention symbol name");
    assert!(result.content.contains("src/greet.rs"), "should mention file path");
    assert!(
        result.content.contains("Hello"),
        "should contain the function body: {}",
        result.content
    );
    // Should NOT contain farewell function
    assert!(
        !result.content.contains("Goodbye"),
        "should not contain unrelated function: {}",
        result.content
    );
}

#[tokio::test]
async fn read_symbol_not_found() {
    let (_tmp, config) = helpers::create_test_project();

    let args = json!({"name": "nonexistent_symbol"});
    let result = tools::execute_tool("read_symbol", &args, &config, &perms(&config))
        .await
        .unwrap();

    assert!(!result.success);
    assert!(result.content.contains("not found"));
}

// ── Stale index after file modification ─────────────────────────────

#[tokio::test]
async fn read_symbol_stale_after_external_edit() {
    let (_tmp, config) = helpers::create_test_project();

    // Phase 1: create and index a file
    let source_v1 = "\
pub fn target() {
    println!(\"version 1\");
}
";
    fs::create_dir_all(helpers::project_path(&config, "src")).unwrap();
    fs::write(helpers::project_path(&config, "src/target.rs"), source_v1).unwrap();

    let index = indexer::index_project(&config.project_root, None).unwrap();
    index.save(&config.miniswe_dir()).unwrap();

    // Phase 2: modify the file externally (add lines before the function)
    let source_v2 = "\
// New comment line 1
// New comment line 2
// New comment line 3

pub fn target() {
    println!(\"version 2\");
}
";
    fs::write(helpers::project_path(&config, "src/target.rs"), source_v2).unwrap();

    // Phase 3: read_symbol should detect that the file is stale and re-extract,
    // returning the correct content from the updated file.
    let args = json!({"name": "target"});
    let result = tools::execute_tool("read_symbol", &args, &config, &perms(&config))
        .await
        .unwrap();

    assert!(result.success);
    assert!(
        result.content.contains("version 2"),
        "read_symbol should return updated content after file was modified externally.\n\
         Got: {}",
        result.content
    );
}

// ── Repo map rendering ──────────────────────────────────────────────

#[test]
fn repo_map_includes_indexed_files() {
    let (_tmp, config) = helpers::create_test_project();

    let source1 = "pub struct Config { pub name: String }\n";
    let source2 = "pub fn run(config: Config) {}\n";

    fs::create_dir_all(helpers::project_path(&config, "src")).unwrap();
    fs::write(helpers::project_path(&config, "src/config.rs"), source1).unwrap();
    fs::write(helpers::project_path(&config, "src/run.rs"), source2).unwrap();

    let mut index = indexer::index_project(&config.project_root, None).unwrap();
    populate_symbol_deps(&mut index);
    let graph = DependencyGraph::build(&index);

    let map = repo_map::render(&index, &graph, 5000, &[], &config.project_root);

    assert!(!map.is_empty(), "repo map should not be empty");
    assert!(map.contains("src/config.rs"), "should mention config.rs");
    assert!(map.contains("src/run.rs"), "should mention run.rs");
    assert!(map.contains("Config"), "should mention Config symbol");
}

#[test]
fn repo_map_all_files_exist_on_disk() {
    let (_tmp, config) = helpers::create_test_project();

    // Create files, index them
    fs::create_dir_all(helpers::project_path(&config, "src")).unwrap();
    fs::write(
        helpers::project_path(&config, "src/lib.rs"),
        "pub fn hello() {}\npub fn world() {}\n",
    )
    .unwrap();
    fs::write(
        helpers::project_path(&config, "src/util.rs"),
        "pub fn helper() {}\n",
    )
    .unwrap();

    let index = indexer::index_project(&config.project_root, None).unwrap();
    let graph = DependencyGraph::build(&index);
    let map = repo_map::render(&index, &graph, 5000, &[], &config.project_root);

    // Every file mentioned in the repo map should exist on disk
    for line in map.lines() {
        let trimmed = line.trim();
        // File headers look like "src/lib.rs:" or "src/lib.rs: (names)"
        if trimmed.ends_with(':') || trimmed.ends_with("(names)") {
            let file = trimmed
                .trim_end_matches(':')
                .trim_end_matches("(names)")
                .trim();
            let path = helpers::project_path(&config, file);
            assert!(
                path.exists(),
                "Repo map references file '{}' which doesn't exist on disk",
                file
            );
        }
    }
}

#[test]
fn repo_map_files_match_after_deletion() {
    let (_tmp, config) = helpers::create_test_project();

    fs::create_dir_all(helpers::project_path(&config, "src")).unwrap();
    fs::write(helpers::project_path(&config, "src/keep.rs"), "pub fn kept() {}\n").unwrap();
    fs::write(helpers::project_path(&config, "src/delete.rs"), "pub fn deleted() {}\n").unwrap();

    // Index with both files
    let index = indexer::index_project(&config.project_root, None).unwrap();

    // Delete one file
    fs::remove_file(helpers::project_path(&config, "src/delete.rs")).unwrap();

    // Repo map should NOT reference the deleted file — render should skip
    // files that no longer exist on disk.
    let graph = DependencyGraph::build(&index);
    let map = repo_map::render(&index, &graph, 5000, &[], &config.project_root);

    assert!(
        !map.contains("delete.rs"),
        "Repo map should not reference deleted file 'src/delete.rs'"
    );
}

// ── Incremental reindex after tool edit ─────────────────────────────

#[tokio::test]
async fn reindex_after_write_file_updates_symbols() {
    let (_tmp, config) = helpers::create_test_project();

    // Phase 1: create file and initial index
    fs::create_dir_all(helpers::project_path(&config, "src")).unwrap();
    fs::write(
        helpers::project_path(&config, "src/lib.rs"),
        "pub fn original() {}\n",
    )
    .unwrap();

    let index = indexer::index_project(&config.project_root, None).unwrap();
    index.save(&config.miniswe_dir()).unwrap();

    // Verify original symbol exists
    assert!(!index.lookup("original").is_empty());
    assert!(index.lookup("replacement").is_empty());

    // Phase 2: use write_file tool (which triggers reindex)
    let args = json!({
        "path": "src/lib.rs",
        "content": "pub fn replacement() {\n    println!(\"new\");\n}\n"
    });
    let result = tools::execute_tool("write_file", &args, &config, &perms(&config))
        .await
        .unwrap();
    assert!(result.success);

    // Phase 3: check the index was updated
    let updated_index = ProjectIndex::load(&config.miniswe_dir()).unwrap();
    assert!(
        !updated_index.lookup("replacement").is_empty(),
        "Index should contain the new symbol after write_file reindex"
    );
    // The old symbol should be gone
    assert!(
        updated_index.lookup("original").is_empty(),
        "Index should no longer contain the old symbol"
    );
}

// ── Repo map respects token budget ──────────────────────────────────

#[test]
fn repo_map_respects_budget() {
    let (_tmp, config) = helpers::create_test_project();

    // Create many files to exceed budget
    fs::create_dir_all(helpers::project_path(&config, "src")).unwrap();
    for i in 0..50 {
        let content = format!(
            "pub struct Type{i} {{}}\npub fn func{i}() {{}}\npub fn helper{i}() {{}}\n"
        );
        fs::write(
            helpers::project_path(&config, &format!("src/mod{i}.rs")),
            content,
        )
        .unwrap();
    }

    let index = indexer::index_project(&config.project_root, None).unwrap();
    let graph = DependencyGraph::build(&index);

    // Small budget — should truncate
    let small_map = repo_map::render(&index, &graph, 200, &[], &config.project_root);
    let large_map = repo_map::render(&index, &graph, 50000, &[], &config.project_root);

    assert!(
        small_map.len() < large_map.len(),
        "Small budget map ({}) should be shorter than large budget map ({})",
        small_map.len(),
        large_map.len()
    );

    // Token estimate of small map should be roughly within budget
    let est_tokens = small_map.len() / 4;
    assert!(
        est_tokens <= 300, // some slack for estimation
        "Small map should be roughly within 200 token budget, got ~{est_tokens}"
    );
}

// ── Task keyword personalization ────────────────────────────────────

#[test]
fn repo_map_boosts_keyword_matches() {
    let (_tmp, config) = helpers::create_test_project();

    fs::create_dir_all(helpers::project_path(&config, "src")).unwrap();
    fs::write(
        helpers::project_path(&config, "src/auth.rs"),
        "pub fn authenticate() {}\npub fn authorize() {}\n",
    )
    .unwrap();
    fs::write(
        helpers::project_path(&config, "src/database.rs"),
        "pub fn connect() {}\npub fn query() {}\n",
    )
    .unwrap();

    let index = indexer::index_project(&config.project_root, None).unwrap();
    let graph = DependencyGraph::build(&index);

    // With "auth" keyword, auth.rs should appear prominently
    let map = repo_map::render(&index, &graph, 5000, &["auth"], &config.project_root);
    let auth_pos = map.find("auth.rs");
    let db_pos = map.find("database.rs");

    assert!(auth_pos.is_some(), "auth.rs should be in the map");
    if let (Some(a), Some(d)) = (auth_pos, db_pos) {
        assert!(
            a < d,
            "auth.rs (pos {a}) should appear before database.rs (pos {d}) when searching for 'auth'"
        );
    }
}
