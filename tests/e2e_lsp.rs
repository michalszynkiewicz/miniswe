//! E2E tests for LSP integration.
//!
//! These tests spawn a real LSP server (rust-analyzer) against a temp project.
//! rust-analyzer is auto-downloaded to ~/.miniswe/lsp-servers/ if not found.
//!
//! Tests are skipped if download fails (no network, unsupported platform).

mod helpers;

use std::fs;
use std::path::Path;
use std::time::Duration;

use miniswe::lsp::{LspClient, LspServer};
use serde_json::json;

/// Try to ensure rust-analyzer is available (downloads if needed).
/// Returns false if unavailable (no network, unsupported platform).
async fn ensure_rust_analyzer() -> bool {
    LspServer::RustAnalyzer.ensure_binary().await.is_ok()
}

/// Create a minimal Rust project in the temp dir for LSP testing.
fn create_rust_project(root: &Path) {
    fs::write(
        root.join("Cargo.toml"),
        "[package]\nname = \"lsp-test\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
    )
    .unwrap();
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(
        root.join("src/main.rs"),
        "fn main() {\n    println!(\"hello\");\n}\n",
    )
    .unwrap();
}

#[tokio::test]
async fn lsp_spawn_and_initialize() {
    if !ensure_rust_analyzer().await {
        eprintln!("skipping: rust-analyzer not available");
        return;
    }

    let (_tmp, config) = helpers::create_test_project();
    create_rust_project(&config.project_root);

    let client = LspClient::spawn(config.project_root.clone())
        .await
        .expect("failed to spawn LSP");

    // Wait for initialization (up to 60s — first load is slow)
    let start = std::time::Instant::now();
    while !client.is_ready() && start.elapsed() < Duration::from_secs(60) {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(client.is_ready(), "LSP should be ready after initialization");
    assert!(!client.has_crashed(), "LSP should not have crashed");

    client.shutdown().await;
}

#[tokio::test]
async fn lsp_diagnostics_on_type_error() {
    if !ensure_rust_analyzer().await {
        eprintln!("skipping: rust-analyzer not available");
        return;
    }

    let (_tmp, config) = helpers::create_test_project();
    create_rust_project(&config.project_root);

    let client = LspClient::spawn(config.project_root.clone())
        .await
        .expect("failed to spawn LSP");

    // Wait for ready
    let start = std::time::Instant::now();
    while !client.is_ready() && start.elapsed() < Duration::from_secs(60) {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(client.is_ready());

    // Write a file with a type error
    let bad_code = "fn main() {\n    let x: u32 = \"not a number\";\n    println!(\"{x}\");\n}\n";
    let main_rs = config.project_root.join("src/main.rs");
    fs::write(&main_rs, bad_code).unwrap();

    // Notify LSP and get diagnostics
    client.notify_file_changed(&main_rs).expect("notify failed");
    let diags = client.get_diagnostics(&main_rs, Duration::from_secs(30)).await;

    // Should have at least one error diagnostic
    assert!(
        !diags.is_empty(),
        "expected diagnostics for type error, got none"
    );
    let has_error = diags.iter().any(|d| {
        d.severity == Some(lsp_types::DiagnosticSeverity::ERROR)
    });
    assert!(has_error, "expected at least one ERROR diagnostic, got: {diags:?}");

    // Error message should mention type mismatch
    let error_msgs: Vec<&str> = diags.iter()
        .filter(|d| d.severity == Some(lsp_types::DiagnosticSeverity::ERROR))
        .map(|d| d.message.as_str())
        .collect();
    let has_type_msg = error_msgs.iter().any(|m| {
        m.contains("mismatched") || m.contains("expected") || m.contains("type")
    });
    assert!(
        has_type_msg,
        "error should mention type mismatch, got: {error_msgs:?}"
    );

    client.shutdown().await;
}

#[tokio::test]
async fn lsp_diagnostics_clear_on_fix() {
    if !ensure_rust_analyzer().await {
        eprintln!("skipping: rust-analyzer not available");
        return;
    }

    let (_tmp, config) = helpers::create_test_project();
    create_rust_project(&config.project_root);

    let client = LspClient::spawn(config.project_root.clone())
        .await
        .expect("failed to spawn LSP");

    // Wait for ready
    let start = std::time::Instant::now();
    while !client.is_ready() && start.elapsed() < Duration::from_secs(60) {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    let main_rs = config.project_root.join("src/main.rs");

    // Write bad code
    fs::write(&main_rs, "fn main() {\n    let x: u32 = \"bad\";\n}\n").unwrap();
    client.notify_file_changed(&main_rs).unwrap();
    let diags = client.get_diagnostics(&main_rs, Duration::from_secs(30)).await;
    assert!(!diags.is_empty(), "should have errors for bad code");

    // Fix the code
    fs::write(&main_rs, "fn main() {\n    let x: u32 = 42;\n    println!(\"{x}\");\n}\n").unwrap();
    client.notify_file_changed(&main_rs).unwrap();
    let diags = client.get_diagnostics(&main_rs, Duration::from_secs(30)).await;

    let errors: Vec<_> = diags.iter()
        .filter(|d| d.severity == Some(lsp_types::DiagnosticSeverity::ERROR))
        .collect();
    assert!(
        errors.is_empty(),
        "should have no errors after fix, got: {errors:?}"
    );

    client.shutdown().await;
}

#[tokio::test]
async fn lsp_auto_check_integration() {
    if !ensure_rust_analyzer().await {
        eprintln!("skipping: rust-analyzer not available");
        return;
    }

    let (_tmp, mut config) = helpers::create_test_project();
    create_rust_project(&config.project_root);
    config.lsp.enabled = true;
    config.lsp.diagnostic_timeout_ms = 30000; // 30s for CI

    let client = LspClient::spawn(config.project_root.clone())
        .await
        .expect("failed to spawn LSP");

    // Wait for ready
    let start = std::time::Instant::now();
    while !client.is_ready() && start.elapsed() < Duration::from_secs(60) {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    let perms = miniswe::tools::permissions::PermissionManager::headless(&config);

    // Write a file with a type error via the tool system
    let args = json!({
        "path": "src/main.rs",
        "content": "fn main() {\n    let x: u32 = \"type error\";\n}\n"
    });
    let result = miniswe::tools::execute_tool("write_file", &args, &config, &perms, Some(&client))
        .await
        .unwrap();

    // auto_check should use LSP and report the error
    assert!(
        !result.success,
        "should fail due to type error, result: {}",
        result.content
    );
    assert!(
        result.content.contains("[rust-analyzer]") || result.content.contains("[cargo check]"),
        "should have diagnostics from LSP or cargo check: {}",
        result.content
    );
    assert!(
        result.content.contains("error"),
        "should contain error text: {}",
        result.content
    );

    client.shutdown().await;
}

// ── Server detection tests (no network needed) ────────────────────────

#[test]
fn detect_rust_project() {
    let tmp = tempfile::TempDir::new().unwrap();
    fs::write(tmp.path().join("Cargo.toml"), "[package]").unwrap();
    assert_eq!(LspServer::detect(tmp.path()), Some(LspServer::RustAnalyzer));
}

#[test]
fn detect_typescript_project() {
    let tmp = tempfile::TempDir::new().unwrap();
    fs::write(tmp.path().join("tsconfig.json"), "{}").unwrap();
    assert_eq!(LspServer::detect(tmp.path()), Some(LspServer::TypeScriptLanguageServer));
}

#[test]
fn detect_python_project() {
    let tmp = tempfile::TempDir::new().unwrap();
    fs::write(tmp.path().join("pyproject.toml"), "[project]").unwrap();
    assert_eq!(LspServer::detect(tmp.path()), Some(LspServer::Pyright));
}

#[test]
fn detect_go_project() {
    let tmp = tempfile::TempDir::new().unwrap();
    fs::write(tmp.path().join("go.mod"), "module test").unwrap();
    assert_eq!(LspServer::detect(tmp.path()), Some(LspServer::Gopls));
}

#[test]
fn detect_java_maven_project() {
    let tmp = tempfile::TempDir::new().unwrap();
    fs::write(tmp.path().join("pom.xml"), "<project/>").unwrap();
    assert_eq!(LspServer::detect(tmp.path()), Some(LspServer::Jdtls));
}

#[test]
fn detect_java_gradle_project() {
    let tmp = tempfile::TempDir::new().unwrap();
    fs::write(tmp.path().join("build.gradle"), "").unwrap();
    assert_eq!(LspServer::detect(tmp.path()), Some(LspServer::Jdtls));
}

#[test]
fn detect_no_project() {
    let tmp = tempfile::TempDir::new().unwrap();
    assert_eq!(LspServer::detect(tmp.path()), None);
}
