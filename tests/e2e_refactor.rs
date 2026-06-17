//! E2E tests for `change_signature` (add_param/drop_param) and `rename`.
//!
//! Spawns a real rust-analyzer instance and a wiremock LLM server. Tests
//! are skipped when rust-analyzer can't be downloaded.

mod helpers;

use std::fs;
use std::path::Path;
use std::time::Duration;

use miniswe::lsp::{LspClient, LspServer};
use miniswe::tools;
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer};

async fn ensure_rust_analyzer() -> bool {
    LspServer::RustAnalyzer.ensure_binary().await.is_ok()
}

/// Build a Cargo project with a function `assemble(a, b)` and two callsites.
fn create_project_for_add_param(root: &Path) {
    fs::write(
        root.join("Cargo.toml"),
        "[package]\nname = \"refactor-test\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n\
         [lib]\npath = \"src/lib.rs\"\n\n[[bin]]\nname = \"refactor-test\"\npath = \"src/main.rs\"\n",
    )
    .unwrap();
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(
        root.join("src/lib.rs"),
        "pub fn assemble(a: u32, b: u32) -> u32 {\n    a + b\n}\n",
    )
    .unwrap();
    fs::write(
        root.join("src/main.rs"),
        "fn main() {\n    \
            let x = refactor_test::assemble(1, 2);\n    \
            let y = refactor_test::assemble(3, 4);\n    \
            println!(\"{x} {y}\");\n\
         }\n",
    )
    .unwrap();
}

async fn spawn_lsp_for(root: &Path) -> LspClient {
    let client = LspClient::spawn(root.to_path_buf())
        .await
        .expect("spawn rust-analyzer");

    let start = std::time::Instant::now();
    while !client.is_ready() && start.elapsed() < Duration::from_secs(60) {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(client.is_ready(), "LSP did not become ready in 60s");

    // Open the files so rust-analyzer indexes them. Then ask for
    // diagnostics — this blocks until rust-analyzer has actually parsed
    // and analyzed the file, which is what makes subsequent
    // find_references / rename calls return real results. Without this,
    // queries hit an unindexed workspace and silently return empty.
    client
        .notify_file_changed(&root.join("src/lib.rs"))
        .unwrap();
    client
        .notify_file_changed(&root.join("src/main.rs"))
        .unwrap();

    // Force analysis by waiting for diagnostics on each open file.
    let _ = client
        .get_diagnostics(&root.join("src/lib.rs"), Duration::from_secs(120))
        .await;
    let _ = client
        .get_diagnostics(&root.join("src/main.rs"), Duration::from_secs(120))
        .await;
    let _ = client.wait_for_idle(Duration::from_secs(30)).await;
    client
}

/// Probe `find_references` with backoff so we don't proceed past a
/// just-spawned rust-analyzer that hasn't yet computed cross-file refs.
async fn wait_for_references(
    client: &LspClient,
    path: &Path,
    line: u32,
    column: u32,
    expected_min: usize,
    timeout: Duration,
) -> bool {
    let start = std::time::Instant::now();
    while start.elapsed() < timeout {
        if let Ok(refs) = client.find_references(path, line, column).await
            && refs.len() >= expected_min
        {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    false
}

#[tokio::test]
async fn add_param_updates_signature_and_callsites() {
    if !ensure_rust_analyzer().await {
        eprintln!("skipping: rust-analyzer not available");
        return;
    }

    let (_tmp, mut config) = helpers::create_test_project();
    create_project_for_add_param(&config.project_root);

    // Mock model: sequence is signature edit, then one callsite edit per
    // call-line. Each response is a strict OLD/NEW block matching the
    // exact verbatim window we know the tool extracts.
    let mock_server = MockServer::start().await;
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |req: &wiremock::Request| {
            // The tool's instruction includes "line N (1-based" — match on
            // that to disambiguate between the two callsites whose windows
            // overlap.
            let body: serde_json::Value = serde_json::from_slice(&req.body).unwrap();
            let user_msg = body["messages"]
                .as_array()
                .unwrap()
                .iter()
                .rev()
                .find(|m| m["role"] == "user")
                .unwrap()["content"]
                .as_str()
                .unwrap()
                .to_string();
            // Each callsite's snippet now starts at its own target line. The
            // line-2 snippet uniquely contains `assemble(1, 2)` (line-3's
            // snippet starts at line 3 and never sees line 2). Anything with
            // "Update the call expression" but not the line-2 marker is the
            // line-3 callsite. Everything else is the signature edit.
            // Returns content that comes AFTER the prefill that ask_rewrite
            // sends as a partial assistant message (`OLD:\n<first_line>\n`).
            // The full assembled OLD/NEW block is prefill + this response.
            let response = if user_msg.contains("assemble(1, 2)") {
                "END_OLD\n\
                 NEW:\n    let x = refactor_test::assemble(1, 2, 0);\nEND_NEW"
                    .to_string()
            } else if user_msg.contains("Update the call expression") {
                "END_OLD\n\
                 NEW:\n    let y = refactor_test::assemble(3, 4, 0);\nEND_NEW"
                    .to_string()
            } else if user_msg.contains("pub fn assemble(a: u32, b: u32) -> u32") {
                "END_OLD\n\
                 NEW:\npub fn assemble(a: u32, b: u32, c: u32) -> u32 {\nEND_NEW"
                    .to_string()
            } else {
                panic!("unexpected model prompt: {user_msg}");
            };
            helpers::mock_text_response(&response)
        })
        .mount(&mock_server)
        .await;

    let lsp = spawn_lsp_for(&config.project_root).await;
    let router = miniswe::llm::ModelRouter::new(&config);

    // The function `assemble` is on line 1 of src/lib.rs, with the name
    // starting at column 8 (1-based: "pub fn assemble" — `a` of `assemble`
    // is at byte 8 → column 8 in 1-based reckoning).
    if !wait_for_references(
        &lsp,
        &config.project_root.join("src/lib.rs"),
        0,
        7,
        3, // declaration + 2 callers
        Duration::from_secs(120),
    )
    .await
    {
        eprintln!(
            "skipping: rust-analyzer didn't return cross-file references in 120s — \
             environment likely lacks Cargo metadata access"
        );
        lsp.shutdown().await;
        return;
    }

    let args = json!({
        "action": "add_param",
        "path": "src/lib.rs",
        "name": "assemble",
        "new_param": "c: u32",
        "position": "after:b",
        "callsite_fill_in": "0",
    });
    let result =
        tools::execute_refactor_tool(&args, &config, &router, Some(&lsp), None, None, None)
            .await
            .unwrap();

    assert!(result.success, "tool failed: {}", result.content);

    // Signature updated.
    let lib_after = fs::read_to_string(config.project_root.join("src/lib.rs")).unwrap();
    assert!(
        lib_after.contains("pub fn assemble(a: u32, b: u32, c: u32) -> u32"),
        "signature not updated: {lib_after}"
    );

    // Both callsites updated.
    let main_after = fs::read_to_string(config.project_root.join("src/main.rs")).unwrap();
    assert!(
        main_after.contains("refactor_test::assemble(1, 2, 0)"),
        "first callsite not updated: {main_after}"
    );
    assert!(
        main_after.contains("refactor_test::assemble(3, 4, 0)"),
        "second callsite not updated: {main_after}"
    );

    lsp.shutdown().await;
}

#[tokio::test]
async fn add_param_reports_failure_when_model_skips() {
    if !ensure_rust_analyzer().await {
        eprintln!("skipping: rust-analyzer not available");
        return;
    }

    let (_tmp, mut config) = helpers::create_test_project();
    create_project_for_add_param(&config.project_root);

    // Model that always emits SKIP.
    let mock_server = MockServer::start().await;
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(|_req: &wiremock::Request| {
            helpers::mock_text_response("SKIP\nrefusing to edit signature for test")
        })
        .mount(&mock_server)
        .await;

    let lsp = spawn_lsp_for(&config.project_root).await;
    let router = miniswe::llm::ModelRouter::new(&config);

    let args = json!({
        "action": "add_param",
        "path": "src/lib.rs",
        "name": "assemble",
        "new_param": "c: u32",
        "position": "after:b",
        "callsite_fill_in": "0",
    });
    let result =
        tools::execute_refactor_tool(&args, &config, &router, Some(&lsp), None, None, None)
            .await
            .unwrap();

    assert!(!result.success);
    assert!(
        result.content.contains("signature rewrite failed"),
        "expected signature rewrite failure, got: {}",
        result.content
    );

    // Source unchanged.
    let lib_after = fs::read_to_string(config.project_root.join("src/lib.rs")).unwrap();
    assert!(
        lib_after.contains("pub fn assemble(a: u32, b: u32) -> u32"),
        "signature should be untouched on signature failure"
    );

    lsp.shutdown().await;
}

#[tokio::test]
async fn rename_via_lsp_updates_definition_and_callers() {
    if !ensure_rust_analyzer().await {
        eprintln!("skipping: rust-analyzer not available");
        return;
    }

    let (_tmp, config) = helpers::create_test_project();
    create_project_for_add_param(&config.project_root);

    let lsp = spawn_lsp_for(&config.project_root).await;

    if !wait_for_references(
        &lsp,
        &config.project_root.join("src/lib.rs"),
        0,
        7,
        3,
        Duration::from_secs(120),
    )
    .await
    {
        eprintln!(
            "skipping: rust-analyzer didn't return cross-file references in 120s — \
             environment likely lacks Cargo metadata access"
        );
        lsp.shutdown().await;
        return;
    }

    let args = json!({
        "action": "rename",
        "path": "src/lib.rs",
        "line": 1,
        "name": "assemble",
        "new_name": "build_context",
    });
    let router = miniswe::llm::ModelRouter::new(&config);
    let result =
        tools::execute_refactor_tool(&args, &config, &router, Some(&lsp), None, None, None)
            .await
            .unwrap();

    assert!(result.success, "rename failed: {}", result.content);

    let lib_after = fs::read_to_string(config.project_root.join("src/lib.rs")).unwrap();
    assert!(
        lib_after.contains("pub fn build_context"),
        "definition not renamed: {lib_after}"
    );
    assert!(!lib_after.contains("pub fn assemble"));

    let main_after = fs::read_to_string(config.project_root.join("src/main.rs")).unwrap();
    assert!(
        main_after.contains("refactor_test::build_context(1, 2)"),
        "first caller not renamed: {main_after}"
    );
    assert!(
        main_after.contains("refactor_test::build_context(3, 4)"),
        "second caller not renamed: {main_after}"
    );

    lsp.shutdown().await;
}

#[tokio::test]
async fn drop_param_updates_signature_and_callsites() {
    if !ensure_rust_analyzer().await {
        eprintln!("skipping: rust-analyzer not available");
        return;
    }

    let (_tmp, mut config) = helpers::create_test_project();
    create_project_for_add_param(&config.project_root);

    let mock_server = MockServer::start().await;
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |req: &wiremock::Request| {
            let body: serde_json::Value = serde_json::from_slice(&req.body).unwrap();
            let user_msg = body["messages"]
                .as_array()
                .unwrap()
                .iter()
                .rev()
                .find(|m| m["role"] == "user")
                .unwrap()["content"]
                .as_str()
                .unwrap()
                .to_string();
            // Returns content that comes AFTER the prefill that ask_rewrite
            // sends as a partial assistant message (`OLD:\n<first_line>\n`).
            let response = if user_msg.contains("assemble(1, 2)") {
                "END_OLD\nNEW:\n    let x = refactor_test::assemble(1);\nEND_NEW".to_string()
            } else if user_msg.contains("Update the call expression") {
                "END_OLD\nNEW:\n    let y = refactor_test::assemble(3);\nEND_NEW".to_string()
            } else if user_msg.contains("Remove the parameter `b`") {
                "END_OLD\nNEW:\npub fn assemble(a: u32) -> u32 {\nEND_NEW".to_string()
            } else {
                panic!("unexpected model prompt: {user_msg}");
            };
            helpers::mock_text_response(&response)
        })
        .mount(&mock_server)
        .await;

    let lsp = spawn_lsp_for(&config.project_root).await;
    let router = miniswe::llm::ModelRouter::new(&config);

    if !wait_for_references(
        &lsp,
        &config.project_root.join("src/lib.rs"),
        0,
        7,
        3,
        Duration::from_secs(120),
    )
    .await
    {
        eprintln!(
            "skipping: rust-analyzer didn't return cross-file references in 120s — \
             environment likely lacks Cargo metadata access"
        );
        lsp.shutdown().await;
        return;
    }

    let args = json!({
        "action": "drop_param",
        "path": "src/lib.rs",
        "name": "assemble",
        "param": "b",
    });
    let result =
        tools::execute_refactor_tool(&args, &config, &router, Some(&lsp), None, None, None)
            .await
            .unwrap();

    assert!(result.success, "drop_param failed: {}", result.content);
    let lib_after = fs::read_to_string(config.project_root.join("src/lib.rs")).unwrap();
    assert!(
        lib_after.contains("pub fn assemble(a: u32) -> u32"),
        "signature not updated: {lib_after}"
    );
    let main_after = fs::read_to_string(config.project_root.join("src/main.rs")).unwrap();
    assert!(
        main_after.contains("refactor_test::assemble(1)") && !main_after.contains("assemble(1, 2)"),
        "first callsite not updated: {main_after}"
    );
    assert!(
        main_after.contains("refactor_test::assemble(3)") && !main_after.contains("assemble(3, 4)"),
        "second callsite not updated: {main_after}"
    );

    lsp.shutdown().await;
}

#[tokio::test]
async fn refactor_help_returns_usage() {
    let (_tmp, config) = helpers::create_test_project();
    let router = miniswe::llm::ModelRouter::new(&config);

    let result = tools::execute_refactor_tool(
        &json!({"action": "help"}),
        &config,
        &router,
        None,
        None,
        None,
        None,
    )
    .await
    .unwrap();
    assert!(result.success);
    assert!(result.content.contains("add_param"));
    assert!(result.content.contains("drop_param"));
    assert!(result.content.contains("EXAMPLE"));
}

#[tokio::test]
async fn refactor_without_lsp_returns_clear_error() {
    let (_tmp, config) = helpers::create_test_project();
    let router = miniswe::llm::ModelRouter::new(&config);

    let args = json!({
        "action": "add_param",
        "path": "src/lib.rs",
        "name": "assemble",
        "new_param": "c: u32",
        "position": "after:b",
        "callsite_fill_in": "0",
    });
    let result = tools::execute_refactor_tool(&args, &config, &router, None, None, None, None)
        .await
        .unwrap();
    assert!(!result.success);
    assert!(
        result.content.contains("requires LSP support"),
        "expected LSP-missing error, got: {}",
        result.content
    );
}

#[tokio::test]
async fn rename_without_lsp_returns_clear_error() {
    let (_tmp, config) = helpers::create_test_project();

    let args = json!({
        "action": "rename",
        "path": "src/lib.rs",
        "line": 1,
        "name": "assemble",
        "new_name": "foo",
    });
    let router = miniswe::llm::ModelRouter::new(&config);
    let result = tools::execute_refactor_tool(&args, &config, &router, None, None, None, None)
        .await
        .unwrap();
    assert!(!result.success);
    assert!(
        result.content.contains("requires LSP support"),
        "expected LSP-missing error, got: {}",
        result.content
    );
}
