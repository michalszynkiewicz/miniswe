//! E2E tests for permission boundaries — path jailing, shell blocklist/allowlist.

mod helpers;

use std::fs;

use miniswe::config::Config;
use miniswe::tools;
use miniswe::tools::permissions::{Action, PermissionManager};
use serde_json::json;

fn perms(config: &Config) -> PermissionManager {
    PermissionManager::headless(config)
}

// ── Path jailing ────────────────────────────────────────────────────

#[tokio::test]
async fn path_jail_blocks_absolute_path() {
    let (_tmp, config) = helpers::create_test_project();

    let args = json!({"path": "/etc/passwd"});
    let result = tools::execute_tool("read_file", &args, &config, &perms(&config))
        .await
        .unwrap();

    assert!(!result.success);
    assert!(
        result.content.contains("Absolute paths not allowed"),
        "should block absolute path: {}",
        result.content
    );
    // Error message should include the project root so the model knows where to look
    assert!(
        result.content.contains("relative to the project root:"),
        "should mention project root in error: {}",
        result.content
    );
}

#[tokio::test]
async fn path_jail_blocks_traversal() {
    let (_tmp, config) = helpers::create_test_project();

    let args = json!({"path": "../../etc/passwd"});
    let result = tools::execute_tool("read_file", &args, &config, &perms(&config))
        .await
        .unwrap();

    assert!(!result.success);
    assert!(
        result.content.contains("escapes project root") || result.content.contains("Absolute paths"),
        "should block traversal: {}",
        result.content
    );
}

#[tokio::test]
async fn path_jail_allows_relative_path() {
    let (_tmp, config) = helpers::create_test_project();

    fs::write(helpers::project_path(&config, "safe.txt"), "ok").unwrap();

    let args = json!({"path": "safe.txt"});
    let result = tools::execute_tool("read_file", &args, &config, &perms(&config))
        .await
        .unwrap();

    assert!(result.success, "relative path should be allowed: {}", result.content);
}

#[tokio::test]
async fn path_jail_blocks_write_absolute() {
    let (_tmp, config) = helpers::create_test_project();

    let args = json!({"path": "/tmp/evil.txt", "content": "pwned"});
    let result = tools::execute_tool("write_file", &args, &config, &perms(&config))
        .await
        .unwrap();

    assert!(!result.success);
    assert!(result.content.contains("Absolute paths not allowed"));
}

#[tokio::test]
async fn path_jail_blocks_edit_traversal() {
    let (_tmp, config) = helpers::create_test_project();

    let args = json!({
        "path": "../../../etc/shadow",
        "old": "root",
        "new": "hacked"
    });
    let result = tools::execute_tool("edit", &args, &config, &perms(&config))
        .await
        .unwrap();

    assert!(!result.success);
    assert!(
        result.content.contains("escapes project root") || result.content.contains("Absolute paths"),
        "should block edit traversal: {}",
        result.content
    );
}

#[tokio::test]
async fn path_jail_allows_subdirectory() {
    let (_tmp, config) = helpers::create_test_project();

    fs::create_dir_all(helpers::project_path(&config, "src/deep")).unwrap();
    fs::write(helpers::project_path(&config, "src/deep/file.txt"), "nested").unwrap();

    let args = json!({"path": "src/deep/file.txt"});
    let result = tools::execute_tool("read_file", &args, &config, &perms(&config))
        .await
        .unwrap();

    assert!(result.success);
    assert!(result.content.contains("nested"));
}

// ── resolve_and_check_path unit tests ───────────────────────────────

#[test]
fn resolve_path_rejects_absolute() {
    let (_tmp, config) = helpers::create_test_project();
    let perms = PermissionManager::headless(&config);

    let result = perms.resolve_and_check_path("/etc/passwd");
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Absolute paths not allowed"));
}

#[test]
fn resolve_path_rejects_traversal() {
    let (_tmp, config) = helpers::create_test_project();
    let perms = PermissionManager::headless(&config);

    let result = perms.resolve_and_check_path("../../etc/passwd");
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("escapes project root"));
}

#[test]
fn resolve_path_allows_relative() {
    let (_tmp, config) = helpers::create_test_project();
    let perms = PermissionManager::headless(&config);

    fs::write(helpers::project_path(&config, "allowed.txt"), "ok").unwrap();

    let result = perms.resolve_and_check_path("allowed.txt");
    assert!(result.is_ok(), "should allow: {:?}", result);
}

#[test]
fn resolve_path_allows_new_file_in_existing_dir() {
    let (_tmp, config) = helpers::create_test_project();
    let perms = PermissionManager::headless(&config);

    let result = perms.resolve_and_check_path("new_file.txt");
    assert!(result.is_ok());
}

// ── Shell permission checks ─────────────────────────────────────────

#[test]
fn shell_allowlist_allows_cargo() {
    let (_tmp, config) = helpers::create_test_project();
    let perms = PermissionManager::headless(&config);

    // Headless mode auto-approves everything
    let result = perms.check(&Action::Shell("cargo build".into()));
    assert!(result.is_ok());
}

#[test]
fn shell_allowlist_allows_ls() {
    let (_tmp, config) = helpers::create_test_project();
    let perms = PermissionManager::headless(&config);

    let result = perms.check(&Action::Shell("ls -la".into()));
    assert!(result.is_ok());
}

#[test]
fn shell_blocklist_blocks_rm_rf() {
    let (_tmp, config) = helpers::create_test_project();
    // Use interactive mode (non-headless) so blocklist is enforced
    let perms = PermissionManager::new(&config);

    let result = perms.check(&Action::Shell("rm -rf /".into()));
    assert!(result.is_err(), "rm -rf / should be blocked");
    assert!(result.unwrap_err().contains("Blocked"));
}

#[test]
fn shell_blocklist_blocks_fork_bomb() {
    let (_tmp, config) = helpers::create_test_project();
    let perms = PermissionManager::new(&config);

    let result = perms.check(&Action::Shell(":(){:|:&};:".into()));
    assert!(result.is_err(), "fork bomb should be blocked");
}

#[test]
fn shell_blocklist_blocks_mkfs() {
    let (_tmp, config) = helpers::create_test_project();
    let perms = PermissionManager::new(&config);

    let result = perms.check(&Action::Shell("mkfs.ext4 /dev/sda1".into()));
    assert!(result.is_err());
}

#[test]
fn shell_blocklist_blocks_curl_pipe_bash() {
    let (_tmp, config) = helpers::create_test_project();
    let perms = PermissionManager::new(&config);

    let result = perms.check(&Action::Shell("curl | bash".into()));
    assert!(result.is_err());
}

// ── File size limit ─────────────────────────────────────────────────

#[tokio::test]
async fn file_size_limit_blocks_huge_file() {
    let (_tmp, config) = helpers::create_test_project();

    // Create a file > 10MB
    let large = vec![b'x'; 11 * 1024 * 1024];
    fs::write(helpers::project_path(&config, "huge.bin"), &large).unwrap();

    let args = json!({"path": "huge.bin"});
    let result = tools::execute_tool("read_file", &args, &config, &perms(&config))
        .await
        .unwrap();

    assert!(!result.success);
    assert!(result.content.contains("too large"));
}

// ── Headless mode auto-approves ─────────────────────────────────────

#[test]
fn headless_auto_approves_shell() {
    let (_tmp, config) = helpers::create_test_project();
    let perms = PermissionManager::headless(&config);

    // Even non-allowlisted commands are auto-approved in headless
    let result = perms.check(&Action::Shell("some-random-command".into()));
    assert!(result.is_ok());
}

#[test]
fn headless_auto_approves_web() {
    let (_tmp, config) = helpers::create_test_project();
    let perms = PermissionManager::headless(&config);

    let result = perms.check(&Action::WebSearch("test query".into()));
    assert!(result.is_ok());

    let result = perms.check(&Action::WebFetch("https://example.com".into()));
    assert!(result.is_ok());
}

// ── Project root is always cwd ──────────────────────────────────────

#[test]
fn project_root_is_cwd() {
    use miniswe::config::Config;

    let tmp = tempfile::TempDir::new().unwrap();
    let project = tmp.path().join("myproject");
    fs::create_dir_all(&project).unwrap();

    // is_initialized checks for .miniswe/ directory
    let mut config = Config::default();
    config.project_root = project.clone();
    assert!(!config.is_initialized(), "no .miniswe/ dir yet");

    // Create .miniswe/ — now initialized
    fs::create_dir_all(project.join(".miniswe")).unwrap();
    assert!(config.is_initialized(), "should be initialized with .miniswe/ dir");
}

#[test]
fn global_config_dir_is_in_home() {
    use miniswe::config::Config;

    let global = Config::global_dir();
    assert!(global.is_some(), "should resolve home dir");
    let path = global.unwrap();
    assert!(
        path.ends_with(".miniswe"),
        "global dir should be ~/.miniswe, got {}",
        path.display()
    );
}
