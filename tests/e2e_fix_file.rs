//! Tests for fix_file patch parsing, atomic application, and repair behavior.

mod helpers;

use std::fs;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use miniswe::tools::fix_file::{self, PatchOp};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ── Patch parsing ─────────────────────────────────────────────────

#[test]
fn parse_insert_after_preserves_indentation() {
    let patch = "\
INSERT_AFTER 2
CONTENT:
    let value = 42;

    println!(\"{value}\");
END
";

    let ops = fix_file::parse_patch(patch).unwrap();
    assert_eq!(
        ops,
        vec![PatchOp::InsertAfter {
            line: 2,
            content: vec![
                "    let value = 42;".into(),
                "".into(),
                "    println!(\"{value}\");".into(),
            ],
        }]
    );
}

#[test]
fn parse_insert_before() {
    let patch = "\
INSERT_BEFORE 1
CONTENT:
// header
END
";

    let ops = fix_file::parse_patch(patch).unwrap();
    assert_eq!(
        ops,
        vec![PatchOp::InsertBefore {
            line: 1,
            content: vec!["// header".into()],
        }]
    );
}

#[test]
fn parse_replace_at() {
    let patch = "\
REPLACE_AT 2
OLD:
    a();
    b();
END_OLD
NEW:
    c();
END_NEW
";

    let ops = fix_file::parse_patch(patch).unwrap();
    assert_eq!(
        ops,
        vec![PatchOp::ReplaceAt {
            start: 2,
            old: vec!["    a();".into(), "    b();".into()],
            new: vec!["    c();".into()],
        }]
    );
}

#[test]
fn parse_delete_at() {
    let patch = "\
DELETE_AT 2
OLD:
remove_me();
END_OLD
";

    let ops = fix_file::parse_patch(patch).unwrap();
    assert_eq!(
        ops,
        vec![PatchOp::DeleteAt {
            start: 2,
            old: vec!["remove_me();".into()],
        }]
    );
}

#[test]
fn parse_no_changes() {
    let ops = fix_file::parse_patch("NO_CHANGES").unwrap();
    assert!(ops.is_empty());
}

#[test]
fn parse_rejects_preamble_and_malformed_blocks() {
    assert!(fix_file::parse_patch("Here is the patch:\nINSERT_AFTER 1\nCONTENT:\nx\nEND").is_err());
    assert!(fix_file::parse_patch("INSERT_AFTER 1\nCONTENT:\nx\n").is_err());
    assert!(fix_file::parse_patch("UNKNOWN 1\nCONTENT:\nx\nEND").is_err());
    assert!(fix_file::parse_patch("INSERT_AFTER 0\nCONTENT:\nx\nEND").is_err());
    assert!(fix_file::parse_patch("REPLACE_AT 0\nOLD:\nx\nEND_OLD\nNEW:\ny\nEND_NEW").is_err());
}

// ── Dry-run apply ─────────────────────────────────────────────────

#[test]
fn apply_insert_and_replace_at() {
    let content = "fn main() {\n    old();\n}\n";
    let ops = vec![
        PatchOp::InsertAfter {
            line: 1,
            content: vec!["    setup();".into()],
        },
        PatchOp::ReplaceAt {
            start: 2,
            old: vec!["    old();".into()],
            new: vec!["    new();".into()],
        },
    ];

    let out = fix_file::apply_patch_dry_run(content, &ops).unwrap();
    assert_eq!(out, "fn main() {\n    setup();\n    new();\n}\n");
}

#[test]
fn apply_delete_at() {
    let content = "a\nb\nc\n";
    let ops = vec![PatchOp::DeleteAt {
        start: 2,
        old: vec!["b".into()],
    }];

    let out = fix_file::apply_patch_dry_run(content, &ops).unwrap();
    assert_eq!(out, "a\nc\n");
}

#[test]
fn apply_preserves_no_trailing_newline() {
    let content = "a\nb";
    let ops = vec![PatchOp::InsertAfter {
        line: 2,
        content: vec!["c".into()],
    }];

    let out = fix_file::apply_patch_dry_run(content, &ops).unwrap();
    assert_eq!(out, "a\nb\nc");
}

#[test]
fn apply_rejects_mismatch_and_out_of_range() {
    let mismatch = vec![PatchOp::ReplaceAt {
        start: 1,
        old: vec!["not a".into()],
        new: vec!["z".into()],
    }];
    assert!(fix_file::apply_patch_dry_run("a\n", &mismatch).is_err());

    let out_of_range = vec![PatchOp::InsertAfter {
        line: 3,
        content: vec!["z".into()],
    }];
    assert!(fix_file::apply_patch_dry_run("a\n", &out_of_range).is_err());
}

#[test]
fn replace_at_uses_old_length_not_end_line() {
    let content = "a\nb\nc\nd\n";
    let ops = vec![PatchOp::ReplaceAt {
        start: 2,
        old: vec!["b".into(), "c".into()],
        new: vec!["x".into()],
    }];

    let out = fix_file::apply_patch_dry_run(content, &ops).unwrap();
    assert_eq!(out, "a\nx\nd\n");
}

#[test]
fn replace_at_relocates_unique_old_block_if_anchor_is_stale() {
    let content = "a\nb\nc\nd\n";
    let ops = vec![PatchOp::ReplaceAt {
        start: 1,
        old: vec!["c".into()],
        new: vec!["x".into()],
    }];

    let out = fix_file::apply_patch_dry_run(content, &ops).unwrap();
    assert_eq!(out, "a\nb\nx\nd\n");
}

#[test]
fn replace_at_rejects_ambiguous_old_block() {
    let content = "same\nother\nsame\n";
    let ops = vec![PatchOp::ReplaceAt {
        start: 2,
        old: vec!["same".into()],
        new: vec!["x".into()],
    }];

    assert!(fix_file::apply_patch_dry_run(content, &ops).is_err());
}

// ── Execute/atomicity with mocked LLM ──────────────────────────────

#[tokio::test]
async fn execute_valid_patch_writes_file() {
    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(helpers::mock_text_response(
            "INSERT_AFTER 1\nCONTENT:\n    added();\nEND\n",
        ))
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    fs::write(config.project_root.join("main.rs"), "fn main() {\n}\n").unwrap();

    let router = miniswe::llm::ModelRouter::new(&config);
    let args = serde_json::json!({"path": "main.rs", "task": "add setup call"});
    let result = fix_file::execute(&args, &config, &router).await.unwrap();

    assert!(result.success, "{}", result.content);
    assert!(result.content.contains("Applied 1 operation"));
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "fn main() {\n    added();\n}\n"
    );
}

#[tokio::test]
async fn execute_failed_patch_writes_nothing() {
    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(helpers::mock_text_response(
            "REPLACE_AT 1\nOLD:\nwrong\nEND_OLD\nNEW:\nnew\nEND_NEW\n",
        ))
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    fs::write(config.project_root.join("main.rs"), "original\n").unwrap();

    let router = miniswe::llm::ModelRouter::new(&config);
    let args = serde_json::json!({"path": "main.rs", "task": "change line"});
    let result = fix_file::execute(&args, &config, &router).await.unwrap();

    assert!(!result.success);
    assert!(result.content.contains("patch was not applied"));
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "original\n"
    );
}

#[tokio::test]
async fn execute_repairs_failed_first_patch() {
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                helpers::mock_text_response(
                    "REPLACE_AT 1\nOLD:\nwrong\nEND_OLD\nNEW:\nfixed\nEND_NEW\n",
                )
            } else {
                helpers::mock_text_response(
                    "REPLACE_AT 1\nOLD:\noriginal\nEND_OLD\nNEW:\nfixed\nEND_NEW\n",
                )
            }
        })
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    fs::write(config.project_root.join("main.rs"), "original\n").unwrap();

    let router = miniswe::llm::ModelRouter::new(&config);
    let args = serde_json::json!({"path": "main.rs", "task": "change line"});
    let result = fix_file::execute(&args, &config, &router).await.unwrap();

    assert!(result.success, "{}", result.content);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "fixed\n"
    );
}

#[tokio::test]
async fn execute_repairs_until_third_patch() {
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            if n < 2 {
                helpers::mock_text_response(
                    "REPLACE_AT 1\nOLD:\nwrong\nEND_OLD\nNEW:\nfixed\nEND_NEW\n",
                )
            } else {
                helpers::mock_text_response(
                    "REPLACE_AT 1\nOLD:\noriginal\nEND_OLD\nNEW:\nfixed\nEND_NEW\n",
                )
            }
        })
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    fs::write(config.project_root.join("main.rs"), "original\n").unwrap();

    let router = miniswe::llm::ModelRouter::new(&config);
    let args = serde_json::json!({"path": "main.rs", "task": "change line"});
    let result = fix_file::execute(&args, &config, &router).await.unwrap();

    assert!(result.success, "{}", result.content);
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "fixed\n"
    );
}

#[tokio::test]
async fn execute_no_changes_leaves_file_unchanged() {
    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "choices": [{
                "message": {"role": "assistant", "content": "NO_CHANGES"},
                "finish_reason": "stop"
            }]
        })))
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    fs::write(config.project_root.join("main.rs"), "original\n").unwrap();

    let router = miniswe::llm::ModelRouter::new(&config);
    let args = serde_json::json!({"path": "main.rs", "task": "no op"});
    let result = fix_file::execute(&args, &config, &router).await.unwrap();

    assert!(result.success);
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "original\n"
    );
}

// ── Window building ───────────────────────────────────────────────

#[test]
fn single_window_for_small_file() {
    let windows = fix_file::build_windows(100, 800, 100);
    assert_eq!(windows, vec![(0, 100)]);
}

#[test]
fn windows_cover_entire_file() {
    let windows = fix_file::build_windows(1500, 800, 100);
    for line in 0..1500 {
        let covered = windows.iter().any(|(s, e)| line >= *s && line < *e);
        assert!(covered, "line {line} not covered by any window");
    }
}
