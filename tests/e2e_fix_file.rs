//! Tests for fix_file patch parsing, atomic application, and repair behavior.

mod helpers;

use std::fs;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use miniswe::knowledge::ProjectIndex;
use miniswe::knowledge::indexer;
use miniswe::tools;
use miniswe::tools::fix_file::{self, PatchOp};
use miniswe::tools::permissions::PermissionManager;
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
fn parse_region_plan() {
    let plan = "\
REGION 10 20
TASK: update one logical block
END

REGION 30 35
TASK: update another logical block
END
";

    let regions = fix_file::parse_region_plan(plan).unwrap();
    assert_eq!(regions.len(), 2);
    assert_eq!(regions[0].start, 10);
    assert_eq!(regions[0].end, 20);
    assert_eq!(regions[0].task, "update one logical block");
    assert_eq!(regions[1].start, 30);
    assert_eq!(regions[1].end, 35);
    assert_eq!(regions[1].task, "update another logical block");
}

#[test]
fn parse_region_plan_allows_five_regions() {
    let plan = "\
REGION 1 1
TASK: one
END
REGION 3 3
TASK: two
END
REGION 5 5
TASK: three
END
REGION 7 7
TASK: four
END
REGION 9 9
TASK: five
END
";

    let regions = fix_file::parse_region_plan(plan).unwrap();
    assert_eq!(regions.len(), 5);
}

#[test]
fn parse_region_plan_rejects_overlap_and_preamble() {
    assert!(
        fix_file::parse_region_plan("NO_REGIONS")
            .unwrap()
            .is_empty()
    );
    assert!(fix_file::parse_region_plan("Here are regions:\nREGION 1 2\nTASK: x\nEND").is_err());
    assert!(
        fix_file::parse_region_plan("REGION 1 3\nTASK: x\nEND\nREGION 3 5\nTASK: y\nEND").is_err()
    );

    let err = fix_file::parse_region_plan("REGION L1 3\nTASK: x\nEND")
        .unwrap_err()
        .to_string();
    assert!(err.contains("invalid region start 'L1'"));
    assert!(err.contains("REGION L1 3"));
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
    let err = fix_file::apply_patch_dry_run("a\n", &mismatch)
        .unwrap_err()
        .to_string();
    assert!(err.contains("OLD mismatch for REPLACE_AT 1"));
    assert!(err.contains("OLD1: \"not a\""));
    assert!(err.contains("Actual text at anchor"));
    assert!(err.contains("L1: \"a\""));

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

    let err = fix_file::apply_patch_dry_run(content, &ops)
        .unwrap_err()
        .to_string();
    assert!(err.contains("matched 2 locations"));
    assert!(err.contains("L1"));
    assert!(err.contains("L3"));
    assert!(err.contains("Use a more specific OLD block"));
}

#[test]
fn replace_at_reports_trimmed_match_hint() {
    let content = "fn main() {\n        call();\n}\n";
    let ops = vec![PatchOp::ReplaceAt {
        start: 2,
        old: vec!["call();".into()],
        new: vec!["        other();".into()],
    }];

    let err = fix_file::apply_patch_dry_run(content, &ops)
        .unwrap_err()
        .to_string();
    assert!(err.contains("Whitespace-trimmed OLD would match at L2"));
    assert!(err.contains("preserve exact indentation"));
}

#[test]
fn overlapping_replacements_report_conflicting_spans() {
    let content = "a\nb\nc\nd\n";
    let ops = vec![
        PatchOp::ReplaceAt {
            start: 2,
            old: vec!["b".into(), "c".into()],
            new: vec!["x".into()],
        },
        PatchOp::ReplaceAt {
            start: 3,
            old: vec!["c".into()],
            new: vec!["y".into()],
        },
    ];

    let err = fix_file::apply_patch_dry_run(content, &ops)
        .unwrap_err()
        .to_string();
    assert!(err.contains("overlapping replacement/delete spans"));
    assert!(err.contains("op 1 REPLACE_AT 2"));
    assert!(err.contains("op 2 REPLACE_AT 3"));
    assert!(err.contains("L2-L3"));
    assert!(err.contains("L3"));
    assert!(err.contains("smallest enclosing REPLACE_AT"));
    assert!(err.contains("narrower fix_file task"));
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
    let result = fix_file::execute(&args, &config, &router, None)
        .await
        .unwrap();

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
    let result = fix_file::execute(&args, &config, &router, None)
        .await
        .unwrap();

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
    let result = fix_file::execute(&args, &config, &router, None)
        .await
        .unwrap();

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
    let result = fix_file::execute(&args, &config, &router, None)
        .await
        .unwrap();

    assert!(result.success, "{}", result.content);
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "fixed\n"
    );
}

#[tokio::test]
async fn execute_split_fallback_after_broad_overlap_failure() {
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                0..=2 => helpers::mock_text_response(
                    "REPLACE_AT 1\nOLD:\na\nb\nEND_OLD\nNEW:\nx\nEND_NEW\n\nREPLACE_AT 2\nOLD:\nb\nEND_OLD\nNEW:\ny\nEND_NEW\n",
                ),
                3 => helpers::mock_text_response(
                    "REGION 1 2\nTASK: replace the first two-line block\nEND\n",
                ),
                _ => helpers::mock_text_response(
                    "REPLACE_AT 1\nOLD:\na\nb\nEND_OLD\nNEW:\nx\nEND_NEW\n",
                ),
            }
        })
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    fs::write(config.project_root.join("main.rs"), "a\nb\nc\n").unwrap();

    let router = miniswe::llm::ModelRouter::new(&config);
    let args = serde_json::json!({"path": "main.rs", "task": "change first block"});
    let result = fix_file::execute(&args, &config, &router, None)
        .await
        .unwrap();

    assert!(result.success, "{}", result.content);
    assert!(result.content.contains("Split fallback"));
    assert_eq!(calls.load(Ordering::SeqCst), 5);
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "x\nc\n"
    );
}

#[tokio::test]
async fn execute_preplans_bulk_edit_into_regions() {
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                0 => helpers::mock_text_response(
                    "REGION 1 1\nTASK: update first call\nEND\n\nREGION 3 3\nTASK: update last call\nEND\n",
                ),
                1 => helpers::mock_text_response(
                    "REPLACE_AT 3\nOLD:\ncall_c();\nEND_OLD\nNEW:\ncall_c(None);\nEND_NEW\n",
                ),
                _ => helpers::mock_text_response(
                    "REPLACE_AT 1\nOLD:\ncall_a();\nEND_OLD\nNEW:\ncall_a(None);\nEND_NEW\n",
                ),
            }
        })
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    fs::write(
        config.project_root.join("main.rs"),
        "call_a();\nkeep();\ncall_c();\n",
    )
    .unwrap();

    let router = miniswe::llm::ModelRouter::new(&config);
    let args = serde_json::json!({
        "path": "main.rs",
        "task": "update all call sites",
        "lsp_validation": "off"
    });
    let result = fix_file::execute(&args, &config, &router, None)
        .await
        .unwrap();

    assert!(result.success, "{}", result.content);
    assert!(result.content.contains("Pre-plan: 2 region(s) planned"));
    assert!(result.content.contains("via pre-plan"));
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "call_a(None);\nkeep();\ncall_c(None);\n"
    );
}

#[tokio::test]
async fn execute_preplan_parse_failure_falls_back_to_broad_patch() {
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                helpers::mock_text_response("I would edit lines 1-2.")
            } else {
                helpers::mock_text_response(
                    "REPLACE_AT 1\nOLD:\nold();\nEND_OLD\nNEW:\nnew();\nEND_NEW\n",
                )
            }
        })
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    fs::write(config.project_root.join("main.rs"), "old();\n").unwrap();

    let router = miniswe::llm::ModelRouter::new(&config);
    let args = serde_json::json!({
        "path": "main.rs",
        "task": "update all calls",
        "lsp_validation": "off"
    });
    let result = fix_file::execute(&args, &config, &router, None)
        .await
        .unwrap();

    assert!(result.success, "{}", result.content);
    assert!(result.content.contains("Window 1: 1 operation"));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "new();\n"
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
    let result = fix_file::execute(&args, &config, &router, None)
        .await
        .unwrap();

    assert!(result.success);
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "original\n"
    );
}

#[tokio::test]
async fn execute_lsp_off_succeeds_without_lsp_client() {
    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(helpers::mock_text_response(
            "REPLACE_AT 1\nOLD:\noriginal\nEND_OLD\nNEW:\nfixed\nEND_NEW\n",
        ))
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    fs::write(config.project_root.join("main.rs"), "original\n").unwrap();

    let router = miniswe::llm::ModelRouter::new(&config);
    let args = serde_json::json!({
        "path": "main.rs",
        "task": "change line",
        "lsp_validation": "off"
    });
    let result = fix_file::execute(&args, &config, &router, None)
        .await
        .unwrap();

    assert!(result.success, "{}", result.content);
    assert!(result.content.contains("[lsp] skipped (off)"));
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "fixed\n"
    );
}

#[tokio::test]
async fn execute_rejects_invalid_lsp_validation_mode() {
    let mock_server = MockServer::start().await;
    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    fs::write(config.project_root.join("main.rs"), "original\n").unwrap();

    let router = miniswe::llm::ModelRouter::new(&config);
    let args = serde_json::json!({
        "path": "main.rs",
        "task": "change line",
        "lsp_validation": "sometimes"
    });
    let result = fix_file::execute(&args, &config, &router, None)
        .await
        .unwrap();

    assert!(!result.success);
    assert!(result.content.contains("Invalid lsp_validation"));
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "original\n"
    );
}

#[tokio::test]
async fn execute_fix_file_tool_reindexes_successful_edit() {
    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(helpers::mock_text_response(
            "REPLACE_AT 1\nOLD:\npub fn original() {}\nEND_OLD\nNEW:\npub fn replacement() {}\nEND_NEW\n",
        ))
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    fs::create_dir_all(config.project_root.join("src")).unwrap();
    fs::write(
        config.project_root.join("src/lib.rs"),
        "pub fn original() {}\n",
    )
    .unwrap();

    let index = indexer::index_project(&config.project_root, None).unwrap();
    index.save(&config.miniswe_dir()).unwrap();
    assert!(!index.lookup("original").is_empty());
    assert!(index.lookup("replacement").is_empty());

    let router = miniswe::llm::ModelRouter::new(&config);
    let perms = PermissionManager::headless(&config);
    let args = serde_json::json!({
        "path": "src/lib.rs",
        "task": "rename original to replacement",
        "lsp_validation": "off"
    });
    let result = tools::execute_fix_file_tool(&args, &config, &perms, &router, None)
        .await
        .unwrap();

    assert!(result.success, "{}", result.content);
    let updated_index = ProjectIndex::load(&config.miniswe_dir()).unwrap();
    assert!(
        !updated_index.lookup("replacement").is_empty(),
        "Index should contain replacement after fix_file reindex"
    );
    assert!(
        updated_index.lookup("original").is_empty(),
        "Index should no longer contain original after fix_file reindex"
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
