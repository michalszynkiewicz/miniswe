//! Tests for edit_file patch parsing, atomic application, and repair behavior.

mod helpers;

use std::fs;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use miniswe::knowledge::ProjectIndex;
use miniswe::knowledge::indexer;
use miniswe::tools;
use miniswe::tools::edit_file::{self, EditPlanStep, PatchOp};
use miniswe::tools::permissions::PermissionManager;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer};

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

    let ops = edit_file::parse_patch(patch).unwrap();
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

    let ops = edit_file::parse_patch(patch).unwrap();
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

    let ops = edit_file::parse_patch(patch).unwrap();
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

    let ops = edit_file::parse_patch(patch).unwrap();
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
    let ops = edit_file::parse_patch("NO_CHANGES").unwrap();
    assert!(ops.is_empty());
}

#[test]
fn parse_edit_plan_supports_literal_and_smart_steps() {
    let plan = "\
LITERAL_REPLACE
SCOPE 1 20
ALL true
OLD:
context::assemble(&config, \"test\", &[], false, None)
END_OLD
NEW:
context::assemble(&config, \"test\", &[], false, None, None)
END_NEW
END

SMART_EDIT
REGION 30 40
TASK: update the multi-line call
END
";

    let steps = edit_file::parse_edit_plan(plan).unwrap();
    assert_eq!(steps.len(), 2);
    assert_eq!(
        steps[0],
        EditPlanStep::LiteralReplace {
            scope_start: 1,
            scope_end: 20,
            all: true,
            old: vec!["context::assemble(&config, \"test\", &[], false, None)".into()],
            new: vec!["context::assemble(&config, \"test\", &[], false, None, None)".into()],
        }
    );
    assert_eq!(
        steps[1],
        EditPlanStep::SmartEdit(edit_file::EditRegion {
            start: 30,
            end: 40,
            task: "update the multi-line call".into(),
        })
    );
}

#[test]
fn parse_edit_plan_accepts_overlapping_steps() {
    // Overlap is no longer a parse error — the planner caller
    // (`partition_overlapping_steps`) drops overlappers in source order
    // and reports them as failed steps. The parser just returns whatever
    // structurally-valid blocks it finds.
    let plan = "\
LITERAL_REPLACE
SCOPE 1 10
ALL true
OLD:
a
END_OLD
NEW:
b
END_NEW
END

SMART_EDIT
REGION 10 20
TASK: edit another block
END
";

    let steps = edit_file::parse_edit_plan(plan).unwrap();
    assert_eq!(steps.len(), 2);
}

#[test]
fn parse_rejects_preamble_and_malformed_blocks() {
    assert!(
        edit_file::parse_patch("Here is the patch:\nINSERT_AFTER 1\nCONTENT:\nx\nEND").is_err()
    );
    assert!(edit_file::parse_patch("INSERT_AFTER 1\nCONTENT:\nx\n").is_err());
    assert!(edit_file::parse_patch("UNKNOWN 1\nCONTENT:\nx\nEND").is_err());
    assert!(edit_file::parse_patch("INSERT_AFTER 0\nCONTENT:\nx\nEND").is_err());
    assert!(edit_file::parse_patch("REPLACE_AT 0\nOLD:\nx\nEND_OLD\nNEW:\ny\nEND_NEW").is_err());
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

    let out = edit_file::apply_patch_dry_run(content, &ops).unwrap();
    assert_eq!(out, "fn main() {\n    setup();\n    new();\n}\n");
}

#[test]
fn apply_delete_at() {
    let content = "a\nb\nc\n";
    let ops = vec![PatchOp::DeleteAt {
        start: 2,
        old: vec!["b".into()],
    }];

    let out = edit_file::apply_patch_dry_run(content, &ops).unwrap();
    assert_eq!(out, "a\nc\n");
}

#[test]
fn apply_preserves_no_trailing_newline() {
    let content = "a\nb";
    let ops = vec![PatchOp::InsertAfter {
        line: 2,
        content: vec!["c".into()],
    }];

    let out = edit_file::apply_patch_dry_run(content, &ops).unwrap();
    assert_eq!(out, "a\nb\nc");
}

#[test]
fn apply_rejects_mismatch_and_out_of_range() {
    let mismatch = vec![PatchOp::ReplaceAt {
        start: 1,
        old: vec!["not a".into()],
        new: vec!["z".into()],
    }];
    let err = edit_file::apply_patch_dry_run("a\n", &mismatch)
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
    assert!(edit_file::apply_patch_dry_run("a\n", &out_of_range).is_err());
}

#[test]
fn replace_at_uses_old_length_not_end_line() {
    let content = "a\nb\nc\nd\n";
    let ops = vec![PatchOp::ReplaceAt {
        start: 2,
        old: vec!["b".into(), "c".into()],
        new: vec!["x".into()],
    }];

    let out = edit_file::apply_patch_dry_run(content, &ops).unwrap();
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

    let out = edit_file::apply_patch_dry_run(content, &ops).unwrap();
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

    let err = edit_file::apply_patch_dry_run(content, &ops)
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

    let err = edit_file::apply_patch_dry_run(content, &ops)
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

    let err = edit_file::apply_patch_dry_run(content, &ops)
        .unwrap_err()
        .to_string();
    assert!(err.contains("overlapping replacement/delete spans"));
    assert!(err.contains("op 1 REPLACE_AT 2"));
    assert!(err.contains("op 2 REPLACE_AT 3"));
    assert!(err.contains("L2-L3"));
    assert!(err.contains("L3"));
    assert!(err.contains("smallest enclosing REPLACE_AT"));
    assert!(err.contains("narrower edit_file task"));
}

#[test]
fn literal_replace_in_scope_replaces_all_exact_matches() {
    let content = "call(None)\nkeep(None)\ncall(None)\noutside(None)\n";
    let (out, count) = edit_file::apply_literal_replace_in_scope(
        content,
        1,
        3,
        &["call(None)".into()],
        &["call(None, None)".into()],
        true,
    )
    .unwrap();

    assert_eq!(count, 2);
    assert_eq!(
        out,
        "call(None, None)\nkeep(None)\ncall(None, None)\noutside(None)\n"
    );
}

#[test]
fn literal_replace_in_scope_requires_exact_match_count_for_single() {
    let err = edit_file::apply_literal_replace_in_scope(
        "a\na\n",
        1,
        2,
        &["a".into()],
        &["b".into()],
        false,
    )
    .unwrap_err()
    .to_string();

    assert!(err.contains("matched 2 occurrence"));
    assert!(err.contains("expected exactly 1"));
}

// ── Execute/atomicity with mocked LLM ──────────────────────────────

#[tokio::test]
async fn execute_valid_patch_writes_file() {
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                0 => helpers::mock_text_response(
                    "SMART_EDIT\nREGION 1-2\nTASK: add setup call\nEND\n",
                ),
                1 => helpers::mock_text_response("DONE_EXPLORING"),
                2 => helpers::mock_text_response(
                    "SMART_EDIT\nREGION 1-2\nTASK: add setup call\nEND\n",
                ),
                _ => helpers::mock_text_response(
                    "INSERT_AFTER 1\nCONTENT:\n    added();\nEND\n",
                ),
            }
        })
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    fs::write(config.project_root.join("main.rs"), "fn main() {\n}\n").unwrap();

    let router = miniswe::llm::ModelRouter::new(&config);
    let args = serde_json::json!({"path": "main.rs", "task": "add setup call"});
    let result = edit_file::execute(&args, &config, &router, None, None, None)
        .await
        .unwrap();

    assert!(result.success, "{}", result.content);
    assert!(result.content.contains("via pre-plan"));
    assert_eq!(calls.load(Ordering::SeqCst), 4);
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "fn main() {\n    added();\n}\n"
    );
}

#[tokio::test]
async fn execute_failed_patch_writes_nothing() {
    // 1-line file → every plan attempt runs window + recon + finalize +
    // 3 patch retries (= 6 LLM calls). 4 attempts × 6 = 24 calls.
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n % 6 {
                0 => helpers::mock_text_response("NOTE noop"),
                1 => helpers::mock_text_response("DONE"),
                2 => helpers::mock_text_response(
                    "SMART_EDIT\nREGION 1\nTASK: change line\nEND\n",
                ),
                _ => helpers::mock_text_response(
                    "REPLACE_AT 1\nOLD:\nwrong\nEND_OLD\nNEW:\nnew\nEND_NEW\n",
                ),
            }
        })
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    fs::write(config.project_root.join("main.rs"), "original\n").unwrap();

    let router = miniswe::llm::ModelRouter::new(&config);
    let args = serde_json::json!({"path": "main.rs", "task": "change line"});
    let result = edit_file::execute(&args, &config, &router, None, None, None)
        .await
        .unwrap();

    assert!(!result.success);
    assert!(result.content.contains("patch was not applied"));
    assert!(result.content.contains("Pre-plan exhausted after 4 attempt(s)"));
    // 4 plan attempts × (window + recon + finalize + 3 patch retries) = 24
    assert_eq!(calls.load(Ordering::SeqCst), 24);
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "original\n"
    );
}

#[tokio::test]
async fn execute_repairs_failed_first_patch() {
    // 1-line file. Plan 1: window + recon + finalize + 3 failing patches.
    // Plan 2 (repair): window + recon + finalize + 1 successful patch.
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                // Plan attempt 1
                0 => helpers::mock_text_response("NOTE noop"),
                1 => helpers::mock_text_response("DONE"),
                2 => helpers::mock_text_response(
                    "SMART_EDIT\nREGION 1\nTASK: change line\nEND\n",
                ),
                3..=5 => helpers::mock_text_response(
                    "REPLACE_AT 1\nOLD:\nwrong\nEND_OLD\nNEW:\nfixed\nEND_NEW\n",
                ),
                // Plan attempt 2 (repair)
                6 => helpers::mock_text_response("NOTE noop"),
                7 => helpers::mock_text_response("DONE"),
                8 => helpers::mock_text_response(
                    "SMART_EDIT\nREGION 1\nTASK: change line\nEND\n",
                ),
                _ => helpers::mock_text_response(
                    "REPLACE_AT 1\nOLD:\noriginal\nEND_OLD\nNEW:\nfixed\nEND_NEW\n",
                ),
            }
        })
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    fs::write(config.project_root.join("main.rs"), "original\n").unwrap();

    let router = miniswe::llm::ModelRouter::new(&config);
    let args = serde_json::json!({"path": "main.rs", "task": "change line"});
    let result = edit_file::execute(&args, &config, &router, None, None, None)
        .await
        .unwrap();

    assert!(result.success, "{}", result.content);
    assert!(result.content.contains("Pre-plan repair attempt 2"));
    // plan1: 3 preplan + 3 failed patches = 6
    // plan2: 3 preplan + 1 success patch    = 4
    assert_eq!(calls.load(Ordering::SeqCst), 10);
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "fixed\n"
    );
}

#[tokio::test]
async fn execute_repairs_until_third_patch() {
    // 1-line file. Plans 1 and 2 fail (3 preplan + 3 patch retries each).
    // Plan 3 succeeds on its first patch (3 preplan + 1 patch).
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                // Per-attempt preplan: window (NOTE), recon (DONE), finalize (plan).
                0 | 6 | 12 => helpers::mock_text_response("NOTE noop"),
                1 | 7 | 13 => helpers::mock_text_response("DONE"),
                2 | 8 | 14 => helpers::mock_text_response(
                    "SMART_EDIT\nREGION 1\nTASK: change line\nEND\n",
                ),
                // Plans 1 and 2 fail their patches.
                3..=5 | 9..=11 => helpers::mock_text_response(
                    "REPLACE_AT 1\nOLD:\nwrong\nEND_OLD\nNEW:\nfixed\nEND_NEW\n",
                ),
                // Plan 3's first patch succeeds.
                _ => helpers::mock_text_response(
                    "REPLACE_AT 1\nOLD:\noriginal\nEND_OLD\nNEW:\nfixed\nEND_NEW\n",
                ),
            }
        })
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    fs::write(config.project_root.join("main.rs"), "original\n").unwrap();

    let router = miniswe::llm::ModelRouter::new(&config);
    let args = serde_json::json!({"path": "main.rs", "task": "change line"});
    let result = edit_file::execute(&args, &config, &router, None, None, None)
        .await
        .unwrap();

    assert!(result.success, "{}", result.content);
    assert!(result.content.contains("Pre-plan repair attempt 3"));
    // plan1: 3 + 3, plan2: 3 + 3, plan3: 3 + 1 = 16
    assert_eq!(calls.load(Ordering::SeqCst), 16);
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "fixed\n"
    );
}

#[tokio::test]
async fn execute_preplan_repair_after_failed_plan() {
    // 3-line file. First plan generates overlapping patch ops that get
    // rejected, so the whole plan is repaired and the second plan succeeds.
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                // Plan attempt 1
                0 => helpers::mock_text_response("NOTE noop"),
                1 => helpers::mock_text_response("DONE"),
                2 => helpers::mock_text_response(
                    "SMART_EDIT\nREGION 1-2\nTASK: replace the first two-line block\nEND\n",
                ),
                3..=5 => helpers::mock_text_response(
                    "REPLACE_AT 1\nOLD:\na\nb\nEND_OLD\nNEW:\nx\nEND_NEW\n\nREPLACE_AT 2\nOLD:\nb\nEND_OLD\nNEW:\ny\nEND_NEW\n",
                ),
                // Plan attempt 2 (repair)
                6 => helpers::mock_text_response("NOTE noop"),
                7 => helpers::mock_text_response("DONE"),
                8 => helpers::mock_text_response(
                    "SMART_EDIT\nREGION 1-2\nTASK: replace the first two-line block\nEND\n",
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
    let result = edit_file::execute(&args, &config, &router, None, None, None)
        .await
        .unwrap();

    assert!(result.success, "{}", result.content);
    assert!(result.content.contains("Pre-plan repair attempt 2"));
    // plan1: 3 preplan + 3 failed patches + plan2: 3 preplan + 1 success patch
    assert_eq!(calls.load(Ordering::SeqCst), 10);
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "x\nc\n"
    );
}

#[tokio::test]
async fn execute_preplan_repair_can_inspect_with_search_and_read() {
    // Large file (>200 lines). The repair attempt uses the recon phase to
    // SEARCH/READ before re-planning.
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                // Plan attempt 1: window + recon + finalize, then 3 failing patches.
                0 => helpers::mock_text_response("NOTE first pass"),
                1 => helpers::mock_text_response("DONE"),
                2 => helpers::mock_text_response(
                    "SMART_EDIT\nREGION 1-2\nTASK: change first block\nEND\n",
                ),
                3..=5 => helpers::mock_text_response(
                    "REPLACE_AT 1\nOLD:\na\nb\nEND_OLD\nNEW:\nx\nEND_NEW\n\nREPLACE_AT 2\nOLD:\nb\nEND_OLD\nNEW:\ny\nEND_NEW\n",
                ),
                // Plan attempt 2 (repair): window emits a note, recon issues
                // SEARCH/READ + DONE in a single round, finalize re-plans,
                // and a single patch succeeds.
                6 => helpers::mock_text_response("NOTE revisit first block"),
                7 => helpers::mock_text_response("SEARCH: a\nREAD: 1-2\nDONE"),
                8 => helpers::mock_text_response(
                    "SMART_EDIT\nREGION 1-2\nTASK: change first block\nEND\n",
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
    fs::write(
        config.project_root.join("main.rs"),
        &large_file_with_block_at_top("a\nb\nc"),
    )
    .unwrap();

    let router = miniswe::llm::ModelRouter::new(&config);
    let args = serde_json::json!({"path": "main.rs", "task": "change first block"});
    let result = edit_file::execute(&args, &config, &router, None, None, None)
        .await
        .unwrap();

    assert!(result.success, "{}", result.content);
    assert!(result.content.contains("Pre-plan repair attempt 2"));
    // plan1: window + recon + finalize + 3 failed patches = 6
    // plan2: window + recon + finalize + 1 success patch  = 4
    assert_eq!(calls.load(Ordering::SeqCst), 10);
    assert!(
        fs::read_to_string(config.project_root.join("main.rs"))
            .unwrap()
            .starts_with("x\nc\n")
    );
}

/// Build a >200-line file whose first lines are `block` and the remainder is
/// padding. Used to opt out of the small-file fast path so the windowed
/// pre-plan pass actually runs.
fn large_file_with_block_at_top(block: &str) -> String {
    let mut s = String::from(block);
    s.push('\n');
    for i in 0..250 {
        s.push_str(&format!("// pad line {i}\n"));
    }
    s
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
                    "SMART_EDIT\nREGION 1 1\nTASK: update first call\nEND\n\nSMART_EDIT\nREGION 3 3\nTASK: update last call\nEND\n",
                ),
                1 => helpers::mock_text_response("DONE_EXPLORING"),
                2 => helpers::mock_text_response(
                    "REGION 1 1\nTASK: update first call\nEND\n\nREGION 3 3\nTASK: update last call\nEND\n",
                ),
                3 => helpers::mock_text_response(
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
    let result = edit_file::execute(&args, &config, &router, None, None, None)
        .await
        .unwrap();

    assert!(result.success, "{}", result.content);
    assert!(
        result
            .content
            .contains("✓ via pre-plan: 2/2 step(s) completed")
    );
    assert!(result.content.contains("Pre-plan attempt 1"));
    assert!(result.content.contains("Raw Pre-plan attempt 1"));
    assert!(result.content.contains("SMART_EDIT"));
    assert_eq!(calls.load(Ordering::SeqCst), 5);
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "call_a(None);\nkeep();\ncall_c(None);\n"
    );
}

#[tokio::test]
async fn execute_preplan_can_inspect_with_search_and_read() {
    // Large file → window emits a note, recon emits SEARCH/READ + DONE,
    // finalize plans the edit. Tests that recon can collect inspection
    // commands and that their results reach the planner.
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                // Window: notes only.
                0 => helpers::mock_text_response("NOTE update first and last call"),
                // Recon: queue SEARCH/READ alongside DONE so it terminates after one round.
                1 => helpers::mock_text_response("SEARCH: call_a\nREAD: 1-3\nDONE"),
                // Finalize: emit the actual edit plan.
                2 => helpers::mock_text_response(
                    "SMART_EDIT\nREGION 1 1\nTASK: update first call\nEND\n\nSMART_EDIT\nREGION 3 3\nTASK: update last call\nEND\n",
                ),
                // Patches — steps run in descending order, so region 3 first.
                3 => helpers::mock_text_response(
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
        &large_file_with_block_at_top("call_a();\nkeep();\ncall_c();"),
    )
    .unwrap();

    let router = miniswe::llm::ModelRouter::new(&config);
    let args = serde_json::json!({
        "path": "main.rs",
        "task": "update all call sites",
        "lsp_validation": "off"
    });
    let result = edit_file::execute(&args, &config, &router, None, None, None)
        .await
        .unwrap();

    assert!(result.success, "{}", result.content);
    assert!(result.content.contains("✓ via pre-plan: 2/2 step(s) completed"));
    // window + recon + finalize + 2 patches = 5
    assert_eq!(calls.load(Ordering::SeqCst), 5);
    assert!(
        fs::read_to_string(config.project_root.join("main.rs"))
            .unwrap()
            .starts_with("call_a(None);\nkeep();\ncall_c(None);\n")
    );
}

#[tokio::test]
async fn execute_preplan_can_handle_multiple_inspection_commands_in_one_response() {
    // The recon phase returns multiple SEARCH and READ commands in a single
    // response and they all get collected and batch-executed before finalize.
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                0 => helpers::mock_text_response("NOTE update both call sites"),
                1 => helpers::mock_text_response(
                    "SEARCH: call_a();\nSEARCH: call_c();\nREAD: 1-3\nDONE",
                ),
                2 => helpers::mock_text_response(
                    "SMART_EDIT\nREGION 1 1\nTASK: update first call\nEND\n\nSMART_EDIT\nREGION 3 3\nTASK: update last call\nEND\n",
                ),
                3 => helpers::mock_text_response(
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
        &large_file_with_block_at_top("call_a();\nkeep();\ncall_c();"),
    )
    .unwrap();

    let router = miniswe::llm::ModelRouter::new(&config);
    let args = serde_json::json!({
        "path": "main.rs",
        "task": "update all call sites",
        "lsp_validation": "off"
    });
    let result = edit_file::execute(&args, &config, &router, None, None, None)
        .await
        .unwrap();

    assert!(result.success, "{}", result.content);
    // window + recon + finalize + 2 patches = 5
    assert_eq!(calls.load(Ordering::SeqCst), 5);
    assert!(
        fs::read_to_string(config.project_root.join("main.rs"))
            .unwrap()
            .starts_with("call_a(None);\nkeep();\ncall_c(None);\n")
    );
}

#[tokio::test]
async fn execute_preplan_uses_literal_replacements_before_smart_edits() {
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                0 => helpers::mock_text_response(
                    "LITERAL_REPLACE\nSCOPE 1 3\nALL true\nOLD:\ncall(None)\nEND_OLD\nNEW:\ncall(None, None)\nEND_NEW\nEND\n\nSMART_EDIT\nREGION 4 6\nTASK: update multi-line call\nEND\n",
                ),
                1 => helpers::mock_text_response("DONE_EXPLORING"),
                2 => helpers::mock_text_response(
                    "LITERAL_REPLACE\nSCOPE 1 3\nALL true\nOLD:\ncall(None)\nEND_OLD\nNEW:\ncall(None, None)\nEND_NEW\nEND\n\nSMART_EDIT\nREGION 4 6\nTASK: update multi-line call\nEND\n",
                ),
                _ => helpers::mock_text_response(
                    "REPLACE_AT 4\nOLD:\ncall(\n    None,\n)\nEND_OLD\nNEW:\ncall(\n    None,\n    None,\n)\nEND_NEW\n",
                ),
            }
        })
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    fs::write(
        config.project_root.join("main.rs"),
        "call(None)\nkeep();\ncall(None)\ncall(\n    None,\n)\n",
    )
    .unwrap();

    let router = miniswe::llm::ModelRouter::new(&config);
    let args = serde_json::json!({
        "path": "main.rs",
        "task": "update all call sites",
        "lsp_validation": "off"
    });
    let result = edit_file::execute(&args, &config, &router, None, None, None)
        .await
        .unwrap();

    assert!(result.success, "{}", result.content);
    assert!(result.content.contains("Raw Pre-plan attempt 1"));
    assert!(result.content.contains("LITERAL_REPLACE"));
    assert!(result.content.contains("SMART_EDIT"));
    assert!(
        result
            .content
            .contains("literal L1-L3: replaced 2 occurrence")
    );
    assert!(result.content.contains("smart L4-L6: applied 1 operation"));
    assert_eq!(calls.load(Ordering::SeqCst), 4);
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "call(None, None)\nkeep();\ncall(None, None)\ncall(\n    None,\n    None,\n)\n"
    );
}

#[tokio::test]
async fn execute_preplan_applies_literal_replace_without_old() {
    // Small file. The plan emits a LITERAL_REPLACE block that omits the
    // OLD: section — i.e. the new ReplaceScope form. The executor should
    // wholesale-replace the SCOPE range with the NEW content without
    // any LLM round-trip for a smart fallback.
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                0 => helpers::mock_text_response("NOTE noop"),
                1 => helpers::mock_text_response("DONE"),
                2 => helpers::mock_text_response(
                    "LITERAL_REPLACE\nSCOPE 2 3\nALL true\nNEW:\nfresh two\nfresh three\nEND_NEW\nEND\n",
                ),
                _ => unreachable!("scope replace should not need a smart fallback"),
            }
        })
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    fs::write(
        config.project_root.join("main.rs"),
        "line one\nline two\nline three\nline four\n",
    )
    .unwrap();

    let router = miniswe::llm::ModelRouter::new(&config);
    let args = serde_json::json!({
        "path": "main.rs",
        "task": "rewrite middle two lines",
        "lsp_validation": "off"
    });
    let result = edit_file::execute(&args, &config, &router, None, None, None)
        .await
        .unwrap();

    assert!(result.success, "{}", result.content);
    assert!(result.content.contains("scope L2-L3"));
    // window + recon + finalize, no patch round needed = 3
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "line one\nfresh two\nfresh three\nline four\n"
    );
}

#[tokio::test]
async fn execute_preplan_literal_step_falls_back_to_smart_edit() {
    // Small file. The literal step's OLD doesn't match the file content,
    // so the smart fallback kicks in.
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                0 => helpers::mock_text_response("NOTE noop"),
                1 => helpers::mock_text_response("DONE"),
                2 => helpers::mock_text_response(
                    "LITERAL_REPLACE\nSCOPE 1 1\nALL false\nOLD:\ncall(None)\nEND_OLD\nNEW:\ncall(None, None)\nEND_NEW\nEND\n",
                ),
                _ => helpers::mock_text_response(
                    "REPLACE_AT 1\nOLD:\ncall( None)\nEND_OLD\nNEW:\ncall(None, None)\nEND_NEW\n",
                ),
            }
        })
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    fs::write(config.project_root.join("main.rs"), "call( None)\n").unwrap();

    let router = miniswe::llm::ModelRouter::new(&config);
    let args = serde_json::json!({
        "path": "main.rs",
        "task": "update all calls",
        "lsp_validation": "off"
    });
    let result = edit_file::execute(&args, &config, &router, None, None, None)
        .await
        .unwrap();

    assert!(result.success, "{}", result.content);
    assert!(result.content.contains("literal fallback L1-L1"));
    // window + recon + finalize + smart-fallback patch = 4
    assert_eq!(calls.load(Ordering::SeqCst), 4);
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "call(None, None)\n"
    );
}

#[tokio::test]
async fn execute_preplan_repairs_whole_plan_after_step_retries_fail() {
    // Plan 1 has a literal whose OLD doesn't match. The smart fallback
    // returns NO_CHANGES on every attempt and exhausts, so the whole plan
    // is repaired.
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                // Plan attempt 1
                0 => helpers::mock_text_response("NOTE noop"),
                1 => helpers::mock_text_response("DONE"),
                2 => helpers::mock_text_response(
                    "LITERAL_REPLACE\nSCOPE 1 1\nALL false\nOLD:\nmissing(None)\nEND_OLD\nNEW:\ncall(None, None)\nEND_NEW\nEND\n",
                ),
                3..=5 => helpers::mock_text_response("NO_CHANGES"),
                // Plan attempt 2 (repair)
                6 => helpers::mock_text_response("NOTE noop"),
                7 => helpers::mock_text_response("DONE"),
                8 => helpers::mock_text_response(
                    "LITERAL_REPLACE\nSCOPE 1 1\nALL false\nOLD:\ncall(None)\nEND_OLD\nNEW:\ncall(None, None)\nEND_NEW\nEND\n",
                ),
                _ => unreachable!("unexpected extra LLM call"),
            }
        })
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    fs::write(config.project_root.join("main.rs"), "call(None)\n").unwrap();

    let router = miniswe::llm::ModelRouter::new(&config);
    let args = serde_json::json!({
        "path": "main.rs",
        "task": "update all calls",
        "lsp_validation": "off"
    });
    let result = edit_file::execute(&args, &config, &router, None, None, None)
        .await
        .unwrap();

    assert!(result.success, "{}", result.content);
    assert!(result.content.contains("Pre-plan repair attempt 2"));
    assert!(result.content.contains("Raw Pre-plan repair attempt 2"));
    assert!(result.content.contains("OLD:\ncall(None)"));
    // plan1: 3 preplan + 3 smart-fallback NO_CHANGES + plan2: 3 preplan = 9
    assert_eq!(calls.load(Ordering::SeqCst), 9);
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "call(None, None)\n"
    );
}

#[tokio::test]
async fn execute_preplan_repair_attempt_includes_structured_repair_context() {
    // Plan 1 has a literal that doesn't match. After 3 smart-fallback
    // NO_CHANGES the whole plan is repaired. Plan 2 fixes the file with a
    // working literal. The plan-2 finalize prompt should carry the
    // structured repair-context block — that's what this test asserts on
    // top of the existing repair flow coverage.
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                // Plan attempt 1
                0 => helpers::mock_text_response("NOTE noop"),
                1 => helpers::mock_text_response("DONE"),
                2 => helpers::mock_text_response(
                    "LITERAL_REPLACE\nSCOPE 1 1\nALL false\nOLD:\nmissing(None)\nEND_OLD\nNEW:\ncall(None, None)\nEND_NEW\nEND\n",
                ),
                3..=5 => helpers::mock_text_response("NO_CHANGES"),
                // Plan attempt 2 (repair)
                6 => helpers::mock_text_response("NOTE noop"),
                7 => helpers::mock_text_response("DONE"),
                8 => helpers::mock_text_response(
                    "LITERAL_REPLACE\nSCOPE 1 1\nALL false\nOLD:\ncall(None)\nEND_OLD\nNEW:\ncall(None, None)\nEND_NEW\nEND\n",
                ),
                _ => unreachable!("unexpected extra LLM call"),
            }
        })
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    fs::write(config.project_root.join("main.rs"), "call(None)\n").unwrap();

    let router = miniswe::llm::ModelRouter::new(&config);
    let args = serde_json::json!({
        "path": "main.rs",
        "task": "update all calls",
        "lsp_validation": "off"
    });
    let result = edit_file::execute(&args, &config, &router, None, None, None)
        .await
        .unwrap();

    assert!(result.success, "{}", result.content);
    // plan1: 3 preplan + 3 smart-fallback + plan2: 3 preplan = 9
    assert_eq!(calls.load(Ordering::SeqCst), 9);

    // Inspect the body of the plan-2 finalize request (the 9th, index 8)
    // and verify the structured repair-context block reached the model.
    let requests = mock_server
        .received_requests()
        .await
        .expect("mock server should record requests");
    assert_eq!(requests.len(), 9);
    let plan2_body = String::from_utf8(requests[8].body.clone()).unwrap();

    // Marker text from format_repair_context.
    assert!(
        plan2_body.contains("A previous edit plan was attempted and failed."),
        "plan-2 prompt missing repair-context preface: {plan2_body}"
    );
    assert!(
        plan2_body.contains("Previous edit plan (as tried):"),
        "plan-2 prompt missing previous-plan section"
    );
    assert!(
        plan2_body.contains("Steps that succeeded and have ALREADY been applied"),
        "plan-2 prompt missing completed-steps section"
    );
    assert!(
        plan2_body.contains("Step that FAILED:"),
        "plan-2 prompt missing failed-step section"
    );
    // The literal we tried in plan 1 should appear in the previous plan
    // section, with its non-matching OLD payload.
    assert!(
        plan2_body.contains("missing(None)"),
        "plan-2 prompt should echo the failed plan's OLD payload"
    );
    // Plan 1 had no successful steps, so the completed section should be
    // the explicit empty stub, not silently absent.
    assert!(
        plan2_body.contains("(none — the first step failed"),
        "plan-2 prompt should mark completed-steps as empty when plan 1 made zero progress"
    );
    assert!(
        plan2_body.contains("Failure reason:"),
        "plan-2 prompt missing failure-reason section"
    );

    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "call(None, None)\n"
    );
}

#[tokio::test]
async fn execute_preplan_invalid_task_short_circuits_with_reason() {
    // The finalize call returns INVALID_TASK with a reason. The retry loop
    // must short-circuit after the planning phase, the file must be left
    // unchanged, and the tool result must surface the rejection reason.
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                0 => helpers::mock_text_response("NOTE noop"),
                1 => helpers::mock_text_response("DONE"),
                2 => helpers::mock_text_response(
                    "INVALID_TASK: file does not contain any auth code to update",
                ),
                _ => unreachable!("INVALID_TASK should short-circuit the retry loop"),
            }
        })
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    fs::write(config.project_root.join("main.rs"), "println!(\"hi\");\n").unwrap();

    let router = miniswe::llm::ModelRouter::new(&config);
    let args = serde_json::json!({
        "path": "main.rs",
        "task": "remove the auth middleware",
        "lsp_validation": "off"
    });
    let result = edit_file::execute(&args, &config, &router, None, None, None)
        .await
        .unwrap();

    assert!(!result.success, "INVALID_TASK should surface as a failure");
    assert!(
        result.content.contains("rejected task as invalid"),
        "expected rejection marker, got: {}",
        result.content
    );
    assert!(
        result
            .content
            .contains("file does not contain any auth code to update"),
        "expected reason in tool output, got: {}",
        result.content
    );
    assert!(
        result.content.contains("file was not modified"),
        "expected unmodified note, got: {}",
        result.content
    );
    // window + recon + finalize = 3, no retries after INVALID_TASK.
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    // File on disk unchanged.
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "println!(\"hi\");\n"
    );
}

#[tokio::test]
async fn execute_preplan_invalid_task_without_reason_uses_placeholder() {
    // Bare `INVALID_TASK` (no colon, no reason) at the finalize phase
    // should still short-circuit and surface a "no reason provided"
    // placeholder rather than an empty string in the user-facing message.
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                0 => helpers::mock_text_response("NOTE noop"),
                1 => helpers::mock_text_response("DONE"),
                2 => helpers::mock_text_response("INVALID_TASK"),
                _ => unreachable!("INVALID_TASK should short-circuit the retry loop"),
            }
        })
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    fs::write(config.project_root.join("main.rs"), "println!(\"hi\");\n").unwrap();

    let router = miniswe::llm::ModelRouter::new(&config);
    let args = serde_json::json!({
        "path": "main.rs",
        "task": "incoherent task",
        "lsp_validation": "off"
    });
    let result = edit_file::execute(&args, &config, &router, None, None, None)
        .await
        .unwrap();

    assert!(!result.success);
    assert!(result.content.contains("no reason provided"));
    // window + recon + finalize = 3
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "println!(\"hi\");\n"
    );
}

#[tokio::test]
async fn execute_preplan_invalid_task_during_repair_short_circuits() {
    // Plan 1 fails to apply (literal OLD doesn't match, smart fallback
    // returns NO_CHANGES three times). On the repair attempt, the model
    // realizes the task is impossible and emits INVALID_TASK. The retry
    // loop must stop immediately and surface the reason — no further
    // planning attempts after the rejection.
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                // Plan attempt 1
                0 => helpers::mock_text_response("NOTE noop"),
                1 => helpers::mock_text_response("DONE"),
                2 => helpers::mock_text_response(
                    "LITERAL_REPLACE\nSCOPE 1 1\nALL false\nOLD:\nmissing(None)\nEND_OLD\nNEW:\ncall(None, None)\nEND_NEW\nEND\n",
                ),
                3..=5 => helpers::mock_text_response("NO_CHANGES"),
                // Plan attempt 2 (repair) — model decides the task is impossible.
                6 => helpers::mock_text_response("NOTE noop"),
                7 => helpers::mock_text_response("DONE"),
                8 => helpers::mock_text_response(
                    "INVALID_TASK: file structure does not match the task description",
                ),
                _ => unreachable!("INVALID_TASK should short-circuit the retry loop"),
            }
        })
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    fs::write(config.project_root.join("main.rs"), "call(None)\n").unwrap();

    let router = miniswe::llm::ModelRouter::new(&config);
    let args = serde_json::json!({
        "path": "main.rs",
        "task": "update all calls",
        "lsp_validation": "off"
    });
    let result = edit_file::execute(&args, &config, &router, None, None, None)
        .await
        .unwrap();

    assert!(!result.success, "{}", result.content);
    assert!(
        result.content.contains("rejected task as invalid"),
        "expected rejection marker, got: {}",
        result.content
    );
    assert!(
        result
            .content
            .contains("file structure does not match the task description"),
        "expected reason in tool output, got: {}",
        result.content
    );
    // plan1: 3 preplan + 3 fallback NO_CHANGES + plan2: 3 preplan (final = INVALID_TASK) = 9
    assert_eq!(calls.load(Ordering::SeqCst), 9);
    // File unchanged — plan 1's literal didn't match, so no edits applied.
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "call(None)\n"
    );
}

#[tokio::test]
async fn execute_preplan_invalid_task_with_missing_target_reason_is_suppressed_and_retries() {
    // The model rejects with INVALID_TASK but the reason is "target not
    // found in current state" — that's a missing prerequisite, NOT an
    // incoherent task. We should suppress the rejection, set up a repair
    // context with a hint, and retry. On the second attempt the model
    // produces a valid plan and the edit applies.
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                // Plan attempt 1: window, recon (DONE), finalize → false-positive INVALID_TASK
                0 => helpers::mock_text_response("NOTE noop"),
                1 => helpers::mock_text_response("DONE"),
                2 => helpers::mock_text_response(
                    "INVALID_TASK: target function `foo` not found in the file",
                ),
                // Plan attempt 2 (after suppression + repair context): window, recon, finalize → real plan
                3 => helpers::mock_text_response("NOTE understood"),
                4 => helpers::mock_text_response("DONE"),
                5 => helpers::mock_text_response(
                    "LITERAL_REPLACE\nSCOPE 1 1\nALL false\nOLD:\noriginal();\nEND_OLD\nNEW:\nfoo();\nEND_NEW\nEND\n",
                ),
                _ => panic!("unexpected extra call #{n} — repair retry should have succeeded"),
            }
        })
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    fs::write(config.project_root.join("main.rs"), "original();\n").unwrap();

    let router = miniswe::llm::ModelRouter::new(&config);
    let args = serde_json::json!({
        "path": "main.rs",
        "task": "rename original to foo",
        "lsp_validation": "off"
    });
    let result = edit_file::execute(&args, &config, &router, None, None, None)
        .await
        .unwrap();

    assert!(
        result.success,
        "expected success after suppression+repair, got: {}",
        result.content
    );
    assert!(
        result.content.contains("suppressed false-positive INVALID_TASK"),
        "expected suppression note in tool output, got: {}",
        result.content
    );
    // plan1: 3 preplan calls (rejected) + plan2: 3 preplan calls (succeeded) = 6 total
    assert_eq!(calls.load(Ordering::SeqCst), 6);
    // File should now contain the renamed call.
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "foo();\n"
    );
}

#[tokio::test]
async fn execute_preplan_invalid_task_with_real_incoherent_reason_still_short_circuits() {
    // Counterpoint to the suppression test: a "real" INVALID_TASK reason
    // that doesn't mention any missing/not-found phrases should still
    // short-circuit the loop (no retry) and surface the rejection.
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                0 => helpers::mock_text_response("NOTE noop"),
                1 => helpers::mock_text_response("DONE"),
                2 => helpers::mock_text_response(
                    "INVALID_TASK: the task asks to write Python in this Rust file",
                ),
                _ => unreachable!(
                    "real INVALID_TASK should short-circuit the retry loop without further calls"
                ),
            }
        })
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    fs::write(config.project_root.join("main.rs"), "println!(\"hi\");\n").unwrap();

    let router = miniswe::llm::ModelRouter::new(&config);
    let args = serde_json::json!({
        "path": "main.rs",
        "task": "convert this file to Python",
        "lsp_validation": "off"
    });
    let result = edit_file::execute(&args, &config, &router, None, None, None)
        .await
        .unwrap();

    assert!(!result.success);
    assert!(
        result.content.contains("rejected task as invalid"),
        "expected rejection marker, got: {}",
        result.content
    );
    assert!(
        result.content.contains("write Python in this Rust file"),
        "expected reason in tool output, got: {}",
        result.content
    );
    // window + recon + finalize = 3 calls, no retry.
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "println!(\"hi\");\n"
    );
}

#[tokio::test]
async fn execute_preplan_parse_failure_returns_error() {
    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(helpers::mock_text_response("I would edit lines 1-2."))
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
    let result = edit_file::execute(&args, &config, &router, None, None, None)
        .await
        .unwrap();

    assert!(!result.success);
    assert!(result.content.contains("unexpected text in edit plan"));
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "old();\n"
    );
}

#[tokio::test]
async fn execute_preplan_overlapping_steps_apply_first_and_report_rest_failed() {
    // The planner emits two SMART_EDIT steps that overlap (shared line 3).
    // The first wins by source order, the second is reported as a failed
    // step in the per-step output. The overall edit_file call still
    // succeeds because at least one step applied.
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                // window phase
                0 => helpers::mock_text_response("NOTE noop"),
                // recon phase
                1 => helpers::mock_text_response("DONE"),
                // finalize: two overlapping SMART_EDIT steps
                2 => helpers::mock_text_response(
                    "SMART_EDIT\nREGION 1 3\nTASK: rewrite the first three lines\nEND\n\
                     \n\
                     SMART_EDIT\nREGION 3 5\nTASK: rewrite lines three through five\nEND\n",
                ),
                // smart-edit patch for the kept (first) step L1-L3
                _ => helpers::mock_text_response(
                    "REPLACE_AT 1\nOLD:\nalpha\nbeta\ngamma\nEND_OLD\nNEW:\nALPHA\nBETA\nGAMMA\nEND_NEW\n",
                ),
            }
        })
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    fs::write(
        config.project_root.join("main.rs"),
        "alpha\nbeta\ngamma\ndelta\nepsilon\n",
    )
    .unwrap();

    let router = miniswe::llm::ModelRouter::new(&config);
    let args = serde_json::json!({
        "path": "main.rs",
        "task": "rewrite the file",
        "lsp_validation": "off"
    });
    let result = edit_file::execute(&args, &config, &router, None, None, None)
        .await
        .unwrap();

    // The kept step applied — overall success.
    assert!(result.success, "{}", result.content);
    // Summary reflects 1/2 with the dropped count called out.
    assert!(
        result.content.contains("1/2 step(s) completed"),
        "missing 1/2 summary in: {}",
        result.content
    );
    assert!(
        result.content.contains("1 dropped (overlap)"),
        "missing dropped (overlap) note in: {}",
        result.content
    );
    // Per-step output: the dropped step is reported as a FAILED step
    // pointing at the kept one.
    assert!(
        result.content.contains("Pre-plan step L3-L5: FAILED"),
        "missing per-step failure for dropped overlap in: {}",
        result.content
    );
    assert!(
        result.content.contains("overlaps earlier step L1-L3"),
        "missing overlap reason in: {}",
        result.content
    );
    // The kept step actually wrote the file.
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "ALPHA\nBETA\nGAMMA\ndelta\nepsilon\n"
    );
}

#[tokio::test]
async fn execute_no_changes_leaves_file_unchanged() {
    // The mock returns NO_CHANGES for every call. The window parser ignores
    // it (not a NOTE), the recon parser sees no commands and terminates,
    // the finalize parser yields zero steps → file untouched.
    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(helpers::mock_text_response("NO_CHANGES"))
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    fs::write(config.project_root.join("main.rs"), "original\n").unwrap();

    let router = miniswe::llm::ModelRouter::new(&config);
    let args = serde_json::json!({"path": "main.rs", "task": "no op"});
    let result = edit_file::execute(&args, &config, &router, None, None, None)
        .await
        .unwrap();

    assert!(result.success);
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "original\n"
    );
}

#[tokio::test]
async fn execute_empty_file_skips_window_and_recon_phases() {
    // For an empty file there is nothing to observe and nothing to inspect,
    // so Phase 1 (window) and Phase 2 (recon) are skipped entirely and the
    // first LLM call is the planning prompt directly. Without the skip, the
    // model would have been called with a degenerate 0-line window slice
    // ("This is slice 1 of 1, covering lines 1-0 of 0").
    //
    // We make the planner short-circuit with INVALID_TASK so the test only
    // exercises the preplan phase pipeline, not downstream edit application.
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            calls_for_mock.fetch_add(1, Ordering::SeqCst);
            helpers::mock_text_response("INVALID_TASK: empty file, nothing to do")
        })
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    // Create a truly empty file (0 bytes, 0 lines).
    fs::write(config.project_root.join("main.rs"), "").unwrap();

    let router = miniswe::llm::ModelRouter::new(&config);
    let args = serde_json::json!({
        "path": "main.rs",
        "task": "create a hello file",
        "lsp_validation": "off",
    });
    let result = edit_file::execute(&args, &config, &router, None, None, None)
        .await
        .unwrap();

    // INVALID_TASK short-circuits → exactly one LLM call total.
    // That single call MUST be the finalize prompt — no window, no recon.
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "empty file should skip window+recon and go straight to finalize"
    );

    // Verify the one call we made was the planning prompt, not the
    // observation or reconnaissance prompt. Each phase has a distinct
    // system message that we can match against.
    let requests = mock_server
        .received_requests()
        .await
        .expect("mock server should record requests");
    assert_eq!(requests.len(), 1);
    let body = String::from_utf8(requests[0].body.clone()).unwrap();
    assert!(
        body.contains("Final planning phase"),
        "first call should be the finalize prompt, got: {body}"
    );
    assert!(
        !body.contains("Observation phase"),
        "first call should NOT be the observation prompt"
    );
    assert!(
        !body.contains("Reconnaissance phase"),
        "first call should NOT be the recon prompt"
    );

    // The planner rejected the task, so the file remains empty.
    assert!(!result.success);
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        ""
    );
}

#[tokio::test]
async fn execute_lsp_off_succeeds_without_lsp_client() {
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                0 => helpers::mock_text_response(
                    "SMART_EDIT\nREGION 1\nTASK: change line\nEND\n",
                ),
                1 => helpers::mock_text_response("DONE_EXPLORING"),
                2 => helpers::mock_text_response(
                    "SMART_EDIT\nREGION 1\nTASK: change line\nEND\n",
                ),
                _ => helpers::mock_text_response(
                    "REPLACE_AT 1\nOLD:\noriginal\nEND_OLD\nNEW:\nfixed\nEND_NEW\n",
                ),
            }
        })
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
    let result = edit_file::execute(&args, &config, &router, None, None, None)
        .await
        .unwrap();

    assert!(result.success, "{}", result.content);
    assert!(result.content.contains("[lsp] skipped (off)"));
    assert_eq!(calls.load(Ordering::SeqCst), 4);
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
    let result = edit_file::execute(&args, &config, &router, None, None, None)
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
async fn execute_edit_file_tool_reindexes_successful_edit() {
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                0 => helpers::mock_text_response(
                    "SMART_EDIT\nREGION 1\nTASK: rename original to replacement\nEND\n",
                ),
                1 => helpers::mock_text_response("DONE_EXPLORING"),
                2 => helpers::mock_text_response(
                    "SMART_EDIT\nREGION 1\nTASK: rename original to replacement\nEND\n",
                ),
                _ => helpers::mock_text_response(
                    "REPLACE_AT 1\nOLD:\npub fn original() {}\nEND_OLD\nNEW:\npub fn replacement() {}\nEND_NEW\n",
                ),
            }
        })
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
    let result =
        tools::execute_edit_file_tool(&args, &config, &perms, &router, None, None, None)
        .await
        .unwrap();

    assert!(result.success, "{}", result.content);
    assert_eq!(calls.load(Ordering::SeqCst), 4);
    let updated_index = ProjectIndex::load(&config.miniswe_dir()).unwrap();
    assert!(
        !updated_index.lookup("replacement").is_empty(),
        "Index should contain replacement after edit_file reindex"
    );
    assert!(
        updated_index.lookup("original").is_empty(),
        "Index should no longer contain original after edit_file reindex"
    );
}

// ── Window building ───────────────────────────────────────────────

#[test]
fn single_window_for_small_file() {
    let windows = edit_file::build_windows(100, 800, 100);
    assert_eq!(windows, vec![(0, 100)]);
}

#[test]
fn windows_cover_entire_file() {
    let windows = edit_file::build_windows(1500, 800, 100);
    for line in 0..1500 {
        let covered = windows.iter().any(|(s, e)| line >= *s && line < *e);
        assert!(covered, "line {line} not covered by any window");
    }
}
