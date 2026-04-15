//! Tests for edit_file patch parsing, atomic application, and repair behavior.

mod helpers;

#[path = "edit_file_parts/execute_preplan.rs"]
mod execute_preplan;

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
    // Small file → the pre-plan pipeline skips the windowed observation
    // pass. Call 0: finalize emits a plan. Call 1: smart-edit patch.
    // Call 2: finalize re-asks "is the task done?" and the model emits
    // COMPLETE — that terminal verdict is what the new loop requires
    // before it'll report success.
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                0 => helpers::mock_text_response(
                    "SMART_EDIT\nREGION 1 2\nTASK: add setup call\nEND\n",
                ),
                1 => helpers::mock_text_response("INSERT_AFTER 1\nCONTENT:\n    added();\nEND\n"),
                _ => helpers::mock_text_response("COMPLETE\n"),
            }
        })
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    fs::write(config.project_root.join("main.rs"), "fn main() {\n}\n").unwrap();

    let router = miniswe::llm::ModelRouter::new(&config);
    let args = serde_json::json!({"path": "main.rs", "task": "add setup call"});
    let result = edit_file::execute(&args, &config, &router, None, None, None, None, None)
        .await
        .unwrap();

    assert!(result.success, "{}", result.content);
    assert_eq!(result.content.trim(), "✓ edit_file(main.rs): done");
    // finalize + smart-edit patch + verdict-finalize = 3
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "fn main() {\n    added();\n}\n"
    );
}

#[tokio::test]
async fn execute_failed_patch_writes_nothing() {
    // 1-line file → small-file fast path. Each plan attempt now runs
    // exactly one finalize call plus a single patch attempt (no inner
    // retry). 4 attempts × 2 calls = 8.
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n % 2 {
                0 => helpers::mock_text_response("SMART_EDIT\nREGION 1\nTASK: change line\nEND\n"),
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
    let result = edit_file::execute(&args, &config, &router, None, None, None, None, None)
        .await
        .unwrap();

    assert!(!result.success);
    assert!(
        result
            .content
            .contains("✗ edit_file(main.rs): could not complete after 4 attempts"),
        "expected budget-exhausted failure template, got: {}",
        result.content
    );
    assert!(
        result.content.contains("Last obstacle:"),
        "expected last-obstacle trailer, got: {}",
        result.content
    );
    // 4 plan attempts × (finalize + 1 patch attempt) = 8
    assert_eq!(calls.load(Ordering::SeqCst), 8);
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "original\n"
    );
}

#[tokio::test]
async fn execute_repairs_failed_first_patch() {
    // 1-line file (small-file fast path). Plan 1: finalize + 1 failing
    // patch (no inner retry). Plan 2 (repair): finalize + 1 successful
    // patch.
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                // Plan attempt 1
                0 => helpers::mock_text_response("SMART_EDIT\nREGION 1\nTASK: change line\nEND\n"),
                1 => helpers::mock_text_response(
                    "REPLACE_AT 1\nOLD:\nwrong\nEND_OLD\nNEW:\nfixed\nEND_NEW\n",
                ),
                // Plan attempt 2 (repair)
                2 => helpers::mock_text_response("SMART_EDIT\nREGION 1\nTASK: change line\nEND\n"),
                3 => helpers::mock_text_response(
                    "REPLACE_AT 1\nOLD:\noriginal\nEND_OLD\nNEW:\nfixed\nEND_NEW\n",
                ),
                // Terminal verdict after the successful patch execution.
                _ => helpers::mock_text_response("COMPLETE\n"),
            }
        })
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    fs::write(config.project_root.join("main.rs"), "original\n").unwrap();

    let router = miniswe::llm::ModelRouter::new(&config);
    let args = serde_json::json!({"path": "main.rs", "task": "change line"});
    let result = edit_file::execute(&args, &config, &router, None, None, None, None, None)
        .await
        .unwrap();

    assert!(result.success, "{}", result.content);
    assert_eq!(result.content.trim(), "✓ edit_file(main.rs): done");
    // plan1: finalize + 1 failed patch  = 2
    // plan2: finalize + 1 success patch = 2
    // verdict-finalize                   = 1
    assert_eq!(calls.load(Ordering::SeqCst), 5);
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "fixed\n"
    );
}

#[tokio::test]
async fn execute_repairs_until_third_patch() {
    // 1-line file (small-file fast path). Plans 1 and 2 fail their
    // single patch attempt (no inner retry). Plan 3 succeeds on its
    // first patch.
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                // Per-attempt preplan: just the finalize call in the small-file fast path.
                0 | 2 | 4 => {
                    helpers::mock_text_response("SMART_EDIT\nREGION 1\nTASK: change line\nEND\n")
                }
                // Plans 1 and 2 fail their single patch attempt.
                1 | 3 => helpers::mock_text_response(
                    "REPLACE_AT 1\nOLD:\nwrong\nEND_OLD\nNEW:\nfixed\nEND_NEW\n",
                ),
                // Plan 3's first patch succeeds.
                5 => helpers::mock_text_response(
                    "REPLACE_AT 1\nOLD:\noriginal\nEND_OLD\nNEW:\nfixed\nEND_NEW\n",
                ),
                // Terminal verdict after the successful patch execution.
                _ => helpers::mock_text_response("COMPLETE\n"),
            }
        })
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    fs::write(config.project_root.join("main.rs"), "original\n").unwrap();

    let router = miniswe::llm::ModelRouter::new(&config);
    let args = serde_json::json!({"path": "main.rs", "task": "change line"});
    let result = edit_file::execute(&args, &config, &router, None, None, None, None, None)
        .await
        .unwrap();

    assert!(result.success, "{}", result.content);
    assert_eq!(result.content.trim(), "✓ edit_file(main.rs): done");
    // plan1: 1 + 1, plan2: 1 + 1, plan3: 1 + 1 = 6, plus terminal verdict = 7
    assert_eq!(calls.load(Ordering::SeqCst), 7);
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "fixed\n"
    );
}

#[tokio::test]
async fn execute_no_changes_leaves_file_unchanged() {
    // The mock returns NO_CHANGES for every call. In the small-file fast path
    // the only preplan call is finalize, whose parser treats NO_CHANGES as an
    // empty plan → zero steps → file untouched.
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            calls_for_mock.fetch_add(1, Ordering::SeqCst);
            helpers::mock_text_response("NO_CHANGES")
        })
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    fs::write(config.project_root.join("main.rs"), "original\n").unwrap();

    let router = miniswe::llm::ModelRouter::new(&config);
    let args = serde_json::json!({"path": "main.rs", "task": "no op"});
    let result = edit_file::execute(&args, &config, &router, None, None, None, None, None)
        .await
        .unwrap();

    assert!(result.success);
    // Exactly one finalize call — NO_CHANGES is the legitimate "nothing to do"
    // signal and must NOT trigger a retry loop. (Empty responses now do.)
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "NO_CHANGES should converge in a single finalize call without retrying"
    );
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "original\n"
    );
}

#[tokio::test]
async fn execute_empty_response_at_finalize_retries_and_recovers() {
    // Bench logs show the planner sometimes returns an empty response as a
    // transient pathology (stalled inference, template misfire). Historically
    // we collapsed that to an empty plan and exited the retry loop, wasting
    // the whole edit_file invocation on one bad response. Now we treat it
    // as a failure, feed the model explicit feedback via RepairContext, and
    // retry within MAX_PLAN_ATTEMPTS. This test proves the retry path works:
    // the first finalize call returns empty, the second returns a valid plan,
    // and the edit applies.
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                // Plan attempt 1: finalize → empty response (pathology).
                0 => helpers::mock_text_response(""),
                // Plan attempt 2 (after empty-response repair): finalize → real plan.
                1 => helpers::mock_text_response(
                    "LITERAL_REPLACE\nSCOPE 1 1\nALL false\nOLD:\nfoo();\nEND_OLD\nNEW:\nbar();\nEND_NEW\nEND\n",
                ),
                // Terminal verdict after the successful literal replace.
                _ => helpers::mock_text_response("COMPLETE\n"),
            }
        })
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    fs::write(config.project_root.join("main.rs"), "foo();\n").unwrap();

    let router = miniswe::llm::ModelRouter::new(&config);
    let args = serde_json::json!({
        "path": "main.rs",
        "task": "rename foo to bar",
        "lsp_validation": "off",
    });
    let result = edit_file::execute(&args, &config, &router, None, None, None, None, None)
        .await
        .unwrap();

    assert!(
        result.success,
        "expected success after empty-response retry, got: {}",
        result.content
    );
    assert_eq!(result.content.trim(), "✓ edit_file(main.rs): done");
    // plan1: 1 finalize (empty) + plan2: 1 finalize (succeeded) + verdict = 3 total
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "bar();\n"
    );
}

#[tokio::test]
async fn execute_empty_response_exhausts_retries_and_reports_failure() {
    // If every finalize call returns empty, we exhaust MAX_PLAN_ATTEMPTS and
    // report a clean failure to the agent instead of silently returning
    // "nothing to do" (the old behavior, which hid the stall).
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            calls_for_mock.fetch_add(1, Ordering::SeqCst);
            helpers::mock_text_response("")
        })
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    fs::write(config.project_root.join("main.rs"), "foo();\n").unwrap();

    let router = miniswe::llm::ModelRouter::new(&config);
    let args = serde_json::json!({
        "path": "main.rs",
        "task": "rename foo to bar",
        "lsp_validation": "off",
    });
    let result = edit_file::execute(&args, &config, &router, None, None, None, None, None)
        .await
        .unwrap();

    assert!(
        !result.success,
        "expected failure after repeated empty responses, got success: {}",
        result.content
    );
    // MAX_PLAN_ATTEMPTS = 4, so we expect 4 finalize calls before bailing.
    assert_eq!(
        calls.load(Ordering::SeqCst),
        4,
        "expected one call per plan attempt before exhaustion"
    );
    assert!(
        result.content.contains("empty response"),
        "expected empty-response reason in failure message, got: {}",
        result.content
    );
    // File untouched.
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "foo();\n"
    );
}

#[tokio::test]
async fn execute_empty_file_skips_windowed_pass() {
    // An empty file is 0 lines, which is ≤ SMALL_FILE_THRESHOLD, so the
    // small-file fast path kicks in and Phase 1 (windowed observation) is
    // skipped entirely — the first LLM call is the finalize prompt. Without
    // the skip, the model would have been called with a degenerate 0-line
    // window slice ("Slice 1/1, lines 1-0 of 0").
    //
    // We make the planner short-circuit with NEEDS_CLARIFICATION so the
    // test only exercises the preplan phase pipeline, not downstream edit
    // application.
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            calls_for_mock.fetch_add(1, Ordering::SeqCst);
            helpers::mock_text_response(
                "NEEDS_CLARIFICATION: file is empty — what contents should it have?",
            )
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
    let result = edit_file::execute(&args, &config, &router, None, None, None, None, None)
        .await
        .unwrap();

    // NEEDS_CLARIFICATION short-circuits → exactly one LLM call total.
    // That single call MUST be the finalize prompt — no windowed pass.
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "empty file should skip the windowed pass and go straight to finalize"
    );

    // Verify the one call we made was the planning prompt, not the
    // observation prompt. Each phase has a distinct system message that we
    // can match against.
    let requests = mock_server
        .received_requests()
        .await
        .expect("mock server should record requests");
    assert_eq!(requests.len(), 1);
    let body = String::from_utf8(requests[0].body.clone()).unwrap();
    assert!(
        body.contains("Verdict phase"),
        "first call should be the verdict/finalize prompt, got: {body}"
    );
    assert!(
        !body.contains("Observation phase"),
        "first call should NOT be the observation prompt"
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
    // Small file (1 line): finalize + 1 patch = 2 LLM calls.
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                0 => helpers::mock_text_response("SMART_EDIT\nREGION 1\nTASK: change line\nEND\n"),
                1 => helpers::mock_text_response(
                    "REPLACE_AT 1\nOLD:\noriginal\nEND_OLD\nNEW:\nfixed\nEND_NEW\n",
                ),
                // Terminal verdict after the successful patch.
                _ => helpers::mock_text_response("COMPLETE\n"),
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
    let result = edit_file::execute(&args, &config, &router, None, None, None, None, None)
        .await
        .unwrap();

    assert!(result.success, "{}", result.content);
    assert_eq!(result.content.trim(), "✓ edit_file(main.rs): done");
    // finalize + patch + verdict-finalize = 3
    assert_eq!(calls.load(Ordering::SeqCst), 3);
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
    let result = edit_file::execute(&args, &config, &router, None, None, None, None, None)
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
    // Small file (1 line): finalize + 1 patch = 2 LLM calls.
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
                1 => helpers::mock_text_response(
                    "REPLACE_AT 1\nOLD:\npub fn original() {}\nEND_OLD\nNEW:\npub fn replacement() {}\nEND_NEW\n",
                ),
                // Terminal verdict after the successful patch.
                _ => helpers::mock_text_response("COMPLETE\n"),
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
    let result = tools::execute_edit_file_tool(&args, &config, &perms, &router, None, None, None)
        .await
        .unwrap();

    assert!(result.success, "{}", result.content);
    // finalize + patch + verdict-finalize = 3
    assert_eq!(calls.load(Ordering::SeqCst), 3);
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
