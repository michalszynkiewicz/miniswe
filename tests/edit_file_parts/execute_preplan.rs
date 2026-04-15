//! Pre-plan retry-loop e2e tests. Included from `tests/e2e_edit_file.rs`
//! via a `#[path]` module declaration — cargo does not compile files
//! under `tests/edit_file_parts/` as their own test binaries.

use std::fs;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use miniswe::tools::edit_file;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer};

#[path = "../helpers/mod.rs"]
mod helpers;

#[tokio::test]
async fn execute_preplan_repair_after_failed_plan() {
    // 3-line file (small-file fast path). First plan generates overlapping
    // patch ops that get rejected, so the whole plan is repaired and the
    // second plan succeeds.
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                // Plan attempt 1
                0 => helpers::mock_text_response(
                    "SMART_EDIT\nREGION 1 2\nTASK: replace the first two-line block\nEND\n",
                ),
                1 => helpers::mock_text_response(
                    "REPLACE_AT 1\nOLD:\na\nb\nEND_OLD\nNEW:\nx\nEND_NEW\n\nREPLACE_AT 2\nOLD:\nb\nEND_OLD\nNEW:\ny\nEND_NEW\n",
                ),
                // Plan attempt 2 (repair)
                2 => helpers::mock_text_response(
                    "SMART_EDIT\nREGION 1 2\nTASK: replace the first two-line block\nEND\n",
                ),
                3 => helpers::mock_text_response(
                    "REPLACE_AT 1\nOLD:\na\nb\nEND_OLD\nNEW:\nx\nEND_NEW\n",
                ),
                // Terminal verdict after the successful patch execution.
                _ => helpers::mock_text_response("COMPLETE\n"),
            }
        })
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    fs::write(config.project_root.join("main.rs"), "a\nb\nc\n").unwrap();

    let router = miniswe::llm::ModelRouter::new(&config);
    let args = serde_json::json!({"path": "main.rs", "task": "change first block"});
    let result = edit_file::execute(&args, &config, &router, None, None, None, None, None)
        .await
        .unwrap();

    assert!(result.success, "{}", result.content);
    assert_eq!(result.content.trim(), "✓ edit_file(main.rs): done");
    // plan1: finalize + 1 failed patch + plan2: finalize + 1 success patch + verdict = 5
    assert_eq!(calls.load(Ordering::SeqCst), 5);
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "x\nc\n"
    );
}

#[tokio::test]
async fn execute_preplan_repair_can_inspect_with_search_and_read() {
    // Large file (>200 lines). Each plan attempt runs the windowed
    // observation pass (1 window because the file fits under WINDOW_SIZE)
    // followed by finalize. On repair, the window prompt includes step
    // outcomes (✓/✗) for the slice so the model knows what was already
    // handled and can emit SEARCH/READ for targeted re-inspection.
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                // Plan attempt 1: window + finalize, then 1 failing patch.
                0 => helpers::mock_text_response("NOTE first pass"),
                1 => helpers::mock_text_response(
                    "SMART_EDIT\nREGION 1 2\nTASK: change first block\nEND\n",
                ),
                2 => helpers::mock_text_response(
                    "REPLACE_AT 1\nOLD:\na\nb\nEND_OLD\nNEW:\nx\nEND_NEW\n\nREPLACE_AT 2\nOLD:\nb\nEND_OLD\nNEW:\ny\nEND_NEW\n",
                ),
                // Plan attempt 2 (repair): window (with step outcomes) +
                // finalize. The window prompt now shows which steps
                // succeeded/failed in this slice.
                3 => helpers::mock_text_response(
                    "NOTE revisit first block\nSEARCH: a\nREAD: 1-2",
                ),
                4 => helpers::mock_text_response(
                    "SMART_EDIT\nREGION 1 2\nTASK: change first block\nEND\n",
                ),
                5 => helpers::mock_text_response(
                    "REPLACE_AT 1\nOLD:\na\nb\nEND_OLD\nNEW:\nx\nEND_NEW\n",
                ),
                // Terminal verdict round — window + finalize.
                6 => helpers::mock_text_response(""),
                _ => helpers::mock_text_response("COMPLETE\n"),
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
    let result = edit_file::execute(&args, &config, &router, None, None, None, None, None)
        .await
        .unwrap();

    assert!(result.success, "{}", result.content);
    assert_eq!(result.content.trim(), "✓ edit_file(main.rs): done");
    // plan1: window + finalize + 1 failed patch  = 3
    // plan2: window + finalize + 1 success patch = 3
    // verdict: window + finalize                 = 2
    assert_eq!(calls.load(Ordering::SeqCst), 8);
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
    // 3-line file (small-file fast path): finalize + 2 patches (one per region).
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
                1 => helpers::mock_text_response(
                    "REPLACE_AT 3\nOLD:\ncall_c();\nEND_OLD\nNEW:\ncall_c(None);\nEND_NEW\n",
                ),
                2 => helpers::mock_text_response(
                    "REPLACE_AT 1\nOLD:\ncall_a();\nEND_OLD\nNEW:\ncall_a(None);\nEND_NEW\n",
                ),
                // Terminal verdict after the successful patch execution.
                _ => helpers::mock_text_response("COMPLETE\n"),
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
    let result = edit_file::execute(&args, &config, &router, None, None, None, None, None)
        .await
        .unwrap();

    assert!(result.success, "{}", result.content);
    assert_eq!(result.content.trim(), "✓ edit_file(main.rs): done");
    // finalize + 2 patches + verdict-finalize = 4
    assert_eq!(calls.load(Ordering::SeqCst), 4);
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "call_a(None);\nkeep();\ncall_c(None);\n"
    );
}

#[tokio::test]
async fn execute_preplan_can_inspect_with_search_and_read() {
    // Large file → window emits NOTE + SEARCH/READ commands in one response,
    // the commands are batch-executed, then finalize plans the edit. Tests
    // that the windowed pass can collect inspection commands and that their
    // results reach the planner.
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                // Window: emit a note plus SEARCH/READ commands in one response.
                0 => helpers::mock_text_response(
                    "NOTE update first and last call\nSEARCH: call_a\nREAD: 1-3",
                ),
                // Finalize: emit the actual edit plan.
                1 => helpers::mock_text_response(
                    "SMART_EDIT\nREGION 1 1\nTASK: update first call\nEND\n\nSMART_EDIT\nREGION 3 3\nTASK: update last call\nEND\n",
                ),
                // Patches — steps run in descending order, so region 3 first.
                2 => helpers::mock_text_response(
                    "REPLACE_AT 3\nOLD:\ncall_c();\nEND_OLD\nNEW:\ncall_c(None);\nEND_NEW\n",
                ),
                3 => helpers::mock_text_response(
                    "REPLACE_AT 1\nOLD:\ncall_a();\nEND_OLD\nNEW:\ncall_a(None);\nEND_NEW\n",
                ),
                // Terminal verdict round — large file still does window + finalize.
                4 => helpers::mock_text_response(""),
                _ => helpers::mock_text_response("COMPLETE\n"),
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
    let result = edit_file::execute(&args, &config, &router, None, None, None, None, None)
        .await
        .unwrap();

    assert!(result.success, "{}", result.content);
    assert_eq!(result.content.trim(), "✓ edit_file(main.rs): done");
    // window + finalize + 2 patches + verdict window + verdict finalize = 6
    assert_eq!(calls.load(Ordering::SeqCst), 6);
    assert!(
        fs::read_to_string(config.project_root.join("main.rs"))
            .unwrap()
            .starts_with("call_a(None);\nkeep();\ncall_c(None);\n")
    );
}

#[tokio::test]
async fn execute_preplan_can_handle_multiple_inspection_commands_in_one_response() {
    // The windowed pass returns multiple SEARCH and READ commands in a single
    // response and they all get collected and batch-executed before finalize.
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                0 => helpers::mock_text_response(
                    "NOTE update both call sites\nSEARCH: call_a();\nSEARCH: call_c();\nREAD: 1-3",
                ),
                1 => helpers::mock_text_response(
                    "SMART_EDIT\nREGION 1 1\nTASK: update first call\nEND\n\nSMART_EDIT\nREGION 3 3\nTASK: update last call\nEND\n",
                ),
                2 => helpers::mock_text_response(
                    "REPLACE_AT 3\nOLD:\ncall_c();\nEND_OLD\nNEW:\ncall_c(None);\nEND_NEW\n",
                ),
                3 => helpers::mock_text_response(
                    "REPLACE_AT 1\nOLD:\ncall_a();\nEND_OLD\nNEW:\ncall_a(None);\nEND_NEW\n",
                ),
                // Terminal verdict round — large file still does window + finalize.
                4 => helpers::mock_text_response(""),
                _ => helpers::mock_text_response("COMPLETE\n"),
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
    let result = edit_file::execute(&args, &config, &router, None, None, None, None, None)
        .await
        .unwrap();

    assert!(result.success, "{}", result.content);
    assert_eq!(result.content.trim(), "✓ edit_file(main.rs): done");
    // window + finalize + 2 patches + verdict window + verdict finalize = 6
    assert_eq!(calls.load(Ordering::SeqCst), 6);
    assert!(
        fs::read_to_string(config.project_root.join("main.rs"))
            .unwrap()
            .starts_with("call_a(None);\nkeep();\ncall_c(None);\n")
    );
}

#[tokio::test]
async fn execute_preplan_uses_literal_replacements_before_smart_edits() {
    // Small file (6 lines): finalize + 1 smart-edit patch. The LITERAL_REPLACE
    // step is applied directly without needing an LLM patch call.
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
                1 => helpers::mock_text_response(
                    "REPLACE_AT 4\nOLD:\ncall(\n    None,\n)\nEND_OLD\nNEW:\ncall(\n    None,\n    None,\n)\nEND_NEW\n",
                ),
                // Terminal verdict after the successful patch execution.
                _ => helpers::mock_text_response("COMPLETE\n"),
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
    let result = edit_file::execute(&args, &config, &router, None, None, None, None, None)
        .await
        .unwrap();

    assert!(result.success, "{}", result.content);
    assert_eq!(result.content.trim(), "✓ edit_file(main.rs): done");
    // finalize + 1 smart-edit patch + verdict-finalize = 3
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "call(None, None)\nkeep();\ncall(None, None)\ncall(\n    None,\n    None,\n)\n"
    );
}

#[tokio::test]
async fn execute_preplan_rejects_literal_replace_without_old_and_recovers() {
    // Small file. On attempt 1 the planner emits the now-forbidden
    // OLD-less LITERAL_REPLACE shortcut — the parser should reject it
    // and feed the error into the repair prompt, then attempt 2 emits
    // the proper OLD-bearing form and the edit succeeds. This is the
    // regression guard for the "hallucinated replacement" failure
    // mode we saw in the docker_20260411 bench run.
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                // Attempt 1: the banned shortcut form — no OLD: block.
                0 => helpers::mock_text_response(
                    "LITERAL_REPLACE\nSCOPE 2 3\nALL true\nNEW:\nfresh two\nfresh three\nEND_NEW\nEND\n",
                ),
                // Attempt 2 (repair): proper form with OLD: verbatim.
                1 => helpers::mock_text_response(
                    "LITERAL_REPLACE\nSCOPE 2 3\nALL true\nOLD:\nline two\nline three\nEND_OLD\nNEW:\nfresh two\nfresh three\nEND_NEW\nEND\n",
                ),
                // Terminal verdict after the successful literal replace.
                _ => helpers::mock_text_response("COMPLETE\n"),
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
    let result = edit_file::execute(&args, &config, &router, None, None, None, None, None)
        .await
        .unwrap();

    assert!(result.success, "{}", result.content);
    assert_eq!(result.content.trim(), "✓ edit_file(main.rs): done");
    // finalize attempt 1 (rejected) + finalize attempt 2 (succeeded) + verdict = 3
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "line one\nfresh two\nfresh three\nline four\n"
    );
}

#[tokio::test]
async fn execute_preplan_literal_step_bubbles_up_when_no_relocation_candidate() {
    // Small file. The literal step's OLD is completely unrelated to any
    // line in the file — byte-exact, whitespace-normalized, and fuzzy
    // line-similarity all decline. With the smart-edit fallback removed,
    // the failure bubbles straight up to plan-level repair, which emits
    // a corrected literal and the edit succeeds.
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                // Plan attempt 1: hallucinated OLD, not in file at all
                0 => helpers::mock_text_response(
                    "LITERAL_REPLACE\nSCOPE 1 1\nALL false\nOLD:\nregister_unrelated_middleware_hook();\nEND_OLD\nNEW:\ncall(None, None)\nEND_NEW\nEND\n",
                ),
                // Plan attempt 2 (repair): correct OLD
                1 => helpers::mock_text_response(
                    "LITERAL_REPLACE\nSCOPE 1 1\nALL false\nOLD:\ncall(None)\nEND_OLD\nNEW:\ncall(None, None)\nEND_NEW\nEND\n",
                ),
                // Terminal verdict after the successful literal replace.
                _ => helpers::mock_text_response("COMPLETE\n"),
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
    let result = edit_file::execute(&args, &config, &router, None, None, None, None, None)
        .await
        .unwrap();

    assert!(result.success, "{}", result.content);
    assert_eq!(result.content.trim(), "✓ edit_file(main.rs): done");
    // plan1 finalize + plan2 finalize + verdict = 3 (no smart-fallback round-trip)
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "call(None, None)\n"
    );
}

#[tokio::test]
async fn execute_literal_replace_recovers_from_whitespace_drift_with_confirmation() {
    // Reproduces the Gemma `</div >` pathology: the planner emits a
    // LITERAL_REPLACE whose OLD has trivial whitespace drift (extra
    // space inside an HTML tag). The byte-exact matcher rejects it, the
    // new whitespace-tolerant fallback locates the real bytes in scope,
    // asks the planner via a single round-trip to confirm, and applies
    // the replacement directly — no smart-edit fallback needed.
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                // Plan finalize: literal with hallucinated whitespace
                0 => helpers::mock_text_response(
                    "LITERAL_REPLACE\nSCOPE 1 1\nALL false\nOLD:\n<div >hello</div >\nEND_OLD\nNEW:\n<div>world</div>\nEND_NEW\nEND\n",
                ),
                // Whitespace-drift confirmation round-trip
                1 => helpers::mock_text_response("YES"),
                // Terminal verdict after the successful literal replace.
                _ => helpers::mock_text_response("COMPLETE\n"),
            }
        })
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    fs::write(config.project_root.join("main.html"), "<div>hello</div>\n").unwrap();

    let router = miniswe::llm::ModelRouter::new(&config);
    let args = serde_json::json!({
        "path": "main.html",
        "task": "change hello to world",
        "lsp_validation": "off",
    });
    let result = edit_file::execute(&args, &config, &router, None, None, None, None, None)
        .await
        .unwrap();

    assert!(result.success, "{}", result.content);
    assert_eq!(result.content.trim(), "✓ edit_file(main.html): done");
    // finalize + confirmation + verdict-finalize = 3 (no smart-fallback round-trip)
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.html")).unwrap(),
        "<div>world</div>\n"
    );
}

#[tokio::test]
async fn execute_literal_replace_recovers_from_single_char_typo_via_fuzzy() {
    // The planner emits a LITERAL_REPLACE whose OLD has a single-char
    // typo that whitespace normalization cannot rescue. The fuzzy
    // line-similarity search locates the real line, asks the planner
    // to confirm, and applies the replacement at the corrected scope.
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                // Plan finalize: OLD is `calback` but file says `callback`
                0 => helpers::mock_text_response(
                    "LITERAL_REPLACE\nSCOPE 1 1\nALL false\nOLD:\nlet calback = register_handler();\nEND_OLD\nNEW:\nlet callback = register_handler_v2();\nEND_NEW\nEND\n",
                ),
                // Fuzzy-relocation confirmation: planner says YES
                1 => helpers::mock_text_response("YES"),
                // Terminal verdict after the successful literal replace.
                _ => helpers::mock_text_response("COMPLETE\n"),
            }
        })
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    fs::write(
        config.project_root.join("main.rs"),
        "let callback = register_handler();\n",
    )
    .unwrap();

    let router = miniswe::llm::ModelRouter::new(&config);
    let args = serde_json::json!({
        "path": "main.rs",
        "task": "upgrade handler registration",
        "lsp_validation": "off",
    });
    let result = edit_file::execute(&args, &config, &router, None, None, None, None, None)
        .await
        .unwrap();

    assert!(result.success, "{}", result.content);
    assert_eq!(result.content.trim(), "✓ edit_file(main.rs): done");
    // finalize + fuzzy confirmation + verdict-finalize = 3
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "let callback = register_handler_v2();\n"
    );
}

#[tokio::test]
async fn execute_literal_replace_with_whitespace_drift_rejected_bails_to_plan_repair() {
    // Same setup, but the planner rejects the candidate ("NO"). We
    // must NOT run the smart-edit fallback after that — bubble straight
    // up to plan-level repair so the planner can re-emit a correct OLD
    // with full structured context.
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                // Plan attempt 1: literal with hallucinated whitespace
                0 => helpers::mock_text_response(
                    "LITERAL_REPLACE\nSCOPE 1 1\nALL false\nOLD:\n<div >hello</div >\nEND_OLD\nNEW:\n<div>WRONG</div>\nEND_NEW\nEND\n",
                ),
                // Whitespace-drift confirmation: planner says NO
                1 => helpers::mock_text_response("NO"),
                // Plan attempt 2 (repair): correct literal applied
                2 => helpers::mock_text_response(
                    "LITERAL_REPLACE\nSCOPE 1 1\nALL false\nOLD:\n<div>hello</div>\nEND_OLD\nNEW:\n<div>world</div>\nEND_NEW\nEND\n",
                ),
                // Terminal verdict after the successful literal replace.
                _ => helpers::mock_text_response("COMPLETE\n"),
            }
        })
        .mount(&mock_server)
        .await;

    let (_tmp, mut config) = helpers::create_test_project();
    helpers::config_with_mock_endpoint(&mut config, &mock_server.uri());
    fs::write(config.project_root.join("main.html"), "<div>hello</div>\n").unwrap();

    let router = miniswe::llm::ModelRouter::new(&config);
    let args = serde_json::json!({
        "path": "main.html",
        "task": "change hello to world",
        "lsp_validation": "off",
    });
    let result = edit_file::execute(&args, &config, &router, None, None, None, None, None)
        .await
        .unwrap();

    assert!(result.success, "{}", result.content);
    assert_eq!(result.content.trim(), "✓ edit_file(main.html): done");
    // plan1: 1 finalize + 1 confirmation (NO) + plan2: 1 finalize + verdict = 4
    // Crucially NO smart-edit fallback round-trip in between.
    assert_eq!(calls.load(Ordering::SeqCst), 4);
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.html")).unwrap(),
        "<div>world</div>\n"
    );
}

#[tokio::test]
async fn execute_preplan_repairs_whole_plan_after_literal_failure() {
    // Plan 1 has a literal whose OLD doesn't match and the OLD-relocation
    // rescue finds no candidate anywhere in the file. The failure bubbles
    // up immediately to plan-level repair, which re-plans with full
    // repair context and the second plan succeeds.
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                // Plan attempt 1
                0 => helpers::mock_text_response(
                    "LITERAL_REPLACE\nSCOPE 1 1\nALL false\nOLD:\nmissing(None)\nEND_OLD\nNEW:\ncall(None, None)\nEND_NEW\nEND\n",
                ),
                // Plan attempt 2 (repair)
                1 => helpers::mock_text_response(
                    "LITERAL_REPLACE\nSCOPE 1 1\nALL false\nOLD:\ncall(None)\nEND_OLD\nNEW:\ncall(None, None)\nEND_NEW\nEND\n",
                ),
                // Terminal verdict after the successful literal replace.
                _ => helpers::mock_text_response("COMPLETE\n"),
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
    let result = edit_file::execute(&args, &config, &router, None, None, None, None, None)
        .await
        .unwrap();

    assert!(result.success, "{}", result.content);
    assert_eq!(result.content.trim(), "✓ edit_file(main.rs): done");
    // plan1 finalize + plan2 finalize + verdict = 3 (no smart-fallback round-trip)
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "call(None, None)\n"
    );
}

#[tokio::test]
async fn execute_preplan_repair_attempt_includes_structured_repair_context() {
    // Plan 1 has a literal that doesn't match, and the OLD-relocation
    // rescue finds no candidate, so the failure bubbles up immediately to
    // plan-level repair. Plan 2 fixes the file with a working literal.
    // The plan-2 finalize prompt should carry the structured repair-context
    // block — that's what this test asserts on top of the existing repair
    // flow coverage.
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                // Plan attempt 1
                0 => helpers::mock_text_response(
                    "LITERAL_REPLACE\nSCOPE 1 1\nALL false\nOLD:\nmissing(None)\nEND_OLD\nNEW:\ncall(None, None)\nEND_NEW\nEND\n",
                ),
                // Plan attempt 2 (repair)
                1 => helpers::mock_text_response(
                    "LITERAL_REPLACE\nSCOPE 1 1\nALL false\nOLD:\ncall(None)\nEND_OLD\nNEW:\ncall(None, None)\nEND_NEW\nEND\n",
                ),
                // Terminal verdict after the successful literal replace.
                _ => helpers::mock_text_response("COMPLETE\n"),
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
    let result = edit_file::execute(&args, &config, &router, None, None, None, None, None)
        .await
        .unwrap();

    assert!(result.success, "{}", result.content);
    // plan1 finalize + plan2 finalize + verdict = 3 (no smart-fallback round-trip)
    assert_eq!(calls.load(Ordering::SeqCst), 3);

    // Inspect the body of the plan-2 finalize request (index 1)
    // and verify the structured repair-context block reached the model.
    let requests = mock_server
        .received_requests()
        .await
        .expect("mock server should record requests");
    assert_eq!(requests.len(), 3);
    let plan2_body = String::from_utf8(requests[1].body.clone()).unwrap();

    // Marker text from format_repair_context.
    assert!(
        plan2_body.contains("The previous iteration failed."),
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
async fn execute_preplan_needs_clarification_short_circuits_with_question() {
    // The finalize call returns NEEDS_CLARIFICATION with a specific question.
    // The retry loop must short-circuit after the planning phase, the file
    // must be left unchanged, and the tool result must surface the question
    // plus the original task so the outer agent can rephrase.
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                0 => helpers::mock_text_response(
                    "NEEDS_CLARIFICATION: which module owns the auth middleware you want removed?",
                ),
                _ => unreachable!("NEEDS_CLARIFICATION should short-circuit the retry loop"),
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
    let result = edit_file::execute(&args, &config, &router, None, None, None, None, None)
        .await
        .unwrap();

    assert!(
        !result.success,
        "NEEDS_CLARIFICATION should surface as a failure"
    );
    assert!(
        result.content.contains("needs clarification"),
        "expected clarification marker, got: {}",
        result.content
    );
    assert!(
        result
            .content
            .contains("which module owns the auth middleware you want removed?"),
        "expected question in tool output, got: {}",
        result.content
    );
    assert!(
        result.content.contains("remove the auth middleware"),
        "expected original task echoed, got: {}",
        result.content
    );
    assert!(
        result.content.contains("file was not modified"),
        "expected unmodified note, got: {}",
        result.content
    );
    // finalize only = 1, no retries after NEEDS_CLARIFICATION.
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    // File on disk unchanged.
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "println!(\"hi\");\n"
    );
}

#[tokio::test]
async fn execute_preplan_needs_clarification_without_question_uses_placeholder() {
    // Bare `NEEDS_CLARIFICATION` (no colon, no question) at the finalize phase
    // should still short-circuit and surface a "no question provided"
    // placeholder rather than an empty string in the user-facing message.
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                0 => helpers::mock_text_response("NEEDS_CLARIFICATION"),
                _ => unreachable!("NEEDS_CLARIFICATION should short-circuit the retry loop"),
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
        "task": "ambiguous task",
        "lsp_validation": "off"
    });
    let result = edit_file::execute(&args, &config, &router, None, None, None, None, None)
        .await
        .unwrap();

    assert!(!result.success);
    assert!(result.content.contains("no question provided"));
    // finalize only = 1
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "println!(\"hi\");\n"
    );
}

#[tokio::test]
async fn execute_preplan_needs_clarification_during_repair_short_circuits() {
    // Plan 1 fails to apply (literal OLD doesn't match, OLD-relocation
    // rescue finds no candidate, bubbles up). On the repair attempt, the
    // model realizes the task is too ambiguous and emits
    // NEEDS_CLARIFICATION. The retry loop must stop immediately and
    // surface the question — no further planning attempts after it.
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                // Plan attempt 1
                0 => helpers::mock_text_response(
                    "LITERAL_REPLACE\nSCOPE 1 1\nALL false\nOLD:\nmissing(None)\nEND_OLD\nNEW:\ncall(None, None)\nEND_NEW\nEND\n",
                ),
                // Plan attempt 2 (repair) — model decides the task is ambiguous.
                1 => helpers::mock_text_response(
                    "NEEDS_CLARIFICATION: which calls should be updated and what are the new arguments?",
                ),
                _ => unreachable!(
                    "NEEDS_CLARIFICATION should short-circuit the retry loop"
                ),
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
    let result = edit_file::execute(&args, &config, &router, None, None, None, None, None)
        .await
        .unwrap();

    assert!(!result.success, "{}", result.content);
    assert!(
        result.content.contains("needs clarification"),
        "expected clarification marker, got: {}",
        result.content
    );
    assert!(
        result
            .content
            .contains("which calls should be updated and what are the new arguments?"),
        "expected question in tool output, got: {}",
        result.content
    );
    // plan1 finalize + plan2 finalize (NEEDS_CLARIFICATION) = 2
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    // File unchanged — plan 1's literal didn't match, so no edits applied.
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "call(None)\n"
    );
}

#[tokio::test]
async fn execute_preplan_missing_target_is_not_treated_as_clarification_and_retries() {
    // The first attempt fails to find the target (plan parses but nothing
    // applies and the OLD-relocation rescue finds no candidate), but the
    // model must NOT use NEEDS_CLARIFICATION for a missing target — that's
    // exactly what the edit is for. The second attempt produces a valid
    // plan and the edit applies. This test guards the retry flow for
    // "target doesn't exist yet" cases.
    let mock_server = MockServer::start().await;
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_mock = calls.clone();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(move |_req: &wiremock::Request| {
            let n = calls_for_mock.fetch_add(1, Ordering::SeqCst);
            match n {
                // Plan attempt 1: literal whose OLD doesn't match
                0 => helpers::mock_text_response(
                    "LITERAL_REPLACE\nSCOPE 1 1\nALL false\nOLD:\nfoo();\nEND_OLD\nNEW:\nbar();\nEND_NEW\nEND\n",
                ),
                // Plan attempt 2 (repair) — now it reads the file correctly
                1 => helpers::mock_text_response(
                    "LITERAL_REPLACE\nSCOPE 1 1\nALL false\nOLD:\noriginal();\nEND_OLD\nNEW:\nfoo();\nEND_NEW\nEND\n",
                ),
                // Terminal verdict after the successful literal replace.
                _ => helpers::mock_text_response("COMPLETE\n"),
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
    let result = edit_file::execute(&args, &config, &router, None, None, None, None, None)
        .await
        .unwrap();

    assert!(
        result.success,
        "expected success after repair retry, got: {}",
        result.content
    );
    // plan1 finalize + plan2 finalize + verdict = 3 (no smart-fallback round-trip)
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    // File should now contain the renamed call.
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "foo();\n"
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
    let result = edit_file::execute(&args, &config, &router, None, None, None, None, None)
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
                // Small-file fast path — call 0 is finalize: two overlapping SMART_EDIT steps.
                0 => helpers::mock_text_response(
                    "SMART_EDIT\nREGION 1 3\nTASK: rewrite the first three lines\nEND\n\
                     \n\
                     SMART_EDIT\nREGION 3 5\nTASK: rewrite lines three through five\nEND\n",
                ),
                // smart-edit patch for the kept (first) step L1-L3
                1 => helpers::mock_text_response(
                    "REPLACE_AT 1\nOLD:\nalpha\nbeta\ngamma\nEND_OLD\nNEW:\nALPHA\nBETA\nGAMMA\nEND_NEW\n",
                ),
                // Terminal verdict after the successful patch.
                _ => helpers::mock_text_response("COMPLETE\n"),
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
    let result = edit_file::execute(&args, &config, &router, None, None, None, None, None)
        .await
        .unwrap();

    // The kept step applied — overall success. The agent-facing message is
    // now a single-line success template; the per-step dropped-step trail
    // lives in the session log, not in the tool output.
    assert!(result.success, "{}", result.content);
    assert_eq!(result.content.trim(), "✓ edit_file(main.rs): done");
    // The kept step actually wrote the file. This file-content assertion
    // is what proves the overlap dropping still works end-to-end: the
    // first step applied (L1-L3 rewritten in uppercase) and the second,
    // overlapping step was dropped (delta/epsilon untouched).
    assert_eq!(
        fs::read_to_string(config.project_root.join("main.rs")).unwrap(),
        "ALPHA\nBETA\nGAMMA\ndelta\nepsilon\n"
    );
}
