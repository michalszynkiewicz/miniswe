//! Replay of real historical `add_param` failures against a REAL model
//! (gemma-4, not a mock) to verify the `known_old` fix (deterministic
//! prefill for signature rewrites — see `model_edit::ask_rewrite_validated`,
//! `sites::signature_old_block`). These calls, byte-for-byte, previously
//! failed with "signature rewrite failed: ... OLD line N doesn't match
//! source" across dozens of real bench runs (2026-07-03/04 compaction
//! matrices), forcing the model to abandon the atomic tool and fall back to
//! a manual single-file edit that skipped every callsite.
//!
//! Requires a live LLM at MINISWE_TEST_ENDPOINT (default
//! http://localhost:8464) and real rust-analyzer. `#[ignore]`d — not part
//! of the default `cargo test` run; invoke with
//! `cargo test --test e2e_refactor_replay -- --ignored --nocapture`.

mod helpers;

use std::fs;
use std::path::Path;
use std::time::Duration;

use miniswe::lsp::{LspClient, LspServer};
use miniswe::tools;
use serde_json::json;

async fn ensure_rust_analyzer() -> bool {
    LspServer::RustAnalyzer.ensure_binary().await.is_ok()
}

fn fixture_tree() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("benchmark_results/_fixtures/run2-depoisoned/tree")
}

fn copy_dir_all(src: &Path, dst: &Path) {
    fs::create_dir_all(dst).unwrap();
    for entry in fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        let ty = entry.file_type().unwrap();
        let dst_path = dst.join(entry.file_name());
        if ty.is_dir() {
            if entry.file_name() == "target" || entry.file_name() == ".git" {
                continue;
            }
            copy_dir_all(&entry.path(), &dst_path);
        } else {
            fs::copy(entry.path(), &dst_path).unwrap();
        }
    }
}

async fn spawn_lsp_for(root: &Path, target_file: &Path) -> LspClient {
    let client = LspClient::spawn(root.to_path_buf())
        .await
        .expect("spawn rust-analyzer");

    let start = std::time::Instant::now();
    while !client.is_ready() && start.elapsed() < Duration::from_secs(120) {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(client.is_ready(), "LSP did not become ready in 120s");

    // This fixture is the full miniswe codebase (not a toy 2-file project),
    // so first-time indexing (crate graph, cargo metadata) takes a while.
    // Force analysis by opening the target file and waiting for diagnostics
    // on it, same as e2e_refactor.rs's spawn_lsp_for — without this,
    // find_references/goto_definition hit an unindexed workspace and
    // silently return empty ("no function named X defined").
    client.notify_file_changed(target_file).unwrap();
    let _ = client
        .get_diagnostics(target_file, Duration::from_secs(180))
        .await;
    let _ = client.wait_for_idle(Duration::from_secs(60)).await;
    client
}

fn real_gemma_config(project_root: std::path::PathBuf) -> miniswe::config::Config {
    let mut config = miniswe::config::Config::default();
    config.project_root = project_root;
    let endpoint =
        std::env::var("MINISWE_TEST_ENDPOINT").unwrap_or_else(|_| "http://localhost:8464".into());
    config.model.provider = "llama-cpp".into();
    config.model.endpoint = endpoint;
    config.model.model = "gemma-4-26B-A4B-it".into();
    config.model.context_window = 40000;
    config.model.temperature = 0.2;
    config.model.max_output_tokens = 4000;
    config.model.max_retries = 3;
    config
}

/// Historical `add_param` calls, byte-for-byte, that failed against real
/// gemma in real bench runs. Source noted per case.
struct ReplayCase {
    label: &'static str,
    source: &'static str,
    args: serde_json::Value,
    /// Substring the signature should contain after a successful rewrite.
    expect_signature_contains: &'static str,
}

fn replay_cases() -> Vec<ReplayCase> {
    vec![
        ReplayCase {
            label: "run4_assemble",
            source: "compaction_20260704_164112 unified/run4 dump req-1783180130-00054-000019",
            args: json!({
                "action": "add_param",
                "path": "src/context/mod.rs",
                "name": "assemble",
                "new_param": "system_prompt_override: Option<String>",
                "position": "after:mcp_summary",
                "callsite_fill_in": "None",
            }),
            expect_signature_contains: "system_prompt_override: Option<String>",
        },
        ReplayCase {
            label: "run3_run_fn",
            source: "compaction_20260704_164112 unified/run3 dump req-1783179088-00055-000079",
            args: json!({
                "action": "add_param",
                "path": "src/cli/commands/run.rs",
                "name": "run",
                "new_param": "system_prompt_override: Option<String>",
                "position": "after:headless",
                "callsite_fill_in": "system_prompt_override",
            }),
            expect_signature_contains: "system_prompt_override: Option<String>",
        },
        ReplayCase {
            label: "tiered_run5_run_fn_refstr",
            source: "compaction_20260703_153837 tiered/run5",
            args: json!({
                "action": "add_param",
                "path": "src/cli/commands/run.rs",
                "name": "run",
                "new_param": "system_prompt_override: Option<&str>",
                "position": "after:headless",
                "callsite_fill_in": "None",
            }),
            expect_signature_contains: "system_prompt_override: Option<&str>",
        },
    ]
}

#[tokio::test]
#[ignore]
async fn replay_historical_add_param_failures() {
    if !ensure_rust_analyzer().await {
        eprintln!("skipping: rust-analyzer not available");
        return;
    }

    let mut results = Vec::new();

    for case in replay_cases() {
        let temp = tempfile::TempDir::new().unwrap();
        let root = temp.path().to_path_buf();
        copy_dir_all(&fixture_tree(), &root);
        fs::create_dir_all(root.join(".miniswe/index")).unwrap();
        fs::write(root.join(".miniswe/index/symbols.json"), "{}").unwrap();
        fs::write(root.join(".miniswe/index/summaries.json"), "{}").unwrap();
        fs::write(root.join(".miniswe/index/file_tree.txt"), "").unwrap();
        fs::write(root.join(".miniswe/index/mtimes.json"), "{}").unwrap();
        fs::write(root.join(".mcp.json"), r#"{"servers":{}}"#).unwrap();

        let config = real_gemma_config(root.clone());
        let router = miniswe::llm::ModelRouter::new(&config);

        eprintln!("\n=== REPLAY: {} ({}) ===", case.label, case.source);
        eprintln!("args: {}", case.args);

        let path_str = case.args["path"].as_str().unwrap().to_string();
        let target_path = root.join(&path_str);
        let before = fs::read_to_string(&target_path).unwrap();

        let lsp = spawn_lsp_for(&root, &target_path).await;

        let result = tools::execute_refactor_tool(
            &case.args,
            &config,
            &router,
            Some(&lsp),
            None,
            None,
            None,
        )
        .await
        .unwrap();

        eprintln!(
            "result.success={} content=\n{}",
            result.success, result.content
        );

        let after = fs::read_to_string(&target_path).unwrap();
        let sig_ok = after.contains(case.expect_signature_contains);
        let changed = after != before;

        // Check the KNOWN historical blind spot: does the test file (if
        // present) also get updated? This is what a fully successful atomic
        // refactor should do — the ORIGINAL bug's downstream damage was
        // exactly that this never happened.
        let test_file = root.join("tests/e2e_context.rs");
        let test_file_status = if test_file.exists() {
            let test_src = fs::read_to_string(&test_file).unwrap();
            // Old 5-arg (assemble) / 4-arg (run's own callers aren't in this
            // test file) calls would now be arity mismatches if untouched.
            // We just report whether the tool's own report mentions the
            // test file, plus whether cargo check passes overall below.
            format!(
                "{} bytes, mentions fn name: {}",
                test_src.len(),
                test_src.contains(&format!("{}(", case.args["name"].as_str().unwrap()))
            )
        } else {
            "n/a".to_string()
        };

        // Real ground truth: does the WHOLE project compile after this one
        // tool call? (Not just the signature file — every callsite too.)
        let build = std::process::Command::new("cargo")
            .arg("check")
            .arg("--tests")
            .current_dir(&root)
            .output();
        let compiles = match &build {
            Ok(o) => o.status.success(),
            Err(_) => false,
        };
        let build_tail = build
            .as_ref()
            .map(|o| {
                let s = String::from_utf8_lossy(&o.stderr);
                s.lines().rev().take(15).collect::<Vec<_>>().join("\n")
            })
            .unwrap_or_default();

        if !compiles {
            // Dump the exact signature area so we can see precisely what
            // the rewrite produced, not just the cascading compiler error.
            let sig_line_0based = after
                .lines()
                .position(|l| l.contains(case.args["name"].as_str().unwrap()) && l.contains("fn "))
                .unwrap_or(0);
            let start = sig_line_0based.saturating_sub(2);
            let end = (sig_line_0based + 15).min(after.lines().count());
            eprintln!(
                "--- signature area of {path_str} (lines {}-{}) ---",
                start + 1,
                end
            );
            for (i, l) in after.lines().enumerate().take(end).skip(start) {
                eprintln!("{:4}| {l}", i + 1);
            }
        }

        eprintln!(
            "signature_ok={sig_ok} changed={changed} compiles_after={compiles}\n\
             test_file_status={test_file_status}"
        );
        if !compiles {
            eprintln!("--- cargo check tail ---\n{build_tail}");
        }

        results.push((
            case.label,
            result.success,
            sig_ok,
            compiles,
            result.content.clone(),
        ));

        lsp.shutdown().await;
    }

    eprintln!("\n=== SUMMARY ===");
    let mut any_fail = false;
    for (label, tool_success, sig_ok, compiles, content) in &results {
        let verdict = if *tool_success && *sig_ok && *compiles {
            "PASS"
        } else {
            any_fail = true;
            "FAIL"
        };
        eprintln!(
            "{verdict}  {label}: tool_success={tool_success} sig_ok={sig_ok} compiles={compiles}"
        );
        if verdict == "FAIL" {
            eprintln!("  content: {}", content.lines().next().unwrap_or(""));
        }
    }
    assert!(
        !any_fail,
        "one or more replay cases failed — see output above"
    );
}
