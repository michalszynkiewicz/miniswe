use super::{FORCE_COMPRESS_MAX_RETRIES, estimated_context_tokens, force_compress};
use crate::config::{CompactionStrategy, Config};
use crate::llm::{Message, ModelRouter};
use crate::runtime::LlmWorkerHandle;
use std::sync::Arc;

/// Test config rooted in a temp dir. The endpoint points at a port
/// nothing listens on and retries are disabled, so any LLM-based
/// summarization fails instantly (connection refused) and falls back to
/// the heuristic summary — tests never touch a live server.
fn config_in(dir: &std::path::Path, strategy: CompactionStrategy) -> Config {
    std::fs::create_dir_all(dir.join(".miniswe")).unwrap();
    let mut config = Config::default();
    config.project_root = dir.to_path_buf();
    config.ensure_session_dir().unwrap();
    config.model.endpoint = "http://127.0.0.1:9".into();
    config.model.max_retries = 0;
    config.model.context_window = 60_000;
    config.context.compaction = strategy;
    config
}

/// A message list far over `raw_budget` (~16.7K tokens at a 60K window
/// with tool_def_tokens=0): 20 tool results of 5K chars ≈ 25K tokens.
fn over_budget_messages() -> Vec<Message> {
    let mut msgs = vec![
        Message::system("You are miniswe."),
        Message::user("do the task"),
    ];
    for i in 0..20 {
        msgs.push(Message::assistant(&format!("reading file {i}")));
        msgs.push(Message::tool_result(
            &format!("call{i}"),
            &"line of tool output\n".repeat(250),
        ));
    }
    msgs
}

#[tokio::test]
async fn lazy_is_a_no_op_in_maybe_compress() {
    let tmp = tempfile::TempDir::new().unwrap();
    let config = config_in(tmp.path(), CompactionStrategy::Lazy);
    let router = Arc::new(ModelRouter::new(&config));
    let worker = LlmWorkerHandle::new(router.clone(), 1);

    let mut messages = over_budget_messages();
    let before_tokens = estimated_context_tokens(&messages, 0);
    let before_len = messages.len();

    let mut plan_flag = false;
    super::maybe_compress(&mut messages, &config, &router, &worker, 0, &mut plan_flag).await;

    // Far over budget, yet nothing was compacted — Lazy never fires
    // proactively. (No plan/scratchpad exists in the temp project, so
    // the current-state refresh is also a no-op here.)
    assert_eq!(messages.len(), before_len);
    assert_eq!(estimated_context_tokens(&messages, 0), before_tokens);
}

#[tokio::test]
async fn force_compress_lazy_shrinks_via_unified_path() {
    let tmp = tempfile::TempDir::new().unwrap();
    let config = config_in(tmp.path(), CompactionStrategy::Lazy);
    let router = Arc::new(ModelRouter::new(&config));
    let worker = LlmWorkerHandle::new(router.clone(), 1);

    let mut messages = over_budget_messages();
    let before_tokens = estimated_context_tokens(&messages, 0);

    let freed = force_compress(&mut messages, &config, &router, &worker, 0).await;

    // The LLM summarizer can't be reached (dead endpoint, 0 retries) —
    // the heuristic fallback must still compact.
    assert!(freed, "force_compress should report freed tokens");
    assert!(
        estimated_context_tokens(&messages, 0) < before_tokens,
        "history should shrink"
    );
    // The unified path archives what it elided.
    assert!(config.miniswe_path("session_archive.md").exists());
}

#[tokio::test]
async fn force_compress_sliding_window_shrinks_without_llm() {
    let tmp = tempfile::TempDir::new().unwrap();
    let config = config_in(tmp.path(), CompactionStrategy::SlidingWindow);
    let router = Arc::new(ModelRouter::new(&config));
    let worker = LlmWorkerHandle::new(router.clone(), 1);

    let mut messages = over_budget_messages();
    let before_tokens = estimated_context_tokens(&messages, 0);

    let freed = force_compress(&mut messages, &config, &router, &worker, 0).await;

    assert!(freed);
    assert!(estimated_context_tokens(&messages, 0) < before_tokens);
}

#[tokio::test]
async fn force_compress_reports_false_when_nothing_to_free() {
    let tmp = tempfile::TempDir::new().unwrap();
    let config = config_in(tmp.path(), CompactionStrategy::Lazy);
    let router = Arc::new(ModelRouter::new(&config));
    let worker = LlmWorkerHandle::new(router.clone(), 1);

    // Tiny history — nothing old enough to compact. Callers rely on
    // `false` here to avoid resending a request that will fail
    // identically.
    let mut messages = vec![
        Message::system("You are miniswe."),
        Message::user("hi"),
        Message::assistant("hello"),
    ];
    let freed = force_compress(&mut messages, &config, &router, &worker, 0).await;
    assert!(!freed);
    assert_eq!(messages.len(), 3);
}

#[test]
fn retry_cap_is_small_and_nonzero() {
    // The cap bounds consecutive futile retries of ONE failing request;
    // it must allow at least one retry and stay small enough that a
    // truly unfixable request fails fast.
    assert!((1..=3).contains(&FORCE_COMPRESS_MAX_RETRIES));
}

#[test]
fn estimated_context_tokens_counts_system_and_tools() {
    let messages = vec![
        Message::system(&"s".repeat(400)), // ~100 tokens
        Message::user(&"u".repeat(400)),   // ~100 tokens
    ];
    let with_tools = estimated_context_tokens(&messages, 500);
    let without_tools = estimated_context_tokens(&messages, 0);
    assert_eq!(with_tools - without_tools, 500);
    // System message IS counted — unlike needs_compression's history
    // total, this estimates the full prompt as the server sees it.
    assert!(without_tools >= 200);
}

#[test]
fn strip_summary_envelope_keeps_only_content() {
    let injected = format!(
        "{}\n- run.rs: threaded the new param\n- mod.rs: added flag\n\
         [Details: file(action='read', path='.miniswe/session_archive.md'). Continue from where you left off.]",
        super::UNIFIED_SUMMARY_HEADER
    );
    let stripped = super::strip_summary_envelope(&injected);
    assert_eq!(
        stripped,
        "- run.rs: threaded the new param\n- mod.rs: added flag"
    );
}

#[tokio::test]
async fn unified_writes_the_marker_its_search_looks_for() {
    // Regression guard for the writer/search drift that silently killed
    // carry-forward for months: the message compact_unified injects must
    // start with the exact prefix its existing-summary search matches on.
    let tmp = tempfile::TempDir::new().unwrap();
    let config = config_in(tmp.path(), CompactionStrategy::Lazy);
    let router = Arc::new(ModelRouter::new(&config));
    let worker = LlmWorkerHandle::new(router.clone(), 1);

    let mut messages = over_budget_messages();
    force_compress(&mut messages, &config, &router, &worker, 0).await;

    let summary_msg = messages
        .iter()
        .find(|m| {
            m.role == "user"
                && m.content
                    .as_deref()
                    .is_some_and(|c| c.starts_with(super::UNIFIED_SUMMARY_HEADER))
        })
        .expect("compact_unified should inject a summary the search can find");
    assert!(super::is_summary_marker(
        summary_msg.content.as_deref().unwrap()
    ));
}

#[tokio::test]
async fn empty_summarize_window_short_circuits_without_llm() {
    // A window holding only a previous summary marker builds an empty
    // timeline; the summarizer must not ask an LLM to "list what you
    // accomplished" over nothing (probed on nemotron: 2/2 fabricated
    // changelogs). With an existing summary it is carried forward
    // verbatim; without one the caller falls back to the heuristic.
    let tmp = tempfile::TempDir::new().unwrap();
    let config = config_in(tmp.path(), CompactionStrategy::Lazy);
    let router = Arc::new(ModelRouter::new(&config));
    let worker = LlmWorkerHandle::new(router.clone(), 1);

    let blob = format!("{}\nold facts", super::UNIFIED_SUMMARY_HEADER);
    let only_marker = [Message::user(&blob)];
    let refs: Vec<&Message> = only_marker.iter().collect();

    // The dead endpoint (config_in) would return None if the LLM path
    // were reached; getting the existing summary back proves the
    // short-circuit fired before any request.
    let carried = super::llm_summarize_timeline(
        &refs,
        "earlier facts",
        1000,
        &router,
        &worker,
        super::SummaryStyle::Structured,
    )
    .await;
    assert_eq!(carried.as_deref(), Some("earlier facts"));

    let none = super::llm_summarize_timeline(
        &refs,
        "",
        1000,
        &router,
        &worker,
        super::SummaryStyle::Structured,
    )
    .await;
    assert_eq!(none, None);
}
