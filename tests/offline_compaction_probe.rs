//! Offline compaction retention probe.
//!
//! Drives the REAL `maybe_compress` over a captured agent trajectory and dumps
//! the final compacted context for each strategy. Retention is then scored
//! out-of-band by `scripts/compaction-judge.py` (exact-match + a semantic judge),
//! so this harness stays simple and the scoring can be fair to summaries.
//!
//! Measures *retention*, not task success — read it as the retention/cost
//! frontier, not an outcome ranking.
//!
//! Run manually (summary arms need a live LLM endpoint):
//!   PROBE_TRAJECTORY=/path/to/trajectory.json \
//!   PROBE_DUMP_DIR=/path/to/out \
//!   LLAMA_ENDPOINT=http://localhost:8464 \
//!   cargo test --test offline_compaction_probe -- --ignored --nocapture
//!
//! `trajectory.json` is a JSON array of `Message` objects (no system message);
//! build one with `scripts/replay/reconstruct-trajectory.py`.

use miniswe::config::{CompactionStrategy, Config};
use miniswe::context::compressor::maybe_compress;
use miniswe::context::estimate_tokens;
use miniswe::llm::{Message, ModelRouter};
use miniswe::runtime::LlmWorkerHandle;
use std::sync::Arc;

/// Load the captured trajectory (Message array, no system msg) from
/// `PROBE_TRAJECTORY`, or fall back to a tiny synthetic stub.
fn load_trajectory() -> Vec<Message> {
    match std::env::var("PROBE_TRAJECTORY") {
        Ok(path) => {
            let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
            serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse {path}: {e}"))
        }
        Err(_) => vec![
            Message::user("Add the --system-prompt-override flag."),
            Message::assistant("Reading src/context/mod.rs."),
            Message::tool_result("c1", "fn assemble(...) { /* needle: QUOKKA_4815 */ }"),
        ],
    }
}

async fn run_strategy(strategy: CompactionStrategy, endpoint: &str) -> (usize, usize, usize) {
    let mut config = Config::default();
    config.model.endpoint = endpoint.to_string();
    config.model.model = "gemma-4-26B-A4B-it".to_string();
    config.model.context_window = 60_000;
    config.context.compaction = strategy;
    config.tools.plan = false; // skip the plan-anchor early-return; compact directly

    let router = Arc::new(ModelRouter::new(&config));
    let worker = LlmWorkerHandle::new(router.clone(), 2);

    // Replay the captured trajectory through the real compaction loop: grow the
    // history one turn at a time, compacting each step like the agent loop does.
    let traj = load_trajectory();
    let mut messages = vec![Message::system(
        "You are miniswe, a coding agent. Complete the task using your tools.",
    )];
    let mut plan_flag = false;
    let mut compactions = 0usize;
    for turn in &traj {
        messages.push(turn.clone());
        let before = messages.len();
        maybe_compress(&mut messages, &config, &router, &worker, 0, &mut plan_flag).await;
        if messages.len() != before
            || messages.iter().any(|m| {
                m.content
                    .as_deref()
                    .is_some_and(|c| c == "[earlier tool output elided to save context]")
            })
        {
            compactions += 1; // rough: any step that masked/collapsed
        }
    }

    let ctx: String = messages
        .iter()
        .map(|m| {
            format!(
                "=== [{}] ===\n{}",
                m.role,
                m.content.as_deref().unwrap_or("")
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    if let Ok(dir) = std::env::var("PROBE_DUMP_DIR") {
        let _ = std::fs::write(format!("{dir}/probe_ctx_{strategy:?}.txt"), &ctx);
    }
    let tokens: usize = messages
        .iter()
        .map(|m| estimate_tokens(m.content.as_deref().unwrap_or("")))
        .sum();
    (tokens, messages.len(), compactions)
}

#[tokio::test]
#[ignore = "needs a captured trajectory + (for summary arms) a live LLM endpoint; run manually"]
async fn offline_compaction_retention_probe() {
    let endpoint =
        std::env::var("LLAMA_ENDPOINT").unwrap_or_else(|_| "http://localhost:8464".to_string());
    let n = load_trajectory().len();
    println!("\n=== Offline compaction probe: {n}-message trajectory, endpoint={endpoint} ===");
    println!(
        "{:<22} {:>8} {:>6} {:>10}",
        "strategy", "tokens", "msgs", "compactions"
    );
    for strat in [
        CompactionStrategy::SlidingWindow,
        CompactionStrategy::ObservationMasking,
        CompactionStrategy::Unified,
        CompactionStrategy::RollingSummary,
        CompactionStrategy::Tiered,
    ] {
        let (tokens, nmsgs, comp) = run_strategy(strat, &endpoint).await;
        println!(
            "{:<22} {tokens:>8} {nmsgs:>6} {comp:>10}",
            format!("{strat:?}")
        );
    }
    println!("Contexts dumped to PROBE_DUMP_DIR — score with scripts/compaction-judge.py\n");
}
