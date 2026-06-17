//! Reactive debugger sub-agent (experimental, opt-in via
//! `tools.reactive_debugger`). See GitHub #40.
//!
//! When the behavioral done-gate (`[validation]`) blocks completion several
//! times in a single turn, the primary agent is in a "stuck-but-knows" state:
//! its own test/check is failing and it can't recover within the attempt
//! (observed on Gemma 4: writes `assembly_respects_system_prompt_override`,
//! the test fails, and it thrashes — reverts, replace_range↔refactor loops,
//! brace slips). This spins up a *fresh-context* sub-agent given ONLY the
//! specific failing-check output and the changed files, told to fix that one
//! thing. The bet is **attention reset / fresh eyes**, not extra capability
//! (same weights). It edits through the same revision store as the primary
//! agent, so its fix persists and the next gate re-check sees it.
//!
//! Cost is paid only when flailing (reactive, not every run), which on a
//! single-GPU serial setup matters — see #40's caveats.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use parking_lot::Mutex;

use crate::config::Config;
use crate::lsp::LspClient;
use crate::mcp::McpRegistry;
use crate::runtime::{LlmWorkerHandle, ToolWorkerPool};
use crate::tools;
use crate::tools::permissions::PermissionManager;

use super::subagent::{self, AgentTask};

/// Number of done-gate blocks in a turn after which the debugger fires once.
/// Two lets the primary agent take its own swings first (the cheap path);
/// the debugger is the escalation when those swings keep missing.
pub const DEBUGGER_TRIGGER_BLOCKS: usize = 2;

/// Run the fresh-context debugger sub-agent against a blocking check failure.
/// Returns the sub-agent's final report (to surface back to the primary agent),
/// or `None` if it produced nothing. Any fix it makes is already on disk via
/// the shared revision store — the report is advisory, the edits are the work.
#[allow(clippy::too_many_arguments)]
pub async fn run_debugger(
    failure_output: &str,
    config: &Config,
    llm_worker: &LlmWorkerHandle,
    tool_pool: &ToolWorkerPool,
    parent_tool_defs: &[crate::llm::ToolDefinition],
    perms: &Arc<PermissionManager>,
    mcp_registry: &Option<Arc<Mutex<McpRegistry>>>,
    lsp: &Option<Arc<LspClient>>,
    fast_revisions: &Option<Arc<tools::RevisionStore>>,
    fast_baseline_errors: usize,
    cancelled: &Arc<AtomicBool>,
) -> Option<String> {
    let changed = changed_files(config);
    let task = AgentTask {
        label: "debugger".to_string(),
        prompt: build_prompt(failure_output, &changed),
    };

    let mut outputs = subagent::run_subagents(
        vec![task],
        config,
        llm_worker,
        tool_pool,
        parent_tool_defs,
        perms,
        mcp_registry,
        lsp,
        fast_revisions,
        fast_baseline_errors,
        cancelled,
        None,
    )
    .await;

    outputs
        .pop()
        .map(|o| o.content.trim().to_string())
        .filter(|c| !c.is_empty())
}

/// Best-effort list of files changed in the working tree (modified + staged +
/// untracked), so the debugger knows where to look without re-discovering the
/// whole task. Empty vec if git is unavailable — the debugger can still search.
fn changed_files(config: &Config) -> Vec<String> {
    let Ok(out) = std::process::Command::new("git")
        .arg("-C")
        .arg(&config.project_root)
        .args(["status", "--porcelain"])
        .output()
    else {
        return Vec::new();
    };
    if !out.status.success() {
        return Vec::new();
    }
    parse_porcelain(&String::from_utf8_lossy(&out.stdout))
}

/// Parse `git status --porcelain` output into a list of paths. Strips the
/// two-column status prefix and handles rename arrows (`old -> new` keeps new).
fn parse_porcelain(stdout: &str) -> Vec<String> {
    stdout
        .lines()
        .filter_map(|line| {
            if line.len() < 4 {
                return None;
            }
            let path = line[3..].trim();
            let path = path.rsplit(" -> ").next().unwrap_or(path);
            (!path.is_empty()).then(|| path.trim_matches('"').to_string())
        })
        .collect()
}

fn build_prompt(failure_output: &str, changed: &[String]) -> String {
    let files = if changed.is_empty() {
        "(could not list changed files — use file(action='search') to locate the relevant code)"
            .to_string()
    } else {
        changed
            .iter()
            .map(|f| format!("  - {f}"))
            .collect::<Vec<_>>()
            .join("\n")
    };

    format!(
        "You are a focused debugger. A verification check is BLOCKING task completion. \
         Your ONLY job is to make that check pass by fixing the specific problem it reports. \
         Do NOT refactor unrelated code, add features, or change anything the failure does not point at.\n\
         \n\
         The check failed with this output:\n\
         ----------------------------------------\n\
         {failure_output}\n\
         ----------------------------------------\n\
         \n\
         Files changed so far this session:\n\
         {files}\n\
         \n\
         Approach: read the exact location the failure names, find the root cause of THIS error \
         (it may be a compile error, a plumbed-but-unconsumed value, or a missing guard — read \
         carefully, do not assume), make one targeted fix, and verify it with `check`. \
         When the specific failure is addressed, stop and report in 1-2 sentences what was wrong \
         and what you changed. The verification will be re-run automatically after you finish."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_porcelain_extracts_paths() {
        let out = " M src/cli/mod.rs\n?? new_file.rs\nA  staged.rs\n";
        let paths = parse_porcelain(out);
        assert_eq!(paths, vec!["src/cli/mod.rs", "new_file.rs", "staged.rs"]);
    }

    #[test]
    fn parse_porcelain_handles_rename_arrow() {
        let out = "R  old/name.rs -> new/name.rs\n";
        assert_eq!(parse_porcelain(out), vec!["new/name.rs"]);
    }

    #[test]
    fn parse_porcelain_empty() {
        assert!(parse_porcelain("").is_empty());
    }

    #[test]
    fn build_prompt_includes_failure_and_files() {
        let p = build_prompt("EXPECTED foo GOT bar", &["src/a.rs".to_string()]);
        assert!(p.contains("EXPECTED foo GOT bar"));
        assert!(p.contains("src/a.rs"));
        assert!(p.contains("ONLY job"));
    }

    #[test]
    fn build_prompt_handles_no_changed_files() {
        let p = build_prompt("boom", &[]);
        assert!(p.contains("could not list changed files"));
    }
}
