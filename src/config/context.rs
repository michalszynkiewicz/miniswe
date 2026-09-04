//! Context-assembly configuration: token budgets, providers, and the
//! conversation-compaction strategy.

use serde::{Deserialize, Serialize};

/// Token budget allocation for context assembly.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ContextConfig {
    /// Token budget for the repo map slice
    pub repo_map_budget: usize,
    /// Maximum tool call rounds before stopping
    pub max_rounds: usize,
    /// Ask user to confirm continuation after this many rounds
    pub pause_after_rounds: usize,
    /// Toggle individual context providers on/off.
    pub providers: ProvidersConfig,
    /// Conversation-compaction strategy — how over-budget history is reduced.
    pub compaction: CompactionStrategy,
}

/// Conversation-compaction strategy: how the agent reduces conversation
/// history once it exceeds the raw-history token budget.
///
/// All strategies fire at the **same** trigger threshold (`raw_budget`, see
/// `compressor::needs_compression`); they differ only in the *action* taken,
/// so they can be A/B'd cleanly. `Unified` is miniswe's production behavior;
/// the others are canonical baselines used for benchmarking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CompactionStrategy {
    /// miniswe production: rolling LLM summary anchored on the plan, keeping
    /// recent turns raw, with the full pre-compression text archived to
    /// `.miniswe/session_archive.md` (and a pointer to it in the summary).
    Unified,
    /// Pure truncation: drop the oldest turns, keep the most-recent turns
    /// within budget. No summary, no LLM call, no archive.
    SlidingWindow,
    /// Textbook rolling LLM summarization: summarize the old turns into a
    /// running summary and keep recent turns raw. No plan-anchor, no disk
    /// archive, neutral summarization prompt.
    RollingSummary,
    /// Observation masking: keep the full action trajectory (assistant
    /// messages, tool calls, user turns) but replace old tool *observations*
    /// (results) with a short placeholder, keeping the last few raw. No LLM.
    ObservationMasking,
    /// Tiered hybrid: mask old observations first (cheap, free), and only if
    /// that doesn't get under budget fall through to the `Unified` summary +
    /// archive (the hard cap). Avoids observation-masking's edit-heavy thrash
    /// while keeping its cheapness when observations dominate.
    Tiered,
    /// Like `Tiered`, but the tier-2 cap is `RollingSummary` (running summary,
    /// no plan-anchor, no disk archive) instead of `Unified`.
    TieredRolling,
    /// `Tiered` plus a system-prompt nudge telling the model to record
    /// non-re-derivable findings (command output, search results, errors) to
    /// its scratchpad before they're elided. Same compaction behavior as
    /// `Tiered`; differs only in the prompt.
    TieredSmart,
    /// Reactive ("lazy") compaction, OpenCode-style: never compact
    /// proactively — let history grow until the SERVER signals context
    /// exhaustion (a rejected over-size request, or a generation truncated
    /// by the context ceiling), then compact once via the `Unified`
    /// summary+archive action and retry. Rationale: proactive strategies
    /// here trigger at ~26% of the window and re-fire every 1-2 rounds in
    /// steady state (compaction lands a mean of only ~145 tokens below its
    /// own trigger); a reactive policy uses ~the whole window and fires
    /// rarely. Trade-off: bigger per-round prompts (weaker KV-cache
    /// locality), and each compaction event is a large, expensive summary
    /// instead of many small ones.
    ///
    /// The DEFAULT since 2026-07-15: deepest-validated strategy of the
    /// matrix (9x 6/6 across three bench batches + the jobs e2e), fires
    /// rarely, and recovers cleanly at the ceiling.
    #[default]
    Lazy,
}

/// Which context providers are enabled.
///
/// Each field corresponds to a `ContextProvider::name()`. Set to `false` in
/// config.toml to disable that provider:
///
/// ```toml
/// [context.providers]
/// lessons = false
/// repo_map = false
/// ```
///
/// Plan and scratchpad are NOT here — they used to be system-prompt
/// providers, but they're the only two pieces of injected state the agent
/// itself mutates mid-run (via `plan(action=...)`), which made the system
/// prompt go stale between refreshes. They're now attached at the point
/// they change instead — see `cli::commands::run::refresh_current_state`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ProvidersConfig {
    pub profile: bool,
    pub guide: bool,
    pub project_notes: bool,
    pub lessons: bool,
    pub repo_map: bool,
    pub mcp: bool,
    pub usage_guide: bool,
    pub plan_mode: bool,
    /// Installed-skills listing (names + descriptions, bodies on demand).
    pub skills: bool,
}

impl Default for ProvidersConfig {
    fn default() -> Self {
        Self {
            // Auto-injected by default: the compaction benchmark (2026-06-27)
            // showed leaving these off costs gemma-4 the 6/6 (5/6 -> 6/6 when on)
            // — the codebase orientation + lessons help the model thread a change
            // end-to-end. Each is a no-op when its source file is absent, and all
            // remain fetchable on demand via the get_project_info()/notes tools.
            profile: true,
            guide: true,
            project_notes: true,
            lessons: true,
            repo_map: false, // still on-demand via code(action='repo_map')
            mcp: true,
            usage_guide: true,
            plan_mode: true,
            skills: true,
        }
    }
}

impl ProvidersConfig {
    /// Check if a provider is enabled by name.
    pub fn is_enabled(&self, name: &str) -> bool {
        match name {
            "profile" => self.profile,
            "guide" => self.guide,
            "project_notes" => self.project_notes,
            "lessons" => self.lessons,
            "repo_map" => self.repo_map,
            "mcp" => self.mcp,
            "usage_guide" => self.usage_guide,
            "plan_mode" => self.plan_mode,
            "skills" => self.skills,
            _ => true, // unknown providers default to enabled
        }
    }
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            repo_map_budget: 5000,
            max_rounds: 100,
            pause_after_rounds: 50,
            providers: ProvidersConfig::default(),
            compaction: CompactionStrategy::default(),
        }
    }
}

#[cfg(test)]
mod compaction_strategy_tests {
    use super::*;

    #[test]
    fn defaults_to_lazy() {
        // Bench-validated default (3 batches of 6/6): reactive compaction.
        assert_eq!(
            ContextConfig::default().compaction,
            CompactionStrategy::Lazy
        );
    }

    #[test]
    fn parses_snake_case_from_toml() {
        let c: ContextConfig = toml::from_str("compaction = \"sliding_window\"").unwrap();
        assert_eq!(c.compaction, CompactionStrategy::SlidingWindow);
        let c: ContextConfig = toml::from_str("compaction = \"observation_masking\"").unwrap();
        assert_eq!(c.compaction, CompactionStrategy::ObservationMasking);
        let c: ContextConfig = toml::from_str("compaction = \"rolling_summary\"").unwrap();
        assert_eq!(c.compaction, CompactionStrategy::RollingSummary);
        let c: ContextConfig = toml::from_str("compaction = \"tiered\"").unwrap();
        assert_eq!(c.compaction, CompactionStrategy::Tiered);
        let c: ContextConfig = toml::from_str("compaction = \"tiered_smart\"").unwrap();
        assert_eq!(c.compaction, CompactionStrategy::TieredSmart);
        let c: ContextConfig = toml::from_str("compaction = \"tiered_rolling\"").unwrap();
        assert_eq!(c.compaction, CompactionStrategy::TieredRolling);
        let c: ContextConfig = toml::from_str("compaction = \"lazy\"").unwrap();
        assert_eq!(c.compaction, CompactionStrategy::Lazy);
    }

    #[test]
    fn missing_field_keeps_default() {
        // Old configs with no `compaction` key still parse (struct-level serde default).
        let c: ContextConfig = toml::from_str("repo_map_budget = 1234").unwrap();
        assert_eq!(c.compaction, CompactionStrategy::Lazy);
        assert_eq!(c.repo_map_budget, 1234);
    }
}
