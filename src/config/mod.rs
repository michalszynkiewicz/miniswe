//! Configuration management for miniswe.
//!
//! Single config file at `~/.miniswe/config.toml` for all settings (model,
//! hardware, API keys). Project root is always the current working directory.
//! Per-project data (index, scratchpad, profile) lives in `.miniswe/` in the
//! project directory — created by `miniswe init`.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Top-level configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub model: ModelConfig,
    /// Named model slots for multi-model routing.
    /// If present, these override `model` for the corresponding roles.
    /// Requires an OpenAI-compatible proxy like llama-swap to handle
    /// on-demand model loading behind a single endpoint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub models: Option<HashMap<String, ModelConfig>>,
    /// Which model slot to use for each role.
    pub routing: RoutingConfig,
    pub context: ContextConfig,
    pub hardware: HardwareConfig,
    pub web: WebConfig,
    pub shell: ShellConfig,
    pub runtime: RuntimeConfig,
    pub logging: LogConfig,
    pub lsp: LspConfig,
    pub tools: ToolsConfig,
    pub validation: ValidationConfig,
    /// Resolved project root directory (not serialized).
    #[serde(skip)]
    pub project_root: PathBuf,
}

/// Which named model slot to use for each role.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RoutingConfig {
    /// Model for general use / coding (default fallback).
    pub default: String,
    /// Model for planning and complex reasoning.
    pub plan: String,
    /// Model for fast/lightweight tasks (summaries, scratchpad).
    pub fast: String,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            default: "default".into(),
            plan: "default".into(),
            fast: "default".into(),
        }
    }
}

/// Which model to use for a given operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelRole {
    /// General / coding tasks.
    Default,
    /// Planning, complex reasoning, architecture.
    Plan,
    /// Fast lightweight tasks (summaries, scratchpad).
    Fast,
}

/// Logging configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LogConfig {
    /// Log verbosity: "info", "debug", "trace"
    /// - info: tool calls and outcomes (one-liner per action)
    /// - debug: full interactions — LLM messages, tool args/results, file changes
    /// - trace: everything + context assembly stats, token counts, masking decisions
    pub level: String,
    /// Whether to write session logs to .miniswe/logs/
    pub enabled: bool,
}

/// LLM provider and model settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    /// Provider type: "llama-cpp", "ollama", "vllm", "openai-compatible"
    pub provider: String,
    /// API endpoint URL
    pub endpoint: String,
    /// Model name/identifier
    pub model: String,
    /// Context window size in tokens
    pub context_window: usize,
    /// Sampling temperature (low for code tasks)
    pub temperature: f64,
    /// Maximum output tokens per response
    pub max_output_tokens: usize,
    /// Timeout in seconds for non-streaming chat requests.
    pub request_timeout_secs: u64,
    /// Idle timeout (seconds) for streamed LLM responses. If no token
    /// activity is observed for this many seconds the request is killed
    /// and retried as a transient failure. Distinguishes "stuck
    /// connection" from "model thinking hard" — a model that is
    /// producing tokens steadily, even slowly, is *not* idle.
    #[serde(default = "default_stream_idle_timeout_secs")]
    pub stream_idle_timeout_secs: u64,
    /// Absolute wall-clock deadline (seconds) for a single logical LLM
    /// call, spanning all internal retries. A BACKSTOP, not the primary
    /// guard: the idle timeout (above) catches normal stalls in seconds,
    /// but it measures inter-token *silence* — a wedged server that keeps
    /// the connection alive with keep-alive bytes can reset it forever
    /// (observed: a refactor inner-call hung ~47min with no timeout). This
    /// ceiling fires regardless. Must clear the worst-case *legitimate*
    /// full generation (max_output_tokens at the model's slowest healthy
    /// rate) or it false-kills good requests — default 600s is ~2-3x the
    /// ~200-300s worst case for an 8k-token Gemma-4 response on a 3090.
    #[serde(default = "default_request_deadline_secs")]
    pub request_deadline_secs: u64,
    /// Maximum number of transient retry attempts for LLM requests.
    pub max_retries: usize,
    /// Server-reported model identity from `/v1/models`, populated at
    /// startup. Drives model-family checks more reliably than the
    /// user-supplied `model` string, which may be a generic alias like
    /// `"default"` or a llama-swap slot name. Skipped from (de)serialization
    /// since it's a runtime probe result, not user config.
    #[serde(skip)]
    pub probed_model: Option<String>,
    /// How to interpret tool calls from the model. `auto` (default) accepts
    /// OpenAI JSON tool_calls and falls back to parsing Anthropic-style XML
    /// embedded in content when tool_calls is empty. `json` ignores XML;
    /// `xml` skips JSON parsing and always treats content as the source.
    /// The override exists so future models with known formats can be pinned
    /// without relying on detection.
    #[serde(default)]
    pub tool_call_format: ToolCallFormat,
}

/// Wire format we expect the model to use for tool invocations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ToolCallFormat {
    /// Try OpenAI JSON first; fall back to XML in content when tool_calls
    /// is empty and the content looks like a tool-call block.
    #[default]
    Auto,
    /// JSON tool_calls only. Ignore any XML in content.
    Json,
    /// Always parse XML from content. Ignore the tool_calls array.
    Xml,
}

fn default_stream_idle_timeout_secs() -> u64 {
    30
}

fn default_request_deadline_secs() -> u64 {
    600
}

impl ModelConfig {
    // (Removed `is_devstral_family`: it gated a Devstral-only carve-out —
    // hide `refactor`, keep `edit_file` — that protected against
    // `position`-arg mangling from the *old* `change_signature` tool.
    // The rename to `refactor` fixed the formatting; the gate's only
    // remaining effect was suppressing refactor adoption. All models now
    // get the uniform surface and a phase-aware system prompt drives
    // adoption instead. See context::build_system_prompt's plan_set branch.)

    /// True if the served model is Mistral Small 4 (the unified MoE that
    /// folded Magistral/Pixtral/Devstral into one model). Used to gate the
    /// `reasoning_effort` knob: Mistral Small 4 exposes a per-request
    /// reasoning_effort kwarg (`none`/`high`) — we want `high` when the
    /// model is deciding task decomposition (pre-plan) and `none` once
    /// edits are flowing. Probe-only: matched against the server-reported
    /// model identity, not the user-supplied config alias.
    pub fn is_mistral_small_4_family(&self) -> bool {
        match &self.probed_model {
            Some(probed) => {
                let p = probed.to_ascii_lowercase();
                p.contains("mistral-small-4") || p.contains("mistral_small_4")
            }
            None => false,
        }
    }
}

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
    #[default]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ProvidersConfig {
    pub profile: bool,
    pub guide: bool,
    pub project_notes: bool,
    pub plan: bool,
    pub lessons: bool,
    pub repo_map: bool,
    pub mcp: bool,
    pub scratchpad: bool,
    pub usage_guide: bool,
    pub plan_mode: bool,
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
            plan: true,
            lessons: true,
            repo_map: false, // still on-demand via code(action='repo_map')
            mcp: true,
            scratchpad: true,
            usage_guide: true,
            plan_mode: true,
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
            "plan" => self.plan,
            "lessons" => self.lessons,
            "repo_map" => self.repo_map,
            "mcp" => self.mcp,
            "scratchpad" => self.scratchpad,
            "usage_guide" => self.usage_guide,
            "plan_mode" => self.plan_mode,
            _ => true, // unknown providers default to enabled
        }
    }
}

/// Hardware configuration hints.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HardwareConfig {
    /// Total VRAM in GB
    pub vram_gb: f64,
    /// VRAM to reserve for OS/display (subtracted from vram_gb for model budget)
    pub vram_reserve_gb: f64,
    /// RAM budget for KV cache overflow
    pub ram_budget_gb: f64,
}

/// Web access configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WebConfig {
    /// Search backend: "serper" (default), "searxng"
    pub search_backend: String,
    /// API key for search provider (Serper: free at serper.dev)
    pub search_api_key: Option<String>,
    /// SearXNG URL (if search_backend = "searxng")
    pub searxng_url: Option<String>,
    /// Fetch backend: "jina" or "local"
    pub fetch_backend: String,
}

/// Shell tool configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ShellConfig {
    /// Default timeout in seconds for shell commands.
    pub default_timeout_secs: u64,
}

/// Runtime execution configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RuntimeConfig {
    /// Size of the shared tool worker pool.
    pub tool_worker_pool_size: usize,
    /// Maximum number of concurrent LLM requests across all agents.
    /// Default 1 serializes all LLM calls — appropriate for local models
    /// (llama.cpp, Ollama) that can only run one inference at a time.
    /// Increase for API providers that support true parallelism.
    pub llm_concurrency: usize,
}

/// Behavioral "done-gate" validation. When `command` is non-empty it runs
/// when the agent would otherwise finish; a non-zero exit blocks completion
/// and feeds the command's output back to the model so it can fix a change
/// that compiles/tests-green but doesn't actually work at runtime.
/// Default: empty command = gate disabled (no behavior change).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ValidationConfig {
    /// Shell command exercising the feature end-to-end. Empty = disabled.
    pub command: String,
    /// Timeout in seconds for the validation command.
    pub timeout_secs: u64,
    /// How many times to block-and-retry before accepting completion anyway,
    /// so a model that cannot fix it doesn't loop forever.
    pub max_retries: usize,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            command: String::new(),
            timeout_secs: 120,
            max_retries: 3,
        }
    }
}

impl ValidationConfig {
    /// The configured behavioral check, or `None` when disabled.
    pub fn command(&self) -> Option<&str> {
        let c = self.command.trim();
        if c.is_empty() { None } else { Some(c) }
    }
}

/// LSP integration configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LspConfig {
    /// Enable LSP integration (rust-analyzer for Rust projects).
    pub enabled: bool,
    /// Timeout in milliseconds for diagnostic responses after file changes.
    pub diagnostic_timeout_ms: u64,
}

/// Toggle which tool groups are available to the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ToolsConfig {
    /// Context tools: get_repo_map, get_project_info, get_architecture_notes
    pub context_tools: bool,
    /// LSP tools: goto_definition, find_references
    pub lsp_tools: bool,
    /// Web tools: web_search, web_fetch
    pub web_tools: bool,
    /// Structured plan tool
    pub plan: bool,
    /// Scratchpad (task_update)
    pub scratchpad: bool,
    /// Agent ceremony level. `"strict"` (DEFAULT): plan-first gating +
    /// phase-aware prompt + progress nudges — the proven-good behavior
    /// (Qwen 6/6, passes `smoke`). `"off"`: leaner/faster minimal
    /// prompt with no plan machinery, but the real docker bench proved
    /// it regresses the end-to-end `smoke` check (opt-in only). See
    /// `docs/tiered-agent-design.md` §Real-bench refutation.
    pub ceremony: CeremonyMode,
    /// Flat refactor tools. `false` (default): grouped
    /// `refactor{action,position,callsite_fill_in}` (proven 6/6).
    /// `true`: replace it with flat single-purpose
    /// `add_function_param`/`drop_function_param`/`rename_symbol` (no
    /// `position`/`callsite_fill_in` DSL — removes the documented
    /// deterministic Devstral mangling). Under A/B evaluation; see
    /// `docs/tiered-agent-design.md`.
    pub flat: bool,
    /// Edit-tool surface: `"fast"` (default) exposes the primitive
    /// `replace_range` / `insert_at` / `revert` / `check` surface from
    /// `src/tools/fast/`; `"smart"` replaces it with `edit_file`, which
    /// delegates to an inner-model planner. See `docs/fast-mode-design.md`.
    pub edit_mode: EditMode,
    /// EXPERIMENTAL (fast mode only). When `true`, after a structural edit
    /// (`replace_range` / `insert_at`) leaves the file's AST broken for
    /// `CASCADE_THRESHOLD` consecutive edits in a row, the file is forcibly
    /// reverted to the most recent AST-clean revision and the model is told
    /// to stop digging and make one balanced edit. Targets the observed
    /// brace-cascade loop (small models patch line-by-line into ever-deeper
    /// breakage). On by default: a 10-run Gemma-4 A/B showed it removes the
    /// catastrophic tail (ON ~5.8 with no 0/6 vs OFF ~3.75 incl. a 17-broken
    /// 0/6); set `false` for the pure fast-mode philosophy of tolerating
    /// transient broken AST. Triggers only on `CASCADE_THRESHOLD` *consecutive*
    /// broken-AST edits, so a deliberate 1–2-step broken intermediate is safe.
    pub auto_revert_ast_cascade: bool,
    /// EXPERIMENTAL. When `true`, after the behavioral done-gate
    /// (`[validation]`) blocks completion `DEBUGGER_TRIGGER_BLOCKS` times in a
    /// turn, spin up a fresh-context "debugger" sub-agent handed only the
    /// specific failing check output + the changed files and told to fix only
    /// that. The bet (see GitHub #40) is *attention reset / fresh eyes* on a
    /// "knows-it's-wrong-but-can't-recover" stall — not extra capability
    /// (same weights). `false` (default) keeps the gate's plain retry-nudge
    /// loop. Requires a `[validation]` command to do anything. A/B only.
    pub reactive_debugger: bool,
    /// EXPERIMENTAL (fast mode). When `true`, detect a *revert-loop* spiral —
    /// the agent reverting the same file to a clean revision
    /// `SPIRAL_REVERT_THRESHOLD` times in a turn (it's cycling: re-trying the
    /// same failing edits and undoing them). On detection, inject a reset
    /// message that names what was tried, says it failed, and forces a
    /// `plan(action='set'/'refine')` with a concrete redirection (use
    /// `refactor` for signature/callsite changes; one balanced edit for a
    /// thrashed region). API-probe on Gemma 4: silent revert 0/8 vs this
    /// framing ~8/8 at making it switch approach. `false` (default). A/B only.
    pub spiral_reset: bool,
    /// EXPERIMENTAL. When `true`, after the done-gate (`[validation]`) blocks
    /// `GATE_RESET_AFTER_BLOCKS` times in a turn, replace its in-context retry
    /// grinding with a CONTEXT RESET: drop the polluted conversation history and
    /// re-assemble a clean context (files persist on disk) — the in-session
    /// equivalent of a best-of-3 fresh attempt. Motivated by: in-context
    /// grinding thrashes in the failure-primed context (qwen: 121 rounds over 3
    /// blocks, still failed) while a fresh attempt fixed it fast (53 rounds).
    /// `false` by default: a controlled gemma-4 A/B (2026-06-29, 3 runs each,
    /// auto_revert on, unified) showed OFF is strictly better — 6.0 vs 5.67 and
    /// ~1.6× faster (≈839s vs ≈1380s). The reset fired 2–4×/run and caused
    /// re-work churn (≈336 vs ≈199 rounds) with no reliability payoff. The qwen
    /// motivation above may still hold on harder/long-repo tasks; opt in there.
    pub gate_context_reset: bool,
    /// EXPERIMENTAL (fast mode + snapshots). When `true`, if the project's LSP
    /// error count stays ABOVE the session baseline for `REVERT_TO_GREEN_BLOCKS`
    /// consecutive rounds (the agent is stuck not converging), revert the ENTIRE
    /// working tree to the last round that was green (compiled ≤ baseline) and
    /// tell the agent to restart from that clean base. Unlike
    /// `auto_revert_ast_cascade` (per-file, AST-syntax only), this is tree-wide
    /// and triggers on SEMANTIC breaks (type errors, deleted methods) that parse
    /// fine. Motivated by run2 (deleted `is_enabled`, broke a caller, ground 100+
    /// rounds unable to untangle its own change). `false` (default). A/B only.
    pub revert_to_green: bool,
}

/// Agent ceremony level — see `ToolsConfig::ceremony`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum CeremonyMode {
    /// No plan gate, no `PLAN CHECK`, no nudge epicycles, all edit
    /// tools visible, one minimal prompt. Leaner and faster, but the
    /// real docker bench proved it REGRESSES the end-to-end `smoke`
    /// check (Qwen3-Coder-Next: strict 6/6 vs off 5/6 smoke:FAIL, same
    /// HEAD/harness). Opt-in only — the synthetic probe that motivated
    /// it could not measure real multi-step value-threading. See
    /// docs/tiered-agent-design.md §Real-bench refutation.
    Off,
    /// Lean code path (no gate / nudges / phase, like Off) BUT the
    /// prompt strongly *advises* outlining the value-threading steps
    /// before editing. Tests whether decomposition *advice* (not gate
    /// *enforcement*) is the active ingredient for `smoke`. Opt-in,
    /// under evaluation — see docs/tiered-agent-design.md.
    Advise,
    /// Plan-first gating + phase-aware prompt + progress nudges. The
    /// proven-good default: matches Qwen's reliable historical 6/6 and
    /// passes `smoke` (the check that proves the feature works).
    #[default]
    Strict,
}

/// Which edit-tool surface to expose to the outer model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum EditMode {
    /// `edit_file` + inner-model planner.
    Smart,
    /// Fast-mode primitives: `replace_range`, `insert_at`, `revert`, `check`.
    #[default]
    Fast,
}

impl Default for ToolsConfig {
    fn default() -> Self {
        Self {
            context_tools: true,
            lsp_tools: true,
            web_tools: true,
            plan: true,
            scratchpad: true,
            ceremony: CeremonyMode::Strict,
            flat: false,
            edit_mode: EditMode::Fast,
            auto_revert_ast_cascade: true,
            reactive_debugger: false,
            spiral_reset: false,
            // Off: the controlled gemma A/B (2026-06-29) showed OFF is strictly
            // better (6.0 vs 5.67, ~1.6× faster) — the reset causes re-work churn
            // with no reliability gain on this task. See the field doc above.
            gate_context_reset: false,
            revert_to_green: false,
        }
    }
}

impl Default for LspConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            diagnostic_timeout_ms: 2000,
        }
    }
}

// --- Defaults ---

impl Default for Config {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            models: None,
            routing: RoutingConfig::default(),
            context: ContextConfig::default(),
            hardware: HardwareConfig::default(),
            web: WebConfig::default(),
            shell: ShellConfig::default(),
            runtime: RuntimeConfig::default(),
            logging: LogConfig::default(),
            lsp: LspConfig::default(),
            tools: ToolsConfig::default(),
            validation: ValidationConfig::default(),
            project_root: PathBuf::from("."),
        }
    }
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            level: "debug".into(),
            enabled: true,
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            provider: "llama-cpp".into(),
            endpoint: "http://localhost:8464".into(),
            model: "devstral-small-2".into(),
            context_window: 50000,
            temperature: 0.15,
            max_output_tokens: 16384,
            request_timeout_secs: 120,
            stream_idle_timeout_secs: default_stream_idle_timeout_secs(),
            request_deadline_secs: default_request_deadline_secs(),
            max_retries: 6,
            probed_model: None,
            tool_call_format: ToolCallFormat::Auto,
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

impl Default for HardwareConfig {
    fn default() -> Self {
        Self {
            vram_gb: 24.0,
            vram_reserve_gb: 3.0,
            ram_budget_gb: 80.0,
        }
    }
}

impl Default for WebConfig {
    fn default() -> Self {
        Self {
            search_backend: "serper".into(),
            search_api_key: None,
            searxng_url: None,
            fetch_backend: "jina".into(),
        }
    }
}

impl Default for ShellConfig {
    fn default() -> Self {
        Self {
            default_timeout_secs: 60,
        }
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            tool_worker_pool_size: 10,
            llm_concurrency: 1,
        }
    }
}

impl Config {
    /// Path to the global config directory (`~/.miniswe/`).
    pub fn global_dir() -> Option<PathBuf> {
        dirs::home_dir().map(|h| h.join(".miniswe"))
    }

    /// Load config with layered resolution:
    /// 1. Built-in defaults
    /// 2. `~/.miniswe/config.toml` — global settings (API keys, model, hardware)
    /// 3. `.miniswe/config.toml` in cwd — per-project overrides (optional)
    ///
    /// Project root is always the current working directory.
    pub fn load() -> Result<Self> {
        let project_root =
            std::env::current_dir().context("Failed to determine current directory")?;

        // Layer 1: global config (~/.miniswe/config.toml), or defaults
        let mut config = if let Some(global_path) = Self::global_dir()
            .map(|d| d.join("config.toml"))
            .filter(|p| p.exists())
        {
            let contents = std::fs::read_to_string(&global_path)
                .with_context(|| format!("Failed to read {}", global_path.display()))?;
            toml::from_str(&contents)
                .with_context(|| format!("Failed to parse {}", global_path.display()))?
        } else {
            Config::default()
        };

        // Layer 2: project config (.miniswe/config.toml), if present
        let project_config_path = project_root.join(".miniswe").join("config.toml");
        if project_config_path.exists() {
            let contents = std::fs::read_to_string(&project_config_path)
                .with_context(|| format!("Failed to read {}", project_config_path.display()))?;
            let project: Config = toml::from_str(&contents)
                .with_context(|| format!("Failed to parse {}", project_config_path.display()))?;

            // Project values override global. Inherit secrets from global
            // if not set in project (serde fills Options with None by default).
            let global_web = config.web.clone();
            config = project;
            if config.web.search_api_key.is_none() {
                config.web.search_api_key = global_web.search_api_key;
            }
            if config.web.searxng_url.is_none() {
                config.web.searxng_url = global_web.searxng_url;
            }
        }

        config.project_root = project_root;
        Ok(config)
    }

    /// Path to the `.miniswe/` data directory in the project.
    pub fn miniswe_dir(&self) -> PathBuf {
        self.project_root.join(".miniswe")
    }

    /// Path to a specific file within the project's `.miniswe/`.
    pub fn miniswe_path(&self, relative: &str) -> PathBuf {
        self.miniswe_dir().join(relative)
    }

    /// Check if this project has been initialized (`miniswe init` was run).
    pub fn is_initialized(&self) -> bool {
        self.miniswe_dir().is_dir()
    }

    /// Max characters for a single tool result.
    ///
    /// Budget: raw history gets 1/4 of context. We want ~10 recent results
    /// to fit unmasked. So each result ≈ context_window/40 tokens ≈ context_window/10 chars.
    /// For 32K context: ~3200 chars (~80 lines). For 50K: ~5000 chars (~125 lines).
    pub fn tool_output_budget_chars(&self) -> usize {
        self.model.context_window / 10
    }

    /// Get the model config for a given role.
    /// Returns the named model from `[models]` if configured, otherwise falls
    /// back to the single `[model]` config.
    pub fn model_for_role(&self, role: ModelRole) -> &ModelConfig {
        let slot_name = match role {
            ModelRole::Default => &self.routing.default,
            ModelRole::Plan => &self.routing.plan,
            ModelRole::Fast => &self.routing.fast,
        };

        self.models
            .as_ref()
            .and_then(|m| m.get(slot_name))
            .unwrap_or(&self.model)
    }

    /// Whether multiple distinct models are configured.
    pub fn is_multi_model(&self) -> bool {
        self.models.as_ref().is_some_and(|m| m.len() > 1)
    }
}

#[cfg(test)]
mod compaction_strategy_tests {
    use super::*;

    #[test]
    fn defaults_to_unified() {
        assert_eq!(
            ContextConfig::default().compaction,
            CompactionStrategy::Unified
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
    }

    #[test]
    fn missing_field_keeps_default() {
        // Old configs with no `compaction` key still parse (struct-level serde default).
        let c: ContextConfig = toml::from_str("repo_map_budget = 1234").unwrap();
        assert_eq!(c.compaction, CompactionStrategy::Unified);
        assert_eq!(c.repo_map_budget, 1234);
    }
}
