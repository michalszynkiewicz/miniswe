//! Model, routing, and tool-call-format configuration.

use serde::{Deserialize, Serialize};

use crate::llm::Provider;

/// Local-provider default endpoint. Shared between `ModelConfig::default`
/// and `providers::endpoint::effective_endpoint`'s "configured endpoint is
/// still the untouched local default" check — kept as one constant so the
/// two can't drift.
pub(crate) const DEFAULT_ENDPOINT: &str = "http://localhost:8464";

/// Fallback context window (tokens) when neither the user configured one
/// nor the server's startup probe reported one. See
/// [`ModelConfig::context_window`].
pub const DEFAULT_CONTEXT_WINDOW: usize = 50_000;

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

/// LLM provider and model settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    /// Provider type: "llama-cpp", "ollama", "vllm", "openai-compatible",
    /// "openrouter", "openai", or "anthropic". Unknown values are treated
    /// as "openai-compatible" with a warning — see `Provider::parse`.
    pub provider: String,
    /// API endpoint URL
    pub endpoint: String,
    /// Model name/identifier
    pub model: String,
    /// Context window in tokens. Unset (the default) means auto: taken
    /// from the server's startup probe when it reports one, else 50000. A
    /// configured value always wins and is never validated against the
    /// server.
    pub context_window: Option<usize>,
    /// Sampling temperature (low for code tasks)
    pub temperature: f64,
    /// Enable thinking-mode reasoning (`enable_thinking: true`) on the main
    /// agent loop and the debugger. Mechanical sub-roles (edit apply,
    /// summarizer, routers) always run without thinking — they don't benefit
    /// and reasoning tokens would slow their tight loops.
    pub thinking: bool,
    /// Sampling temperature for thinking-mode requests. Reasoning traces
    /// degenerate at code-task temperatures (Nemotron 3.5 whitespace-flood
    /// probes; Unsloth recommends 0.6 for Nemotron thinking vs 0.2 instruct),
    /// so thinking requests override `temperature` with this value.
    pub thinking_temperature: f64,
    /// Maximum output tokens per response. This is a runaway BRAKE, not a
    /// capability budget: a real tool call is small (the 08-28 demo-e2e-task
    /// run had a 93-token median generation, p90 289, and the largest output
    /// the harness actually consumed was 1077 tokens), while a repetition
    /// loop inside a tool argument runs to whatever ceiling this sets. At
    /// 16384 that cost three requests 583s/571s/451s -- 26.7 min, 10% of all
    /// server time -- and at least two were discarded outright ("arguments
    /// were cut off by the output limit", "generation truncated by context
    /// ceiling"). `tool_call_repair` already recovers from those; the cap
    /// decides how long it waits first. Models that genuinely need more get
    /// it explicitly (see the Mistral Small 4 reasoning override in `run.rs`)
    /// or from a bench script's own config.
    pub max_output_tokens: usize,
    /// Ceiling on the connect phase of an LLM request — from send to
    /// receiving response headers (the start of the SSE stream), not the
    /// full generation (see `stream_idle_timeout_secs`/`request_deadline_secs`
    /// for that). A server-side connect wedge is retried as a transient
    /// failure on timeout. Empirically (27K+ requests analyzed across many
    /// benchmark runs on this project's local llama-server setup) the
    /// largest clean connect+prefill observed was ~24s even at the biggest
    /// real context sizes (24K+ tokens); genuine wedges cluster tightly at
    /// the old 120s default with a near-empty gap in between (42-100s),
    /// confirming wedges are a distinct failure mode, not slow-but-healthy
    /// prefill. 30s gives ~25% margin over the largest clean case observed
    /// while cutting wasted wedge-recovery time 4x (120s → 30s); the rare
    /// case of a legitimate prefill exceeding this just costs one extra
    /// retry cycle, safely bounded by `request_deadline_secs`.
    pub request_timeout_secs: u64,
    /// Idle-timeout FLOOR (seconds) for streamed LLM responses. If no token
    /// activity is observed for this long the request is killed and retried
    /// as a transient failure. Distinguishes "stuck connection" from "model
    /// thinking hard" — a model that is producing tokens steadily, even
    /// slowly, is *not* idle.
    ///
    /// A floor, not the whole story: prompt *prefill* emits no token at all,
    /// so `attempt_idle_timeout` widens this per request to cover the prompt
    /// the server has to evaluate (capped at `request_deadline_secs`). At the
    /// old flat 30s a warm request whose cached prefix had been evicted was
    /// killed mid-prefill — 215 times in one run, ~61% of its wall clock.
    /// 120s covers a ~5k-token re-prefill on a CPU-offloaded MoE outright,
    /// and the size-aware widening covers the rest.
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
    /// Context window reported by the server's startup probe (llama.cpp
    /// `/props`, vLLM `max_model_len`, OpenRouter `context_length`).
    /// Skipped from (de)serialization since it's a runtime probe result,
    /// not user config. See [`Self::context_window`].
    #[serde(skip)]
    pub probed_context_window: Option<usize>,
    /// How to interpret tool calls from the model. `auto` (default) accepts
    /// OpenAI JSON tool_calls and falls back to parsing Anthropic-style XML
    /// embedded in content when tool_calls is empty. `json` ignores XML;
    /// `xml` skips JSON parsing and always treats content as the source.
    /// The override exists so future models with known formats can be pinned
    /// without relying on detection.
    #[serde(default)]
    pub tool_call_format: ToolCallFormat,
    /// API key for a hosted provider. Belongs in `~/.miniswe/config.toml`
    /// or the environment (`api_key_env` / the provider's conventional
    /// env var) — never in a project's `.miniswe/config.toml`, which a
    /// user's project may not gitignore. `miniswe init` never writes this
    /// field (see `init_default_config_has_no_api_key`); `miniswe config`
    /// / `miniswe info` show only where a key came from, never its value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Name of an environment variable to read the API key from, when
    /// `api_key` isn't set directly. Falls back further to the provider's
    /// conventional variable (`OPENROUTER_API_KEY`, `OPENAI_API_KEY`,
    /// `ANTHROPIC_API_KEY`) — see `providers::auth::resolve_api_key`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    /// Reasoning effort sent to OpenRouter/OpenAI when `thinking = true`
    /// (`"low"` / `"medium"` / `"high"`, provider-dependent).
    #[serde(default = "default_thinking_effort")]
    pub thinking_effort: String,
    /// Anthropic extended-thinking token budget (`thinking.budget_tokens`),
    /// used when `thinking = true` and the provider is `anthropic`.
    #[serde(default = "default_thinking_budget_tokens")]
    pub thinking_budget_tokens: usize,
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
    120
}

fn default_request_deadline_secs() -> u64 {
    600
}

fn default_thinking_effort() -> String {
    "medium".into()
}

fn default_thinking_budget_tokens() -> usize {
    2048
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

    /// True if the served model's attention window is narrower than a typical
    /// `[CURRENT STATE]` block, so re-anchoring that block every round throws
    /// away the whole KV cache instead of trimming its tail.
    ///
    /// llama.cpp reuses a cached prompt only as a pure EXTENSION. A shorter
    /// prefix is normally served by trimming the KV tail, but positions older
    /// than a sliding-window model's window cannot be rolled back to at all,
    /// so the sequence is cleared and re-prefilled from token zero. The
    /// threshold is exactly that model's `attention.sliding_window` — measured
    /// on Laguna XS at 512 tokens, where a 512-token rewind cost 496 tokens of
    /// prefill and a 528-token rewind cost the full 21,491.
    ///
    /// Only two families we run sit below a real block (~600 bytes median,
    /// ~2.3 KB on long plans): Laguna 2.1 at 512 tokens (~1.8 KB) and
    /// gpt-oss-20b at 128 (~444 B). Gemma 4 (1024), Muse Glimmer (2048) and
    /// North Mini Code (4096) are wide enough that re-anchoring never crosses
    /// the cliff, and Devstral / Mistral Small 4 / Nemotron 3.5 / Qwen3 use
    /// full attention and have no cliff at all.
    ///
    /// Unknown models answer `false`, which is both the safe default and the
    /// empirically better one: replaying 223 benchmark runs, always
    /// re-anchoring won or tied on every family except these two, and it never
    /// leaves a superseded copy in history.
    ///
    /// Matched against the server-reported identity rather than the
    /// user-supplied alias, same as [`Self::is_mistral_small_4_family`]. Every
    /// llama-server we run reports the GGUF path, which carries the family
    /// name.
    pub fn has_narrow_attention_window(&self) -> bool {
        // Unlike the reasoning_effort gate, this falls back to the configured
        // alias when the probe failed. A silent `None` would disable the
        // stickiness on exactly the models that need it, and the alias is a
        // usable signal here: the benchmark harness writes the full GGUF path
        // into `model`, and so does anyone pointing miniswe at a local file.
        let name = self
            .probed_model
            .as_deref()
            .unwrap_or(&self.model)
            .to_ascii_lowercase();
        name.contains("laguna") || name.contains("gpt-oss")
    }

    /// Effective context window: the configured value, else the server's
    /// probed value, else [`DEFAULT_CONTEXT_WINDOW`]. A configured value
    /// always wins and is never validated against what the server reports.
    pub fn context_window(&self) -> usize {
        self.context_window
            .or(self.probed_context_window)
            .unwrap_or(DEFAULT_CONTEXT_WINDOW)
    }

    /// Where [`Self::context_window`]'s effective value came from:
    /// `"config"`, `"server"`, or `"default"`.
    pub fn context_window_source(&self) -> &'static str {
        if self.context_window.is_some() {
            "config"
        } else if self.probed_context_window.is_some() {
            "server"
        } else {
            "default"
        }
    }

    /// Parsed [`Provider`] for `self.provider`.
    pub fn provider_kind(&self) -> Provider {
        Provider::parse(&self.provider)
    }

    /// The endpoint actually used: `self.endpoint` unless it is empty or
    /// still the untouched local default, in which case a hosted
    /// provider's own default endpoint takes over. See
    /// `providers::endpoint::effective_endpoint`.
    pub fn effective_endpoint(&self) -> String {
        crate::llm::providers::effective_endpoint(self.provider_kind(), &self.endpoint)
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            provider: "llama-cpp".into(),
            endpoint: DEFAULT_ENDPOINT.into(),
            model: "devstral-small-2".into(),
            context_window: None,
            temperature: 0.15,
            thinking: false,
            thinking_temperature: 0.6,
            max_output_tokens: 4096,
            request_timeout_secs: 30,
            stream_idle_timeout_secs: default_stream_idle_timeout_secs(),
            request_deadline_secs: default_request_deadline_secs(),
            max_retries: 6,
            probed_model: None,
            probed_context_window: None,
            tool_call_format: ToolCallFormat::Auto,
            api_key: None,
            api_key_env: None,
            thinking_effort: default_thinking_effort(),
            thinking_budget_tokens: default_thinking_budget_tokens(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_window_precedence() {
        // Configured value always wins, even over a probed one.
        let mut config = ModelConfig {
            context_window: Some(40_000),
            probed_context_window: Some(60_000),
            ..ModelConfig::default()
        };
        assert_eq!(config.context_window(), 40_000);
        assert_eq!(config.context_window_source(), "config");

        // No configured value: fall back to the server's probed value.
        config.context_window = None;
        assert_eq!(config.context_window(), 60_000);
        assert_eq!(config.context_window_source(), "server");

        // Neither configured nor probed: the hardcoded default.
        config.probed_context_window = None;
        assert_eq!(config.context_window(), DEFAULT_CONTEXT_WINDOW);
        assert_eq!(config.context_window_source(), "default");
    }
}
