//! The top-level [`Config`]: layered loading, project/session paths, and
//! the small leaf config sections.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use super::session;
use super::{ContextConfig, ModelConfig, ModelRole, RoutingConfig, ToolsConfig};

/// Top-level configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub model: ModelConfig,
    /// Named model slots for multi-model routing.
    /// If present, these override `model` for the corresponding roles.
    /// Each slot is an independent `[models.<name>]` table with its own
    /// `provider`/`endpoint`/`model` — a local llama-swap/vLLM instance
    /// serving multiple models behind one endpoint, a mix of local and
    /// hosted providers, or several hosted accounts side by side.
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
    /// Id of this process's session (not serialized). Session working
    /// state — `plan.md`, `scratchpad.md` — lives under
    /// `.miniswe/sessions/<session_id>/` so concurrent or nested miniswe
    /// runs in one project can't clobber each other. See `config::session`.
    #[serde(skip)]
    pub session_id: String,
    /// Whether current-state refresh injects the active skill-step block
    /// (`[SKILL STEP]`). Runtime-only: set by the headless `run` surface,
    /// which registers the `skill` tool the block tells the model to call.
    /// The repl shares the same on-disk cursor (a killed run leaves one
    /// behind) but has no `skill` tool, so injecting there would demand a
    /// call the model cannot make — with no way to advance or clear it.
    #[serde(skip)]
    pub skill_step_injection: bool,
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
    /// Opt-in budget guard: stop the turn once this session's cumulative
    /// prompt-token usage (summed across every role, from `[usage]`
    /// logging) exceeds this many tokens. `0` disables the guard — the
    /// default, since it only matters once a hosted provider is billing
    /// per token. A bench run is on the order of ~100 rounds × ~30k prompt
    /// tokens; this is meant to catch a run that's grinding far past that,
    /// not to cap a normal one.
    #[serde(default)]
    pub max_session_input_tokens: u64,
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
            session_id: session::new_id(),
            skill_step_injection: false,
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
            max_session_input_tokens: 0,
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
        let global_path = Self::global_dir().map(|d| d.join("config.toml"));
        Self::load_from(global_path.as_deref(), project_root)
    }

    /// [`Config::load`] with the global config path given explicitly.
    ///
    /// The project file is deep-merged over the global one key by key, so a
    /// project overrides only what it sets: a project that tweaks
    /// `[context]` still uses the global `[model]`.
    pub fn load_from(global_path: Option<&Path>, project_root: PathBuf) -> Result<Self> {
        let global = match global_path.filter(|p| p.exists()) {
            Some(path) => Some(read_toml_table(path)?),
            None => None,
        };
        let project_config_path = project_root.join(".miniswe").join("config.toml");
        let project = if project_config_path.exists() {
            Some(read_toml_table(&project_config_path)?)
        } else {
            None
        };

        let mut config = match (global, project) {
            (None, None) => Config::default(),
            (Some(global), None) => global
                .try_into()
                .context("Failed to parse global config.toml")?,
            (global, Some(project)) => {
                let global_config: Config = match &global {
                    Some(g) => g
                        .clone()
                        .try_into()
                        .context("Failed to parse global config.toml")?,
                    None => Config::default(),
                };
                let mut merged = global.unwrap_or_else(|| toml::Value::Table(Default::default()));
                merge_toml(&mut merged, project);
                let mut config: Config = merged.try_into().with_context(|| {
                    format!("Failed to parse {}", project_config_path.display())
                })?;
                // The merge already carries global secrets into unset project
                // fields; this keeps the explicit guarantee in one place.
                super::secrets::inherit_secrets(&global_config, &mut config);
                config
            }
        };

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

    /// Directory holding every session's state directory.
    pub fn sessions_dir(&self) -> PathBuf {
        self.miniswe_dir().join("sessions")
    }

    /// This session's private state directory.
    pub fn session_dir(&self) -> PathBuf {
        self.sessions_dir().join(&self.session_id)
    }

    /// Path to a file within this session's state directory. Use this for
    /// anything a concurrent or nested run must not see or overwrite.
    pub fn session_path(&self, relative: &str) -> PathBuf {
        self.session_dir().join(relative)
    }

    /// Create this session's state directory. Call before writing session
    /// state; cheap and idempotent.
    pub fn ensure_session_dir(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(self.session_dir())
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
        self.model.context_window() / 10
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

fn read_toml_table(path: &Path) -> Result<toml::Value> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read {}", path.display()))?;
    toml::from_str(&contents).with_context(|| format!("Failed to parse {}", path.display()))
}

/// Overlay `overlay` onto `base`: tables merge recursively, any other value
/// (including arrays) in `overlay` replaces the one in `base`.
fn merge_toml(base: &mut toml::Value, overlay: toml::Value) {
    match (base, overlay) {
        (toml::Value::Table(base), toml::Value::Table(overlay)) => {
            for (key, value) in overlay {
                match base.get_mut(&key) {
                    Some(existing) => merge_toml(existing, value),
                    None => {
                        base.insert(key, value);
                    }
                }
            }
        }
        (base, overlay) => *base = overlay,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(path: &Path, contents: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, contents).unwrap();
    }

    const GLOBAL_OPENROUTER: &str = "[model]\nprovider = \"openrouter\"\nmodel = \"anthropic/claude-sonnet-4.5\"\napi_key_env = \"OPENROUTER_API_KEY\"\n";

    #[test]
    fn project_override_keeps_unset_global_keys() {
        let tmp = tempfile::tempdir().unwrap();
        let global = tmp.path().join("home/config.toml");
        write(&global, GLOBAL_OPENROUTER);
        let project = tmp.path().join("proj");
        write(
            &project.join(".miniswe/config.toml"),
            "[model]\ntemperature = 0.5\n\n[context]\nmax_rounds = 7\n",
        );

        let cfg = Config::load_from(Some(&global), project).unwrap();
        assert_eq!(cfg.model.model, "anthropic/claude-sonnet-4.5");
        assert_eq!(cfg.model.api_key_env.as_deref(), Some("OPENROUTER_API_KEY"));
        assert_eq!(cfg.model.temperature, 0.5);
        assert_eq!(cfg.context.max_rounds, 7);
    }

    #[test]
    fn project_value_wins_over_global() {
        let tmp = tempfile::tempdir().unwrap();
        let global = tmp.path().join("home/config.toml");
        write(&global, GLOBAL_OPENROUTER);
        let project = tmp.path().join("proj");
        write(
            &project.join(".miniswe/config.toml"),
            "[model]\nmodel = \"openai/gpt-5\"\n",
        );

        let cfg = Config::load_from(Some(&global), project).unwrap();
        assert_eq!(cfg.model.model, "openai/gpt-5");
        assert_eq!(cfg.model.api_key_env.as_deref(), Some("OPENROUTER_API_KEY"));
    }

    #[test]
    fn missing_files_fall_back_to_defaults() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg =
            Config::load_from(Some(&tmp.path().join("nope.toml")), tmp.path().into()).unwrap();
        assert_eq!(cfg.model.model, Config::default().model.model);
        assert_eq!(cfg.project_root, tmp.path());
    }
}
