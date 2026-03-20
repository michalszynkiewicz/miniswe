//! Configuration management for miniswe.
//!
//! Single config file at `~/.miniswe/config.toml` for all settings (model,
//! hardware, API keys). Project root is always the current working directory.
//! Per-project data (index, scratchpad, profile) lives in `.miniswe/` in the
//! project directory — created by `miniswe init`.

use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Top-level configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub model: ModelConfig,
    pub context: ContextConfig,
    pub hardware: HardwareConfig,
    pub web: WebConfig,
    pub logging: LogConfig,
    /// Resolved project root directory (not serialized).
    #[serde(skip)]
    pub project_root: PathBuf,
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
}

/// Token budget allocation for context assembly.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ContextConfig {
    /// Token budget for the repo map slice
    pub repo_map_budget: usize,
    /// Token budget for retrieved code snippets
    pub snippet_budget: usize,
    /// Number of raw conversation turns to keep
    pub history_turns: usize,
    /// Token budget for conversation history
    pub history_budget: usize,
    /// Token budget for the scratchpad
    pub scratchpad_budget: usize,
    /// Maximum tool call rounds before stopping
    pub max_rounds: usize,
    /// Ask user to confirm continuation after this many rounds
    pub pause_after_rounds: usize,
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

// --- Defaults ---

impl Default for Config {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            context: ContextConfig::default(),
            hardware: HardwareConfig::default(),
            web: WebConfig::default(),
            logging: LogConfig::default(),
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
        }
    }
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            repo_map_budget: 5000,
            snippet_budget: 12000,
            history_turns: 5,
            history_budget: 6000,
            scratchpad_budget: 1500,
            max_rounds: 100,
            pause_after_rounds: 50,
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
        let project_root = std::env::current_dir()
            .context("Failed to determine current directory")?;

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

    /// Save configuration to `~/.miniswe/config.toml`.
    pub fn save(&self) -> Result<()> {
        let config_dir = Self::global_dir()
            .context("Cannot determine home directory")?;
        std::fs::create_dir_all(&config_dir)?;
        let config_path = config_dir.join("config.toml");
        let contents = toml::to_string_pretty(self)?;
        std::fs::write(&config_path, contents)?;
        Ok(())
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
}
