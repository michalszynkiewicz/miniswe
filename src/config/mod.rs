//! Configuration management for miniswe.
//!
//! Loads from `.miniswe/config.toml` in the project root, with defaults
//! for all values. Supports provider configuration for llama.cpp, Ollama, vLLM,
//! or any OpenAI-compatible endpoint.

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
    /// Resolved project root directory (not serialized).
    #[serde(skip)]
    pub project_root: PathBuf,
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
            project_root: PathBuf::from("."),
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
    /// Find the project root by looking for `.miniswe/` directory,
    /// walking up from the current directory.
    pub fn find_project_root() -> Option<PathBuf> {
        let mut dir = std::env::current_dir().ok()?;
        loop {
            if dir.join(".miniswe").is_dir() {
                return Some(dir);
            }
            if !dir.pop() {
                return None;
            }
        }
    }

    /// Load config from `.miniswe/config.toml` in the project root.
    /// Falls back to defaults if the file doesn't exist.
    pub fn load() -> Result<Self> {
        let project_root = Self::find_project_root()
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

        let config_path = project_root.join(".miniswe").join("config.toml");

        let mut config = if config_path.exists() {
            let contents = std::fs::read_to_string(&config_path)
                .with_context(|| format!("Failed to read config from {}", config_path.display()))?;
            toml::from_str(&contents)
                .with_context(|| format!("Failed to parse config from {}", config_path.display()))?
        } else {
            Config::default()
        };

        config.project_root = project_root;
        Ok(config)
    }

    /// Save configuration to `.miniswe/config.toml`.
    pub fn save(&self) -> Result<()> {
        let config_dir = self.project_root.join(".miniswe");
        std::fs::create_dir_all(&config_dir)?;
        let config_path = config_dir.join("config.toml");
        let contents = toml::to_string_pretty(self)?;
        std::fs::write(&config_path, contents)?;
        Ok(())
    }

    /// Path to the `.miniswe/` directory.
    pub fn miniswe_dir(&self) -> PathBuf {
        self.project_root.join(".miniswe")
    }

    /// Path to a specific file within `.miniswe/`.
    pub fn miniswe_path(&self, relative: &str) -> PathBuf {
        self.miniswe_dir().join(relative)
    }

    /// Check if this project has been initialized.
    pub fn is_initialized(&self) -> bool {
        self.miniswe_dir().is_dir()
    }
}
