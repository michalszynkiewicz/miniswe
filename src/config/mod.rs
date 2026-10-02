//! Configuration management for miniswe.
//!
//! Single config file at `~/.miniswe/config.toml` for all settings (model,
//! hardware, API keys). Project root is always the current working directory.
//! Per-project data (index, scratchpad, profile) lives in `.miniswe/` in the
//! project directory — created by `miniswe init`.

pub mod session;

mod context;
mod model;
mod root;
mod secrets;
mod tools;

pub use context::{CompactionStrategy, ContextConfig, ProvidersConfig};
pub(crate) use model::DEFAULT_ENDPOINT;
pub use model::{ModelConfig, ModelRole, RoutingConfig, ToolCallFormat};
pub use root::{
    Config, HardwareConfig, LogConfig, LspConfig, RuntimeConfig, ShellConfig, ValidationConfig,
    WebConfig,
};
pub use tools::{CeremonyMode, EditMode, ToolsConfig};
