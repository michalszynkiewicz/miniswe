//! Configuration management for buzztr.
//!
//! Global settings (model, hardware, API keys) live in
//! `~/.buzztr/config.toml`; an optional project `.buzztr/config.toml` is
//! deep-merged over it. Project root is always the current working directory.
//! Per-project data (index, scratchpad, profile) lives in `.buzztr/` in the
//! project directory — created by `buzztr init`.

pub mod session;

mod context;
mod model;
mod root;
mod secrets;
mod tools;

pub use context::{CompactionStrategy, ContextConfig, ProvidersConfig};
pub(crate) use model::DEFAULT_ENDPOINT;
pub use model::{DEFAULT_CONTEXT_WINDOW, ModelConfig, ModelRole, RoutingConfig, ToolCallFormat};
pub use root::{
    Config, HardwareConfig, LogConfig, LspConfig, RuntimeConfig, ShellConfig, ValidationConfig,
    WebConfig,
};
pub use tools::{CeremonyMode, EditMode, ToolsConfig};
