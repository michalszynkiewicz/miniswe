//! Model router — selects the right LLM client based on task role.
//!
//! In single-model mode (just `[model]` in config), all roles use the same
//! client. In multi-model mode (`[models.*]` + `[routing]`), different roles
//! can target different models/endpoints — typically via llama-swap or
//! separate server instances.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use anyhow::Result;

use crate::config::{Config, ModelConfig, ModelRole};
use super::{ChatRequest, ChatResponse, LlmClient};

/// Routes LLM requests to the appropriate client based on model role.
pub struct ModelRouter {
    clients: HashMap<String, LlmClient>,
    configs: HashMap<String, ModelConfig>,
    default_key: String,
    routing_default: String,
    routing_plan: String,
    routing_fast: String,
}

impl ModelRouter {
    /// Build a router from the application config.
    pub fn new(config: &Config) -> Self {
        if let Some(models) = &config.models {
            let clients: HashMap<String, LlmClient> = models
                .iter()
                .map(|(k, v)| (k.clone(), LlmClient::new(v.clone())))
                .collect();
            let configs: HashMap<String, ModelConfig> = models.clone();
            let default_key = config.routing.default.clone();
            Self {
                clients,
                configs,
                default_key,
                routing_default: config.routing.default.clone(),
                routing_plan: config.routing.plan.clone(),
                routing_fast: config.routing.fast.clone(),
            }
        } else {
            // Single-model mode: wrap the lone [model] as "default"
            let mut clients = HashMap::new();
            clients.insert("default".into(), LlmClient::new(config.model.clone()));
            let mut configs = HashMap::new();
            configs.insert("default".into(), config.model.clone());
            Self {
                clients,
                configs,
                default_key: "default".into(),
                routing_default: "default".into(),
                routing_plan: "default".into(),
                routing_fast: "default".into(),
            }
        }
    }

    /// Get the LLM client for a given role.
    /// Falls back to the default client if the role's model isn't configured.
    pub fn client_for(&self, role: ModelRole) -> &LlmClient {
        let key = self.key_for(role);
        self.clients
            .get(key)
            .or_else(|| self.clients.get(&self.default_key))
            .expect("no LLM client configured — check [model] or [models] in config.toml")
    }

    /// Get the model config for a given role (for display / context budgeting).
    pub fn config_for(&self, role: ModelRole) -> &ModelConfig {
        let key = self.key_for(role);
        self.configs
            .get(key)
            .or_else(|| self.configs.get(&self.default_key))
            .expect("no model config found — check [model] or [models] in config.toml")
    }

    /// Whether multiple distinct models are configured.
    pub fn is_multi_model(&self) -> bool {
        self.clients.len() > 1
    }

    /// Model name for a given role (for display).
    pub fn model_name(&self, role: ModelRole) -> &str {
        &self.config_for(role).model
    }

    /// Summary of configured models for display at startup.
    pub fn startup_summary(&self) -> Vec<String> {
        if !self.is_multi_model() {
            let cfg = self.config_for(ModelRole::Default);
            return vec![format!(
                "Model: {} @ {}",
                cfg.model, cfg.endpoint
            )];
        }

        let mut lines = vec!["Models:".to_string()];
        let roles = [
            ("default", ModelRole::Default),
            ("plan", ModelRole::Plan),
            ("fast", ModelRole::Fast),
        ];
        // Deduplicate — don't repeat if plan/fast point to the same as default
        let default_key = self.key_for(ModelRole::Default);
        for (label, role) in &roles {
            let key = self.key_for(*role);
            if *label != "default" && key == default_key {
                continue; // Same as default, skip
            }
            let cfg = self.config_for(*role);
            lines.push(format!(
                "  {label}: {} @ {}",
                cfg.model, cfg.endpoint
            ));
        }
        lines
    }

    /// Non-streaming chat request (for summarization, etc.).
    pub async fn chat(&self, role: ModelRole, request: &ChatRequest) -> Result<ChatResponse> {
        self.client_for(role).chat(request).await
    }

    /// Convenience: stream a chat request using the client for the given role.
    pub async fn chat_stream<F>(
        &self,
        role: ModelRole,
        request: &ChatRequest,
        on_token: F,
        cancelled: &Arc<AtomicBool>,
    ) -> Result<ChatResponse>
    where
        F: FnMut(&str),
    {
        self.client_for(role).chat_stream(request, on_token, cancelled).await
    }

    fn key_for(&self, role: ModelRole) -> &str {
        match role {
            ModelRole::Default => &self.routing_default,
            ModelRole::Plan => &self.routing_plan,
            ModelRole::Fast => &self.routing_fast,
        }
    }
}
