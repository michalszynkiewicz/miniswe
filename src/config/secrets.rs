//! Secret inheritance across the global/project config layers.
//!
//! `Config::load()` replaces the global config wholesale with the project
//! config when `.miniswe/config.toml` is present (it is not a deep merge).
//! That means a secret set only in the global file — an API key the user
//! deliberately keeps out of a project directory that may not gitignore
//! it — would otherwise vanish the moment a project overrides anything
//! else. This module is the one place that reinherits those fields.

use super::{Config, ModelConfig};

/// Fold secrets from `global` into `project` wherever `project` left them
/// unset. Called after a project config has fully overridden the global
/// one, so every other field keeps the project's value untouched.
pub fn inherit_secrets(global: &Config, project: &mut Config) {
    if project.web.search_api_key.is_none() {
        project.web.search_api_key = global.web.search_api_key.clone();
    }
    if project.web.searxng_url.is_none() {
        project.web.searxng_url = global.web.searxng_url.clone();
    }

    inherit_model_secrets(&mut project.model, &global.model);

    // Per-slot inheritance only applies between slots of the same name —
    // a key configured for `[models.fast]` globally has no business
    // leaking into a project's differently-purposed `[models.fast]` or
    // any other slot, so an unmatched project slot is left exactly as the
    // project wrote it.
    if let (Some(project_models), Some(global_models)) =
        (project.models.as_mut(), global.models.as_ref())
    {
        for (slot, cfg) in project_models.iter_mut() {
            if let Some(global_cfg) = global_models.get(slot) {
                inherit_model_secrets(cfg, global_cfg);
            }
        }
    }
}

fn inherit_model_secrets(project: &mut ModelConfig, global: &ModelConfig) {
    if project.api_key.is_none() {
        project.api_key = global.api_key.clone();
    }
    if project.api_key_env.is_none() {
        project.api_key_env = global.api_key_env.clone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_with_key(key: Option<&str>) -> Config {
        Config {
            model: ModelConfig {
                api_key: key.map(String::from),
                ..ModelConfig::default()
            },
            ..Config::default()
        }
    }

    #[test]
    fn project_without_api_key_inherits_global_top_level_key() {
        let global = config_with_key(Some("global-secret"));
        let mut project = config_with_key(None);
        inherit_secrets(&global, &mut project);
        assert_eq!(project.model.api_key.as_deref(), Some("global-secret"));
    }

    #[test]
    fn project_with_its_own_api_key_is_not_overwritten() {
        let global = config_with_key(Some("global-secret"));
        let mut project = config_with_key(Some("project-secret"));
        inherit_secrets(&global, &mut project);
        assert_eq!(project.model.api_key.as_deref(), Some("project-secret"));
    }

    #[test]
    fn per_slot_inheritance_only_matches_same_named_slot() {
        let fast_global = ModelConfig {
            api_key: Some("fast-global-key".into()),
            ..ModelConfig::default()
        };
        let mut global_models = std::collections::HashMap::new();
        global_models.insert("fast".to_string(), fast_global);
        let global = Config {
            models: Some(global_models),
            ..Config::default()
        };

        let mut project_models = std::collections::HashMap::new();
        project_models.insert("fast".to_string(), ModelConfig::default());
        project_models.insert("plan".to_string(), ModelConfig::default());
        let mut project = Config {
            models: Some(project_models),
            ..Config::default()
        };

        inherit_secrets(&global, &mut project);

        let models = project.models.as_ref().unwrap();
        assert_eq!(
            models.get("fast").unwrap().api_key.as_deref(),
            Some("fast-global-key")
        );
        // "plan" has no counterpart in the global models map, so it stays unset.
        assert_eq!(models.get("plan").unwrap().api_key, None);
    }

    #[test]
    fn web_secrets_still_inherit() {
        let global = Config {
            web: crate::config::WebConfig {
                search_api_key: Some("serper-key".into()),
                ..Default::default()
            },
            ..Config::default()
        };
        let mut project = Config::default();
        inherit_secrets(&global, &mut project);
        assert_eq!(project.web.search_api_key.as_deref(), Some("serper-key"));
    }
}
