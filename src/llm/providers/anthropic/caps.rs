//! Model capability lookup: one `GET /v1/models/{id}` per client.

use std::time::Duration;

use reqwest::Client;
use serde_json::Value;

use super::ModelCaps;
use crate::llm::providers::{Provider, apply_auth};

/// Every current model is adaptive-capable, so that is the assumption when
/// the lookup can't say otherwise.
impl Default for ModelCaps {
    fn default() -> Self {
        Self { adaptive: true }
    }
}

/// `capabilities.thinking.types.adaptive.supported` from a model object,
/// `None` when the field is missing or not a bool.
pub(super) fn adaptive_from_model_json(body: &Value) -> Option<bool> {
    body.pointer("/capabilities/thinking/types/adaptive/supported")?
        .as_bool()
}

fn model_url(endpoint: &str, model: &str) -> String {
    let base = endpoint.trim_end_matches('/');
    if base.ends_with("/v1") {
        format!("{base}/models/{model}")
    } else {
        format!("{base}/v1/models/{model}")
    }
}

/// Look the model up. Never fails: any problem (network, non-200, missing
/// field) assumes adaptive and is logged at `debug`.
pub async fn lookup_caps(
    client: &Client,
    endpoint: &str,
    model: &str,
    api_key: Option<&str>,
) -> ModelCaps {
    let url = model_url(endpoint, model);
    let request = apply_auth(client.get(&url), Provider::Anthropic, api_key);
    let resp = match tokio::time::timeout(Duration::from_secs(10), request.send()).await {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            tracing::debug!("model capability lookup: GET {url} failed: {e}");
            return ModelCaps::default();
        }
        Err(_) => {
            tracing::debug!("model capability lookup: GET {url} timed out");
            return ModelCaps::default();
        }
    };
    if !resp.status().is_success() {
        tracing::debug!(
            "model capability lookup: GET {url} returned HTTP {}",
            resp.status().as_u16()
        );
        return ModelCaps::default();
    }
    let body: Value = match resp.json().await {
        Ok(b) => b,
        Err(e) => {
            tracing::debug!("model capability lookup: GET {url} returned bad JSON: {e}");
            return ModelCaps::default();
        }
    };
    match adaptive_from_model_json(&body) {
        Some(adaptive) => ModelCaps { adaptive },
        None => {
            tracing::debug!("model capability lookup: {url} response had no adaptive field");
            ModelCaps::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn adaptive_flag_is_read_from_capabilities() {
        let yes = json!({"capabilities": {"thinking": {"supported": true,
            "types": {"enabled": {"supported": false}, "adaptive": {"supported": true}}}}});
        assert_eq!(adaptive_from_model_json(&yes), Some(true));
        let no = json!({"capabilities": {"thinking": {"supported": true,
            "types": {"enabled": {"supported": true}, "adaptive": {"supported": false}}}}});
        assert_eq!(adaptive_from_model_json(&no), Some(false));
    }

    #[test]
    fn missing_or_malformed_capability_is_none() {
        assert_eq!(adaptive_from_model_json(&json!({})), None);
        assert_eq!(
            adaptive_from_model_json(&json!({"capabilities": {"thinking": {"types": {}}}})),
            None
        );
        assert_eq!(
            adaptive_from_model_json(
                &json!({"capabilities": {"thinking": {"types": {"adaptive": {"supported": "yes"}}}}})
            ),
            None
        );
    }

    #[test]
    fn model_url_follows_the_v1_rule() {
        assert_eq!(
            model_url("https://api.anthropic.com/v1/", "claude-x"),
            "https://api.anthropic.com/v1/models/claude-x"
        );
        assert_eq!(model_url("http://h", "m"), "http://h/v1/models/m");
    }

    #[test]
    fn default_assumes_adaptive() {
        assert!(ModelCaps::default().adaptive);
    }
}
