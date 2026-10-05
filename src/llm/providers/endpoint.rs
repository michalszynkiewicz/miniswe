//! Endpoint resolution: hosted-provider defaults, and chat/models URL
//! construction per dialect.

use super::Provider;

/// Resolve the endpoint actually used for requests. A hosted provider's
/// built-in default takes over when `configured` is empty or is still the
/// untouched local default (`ModelConfig::default().endpoint`) — the
/// signal that the user picked a provider but never edited `endpoint`.
/// Otherwise the configured value wins (a self-hosted proxy, a custom
/// OpenRouter-compatible gateway, etc). A trailing slash is always
/// trimmed so callers can append paths without doubling `/`.
pub fn effective_endpoint(provider: Provider, configured: &str) -> String {
    let trimmed = configured.trim_end_matches('/');
    match provider.default_endpoint() {
        Some(default) if trimmed.is_empty() || trimmed == crate::config::DEFAULT_ENDPOINT => {
            default.to_string()
        }
        _ => trimmed.to_string(),
    }
}

/// Chat URL for an (already effective) endpoint. Ollama uses its own
/// `/api/chat`, Anthropic its native `/v1/messages`; every other dialect
/// uses the OpenAI `/v1/chat/completions` path — except a base that already
/// ends in `/v1` (every hosted provider's default does), where appending
/// the full `/v1/...` suffix would double it.
pub fn chat_url(provider: Provider, endpoint: &str) -> String {
    let base = endpoint.trim_end_matches('/');
    match provider {
        Provider::Ollama => format!("{base}/api/chat"),
        Provider::Anthropic if base.ends_with("/v1") => format!("{base}/messages"),
        Provider::Anthropic => format!("{base}/v1/messages"),
        _ if base.ends_with("/v1") => format!("{base}/chat/completions"),
        _ => format!("{base}/v1/chat/completions"),
    }
}

/// Models-listing URL, same `/v1`-doubling rule as [`chat_url`]. Ollama
/// uses `/api/tags`. Anthropic paginates the list (20 per page by default,
/// newest first), so a valid older id would fall off the first page and
/// the probe would wrongly report it "not listed"; ask for the maximum.
pub fn models_url(provider: Provider, endpoint: &str) -> String {
    let base = endpoint.trim_end_matches('/');
    let path = match provider {
        Provider::Ollama => return format!("{base}/api/tags"),
        _ if base.ends_with("/v1") => format!("{base}/models"),
        _ => format!("{base}/v1/models"),
    };
    match provider {
        Provider::Anthropic => format!("{path}?limit=1000"),
        _ => path,
    }
}

/// llama.cpp's `/props` endpoint URL for an (already effective) endpoint.
/// Strips a trailing `/` and a trailing `/v1` (every hosted provider's
/// default endpoint, and any local endpoint pointed at an OpenAI-style
/// base, carries one) before appending `/props`.
pub fn props_url(endpoint: &str) -> String {
    let base = endpoint
        .trim_end_matches('/')
        .trim_end_matches("/v1")
        .trim_end_matches('/');
    format!("{base}/props")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_endpoint_rules() {
        assert_eq!(
            effective_endpoint(Provider::OpenRouter, ""),
            "https://openrouter.ai/api/v1"
        );
        assert_eq!(
            effective_endpoint(Provider::OpenRouter, "http://localhost:8464"),
            "https://openrouter.ai/api/v1"
        );
        assert_eq!(
            effective_endpoint(Provider::OpenRouter, "http://proxy:1/v1"),
            "http://proxy:1/v1"
        );
        assert_eq!(
            effective_endpoint(Provider::LlamaCpp, "http://localhost:8464"),
            "http://localhost:8464"
        );
    }

    #[test]
    fn chat_url_rules() {
        assert_eq!(
            chat_url(Provider::OpenRouter, "http://x/v1"),
            "http://x/v1/chat/completions"
        );
        assert_eq!(
            chat_url(Provider::LlamaCpp, "http://localhost:8464"),
            "http://localhost:8464/v1/chat/completions"
        );
        assert_eq!(chat_url(Provider::Ollama, "http://x"), "http://x/api/chat");
        assert_eq!(
            chat_url(Provider::Anthropic, "https://api.anthropic.com/v1"),
            "https://api.anthropic.com/v1/messages"
        );
        assert_eq!(
            chat_url(Provider::Anthropic, "http://x"),
            "http://x/v1/messages"
        );
    }

    #[test]
    fn props_url_rules() {
        assert_eq!(
            props_url("http://localhost:8464/v1"),
            "http://localhost:8464/props"
        );
        assert_eq!(
            props_url("http://localhost:8464"),
            "http://localhost:8464/props"
        );
        assert_eq!(
            props_url("http://localhost:8464/"),
            "http://localhost:8464/props"
        );
    }

    #[test]
    fn models_url_rules() {
        assert_eq!(
            models_url(Provider::OpenAi, "http://x/v1"),
            "http://x/v1/models"
        );
        assert_eq!(
            models_url(Provider::LlamaCpp, "http://localhost:8464"),
            "http://localhost:8464/v1/models"
        );
        assert_eq!(
            models_url(Provider::Ollama, "http://x"),
            "http://x/api/tags"
        );
        assert_eq!(
            models_url(Provider::Anthropic, "https://api.anthropic.com/v1"),
            "https://api.anthropic.com/v1/models?limit=1000"
        );
    }
}
