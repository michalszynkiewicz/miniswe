//! API key resolution and request authentication.

use reqwest::RequestBuilder;

use super::Provider;

/// Resolve the API key to use, in priority order: an inline `api_key` in
/// config, then the environment variable named by `api_key_env`, then the
/// provider's conventional default environment variable. `None` if none of
/// those produced a non-empty value.
pub fn resolve_api_key(
    provider: Provider,
    api_key: Option<&str>,
    api_key_env: Option<&str>,
) -> Option<String> {
    if let Some(key) = api_key
        && !key.is_empty()
    {
        return Some(key.to_string());
    }
    let env_var = api_key_env.or_else(|| provider.default_api_key_env())?;
    std::env::var(env_var).ok().filter(|v| !v.is_empty())
}

/// Human-readable description of where the resolved key (if any) came
/// from, for display in `miniswe config` / `miniswe info`. Never returns
/// the key value itself.
pub fn api_key_source(
    provider: Provider,
    api_key: Option<&str>,
    api_key_env: Option<&str>,
) -> String {
    if let Some(key) = api_key
        && !key.is_empty()
    {
        return "inline (model.api_key)".to_string();
    }
    if let Some(env_var) = api_key_env {
        if std::env::var(env_var).is_ok_and(|v| !v.is_empty()) {
            return format!("env {env_var} (model.api_key_env)");
        }
        return format!("unset (model.api_key_env={env_var})");
    }
    if let Some(env_var) = provider.default_api_key_env() {
        if std::env::var(env_var).is_ok_and(|v| !v.is_empty()) {
            return format!("env {env_var} (default)");
        }
        return format!("unset (default env {env_var})");
    }
    "none".to_string()
}

/// Apply the provider's auth headers to an outgoing request. No-op when
/// `api_key` is `None` (a local provider with no key configured).
pub fn apply_auth(
    builder: RequestBuilder,
    provider: Provider,
    api_key: Option<&str>,
) -> RequestBuilder {
    let Some(key) = api_key else {
        return builder;
    };
    match provider {
        Provider::Anthropic => builder
            .bearer_auth(key)
            .header("x-api-key", key)
            .header("anthropic-version", "2023-06-01"),
        _ => builder.bearer_auth(key),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_prefers_inline_key() {
        assert_eq!(
            resolve_api_key(Provider::OpenAi, Some("inline"), None),
            Some("inline".to_string())
        );
    }

    #[test]
    fn resolve_falls_back_to_named_env_var() {
        // SAFETY: test-only env mutation, no concurrent access to this var.
        unsafe {
            std::env::set_var("MINISWE_TEST_AUTH_KEY_1", "from-env");
        }
        assert_eq!(
            resolve_api_key(Provider::OpenAi, None, Some("MINISWE_TEST_AUTH_KEY_1")),
            Some("from-env".to_string())
        );
        unsafe {
            std::env::remove_var("MINISWE_TEST_AUTH_KEY_1");
        }
    }

    #[test]
    fn resolve_empty_inline_falls_through_to_env() {
        unsafe {
            std::env::set_var("MINISWE_TEST_AUTH_KEY_2", "from-env-2");
        }
        assert_eq!(
            resolve_api_key(Provider::OpenAi, Some(""), Some("MINISWE_TEST_AUTH_KEY_2")),
            Some("from-env-2".to_string())
        );
        unsafe {
            std::env::remove_var("MINISWE_TEST_AUTH_KEY_2");
        }
    }

    #[test]
    fn resolve_local_provider_has_no_default_env() {
        assert_eq!(resolve_api_key(Provider::LlamaCpp, None, None), None);
    }

    #[test]
    fn source_never_leaks_key_value() {
        let src = api_key_source(Provider::OpenAi, Some("sk-super-secret"), None);
        assert!(!src.contains("sk-super-secret"));
    }
}
