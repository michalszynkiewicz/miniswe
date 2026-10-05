//! Pure parsing of a context-window hint out of a startup probe response.
//! No I/O here — see `LlmClient::probe` for where these are actually
//! called against a live server.

use serde_json::Value;

use super::Provider;

/// llama.cpp's `GET /props`: `default_generation_settings.n_ctx` is the
/// per-slot context the running server will actually serve. Deliberately
/// does NOT read `n_ctx_train` (present on some builds) — that is the
/// GGUF's training context, not what this server instance is configured
/// to serve.
pub fn context_window_from_props(body: &Value) -> Option<usize> {
    body.get("default_generation_settings")?
        .get("n_ctx")?
        .as_u64()
        .map(|n| n as usize)
}

/// From a `/v1/models`-style listing body, the served context window for
/// the entry whose `id` matches `model`.
///
/// - vLLM / openai-compatible: `max_model_len`.
/// - OpenRouter: `context_length`.
/// - OpenAI, Anthropic, Ollama, llama.cpp: `None` — their listings carry
///   no served window (llama.cpp's `meta.n_ctx_train` is the model's
///   TRAINING context, not the running server's configured window, and
///   must never be used here — `context_window_from_props` is the real
///   source for llama.cpp).
///
/// Falls back to the first entry in the list, for local providers only
/// (`Provider::is_hosted() == false`), when none matches `model` by id —
/// mirroring how [`super::LlmClient::probe`]'s model-identity probe picks
/// "whichever id comes first" for a local server that only ever serves
/// one model. A hosted catalogue never falls back: an unmatched model
/// means the configured alias isn't being served, and guessing another
/// entry's window would be actively misleading.
pub fn context_window_from_models(provider: Provider, body: &Value, model: &str) -> Option<usize> {
    let field = match provider {
        Provider::Vllm | Provider::OpenAiCompatible => "max_model_len",
        Provider::OpenRouter => "context_length",
        Provider::LlamaCpp | Provider::Ollama | Provider::OpenAi | Provider::Anthropic => {
            return None;
        }
    };

    let list = body.get("data").and_then(Value::as_array)?;
    let entry = list
        .iter()
        .find(|m| m.get("id").and_then(Value::as_str) == Some(model))
        .or_else(|| (!provider.is_hosted()).then(|| list.first()).flatten())?;

    entry.get(field)?.as_u64().map(|n| n as usize)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn props_reads_n_ctx_from_default_generation_settings() {
        let body = json!({
            "default_generation_settings": {"n_ctx": 60000},
            "total_slots": 1,
        });
        assert_eq!(context_window_from_props(&body), Some(60000));
    }

    #[test]
    fn props_missing_field_is_none() {
        assert_eq!(context_window_from_props(&json!({})), None);
        assert_eq!(
            context_window_from_props(&json!({"default_generation_settings": {}})),
            None
        );
    }

    #[test]
    fn vllm_reads_max_model_len_for_the_configured_model() {
        let body = json!({
            "data": [
                {"id": "other-model", "max_model_len": 8192},
                {"id": "test-model", "max_model_len": 32768},
            ]
        });
        assert_eq!(
            context_window_from_models(Provider::Vllm, &body, "test-model"),
            Some(32768)
        );
    }

    #[test]
    fn vllm_falls_back_to_first_entry_when_unmatched() {
        let body = json!({"data": [{"id": "whatever", "max_model_len": 4096}]});
        assert_eq!(
            context_window_from_models(Provider::Vllm, &body, "not-listed"),
            Some(4096)
        );
    }

    #[test]
    fn openrouter_matches_configured_model_not_the_first_entry() {
        let body = json!({
            "data": [
                {"id": "openai/gpt-oss-120b", "context_length": 131072},
                {"id": "test-model", "context_length": 200000},
            ]
        });
        assert_eq!(
            context_window_from_models(Provider::OpenRouter, &body, "test-model"),
            Some(200000)
        );
    }

    #[test]
    fn openrouter_unmatched_model_does_not_fall_back() {
        let body = json!({"data": [{"id": "other", "context_length": 200000}]});
        assert_eq!(
            context_window_from_models(Provider::OpenRouter, &body, "not-listed"),
            None
        );
    }

    #[test]
    fn providers_with_no_served_window_are_always_none() {
        let body = json!({"data": [{"id": "test-model", "context_length": 200000, "max_model_len": 32768}]});
        for provider in [
            Provider::LlamaCpp,
            Provider::Ollama,
            Provider::OpenAi,
            Provider::Anthropic,
        ] {
            assert_eq!(
                context_window_from_models(provider, &body, "test-model"),
                None
            );
        }
    }
}
