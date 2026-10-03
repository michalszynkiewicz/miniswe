//! The single body-building choke point. Every dialect speaks the OpenAI
//! chat-completions wire protocol at the top level; this is where the
//! per-provider differences from `docs/hosted-providers.md`'s decision
//! table are applied to one `serde_json::Value` body.

use serde_json::{Value, json};

use crate::config::ModelConfig;

use super::Provider;
use crate::llm::types::ChatRequest;

/// What the caller asked for in terms of reasoning, read off the
/// `chat_template_kwargs` every call site threads through today. Two
/// spellings exist: `{"enable_thinking": bool}` (set from
/// `ModelConfig::thinking` on the main loop / debugger, hardcoded `false`
/// on mechanical sub-roles such as the summarizer or skill router) and
/// Mistral Small 4's `{"reasoning_effort": "none" | "high"}` (main loop
/// only, keyed on the probed model). Hosted dialects never see the kwarg
/// itself, so this is the one signal they translate into their own shape.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Thinking {
    Off,
    /// On, with an explicit effort when the caller named one
    /// (`reasoning_effort`); `None` means "use `ModelConfig::thinking_effort`".
    On {
        effort: Option<String>,
    },
}

fn thinking_requested(request: &ChatRequest) -> Thinking {
    let Some(kwargs) = request.chat_template_kwargs.as_ref() else {
        return Thinking::Off;
    };
    if let Some(effort) = kwargs.get("reasoning_effort").and_then(|v| v.as_str()) {
        return if effort.eq_ignore_ascii_case("none") {
            Thinking::Off
        } else {
            Thinking::On {
                effort: Some(effort.to_string()),
            }
        };
    }
    match kwargs.get("enable_thinking").and_then(|v| v.as_bool()) {
        Some(true) => Thinking::On { effort: None },
        _ => Thinking::Off,
    }
}

/// Extra room left above `thinking.budget_tokens` for the visible answer
/// when Anthropic extended thinking is on: the API requires
/// `max_tokens > budget_tokens`, and a bare `+1` would satisfy it while
/// leaving no room for the reply.
const ANTHROPIC_ANSWER_HEADROOM: u64 = 1024;

/// Build the full JSON request body for `provider`. This replaces the
/// duplicated body-building code that used to live directly in
/// `LlmClient::chat_with_cancel` / `chat_stream` — both now call this
/// instead, so the two code paths can't drift from each other.
///
/// IMPORTANT: for `Provider::LlamaCpp` (and the other local dialects —
/// ollama/vllm/openai-compatible) this must keep producing exactly the
/// fields the wire body had before this module existed: `model`,
/// `temperature`, `max_tokens`, `stream`, `chat_template_kwargs` /
/// `cache_prompt` when the request carries them, plus whatever
/// `ChatRequest` itself serializes to (`messages`, `tools`,
/// `tool_choice`). This is benchmarked code — see the wiremock test
/// `llama_cpp_body_is_unchanged` for the exact key-set assertion.
pub fn build_body(
    provider: Provider,
    config: &ModelConfig,
    request: &ChatRequest,
) -> anyhow::Result<Value> {
    let mut body = serde_json::to_value(request)?;

    body["model"] = Value::String(config.model.clone());
    body["stream"] = Value::Bool(true);

    let thinking = thinking_requested(request);
    let thinking_on = thinking != Thinking::Off;

    // --- temperature: sent for local dialects + OpenRouter, never for OpenAI
    // or Anthropic. OpenAI's reasoning models (gpt-5 family) reject any
    // value other than the default even without reasoning requested —
    // confirmed live 2026-10-02 ("Unsupported value: 'temperature' does
    // not support 0.15 with this model"); Anthropic deprecates it on
    // current models. The 0.15 default exists for small local models.
    let send_temperature = !matches!(provider, Provider::OpenAi | Provider::Anthropic);
    if send_temperature {
        body["temperature"] =
            Value::from(request.temperature_override.unwrap_or(config.temperature));
    }

    // --- output cap: max_tokens everywhere except OpenAI's max_completion_tokens ---
    let max_tokens = request
        .max_tokens_override
        .unwrap_or(config.max_output_tokens as u64);
    match provider {
        Provider::OpenAi => {
            body["max_completion_tokens"] = Value::from(max_tokens);
        }
        Provider::Anthropic if thinking_on => {
            // Anthropic requires max_tokens to exceed thinking.budget_tokens;
            // keep real headroom for the answer, not just the required +1.
            let budget = config.thinking_budget_tokens as u64;
            body["max_tokens"] = Value::from(max_tokens.max(budget + ANTHROPIC_ANSWER_HEADROOM));
        }
        _ => {
            body["max_tokens"] = Value::from(max_tokens);
        }
    }

    // --- llama-only fields: sent for local dialects, stripped for hosted ---
    if provider.strips_llama_fields() {
        if let Some(obj) = body.as_object_mut() {
            obj.remove("chat_template_kwargs");
            obj.remove("cache_prompt");
        }
    } else {
        if let Some(kwargs) = &request.chat_template_kwargs {
            body["chat_template_kwargs"] = kwargs.clone();
        }
        if let Some(cp) = request.cache_prompt {
            body["cache_prompt"] = Value::Bool(cp);
        }
    }

    // --- thinking translation for hosted dialects ---
    if let Thinking::On { effort } = &thinking {
        let effort = effort
            .clone()
            .unwrap_or_else(|| config.thinking_effort.clone());
        match provider {
            Provider::OpenRouter => {
                body["reasoning"] = json!({ "effort": effort });
            }
            Provider::OpenAi => {
                body["reasoning_effort"] = Value::String(effort);
            }
            Provider::Anthropic => {
                body["thinking"] = json!({
                    "type": "enabled",
                    "budget_tokens": config.thinking_budget_tokens,
                });
            }
            _ => {}
        }
    }

    // --- usage: local dialects + OpenRouter get it unconditionally; OpenAI
    // and Anthropic require asking for it explicitly ---
    if provider.wants_stream_usage() {
        body["stream_options"] = json!({ "include_usage": true });
    }

    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::Message;

    fn request() -> ChatRequest {
        ChatRequest {
            messages: vec![Message::user("hi")],
            ..Default::default()
        }
    }

    #[test]
    fn llama_cpp_has_exact_legacy_field_set() {
        let config = ModelConfig::default();
        let body = build_body(Provider::LlamaCpp, &config, &request()).unwrap();
        let obj = body.as_object().unwrap();
        let mut keys: Vec<&str> = obj.keys().map(String::as_str).collect();
        keys.sort_unstable();
        let mut expected = vec!["max_tokens", "messages", "model", "stream", "temperature"];
        expected.sort_unstable();
        assert_eq!(keys, expected);
        assert!(!obj.contains_key("stream_options"));
        assert!(!obj.contains_key("reasoning"));
        assert!(!obj.contains_key("thinking"));
    }

    #[test]
    fn openai_uses_max_completion_tokens_and_never_sends_temperature() {
        let config = ModelConfig {
            provider: "openai".into(),
            ..ModelConfig::default()
        };
        let mut req = request();
        req.chat_template_kwargs = Some(json!({"enable_thinking": true}));
        let body = build_body(Provider::OpenAi, &config, &req).unwrap();
        assert!(body.get("max_completion_tokens").is_some());
        assert!(body.get("max_tokens").is_none());
        assert!(body.get("temperature").is_none());
        assert_eq!(body["reasoning_effort"], Value::String("medium".into()));
        // Also without thinking: gpt-5 family rejects temperature != 1.
        let body = build_body(Provider::OpenAi, &config, &request()).unwrap();
        assert!(body.get("temperature").is_none());
        assert!(body.get("reasoning_effort").is_none());
    }

    #[test]
    fn anthropic_never_sends_temperature_and_bumps_max_tokens_for_thinking() {
        let config = ModelConfig {
            provider: "anthropic".into(),
            max_output_tokens: 100,
            thinking_budget_tokens: 2048,
            ..ModelConfig::default()
        };
        let mut req = request();
        req.chat_template_kwargs = Some(json!({"enable_thinking": true}));
        let body = build_body(Provider::Anthropic, &config, &req).unwrap();
        assert!(body.get("temperature").is_none());
        // budget + 1024 of answer headroom beats the configured 100.
        assert_eq!(body["max_tokens"], Value::from(3072u64));
        assert_eq!(
            body["thinking"],
            json!({"type": "enabled", "budget_tokens": 2048})
        );
    }

    #[test]
    fn openrouter_gets_usage_without_stream_options() {
        let config = ModelConfig {
            provider: "openrouter".into(),
            ..ModelConfig::default()
        };
        let body = build_body(Provider::OpenRouter, &config, &request()).unwrap();
        assert!(body.get("stream_options").is_none());
    }

    #[test]
    fn llama_cpp_forwards_template_kwargs_and_cache_prompt() {
        let config = ModelConfig::default();
        let mut req = request();
        req.chat_template_kwargs = Some(json!({"enable_thinking": false}));
        req.cache_prompt = Some(false);
        let body = build_body(Provider::LlamaCpp, &config, &req).unwrap();
        assert_eq!(
            body["chat_template_kwargs"],
            json!({"enable_thinking": false})
        );
        assert_eq!(body["cache_prompt"], Value::Bool(false));
    }

    #[test]
    fn reasoning_effort_kwarg_maps_to_hosted_thinking() {
        // Mistral Small 4's main-loop spelling: "none" = off, anything
        // else = on with that effort, overriding `thinking_effort`.
        let config = ModelConfig {
            thinking_effort: "medium".into(),
            ..ModelConfig::default()
        };
        let mut req = request();
        req.chat_template_kwargs = Some(json!({"reasoning_effort": "high"}));
        let body = build_body(Provider::OpenRouter, &config, &req).unwrap();
        assert_eq!(body["reasoning"], json!({"effort": "high"}));
        let body = build_body(Provider::OpenAi, &config, &req).unwrap();
        assert_eq!(body["reasoning_effort"], Value::String("high".into()));
        assert!(body.get("temperature").is_none());

        req.chat_template_kwargs = Some(json!({"reasoning_effort": "none"}));
        let body = build_body(Provider::OpenRouter, &config, &req).unwrap();
        assert!(body.get("reasoning").is_none());
        let body = build_body(Provider::Anthropic, &config, &req).unwrap();
        assert!(body.get("thinking").is_none());
    }
}
