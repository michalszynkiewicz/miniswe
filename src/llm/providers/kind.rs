//! Parsing and per-provider classification.

use super::Provider;

impl Provider {
    /// Parse a `[model] provider` string. Case-insensitive; `"llamacpp"`
    /// and `"llama.cpp"` are accepted spellings of `"llama-cpp"`. Anything
    /// unrecognized is treated as `OpenAiCompatible` (the OpenAI chat wire
    /// protocol with no provider-specific shaping) with a warning, so a
    /// typo degrades gracefully instead of hard-failing at startup.
    pub fn parse(name: &str) -> Provider {
        match name.to_ascii_lowercase().as_str() {
            "llama-cpp" | "llamacpp" | "llama.cpp" => Provider::LlamaCpp,
            "ollama" => Provider::Ollama,
            "vllm" => Provider::Vllm,
            "openai-compatible" => Provider::OpenAiCompatible,
            "openrouter" => Provider::OpenRouter,
            "openai" => Provider::OpenAi,
            "anthropic" => Provider::Anthropic,
            _ => {
                tracing::warn!("unknown provider {name:?}; treating as openai-compatible");
                Provider::OpenAiCompatible
            }
        }
    }

    /// Canonical config-string name, the inverse of `parse` for the known
    /// variants.
    pub fn name(self) -> &'static str {
        match self {
            Provider::LlamaCpp => "llama-cpp",
            Provider::Ollama => "ollama",
            Provider::Vllm => "vllm",
            Provider::OpenAiCompatible => "openai-compatible",
            Provider::OpenRouter => "openrouter",
            Provider::OpenAi => "openai",
            Provider::Anthropic => "anthropic",
        }
    }

    /// True for a provider reached over the public internet with required
    /// auth, as opposed to a local/self-hosted OpenAI-compatible server.
    pub fn is_hosted(self) -> bool {
        matches!(
            self,
            Provider::OpenRouter | Provider::OpenAi | Provider::Anthropic
        )
    }

    /// Built-in endpoint used when the configured one is empty or still the
    /// local default. `None` for providers that have no sensible default
    /// (local servers vary by port/host).
    pub fn default_endpoint(self) -> Option<&'static str> {
        match self {
            Provider::OpenRouter => Some("https://openrouter.ai/api/v1"),
            Provider::OpenAi => Some("https://api.openai.com/v1"),
            Provider::Anthropic => Some("https://api.anthropic.com/v1"),
            _ => None,
        }
    }

    /// Conventional environment variable a key is read from when neither
    /// `api_key` nor `api_key_env` resolve one.
    pub fn default_api_key_env(self) -> Option<&'static str> {
        match self {
            Provider::OpenRouter => Some("OPENROUTER_API_KEY"),
            Provider::OpenAi => Some("OPENAI_API_KEY"),
            Provider::Anthropic => Some("ANTHROPIC_API_KEY"),
            _ => None,
        }
    }

    /// True if llama.cpp-only request fields (`chat_template_kwargs`,
    /// `cache_prompt`) must never be sent to this provider.
    pub fn strips_llama_fields(self) -> bool {
        self.is_hosted()
    }

    /// True if usage must be requested explicitly via
    /// `stream_options: {include_usage: true}`. OpenRouter (like the local
    /// dialects) already includes usage in its final stream chunk with no
    /// flag needed; only OpenAI and Anthropic require asking for it.
    pub fn wants_stream_usage(self) -> bool {
        matches!(self, Provider::OpenAi | Provider::Anthropic)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_is_case_insensitive_and_accepts_aliases() {
        for (s, want) in [
            ("llama-cpp", Provider::LlamaCpp),
            ("LLAMA-CPP", Provider::LlamaCpp),
            ("llamacpp", Provider::LlamaCpp),
            ("llama.cpp", Provider::LlamaCpp),
            ("ollama", Provider::Ollama),
            ("vllm", Provider::Vllm),
            ("openai-compatible", Provider::OpenAiCompatible),
            ("OpenRouter", Provider::OpenRouter),
            ("OPENAI", Provider::OpenAi),
            ("Anthropic", Provider::Anthropic),
        ] {
            assert_eq!(Provider::parse(s), want, "parsing {s:?}");
        }
    }

    #[test]
    fn parse_unknown_falls_back_to_openai_compatible() {
        assert_eq!(
            Provider::parse("totally-made-up"),
            Provider::OpenAiCompatible
        );
    }

    #[test]
    fn hosted_classification() {
        for p in [Provider::OpenRouter, Provider::OpenAi, Provider::Anthropic] {
            assert!(p.is_hosted());
            assert!(p.strips_llama_fields());
            assert!(p.default_endpoint().is_some());
            assert!(p.default_api_key_env().is_some());
        }
        for p in [
            Provider::LlamaCpp,
            Provider::Ollama,
            Provider::Vllm,
            Provider::OpenAiCompatible,
        ] {
            assert!(!p.is_hosted());
            assert!(!p.strips_llama_fields());
            assert!(!p.wants_stream_usage());
            assert!(p.default_endpoint().is_none());
            assert!(p.default_api_key_env().is_none());
        }
    }

    #[test]
    fn only_openai_and_anthropic_want_explicit_stream_usage() {
        // OpenRouter already includes usage in its final stream chunk with
        // no flag needed, same as the local dialects — only OpenAI and
        // Anthropic require `stream_options.include_usage`.
        assert!(!Provider::OpenRouter.wants_stream_usage());
        assert!(Provider::OpenAi.wants_stream_usage());
        assert!(Provider::Anthropic.wants_stream_usage());
    }
}
