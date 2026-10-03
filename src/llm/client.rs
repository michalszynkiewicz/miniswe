//! The OpenAI-compatible HTTP client: streaming, idle-timeout sizing,
//! retries, and optional request-body dumping.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use futures::StreamExt;
use reqwest::Client;
use serde_json::Value;

use crate::config::ModelConfig;

use super::errors::{
    has_tool_call_leak, is_retryable_llm_error, retry_after_from_message, retry_after_secs,
    with_retry_after_marker,
};
use super::normalize::normalize_xml_tool_calls;
use super::providers::{self, Provider, UsageSnapshot, UsageTotals};
use super::tool_call_repair::{TOOL_CALL_ARGS_CAP_MARKER, tool_call_args_cap};
use super::types::*;

/// Cap on how long a 429's `Retry-After` can push a retry wait to. Honoring
/// the server's requested cooldown is the point, but an unbounded or
/// misbehaving value shouldn't be able to stall the agent indefinitely —
/// the absolute `request_deadline_secs` backstops this further still.
const MAX_RETRY_AFTER_WAIT_SECS: u64 = 60;

/// Idle-window floor (secs) for a forced cold prefill (`cache_prompt=false`).
/// A cold prompt eval reprocesses the whole context and emits no token until
/// prefill completes; on a slow / CPU-offloaded model that first-token latency
/// can exceed the normal stream-idle guard, which would then kill a
/// legitimately-progressing prefill (silence ≠ no progress). The absolute
/// `request_deadline_secs` still backstops a truly wedged server.
pub(super) const COLD_PREFILL_IDLE_SECS: u64 = 60;

/// Worst-case prompt-eval throughput (tokens/sec) assumed when sizing the idle
/// window. A heavily CPU-offloaded MoE (`--n-cpu-moe 40`) was measured at a
/// ~175 tok/s mean over 678 prefill reports, so this leaves ~4x headroom for
/// the slow tail. Deliberately pessimistic: over-estimating the prefill cost
/// only delays wedge detection, which `request_deadline_secs` backstops
/// anyway, whereas under-estimating kills a healthy request outright.
pub(super) const MIN_PREFILL_TOKENS_PER_SEC: u64 = 40;

/// Serialized-JSON bytes per token, for sizing the prefill allowance. The
/// serialized form carries punctuation and escapes the tokenizer never sees,
/// so this over-estimates the token count — again, the safe direction.
pub(super) const PROMPT_BYTES_PER_TOKEN: u64 = 4;

/// Rough token count of the prompt this body will make the server evaluate.
fn estimated_prompt_tokens(body: &Value) -> u64 {
    let bytes = body
        .get("messages")
        .or_else(|| body.get("prompt"))
        .map(|v| v.to_string().len())
        .unwrap_or(0) as u64;
    bytes / PROMPT_BYTES_PER_TOKEN
}

/// The idle timeout for one attempt.
///
/// Prefill emits no SSE token until it completes, so a long *legitimate* prompt
/// eval is byte-for-byte indistinguishable from a wedged server to any purely
/// idle-based guard. Keying the widened window on `cache_prompt=false` was
/// therefore too narrow: a *warm* request whose cached prefix got evicted (one
/// llama.cpp slot shared between 30k main-loop calls and 1-4k fast-role calls)
/// re-prefills thousands of tokens while still advertising `cache_prompt=true`,
/// and got killed by the flat configured window. Observed cost: 215 client-side
/// cancels, every one at exactly the 30s mark, burning ~61% of a 2h56m run.
///
/// So the window is sized from the work the request actually implies —
/// `configured_secs` as the floor, widened to cover the prompt at
/// [`MIN_PREFILL_TOKENS_PER_SEC`], and capped at the absolute deadline, past
/// which `request_deadline_secs` fires regardless and a longer window is a
/// number that can never be reached.
pub(super) fn attempt_idle_timeout(
    body: &Value,
    configured_secs: u64,
    deadline_secs: u64,
) -> Duration {
    let prefill_allowance = estimated_prompt_tokens(body) / MIN_PREFILL_TOKENS_PER_SEC;
    let mut secs = configured_secs.max(prefill_allowance);
    if body.get("cache_prompt") == Some(&Value::Bool(false)) {
        secs = secs.max(COLD_PREFILL_IDLE_SECS);
    }
    Duration::from_secs(secs.min(deadline_secs.max(1)))
}

/// Counter for dumped request bodies. Atomic so multi-threaded callers
/// don't collide on filenames.
static DUMP_COUNTER: AtomicU64 = AtomicU64::new(0);
/// Per-process dump prefix. Without this, multiple agent runs sharing
/// a single dump dir (e.g. successive bench retry attempts mounting the
/// same /output volume) would all start at req-000000 and clobber each
/// other's data — exactly the most diagnostic data when something fails.
/// `seconds-since-epoch + pid` gives chronological sort order across
/// sessions and uniqueness within a host.
static DUMP_SESSION_PREFIX: std::sync::OnceLock<String> = std::sync::OnceLock::new();

fn dump_session_prefix() -> &'static str {
    DUMP_SESSION_PREFIX.get_or_init(|| {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let pid = std::process::id();
        format!("{secs:010}-{pid:05}")
    })
}

/// If `MINISWE_LLM_DUMP_DIR` is set, write the full request body to a
/// numbered JSON file inside that directory. Used to capture exact
/// llama.cpp request bodies for offline replay (the structured logger
/// truncates large bodies, so it can't be used for verbatim replay).
fn maybe_dump_request(body: &Value) {
    let Ok(dir) = std::env::var("MINISWE_LLM_DUMP_DIR") else {
        return;
    };
    let n = DUMP_COUNTER.fetch_add(1, Ordering::SeqCst);
    let prefix = dump_session_prefix();
    let path = std::path::PathBuf::from(&dir).join(format!("req-{prefix}-{n:06}.json"));
    if let Err(e) = std::fs::create_dir_all(&dir) {
        eprintln!("[dump] mkdir {dir:?}: {e}");
        return;
    }
    if let Err(e) = std::fs::write(&path, serde_json::to_vec_pretty(body).unwrap_or_default()) {
        eprintln!("[dump] write {path:?}: {e}");
    }
}

/// Client for communicating with an OpenAI-compatible LLM API.
pub struct LlmClient {
    client: Client,
    config: ModelConfig,
    /// Parsed `config.provider` — the single source of truth for which
    /// dialect this client speaks. Computed once at construction so every
    /// call site doesn't re-parse the string.
    provider: Provider,
    /// Resolved API key (config field, named env var, or the provider's
    /// conventional env var), if any. See `providers::auth::resolve_api_key`.
    api_key: Option<String>,
    /// The endpoint actually used for requests: `config.endpoint` unless a
    /// hosted provider's own default takes over. See
    /// `providers::endpoint::effective_endpoint`.
    endpoint: String,
    /// Running token-usage totals for this client, shared with whoever
    /// holds a clone (cheap — just an `Arc` bump) so usage can be read back
    /// after the client has been handed off to a router.
    usage: Arc<UsageTotals>,
}

impl LlmClient {
    pub fn new(config: ModelConfig) -> Self {
        // Transport-level wall-clock ceiling per request, set to the same
        // deadline the app-level wrapper enforces. Unlike the idle-timeout
        // (which a keep-alive-dribbling wedged server can reset forever),
        // this total-request cap is never reset by incoming bytes. Falls
        // back to an untimed client only if the builder somehow fails.
        let client = Client::builder()
            .timeout(Duration::from_secs(config.request_deadline_secs))
            .build()
            .unwrap_or_else(|_| Client::new());
        let provider = config.provider_kind();
        let api_key = providers::resolve_api_key(
            provider,
            config.api_key.as_deref(),
            config.api_key_env.as_deref(),
        );
        let endpoint = providers::effective_endpoint(provider, &config.endpoint);
        Self {
            client,
            config,
            provider,
            api_key,
            endpoint,
            usage: Arc::new(UsageTotals::new()),
        }
    }

    /// Build the API URL based on provider type.
    fn chat_url(&self) -> String {
        providers::chat_url(self.provider, &self.endpoint)
    }

    /// Ask the server what model it's actually serving, via `/v1/models`
    /// (or `/api/tags` for Ollama). For a local server that returns the
    /// first id it reports — llama-server serves one model and reports
    /// the GGUF path, which is the identity the model-family checks key on.
    /// A hosted provider lists its whole catalogue instead, so there the
    /// probe looks the configured `model` up in that list and returns it
    /// (an early "no such model" check) rather than whichever id happens
    /// to come first. Short timeout so a dead endpoint doesn't stall
    /// startup.
    ///
    /// Error messages are kept short and URL-free — the caller already
    /// displays the endpoint alongside the probe result, so we avoid
    /// repeating it.
    pub async fn probe_model(&self) -> Result<String> {
        let url = providers::models_url(self.provider, &self.endpoint);
        let ollama = matches!(self.provider, Provider::Ollama);
        let request = providers::apply_auth(
            self.client.get(&url),
            self.provider,
            self.api_key.as_deref(),
        );
        // 3s fits a local server; a TLS round trip to a hosted catalogue of
        // a few hundred models timed out at 3s live (OpenAI, 2026-10-02).
        let deadline = if self.provider.is_hosted() {
            Duration::from_secs(10)
        } else {
            Duration::from_secs(3)
        };
        let resp = match tokio::time::timeout(deadline, request.send()).await {
            Err(_) => bail!("timeout"),
            Ok(Err(e)) if e.is_connect() => bail!("unreachable"),
            Ok(Err(e)) => bail!("transport error ({e})"),
            Ok(Ok(r)) => r,
        };
        if !resp.status().is_success() {
            bail!("HTTP {}", resp.status().as_u16());
        }
        let body: Value = resp.json().await.map_err(|_| anyhow::anyhow!("bad JSON"))?;
        let (list, id_key) = if ollama {
            (body["models"].as_array(), "name")
        } else {
            (body["data"].as_array(), "id")
        };
        let ids: Vec<&str> = list
            .into_iter()
            .flatten()
            .filter_map(|m| m[id_key].as_str())
            .collect();
        if self.provider.is_hosted() {
            return if ids.contains(&self.config.model.as_str()) {
                Ok(self.config.model.clone())
            } else {
                bail!("model {:?} not listed", self.config.model)
            };
        }
        ids.first()
            .map(|s| s.to_string())
            .ok_or_else(|| anyhow::anyhow!("no models listed"))
    }

    /// The endpoint actually used for requests (after hosted-provider
    /// default substitution) — see `providers::effective_endpoint`.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Which wire dialect this client speaks.
    pub fn provider(&self) -> Provider {
        self.provider
    }

    /// True if a non-empty API key was resolved at construction time.
    pub fn has_api_key(&self) -> bool {
        self.api_key.is_some()
    }

    /// Fail fast if this client's provider is hosted and no API key could
    /// be resolved, naming the env var the user can set — called right
    /// after startup so a missing key surfaces immediately instead of as a
    /// 401 deep inside a round.
    pub fn check_credentials(&self) -> Result<()> {
        if self.provider.is_hosted() && !self.has_api_key() {
            let hint = self
                .config
                .api_key_env
                .as_deref()
                .or_else(|| self.provider.default_api_key_env())
                .unwrap_or("its conventional");
            bail!(
                "No API key configured for provider '{}' (model {:?} @ {}). Set \
                 `api_key`, `api_key_env`, or the {hint} environment variable in \
                 [model] (or the relevant [models.<slot>]).",
                self.provider.name(),
                self.config.model,
                self.endpoint,
            );
        }
        Ok(())
    }

    /// Snapshot of this client's cumulative token usage across every call
    /// made so far.
    pub fn usage_snapshot(&self) -> UsageSnapshot {
        self.usage.snapshot()
    }

    /// Record one response's usage, if the server reported any. Must only
    /// be called on a response actually handed back to the caller — never
    /// on one discarded by the tool-call-leak retry — or usage would be
    /// double-counted.
    fn record_usage(&self, resp: &ChatResponse) {
        if let Some(u) = &resp.usage {
            self.usage.record(
                u.prompt_tokens as u64,
                u.completion_tokens as u64,
                u.cached_tokens() as u64,
            );
        }
    }

    /// Send a chat completion request and return the full response.
    pub async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        self.chat_with_cancel(request, None).await
    }

    /// Send a chat request with optional cancellation. Internally streams
    /// the response so we can apply an idle-timeout (kill the request if
    /// no tokens have arrived for `stream_idle_timeout_secs`) and retry
    /// the whole call as a transient failure. The caller still receives
    /// a single non-streaming `ChatResponse` — they don't see tokens
    /// piecewise.
    pub async fn chat_with_cancel(
        &self,
        request: &ChatRequest,
        cancelled: Option<&AtomicBool>,
    ) -> Result<ChatResponse> {
        let url = self.chat_url();
        let retry_delays = [1u64, 2, 4, 8, 16, 32];
        let max_retries = self.config.max_retries.min(retry_delays.len());

        // We ALWAYS stream now so we can detect idle connections, even
        // though the public API returns a single ChatResponse to the
        // caller. `build_body` is the single choke point for everything
        // provider-specific — see `providers::shape`.
        let mut body = providers::build_body(self.provider, &self.config, request)?;
        maybe_dump_request(&body);
        let connect_timeout = Duration::from_secs(self.config.request_timeout_secs);

        let mut attempt = 0usize;
        let mut cache_busted = false;
        let mut noop = |_: &str| {};
        // Absolute deadline across all retries. The reqwest client caps each
        // attempt in-flight (the load-bearing guard, never reset by keep-alive
        // bytes); this between-attempts check stops a near-deadline failure
        // from being retried into a multiplied stall. Fast-failing transients
        // (well under the deadline) still retry normally.
        let deadline = Duration::from_secs(self.config.request_deadline_secs);
        let deadline_start = std::time::Instant::now();
        loop {
            if deadline_start.elapsed() >= deadline {
                bail!(
                    "LLM request exceeded {}s deadline across retries",
                    self.config.request_deadline_secs
                );
            }
            // Recomputed per attempt: the tool-call-leak retry below may flip
            // cache_prompt to false mid-loop, and a cold prefill needs the
            // wider idle window.
            let idle_timeout = attempt_idle_timeout(
                &body,
                self.config.stream_idle_timeout_secs,
                self.config.request_deadline_secs,
            );
            let result = self
                .stream_once_assembled(
                    &url,
                    &body,
                    connect_timeout,
                    idle_timeout,
                    cancelled,
                    &mut noop,
                )
                .await;

            match result {
                Ok(resp) if !cache_busted && has_tool_call_leak(&resp) => {
                    // Devstral occasionally emits chat-template tokens
                    // ([TOOL_CALLS]/[ARGS]) embedded inside tool-call
                    // arguments. Verbatim replay shows this is not bytes-
                    // deterministic — it depends on KV-cache state from
                    // prior generations on the same llama.cpp slot. Force
                    // a fresh prompt eval and retry once.
                    tracing::warn!("LLM tool-call leak detected; retrying with cache_prompt=false");
                    // Hosted providers never had this field in the body to
                    // begin with (`build_body` strips it) — don't add it
                    // back just for this retry.
                    if !self.provider.strips_llama_fields() {
                        body["cache_prompt"] = Value::Bool(false);
                    }
                    cache_busted = true;
                    continue;
                }
                Ok(resp) => {
                    self.record_usage(&resp);
                    return Ok(resp);
                }
                Err(err) if attempt < max_retries && is_retryable_llm_error(&err) => {
                    let msg = err.to_string();
                    // A 429's Retry-After is authoritative where present —
                    // it reflects the server's actual rate-limit state,
                    // which our fixed backoff ladder can't know. Capped so
                    // a huge/misbehaving value can't stall the agent.
                    let delay = retry_after_from_message(&msg)
                        .map(|secs| secs.min(MAX_RETRY_AFTER_WAIT_SECS))
                        .unwrap_or(retry_delays[attempt]);
                    attempt += 1;
                    // This branch used to be silent — a connect-phase wedge
                    // (see request_timeout_secs' doc comment) was only
                    // detectable after the fact by diffing llm_dumps
                    // timestamps. Surface it live instead.
                    tracing::warn!(
                        "LLM request failed (retryable), attempt {attempt}/{max_retries}, \
                         retrying in {delay}s: {err}"
                    );
                    match cancelled {
                        Some(flag) => {
                            tokio::select! {
                                _ = tokio::time::sleep(Duration::from_secs(delay)) => {}
                                _ = wait_for_cancel(flag) => bail!("Interrupted by user"),
                            }
                        }
                        None => tokio::time::sleep(Duration::from_secs(delay)).await,
                    }
                }
                Err(err) => return Err(err),
            }
        }
    }

    /// One streamed attempt: connect, drain SSE chunks (each chunk read
    /// must arrive within `idle_timeout` or we bail), assemble into a
    /// `ChatResponse`. `on_token` fires once per content delta — pass a
    /// no-op closure if the caller is not surfacing intermediate tokens
    /// to the UI. Used by both `chat_with_cancel` (no-op) and
    /// `chat_stream` (live UI callback).
    async fn stream_once_assembled<F: FnMut(&str)>(
        &self,
        url: &str,
        body: &Value,
        connect_timeout: Duration,
        idle_timeout: Duration,
        cancelled: Option<&AtomicBool>,
        on_token: &mut F,
    ) -> Result<ChatResponse> {
        // The initial connect/HTTP-handshake gets the wall-clock timeout —
        // we don't want to dial forever if the server is unreachable.
        let connect_future = async {
            let request = providers::apply_auth(
                self.client.post(url),
                self.provider,
                self.api_key.as_deref(),
            )
            .json(body);
            request
                .send()
                .await
                .with_context(|| format!("Failed to connect to LLM at {url}"))
        };

        let response = match cancelled {
            Some(flag) => {
                tokio::select! {
                    result = connect_future => result?,
                    _ = tokio::time::sleep(connect_timeout) => {
                        bail!("LLM request timed out after {}s", self.config.request_timeout_secs);
                    }
                    _ = wait_for_cancel(flag) => bail!("Interrupted by user"),
                }
            }
            None => tokio::time::timeout(connect_timeout, connect_future)
                .await
                .map_err(|_| {
                    anyhow::anyhow!(
                        "LLM request timed out after {}s",
                        self.config.request_timeout_secs
                    )
                })??,
        };

        if !response.status().is_success() {
            let status = response.status();
            // A 429's Retry-After header is the server telling us exactly
            // how long to back off — captured here (before the body is
            // consumed) and carried through the error message since the
            // retry loop only sees the stringified error.
            let retry_after = (status == reqwest::StatusCode::TOO_MANY_REQUESTS)
                .then(|| retry_after_secs(response.headers()))
                .flatten();
            let text = response.text().await.unwrap_or_default();
            bail!(
                "{}",
                with_retry_after_marker(format!("LLM API error ({status}): {text}"), retry_after)
            );
        }

        // Servers that honor `stream: true` return text/event-stream;
        // some servers (and our test mocks) ignore the flag and return
        // a single application/json body. We dispatch on Content-Type
        // and handle both — the idle-timeout still applies to whichever
        // chunk reader we end up using.
        let is_sse = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|ct| ct.contains("event-stream"))
            .unwrap_or(false);

        if !is_sse {
            return self
                .read_non_streaming_body(response, idle_timeout, cancelled, on_token)
                .await;
        }

        let mut stream = response.bytes_stream();
        let mut full_content = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut current_tool_call_parts: std::collections::HashMap<
            usize,
            (String, String, String),
        > = std::collections::HashMap::new();
        let mut sse_buf = String::new();
        // Warn at most once per stream if the server omits tool_call index;
        // a broken server would otherwise spam a line per delta.
        let mut warned_missing_index = false;
        // Real finish_reason/usage from the stream (the final chunks carry
        // them). finish_reason matters downstream: "length" is the only
        // signal that a generation was cut off by the context ceiling
        // rather than finishing on its own — see
        // `is_context_truncated_response`.
        let mut finish_reason: Option<String> = None;
        let mut usage: Option<Usage> = None;

        loop {
            // Wrap each chunk read in an idle-timeout. If the model has
            // not produced any tokens (or even keep-alives) within the
            // idle window, treat the connection as stuck and bail with
            // a retryable error.
            let next_chunk = async {
                if let Some(flag) = cancelled {
                    tokio::select! {
                        chunk = stream.next() => Ok(chunk),
                        _ = wait_for_cancel(flag) => Err(anyhow::anyhow!("Interrupted by user")),
                    }
                } else {
                    Ok(stream.next().await)
                }
            };

            let chunk_opt = match tokio::time::timeout(idle_timeout, next_chunk).await {
                Ok(Ok(chunk_opt)) => chunk_opt,
                Ok(Err(e)) => return Err(e),
                Err(_) => bail!(
                    "LLM stream idle: no tokens received for {}s",
                    idle_timeout.as_secs()
                ),
            };

            let Some(chunk) = chunk_opt else {
                break; // stream ended
            };
            let chunk = chunk.context("Stream read error")?;
            sse_buf.push_str(&String::from_utf8_lossy(&chunk));

            // Drain complete SSE events from the buffer (each event
            // ends with a `\n\n`). Hold any partial trailing event for
            // the next iteration so we don't truncate JSON mid-chunk.
            while let Some(idx) = sse_buf.find("\n\n") {
                let event = sse_buf[..idx].to_string();
                sse_buf.drain(..idx + 2);

                let mut done = false;
                for line in event.lines() {
                    let line = line.trim();
                    if line == "data: [DONE]" {
                        done = true;
                        break;
                    }
                    let Some(data) = line.strip_prefix("data: ") else {
                        continue;
                    };
                    let Ok(parsed) = serde_json::from_str::<StreamChunk>(data) else {
                        continue;
                    };
                    if let Some(u) = parsed.usage {
                        usage = Some(u);
                    }
                    if let Some(choice) = parsed.choices.first() {
                        if let Some(fr) = &choice.finish_reason {
                            finish_reason = Some(fr.clone());
                        }
                        if let Some(content) = &choice.delta.content {
                            on_token(content);
                            full_content.push_str(content);
                        }
                        if let Some(tc_deltas) = &choice.delta.tool_calls {
                            for tc_delta in tc_deltas {
                                // Per OpenAI spec every tool-call delta carries `index`.
                                // Guessing 0 would silently corrupt parallel calls by
                                // merging stray deltas into call #0. Skip instead.
                                let Some(idx) = tc_delta.index else {
                                    if !warned_missing_index {
                                        tracing::warn!(
                                            "LLM stream: tool_call delta missing `index`; skipping. \
                                             The server emitted a non-spec-compliant SSE chunk — \
                                             if you see this often, the upstream tool call may be incomplete."
                                        );
                                        warned_missing_index = true;
                                    }
                                    continue;
                                };
                                let entry =
                                    current_tool_call_parts.entry(idx).or_insert_with(|| {
                                        (
                                            tc_delta.id.clone().unwrap_or_default(),
                                            String::new(),
                                            String::new(),
                                        )
                                    });
                                if let Some(id) = &tc_delta.id
                                    && !id.is_empty()
                                {
                                    entry.0 = id.clone();
                                }
                                if let Some(func) = &tc_delta.function {
                                    if let Some(name) = &func.name {
                                        entry.1.push_str(name);
                                    }
                                    if let Some(args) = &func.arguments {
                                        entry.2.push_str(args);
                                        // Anchor-only tools never need more
                                        // than a few hundred chars; a call
                                        // growing past the cap is the model
                                        // pasting code into an anchor field.
                                        // Abort now (the server cancels the
                                        // slot on disconnect) instead of
                                        // burning minutes until the context
                                        // ceiling truncates it anyway.
                                        if let Some(cap) = tool_call_args_cap(&entry.1)
                                            && entry.2.len() > cap
                                        {
                                            bail!(
                                                "{TOOL_CALL_ARGS_CAP_MARKER}: `{}` arguments \
                                                 reached {} chars (cap {cap}) — generation aborted",
                                                entry.1,
                                                entry.2.len()
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if done {
                    break;
                }
            }
        }

        // Assemble tool calls from accumulated parts
        let mut indices: Vec<usize> = current_tool_call_parts.keys().copied().collect();
        indices.sort();
        for idx in indices {
            let Some((id, name, arguments)) = current_tool_call_parts.remove(&idx) else {
                continue;
            };
            tool_calls.push(ToolCall {
                id,
                r#type: "function".into(),
                function: FunctionCall { name, arguments },
            });
        }

        let mut resp = ChatResponse {
            choices: vec![Choice {
                message: Message {
                    role: "assistant".into(),
                    content: if full_content.is_empty() {
                        None
                    } else {
                        Some(full_content)
                    },
                    tool_calls: if tool_calls.is_empty() {
                        None
                    } else {
                        Some(tool_calls)
                    },
                    tool_call_id: None,
                    name: None,
                },
                // Real finish_reason from the stream when the server sent
                // one; "stop" preserves the old behavior for servers/mocks
                // that never emit it.
                finish_reason: Some(finish_reason.unwrap_or_else(|| "stop".into())),
            }],
            usage,
        };
        normalize_xml_tool_calls(&mut resp, self.config.tool_call_format);
        Ok(resp)
    }

    /// Drain a non-streamed JSON response body chunk-by-chunk so the
    /// idle-timeout still applies (we don't want a hanging body to wedge
    /// the request indefinitely just because the server returned
    /// `application/json` instead of `text/event-stream`). After the
    /// body finishes we parse it as a single `ChatResponse` and forward
    /// any assistant text to `on_token` so streaming-style callers still
    /// get one final UI update.
    async fn read_non_streaming_body<F: FnMut(&str)>(
        &self,
        response: reqwest::Response,
        idle_timeout: Duration,
        cancelled: Option<&AtomicBool>,
        on_token: &mut F,
    ) -> Result<ChatResponse> {
        let mut stream = response.bytes_stream();
        let mut buf: Vec<u8> = Vec::new();

        loop {
            let next_chunk = async {
                if let Some(flag) = cancelled {
                    tokio::select! {
                        chunk = stream.next() => Ok(chunk),
                        _ = wait_for_cancel(flag) => Err(anyhow::anyhow!("Interrupted by user")),
                    }
                } else {
                    Ok(stream.next().await)
                }
            };

            let chunk_opt = match tokio::time::timeout(idle_timeout, next_chunk).await {
                Ok(Ok(chunk_opt)) => chunk_opt,
                Ok(Err(e)) => return Err(e),
                Err(_) => bail!(
                    "LLM stream idle: no tokens received for {}s",
                    idle_timeout.as_secs()
                ),
            };

            let Some(chunk) = chunk_opt else {
                break;
            };
            let chunk = chunk.context("Stream read error")?;
            buf.extend_from_slice(&chunk);
        }

        let mut resp: ChatResponse =
            serde_json::from_slice(&buf).context("Failed to parse LLM response")?;
        normalize_xml_tool_calls(&mut resp, self.config.tool_call_format);
        if let Some(content) = resp
            .choices
            .first()
            .and_then(|c| c.message.content.as_deref())
            .filter(|c| !c.is_empty())
        {
            on_token(content);
        }
        Ok(resp)
    }

    /// Send a streaming chat request. Calls `on_token` for each content
    /// delta and returns the final assembled response. The `cancelled`
    /// flag can be set from another task (e.g., Ctrl+C handler) to abort
    /// mid-stream.
    ///
    /// Internally shares the SSE / non-SSE / idle-timeout machinery with
    /// `chat_with_cancel` via [`Self::stream_once_assembled`]. Connect
    /// failures and idle-timeout errors are retried up to `max_retries`,
    /// but only if no tokens have been delivered yet on the current
    /// attempt — once the UI has seen partial content, retrying would
    /// duplicate it, so we surface the error instead.
    pub async fn chat_stream<F>(
        &self,
        request: &ChatRequest,
        mut on_token: F,
        cancelled: &Arc<AtomicBool>,
    ) -> Result<ChatResponse>
    where
        F: FnMut(&str),
    {
        let url = self.chat_url();
        let retry_delays = [1u64, 2, 4, 8, 16, 32];
        let max_retries = self.config.max_retries.min(retry_delays.len());

        let body = providers::build_body(self.provider, &self.config, request)?;
        maybe_dump_request(&body);
        let connect_timeout = Duration::from_secs(self.config.request_timeout_secs);

        let mut attempt = 0usize;
        // Absolute deadline across retries — see chat_with_cancel. The
        // reqwest client caps each attempt in-flight; this stops a
        // near-deadline failure from being retried into a multiplied stall.
        let deadline = Duration::from_secs(self.config.request_deadline_secs);
        let deadline_start = std::time::Instant::now();
        loop {
            if deadline_start.elapsed() >= deadline {
                bail!(
                    "LLM request exceeded {}s deadline across retries",
                    self.config.request_deadline_secs
                );
            }
            let mut had_progress = false;
            let idle_timeout = attempt_idle_timeout(
                &body,
                self.config.stream_idle_timeout_secs,
                self.config.request_deadline_secs,
            );
            let result = {
                let mut wrapped = |token: &str| {
                    had_progress = true;
                    on_token(token);
                };
                self.stream_once_assembled(
                    &url,
                    &body,
                    connect_timeout,
                    idle_timeout,
                    Some(cancelled.as_ref()),
                    &mut wrapped,
                )
                .await
            };

            match result {
                Ok(resp) => {
                    self.record_usage(&resp);
                    return Ok(resp);
                }
                Err(err)
                    if !had_progress && attempt < max_retries && is_retryable_llm_error(&err) =>
                {
                    let msg = err.to_string();
                    let delay = retry_after_from_message(&msg)
                        .map(|secs| secs.min(MAX_RETRY_AFTER_WAIT_SECS))
                        .unwrap_or(retry_delays[attempt]);
                    attempt += 1;
                    tokio::select! {
                        _ = tokio::time::sleep(Duration::from_secs(delay)) => {}
                        _ = wait_for_cancel(cancelled) => bail!("Interrupted by user"),
                    }
                }
                Err(err) => return Err(err),
            }
        }
    }
}

async fn wait_for_cancel(cancelled: &AtomicBool) {
    while !cancelled.load(Ordering::Relaxed) {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}
