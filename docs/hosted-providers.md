# Hosted providers: OpenRouter, OpenAI, Anthropic

Status: phase 1 done; phase 2 (native Anthropic client) implemented 2026-10-05, not yet exercised live. Phase 3 is not started.

## Why

Until phase 1 the client had no way to send an API key and always sent two
llama.cpp-only request fields (`chat_template_kwargs`, `cache_prompt`), so the
README's "works with any OpenAI-compatible API" was only true for keyless local
servers. Hosted models also make the fresh-install walkthrough possible on a
machine without a GPU.

## Facts the design rests on

Verified 2026-10-01 against the live docs:

- OpenRouter: `https://openrouter.ai/api/v1/chat/completions`, bearer auth,
  unknown body fields are ignored, usage arrives once in the final stream chunk,
  reasoning is requested with a `reasoning` object (`effort` or `max_tokens`).
- Anthropic OpenAI-compatibility layer: base `https://api.anthropic.com/v1/`,
  `authorization: Bearer` supported, unsupported fields silently ignored,
  `max_tokens` and `stream_options` supported, `temperature` only within 0..1
  and deprecated on newer models, thinking via an extra
  `thinking: {type: "enabled", budget_tokens}` object, `reasoning_effort`
  ignored. Anthropic labels the layer as for evaluation, not production: no
  prompt caching, no thinking output, system messages hoisted.
- Anthropic native Messages API: `POST /v1/messages`, `x-api-key` +
  `anthropic-version: 2023-06-01`, separate `system`, tools with
  `input_schema`, `tool_use` / `tool_result` content blocks, SSE events
  `message_start`, `content_block_start/delta/stop`, `message_delta`.

Confirmed live 2026-10-02 (OpenAI's reference is behind a 403 from the dev
container): `gpt-5-mini` rejects `temperature` with HTTP 400
`unsupported_value` even without reasoning requested, so OpenAI never gets
`temperature`. Still unverified: whether unknown top-level fields produce a
400 (none are sent today) and whether `max_completion_tokens` is required.

Also verified 2026-10-02: Anthropic's `/v1/models` is paginated (20 per
page, newest first); the manual `thinking: {type: enabled, budget_tokens}`
form is deprecated on the 4.6 generation and not accepted on Claude 5
models, which think adaptively by default (the compat layer ignores
`reasoning_effort`). Set `thinking = false` on Claude 5 models until the
phase 2 native client maps effort.

## Design

`provider` stays a string in `[model]` for config compatibility and is parsed
into an enum. Every provider except `anthropic` (native Messages client, phase 2) shares the
OpenAI chat-completions wire protocol; a dialect layer in `src/llm/providers/`
decides per provider:

| decision | llama-cpp / ollama / vllm / openai-compatible | openrouter | openai | anthropic (native `/v1/messages`) |
|---|---|---|---|---|
| default endpoint | none, configured | `https://openrouter.ai/api/v1` | `https://api.openai.com/v1` | `https://api.anthropic.com/v1` |
| auth | bearer if a key is configured | bearer, required | bearer, required | `x-api-key` + `anthropic-version` (a bearer header is still sent, harmless) |
| llama-only fields | sent | stripped | stripped | stripped |
| output cap | `max_tokens` | `max_tokens` | `max_completion_tokens` | `max_tokens` (native body) |
| `thinking = true` | `enable_thinking` kwarg | `reasoning: {effort}` | `reasoning_effort` | adaptive models: `thinking: {type: adaptive}` + `output_config.effort` (thinking off sends effort `low`; `max_tokens` floor 32000); Haiku and older: `{type: enabled, budget_tokens}` with `max_tokens` >= budget + 1024, nothing when off |
| model probe | first id from `/v1/models`, 3s deadline | configured model looked up in the catalogue, 10s deadline | same | same, list fetched with `limit=1000` (paginated at 20); thinking capability from `GET /v1/models/{id}` once per client (10s, assume adaptive on failure) |
| context window probe | `/props` `default_generation_settings.n_ctx` (llama-cpp); `max_model_len` (vllm/openai-compatible); none (ollama) | `context_length` from the catalogue entry | none (stays at 50000 unless configured) | none (stays at 50000 unless configured) |
| `temperature` | sent | sent | never sent (gpt-5 family rejects values other than 1) | never sent (deprecated on current models) |
| usage | from stream | final chunk | `stream_options.include_usage` | `message_start` / `message_delta` (`cache_read` + `cache_creation` + input = prompt) |

Keys: `api_key` / `api_key_env` on each model slot, resolved as config field,
then the named env var, then the provider's conventional variable
(`OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`). Keys belong in
`~/.miniswe/config.toml` or the environment, never in a project's
`.miniswe/config.toml` (a user's project may not gitignore it). The project
override merge inherits keys from the global file, `miniswe init` never writes
one, and `miniswe config` / `miniswe info` redact.

Cost visibility ships with the feature: a `[usage]` log line per main-loop
call, a `[usage:total]` line before `[end]` summing every role, and an
opt-in `runtime.max_session_input_tokens` guard that stops the turn instead of
grinding into a bill (a bench run is ~100 rounds × ~30k prompt tokens; the
600-round cap is an order of magnitude more).

`context_window` is unset by default, which means auto: taken from the
server's startup probe when it reports one, else 50000 (see
`ModelConfig::context_window`). The probe reads no window from the OpenAI or
Anthropic listings (Anthropic does report `max_input_tokens`, up to 1M, which is deliberately not used as a default), so both stay at the 50000 fallback unless you configure a
value explicitly. OpenRouter's catalogue does report `context_length`, and
the probe uses it — meaning the effective window (and therefore per-round
compaction cost) can jump to whatever the routed model's real window is. Set
`context_window` explicitly on OpenRouter if you want per-round cost bounded
regardless of which model you route to.

Routing already fits: `[models.<slot>]` tables plus `[routing]` let the main
role run on a hosted model while the fast role (summaries, edit apply,
routers) stays local or on a cheap hosted model.

## Phases

1. **Phase 1 (this change):** provider enum + dialect shaping, keys and
   redaction, usage logging and budget guard, 429 retry with `retry-after`,
   hosted context-exceeded error patterns, OpenRouter + OpenAI + Anthropic via
   the compatibility layer, wiremock tests per dialect, docs.
2. **Phase 2 (in v0.1.0, plan below):** native Anthropic Messages client behind the same `LlmClient`:
   message/tool conversion, second SSE parser into the same accumulator,
   `cache_control` breakpoints on the near-static prefix, thinking-block replay
   across tool-use turns.
3. **Phase 3 (optional):** bench a hosted-main + local-fast mix against the
   all-local baseline.


## Phase 2 plan: native Anthropic Messages client (in v0.1.0)

Facts below were checked 2026-10-05 against the current Claude API reference.

### Why the compat layer is not enough

No prompt caching, no effort control, and current models cannot run without
thinking: Opus 5.5 and Fable 5.1 reject `thinking: disabled`, Sonnet 5.5
only offers `between_tools`, and the `enabled` + `budget_tokens` form is a
400 on every Claude 5 model.

### Behaviour

- `provider = "anthropic"` switches to `POST /v1/messages`. The compat layer
  stays reachable as `provider = "openai-compatible"` with the Anthropic
  endpoint. The Anthropic branches leave `shape.rs`.
- **One request shape for every current model**: `thinking: {type:
  "adaptive"}` plus `output_config: {effort}`. `model.thinking = true` sends
  `model.thinking_effort`, `false` sends `low`. No model-name table.
- **One capability split, read from the API**: `GET /v1/models/{id}` once per
  client, cached in the client. When `capabilities.thinking.types.adaptive`
  is unsupported (Haiku 4.5 and older), no `thinking` / `output_config` is
  sent unless `model.thinking = true`, which sends the old `enabled` +
  `thinking_budget_tokens` form. A failed lookup assumes adaptive.
- `max_tokens`: thinking counts against it, and the 4096 default would
  truncate most turns. Adaptive models get `max(max_output_tokens, 32000)`.
- **Thinking blocks are replayed verbatim.** `Message` gains
  `#[serde(skip)] provider_blocks: Option<Vec<Value>>` holding the assistant
  turn's raw content blocks (`thinking`, `redacted_thinking`, `text`,
  `tool_use`, anything unknown). `serde(skip)` keeps it out of the
  OpenAI-shaped body, so local dialects stay byte-identical.
- **Edited history degrades instead of failing.** Every thinking-capable
  request sends `thinking.block_binding.prefix_mismatch_behavior =
  "drop_block"` with beta `thinking-binding-controls-2026-08-01`. Thinking
  blocks are bound to the exact `system` + `tools` + earlier messages that
  produced them; compaction, masking, read pruning, a rewritten state block
  or a changed visible-tool set all invalidate later blocks, and accounts
  created on or after 2026-08-31 get a 400 without this setting. With it
  the API drops the stale blocks and proceeds. Drops are counted from
  `input_transformations` and logged.
- When the harness has edited an assistant message (tool-call repair,
  pruning) so that `provider_blocks` no longer matches `content` /
  `tool_calls`, the turn is rebuilt from the latter without thinking.
- **Prompt caching**: `cache_control` on the last tool definition, on the
  system block, and on the last content block before the `[CURRENT STATE]`
  marker. The state block is split into its own trailing text block so the
  prefix ahead of it stays byte-stable between rounds.
- `stop_reason`: `end_turn` -> `stop`, `tool_use` -> `tool_calls`,
  `max_tokens` -> `length`, `refusal` -> an error naming the category.
- `fallbacks: "default"` (beta `server-side-fallback-2026-07-01`) is sent
  for `claude-fable-5-1`, `claude-opus-5-5`, `claude-opus-5`,
  `claude-sonnet-5-5`, so a classifier decline is re-run server-side on
  another model instead of ending the turn. `model.anthropic_fallbacks =
  false` turns it off.
- Usage: `prompt_tokens = input + cache_creation + cache_read`,
  `cached = cache_read`; the `[usage]` line gains `cache_write`.
- Tools carry `eager_input_streaming: true` so the existing argument-size
  cap can still abort a runaway call mid-stream.

### Request conversion

- Leading `system` messages -> top-level `system`. A later `system` message
  becomes user text (mid-conversation `role: system` is unsupported on
  Sonnet 5 and has placement rules).
- Assistant `tool_calls` -> `tool_use` blocks, `arguments` parsed to an
  object (unparseable arguments are sent as `{"_raw": "..."}`).
- Consecutive `tool` messages -> one user message of `tool_result` blocks;
  a following user message is merged in as text after them.
- Tool definitions -> `name`, `description`, `input_schema`.

### Code layout

- `src/llm/providers/anthropic/`: `mod.rs` (types, re-exports),
  `request.rs` (conversion, cache breakpoints, thinking shape),
  `stream.rs` (SSE events -> `ChatResponse`), `caps.rs` (model lookup).
- `client.rs`: the connect / idle-timeout / cancel loop of
  `stream_once_assembled` is separated from event parsing so both wire
  formats share it; the OpenAI parser moves unchanged. Retry, deadline and
  429 handling are shared as they are.
- `types.rs`: the new `Message` field (5 struct-literal sites), `cache_write`
  on `Usage`.
- Model listing: `max_input_tokens` is not read. The effective default stays
  at 50000 (a 1M window would make every round expensive), and showing a
  number that is not the window in use would mislead.

### Tests (wiremock, no network)

Request: headers and betas, system hoist, tool_result merging, the three
breakpoints with the state block split out, adaptive + effort, the Haiku
shape, no `temperature`, `max_tokens` floor. Stream: text, `tool_use` via
`input_json_delta`, thinking + signature round trip into the next request,
usage with cache fields, `refusal`, mid-stream `error` event, `max_tokens`.
Also: an edited assistant message falls back to a rebuilt turn, and the
llama.cpp body key-set test still passes untouched.

### Live validation (user-run)

`tmp/hosted-smoke.sh` on `claude-haiku-4-5`, then `claude-sonnet-5-5`:
no 400s, `cached > 0` from round 2, dropped-thinking count per run.

### Not in this phase

Append-only history for Anthropic (mid-conversation system messages for the
state block, server-side compaction, a frozen tool set). Decide from the
measured cache-hit and dropped-block numbers.

## Validation

- Unit: wiremock tests assert, per provider, the auth header, absence of
  llama-only fields, the output-cap field name, the thinking translation,
  usage parsed from a final chunk, a 401 producing an error that names the
  env var, and a 429 that waits and retries.
- Live (user-run): one bench task through OpenRouter on a cheap open model,
  then one each on Claude and OpenAI with the round cap lowered. The OpenAI
  run confirms the unverified parameter behavior above.
