# Hosted providers: OpenRouter, OpenAI, Anthropic

Status: phase 1 in progress (2026-10-01). Phases 2 and 3 are not started.

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

Not verified (OpenAI's reference is behind a 403 from the dev container; from
memory, to be confirmed by the first live run): OpenAI rejects unknown
top-level fields with HTTP 400, reasoning models require
`max_completion_tokens` and reject `temperature`.

## Design

`provider` stays a string in `[model]` for config compatibility and is parsed
into an enum. Every provider except `anthropic` native (phase 2) shares the
OpenAI chat-completions wire protocol; a dialect layer in `src/llm/providers/`
decides per provider:

| decision | llama-cpp / ollama / vllm / openai-compatible | openrouter | openai | anthropic (phase 1 = compat layer) |
|---|---|---|---|---|
| default endpoint | none, configured | `https://openrouter.ai/api/v1` | `https://api.openai.com/v1` | `https://api.anthropic.com/v1` |
| auth | bearer if a key is configured | bearer, required | bearer, required | bearer, required (+ `x-api-key`, `anthropic-version`) |
| llama-only fields | sent | stripped | stripped | stripped |
| output cap | `max_tokens` | `max_tokens` | `max_completion_tokens` | `max_tokens` |
| `thinking = true` | `enable_thinking` kwarg | `reasoning: {effort}` | `reasoning_effort`, no `temperature` | `thinking: {type: enabled, budget_tokens}`, `max_tokens` raised to at least budget + 1024 |
| model probe | first id from `/v1/models` | configured model looked up in the catalogue | same | same |
| `temperature` | sent | sent | sent unless thinking | never sent (deprecated on current models) |
| usage | from stream | final chunk | `stream_options.include_usage` | `stream_options.include_usage` |

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

`context_window` keeps its 50000 default on hosted providers on purpose:
compaction at 50k bounds per-round cost. Raise it per model if wanted.

Routing already fits: `[models.<slot>]` tables plus `[routing]` let the main
role run on a hosted model while the fast role (summaries, edit apply,
routers) stays local or on a cheap hosted model.

## Phases

1. **Phase 1 (this change):** provider enum + dialect shaping, keys and
   redaction, usage logging and budget guard, 429 retry with `retry-after`,
   hosted context-exceeded error patterns, OpenRouter + OpenAI + Anthropic via
   the compatibility layer, wiremock tests per dialect, docs.
2. **Phase 2:** native Anthropic Messages client behind the same `LlmClient`:
   message/tool conversion, second SSE parser into the same accumulator,
   `cache_control` breakpoints on the near-static prefix, thinking-block replay
   across tool-use turns.
3. **Phase 3 (optional):** bench a hosted-main + local-fast mix against the
   all-local baseline.

## Validation

- Unit: wiremock tests assert, per provider, the auth header, absence of
  llama-only fields, the output-cap field name, the thinking translation,
  usage parsed from a final chunk, a 401 producing an error that names the
  env var, and a 429 that waits and retries.
- Live (user-run): one bench task through OpenRouter on a cheap open model,
  then one each on Claude and OpenAI with the round cap lowered. The OpenAI
  run confirms the unverified parameter behavior above.
