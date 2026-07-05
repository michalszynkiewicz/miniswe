//! miniswe-mcp — exposes miniswe's LSP-backed code intelligence and
//! structural refactor tools as an MCP server over stdio.
//!
//! Built on miniswe's existing public library API (`Config`,
//! `PermissionManager`, `LspClient`, `ModelRouter`, `execute_tool`,
//! `execute_refactor_tool`). The only production-crate changes this binary
//! required were additive: `ToolResult::detail` (see `errors` and
//! `miniswe::tools::ToolDetail`) lets this server render its own next-step
//! guidance for certain failures instead of reusing miniswe's own tool-name
//! references (`file(...)`, `refactor(...)`) that don't exist in an MCP
//! client's tool list. See docs/miniswe-mcp.md.
//!
//! Run from within a project directory (mirrors `miniswe` itself): the
//! project root is resolved from the process's current working directory.

mod errors;
mod protocol;
mod telemetry;
mod toolset;

use std::io::{self, BufRead};
use std::sync::Arc;
use std::time::Instant;

use serde_json::json;

use miniswe::config::Config;
use miniswe::llm::ModelRouter;
use miniswe::lsp::LspClient;
use miniswe::tools::PermissionManager;

use telemetry::Telemetry;
use toolset::Context;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Tracing must go to stderr — stdout is reserved for JSON-RPC frames.
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(io::stderr)
        .init();

    let config = Config::load()?;
    let perms = PermissionManager::headless(&config);
    let router = ModelRouter::new(&config);
    let lsp = if config.lsp.enabled {
        match LspClient::spawn(config.project_root.clone()).await {
            Ok(client) => Some(Arc::new(client)),
            Err(e) => {
                tracing::warn!("LSP unavailable, code-intel tools will degrade: {e}");
                None
            }
        }
    } else {
        None
    };

    let mut telemetry = Telemetry::open(&config.project_root)?;
    telemetry.log_start(lsp.as_ref().is_some_and(|l| l.is_ready()));

    let ctx = Context {
        config,
        perms,
        router,
        lsp,
    };

    let result = run_stdio_loop(&ctx, &mut telemetry).await;
    telemetry.log_stop();
    result
}

async fn run_stdio_loop(ctx: &Context, telemetry: &mut Telemetry) -> anyhow::Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    // One request at a time: the blocking line-read is fine here since
    // nothing else needs to make progress concurrently on this thread.
    for line in stdin.lock().lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let req: protocol::Request = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("ignoring malformed JSON-RPC line: {e}");
                continue;
            }
        };

        if req.is_notification() {
            // e.g. notifications/initialized — nothing to do.
            continue;
        }
        let id = req.id.clone().unwrap_or(serde_json::Value::Null);

        match req.method.as_str() {
            "initialize" => {
                let result = json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "miniswe-mcp",
                        "version": env!("CARGO_PKG_VERSION"),
                    },
                });
                protocol::write_response(&mut stdout, id, result)?;
            }

            "tools/list" => {
                protocol::write_response(&mut stdout, id, toolset::list_tools())?;
            }

            "tools/call" => {
                let name = req
                    .params
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let empty = json!({});
                let arguments = req.params.get("arguments").unwrap_or(&empty).clone();

                let started = Instant::now();
                let result = toolset::call_tool(ctx, &name, &arguments).await;
                record_call(telemetry, &name, &result, started.elapsed());

                protocol::write_response(&mut stdout, id, result)?;
            }

            other => {
                protocol::write_error(
                    &mut stdout,
                    id,
                    -32601,
                    format!("Method not found: {other}"),
                )?;
            }
        }
    }

    Ok(())
}

/// Pull outcome + a short error detail out of a `tools/call` result (shaped
/// by `toolset::call_tool` as `{content: [{type, text}], isError}`) and feed
/// it to telemetry.
fn record_call(
    telemetry: &mut Telemetry,
    tool: &str,
    result: &serde_json::Value,
    duration: std::time::Duration,
) {
    let is_error = result
        .get("isError")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let detail = is_error
        .then(|| {
            result
                .get("content")
                .and_then(|c| c.get(0))
                .and_then(|c| c.get("text"))
                .and_then(|t| t.as_str())
        })
        .flatten()
        .map(|text| miniswe::truncate_chars(text, 200));

    telemetry.record_call(tool, !is_error, duration, detail.as_deref());
}
