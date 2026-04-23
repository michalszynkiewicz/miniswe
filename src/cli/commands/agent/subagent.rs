//! Subagent execution — run multiple independent agent loops concurrently.
//!
//! Each subagent gets its own prompt and message history, executes tools
//! through the shared pool, and returns its final output as a string.
//! LLM concurrency is controlled by the shared `LlmWorkerHandle` (jobs queue
//! on the worker threads, so `runtime.llm_concurrency` caps parallel calls).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::Mutex;

use crate::config::{Config, ModelRole};
use crate::context;
use crate::llm::{ChatRequest, Message};
use crate::lsp::LspClient;
use crate::mcp::McpRegistry;
use crate::runtime::{LlmWorkerEvent, LlmWorkerHandle, ToolWorkerPool};
use crate::tools;
use crate::tools::permissions::PermissionManager;
use crate::tui::app::LineStyle;

/// A single subagent task.
pub struct AgentTask {
    pub label: String,
    pub prompt: String,
}

/// The output collected from a completed subagent.
pub struct AgentOutput {
    pub label: String,
    pub content: String,
}

/// Run `tasks` concurrently. Returns outputs in the same order as `tasks`.
///
/// Concurrency at the LLM layer is controlled by the number of worker threads
/// in `llm_worker` (set via `runtime.llm_concurrency` in config).
/// Tool calls from different agents genuinely overlap via the shared `tool_pool`.
pub async fn run_subagents(
    tasks: Vec<AgentTask>,
    config: &Config,
    llm_worker: &LlmWorkerHandle,
    tool_pool: &ToolWorkerPool,
    // Tool defs for the parent — spawn_agents is stripped to prevent recursion.
    parent_tool_defs: &[crate::llm::ToolDefinition],
    perms: &Arc<PermissionManager>,
    mcp_registry: &Option<Arc<Mutex<McpRegistry>>>,
    lsp: &Option<Arc<LspClient>>,
    fast_revisions: &Option<Arc<tools::RevisionStore>>,
    fast_baseline_errors: usize,
    cancelled: &Arc<AtomicBool>,
    // Optional channel to forward live output lines to the TUI.
    output_tx: Option<tokio::sync::mpsc::UnboundedSender<(String, LineStyle)>>,
) -> Vec<AgentOutput> {
    let tool_defs: Vec<_> = parent_tool_defs
        .iter()
        .filter(|t| t.function.name != "spawn_agents")
        .cloned()
        .collect();

    let futures: Vec<_> = tasks
        .into_iter()
        .map(|task| {
            run_single_subagent(
                task,
                config,
                llm_worker,
                tool_pool,
                &tool_defs,
                perms,
                mcp_registry,
                lsp,
                fast_revisions,
                fast_baseline_errors,
                cancelled,
                output_tx.clone(),
            )
        })
        .collect();

    futures::future::join_all(futures).await
}

async fn run_single_subagent(
    task: AgentTask,
    config: &Config,
    llm_worker: &LlmWorkerHandle,
    tool_pool: &ToolWorkerPool,
    tool_defs: &[crate::llm::ToolDefinition],
    perms: &Arc<PermissionManager>,
    mcp_registry: &Option<Arc<Mutex<McpRegistry>>>,
    lsp: &Option<Arc<LspClient>>,
    fast_revisions: &Option<Arc<tools::RevisionStore>>,
    fast_baseline_errors: usize,
    cancelled: &Arc<AtomicBool>,
    output_tx: Option<tokio::sync::mpsc::UnboundedSender<(String, LineStyle)>>,
) -> AgentOutput {
    let label = task.label.clone();

    macro_rules! emit {
        ($style:expr, $fmt:literal $(, $arg:expr)*) => {
            if let Some(ref tx) = output_tx {
                let _ = tx.send((format!("[{}] {}", label, format!($fmt $(, $arg)*)), $style));
            }
        };
    }

    let mcp_summary = mcp_registry
        .as_ref()
        .and_then(|r| r.lock().context_summary());
    let assembled = context::assemble(config, &task.prompt, &[], false, mcp_summary.as_deref());
    let mut messages = assembled.messages;

    // Cap subagent rounds at 30 to prevent runaway
    let max_rounds = config.context.max_rounds.min(30);
    let mut final_content = String::new();

    for round in 0..max_rounds {
        if cancelled.load(Ordering::Relaxed) {
            emit!(LineStyle::Status, "cancelled");
            break;
        }

        context::sanitize_messages(&mut messages);

        let request = ChatRequest {
            messages: messages.clone(),
            tools: Some(tool_defs.to_vec()),
            tool_choice: None,
        };

        let mut llm_events = llm_worker.submit(ModelRole::Default, request, cancelled.clone());

        // Drain the streaming response
        let response = loop {
            match llm_events.recv().await {
                Some(LlmWorkerEvent::Token(_)) => {}
                Some(LlmWorkerEvent::Completed(Ok(r))) => break Some(r),
                Some(LlmWorkerEvent::Completed(Err(e))) => {
                    emit!(LineStyle::Error, "LLM error: {e}");
                    break None;
                }
                None => break None,
            }
        };

        let Some(response) = response else { break };
        let Some(choice) = response.choices.first() else {
            break;
        };
        let assistant_msg = &choice.message;

        if let Some(ref content) = assistant_msg.content {
            if !content.is_empty() {
                final_content = content.clone();
            }
        }

        let has_content = assistant_msg
            .content
            .as_deref()
            .is_some_and(|s| !s.is_empty());
        let has_tc = assistant_msg
            .tool_calls
            .as_deref()
            .is_some_and(|tc| !tc.is_empty());
        if has_content || has_tc {
            messages.push(assistant_msg.clone());
        }

        let tool_calls = match &assistant_msg.tool_calls {
            Some(tc) if !tc.is_empty() => tc.clone(),
            _ => break, // no tool calls → done
        };

        if round == 0 {
            emit!(
                LineStyle::Status,
                "working ({} tool calls)",
                tool_calls.len()
            );
        }

        for tc in &tool_calls {
            let args: serde_json::Value = match serde_json::from_str(&tc.function.arguments) {
                Ok(v) => v,
                Err(e) => {
                    let msg = format!("invalid JSON args: {e}");
                    messages.push(Message::tool_result(&tc.id, &msg));
                    continue;
                }
            };

            let result = execute_subagent_tool(
                &tc.function.name,
                &tc.id,
                &args,
                config,
                tool_pool,
                perms,
                mcp_registry,
                lsp,
                fast_revisions,
                fast_baseline_errors,
            )
            .await;

            let first_line = result.content.lines().next().unwrap_or("(empty)");
            if result.success {
                emit!(LineStyle::ToolOk, "✓ {}: {first_line}", tc.function.name);
            } else {
                emit!(LineStyle::ToolErr, "✗ {}: {first_line}", tc.function.name);
            }

            messages.push(Message::tool_result(&tc.id, &result.content));
        }
    }

    emit!(LineStyle::Status, "done");
    AgentOutput {
        label,
        content: final_content,
    }
}

async fn execute_subagent_tool(
    name: &str,
    id: &str,
    args: &serde_json::Value,
    config: &Config,
    tool_pool: &ToolWorkerPool,
    perms: &Arc<PermissionManager>,
    mcp_registry: &Option<Arc<Mutex<McpRegistry>>>,
    lsp: &Option<Arc<LspClient>>,
    fast_revisions: &Option<Arc<tools::RevisionStore>>,
    fast_baseline_errors: usize,
) -> tools::ToolResult {
    // mcp_use is handled inline (needs registry lock, not thread-pool friendly)
    if name == "mcp_use" {
        let server = args["server"].as_str().unwrap_or("").to_string();
        let tool = args["tool"].as_str().unwrap_or("").to_string();
        if server.is_empty() || tool.is_empty() {
            return tools::ToolResult::err(
                "mcp_use requires top-level 'server' and 'tool' string fields.".into(),
            );
        }
        let tool_args = args.get("arguments").cloned().unwrap_or_default();
        return match mcp_registry {
            Some(registry) => {
                let result = registry.lock().call_tool(&server, &tool, tool_args);
                result
                    .map(tools::ToolResult::ok)
                    .unwrap_or_else(|e| tools::ToolResult::err(format!("MCP error: {e}")))
            }
            None => tools::ToolResult::err("No MCP servers connected".into()),
        };
    }

    // fast-mode tools need the revision store
    if matches!(
        name,
        "replace_range" | "insert_at" | "revert" | "show_rev" | "check"
    ) {
        let tool_name = name.to_string();
        let args = args.clone();
        let config = config.clone();
        let perms = perms.clone();
        let lsp = lsp.clone();
        let revisions = fast_revisions.clone();
        let baseline = fast_baseline_errors;
        let mut rx = tool_pool.submit(move || {
            let Some(revisions) = revisions else {
                return Ok(tools::ToolResult::err(
                    "fast mode: revision store unavailable".into(),
                ));
            };
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| e.to_string())?;
            runtime
                .block_on(async move {
                    tools::execute_fast_tool(
                        &tool_name,
                        &args,
                        &config,
                        perms.as_ref(),
                        lsp.as_deref(),
                        revisions.as_ref(),
                        baseline,
                    )
                    .await
                })
                .map_err(|e| format!("fast tool error: {e}"))
        });
        return match rx.await {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => tools::ToolResult::err(e),
            Err(_) => tools::ToolResult::err("tool pool dropped fast-mode job".into()),
        };
    }

    // General tools via thread pool
    let tool_name = name.to_string();
    let args = args.clone();
    let config = config.clone();
    let perms = perms.clone();
    let lsp = lsp.clone();
    let mut rx = tool_pool.submit(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| e.to_string())?;
        runtime
            .block_on(async move {
                tools::execute_tool(&tool_name, &args, &config, perms.as_ref(), lsp.as_deref())
                    .await
            })
            .map_err(|e| format!("tool error: {e}"))
    });
    match rx.await {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => tools::ToolResult::err(e),
        Err(_) => tools::ToolResult::err(format!("tool pool dropped job for {id}")),
    }
}

/// Format all subagent outputs into a single tool result string.
pub fn format_outputs(outputs: Vec<AgentOutput>) -> String {
    outputs
        .into_iter()
        .map(|o| format!("## Agent: {}\n\n{}\n", o.label, o.content.trim()))
        .collect::<Vec<_>>()
        .join("\n---\n\n")
}
