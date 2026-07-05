//! Tool schema + dispatch: bridges MCP `tools/list` and `tools/call` into
//! miniswe's existing, unmodified `execute_tool` / `execute_refactor_tool`
//! entry points — the same functions the interactive agent loop uses.

use std::sync::Arc;

use serde_json::{Value, json};

use miniswe::config::{Config, EditMode};
use miniswe::llm::ModelRouter;
use miniswe::lsp::LspClient;
use miniswe::tools::definitions::{flat_refactor_tool_definitions, flat_to_refactor_args};
use miniswe::tools::{PermissionManager, execute_refactor_tool, execute_tool, tool_definitions};

/// Shared, long-lived state for the life of the server process — built once
/// at startup and reused across every `tools/call`.
pub struct Context {
    pub config: Config,
    pub perms: PermissionManager,
    pub router: ModelRouter,
    pub lsp: Option<Arc<LspClient>>,
}

/// Tool schemas exposed over MCP: the production `code` tool (goto_definition,
/// find_references, diagnostics, repo_map, project_info, architecture_notes)
/// plus the flat, single-purpose refactor tools. Both are pulled verbatim
/// from `miniswe::tools::definitions` — no schema is redefined here.
pub fn list_tools() -> Value {
    let code_tool = tool_definitions(EditMode::Fast)
        .into_iter()
        .find(|t| t.function.name == "code");

    let tools: Vec<Value> = code_tool
        .into_iter()
        .chain(flat_refactor_tool_definitions())
        .map(|t| {
            json!({
                "name": t.function.name,
                "description": t.function.description,
                "inputSchema": t.function.parameters,
            })
        })
        .collect();

    json!({ "tools": tools })
}

pub async fn call_tool(ctx: &Context, name: &str, arguments: &Value) -> Value {
    let result = match name {
        "code" => {
            execute_tool(
                "code",
                arguments,
                &ctx.config,
                &ctx.perms,
                ctx.lsp.as_deref(),
            )
            .await
        }

        "add_function_param" | "drop_function_param" | "rename_symbol" => {
            match flat_to_refactor_args(name, arguments) {
                Some(refactor_args) => {
                    execute_refactor_tool(
                        &refactor_args,
                        &ctx.config,
                        &ctx.router,
                        ctx.lsp.as_deref(),
                        None,
                        None,
                        None,
                    )
                    .await
                }
                None => Ok(miniswe::tools::ToolResult::err(format!(
                    "internal error: '{name}' did not map to refactor args"
                ))),
            }
        }

        other => Ok(miniswe::tools::ToolResult::err(format!(
            "Unknown tool: '{other}'"
        ))),
    };

    match result {
        Ok(r) => {
            let text = match &r.detail {
                Some(detail) => crate::errors::render(detail),
                None => r.content,
            };
            json!({
                "content": [{"type": "text", "text": text}],
                "isError": !r.success,
            })
        }
        Err(e) => json!({
            "content": [{"type": "text", "text": format!("error: {e:#}")}],
            "isError": true,
        }),
    }
}
