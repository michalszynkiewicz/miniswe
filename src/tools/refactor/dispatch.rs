//! Top-level dispatch for the `refactor` and `rename` tools.

use anyhow::Result;
use serde_json::Value;

use crate::config::Config;
use crate::llm::ModelRouter;
use crate::logging::SessionLog;
use crate::lsp::LspClient;
use crate::tools::ToolResult;
use crate::tools::args;
use crate::tools::fast::RevisionStore;

use super::{add_param, drop_param, rename};

/// Help text shown when the agent calls `refactor(action="help")`.
pub const REFACTOR_HELP: &str = "\
Refactor a function signature or rename a symbol across every callsite. Backed by LSP for
both location resolution AND callsite discovery — you supply the function/symbol NAME and
the tool finds the canonical definition position itself.

Actions:

- add_param: Add a new parameter to a function signature, and at every callsite, insert a literal
  expression at the matching slot.
  Required: path, name, new_param, position, callsite_fill_in
  Optional: line (disambiguation hint when multiple methods share `name`)

  EXAMPLE:
    You added a 6th parameter `system_prompt_override: Option<&str>` to `context::assemble`
    in src/context/mod.rs. To update the signature AND every callsite in one shot:

    refactor(action=\"add_param\",
             path=\"src/context/mod.rs\", name=\"assemble\",
             new_param=\"system_prompt_override: Option<&str>\",
             position=\"after:mcp_summary\",
             callsite_fill_in=\"None\")

    Note: `path` must contain the function's DEFINITION, not just a call site. If you only
    have a call, use code(goto_definition) first to find the defining file. `position` accepts:
      - \"start\"            insert before all existing params
      - \"after:<name>\"     insert immediately after the existing param with that name
                          (use the name of the LAST param to append to the end)

- drop_param: Remove a parameter from a function signature and the corresponding argument from
  every callsite. SKIPs callsites where the dropped argument is a side-effecting expression.
  Required: path, name, param
  Optional: line (disambiguation hint)

- rename: LSP-driven cross-file rename of any symbol. See the rename section for required args.

- help: Show this text.";

/// Help text for the standalone `rename` tool.
pub const RENAME_HELP: &str = "\
Rename a symbol (function, type, variable, parameter, field, module) using the language
server's native rename.

Required: path, line, name, new_name

EXAMPLE:
    rename(path=\"src/context/mod.rs\", line=286, name=\"assemble\", new_name=\"build_context\")

The tool finds the column where `name` appears on `line`, then asks the LSP to rename. The LSP
determines the semantic scope (every reference, type-aware, cross-file) and produces a
WorkspaceEdit which this tool applies. Works for any symbol, not just functions. If you want to
add or drop a function parameter, use `refactor` instead.";

pub async fn execute_refactor_tool(
    args: &Value,
    config: &Config,
    router: &ModelRouter,
    lsp: Option<&LspClient>,
    log: Option<&SessionLog>,
    revisions: Option<&RevisionStore>,
) -> Result<ToolResult> {
    let action = match args::require_str(args, "action") {
        Ok(a) => a,
        Err(e) => return Ok(ToolResult::err(e)),
    };
    let result = match action {
        "help" => {
            return Ok(ToolResult::ok(format!(
                "{REFACTOR_HELP}\n\n--- rename action ---\n{RENAME_HELP}"
            )));
        }
        "add_param" => add_param::execute(args, config, router, lsp, log, revisions).await,
        "drop_param" => drop_param::execute(args, config, router, lsp, log, revisions).await,
        "rename" => rename::execute(args, config, lsp).await,
        _ => {
            return Ok(ToolResult::err(format!(
                "Unknown refactor action: '{action}'. Use action='help' to see options (add_param, drop_param, rename)."
            )));
        }
    };

    // Refactor touches the definition file plus every callsite — possibly
    // across many files. We don't have a single path to reindex, so do an
    // incremental project reindex (only mtime-changed files get
    // re-extracted). Without this the symbol index serves pre-refactor
    // signatures for code the model just rewrote.
    if result.as_ref().is_ok_and(|r| r.success) {
        crate::tools::edit_orchestration::reindex_project_incremental(config);
    }

    result
}
