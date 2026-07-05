//! Renders `miniswe::tools::ToolDetail` into text appropriate for an MCP
//! client (Claude Code), instead of reusing `ToolResult::content` verbatim —
//! that string is written for miniswe's own agent loop and, for some
//! failures, names miniswe-only tools (`file(...)`, `refactor(...)`) that
//! don't exist in an MCP client's tool list. See docs/miniswe-mcp.md.

use miniswe::tools::ToolDetail;

/// The MCP-facing example for a `refactor` action, in the flat tool syntax
/// this server actually exposes (see `toolset::list_tools`) — the shared
/// `ToolDetail::InvalidArgs.action` names the grouped miniswe action
/// (`add_param`/`drop_param`), which this maps to our own tool names.
fn mcp_example(action: &'static str) -> &'static str {
    match action {
        "add_param" => {
            "add_function_param(path=\"src/lib.rs\", function=\"assemble\", param=\"x: u32\", call_value=\"0\", after=\"b\")"
        }
        "drop_param" => {
            "drop_function_param(path=\"src/lib.rs\", function=\"assemble\", param=\"x\")"
        }
        other => other,
    }
}

/// Render a `ToolDetail` into the text this server should actually return,
/// in place of `ToolResult::content`.
pub fn render(detail: &ToolDetail) -> String {
    match detail {
        ToolDetail::LspUnavailable => {
            "LSP is not available for this project. Use your own file-search/grep tool to locate it instead.".into()
        }

        ToolDetail::InvalidArgs {
            action,
            missing,
            bad_type,
            unknown,
        } => {
            let mut parts = Vec::new();
            if !missing.is_empty() {
                parts.push(format!(
                    "missing required parameter(s): {}",
                    missing.join(", ")
                ));
            }
            if !bad_type.is_empty() {
                parts.push(format!("type error(s): {}", bad_type.join("; ")));
            }
            if !unknown.is_empty() {
                parts.push(format!("unknown parameter(s): {}", unknown.join(", ")));
            }
            format!(
                "Invalid arguments for {action}: {}\nExample: {}",
                parts.join("; "),
                mcp_example(action),
            )
        }

        ToolDetail::PartialSignatureChange {
            action,
            total,
            succeeded,
            callsite_failures,
            callsite_report,
        } => {
            let mut msg = format!(
                "{action}: signature updated, but only {succeeded}/{total} callsite(s) were rewritten automatically.\n"
            );
            if !callsite_report.is_empty() {
                msg.push_str("Updated:\n");
                for line in callsite_report {
                    msg.push_str(line);
                    msg.push('\n');
                }
            }
            if !callsite_failures.is_empty() {
                msg.push_str(&format!("Not updated ({} callsite(s)):\n", callsite_failures.len()));
                for f in callsite_failures {
                    msg.push_str(&format!("  - {f}\n"));
                }
            }
            msg.push_str(
                "\nThis server has no revert tool for this change. To recover: edit the \
                 unresolved callsite(s) above directly, or use git to inspect/discard the \
                 change (e.g. `git diff -- <file>`, `git checkout -- <file>`) if you'd rather \
                 redo the refactor.",
            );
            msg
        }
    }
}
