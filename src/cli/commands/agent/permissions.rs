//! Map a `(tool_name, args)` pair to the `Action` that the
//! [`PermissionManager`](crate::tools::permissions::PermissionManager)
//! should approve or deny. Returns `None` for tools that don't require
//! permission gating.
//!
//! Uses the cosmetic `get_str_or` helpers: malformed args produce an
//! empty-string Action which no allowlist matches, so it fails safe.
//! The per-tool `require_*` checks in `src/tools/dispatch.rs` produce
//! the user-facing error.

use crate::tools::args::get_str_or;
use crate::tools::permissions::Action;

pub fn permission_action(tool_name: &str, args: &serde_json::Value) -> Option<Action> {
    match tool_name {
        "shell" => {
            let check = get_str_or(args, "check", "");
            if !check.is_empty() {
                // wait's probe command — same permission surface as running it
                Some(Action::Shell(check.into()))
            } else if get_str_or(args, "action", "") == "run" || args.get("action").is_none() {
                Some(Action::Shell(get_str_or(args, "command", "").into()))
            } else {
                None // wait/status/kill without check: no new execution
            }
        }
        "file" if get_str_or(args, "action", "") == "shell" => {
            Some(Action::Shell(get_str_or(args, "command", "").into()))
        }
        "web_search" => Some(Action::WebSearch(get_str_or(args, "query", "").into())),
        "web_fetch" => Some(Action::WebFetch(get_str_or(args, "url", "").into())),
        "mcp_use" => Some(Action::McpUse(
            get_str_or(args, "server", "").into(),
            get_str_or(args, "tool", "").into(),
        )),
        _ => None,
    }
}
