//! Map a `(tool_name, args)` pair to the `Action` that the
//! [`PermissionManager`](crate::tools::permissions::PermissionManager)
//! should approve or deny. Returns `None` for tools that don't require
//! permission gating.

use crate::tools::permissions::Action;

pub fn permission_action(tool_name: &str, args: &serde_json::Value) -> Option<Action> {
    match tool_name {
        "shell" => Some(Action::Shell(args["command"].as_str().unwrap_or("").into())),
        "file" if args["action"].as_str().unwrap_or("") == "shell" => {
            Some(Action::Shell(args["command"].as_str().unwrap_or("").into()))
        }
        "web_search" => Some(Action::WebSearch(
            args["query"].as_str().unwrap_or("").into(),
        )),
        "web_fetch" => Some(Action::WebFetch(args["url"].as_str().unwrap_or("").into())),
        "mcp_use" => Some(Action::McpUse(
            args["server"].as_str().unwrap_or("").into(),
            args["tool"].as_str().unwrap_or("").into(),
        )),
        _ => None,
    }
}
