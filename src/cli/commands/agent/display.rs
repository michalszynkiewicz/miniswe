//! One-line tool-call summaries for the transcript and session log.
//!
//! Every tool that needs to appear in the "→ tool(...)" line in the UI
//! should have a dedicated arm here. Falling through to
//! `format!("{args}")` dumps raw JSON, which breaks scrolling in the
//! ratatui line-count math — once bit us for fast-mode tools that were
//! missing from this match for weeks.
//!
//! TODO(design): move `summarize(args) -> String` onto the tool
//! definition itself (next to `name` / `parameters` in
//! `src/tools/definitions.rs`) so a new tool can't be added without also
//! supplying a summary.

/// One-line summary of a tool call's arguments.
pub fn summarize_args(tool_name: &str, args: &serde_json::Value) -> String {
    let action = args["action"].as_str().unwrap_or("");
    match tool_name {
        "file" => match action {
            "read" => {
                let path = args["path"].as_str().unwrap_or("?");
                let start = args["start_line"].as_u64();
                let end = args["end_line"].as_u64();
                match (start, end) {
                    (Some(s), Some(e)) => format!("read {path}:{s}-{e}"),
                    (Some(s), None) => format!("read {path}:{s}-"),
                    _ => format!("read {path}"),
                }
            }
            "search" => {
                let query = args["query"]
                    .as_str()
                    .or_else(|| args["pattern"].as_str())
                    .unwrap_or("?");
                let scope = args["scope"]
                    .as_str()
                    .or_else(|| args["path"].as_str())
                    .unwrap_or("project");
                format!("search \"{query}\" in {scope}")
            }
            "delete" => {
                let path = args["path"].as_str().unwrap_or("?");
                format!("delete {path}")
            }
            "shell" => {
                let cmd = args["command"].as_str().unwrap_or("?");
                let timeout = args["timeout"].as_u64();
                match timeout {
                    Some(t) => format!("shell {} [timeout={t}]", crate::truncate_chars(cmd, 40)),
                    None => format!("shell {}", crate::truncate_chars(cmd, 40)),
                }
            }
            "revert" => "revert".to_string(),
            _ => action.to_string(),
        },
        "code" => match action {
            "goto_definition" | "find_references" => {
                let path = args["path"].as_str().unwrap_or("?");
                let line = args["line"].as_u64().unwrap_or(0);
                format!("{action} {path}:{line}")
            }
            _ => action.to_string(),
        },
        "web" => match action {
            "search" => {
                let query = args["query"].as_str().unwrap_or("?");
                format!("search \"{query}\"")
            }
            "fetch" => args["url"].as_str().unwrap_or("?").to_string(),
            _ => action.to_string(),
        },
        "plan" => match action {
            "scratchpad" => "scratchpad".to_string(),
            "check" => format!("check step {}", args["step"].as_u64().unwrap_or(0)),
            "refine" => format!("refine step {}", args["step"].as_u64().unwrap_or(0)),
            _ => action.to_string(),
        },
        "edit_file" => {
            let path = args["path"].as_str().unwrap_or("?");
            let task = args["task"].as_str().unwrap_or("");
            let lsp = args["lsp_validation"].as_str().unwrap_or("auto");
            if task.is_empty() {
                path.to_string()
            } else if lsp == "auto" {
                format!("{path}: {}", crate::truncate_chars(task, 70))
            } else {
                format!("{path}: {} [lsp={lsp}]", crate::truncate_chars(task, 58))
            }
        }
        "write_file" => {
            let path = args["path"].as_str().unwrap_or("?");
            format!("write {path}")
        }
        "mcp_use" => {
            let server = args["server"].as_str().unwrap_or("?");
            let tool = args["tool"].as_str().unwrap_or("?");
            format!("{server}/{tool}")
        }
        "replace_range" => {
            let path = args["path"].as_str().unwrap_or("?");
            let start = args["start"].as_u64().unwrap_or(0);
            let end = args["end"].as_u64().unwrap_or(0);
            format!("{path} L{start}-{end}")
        }
        "insert_at" => {
            let path = args["path"].as_str().unwrap_or("?");
            let after = args["after_line"].as_u64().unwrap_or(0);
            format!("{path} @L{after}")
        }
        "revert" => {
            let path = args["path"].as_str().unwrap_or("?");
            let rev = args["rev"].as_u64().unwrap_or(0);
            format!("{path} to rev_{rev}")
        }
        "show_rev" => {
            let path = args["path"].as_str().unwrap_or("?");
            let rev = args["rev"].as_u64().unwrap_or(0);
            format!("{path} rev_{rev}")
        }
        "check" => String::new(),
        _ => format!("{args}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn file_search_summary_uses_pattern_and_path() {
        let args = json!({
            "action": "search",
            "path": "tests",
            "pattern": "system_prompt_override",
        });

        assert_eq!(
            summarize_args("file", &args),
            "search \"system_prompt_override\" in tests"
        );
    }

    #[test]
    fn plan_refine_summary_includes_step() {
        let args = json!({
            "action": "refine",
            "step": 2,
        });

        assert_eq!(summarize_args("plan", &args), "refine step 2");
    }

    #[test]
    fn web_search_summary_includes_query() {
        let args = json!({
            "action": "search",
            "query": "Michał Szynkiewicz",
        });

        assert_eq!(
            summarize_args("web", &args),
            "search \"Michał Szynkiewicz\""
        );
    }

    #[test]
    fn file_read_summary_includes_line_range() {
        let args = json!({
            "action": "read",
            "path": "src/main.rs",
            "start_line": 10,
            "end_line": 20,
        });
        assert_eq!(summarize_args("file", &args), "read src/main.rs:10-20");
    }

    #[test]
    fn replace_range_summary_shows_line_span() {
        let args = json!({ "path": "src/lib.rs", "start": 5, "end": 12 });
        assert_eq!(summarize_args("replace_range", &args), "src/lib.rs L5-12");
    }
}
