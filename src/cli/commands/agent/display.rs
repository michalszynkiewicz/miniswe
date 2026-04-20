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
    use crate::tools::args::{get_str_or, get_u64_or, opt_str, opt_u64};

    let action = get_str_or(args, "action", "");
    match tool_name {
        "file" => match action {
            "read" => {
                let path = get_str_or(args, "path", "?");
                let start = opt_u64(args, "start_line").unwrap_or(None);
                let end = opt_u64(args, "end_line").unwrap_or(None);
                match (start, end) {
                    (Some(s), Some(e)) => format!("read {path}:{s}-{e}"),
                    (Some(s), None) => format!("read {path}:{s}-"),
                    _ => format!("read {path}"),
                }
            }
            "search" => {
                let query = opt_str(args, "query")
                    .ok()
                    .flatten()
                    .or_else(|| opt_str(args, "pattern").ok().flatten())
                    .unwrap_or("?");
                let scope = opt_str(args, "scope")
                    .ok()
                    .flatten()
                    .or_else(|| opt_str(args, "path").ok().flatten())
                    .unwrap_or("project");
                format!("search \"{query}\" in {scope}")
            }
            "delete" => {
                let path = get_str_or(args, "path", "?");
                format!("delete {path}")
            }
            "shell" => {
                let cmd = get_str_or(args, "command", "?");
                let timeout = opt_u64(args, "timeout").unwrap_or(None);
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
                let path = get_str_or(args, "path", "?");
                let line = get_u64_or(args, "line", 0);
                format!("{action} {path}:{line}")
            }
            _ => action.to_string(),
        },
        "web" => match action {
            "search" => {
                let query = get_str_or(args, "query", "?");
                format!("search \"{query}\"")
            }
            "fetch" => get_str_or(args, "url", "?").to_string(),
            _ => action.to_string(),
        },
        "plan" => match action {
            "scratchpad" => "scratchpad".to_string(),
            "check" => format!("check step {}", get_u64_or(args, "step", 0)),
            "refine" => format!("refine step {}", get_u64_or(args, "step", 0)),
            _ => action.to_string(),
        },
        "edit_file" => {
            let path = get_str_or(args, "path", "?");
            let task = get_str_or(args, "task", "");
            let lsp = get_str_or(args, "lsp_validation", "auto");
            if task.is_empty() {
                path.to_string()
            } else if lsp == "auto" {
                format!("{path}: {}", crate::truncate_chars(task, 70))
            } else {
                format!("{path}: {} [lsp={lsp}]", crate::truncate_chars(task, 58))
            }
        }
        "write_file" => {
            let path = get_str_or(args, "path", "?");
            format!("write {path}")
        }
        "mcp_use" => {
            let server = get_str_or(args, "server", "?");
            let tool = get_str_or(args, "tool", "?");
            format!("{server}/{tool}")
        }
        "replace_range" => {
            let path = get_str_or(args, "path", "?");
            let start = get_u64_or(args, "start", 0);
            let end = get_u64_or(args, "end", 0);
            format!("{path} L{start}-{end}")
        }
        "insert_at" => {
            let path = get_str_or(args, "path", "?");
            let after = get_u64_or(args, "after_line", 0);
            format!("{path} @L{after}")
        }
        "revert" => {
            let path = get_str_or(args, "path", "?");
            let rev = get_u64_or(args, "rev", 0);
            format!("{path} to rev_{rev}")
        }
        "show_rev" => {
            let path = get_str_or(args, "path", "?");
            let rev = get_u64_or(args, "rev", 0);
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
