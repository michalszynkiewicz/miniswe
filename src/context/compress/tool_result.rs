//! Summarize a tool result into a one-line observation. Used by the
//! observation-masking pass to replace stale tool outputs with dense
//! hints as the conversation grows past budget.

/// Summarize a tool result into a one-line observation for history compression.
///
/// This replaces full tool outputs in older conversation turns with dense summaries.
pub fn summarize_tool_result(tool_name: &str, args: &serde_json::Value, content: &str) -> String {
    use crate::tools::args::get_str_or;

    let action = get_str_or(args, "action", "");

    match tool_name {
        "file" => match action {
            "read" => {
                let path = get_str_or(args, "path", "?");
                let line_count = content.lines().count();
                let mut sigs = Vec::new();
                for line in content.lines() {
                    let stripped = if let Some(pos) = line.find('│') {
                        &line[pos + '│'.len_utf8()..]
                    } else {
                        line
                    };
                    let trimmed = stripped.trim();
                    if (trimmed.starts_with("pub fn ")
                        || trimmed.starts_with("pub async fn ")
                        || trimmed.starts_with("fn ")
                        || trimmed.starts_with("async fn "))
                        && trimmed.contains('(')
                    {
                        let sig = trimmed.split('{').next().unwrap_or(trimmed).trim();
                        if sig.len() < 80 {
                            sigs.push(sig.to_string());
                        }
                    }
                    if (trimmed.starts_with("pub struct ")
                        || trimmed.starts_with("pub enum ")
                        || trimmed.starts_with("pub trait ")
                        || trimmed.starts_with("struct ")
                        || trimmed.starts_with("enum ")
                        || trimmed.starts_with("trait "))
                        && !trimmed.contains(';')
                    {
                        let def = trimmed.split('{').next().unwrap_or(trimmed).trim();
                        if def.len() < 80 {
                            sigs.push(def.to_string());
                        }
                    }
                    if sigs.len() >= 10 {
                        break;
                    }
                }
                if sigs.is_empty() {
                    format!(
                        "[read:{path} ({line_count}L) — use file(action='read', path='{path}') to re-read]"
                    )
                } else {
                    format!(
                        "[read:{path} ({line_count}L) — use file(action='read', path='{path}') to re-read]\n{}",
                        sigs.join("\n")
                    )
                }
            }
            "search" => {
                let query = get_str_or(args, "query", "?");
                let match_count = content.lines().filter(|l| !l.starts_with('[')).count();
                format!("[search:\"{query}\"→{match_count} matches]")
            }
            "shell" => {
                let cmd = get_str_or(args, "command", "?");
                let short_cmd = crate::truncate_chars(cmd, 30);
                let exit_code = if content.contains("exit 0") {
                    "ok"
                } else {
                    "err"
                };
                format!("[shell:\"{short_cmd}\"→{exit_code}]")
            }
            _ => format!("[file.{action}→done]"),
        },
        "code" => match action {
            "diagnostics" => {
                let errors = content.lines().filter(|l| l.contains("error")).count();
                let warnings = content.lines().filter(|l| l.contains("warning")).count();
                format!("[diag:{errors}E,{warnings}W]")
            }
            _ => format!("[code.{action}→done]"),
        },
        "web" => match action {
            "search" => {
                let query = get_str_or(args, "query", "?");
                let result_count = content
                    .lines()
                    .filter(|l| l.starts_with(|c: char| c.is_ascii_digit()))
                    .count();
                format!("[web_search:\"{query}\"→{result_count} results]")
            }
            "fetch" => {
                let url = get_str_or(args, "url", "?");
                format!("[web_fetch:{url}→{}chars]", content.len())
            }
            _ => format!("[web.{action}→done]"),
        },
        "plan" => match action {
            "scratchpad" => "[scratchpad→ok]".to_string(),
            _ => format!("[plan.{action}→done]"),
        },
        "edit_file" => {
            let path = get_str_or(args, "path", "?");
            if content.contains('✓') {
                if content.contains("error")
                    && (content.contains("[cargo check]") || content.contains("[lsp]"))
                {
                    let errors: Vec<&str> = content
                        .lines()
                        .filter(|l| l.contains("error"))
                        .take(2)
                        .collect();
                    format!("[edit_file:{path}→ok but errors: {}]", errors.join("; "))
                } else {
                    format!("[edit_file:{path}→ok]")
                }
            } else {
                format!("[edit_file:{path}→failed]")
            }
        }
        "mcp_use" => {
            let server = get_str_or(args, "server", "?");
            let tool = get_str_or(args, "tool", "?");
            format!("[mcp:{server}/{tool}→{}chars]", content.len())
        }
        _ => format!("[{tool_name}→done]"),
    }
}

/// Extract likely exported symbol names from code content.
fn extract_symbol_names_from_content(content: &str) -> Vec<String> {
    let mut names = Vec::new();
    for line in content.lines().take(50) {
        let trimmed = line.trim();
        // Look for function/struct/class definitions
        for keyword in &[
            "pub fn ",
            "fn ",
            "pub struct ",
            "struct ",
            "pub enum ",
            "class ",
            "def ",
            "function ",
            "export function ",
        ] {
            if trimmed.contains(keyword)
                && let Some(after) = trimmed.split(keyword).nth(1)
            {
                let name: String = after
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                if !name.is_empty() && name.len() > 1 {
                    names.push(name);
                }
            }
        }
    }
    names.truncate(5);
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_result_summary() {
        let args = serde_json::json!({"action": "read", "path": "src/main.rs"});
        let content = "pub fn main() {\n    println!(\"hello\");\n}\n";
        let summary = summarize_tool_result("file", &args, content);
        assert!(
            summary.contains("src/main.rs"),
            "should have path: {summary}"
        );
        assert!(
            summary.contains("read"),
            "should hint at re-read: {summary}"
        );
    }
}
