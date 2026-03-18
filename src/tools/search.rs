//! search tool — ripgrep-based code search.

use anyhow::Result;
use serde_json::Value;
use std::process::Command;

use crate::config::Config;
use super::ToolResult;

pub async fn execute(args: &Value, config: &Config) -> Result<ToolResult> {
    let query = args["query"]
        .as_str()
        .unwrap_or("");

    if query.is_empty() {
        return Ok(ToolResult::err("Missing required parameter: query".into()));
    }

    let max_results = args["max_results"]
        .as_u64()
        .unwrap_or(20) as usize;

    let scope = args["scope"]
        .as_str()
        .unwrap_or("project");

    let search_dir = match scope {
        "project" | "symbols" => config.project_root.to_string_lossy().to_string(),
        dir => {
            let path = config.project_root.join(dir);
            if path.is_dir() {
                path.to_string_lossy().to_string()
            } else {
                config.project_root.to_string_lossy().to_string()
            }
        }
    };

    // Use ripgrep (rg) for fast search
    let output = Command::new("rg")
        .args([
            "--line-number",
            "--no-heading",
            "--color=never",
            "--max-count", &max_results.to_string(),
            "--max-columns", "200",
            "--type-add", "code:*.{rs,py,js,ts,tsx,jsx,go,java,c,cpp,h,hpp,rb,php,swift,kt,scala,zig,hs,ml,ex,exs,clj,sh,bash,zsh,toml,yaml,yml,json,md}",
            "-t", "code",
            query,
            &search_dir,
        ])
        .output();

    match output {
        Ok(result) => {
            let stdout = String::from_utf8_lossy(&result.stdout);
            let stderr = String::from_utf8_lossy(&result.stderr);

            if stdout.is_empty() && result.status.code() == Some(1) {
                return Ok(ToolResult::ok(format!("No matches found for: {query}")));
            }

            if !result.status.success() && !stderr.is_empty() {
                // rg might not be installed, fall back to grep
                return fallback_grep(query, &search_dir, max_results).await;
            }

            // Strip the project root prefix from paths for cleaner output
            let root_prefix = format!("{}/", config.project_root.display());
            let cleaned: String = stdout
                .lines()
                .take(max_results)
                .map(|line| line.strip_prefix(&root_prefix).unwrap_or(line))
                .collect::<Vec<_>>()
                .join("\n");

            let match_count = cleaned.lines().count();
            let mut output = format!("[search \"{query}\": {match_count} matches]\n");
            output.push_str(&cleaned);
            Ok(ToolResult::ok(output))
        }
        Err(_) => fallback_grep(query, &search_dir, max_results).await,
    }
}

/// Fallback to grep if rg is not available.
async fn fallback_grep(query: &str, dir: &str, max_results: usize) -> Result<ToolResult> {
    let output = Command::new("grep")
        .args([
            "-rn",
            "--include=*.rs",
            "--include=*.py",
            "--include=*.js",
            "--include=*.ts",
            "--include=*.go",
            "-m", &max_results.to_string(),
            query,
            dir,
        ])
        .output();

    match output {
        Ok(result) => {
            let stdout = String::from_utf8_lossy(&result.stdout);
            if stdout.is_empty() {
                Ok(ToolResult::ok(format!("No matches found for: {query}")))
            } else {
                let match_count = stdout.lines().count();
                Ok(ToolResult::ok(format!(
                    "[search \"{query}\": {match_count} matches]\n{stdout}"
                )))
            }
        }
        Err(e) => Ok(ToolResult::err(format!(
            "Neither rg nor grep available: {e}"
        ))),
    }
}
