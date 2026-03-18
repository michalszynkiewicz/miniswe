//! Web tools — web_search, web_fetch, docs_lookup.

use anyhow::Result;
use serde_json::Value;
use std::fs;

use crate::config::Config;
use super::ToolResult;

/// web_search — Search the web via DuckDuckGo.
pub async fn search(args: &Value, _config: &Config) -> Result<ToolResult> {
    let query = args["query"].as_str().unwrap_or("");
    let max_results = args["max_results"].as_u64().unwrap_or(5) as usize;

    if query.is_empty() {
        return Ok(ToolResult::err("Missing required parameter: query".into()));
    }

    // Use DuckDuckGo HTML API via curl
    // This is a simplified approach; a production version would use a proper HTTP client
    let encoded_query = urlencoded(query);
    let url = format!("https://html.duckduckgo.com/html/?q={encoded_query}");

    let client = reqwest::Client::builder()
        .user_agent("Mozilla/5.0 (compatible; miniswe/0.1)")
        .build()?;

    match client.get(&url).send().await {
        Ok(response) => {
            let html = response.text().await.unwrap_or_default();
            let results = parse_ddg_results(&html, max_results);

            if results.is_empty() {
                return Ok(ToolResult::ok(format!("[search: \"{query}\" — no results]")));
            }

            let mut output = format!("[SEARCH:\"{query}\"]\n");
            for (i, (title, url, snippet)) in results.iter().enumerate() {
                output.push_str(&format!(
                    "{}. {}\n   {}\n   \"{}\"\n",
                    i + 1,
                    title,
                    url,
                    snippet
                ));
            }
            Ok(ToolResult::ok(output))
        }
        Err(e) => Ok(ToolResult::err(format!("Web search failed: {e}"))),
    }
}

/// web_fetch — Fetch a URL and extract content as markdown.
pub async fn fetch(args: &Value, config: &Config) -> Result<ToolResult> {
    let url = args["url"].as_str().unwrap_or("");

    if url.is_empty() {
        return Ok(ToolResult::err("Missing required parameter: url".into()));
    }

    // Use Jina Reader API by default
    let fetch_url = if config.web.fetch_backend == "jina" {
        format!("https://r.jina.ai/{url}")
    } else {
        url.to_string()
    };

    let client = reqwest::Client::builder()
        .user_agent("Mozilla/5.0 (compatible; miniswe/0.1)")
        .build()?;

    match client.get(&fetch_url).send().await {
        Ok(response) => {
            let content = response.text().await.unwrap_or_default();

            // Truncate to ~4K tokens (rough estimate: 4 chars per token)
            let max_chars = 16000;
            let truncated = if content.len() > max_chars {
                format!(
                    "{}\n\n[... truncated, {} chars total]",
                    &content[..max_chars],
                    content.len()
                )
            } else {
                content
            };

            Ok(ToolResult::ok(format!("[fetch: {url}]\n{truncated}")))
        }
        Err(e) => Ok(ToolResult::err(format!("Failed to fetch {url}: {e}"))),
    }
}

/// docs_lookup — Search local documentation cache.
pub async fn docs_lookup(args: &Value, config: &Config) -> Result<ToolResult> {
    let library = args["library"].as_str().unwrap_or("");
    let topic = args["topic"].as_str().unwrap_or("");

    if library.is_empty() {
        return Ok(ToolResult::err(
            "Missing required parameter: library".into(),
        ));
    }

    let docs_dir = config.miniswe_path("docs");

    if !docs_dir.exists() {
        return Ok(ToolResult::ok(format!(
            "No local docs cached. Run `miniswe docs add <url>` to add documentation."
        )));
    }

    // Look for files matching the library name
    let mut found = String::new();
    if let Ok(entries) = fs::read_dir(&docs_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name
                .to_lowercase()
                .contains(&library.to_lowercase())
            {
                let content = fs::read_to_string(entry.path()).unwrap_or_default();

                if topic.is_empty() {
                    // Return first 2K tokens worth
                    let max_chars = 8000;
                    let truncated = if content.len() > max_chars {
                        &content[..max_chars]
                    } else {
                        &content
                    };
                    found.push_str(truncated);
                } else {
                    // Search for relevant sections by keyword
                    let sections = extract_relevant_sections(&content, topic);
                    found.push_str(&sections);
                }
                break;
            }
        }
    }

    if found.is_empty() {
        Ok(ToolResult::ok(format!(
            "No docs found for '{library}'. Available docs:\n{}",
            list_cached_docs(config)
        )))
    } else {
        Ok(ToolResult::ok(format!(
            "[docs: {library}{}]\n{found}",
            if topic.is_empty() {
                String::new()
            } else {
                format!(" / {topic}")
            }
        )))
    }
}

/// Simple URL encoding.
fn urlencoded(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            ' ' => '+'.to_string(),
            c if c.is_alphanumeric() || "-_.~".contains(c) => c.to_string(),
            c => format!("%{:02X}", c as u32),
        })
        .collect()
}

/// Parse DuckDuckGo HTML results (simplified).
fn parse_ddg_results(html: &str, max: usize) -> Vec<(String, String, String)> {
    let mut results = Vec::new();

    // Simple HTML parsing for DuckDuckGo results
    // Look for result links and snippets
    for segment in html.split("class=\"result__a\"").skip(1).take(max) {
        let title = extract_between(segment, ">", "</a>")
            .unwrap_or_default()
            .replace("<b>", "")
            .replace("</b>", "");

        let url = extract_between(segment, "href=\"", "\"").unwrap_or_default();

        let snippet = if let Some(snip_segment) = segment.split("class=\"result__snippet\"").nth(1)
        {
            extract_between(snip_segment, ">", "</")
                .unwrap_or_default()
                .replace("<b>", "")
                .replace("</b>", "")
        } else {
            String::new()
        };

        if !title.is_empty() {
            results.push((title, url, snippet));
        }
    }

    results
}

/// Extract text between two markers.
fn extract_between(s: &str, start: &str, end: &str) -> Option<String> {
    let start_idx = s.find(start)? + start.len();
    let end_idx = s[start_idx..].find(end)? + start_idx;
    Some(s[start_idx..end_idx].to_string())
}

/// Extract sections from a document that match a keyword.
fn extract_relevant_sections(content: &str, keyword: &str) -> String {
    let keyword_lower = keyword.to_lowercase();
    let lines: Vec<&str> = content.lines().collect();
    let mut result = String::new();
    let mut in_relevant_section = false;
    let mut chars_added = 0;
    let max_chars = 8000;

    for (_i, line) in lines.iter().enumerate() {
        let line_lower = line.to_lowercase();

        // Check if this is a heading containing the keyword
        if line.starts_with('#') && line_lower.contains(&keyword_lower) {
            in_relevant_section = true;
            result.push_str(line);
            result.push('\n');
            chars_added += line.len() + 1;
            continue;
        }

        // End of relevant section at next heading of same or higher level
        if in_relevant_section && line.starts_with('#') && !line_lower.contains(&keyword_lower) {
            in_relevant_section = false;
        }

        if in_relevant_section && chars_added < max_chars {
            result.push_str(line);
            result.push('\n');
            chars_added += line.len() + 1;
        }
    }

    result
}

/// List cached documentation files.
fn list_cached_docs(config: &Config) -> String {
    let docs_dir = config.miniswe_path("docs");
    if !docs_dir.exists() {
        return "(none)".into();
    }

    let mut docs = Vec::new();
    if let Ok(entries) = fs::read_dir(&docs_dir) {
        for entry in entries.flatten() {
            docs.push(format!("  - {}", entry.file_name().to_string_lossy()));
        }
    }

    if docs.is_empty() {
        "(none)".into()
    } else {
        docs.join("\n")
    }
}
