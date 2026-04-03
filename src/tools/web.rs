//! Web tools — web_search, web_fetch, docs_lookup.

use anyhow::Result;
use serde_json::Value;
use std::fs;

use crate::config::Config;
use super::ToolResult;

/// web_search — Search the web via Serper (Google results) or SearXNG.
///
/// Requires an API key in config. Get a free key at https://serper.dev
/// (2,500 queries/month, no credit card).
pub async fn search(args: &Value, config: &Config) -> Result<ToolResult> {
    let query = args["query"].as_str().unwrap_or("");
    let max_results = args["max_results"].as_u64().unwrap_or(5) as usize;

    if query.is_empty() {
        return Ok(ToolResult::err("Missing required parameter: query".into()));
    }

    match config.web.search_backend.as_str() {
        "searxng" => search_searxng(query, max_results, config).await,
        "github" => search_github(query, max_results).await,
        _ => {
            // Try Serper if key is available, fall back to GitHub search
            let has_key_file = dirs::home_dir()
                .map(|h| h.join(".miniswe").join("serper.key").exists())
                .unwrap_or(false);
            if config.web.search_api_key.is_some()
                || has_key_file
                || std::env::var("SERPER_API_KEY").is_ok()
                || std::env::var("SEARCH_API_KEY").is_ok()
            {
                search_serper(query, max_results, config).await
            } else {
                search_github(query, max_results).await
            }
        }
    }
}

/// Search via Serper.dev (Google results, free tier: 2,500 queries/month).
async fn search_serper(query: &str, max_results: usize, config: &Config) -> Result<ToolResult> {
    let api_key = match &config.web.search_api_key {
        Some(key) if !key.is_empty() => key,
        _ => {
            // Check ~/.miniswe/serper.key, then environment variable
            let home_key = dirs::home_dir()
                .map(|h| h.join(".miniswe").join("serper.key"))
                .and_then(|p| std::fs::read_to_string(p).ok())
                .map(|k| k.trim().to_string())
                .filter(|k| !k.is_empty());
            match home_key.ok_or(()).or_else(|_|
                std::env::var("SERPER_API_KEY").or_else(|_| std::env::var("SEARCH_API_KEY"))
            ) {
                Ok(key) if !key.is_empty() => {
                    // Use env var (can't store reference, so do the request inline)
                    return do_serper_request(query, max_results, &key).await;
                }
                _ => {
                    return Ok(ToolResult::err(
                        "Web search requires an API key.\n\
                         Get a free key at https://serper.dev (2,500 queries/month).\n\
                         Set it in .miniswe/config.toml:\n\
                         [web]\n\
                         search_api_key = \"your-key-here\"\n\
                         \n\
                         Or set SERPER_API_KEY environment variable."
                            .into(),
                    ));
                }
            }
        }
    };

    do_serper_request(query, max_results, api_key).await
}

async fn do_serper_request(query: &str, max_results: usize, api_key: &str) -> Result<ToolResult> {
    let client = reqwest::Client::new();

    let response = client
        .post("https://google.serper.dev/search")
        .header("X-API-KEY", api_key)
        .header("Content-Type", "application/json")
        .json(&serde_json::json!({
            "q": query,
            "num": max_results
        }))
        .send()
        .await;

    match response {
        Ok(resp) => {
            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                return Ok(ToolResult::err(format!("Search API error ({status}): {text}")));
            }

            let data: Value = resp.json().await.unwrap_or_default();

            let mut output = format!("[SEARCH:\"{query}\"]\n");
            let mut count = 0;

            if let Some(organic) = data["organic"].as_array() {
                for result in organic.iter().take(max_results) {
                    let title = result["title"].as_str().unwrap_or("");
                    let url = result["link"].as_str().unwrap_or("");
                    let snippet = result["snippet"].as_str().unwrap_or("");

                    if !title.is_empty() {
                        count += 1;
                        output.push_str(&format!(
                            "{}. {}\n   {}\n   \"{}\"\n",
                            count, title, url, snippet
                        ));
                    }
                }
            }

            if count == 0 {
                Ok(ToolResult::ok(format!("[search: \"{query}\" — no results]")))
            } else {
                Ok(ToolResult::ok(output))
            }
        }
        Err(e) => Ok(ToolResult::err(format!("Web search failed: {e}"))),
    }
}

/// Search via SearXNG (self-hosted).
async fn search_searxng(query: &str, max_results: usize, config: &Config) -> Result<ToolResult> {
    let base_url = config
        .web
        .searxng_url
        .as_deref()
        .unwrap_or("http://localhost:8080");

    let encoded_query = urlencoded(query);
    let url = format!("{base_url}/search?q={encoded_query}&format=json");

    let client = reqwest::Client::new();

    match client.get(&url).send().await {
        Ok(resp) => {
            let data: Value = resp.json().await.unwrap_or_default();

            let mut output = format!("[SEARCH:\"{query}\"]\n");
            let mut count = 0;

            if let Some(results) = data["results"].as_array() {
                for result in results.iter().take(max_results) {
                    let title = result["title"].as_str().unwrap_or("");
                    let url = result["url"].as_str().unwrap_or("");
                    let snippet = result["content"].as_str().unwrap_or("");

                    if !title.is_empty() {
                        count += 1;
                        output.push_str(&format!(
                            "{}. {}\n   {}\n   \"{}\"\n",
                            count, title, url, snippet
                        ));
                    }
                }
            }

            if count == 0 {
                Ok(ToolResult::ok(format!("[search: \"{query}\" — no results]")))
            } else {
                Ok(ToolResult::ok(output))
            }
        }
        Err(e) => Ok(ToolResult::err(format!("SearXNG search failed: {e}"))),
    }
}

/// Search via GitHub API (no auth needed, 10 req/min).
/// Searches repos + README content. Good for code/library queries.
async fn search_github(query: &str, max_results: usize) -> Result<ToolResult> {
    let encoded = urlencoded(query);
    let url = format!(
        "https://api.github.com/search/repositories?q={encoded}&per_page={max_results}&sort=stars"
    );

    let client = reqwest::Client::builder()
        .user_agent("miniswe/0.1")
        .build()?;

    // Use gh CLI token if available (30/min vs 10/min unauthenticated)
    let mut req = client
        .get(&url)
        .header("Accept", "application/vnd.github.v3+json");
    if let Ok(token) = get_gh_token() {
        req = req.header("Authorization", format!("Bearer {token}"));
    }

    match req.send().await
    {
        Ok(resp) => {
            if resp.status() == 403 || resp.status() == 429 {
                return Ok(ToolResult::err(
                    "GitHub search rate limited (10/min unauthenticated). Wait a minute or set a Serper API key.".into()
                ));
            }
            let data: Value = resp.json().await.unwrap_or_default();

            let mut output = format!("[SEARCH:\"{query}\" via GitHub]\n");
            let mut count = 0;

            if let Some(items) = data["items"].as_array() {
                for item in items.iter().take(max_results) {
                    let name = item["full_name"].as_str().unwrap_or("");
                    let desc = item["description"].as_str().unwrap_or("");
                    let url = item["html_url"].as_str().unwrap_or("");
                    let stars = item["stargazers_count"].as_u64().unwrap_or(0);

                    if !name.is_empty() {
                        count += 1;
                        output.push_str(&format!(
                            "{}. {} ({stars}★)\n   {}\n   \"{}\"\n",
                            count, name, url, desc
                        ));
                    }
                }
            }

            if count == 0 {
                Ok(ToolResult::ok(format!(
                    "[search: \"{query}\" — no GitHub repos found]\n\
                     Tip: GitHub search only finds repositories, not web content.\n\
                     For documentation, use web_fetch on the URL directly, e.g.:\n\
                     web_fetch(\"https://docs.rs/CRATE\") or web_fetch(\"https://LIBRARY.dev\")"
                )))
            } else {
                output.push_str("\n(GitHub repo search — for broader web results, set SERPER_API_KEY)");
                Ok(ToolResult::ok(output))
            }
        }
        Err(e) => Ok(ToolResult::err(format!("GitHub search failed: {e}"))),
    }
}

/// web_fetch — Fetch a URL and extract content as markdown.
pub async fn fetch(args: &Value, config: &Config) -> Result<ToolResult> {
    let url = args["url"].as_str().unwrap_or("");

    if url.is_empty() {
        return Ok(ToolResult::err("Missing required parameter: url".into()));
    }

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

            let truncated = crate::truncate_chars(&content, config.tool_output_budget_chars());

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

    let mut found = String::new();
    if let Ok(entries) = fs::read_dir(&docs_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.to_lowercase().contains(&library.to_lowercase()) {
                let content = fs::read_to_string(entry.path()).unwrap_or_default();

                if topic.is_empty() {
                    found.push_str(&crate::truncate_chars(&content, config.tool_output_budget_chars()));
                } else {
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

/// Try to get a GitHub token from `gh auth token` CLI.
fn get_gh_token() -> std::result::Result<String, ()> {
    std::process::Command::new("gh")
        .args(["auth", "token"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|t| !t.is_empty())
        .ok_or(())
}

fn urlencoded(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            ' ' => '+'.to_string(),
            c if c.is_alphanumeric() || "-_.~".contains(c) => c.to_string(),
            c => format!("%{:02X}", c as u32),
        })
        .collect()
}

fn extract_relevant_sections(content: &str, keyword: &str) -> String {
    let keyword_lower = keyword.to_lowercase();
    let mut result = String::new();
    let mut in_relevant_section = false;
    let mut chars_added = 0;
    let max_chars = 8000;

    for line in content.lines() {
        let line_lower = line.to_lowercase();

        if line.starts_with('#') && line_lower.contains(&keyword_lower) {
            in_relevant_section = true;
            result.push_str(line);
            result.push('\n');
            chars_added += line.len() + 1;
            continue;
        }

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
