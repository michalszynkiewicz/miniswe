//! search tool — regex code search backed by the `grep-*` crates.
//!
//! Uses the same search kernel as ripgrep (matcher + searcher + regex
//! engine) so we don't depend on any external `rg` / `grep` binary being
//! present. Directory walking goes through `ignore`, which respects
//! `.gitignore`, `.ignore`, and hidden-file rules the same way ripgrep
//! does by default.

use anyhow::Result;
use grep_regex::{RegexMatcher, RegexMatcherBuilder};
use grep_searcher::{Searcher, SearcherBuilder, Sink, SinkMatch};
use ignore::WalkBuilder;
use serde_json::Value;
use std::path::Path;

use super::ToolResult;
use crate::config::Config;

/// File extensions we search by default. Matches the historical
/// `--type-add code:*.{…}` list from the shell-out implementation.
const CODE_EXTENSIONS: &[&str] = &[
    "rs", "py", "js", "ts", "tsx", "jsx", "go", "java", "c", "cpp", "h", "hpp", "rb", "php",
    "swift", "kt", "scala", "zig", "hs", "ml", "ex", "exs", "clj", "sh", "bash", "zsh", "toml",
    "yaml", "yml", "json", "md",
];

pub async fn execute(args: &Value, config: &Config) -> Result<ToolResult> {
    let query = match super::args::opt_str(args, "query") {
        Ok(s) => s.unwrap_or(""),
        Err(e) => return Ok(ToolResult::err(e)),
    };
    let pattern = match super::args::opt_str(args, "pattern") {
        Ok(s) => s.unwrap_or(""),
        Err(e) => return Ok(ToolResult::err(e)),
    };

    // Exactly one of query (literal) or pattern (regex) must be provided.
    let (search_term, literal) = if !query.is_empty() {
        (query, true)
    } else if !pattern.is_empty() {
        (pattern, false)
    } else {
        return Ok(ToolResult::err(
            "Provide either 'query' (plain text) or 'pattern' (regex).".into(),
        ));
    };

    let max_results = match super::args::opt_u64(args, "max_results") {
        Ok(n) => n.unwrap_or(20) as usize,
        Err(e) => return Ok(ToolResult::err(e)),
    };
    let scope = match super::args::opt_str(args, "scope") {
        Ok(s) => s.unwrap_or("project"),
        Err(e) => return Ok(ToolResult::err(e)),
    };

    let search_dir = match scope {
        "project" | "symbols" => config.project_root.clone(),
        dir => {
            let path = config.project_root.join(dir);
            if path.is_dir() {
                path
            } else {
                config.project_root.clone()
            }
        }
    };

    let matcher = match build_matcher(search_term, literal) {
        Ok(m) => m,
        Err(e) => {
            return Ok(ToolResult::err(format!(
                "Invalid {} '{}': {e}",
                if literal { "query" } else { "pattern" },
                search_term
            )));
        }
    };

    // Searching is a synchronous, I/O-bound walk — push it onto a blocking
    // pool so we don't stall the tokio runtime on large trees.
    let search_term_owned = search_term.to_string();
    let hits = tokio::task::spawn_blocking(move || run_search(&search_dir, &matcher, max_results))
        .await
        .map_err(|e| anyhow::anyhow!("search task panicked: {e}"))?;

    if hits.is_empty() {
        return Ok(ToolResult::ok(format!(
            "No matches found for: {search_term_owned}"
        )));
    }

    let match_count = hits.len();
    let mut output = format!("[search \"{search_term_owned}\": {match_count} matches]\n");
    for (i, line) in hits.iter().enumerate() {
        if i > 0 {
            output.push('\n');
        }
        output.push_str(line);
    }
    Ok(ToolResult::ok(output))
}

fn build_matcher(
    term: &str,
    literal: bool,
) -> std::result::Result<RegexMatcher, grep_regex::Error> {
    if literal {
        RegexMatcherBuilder::new().fixed_strings(true).build(term)
    } else {
        RegexMatcher::new(term)
    }
}

/// Walk the search dir and collect up to `max_results` matches as
/// `relative_path:line_number:line_text` strings.
fn run_search(root: &Path, matcher: &RegexMatcher, max_results: usize) -> Vec<String> {
    let mut hits: Vec<String> = Vec::new();
    let mut searcher = SearcherBuilder::new().line_number(true).build();

    let walker = WalkBuilder::new(root).build();
    for entry in walker {
        if hits.len() >= max_results {
            break;
        }
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if !CODE_EXTENSIONS.contains(&ext) {
            continue;
        }

        let rel_path = path
            .strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .to_string();

        let sink = FileSink {
            rel_path,
            hits: &mut hits,
            max_results,
        };
        // Errors on individual files (binary file, permission denied, bad
        // UTF-8) aren't fatal — skip and keep going.
        let _ = searcher.search_path(matcher, path, sink);
    }

    hits
}

struct FileSink<'a> {
    rel_path: String,
    hits: &'a mut Vec<String>,
    max_results: usize,
}

impl Sink for FileSink<'_> {
    type Error = std::io::Error;

    fn matched(
        &mut self,
        _searcher: &Searcher,
        m: &SinkMatch<'_>,
    ) -> std::result::Result<bool, Self::Error> {
        if self.hits.len() >= self.max_results {
            return Ok(false);
        }
        let line_number = m.line_number().unwrap_or(0);
        let text = std::str::from_utf8(m.bytes())
            .unwrap_or("")
            .trim_end_matches('\n')
            .trim_end_matches('\r');
        self.hits
            .push(format!("{}:{}:{}", self.rel_path, line_number, text));
        // Keep searching this file unless the global cap was reached.
        Ok(self.hits.len() < self.max_results)
    }
}
