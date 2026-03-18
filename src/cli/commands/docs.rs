//! `miniswe docs` — Manage documentation cache.

use std::fs;

use anyhow::Result;

use crate::cli::DocsSubcommand;
use crate::config::Config;
use crate::tui;

pub async fn run(subcommand: DocsSubcommand) -> Result<()> {
    let config = Config::load()?;
    let docs_dir = config.miniswe_path("docs");
    fs::create_dir_all(&docs_dir)?;

    match subcommand {
        DocsSubcommand::Add { url } => {
            tui::print_status(&format!("Fetching {url}..."));

            let client = reqwest::Client::builder()
                .user_agent("Mozilla/5.0 (compatible; miniswe/0.1)")
                .build()?;

            match client.get(&url).send().await {
                Ok(response) => {
                    let content = response.text().await?;

                    // Derive filename from URL
                    let filename = url
                        .trim_end_matches('/')
                        .rsplit('/')
                        .next()
                        .unwrap_or("docs.txt")
                        .replace(['?', '&', '='], "_");

                    let doc_path = docs_dir.join(&filename);
                    fs::write(&doc_path, &content)?;

                    tui::print_complete(&format!(
                        "Cached {} ({} bytes) → {}",
                        url,
                        content.len(),
                        doc_path.display()
                    ));
                }
                Err(e) => {
                    tui::print_error(&format!("Failed to fetch {url}: {e}"));
                }
            }
        }
        DocsSubcommand::List => {
            tui::print_header("Cached Documentation");

            if !docs_dir.exists() {
                eprintln!("  (no docs cached)");
                return Ok(());
            }

            let mut entries: Vec<_> = fs::read_dir(&docs_dir)?
                .flatten()
                .collect();
            entries.sort_by_key(|e| e.file_name());

            if entries.is_empty() {
                eprintln!("  (no docs cached)");
            } else {
                for entry in entries {
                    let meta = entry.metadata()?;
                    let size = meta.len();
                    let name = entry.file_name().to_string_lossy().to_string();
                    eprintln!("  {name} ({size} bytes)");
                }
            }

            eprintln!();
            tui::print_status("Add docs: `miniswe docs add <url>`");
        }
        DocsSubcommand::Refresh => {
            tui::print_status("Refreshing cached docs...");
            // TODO: Store original URLs and re-fetch
            tui::print_error("Not yet implemented — re-add docs with `miniswe docs add <url>`");
        }
    }

    Ok(())
}
