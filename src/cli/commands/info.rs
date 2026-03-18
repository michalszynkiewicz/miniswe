//! `miniswe info` — Show project info and index stats.

use anyhow::Result;

use crate::config::Config;
use crate::knowledge::ProjectIndex;
use crate::tui;

pub async fn run() -> Result<()> {
    let config = Config::load()?;

    if !config.is_initialized() {
        tui::print_error("Project not initialized. Run `miniswe init` first.");
        return Ok(());
    }

    tui::print_header("Project Info");

    // Load and display profile
    let profile_path = config.miniswe_path("profile.md");
    if let Ok(profile) = std::fs::read_to_string(&profile_path) {
        eprintln!("{profile}");
    }

    tui::print_separator();

    // Load and display index stats
    match ProjectIndex::load(&config.miniswe_dir()) {
        Ok(index) => {
            eprintln!("Index Statistics:");
            eprintln!("  Files indexed:    {}", index.total_files);
            eprintln!("  Symbols extracted: {}", index.total_symbols);
            eprintln!("  File tree entries: {}", index.file_tree.len());

            // Top symbols by kind
            let mut kind_counts: std::collections::HashMap<&str, usize> =
                std::collections::HashMap::new();
            for symbols in index.symbols.values() {
                for sym in symbols {
                    *kind_counts.entry(sym.kind.as_str()).or_default() += 1;
                }
            }
            if !kind_counts.is_empty() {
                eprintln!("  Symbol breakdown:");
                let mut kinds: Vec<(&&str, &usize)> = kind_counts.iter().collect();
                kinds.sort_by(|a, b| b.1.cmp(a.1));
                for (kind, count) in kinds {
                    eprintln!("    {kind}: {count}");
                }
            }
        }
        Err(e) => {
            tui::print_error(&format!("Failed to load index: {e}"));
        }
    }

    tui::print_separator();

    // Display config
    eprintln!("Configuration:");
    eprintln!("  Model: {} ({})", config.model.model, config.model.provider);
    eprintln!("  Endpoint: {}", config.model.endpoint);
    eprintln!("  Context window: {} tokens", config.model.context_window);
    eprintln!("  Temperature: {}", config.model.temperature);
    eprintln!(
        "  Context budget: repo_map={}, snippets={}, history={}",
        config.context.repo_map_budget,
        config.context.snippet_budget,
        config.context.history_budget
    );

    Ok(())
}
