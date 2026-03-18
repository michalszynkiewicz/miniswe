//! `minime init` — Initialize the project knowledge base.

use std::fs;

use anyhow::Result;

use crate::config::Config;
use crate::knowledge::graph::{self, DependencyGraph};
use crate::knowledge::indexer;
use crate::knowledge::profile;
use crate::knowledge::ts_extract;
use crate::tui;

pub async fn run() -> Result<()> {
    let root = std::env::current_dir()?;

    tui::print_header("Initializing minime");

    // Check if already initialized
    let minime_dir = root.join(".minime");
    if minime_dir.exists() {
        tui::print_status("Project already initialized. Re-indexing...");
    }

    // Create directory structure
    tui::print_status("Creating .minime/ directory structure...");
    fs::create_dir_all(minime_dir.join("index"))?;
    fs::create_dir_all(minime_dir.join("snippets"))?;
    fs::create_dir_all(minime_dir.join("sessions"))?;
    fs::create_dir_all(minime_dir.join("docs"))?;

    // Detect project and generate profile
    tui::print_status("Detecting project configuration...");
    let info = profile::detect_project(&root)?;
    let profile_content = profile::generate_profile(&info);
    let profile_path = minime_dir.join("profile.md");
    fs::write(&profile_path, &profile_content)?;
    tui::print_status(&format!("  Generated profile: {}", info.name));

    // Create default config if it doesn't exist
    let config_path = minime_dir.join("config.toml");
    if !config_path.exists() {
        let config = Config::default();
        let config_content = toml::to_string_pretty(&config)?;
        fs::write(&config_path, config_content)?;
        tui::print_status("  Created default config.toml");
    }

    // Create guide.md if it doesn't exist
    let guide_path = minime_dir.join("guide.md");
    if !guide_path.exists() {
        fs::write(
            &guide_path,
            "# Project Guide\n\n\
             <!-- Add project-specific instructions here (keep under 500 tokens) -->\n\
             <!-- These will be included in every LLM context -->\n",
        )?;
        tui::print_status("  Created guide.md (edit this with project-specific tips)");
    }

    // Create lessons.md if it doesn't exist
    let lessons_path = minime_dir.join("lessons.md");
    if !lessons_path.exists() {
        fs::write(
            &lessons_path,
            "# Lessons\n\n\
             <!-- Accumulated tips from past sessions -->\n\
             <!-- Use `minime learn \"tip\"` to add entries -->\n",
        )?;
        tui::print_status("  Created lessons.md");
    }

    // Report tree-sitter status
    let ts_langs = ts_extract::enabled_languages();
    if ts_langs.is_empty() {
        tui::print_status("Parser: regex (tree-sitter not enabled)");
    } else {
        tui::print_status(&format!(
            "Parser: tree-sitter ({})",
            ts_langs.join(", ")
        ));
    }

    // Run indexer
    tui::print_status("Indexing project files...");
    let mut index = indexer::index_project(&root)?;

    // Populate cross-references
    tui::print_status("Building dependency graph...");
    graph::populate_symbol_deps(&mut index);
    let dep_graph = DependencyGraph::build(&index);
    let edge_count: usize = dep_graph.edges.values().map(|v| v.len()).sum();

    index.save(&minime_dir)?;
    dep_graph.save(&minime_dir)?;

    tui::print_complete(&format!(
        "Indexed {} files, {} symbols, {} cross-references",
        index.total_files, index.total_symbols, edge_count
    ));

    // Create .gitignore for .minime/
    let gitignore_path = minime_dir.join(".gitignore");
    if !gitignore_path.exists() {
        fs::write(
            &gitignore_path,
            "# Auto-generated — commit profile.md, guide.md, lessons.md\n\
             # Ignore everything else\n\
             index/\n\
             snippets/\n\
             sessions/\n\
             scratchpad.md\n\
             plan.md\n\
             docs/\n",
        )?;
    }

    // Audit file sizes
    let large_files = indexer::audit_file_sizes(&root, 200);
    if !large_files.is_empty() {
        tui::print_separator();
        tui::print_status(&format!(
            "⚠ {} files exceed 200 lines (consider splitting for better agent performance):",
            large_files.len()
        ));
        for (file, lines) in large_files.iter().take(10) {
            tui::print_status(&format!("  {file}: {lines} lines"));
        }
        if large_files.len() > 10 {
            tui::print_status(&format!("  ... and {} more", large_files.len() - 10));
        }
    }

    tui::print_separator();
    tui::print_status("Review and edit:");
    tui::print_status(&format!(
        "  {} (project profile)",
        profile_path.display()
    ));
    tui::print_status(&format!(
        "  {} (custom instructions)",
        guide_path.display()
    ));
    tui::print_status(&format!(
        "  {} (model & context settings)",
        config_path.display()
    ));

    Ok(())
}
