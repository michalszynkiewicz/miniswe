//! `minime init` — Initialize the project knowledge base.

use std::fs;

use anyhow::Result;

use crate::config::Config;
use crate::knowledge::indexer;
use crate::knowledge::profile;
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

    // Run indexer
    tui::print_status("Indexing project files...");
    let index = indexer::index_project(&root)?;
    index.save(&minime_dir)?;

    tui::print_complete(&format!(
        "Indexed {} files, {} symbols",
        index.total_files, index.total_symbols
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
