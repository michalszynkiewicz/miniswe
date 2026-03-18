//! `miniswe config` — Show/edit configuration.

use anyhow::Result;

use crate::config::Config;
use crate::tui;

pub async fn run() -> Result<()> {
    let config = Config::load()?;

    tui::print_header("Configuration");

    eprintln!("[model]");
    eprintln!("  provider = \"{}\"", config.model.provider);
    eprintln!("  endpoint = \"{}\"", config.model.endpoint);
    eprintln!("  model = \"{}\"", config.model.model);
    eprintln!("  context_window = {}", config.model.context_window);
    eprintln!("  temperature = {}", config.model.temperature);
    eprintln!("  max_output_tokens = {}", config.model.max_output_tokens);
    eprintln!();
    eprintln!("[context]");
    eprintln!("  repo_map_budget = {}", config.context.repo_map_budget);
    eprintln!("  snippet_budget = {}", config.context.snippet_budget);
    eprintln!("  history_turns = {}", config.context.history_turns);
    eprintln!("  history_budget = {}", config.context.history_budget);
    eprintln!("  scratchpad_budget = {}", config.context.scratchpad_budget);
    eprintln!();
    eprintln!("[hardware]");
    eprintln!("  vram_gb = {}", config.hardware.vram_gb);
    eprintln!("  vram_reserve_gb = {} (usable: {}GB)", config.hardware.vram_reserve_gb, config.hardware.vram_gb - config.hardware.vram_reserve_gb);
    eprintln!("  ram_budget_gb = {}", config.hardware.ram_budget_gb);
    eprintln!();
    eprintln!("[web]");
    eprintln!("  search_backend = \"{}\"", config.web.search_backend);
    eprintln!("  fetch_backend = \"{}\"", config.web.fetch_backend);

    if !config.is_initialized() {
        eprintln!();
        tui::print_status("Config file: (using defaults — run `miniswe init` to create)");
    } else {
        eprintln!();
        tui::print_status(&format!(
            "Config file: {}",
            config.miniswe_path("config.toml").display()
        ));
    }

    Ok(())
}
