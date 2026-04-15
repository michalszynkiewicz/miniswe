//! Local installers that shell out to a host package manager (`npm`,
//! `go install`) instead of grabbing a pre-built binary directly.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};

use super::platform::find_in_path;

/// Install an npm-based LSP server locally.
pub fn npm_install(cache_dir: &Path, package: &str, binary_name: &str) -> Result<PathBuf> {
    // Check if npm/node is available
    if find_in_path("npm").is_none() {
        anyhow::bail!("{package}: npm not found — install Node.js to use this LSP server");
    }

    // Initialize package.json if needed
    if !cache_dir.join("package.json").exists() {
        let status = Command::new("npm")
            .args(["init", "-y"])
            .current_dir(cache_dir)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()?;
        if !status.success() {
            anyhow::bail!("npm init failed");
        }
    }

    let status = Command::new("npm")
        .args(["install", "--save", package])
        .current_dir(cache_dir)
        .stdout(std::process::Stdio::null())
        .status()
        .context("npm install")?;

    if !status.success() {
        anyhow::bail!("npm install {package} failed");
    }

    let dest = cache_dir.join("node_modules/.bin").join(binary_name);
    if !dest.exists() {
        anyhow::bail!("{binary_name} not found after npm install");
    }

    eprintln!("[lsp] {package} installed to {}", dest.display());
    Ok(dest)
}

/// Install gopls via `go install`.
pub fn go_install(cache_dir: &Path) -> Result<PathBuf> {
    if find_in_path("go").is_none() {
        anyhow::bail!("gopls: Go not found — install Go to use this LSP server");
    }

    let gobin = cache_dir.join("gobin");
    std::fs::create_dir_all(&gobin)?;

    let status = Command::new("go")
        .args(["install", "golang.org/x/tools/gopls@latest"])
        .env("GOBIN", &gobin)
        .stdout(std::process::Stdio::null())
        .status()
        .context("go install gopls")?;

    if !status.success() {
        anyhow::bail!("go install gopls failed");
    }

    let dest = gobin.join("gopls");
    if !dest.exists() {
        anyhow::bail!("gopls not found after go install");
    }

    eprintln!("[lsp] gopls installed to {}", dest.display());
    Ok(dest)
}
