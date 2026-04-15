//! LSP server discovery, download, and management.
//!
//! Auto-detects project language, downloads the right LSP server binary
//! to `~/.miniswe/lsp-servers/`, and returns the command to spawn it.

mod download;
mod install;
mod platform;
mod verify;

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};

use self::download::{download_clangd, download_jdtls, download_rust_analyzer};
use self::install::{go_install, npm_install};
use self::platform::{find_in_path, has_c_sources};
use self::verify::{VerifyResult, verify_binary_verbose};

/// Supported LSP server types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LspServer {
    RustAnalyzer,
    TypeScriptLanguageServer,
    Pyright,
    Gopls,
    Clangd,
    Jdtls,
}

impl LspServer {
    /// Detect which LSP server to use based on project files.
    pub fn detect(project_root: &Path) -> Option<Self> {
        if project_root.join("Cargo.toml").exists() {
            Some(Self::RustAnalyzer)
        } else if project_root.join("tsconfig.json").exists()
            || project_root.join("package.json").exists()
        {
            Some(Self::TypeScriptLanguageServer)
        } else if project_root.join("pyproject.toml").exists()
            || project_root.join("setup.py").exists()
            || project_root.join("requirements.txt").exists()
        {
            Some(Self::Pyright)
        } else if project_root.join("go.mod").exists() {
            Some(Self::Gopls)
        } else if project_root.join("pom.xml").exists()
            || project_root.join("build.gradle").exists()
            || project_root.join("build.gradle.kts").exists()
        {
            Some(Self::Jdtls)
        } else if project_root.join("CMakeLists.txt").exists()
            || project_root.join("compile_commands.json").exists()
            || project_root.join("Makefile").exists()
        {
            // Only pick clangd if there are C/C++ source files
            if has_c_sources(project_root) {
                Some(Self::Clangd)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Human-readable name for logging.
    pub fn name(&self) -> &'static str {
        match self {
            Self::RustAnalyzer => "rust-analyzer",
            Self::TypeScriptLanguageServer => "typescript-language-server",
            Self::Pyright => "pyright",
            Self::Gopls => "gopls",
            Self::Clangd => "clangd",
            Self::Jdtls => "jdtls",
        }
    }

    /// The binary name to look for / execute.
    fn binary_name(&self) -> &'static str {
        match self {
            Self::RustAnalyzer => "rust-analyzer",
            Self::TypeScriptLanguageServer => "typescript-language-server",
            Self::Pyright => "pyright-langserver",
            Self::Gopls => "gopls",
            Self::Clangd => "clangd",
            Self::Jdtls => "jdtls",
        }
    }

    /// Command-line args to start the server in stdio mode.
    pub fn stdio_args(&self) -> Vec<&'static str> {
        match self {
            Self::RustAnalyzer => vec![],
            Self::TypeScriptLanguageServer => vec!["--stdio"],
            Self::Pyright => vec!["--stdio"],
            Self::Gopls => vec!["serve"],
            Self::Clangd => vec![],
            Self::Jdtls => vec![], // args built in build_command()
        }
    }

    /// Args to verify the binary works (e.g. --version).
    fn version_args(&self) -> &[&str] {
        match self {
            Self::RustAnalyzer => &["--version"],
            Self::TypeScriptLanguageServer => &["--version"],
            Self::Pyright => &["--version"],
            Self::Gopls => &["version"],
            Self::Clangd => &["--version"],
            Self::Jdtls => &["--version"],
        }
    }

    /// Find or install the server binary. Returns the path to the executable.
    pub async fn ensure_binary(&self) -> Result<PathBuf> {
        // 1. Check our local cache first — anything we downloaded ourselves is known-good,
        //    and avoids re-running version verification on every startup.
        let cache_dir = lsp_cache_dir()?;
        let cached = self.cached_binary_path(&cache_dir);
        if cached.exists() {
            return Ok(cached);
        }

        // 2. Check system PATH — verify the binary actually works
        //    (rustup proxies exist in PATH but fail if the component isn't installed)
        if let Some(path) = find_in_path(self.binary_name()) {
            match verify_binary_verbose(&path, self.version_args()) {
                VerifyResult::Ok => return Ok(path),
                VerifyResult::Failed { reason } => {
                    eprintln!(
                        "[lsp] {} in PATH at {} is unusable ({}), will download a working copy",
                        self.binary_name(),
                        path.display(),
                        reason
                    );
                }
            }
        }

        // 3. Download/install
        eprintln!("[lsp] downloading {}...", self.binary_name());
        std::fs::create_dir_all(&cache_dir).context("create lsp-servers cache dir")?;

        match self {
            Self::RustAnalyzer => download_rust_analyzer(&cache_dir).await,
            Self::Clangd => download_clangd(&cache_dir).await,
            Self::TypeScriptLanguageServer => npm_install(
                &cache_dir,
                "typescript-language-server",
                "typescript-language-server",
            ),
            Self::Pyright => npm_install(&cache_dir, "pyright", "pyright-langserver"),
            Self::Gopls => go_install(&cache_dir),
            Self::Jdtls => download_jdtls(&cache_dir).await,
        }
    }

    fn cached_binary_path(&self, cache_dir: &Path) -> PathBuf {
        match self {
            Self::TypeScriptLanguageServer | Self::Pyright => {
                cache_dir.join("node_modules/.bin").join(self.binary_name())
            }
            Self::Jdtls => {
                // jdtls is a directory with a launcher jar inside
                cache_dir.join("jdtls").join("bin").join("jdtls")
            }
            _ => cache_dir.join(self.binary_name()),
        }
    }

    /// Build the Command to spawn this LSP server.
    /// For most servers: binary + stdio_args.
    /// For jdtls: java with special launcher args.
    pub fn build_command(&self, binary_path: &Path, project_root: &Path) -> Command {
        match self {
            Self::Jdtls => {
                // jdtls ships a launcher script at bin/jdtls
                let mut cmd = Command::new(binary_path);
                // -data is the workspace data dir (per-project)
                let data_dir = project_root.join(".miniswe").join("jdtls-data");
                cmd.arg("-data").arg(&data_dir);
                cmd
            }
            _ => {
                let mut cmd = Command::new(binary_path);
                for arg in self.stdio_args() {
                    cmd.arg(arg);
                }
                cmd
            }
        }
    }
}

/// Base directory for cached LSP server binaries.
fn lsp_cache_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().context("no home directory")?;
    Ok(home.join(".miniswe").join("lsp-servers"))
}
