//! LSP server discovery, download, and management.
//!
//! Auto-detects project language, downloads the right LSP server binary
//! to `~/.miniswe/lsp-servers/`, and returns the command to spawn it.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};

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
            Self::Jdtls => vec![],  // args built in build_command()
        }
    }

    /// Find or install the server binary. Returns the path to the executable.
    pub async fn ensure_binary(&self) -> Result<PathBuf> {
        // 1. Check system PATH
        if let Some(path) = find_in_path(self.binary_name()) {
            return Ok(path);
        }

        // 2. Check our local cache
        let cache_dir = lsp_cache_dir()?;
        let cached = self.cached_binary_path(&cache_dir);
        if cached.exists() {
            return Ok(cached);
        }

        // 3. Download/install
        std::fs::create_dir_all(&cache_dir)
            .context("create lsp-servers cache dir")?;

        match self {
            Self::RustAnalyzer => download_rust_analyzer(&cache_dir).await,
            Self::Clangd => download_clangd(&cache_dir).await,
            Self::TypeScriptLanguageServer => npm_install(&cache_dir, "typescript-language-server", "typescript-language-server"),
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

/// Check if a binary exists in PATH.
fn find_in_path(name: &str) -> Option<PathBuf> {
    Command::new("which")
        .arg(name)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| {
            let path = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if path.is_empty() { None } else { Some(PathBuf::from(path)) }
        })
}

/// Check for C/C++ source files in project root (shallow).
fn has_c_sources(root: &Path) -> bool {
    let c_exts = ["c", "cc", "cpp", "cxx", "h", "hpp"];
    if let Ok(entries) = std::fs::read_dir(root) {
        for entry in entries.flatten() {
            if let Some(ext) = entry.path().extension().and_then(|e| e.to_str()) {
                if c_exts.contains(&ext) {
                    return true;
                }
            }
        }
    }
    // Also check src/ subdirectory
    let src = root.join("src");
    if let Ok(entries) = std::fs::read_dir(&src) {
        for entry in entries.flatten() {
            if let Some(ext) = entry.path().extension().and_then(|e| e.to_str()) {
                if c_exts.contains(&ext) {
                    return true;
                }
            }
        }
    }
    false
}

/// Platform identifier for GitHub release downloads.
fn platform_triple() -> &'static str {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    { "x86_64-unknown-linux-gnu" }
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    { "aarch64-unknown-linux-gnu" }
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    { "x86_64-apple-darwin" }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    { "aarch64-apple-darwin" }
    #[cfg(not(any(
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "linux", target_arch = "aarch64"),
        all(target_os = "macos", target_arch = "x86_64"),
        all(target_os = "macos", target_arch = "aarch64"),
    )))]
    { "unsupported" }
}

// ── Downloaders ────────────────────────────────────────────────────────

/// Download rust-analyzer binary from GitHub releases.
async fn download_rust_analyzer(cache_dir: &Path) -> Result<PathBuf> {
    let triple = platform_triple();
    if triple == "unsupported" {
        anyhow::bail!("rust-analyzer: unsupported platform");
    }

    let url = format!(
        "https://github.com/rust-lang/rust-analyzer/releases/latest/download/rust-analyzer-{triple}.gz"
    );

    eprintln!("[lsp] downloading rust-analyzer...");
    let response = reqwest::get(&url).await
        .context("download rust-analyzer")?;

    if !response.status().is_success() {
        anyhow::bail!("download failed: HTTP {}", response.status());
    }

    let compressed = response.bytes().await?;

    // Decompress gzip
    use std::io::Read;
    let mut decoder = flate2::read::GzDecoder::new(&compressed[..]);
    let mut binary = Vec::new();
    decoder.read_to_end(&mut binary)
        .context("decompress rust-analyzer")?;

    let dest = cache_dir.join("rust-analyzer");
    std::fs::write(&dest, &binary)?;

    // Make executable
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&dest, std::fs::Permissions::from_mode(0o755))?;
    }

    eprintln!("[lsp] rust-analyzer installed to {}", dest.display());
    Ok(dest)
}

/// Download clangd binary from GitHub releases.
async fn download_clangd(cache_dir: &Path) -> Result<PathBuf> {
    let platform = if cfg!(target_os = "linux") {
        "linux"
    } else if cfg!(target_os = "macos") {
        "mac"
    } else {
        anyhow::bail!("clangd: unsupported platform");
    };

    // Use the clangd releases API to find the latest version
    let api_url = "https://api.github.com/repos/clangd/clangd/releases/latest";
    let client = reqwest::Client::new();
    let release: serde_json::Value = client.get(api_url)
        .header("User-Agent", "miniswe")
        .send().await?
        .json().await?;

    let assets = release["assets"].as_array()
        .context("no assets in release")?;

    let asset = assets.iter()
        .find(|a| {
            a["name"].as_str()
                .is_some_and(|n| n.contains(platform) && n.ends_with(".zip"))
        })
        .context("no matching clangd asset for platform")?;

    let download_url = asset["browser_download_url"].as_str()
        .context("no download URL")?;

    eprintln!("[lsp] downloading clangd...");
    let response = reqwest::get(download_url).await?;
    let zip_bytes = response.bytes().await?;

    // Extract zip
    let cursor = std::io::Cursor::new(&zip_bytes);
    let mut archive = zip::ZipArchive::new(cursor)
        .context("open clangd zip")?;

    // Find the clangd binary inside the archive
    let mut clangd_data = None;
    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        if file.name().ends_with("/bin/clangd") || file.name() == "clangd" {
            let mut buf = Vec::new();
            std::io::Read::read_to_end(&mut file, &mut buf)?;
            clangd_data = Some(buf);
            break;
        }
    }

    let data = clangd_data.context("clangd binary not found in zip")?;
    let dest = cache_dir.join("clangd");
    std::fs::write(&dest, &data)?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&dest, std::fs::Permissions::from_mode(0o755))?;
    }

    eprintln!("[lsp] clangd installed to {}", dest.display());
    Ok(dest)
}

/// Install an npm-based LSP server locally.
fn npm_install(cache_dir: &Path, package: &str, binary_name: &str) -> Result<PathBuf> {
    // Check if npm/node is available
    if find_in_path("npm").is_none() {
        anyhow::bail!("{package}: npm not found — install Node.js to use this LSP server");
    }

    eprintln!("[lsp] npm install {package}...");

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
fn go_install(cache_dir: &Path) -> Result<PathBuf> {
    if find_in_path("go").is_none() {
        anyhow::bail!("gopls: Go not found — install Go to use this LSP server");
    }

    let gobin = cache_dir.join("gobin");
    std::fs::create_dir_all(&gobin)?;

    eprintln!("[lsp] go install gopls...");
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

/// Download Eclipse JDT Language Server.
/// jdtls ships as a tar.gz with a `bin/jdtls` launcher script.
async fn download_jdtls(cache_dir: &Path) -> Result<PathBuf> {
    if find_in_path("java").is_none() {
        anyhow::bail!("jdtls: Java not found — install a JDK to use Java LSP");
    }

    let platform = if cfg!(target_os = "linux") {
        "linux"
    } else if cfg!(target_os = "macos") {
        "mac"
    } else {
        anyhow::bail!("jdtls: unsupported platform");
    };

    let arch = if cfg!(target_arch = "x86_64") {
        "x86_64"
    } else if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else {
        anyhow::bail!("jdtls: unsupported architecture");
    };

    // Find latest milestone from Eclipse download site
    let api_url = "https://api.github.com/repos/eclipse-jdtls/eclipse.jdt.ls/releases/latest";
    let client = reqwest::Client::new();
    let release: serde_json::Value = client.get(api_url)
        .header("User-Agent", "miniswe")
        .send().await?
        .json().await?;

    let assets = release["assets"].as_array()
        .context("no assets in jdtls release")?;

    // Look for the platform-specific tar.gz
    let search = format!("{platform}-{arch}");
    let asset = assets.iter()
        .find(|a| {
            a["name"].as_str()
                .is_some_and(|n| n.contains(&search) && n.ends_with(".tar.gz"))
        })
        .or_else(|| {
            // Fall back to generic tar.gz
            assets.iter().find(|a| {
                a["name"].as_str()
                    .is_some_and(|n| n.ends_with(".tar.gz") && !n.contains("source"))
            })
        })
        .context("no matching jdtls asset")?;

    let download_url = asset["browser_download_url"].as_str()
        .context("no download URL")?;

    eprintln!("[lsp] downloading jdtls...");
    let response = reqwest::get(download_url).await?;
    let tar_gz_bytes = response.bytes().await?;

    // Extract tar.gz
    let jdtls_dir = cache_dir.join("jdtls");
    std::fs::create_dir_all(&jdtls_dir)?;

    let decoder = flate2::read::GzDecoder::new(&tar_gz_bytes[..]);
    let mut archive = tar::Archive::new(decoder);
    archive.unpack(&jdtls_dir)
        .context("extract jdtls tar.gz")?;

    // The bin/jdtls launcher script needs to be executable
    let launcher = jdtls_dir.join("bin").join("jdtls");
    if launcher.exists() {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&launcher, std::fs::Permissions::from_mode(0o755))?;
        }
        eprintln!("[lsp] jdtls installed to {}", launcher.display());
        Ok(launcher)
    } else {
        // Some versions have the launcher jar directly — create a wrapper script
        let plugins_dir = jdtls_dir.join("plugins");
        let launcher_jar = std::fs::read_dir(&plugins_dir)?
            .filter_map(|e| e.ok())
            .find(|e| {
                e.file_name().to_string_lossy().starts_with("org.eclipse.equinox.launcher_")
                    && e.file_name().to_string_lossy().ends_with(".jar")
            })
            .context("no equinox launcher jar found in jdtls")?;

        let config_dir = if cfg!(target_os = "linux") {
            jdtls_dir.join("config_linux")
        } else {
            jdtls_dir.join("config_mac")
        };

        // Create a launcher script
        let script = format!(
            "#!/bin/sh\nexec java \\\n  -Declipse.application=org.eclipse.jdt.ls.core.id1 \\\n  -Dosgi.bundles.defaultStartLevel=4 \\\n  -Declipse.product=org.eclipse.jdt.ls.core.product \\\n  -Dosgi.checkConfiguration=true \\\n  -Dosgi.sharedConfiguration.area={config} \\\n  -Dosgi.sharedConfiguration.area.readOnly=true \\\n  -Dosgi.configuration.cascaded=true \\\n  -noverify \\\n  --add-modules=ALL-SYSTEM \\\n  --add-opens java.base/java.util=ALL-UNNAMED \\\n  --add-opens java.base/java.lang=ALL-UNNAMED \\\n  -jar {jar} \\\n  \"$@\"\n",
            config = config_dir.display(),
            jar = launcher_jar.path().display(),
        );

        let bin_dir = jdtls_dir.join("bin");
        std::fs::create_dir_all(&bin_dir)?;
        let script_path = bin_dir.join("jdtls");
        std::fs::write(&script_path, script)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&script_path, std::fs::Permissions::from_mode(0o755))?;
        }

        eprintln!("[lsp] jdtls installed to {}", script_path.display());
        Ok(script_path)
    }
}
