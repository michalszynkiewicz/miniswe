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

/// Result of verifying that a binary on disk actually runs.
#[derive(Debug, PartialEq, Eq)]
enum VerifyResult {
    Ok,
    Failed { reason: String },
}

/// Verify a binary actually works by running it with version args.
/// Captures stderr/stdout so failures can be explained instead of swallowed.
fn verify_binary_verbose(path: &Path, args: &[&str]) -> VerifyResult {
    let output = match Command::new(path)
        .args(args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
    {
        Ok(out) => out,
        Err(e) => {
            return VerifyResult::Failed {
                reason: format!("spawn failed: {e}"),
            };
        }
    };

    if output.status.success() {
        return VerifyResult::Ok;
    }

    // Surface the first useful line of stderr so the user can diagnose
    // (rustup proxy errors look like: "error: 'rust-analyzer' is not installed for the toolchain ...").
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let snippet = stderr
        .lines()
        .chain(stdout.lines())
        .find(|l| !l.trim().is_empty())
        .unwrap_or("no output")
        .trim()
        .chars()
        .take(160)
        .collect::<String>();

    let exit = output
        .status
        .code()
        .map(|c| format!("exit {c}"))
        .unwrap_or_else(|| "killed by signal".to_string());

    VerifyResult::Failed {
        reason: format!("{exit}: {snippet}"),
    }
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
            if path.is_empty() {
                None
            } else {
                Some(PathBuf::from(path))
            }
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
    {
        "x86_64-unknown-linux-gnu"
    }
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        "aarch64-unknown-linux-gnu"
    }
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    {
        "x86_64-apple-darwin"
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        "aarch64-apple-darwin"
    }
    #[cfg(not(any(
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "linux", target_arch = "aarch64"),
        all(target_os = "macos", target_arch = "x86_64"),
        all(target_os = "macos", target_arch = "aarch64"),
    )))]
    {
        "unsupported"
    }
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

    let response = reqwest::get(&url).await.context("download rust-analyzer")?;

    if !response.status().is_success() {
        anyhow::bail!("download failed: HTTP {}", response.status());
    }

    let compressed = response.bytes().await?;

    // Decompress gzip
    use std::io::Read;
    let mut decoder = flate2::read::GzDecoder::new(&compressed[..]);
    let mut binary = Vec::new();
    decoder
        .read_to_end(&mut binary)
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
    let release: serde_json::Value = client
        .get(api_url)
        .header("User-Agent", "miniswe")
        .send()
        .await?
        .json()
        .await?;

    let assets = release["assets"]
        .as_array()
        .context("no assets in release")?;

    let asset = assets
        .iter()
        .find(|a| {
            a["name"]
                .as_str()
                .is_some_and(|n| n.contains(platform) && n.ends_with(".zip"))
        })
        .context("no matching clangd asset for platform")?;

    let download_url = asset["browser_download_url"]
        .as_str()
        .context("no download URL")?;

    let response = reqwest::get(download_url).await?;
    let zip_bytes = response.bytes().await?;

    // Extract zip
    let cursor = std::io::Cursor::new(&zip_bytes);
    let mut archive = zip::ZipArchive::new(cursor).context("open clangd zip")?;

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
    let release: serde_json::Value = client
        .get(api_url)
        .header("User-Agent", "miniswe")
        .send()
        .await?
        .json()
        .await?;

    let assets = release["assets"]
        .as_array()
        .context("no assets in jdtls release")?;

    // Look for the platform-specific tar.gz
    let search = format!("{platform}-{arch}");
    let asset = assets
        .iter()
        .find(|a| {
            a["name"]
                .as_str()
                .is_some_and(|n| n.contains(&search) && n.ends_with(".tar.gz"))
        })
        .or_else(|| {
            // Fall back to generic tar.gz
            assets.iter().find(|a| {
                a["name"]
                    .as_str()
                    .is_some_and(|n| n.ends_with(".tar.gz") && !n.contains("source"))
            })
        })
        .context("no matching jdtls asset")?;

    let download_url = asset["browser_download_url"]
        .as_str()
        .context("no download URL")?;

    let response = reqwest::get(download_url).await?;
    let tar_gz_bytes = response.bytes().await?;

    // Extract tar.gz
    let jdtls_dir = cache_dir.join("jdtls");
    std::fs::create_dir_all(&jdtls_dir)?;

    let decoder = flate2::read::GzDecoder::new(&tar_gz_bytes[..]);
    let mut archive = tar::Archive::new(decoder);
    archive.unpack(&jdtls_dir).context("extract jdtls tar.gz")?;

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
                e.file_name()
                    .to_string_lossy()
                    .starts_with("org.eclipse.equinox.launcher_")
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

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    // We deliberately do *not* write our own scripts and exec them
    // here. Doing so in a multi-threaded `cargo test` run is racy on
    // Linux: a concurrent fork in another test can inherit our write
    // FD and trip `ETXTBSY` when the child tries to exec. Driving a
    // pre-existing shell (`/bin/sh -c '…'`) skips the write step
    // entirely and the tests become completely deterministic.

    #[test]
    fn verify_binary_ok_on_zero_exit() {
        // `sh -c 'exit 0'` is the smallest "binary runs and returns
        // success" shape we can construct without creating files.
        assert_eq!(
            verify_binary_verbose(Path::new("/bin/sh"), &["-c", "echo 1.2.3; exit 0"]),
            VerifyResult::Ok
        );
    }

    #[test]
    fn verify_binary_failed_captures_stderr_snippet() {
        // Mimic a rustup proxy failure: nonzero exit + stderr line
        // that the user will want to see verbatim in logs.
        let rustup_msg = "error: 'rust-analyzer' is not installed for the toolchain 'stable-x86_64-unknown-linux-gnu'";
        match verify_binary_verbose(
            Path::new("/bin/sh"),
            &["-c", &format!("echo \"{rustup_msg}\" >&2; exit 1")],
        ) {
            VerifyResult::Ok => panic!("expected failure"),
            VerifyResult::Failed { reason } => {
                assert!(reason.contains("exit 1"), "reason was: {reason}");
                assert!(
                    reason.contains("rust-analyzer") && reason.contains("not installed"),
                    "reason was: {reason}"
                );
            }
        }
    }

    #[test]
    fn verify_binary_failed_when_missing_executable() {
        let result = verify_binary_verbose(
            Path::new("/nonexistent/definitely-missing-binary"),
            &["--version"],
        );
        match result {
            VerifyResult::Ok => panic!("expected failure for missing binary"),
            VerifyResult::Failed { reason } => {
                assert!(reason.starts_with("spawn failed"), "reason was: {reason}");
            }
        }
    }

    #[test]
    fn verify_binary_truncates_long_lines() {
        // 500 chars of 'x' on stderr — the snippet that ends up in
        // the reason string should be truncated to ≤160 chars.
        match verify_binary_verbose(
            Path::new("/bin/sh"),
            &["-c", "printf 'x%.0s' $(seq 1 500) >&2; exit 2"],
        ) {
            VerifyResult::Ok => panic!("expected failure"),
            VerifyResult::Failed { reason } => {
                // exit prefix + ": " + up-to-160-char snippet
                assert!(reason.starts_with("exit 2: "), "reason was: {reason}");
                let snippet = &reason["exit 2: ".len()..];
                assert!(snippet.len() <= 160, "snippet too long: {}", snippet.len());
            }
        }
    }
}
