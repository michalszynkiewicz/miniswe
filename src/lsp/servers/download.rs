//! HTTP downloaders for LSP binaries that ship as GitHub release
//! artifacts (rust-analyzer, clangd, jdtls).

use std::future::Future;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, anyhow};

use super::platform::{find_in_path, platform_triple};
use super::verify::{VerifyResult, verify_binary_verbose};

/// Attempts for a release-artifact download.
///
/// The failure this guards against was observed once in 38 benchmark runs: the
/// rust-analyzer download threw, `ensure_binary` propagated the error, and the
/// session then ran to completion with no LSP at all — which silently removes
/// the `refactor` tools. The model fell back to `sed -i` across 14 callsites
/// and scored 4/6, indistinguishable from an ordinary bad result.
const DOWNLOAD_ATTEMPTS: u32 = 3;

/// Per-attempt ceiling. `reqwest::get` applies no timeout whatsoever, so a
/// stalled connection hangs session startup instead of failing it.
const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(120);

/// Run `f` up to [`DOWNLOAD_ATTEMPTS`] times with exponential backoff.
///
/// The whole install is retried, not just the HTTP GET: a truncated body
/// surfaces as a gzip error and a corrupt artifact as a binary that won't
/// exec, so retrying the transfer alone would miss both.
async fn with_retry<T, F, Fut>(what: &str, mut f: F) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    let mut last_err: Option<anyhow::Error> = None;

    for attempt in 1..=DOWNLOAD_ATTEMPTS {
        if attempt > 1 {
            // Back off rather than hammer. These are unauthenticated GitHub
            // release endpoints and a benchmark round hits them dozens of
            // times from a single IP, so a tight retry loop would turn a
            // transient rate-limit into a sustained one.
            let wait = Duration::from_secs(1 << (attempt - 1));
            eprintln!(
                "[lsp] {what}: retrying in {}s ({attempt}/{DOWNLOAD_ATTEMPTS})",
                wait.as_secs()
            );
            tokio::time::sleep(wait).await;
        }

        match f().await {
            Ok(value) => return Ok(value),
            Err(e) => {
                eprintln!("[lsp] {what}: attempt {attempt}/{DOWNLOAD_ATTEMPTS} failed: {e:#}");
                last_err = Some(e);
            }
        }
    }

    Err(last_err.unwrap_or_else(|| anyhow!("no attempts made")))
        .with_context(|| format!("{what}: giving up after {DOWNLOAD_ATTEMPTS} attempts"))
}

/// GET `url` under [`DOWNLOAD_TIMEOUT`], returning the body only on 2xx.
async fn fetch_bytes(url: &str) -> Result<Vec<u8>> {
    let client = reqwest::Client::builder()
        .timeout(DOWNLOAD_TIMEOUT)
        .build()
        .context("build http client")?;

    let response = client
        .get(url)
        .header("User-Agent", "miniswe")
        .send()
        .await
        .context("http request")?;

    let status = response.status();
    if !status.is_success() {
        anyhow::bail!("HTTP {status}");
    }

    Ok(response
        .bytes()
        .await
        .context("read response body")?
        .to_vec())
}

/// Download rust-analyzer binary from GitHub releases.
pub async fn download_rust_analyzer(cache_dir: &Path) -> Result<PathBuf> {
    let triple = platform_triple();
    if triple == "unsupported" {
        anyhow::bail!("rust-analyzer: unsupported platform");
    }

    let url = format!(
        "https://github.com/rust-lang/rust-analyzer/releases/latest/download/rust-analyzer-{triple}.gz"
    );

    let dest = with_retry("rust-analyzer", || install_rust_analyzer(&url, cache_dir)).await?;

    eprintln!("[lsp] rust-analyzer installed to {}", dest.display());
    Ok(dest)
}

/// One download → decompress → write → verify cycle, retried as a unit.
async fn install_rust_analyzer(url: &str, cache_dir: &Path) -> Result<PathBuf> {
    let compressed = fetch_bytes(url).await?;

    // Decompress gzip
    use std::io::Read;
    let mut decoder = flate2::read::GzDecoder::new(&compressed[..]);
    let mut binary = Vec::new();
    decoder
        .read_to_end(&mut binary)
        .context("decompress rust-analyzer")?;

    let dest = cache_dir.join("rust-analyzer");
    std::fs::write(&dest, &binary).with_context(|| format!("write {}", dest.display()))?;

    // Make executable
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&dest, std::fs::Permissions::from_mode(0o755))
            .context("chmod rust-analyzer")?;
    }

    // Exec it once before declaring success. `ensure_binary` returns anything
    // in the cache without re-verifying it, so a truncated or corrupt artifact
    // written here would be trusted as known-good on every later startup —
    // remove it and let the retry fetch a clean copy.
    if let VerifyResult::Failed { reason } = verify_binary_verbose(&dest, &["--version"]) {
        let _ = std::fs::remove_file(&dest);
        anyhow::bail!("downloaded rust-analyzer does not run ({reason})");
    }

    Ok(dest)
}

/// Download clangd binary from GitHub releases.
pub async fn download_clangd(cache_dir: &Path) -> Result<PathBuf> {
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

/// Download Eclipse JDT Language Server.
/// jdtls ships as a tar.gz with a `bin/jdtls` launcher script.
pub async fn download_jdtls(cache_dir: &Path) -> Result<PathBuf> {
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
