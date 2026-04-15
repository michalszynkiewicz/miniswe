//! Platform / host-tool detection used by the LSP installers.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Platform identifier for GitHub release downloads.
pub fn platform_triple() -> &'static str {
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

/// Check if a binary exists in PATH.
pub fn find_in_path(name: &str) -> Option<PathBuf> {
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

/// Check for C/C++ source files in project root (shallow). Used to
/// decide whether `clangd` is worth spawning for projects that happen
/// to have a `CMakeLists.txt` / `Makefile` but no C source of their own.
pub fn has_c_sources(root: &Path) -> bool {
    let c_exts = ["c", "cc", "cpp", "cxx", "h", "hpp"];
    if let Ok(entries) = std::fs::read_dir(root) {
        for entry in entries.flatten() {
            if let Some(ext) = entry.path().extension().and_then(|e| e.to_str())
                && c_exts.contains(&ext)
            {
                return true;
            }
        }
    }
    // Also check src/ subdirectory
    let src = root.join("src");
    if let Ok(entries) = std::fs::read_dir(&src) {
        for entry in entries.flatten() {
            if let Some(ext) = entry.path().extension().and_then(|e| e.to_str())
                && c_exts.contains(&ext)
            {
                return true;
            }
        }
    }
    false
}
