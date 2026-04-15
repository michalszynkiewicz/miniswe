//! Project profile auto-generation.
//!
//! Scans the project for config files (Cargo.toml, package.json, etc.)
//! and generates `.miniswe/profile.md` — a compressed overview of the project.

use std::path::Path;

use anyhow::Result;

/// Detected project information.
#[derive(Debug, Default)]
pub struct ProjectInfo {
    pub name: String,
    pub languages: Vec<(String, f32)>, // (language, percentage)
    pub framework: Option<String>,
    pub package_manager: Option<String>,
    pub test_runner: Option<String>,
    pub build_cmd: Option<String>,
    pub test_cmd: Option<String>,
    pub lint_cmd: Option<String>,
    pub entry_points: Vec<String>,
    pub description: Option<String>,
}

/// Detect project info by scanning config files.
pub fn detect_project(root: &Path) -> Result<ProjectInfo> {
    let mut info = ProjectInfo::default();

    // Detect from Cargo.toml (Rust)
    let cargo_toml = root.join("Cargo.toml");
    if cargo_toml.exists()
        && let Ok(content) = std::fs::read_to_string(&cargo_toml)
        && let Ok(parsed) = content.parse::<toml::Table>()
    {
        if let Some(pkg) = parsed.get("package").and_then(|v| v.as_table()) {
            info.name = pkg
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            info.description = pkg
                .get("description")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
        }
        info.languages.push(("Rust".into(), 100.0));
        info.package_manager = Some("cargo".into());
        info.build_cmd = Some("cargo build".into());
        info.test_cmd = Some("cargo test".into());
        info.lint_cmd = Some("cargo clippy".into());
    }

    // Detect from package.json (JavaScript/TypeScript)
    let package_json = root.join("package.json");
    if package_json.exists()
        && let Ok(content) = std::fs::read_to_string(&package_json)
        && let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&content)
    {
        info.name = parsed["name"].as_str().unwrap_or("unknown").to_string();
        info.description = parsed["description"].as_str().map(|s| s.to_string());

        // Detect package manager
        if root.join("pnpm-lock.yaml").exists() {
            info.package_manager = Some("pnpm".into());
        } else if root.join("yarn.lock").exists() {
            info.package_manager = Some("yarn".into());
        } else {
            info.package_manager = Some("npm".into());
        }

        // Detect scripts
        if let Some(scripts) = parsed["scripts"].as_object() {
            let pm = info.package_manager.as_deref().unwrap_or("npm");
            if scripts.contains_key("build") {
                info.build_cmd = Some(format!("{pm} run build"));
            }
            if scripts.contains_key("test") {
                info.test_cmd = Some(format!("{pm} test"));
            }
            if scripts.contains_key("lint") {
                info.lint_cmd = Some(format!("{pm} run lint"));
            }
        }

        // Detect TypeScript
        if root.join("tsconfig.json").exists() {
            info.languages.push(("TypeScript".into(), 90.0));
            info.languages.push(("JavaScript".into(), 10.0));
        } else {
            info.languages.push(("JavaScript".into(), 100.0));
        }
    }

    // Detect from go.mod (Go)
    let go_mod = root.join("go.mod");
    if go_mod.exists()
        && let Ok(content) = std::fs::read_to_string(&go_mod)
    {
        for line in content.lines() {
            if line.starts_with("module ") {
                info.name = line
                    .strip_prefix("module ")
                    .unwrap_or("unknown")
                    .trim()
                    .to_string();
                break;
            }
        }
        info.languages.push(("Go".into(), 100.0));
        info.build_cmd = Some("go build ./...".into());
        info.test_cmd = Some("go test ./...".into());
        info.lint_cmd = Some("golangci-lint run".into());
    }

    // Detect from pyproject.toml (Python)
    let pyproject = root.join("pyproject.toml");
    if pyproject.exists()
        && let Ok(content) = std::fs::read_to_string(&pyproject)
        && let Ok(parsed) = content.parse::<toml::Table>()
    {
        if let Some(project) = parsed.get("project").and_then(|v| v.as_table()) {
            info.name = project
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            info.description = project
                .get("description")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
        }
        info.languages.push(("Python".into(), 100.0));
        info.test_cmd = Some("pytest".into());
        info.lint_cmd = Some("ruff check".into());
    }

    // Fallback name from directory
    if info.name.is_empty() {
        info.name = root
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("project")
            .to_string();
    }

    // Detect entry points
    for candidate in &[
        "src/main.rs",
        "src/lib.rs",
        "src/index.ts",
        "src/index.js",
        "src/app.ts",
        "src/app.js",
        "main.go",
        "cmd/main.go",
        "src/main.py",
        "app.py",
        "manage.py",
    ] {
        if root.join(candidate).exists() {
            info.entry_points.push(candidate.to_string());
        }
    }

    Ok(info)
}

/// Generate the profile.md content.
pub fn generate_profile(info: &ProjectInfo) -> String {
    let mut profile = String::new();

    profile.push_str("# Project Profile (auto-generated — edit to refine)\n\n");

    profile.push_str("## Identity\n");
    profile.push_str(&format!("- Name: {}\n", info.name));

    if !info.languages.is_empty() {
        let langs: Vec<String> = info
            .languages
            .iter()
            .map(|(lang, pct)| format!("{lang} ({pct:.0}%)"))
            .collect();
        profile.push_str(&format!("- Language: {}\n", langs.join(", ")));
    }

    if let Some(fw) = &info.framework {
        profile.push_str(&format!("- Framework: {fw}\n"));
    }
    if let Some(pm) = &info.package_manager {
        profile.push_str(&format!("- Package manager: {pm}\n"));
    }
    if let Some(desc) = &info.description {
        profile.push_str(&format!("- Description: {desc}\n"));
    }

    // Commands
    let mut cmds = Vec::new();
    if let Some(cmd) = &info.build_cmd {
        cmds.push(format!("Build: `{cmd}`"));
    }
    if let Some(cmd) = &info.test_cmd {
        cmds.push(format!("Test: `{cmd}`"));
    }
    if let Some(cmd) = &info.lint_cmd {
        cmds.push(format!("Lint: `{cmd}`"));
    }
    if !cmds.is_empty() {
        profile.push_str(&format!("- Commands: {}\n", cmds.join(" | ")));
    }

    // Entry points
    if !info.entry_points.is_empty() {
        profile.push_str("\n## Entry Points\n");
        for ep in &info.entry_points {
            profile.push_str(&format!("- {ep}\n"));
        }
    }

    profile
}
