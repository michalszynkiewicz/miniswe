//! `check`
//!
//! Explicit project-wide check. In fast mode, per-edit feedback already
//! reports the current LSP error count — `check` is the deeper, slower
//! look the model reaches for when it wants to know "does the whole
//! project build?" before moving on.
//!
//! Runs the appropriate compiler (cargo check / tsc / go vet / mvn / …)
//! synchronously, with the same subprocess harness used by edit
//! orchestration's LSP fallback.

use anyhow::Result;
use serde_json::Value;

use crate::config::Config;

use super::super::ToolResult;
use super::super::cargo_check::run_check_with_timeout;

/// Detect the project's checker from marker files in `project_root`.
/// Returns `(command, args, human_name)` or None if no known toolchain.
fn detect_checker(
    project_root: &std::path::Path,
) -> Option<(&'static str, Vec<&'static str>, &'static str)> {
    if project_root.join("Cargo.toml").exists() {
        return Some((
            "cargo",
            vec!["check", "--message-format=short"],
            "cargo check",
        ));
    }
    if project_root.join("tsconfig.json").exists() {
        return Some(("npx", vec!["tsc", "--noEmit", "--pretty", "false"], "tsc"));
    }
    if project_root.join("go.mod").exists() {
        return Some(("go", vec!["vet", "./..."], "go vet"));
    }
    if project_root.join("pom.xml").exists() {
        return Some(("mvn", vec!["compile", "-q"], "mvn compile"));
    }
    if project_root.join("build.gradle").exists() {
        return Some(("gradle", vec!["compileJava", "-q"], "gradle compileJava"));
    }
    None
}

pub async fn execute(_args: &Value, config: &Config) -> Result<ToolResult> {
    let project_root = config.project_root.clone();

    let Some((cmd, args, name)) = detect_checker(&project_root) else {
        return Ok(ToolResult::ok(
            "[check] no recognized project toolchain (looked for Cargo.toml, tsconfig.json, go.mod, pom.xml, build.gradle)".into(),
        ));
    };

    let cmd_s = cmd.to_string();
    let args_s: Vec<String> = args.into_iter().map(|s| s.to_string()).collect();
    let root_for_thread = project_root.clone();

    // 60s budget — `check` is the slow, deliberate option; per-edit feedback
    // covers the fast-path. Still spawn_blocking so the runtime isn't stalled.
    let result = tokio::task::spawn_blocking(move || {
        run_check_with_timeout(&cmd_s, &args_s, &root_for_thread, 60)
    })
    .await;

    let (success, stderr) = match result {
        Ok(Some(r)) => r,
        Ok(None) => {
            return Ok(ToolResult::err(format!(
                "[{name}] failed to spawn — is `{cmd}` on PATH?"
            )));
        }
        Err(e) => {
            return Ok(ToolResult::err(format!("[{name}] task panicked: {e}")));
        }
    };

    if success {
        return Ok(ToolResult::ok(format!("[{name}] OK — no errors")));
    }

    // Surface a reasonable slice of the stderr. The model will see line:col
    // markers in the output and can follow up with read_file / replace_range.
    let relevant: Vec<&str> = stderr
        .lines()
        .filter(|l| {
            l.contains("error")
                || l.contains("Error")
                || l.contains("warning")
                || l.starts_with("  ")
                || l.starts_with("-->")
        })
        .take(40)
        .collect();

    let mut out = format!("[{name}] FAILED\n");
    if relevant.is_empty() {
        out.push_str("(no error details captured)");
    } else {
        out.push_str(&relevant.join("\n"));
    }
    Ok(ToolResult::ok(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn scratch_config(dir: &std::path::Path) -> Config {
        let mut cfg = Config::default();
        cfg.project_root = dir.to_path_buf();
        cfg
    }

    #[tokio::test]
    async fn reports_unknown_toolchain_gracefully() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = scratch_config(tmp.path());

        let r = execute(&serde_json::json!({}), &cfg).await.unwrap();
        assert!(r.success, "unknown toolchain is not a failure");
        assert!(r.content.contains("no recognized project toolchain"));
    }

    #[test]
    fn detects_cargo_project() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("Cargo.toml"), "[package]").unwrap();
        let (cmd, _args, name) = detect_checker(tmp.path()).unwrap();
        assert_eq!(cmd, "cargo");
        assert_eq!(name, "cargo check");
    }

    #[test]
    fn detects_ts_project() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("tsconfig.json"), "{}").unwrap();
        let (cmd, _args, name) = detect_checker(tmp.path()).unwrap();
        assert_eq!(cmd, "npx");
        assert_eq!(name, "tsc");
    }

    #[test]
    fn detects_go_project() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("go.mod"), "module x").unwrap();
        let (cmd, _args, name) = detect_checker(tmp.path()).unwrap();
        assert_eq!(cmd, "go");
        assert_eq!(name, "go vet");
    }

    #[test]
    fn returns_none_for_empty_dir() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(detect_checker(tmp.path()).is_none());
    }
}
