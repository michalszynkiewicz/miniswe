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

pub async fn execute(_args: &Value, _config: &Config) -> Result<ToolResult> {
    todo!(
        "check — pick checker by project markers (Cargo.toml, tsconfig.json, \
         go.mod, pyproject, pom, gradle), run via super::cargo_check::\
         run_check_with_timeout, summarize errors with source context"
    )
}
