//! LSP validation of edit candidates: writes the candidate, diffs error
//! counts against the baseline, and reverts on regression.

use super::*;

impl ValidationError {
    pub(super) fn summary(&self) -> String {
        match self {
            Self::LspRegression(reg) => {
                let mut out = format!(
                    "LSP diagnostics worsened: {} -> {} error(s)",
                    reg.baseline_count,
                    reg.errors.len() + reg.extra_error_count
                );
                for err in reg.errors.iter().take(5) {
                    out.push_str(&format!(
                        "\nL{}:{}: error: {}",
                        err.line, err.column, err.message
                    ));
                }
                if reg.extra_error_count > 0 {
                    out.push_str(&format!(
                        "\n... and {} more error(s)",
                        reg.extra_error_count
                    ));
                }
                out
            }
            Self::Other(e) => e.to_string(),
        }
    }
}

impl From<anyhow::Error> for ValidationError {
    fn from(e: anyhow::Error) -> Self {
        Self::Other(e)
    }
}

impl From<std::io::Error> for ValidationError {
    fn from(e: std::io::Error) -> Self {
        Self::Other(e.into())
    }
}

impl LspValidationMode {
    pub(super) fn from_args(args: &Value) -> Result<Self> {
        let mode = crate::tools::args::opt_str(args, "lsp_validation")
            .map_err(|e| anyhow::anyhow!(e))?
            .unwrap_or("auto");
        match mode {
            "auto" => Ok(Self::Auto),
            "require" => Ok(Self::Require),
            "off" => Ok(Self::Off),
            other => bail!("Invalid lsp_validation: {other}. Expected one of: auto, require, off"),
        }
    }

    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Require => "require",
            Self::Off => "off",
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn validate_candidate_for_write(
    path_str: &str,
    path: &std::path::Path,
    original: &str,
    candidate: &str,
    config: &Config,
    lsp: Option<&LspClient>,
    lsp_validation: LspValidationMode,
    cancelled: Option<&AtomicBool>,
    log: Option<&SessionLog>,
    baseline_lsp_errors: Option<usize>,
    perms: Option<&PermissionManager>,
) -> std::result::Result<Option<String>, ValidationError> {
    ensure_not_cancelled(cancelled).map_err(ValidationError::Other)?;
    validate_candidate(original, candidate).map_err(ValidationError::Other)?;
    gate_truncation(path_str, original, candidate, perms).map_err(ValidationError::Other)?;
    log_stage(log, path_str, "validate:lsp");
    validate_candidate_with_lsp(
        path_str,
        path,
        original,
        candidate,
        config,
        lsp,
        lsp_validation,
        baseline_lsp_errors,
    )
    .await
}

pub(super) async fn validate_candidate_with_lsp(
    path_str: &str,
    path: &std::path::Path,
    original: &str,
    candidate: &str,
    config: &Config,
    lsp: Option<&LspClient>,
    lsp_validation: LspValidationMode,
    baseline_lsp_errors: Option<usize>,
) -> std::result::Result<Option<String>, ValidationError> {
    if lsp_validation == LspValidationMode::Off {
        return Ok(Some("[lsp] skipped (off)".into()));
    }

    let Some(lsp) = lsp else {
        if lsp_validation == LspValidationMode::Require {
            return Err(ValidationError::Other(anyhow::anyhow!(
                "LSP validation required but no LSP client is available"
            )));
        }
        return Ok(None);
    };

    if !lsp.is_ready() || lsp.has_crashed() {
        if lsp_validation == LspValidationMode::Require {
            return Err(ValidationError::Other(anyhow::anyhow!(
                "LSP validation required but LSP is not ready"
            )));
        }
        return Ok(None);
    }

    let timeout = Duration::from_millis(config.lsp.diagnostic_timeout_ms);

    // Prefer the baseline captured by the outer tool dispatcher
    // (`capture_edit_baseline` in tools::mod) when it's available. That
    // baseline is taken once *before* this edit_file call begins, so it
    // stays consistent across pre-plan retries and matches what the outer
    // `auto_check` will compare against. Falling back to a local query is
    // only for legacy / direct callers that don't supply one.
    let baseline_count = match baseline_lsp_errors {
        Some(n) => n,
        None => match diagnostics_for_current_file(lsp, path, timeout).await {
            Ok(diags) => error_diagnostics(&diags).len(),
            Err(e) => {
                if lsp_validation == LspValidationMode::Require {
                    return Err(ValidationError::Other(anyhow::anyhow!(
                        "LSP baseline diagnostics failed: {e}"
                    )));
                }
                return Ok(None);
            }
        },
    };

    std::fs::write(path, candidate)?;

    let candidate_diags = match diagnostics_for_current_file(lsp, path, timeout).await {
        Ok(diags) => diags,
        Err(e) => {
            let _ = std::fs::write(path, original);
            let _ = diagnostics_for_current_file(lsp, path, timeout).await;
            if lsp_validation == LspValidationMode::Require {
                return Err(ValidationError::Other(anyhow::anyhow!(
                    "LSP candidate diagnostics failed: {e}"
                )));
            }
            return Ok(None);
        }
    };

    let candidate_errors = error_diagnostics(&candidate_diags);
    if candidate_errors.len() > baseline_count {
        let regression = build_lsp_regression(baseline_count, &candidate_errors, candidate);
        let _ = std::fs::write(path, original);
        let _ = diagnostics_for_current_file(lsp, path, timeout).await;
        let _ = path_str; // path_str retained for callers formatting the one-line summary
        return Err(ValidationError::LspRegression(regression));
    }

    Ok(Some(format!(
        "[lsp] OK ({} -> {} error(s), mode={})",
        baseline_count,
        candidate_errors.len(),
        lsp_validation.as_str()
    )))
}

/// Build the structured `LspRegression` carried back to the planner.
/// We keep the first 5 errors inline (matching the one-line summary
/// cap) and record the remaining count so the summary still says
/// "and N more".
pub(super) fn build_lsp_regression(
    baseline_count: usize,
    candidate_errors: &[Diagnostic],
    candidate_content: &str,
) -> LspRegression {
    const MAX_ERRORS: usize = 5;
    let total = candidate_errors.len();
    let kept: Vec<LspErrorLocation> = candidate_errors
        .iter()
        .take(MAX_ERRORS)
        .map(|d| LspErrorLocation {
            line: (d.range.start.line + 1) as usize,
            column: (d.range.start.character + 1) as usize,
            message: d.message.clone(),
        })
        .collect();
    let extra_error_count = total.saturating_sub(kept.len());
    LspRegression {
        baseline_count,
        errors: kept,
        extra_error_count,
        candidate_content: candidate_content.to_string(),
    }
}

pub(super) async fn diagnostics_for_current_file(
    lsp: &LspClient,
    path: &std::path::Path,
    timeout: Duration,
) -> Result<Vec<Diagnostic>> {
    lsp.notify_file_changed(path)?;
    Ok(lsp.get_diagnostics(path, timeout).await)
}

pub(super) fn error_diagnostics(diagnostics: &[Diagnostic]) -> Vec<Diagnostic> {
    diagnostics
        .iter()
        .filter(|d| d.severity == Some(DiagnosticSeverity::ERROR))
        .cloned()
        .collect()
}
