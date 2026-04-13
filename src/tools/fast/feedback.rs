//! Per-edit feedback builder: AST status + per-file LSP diagnostics +
//! project-wide error stat + revision table.
//!
//! Every fast-mode write-ish tool appends this block to its `ToolResult`
//! content so the model always sees where it is and what just broke.
//! Format is optimized for tiny-model pattern-matching; see
//! `docs/fast-mode-design.md` for the design rationale.

use crate::config::Config;
use crate::lsp::LspClient;

use super::ast::parse_check;
use super::revisions::{Revision, RevisionStore};

/// Max per-file LSP errors listed verbatim in the feedback. Over this,
/// we show top N and summarize the tail.
const MAX_FILE_ERRORS_INLINE: usize = 10;

/// When the file has more than `MAX_FILE_ERRORS_INLINE` errors, show the
/// first N only and append a "…and K more" line.
const TOP_FILE_ERRORS_WHEN_TRUNCATED: usize = 3;

/// Built feedback block, ready to append to a `ToolResult.content`.
pub struct EditFeedback {
    /// Rendered multi-line feedback text.
    pub text: String,
    /// AST parsed cleanly for this file.
    pub ast_ok: bool,
    /// LSP error count in this file right now.
    pub file_errors: usize,
    /// LSP error count across the project right now.
    pub project_errors: usize,
}

/// Build per-edit feedback for `rel_path` after a successful write.
///
/// Runs AST parse on the new content, polls LSP for current per-file and
/// project-wide diagnostics, and renders the revision table. All IO is
/// best-effort: missing pieces degrade gracefully (no LSP → AST-only,
/// still useful).
///
/// The revision table shown reflects whatever is currently in
/// `revisions` for `rel_path` — the caller is expected to have already
/// called `ensure_pristine` / `record` / `truncate_to` as appropriate
/// before this function runs.
pub async fn build_feedback(
    rel_path: &str,
    new_content: &str,
    config: &Config,
    lsp: Option<&LspClient>,
    revisions: &RevisionStore,
    project_baseline_errors: usize,
) -> EditFeedback {
    // AST
    let ast_result = parse_check(rel_path, new_content);
    let ast_ok = ast_result.is_ok();
    let ast_line = match &ast_result {
        Ok(()) => "[ast] ok".to_string(),
        Err(msg) => format!("[ast] broken: {msg}"),
    };

    // Per-file LSP
    let abs_path = config.project_root.join(rel_path);
    let file_diags = collect_file_diagnostics(&abs_path, config, lsp).await;
    let file_errors = file_diags.len();
    let file_lsp_line = render_file_diagnostics(&file_diags);

    // Project-wide LSP (a single number + delta)
    let project_errors = collect_project_error_count(lsp).await;
    let delta = project_errors as isize - project_baseline_errors as isize;
    let project_lsp_line = format!(
        "[lsp project] {project_errors} error(s) ({sign}{delta} from baseline)",
        sign = if delta >= 0 { "+" } else { "" }
    );

    // Revision table
    let revs = revisions.list(rel_path);
    let rev_table = render_revision_table(rel_path, &revs);

    let mut text = String::new();
    text.push('\n');
    text.push_str(&ast_line);
    text.push('\n');
    text.push_str(&file_lsp_line);
    text.push('\n');
    text.push_str(&project_lsp_line);
    if !rev_table.is_empty() {
        text.push('\n');
        text.push_str(&rev_table);
    }

    EditFeedback {
        text,
        ast_ok,
        file_errors,
        project_errors,
    }
}

/// Snapshot total LSP error count across the project. Returns 0 if LSP is
/// absent or not ready. Used for the `project_errors` line in feedback
/// and as a baseline at session start.
pub async fn project_error_count(lsp: Option<&LspClient>) -> usize {
    collect_project_error_count(lsp).await
}

async fn collect_file_diagnostics(
    abs_path: &std::path::Path,
    config: &Config,
    lsp: Option<&LspClient>,
) -> Vec<lsp_types::Diagnostic> {
    let Some(lsp) = lsp else {
        return Vec::new();
    };
    if !lsp.is_ready() || lsp.has_crashed() {
        return Vec::new();
    }
    if lsp.notify_file_changed(abs_path).is_err() {
        return Vec::new();
    }
    let timeout = std::time::Duration::from_millis(config.lsp.diagnostic_timeout_ms);
    let diags = lsp.get_diagnostics(abs_path, timeout).await;
    diags
        .into_iter()
        .filter(|d| d.severity == Some(lsp_types::DiagnosticSeverity::ERROR))
        .collect()
}

async fn collect_project_error_count(lsp: Option<&LspClient>) -> usize {
    let Some(lsp) = lsp else {
        return 0;
    };
    if !lsp.is_ready() || lsp.has_crashed() {
        return 0;
    }
    let mut count = 0;
    for entry in lsp.diagnostics_snapshot() {
        for diag in entry.1 {
            if diag.severity == Some(lsp_types::DiagnosticSeverity::ERROR) {
                count += 1;
            }
        }
    }
    count
}

fn render_file_diagnostics(diags: &[lsp_types::Diagnostic]) -> String {
    let count = diags.len();
    if count == 0 {
        return "[lsp file] 0 errors".into();
    }

    let mut out = if count <= MAX_FILE_ERRORS_INLINE {
        format!("[lsp file] {count} error(s)\n")
    } else {
        format!(
            "[lsp file] {count} error(s) (showing top {TOP_FILE_ERRORS_WHEN_TRUNCATED})\n"
        )
    };
    let shown = if count <= MAX_FILE_ERRORS_INLINE {
        count
    } else {
        TOP_FILE_ERRORS_WHEN_TRUNCATED
    };
    for d in diags.iter().take(shown) {
        let line = d.range.start.line + 1;
        let col = d.range.start.character + 1;
        out.push_str(&format!("  L{line}:{col}: {}\n", d.message));
    }
    if count > shown {
        out.push_str(&format!("  … and {} more\n", count - shown));
    }
    // Drop trailing newline — caller adds one.
    if out.ends_with('\n') {
        out.pop();
    }
    out
}

fn render_revision_table(rel_path: &str, revs: &[Revision]) -> String {
    if revs.is_empty() {
        return String::new();
    }
    let current = revs.last().map(|r| r.number).unwrap_or(0);
    let mut out = format!("[revisions] {rel_path}\n");

    // Compute column widths for a stable, scannable layout.
    let label_width = revs
        .iter()
        .map(|r| render_label(r).len())
        .max()
        .unwrap_or(0);

    for r in revs {
        let marker = if r.number == current { "*" } else { " " };
        let label = render_label(r);
        let ast = if r.ast_ok { "ok" } else { "broken" };
        out.push_str(&format!(
            "{marker} rev_{num:<2} {label:<label_width$}  ast={ast:<6}  file_errors={fe}  project_errors={pe}",
            num = r.number,
            fe = r.file_errors,
            pe = r.project_errors,
        ));
        if r.number == current {
            out.push_str("  <- current");
        }
        out.push('\n');
    }
    // Drop trailing newline
    if out.ends_with('\n') {
        out.pop();
    }
    out
}

fn render_label(r: &Revision) -> String {
    if r.number == 0 {
        r.label.clone()
    } else if r.added == 0 && r.removed == 0 {
        r.label.clone()
    } else {
        format!("{} (+{} -{})", r.label, r.added, r.removed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rev(n: usize, label: &str, added: usize, removed: usize, fe: usize, pe: usize) -> Revision {
        Revision {
            number: n,
            label: label.into(),
            added,
            removed,
            ast_ok: true,
            file_errors: fe,
            project_errors: pe,
            // Content is private to the module; use a dummy string via the
            // only public constructor — ensure_pristine / record. We don't
            // test `content` here, just formatting of metadata, so we drop
            // through the public API:
            content: String::new(),
        }
    }

    #[test]
    fn revision_table_marks_current_row() {
        let revs = vec![
            rev(0, "initial", 0, 0, 0, 0),
            rev(1, "write_file", 12, 0, 0, 0),
            rev(2, "replace_range L42", 1, 1, 0, 0),
        ];
        let table = render_revision_table("src/x.rs", &revs);
        assert!(table.contains("[revisions] src/x.rs"));
        let last = table.lines().last().unwrap();
        assert!(last.starts_with('*'), "last row should be marked current: {last}");
        assert!(last.contains("<- current"));
        // Earlier rows unmarked
        for l in table.lines().skip(1).take(2) {
            assert!(l.starts_with(' '), "non-current row should not be marked: {l}");
        }
    }

    #[test]
    fn revision_table_shows_line_deltas() {
        let revs = vec![
            rev(0, "initial", 0, 0, 0, 0),
            rev(1, "replace_range L42", 3, 2, 0, 0),
        ];
        let table = render_revision_table("a.rs", &revs);
        assert!(table.contains("(+3 -2)"), "table missing delta: {table}");
    }

    #[test]
    fn file_diagnostics_empty_shows_zero() {
        let out = render_file_diagnostics(&[]);
        assert_eq!(out, "[lsp file] 0 errors");
    }

    #[test]
    fn file_diagnostics_under_cap_show_inline() {
        let diags = vec![
            lsp_types::Diagnostic {
                range: lsp_types::Range {
                    start: lsp_types::Position { line: 4, character: 2 },
                    end: lsp_types::Position { line: 4, character: 8 },
                },
                severity: Some(lsp_types::DiagnosticSeverity::ERROR),
                message: "bad thing".into(),
                ..Default::default()
            },
        ];
        let out = render_file_diagnostics(&diags);
        assert!(out.contains("1 error(s)"));
        assert!(out.contains("L5:3: bad thing"));
    }

    #[test]
    fn file_diagnostics_over_cap_truncate() {
        let many: Vec<_> = (0..15)
            .map(|i| lsp_types::Diagnostic {
                range: lsp_types::Range {
                    start: lsp_types::Position { line: i, character: 0 },
                    end: lsp_types::Position { line: i, character: 1 },
                },
                severity: Some(lsp_types::DiagnosticSeverity::ERROR),
                message: format!("err {i}"),
                ..Default::default()
            })
            .collect();
        let out = render_file_diagnostics(&many);
        assert!(out.contains("15 error(s)"));
        assert!(out.contains("showing top 3"));
        assert!(out.contains("… and 12 more"));
    }
}
