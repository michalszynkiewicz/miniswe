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
    /// First parse error, when `ast_ok=false`. Format: `"L42:9: syntax error"`.
    pub ast_error: Option<String>,
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
    let ast_error = ast_result.as_ref().err().cloned();
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
        ast_error,
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
        format!("[lsp file] {count} error(s) (showing top {TOP_FILE_ERRORS_WHEN_TRUNCATED})\n")
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

/// Number of most-recent tombstones rendered with expanded payload
/// preview in the revision table. Older tombstones drop to one-liners.
const EXPANDED_TOMBSTONES: usize = 3;

/// Max bytes of payload preview shown for an expanded tombstone row.
const TOMBSTONE_PAYLOAD_PREVIEW_BYTES: usize = 300;

/// Max lines of payload preview shown for an expanded tombstone row.
const TOMBSTONE_PAYLOAD_PREVIEW_LINES: usize = 5;

fn render_revision_table(rel_path: &str, revs: &[Revision]) -> String {
    if revs.is_empty() {
        return String::new();
    }

    // Partition into live chain vs tombstones, preserving order.
    let live: Vec<&Revision> = revs.iter().filter(|r| !r.reverted).collect();
    let tombs: Vec<&Revision> = revs.iter().filter(|r| r.reverted).collect();

    let current = live.last().map(|r| r.number).unwrap_or(0);
    let mut out = format!("[revisions] {rel_path}\n");

    // Column width: compute across *live* rows only — tombstones render
    // as one-liners or expanded blocks with their own layout.
    let label_width = live
        .iter()
        .map(|r| render_label(r).len())
        .max()
        .unwrap_or(0);

    for r in &live {
        out.push_str(&render_live_row(r, current, label_width));
        out.push('\n');
    }

    if !tombs.is_empty() {
        // Tombstones: render oldest-first so they read chronologically.
        // Last EXPANDED_TOMBSTONES get expanded payload; older ones get
        // a one-liner.
        let split_at = tombs.len().saturating_sub(EXPANDED_TOMBSTONES);
        for r in &tombs[..split_at] {
            out.push_str(&render_tombstone_oneliner(r, label_width));
            out.push('\n');
        }
        for r in &tombs[split_at..] {
            out.push_str(&render_tombstone_expanded(r, label_width));
            out.push('\n');
        }
    }

    // Drop trailing newline
    if out.ends_with('\n') {
        out.pop();
    }
    out
}

fn render_live_row(r: &Revision, current: usize, label_width: usize) -> String {
    let marker = if r.number == current { "*" } else { " " };
    let label = render_label(r);
    let ast = if r.ast_ok { "ok" } else { "broken" };
    let mut out = format!(
        "{marker} rev_{num:<2} {label:<label_width$}  ast={ast:<6}  file_errors={fe}  project_errors={pe}",
        num = r.number,
        fe = r.file_errors,
        pe = r.project_errors,
    );
    if r.number == current {
        out.push_str("  <- current");
    }
    out
}

fn render_tombstone_oneliner(r: &Revision, label_width: usize) -> String {
    let label = render_label(r);
    format!(
        "  rev_{num:<2} {label:<label_width$}  [reverted, {outcome}]",
        num = r.number,
        outcome = outcome_tag(r),
    )
}

fn render_tombstone_expanded(r: &Revision, label_width: usize) -> String {
    let label = render_label(r);
    let mut out = format!(
        "  rev_{num:<2} {label:<label_width$}  [reverted, {outcome}]",
        num = r.number,
        outcome = outcome_tag(r),
    );
    match (r.operation.as_str(), &r.payload) {
        ("write_file", _) => {
            let bytes = r.content.len();
            let lines = r.content.lines().count().max(1);
            out.push_str(&format!(
                "\n         (full rewrite, {lines} line(s), {bytes} bytes)"
            ));
        }
        (_, Some(payload)) => {
            let key = if r.operation == "insert_at" {
                "text"
            } else {
                "new_text"
            };
            let (preview, extra_lines, extra_bytes) = truncate_preview(
                payload,
                TOMBSTONE_PAYLOAD_PREVIEW_LINES,
                TOMBSTONE_PAYLOAD_PREVIEW_BYTES,
            );
            out.push_str(&format!("\n         {key}: |"));
            for line in preview.lines() {
                out.push_str("\n           ");
                out.push_str(line);
            }
            if extra_lines > 0 || extra_bytes > 0 {
                let detail = if extra_lines > 0 {
                    format!("{extra_lines} more line(s)")
                } else {
                    format!("{extra_bytes} more byte(s)")
                };
                out.push_str(&format!(
                    "\n         … {detail} — use show_rev to see full payload"
                ));
            }
        }
        (_, None) => {
            // No payload stored (unexpected for a replace_range/insert_at
            // tombstone, but safe fallback)
        }
    }
    out
}

/// Short `[reverted, X]` tag describing why the rev was abandoned.
/// Order of precedence: AST break > file error delta > plain "no errors".
fn outcome_tag(r: &Revision) -> String {
    if !r.ast_ok {
        if let Some(err) = &r.ast_error {
            return format!("ast=broken at {err}");
        }
        return "ast=broken".into();
    }
    if r.file_errors > 0 {
        return format!("file_errors={fe}", fe = r.file_errors);
    }
    "no errors".into()
}

/// Trim `text` to `max_lines` lines OR `max_bytes` bytes (whichever hits
/// first). Returns `(preview, extra_lines, extra_bytes)` where `extra_*`
/// describe what was dropped (one of the two is zero — we pick the more
/// informative hint).
fn truncate_preview(text: &str, max_lines: usize, max_bytes: usize) -> (String, usize, usize) {
    let total_lines = text.lines().count();
    let total_bytes = text.len();

    // Collect up to max_lines lines first.
    let collected: Vec<&str> = text.lines().take(max_lines).collect();
    let mut preview = collected.join("\n");

    // If the line-bounded preview is still over max_bytes, cut further.
    if preview.len() > max_bytes {
        // Walk back to the nearest char boundary.
        let mut cut = max_bytes;
        while cut > 0 && !preview.is_char_boundary(cut) {
            cut -= 1;
        }
        preview.truncate(cut);
        let extra_bytes = total_bytes.saturating_sub(preview.len());
        return (preview, 0, extra_bytes);
    }

    let extra_lines = total_lines.saturating_sub(collected.len());
    (preview, extra_lines, 0)
}

fn render_label(r: &Revision) -> String {
    if r.number == 0 || (r.added == 0 && r.removed == 0) {
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
            operation: if n == 0 {
                "initial".into()
            } else {
                "replace_range".into()
            },
            label: label.into(),
            range: None,
            payload: None,
            added,
            removed,
            ast_ok: true,
            ast_error: None,
            file_errors: fe,
            project_errors: pe,
            reverted: false,
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
        assert!(
            last.starts_with('*'),
            "last row should be marked current: {last}"
        );
        assert!(last.contains("<- current"));
        // Earlier rows unmarked
        for l in table.lines().skip(1).take(2) {
            assert!(
                l.starts_with(' '),
                "non-current row should not be marked: {l}"
            );
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
        let diags = vec![lsp_types::Diagnostic {
            range: lsp_types::Range {
                start: lsp_types::Position {
                    line: 4,
                    character: 2,
                },
                end: lsp_types::Position {
                    line: 4,
                    character: 8,
                },
            },
            severity: Some(lsp_types::DiagnosticSeverity::ERROR),
            message: "bad thing".into(),
            ..Default::default()
        }];
        let out = render_file_diagnostics(&diags);
        assert!(out.contains("1 error(s)"));
        assert!(out.contains("L5:3: bad thing"));
    }

    #[test]
    fn file_diagnostics_over_cap_truncate() {
        let many: Vec<_> = (0..15)
            .map(|i| lsp_types::Diagnostic {
                range: lsp_types::Range {
                    start: lsp_types::Position {
                        line: i,
                        character: 0,
                    },
                    end: lsp_types::Position {
                        line: i,
                        character: 1,
                    },
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

    fn tomb(n: usize, label: &str, payload: &str, ast_error: Option<&str>) -> Revision {
        Revision {
            number: n,
            operation: "replace_range".into(),
            label: label.into(),
            range: Some((1, 1)),
            payload: Some(payload.into()),
            added: 1,
            removed: 1,
            ast_ok: ast_error.is_none(),
            ast_error: ast_error.map(String::from),
            file_errors: if ast_error.is_some() { 0 } else { 3 },
            project_errors: 0,
            reverted: true,
            content: String::new(),
        }
    }

    #[test]
    fn tombstone_row_shows_reverted_tag_and_ast_error() {
        let revs = vec![
            rev(0, "initial", 0, 0, 0, 0),
            tomb(
                1,
                "replace_range L42-42",
                "bad()",
                Some("L42:5: syntax error"),
            ),
        ];
        let table = render_revision_table("a.rs", &revs);
        assert!(
            table.contains("[reverted, ast=broken at L42:5: syntax error]"),
            "table missing reverted tag with ast error:\n{table}"
        );
        // Tombstone should show current marker on rev_0 (only live rev)
        let lines: Vec<&str> = table.lines().collect();
        let rev0_line = lines.iter().find(|l| l.contains("rev_0")).unwrap();
        assert!(rev0_line.contains("<- current"));
    }

    #[test]
    fn last_three_tombstones_expanded_older_one_liners() {
        // 5 tombstones; only the last 3 should have `new_text: |` blocks.
        let mut revs = vec![rev(0, "initial", 0, 0, 0, 0)];
        for i in 1..=5 {
            revs.push(tomb(
                i,
                &format!("replace_range L{i}-{i}"),
                &format!("payload_{i}"),
                Some(&format!("L{i}:1: syntax error")),
            ));
        }
        let table = render_revision_table("a.rs", &revs);

        // Last 3 (rev_3, rev_4, rev_5) should have expanded payload
        assert!(
            table.contains("payload_3"),
            "rev_3 should be expanded:\n{table}"
        );
        assert!(table.contains("payload_4"));
        assert!(table.contains("payload_5"));
        // First 2 (rev_1, rev_2) should NOT show payload
        assert!(
            !table.contains("payload_1"),
            "rev_1 should be one-liner only:\n{table}"
        );
        assert!(!table.contains("payload_2"));

        // Expanded rows should have the `new_text: |` yaml marker
        assert!(table.contains("new_text: |"));
    }

    #[test]
    fn tombstone_payload_truncated_past_five_lines() {
        let big_payload = (1..=10)
            .map(|i| format!("line_{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let revs = vec![
            rev(0, "initial", 0, 0, 0, 0),
            tomb(1, "replace_range L1-1", &big_payload, None),
        ];
        let table = render_revision_table("a.rs", &revs);
        assert!(table.contains("line_1"));
        assert!(table.contains("line_5"));
        assert!(
            !table.contains("line_6"),
            "payload should stop at 5 lines:\n{table}"
        );
        assert!(table.contains("5 more line(s)"));
        assert!(table.contains("use show_rev"));
    }

    #[test]
    fn insert_at_tombstone_uses_text_label_not_new_text() {
        let mut t = tomb(1, "insert_at after L1", "hi\n", None);
        t.operation = "insert_at".into();
        t.range = None;
        let revs = vec![rev(0, "initial", 0, 0, 0, 0), t];
        let table = render_revision_table("a.rs", &revs);
        assert!(
            table.contains("text: |"),
            "insert_at should use 'text:' key:\n{table}"
        );
        assert!(!table.contains("new_text: |"));
    }

    #[test]
    fn write_file_tombstone_shows_rewrite_summary_not_content() {
        let mut t = tomb(1, "write_file", "irrelevant_payload", None);
        t.operation = "write_file".into();
        t.payload = None;
        t.content = "a\nb\nc\nd\n".into();
        let revs = vec![rev(0, "initial", 0, 0, 0, 0), t];
        let table = render_revision_table("a.rs", &revs);
        assert!(table.contains("full rewrite"));
        assert!(table.contains("4 line(s)"));
        assert!(!table.contains("irrelevant_payload"));
    }

    #[test]
    fn live_chain_renders_before_tombstones() {
        let mut t = tomb(2, "replace_range L1-1", "bad", Some("L1:1: err"));
        let revs = vec![
            rev(0, "initial", 0, 0, 0, 0),
            rev(1, "replace_range L1-1", 1, 1, 0, 0),
            {
                t.reverted = true;
                t
            },
        ];
        let table = render_revision_table("a.rs", &revs);
        let lines: Vec<&str> = table.lines().collect();
        let rev1_idx = lines.iter().position(|l| l.contains("rev_1 ")).unwrap();
        let rev2_idx = lines.iter().position(|l| l.contains("rev_2 ")).unwrap();
        assert!(
            rev1_idx < rev2_idx,
            "live rev_1 must appear before tombstoned rev_2:\n{table}"
        );
    }

    #[test]
    fn current_marker_tracks_last_live_rev_when_later_are_tombstones() {
        let mut t = tomb(2, "replace_range L1-1", "oops", Some("L1:1: err"));
        t.reverted = true;
        let revs = vec![
            rev(0, "initial", 0, 0, 0, 0),
            rev(1, "replace_range L1-1", 1, 1, 0, 0),
            t,
        ];
        let table = render_revision_table("a.rs", &revs);
        // rev_1 (last live) should be current, rev_2 should NOT be.
        let lines: Vec<&str> = table.lines().collect();
        let rev1_line = lines.iter().find(|l| l.contains("rev_1 ")).unwrap();
        assert!(
            rev1_line.contains("<- current"),
            "rev_1 should be current:\n{table}"
        );
        let rev2_line = lines.iter().find(|l| l.contains("rev_2 ")).unwrap();
        assert!(!rev2_line.contains("<- current"));
    }
}
