//! Locate callsites via LSP `textDocument/references` and extract context
//! windows for the per-callsite model rewrites.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use lsp_types::SymbolKind;

use crate::config::Config;
use crate::lsp::{LspClient, uri_to_path};
use crate::tools::fast::RevisionStore;

/// One callsite, with a window of source lines around it.
#[derive(Debug, Clone)]
pub struct CallSite {
    pub path: PathBuf,
    /// 0-based line number of the call expression's starting line, as
    /// reported by the LSP.
    pub line: u32,
    /// 0-based column of the call expression's start.
    pub column: u32,
    /// The context window's first line (0-based, inclusive).
    pub window_start: u32,
    /// The context window's last line (0-based, inclusive).
    pub window_end: u32,
    /// Verbatim window content (lines `window_start..=window_end`, joined
    /// with `\n`, no trailing newline).
    pub window: String,
}

/// Number of *trailing* context lines to include after the target line in
/// a snippet. The target line is always at position 0 of the snippet; the
/// model is told "edit the FIRST line of the snippet." Trailing context
/// is needed because multi-line calls or signatures extend down past the
/// first line — the model needs to see the whole expression to produce a
/// matching OLD/NEW block.
///
/// Generous (12 lines) on purpose: the model only ever rewrites the
/// expression starting at line 0, so extra trailing context can't lure
/// it into the wrong target.
pub const TRAILING_LINES: u32 = 12;

/// Resolved canonical location of a function definition.
#[derive(Debug, Clone)]
pub struct ResolvedFunction {
    pub line_0: u32,
    pub column_0: u32,
}

/// Resolve a (path, name, optional line hint) into the canonical
/// definition position of a function or method named `name`.
///
/// Determinism > convenience: the agent's `line_hint` is only used to
/// disambiguate when multiple symbols in the file share `name`. The
/// returned position points at the symbol's *name*, which is what
/// `find_references` and `rename` expect.
///
/// On miss, falls back to `workspace/symbol` to suggest where the named
/// function actually lives — that becomes the actionable error message.
pub async fn resolve_function_location(
    lsp: &LspClient,
    path: &std::path::Path,
    name: &str,
    line_hint: Option<u32>,
) -> Result<ResolvedFunction> {
    let symbols = lsp
        .document_symbol(path)
        .await
        .context("LSP documentSymbol failed")?;

    // Filter to function-like kinds. We accept Method and Constructor too —
    // every refactor we support applies equally to free functions and to
    // associated methods.
    let candidates: Vec<&_> = symbols
        .iter()
        .filter(|s| {
            s.name == name
                && matches!(
                    s.kind,
                    SymbolKind::FUNCTION | SymbolKind::METHOD | SymbolKind::CONSTRUCTOR
                )
        })
        .collect();

    if candidates.is_empty() {
        // Surface available functions in this file plus a workspace-wide
        // suggestion so the agent can re-issue the call with the right
        // path. Limit the list so the error stays readable.
        let in_file: Vec<String> = symbols
            .iter()
            .filter(|s| {
                matches!(
                    s.kind,
                    SymbolKind::FUNCTION | SymbolKind::METHOD | SymbolKind::CONSTRUCTOR
                )
            })
            .take(8)
            .map(|s| format!("{} (line {})", s.name, s.name_range.start.line + 1))
            .collect();

        let workspace_hits = lsp.workspace_symbol(name).await.unwrap_or_default();
        let workspace_hint = workspace_hits
            .into_iter()
            .filter(|s| {
                matches!(
                    s.kind,
                    SymbolKind::FUNCTION | SymbolKind::METHOD | SymbolKind::CONSTRUCTOR
                ) && s.name == name
            })
            .take(3)
            .map(|s| format!("{} (line {})", s.path.display(), s.line + 1))
            .collect::<Vec<_>>()
            .join(", ");

        let in_file_hint = if in_file.is_empty() {
            "no functions defined in this file".to_string()
        } else {
            format!("functions in this file: {}", in_file.join(", "))
        };
        let cross_file_hint = if workspace_hint.is_empty() {
            "no other file in the workspace defines this function either".to_string()
        } else {
            format!("but `{name}` is defined in: {workspace_hint} — point refactor there")
        };

        return Err(anyhow!(
            "no function named `{name}` defined in {} ({in_file_hint}; {cross_file_hint})",
            path.display(),
        ));
    }

    // Disambiguate when multiple symbols share the name (e.g. `run` as a
    // method on multiple impls). Pick the candidate whose full range is
    // closest to the agent's line hint; without a hint we just take the
    // first one and surface a warning in the result.
    let chosen = if let Some(hint) = line_hint {
        candidates
            .iter()
            .min_by_key(|s| {
                let range = &s.full_range;
                if range.start.line <= hint && hint <= range.end.line {
                    0u64
                } else {
                    let d_start = (range.start.line as i64 - hint as i64).unsigned_abs();
                    let d_end = (range.end.line as i64 - hint as i64).unsigned_abs();
                    d_start.min(d_end) + 1
                }
            })
            .copied()
            .unwrap()
    } else {
        candidates[0]
    };

    // Older LSPs (and rust-analyzer when the client doesn't advertise
    // hierarchicalDocumentSymbolSupport) return SymbolInformation, which
    // has no separate name span — `name_range` ends up equal to
    // `full_range` (the whole signature + body). Detect that case by
    // checking whether the range spans more than the start line OR
    // covers more than `name.len()` chars on the start line; if so,
    // search the start line's source for the identifier.
    let likely_full_span = chosen.name_range.end.line != chosen.name_range.start.line
        || chosen
            .name_range
            .end
            .character
            .saturating_sub(chosen.name_range.start.character)
            > name.len() as u32 + 4;
    if likely_full_span {
        let src = std::fs::read_to_string(path).context("read source for name column lookup")?;
        if let Some(col) = find_identifier_column(&src, chosen.full_range.start.line, name) {
            return Ok(ResolvedFunction {
                line_0: chosen.full_range.start.line,
                column_0: col,
            });
        }
    }

    Ok(ResolvedFunction {
        line_0: chosen.name_range.start.line,
        column_0: chosen.name_range.start.character,
    })
}

/// Locate `name` as an identifier on `line_0` of `source`. Returns the
/// 0-based column of the first whole-word occurrence.
fn find_identifier_column(source: &str, line_0: u32, name: &str) -> Option<u32> {
    let line = source.lines().nth(line_0 as usize)?;
    let mut start = 0;
    while let Some(idx) = line[start..].find(name) {
        let abs = start + idx;
        let before_ok = abs == 0
            || !line[..abs]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_alphanumeric() || c == '_');
        let after_ok = !line[abs + name.len()..]
            .chars()
            .next()
            .is_some_and(|c| c.is_alphanumeric() || c == '_');
        if before_ok && after_ok {
            return Some(line[..abs].chars().count() as u32);
        }
        start = abs + name.len();
    }
    None
}

/// One file's staged edit: keep the original alongside the updated
/// content so we can record both into the revision store on commit.
#[derive(Debug, Clone)]
pub struct StagedEdit {
    pub original: String,
    pub updated: String,
}

/// Commit `staged` to disk and, when a Fast-mode `RevisionStore` is in
/// scope, also record each file's pristine + post-refactor revision so
/// the agent's `revert(path, rev=N)` works after a refactor (otherwise
/// only the round-based shadow-git revert sees these edits).
///
/// `tool_name` becomes the operation label in the revision table.
pub fn commit_staged(
    staged: &std::collections::BTreeMap<PathBuf, StagedEdit>,
    config: &Config,
    revisions: Option<&RevisionStore>,
    tool_name: &str,
) -> Result<()> {
    use crate::tools::fast::RecordArgs;

    for (path, edit) in staged {
        std::fs::write(path, &edit.updated)
            .with_context(|| format!("write {} after refactor", path.display()))?;

        if let Some(store) = revisions {
            let rel = path
                .strip_prefix(&config.project_root)
                .unwrap_or(path)
                .display()
                .to_string();
            // First touch on this file in the session — record the
            // pristine baseline so a future `revert(rev=0)` actually
            // restores the original.
            store.ensure_pristine(&rel, &edit.original).ok();
            let added = edit
                .updated
                .lines()
                .count()
                .saturating_sub(edit.original.lines().count());
            let removed = edit
                .original
                .lines()
                .count()
                .saturating_sub(edit.updated.lines().count());
            store
                .record(
                    &rel,
                    &edit.updated,
                    RecordArgs {
                        operation: tool_name,
                        label: tool_name,
                        range: None,
                        payload: None,
                        added,
                        removed,
                        ast_ok: true,
                        ast_error: None,
                        file_errors: 0,
                        project_errors: 0,
                    },
                )
                .ok();
        }
    }
    Ok(())
}

/// Wait for the LSP to be ready before issuing a references request.
/// `find_references` against an unready server returns empty results, which
/// would silently make the refactor a no-op.
pub async fn ensure_ready(lsp: &LspClient, timeout: Duration) -> Result<()> {
    if lsp.has_crashed() {
        return Err(anyhow!("LSP has crashed"));
    }
    if lsp.is_ready() {
        return Ok(());
    }
    let deadline = std::time::Instant::now() + timeout;
    while std::time::Instant::now() < deadline {
        if lsp.is_ready() {
            return Ok(());
        }
        if lsp.has_crashed() {
            return Err(anyhow!("LSP crashed while waiting for ready"));
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    Err(anyhow!("LSP did not become ready within {:?}", timeout))
}

/// Find all callsites of the symbol at `(line, column)` (0-based) in `path`,
/// excluding the definition itself.
///
/// Returns a vector of `CallSite` with surrounding context windows already
/// extracted. Same-file references that overlap the definition's own line
/// are filtered out.
pub async fn find_callsites(
    lsp: &LspClient,
    config: &Config,
    def_path: &Path,
    def_line: u32,
    def_column: u32,
) -> Result<Vec<CallSite>> {
    // Let the analyzer finish any work it has in flight so references are
    // computed against a consistent snapshot. Cheap when the server is
    // already idle; cap the wait so a stuck server can't pin the agent.
    let _ = lsp.wait_for_idle(Duration::from_secs(30)).await;

    let locations = lsp
        .find_references(def_path, def_line, def_column)
        .await
        .context("LSP find_references failed")?;

    let mut sites = Vec::new();
    for loc in locations {
        let Some(path) = uri_to_path(&loc.uri) else {
            continue;
        };
        let line = loc.range.start.line;
        let column = loc.range.start.character;
        // Filter out the declaration itself: same file, same line as the
        // user-supplied definition position. We can't filter purely on
        // column because rust-analyzer reports the *name* span (after
        // `fn `), not the keyword, so the column rarely matches what the
        // caller passed.
        if path == def_path && line == def_line {
            continue;
        }
        // Stay within the project to avoid editing dependency sources
        // pulled in by the LSP's index.
        if !path.starts_with(&config.project_root) {
            continue;
        }
        let source = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let window = extract_window(&source, line, TRAILING_LINES);
        sites.push(CallSite {
            path,
            line,
            column,
            window_start: window.start,
            window_end: window.end,
            window: window.text,
        });
    }
    // Sort: same file together (by path then descending line) so when we
    // apply edits in order, earlier edits don't shift later edits' line
    // numbers. We rewrite per-line based on verbatim string match anyway,
    // but the bottom-up order keeps things robust if we ever switch to
    // line-index-based application.
    sites.sort_by(|a, b| {
        a.path
            .cmp(&b.path)
            .then_with(|| b.line.cmp(&a.line))
            .then_with(|| b.column.cmp(&a.column))
    });
    Ok(sites)
}

pub struct Window {
    pub start: u32,
    pub end: u32,
    pub text: String,
}

/// Extract a snippet starting at `line` and including up to `lines_after`
/// trailing lines (clamped to the file). The target line is always
/// `lines[0]` of the snippet, so the model can be told unambiguously
/// "edit the FIRST line of the snippet" without any line-number arithmetic.
pub fn extract_window(source: &str, line: u32, lines_after: u32) -> Window {
    let lines: Vec<&str> = source.lines().collect();
    let total = lines.len() as u32;
    if total == 0 || line >= total {
        return Window {
            start: line,
            end: line,
            text: String::new(),
        };
    }
    let start = line;
    let end = (line + lines_after).min(total.saturating_sub(1));
    let text = lines[start as usize..=end as usize].join("\n");
    Window { start, end, text }
}
