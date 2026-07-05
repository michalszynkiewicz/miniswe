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

/// Lexer state while scanning for balanced parens — tracks the constructs
/// whose `(`/`)` characters must NOT count toward paren depth.
#[derive(Clone, Copy, PartialEq)]
enum ScanState {
    Normal,
    /// Inside a `"..."`, `'...'`, or `` `...` `` (Go raw string / JS-TS
    /// template literal) literal; the char is the matching quote.
    InString(char),
    InLineComment,
    InBlockComment,
}

/// Scan `tail` for the first balanced `(...)` starting no earlier than byte
/// `from`. Returns `(open_byte, close_byte)`, both absolute offsets into
/// `tail`.
///
/// Skips over string/char literal contents (with `\`-escape awareness) and
/// `//` / `/* */` comments so a `(` or `)` inside a string ARGUMENT (e.g.
/// `foo("(unbalanced")`) or a comment can't be mistaken for a real
/// delimiter — confirmed as a real, silent-corruption risk (not just a
/// clean failure: `foo("a) b(c", x)` previously matched `("a)`, an
/// arbitrary wrong span, not an error). Deliberately does NOT treat `#` as
/// a comment marker (correct for Python, but Rust parameter attributes
/// like `#[cfg(test)] a: u32` would then have the rest of the line wrongly
/// skipped) — this is a real limitation, not a full per-language lexer.
///
/// This is a hand-rolled, best-effort scanner, not a real parser — it
/// treats `\`-escapes uniformly across all three quote characters even
/// though e.g. Go raw strings don't actually support escapes (a rare,
/// low-risk simplification: recognizing backtick fencing at all, even
/// slightly wrong on escapes, beats not recognizing it). A tricky
/// cross-language test matrix (`tricky_callsite_tests` below) exercises
/// raw strings, nested comments, template literals with embedded calls,
/// and regex literals across Rust/Python/Go/TS/Java — most pass, but this
/// approach has repeatedly needed follow-up patches as new cases surfaced
/// (2026-07-05); see that decision point noted in project memory before
/// assuming this list is exhaustive.
fn scan_balanced_parens_from(tail: &str, from: usize) -> Option<(usize, usize)> {
    let chars: Vec<(usize, char)> = tail.char_indices().filter(|&(i, _)| i >= from).collect();

    let mut i = 0;
    let mut state = ScanState::Normal;
    let open_idx = loop {
        let &(byte_pos, c) = chars.get(i)?;
        match state {
            ScanState::Normal => match c {
                '"' | '\'' | '`' => state = ScanState::InString(c),
                '/' if chars.get(i + 1).map(|&(_, c2)| c2) == Some('/') => {
                    state = ScanState::InLineComment;
                    i += 1;
                }
                '/' if chars.get(i + 1).map(|&(_, c2)| c2) == Some('*') => {
                    state = ScanState::InBlockComment;
                    i += 1;
                }
                '(' => break byte_pos,
                _ => {}
            },
            ScanState::InString(q) => {
                if c == '\\' {
                    i += 1;
                } else if c == q {
                    state = ScanState::Normal;
                }
            }
            ScanState::InLineComment => {
                if c == '\n' {
                    state = ScanState::Normal;
                }
            }
            ScanState::InBlockComment => {
                if c == '*' && chars.get(i + 1).map(|&(_, c2)| c2) == Some('/') {
                    state = ScanState::Normal;
                    i += 1;
                }
            }
        }
        i += 1;
    };

    let mut depth = 0i32;
    state = ScanState::Normal;
    while let Some(&(byte_pos, c)) = chars.get(i) {
        match state {
            ScanState::Normal => match c {
                '"' | '\'' | '`' => state = ScanState::InString(c),
                '/' if chars.get(i + 1).map(|&(_, c2)| c2) == Some('/') => {
                    state = ScanState::InLineComment;
                    i += 1;
                }
                '/' if chars.get(i + 1).map(|&(_, c2)| c2) == Some('*') => {
                    state = ScanState::InBlockComment;
                    i += 1;
                }
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some((open_idx, byte_pos));
                    }
                }
                _ => {}
            },
            ScanState::InString(q) => {
                if c == '\\' {
                    i += 1;
                } else if c == q {
                    state = ScanState::Normal;
                }
            }
            ScanState::InLineComment => {
                if c == '\n' {
                    state = ScanState::Normal;
                }
            }
            ScanState::InBlockComment => {
                if c == '*' && chars.get(i + 1).map(|&(_, c2)| c2) == Some('/') {
                    state = ScanState::Normal;
                    i += 1;
                }
            }
        }
        i += 1;
    }
    None
}

/// Find the byte range of a function's PARAMETER LIST parens in `tail`
/// (typically the text of a function starting at its `fn`/`func`/`def`
/// line). Returns `(open_byte, close_byte)`. Shared by
/// `add_param::signature_has_param` (checks param names) and
/// `signature_old_block` (deterministic OLD text for a signature rewrite).
///
/// Language note: usually the first balanced `(...)` IS the parameter list
/// (Rust, Python, Java, plain Go/TS functions all match this). The one
/// exception found in the languages the compile-gate supports
/// (`plan::actions::run_compile_check`): a Go METHOD's receiver clause,
/// `func (r *T) name(params) ...` — the first `(...)` there is the
/// receiver, not the parameters. Detected by checking whether the first
/// `(...)` is immediately followed by `identifier(` (the receiver pattern);
/// if so, the SECOND balanced `(...)` is the real parameter list.
pub fn balanced_parens(tail: &str) -> Option<(usize, usize)> {
    let (first_open, first_close) = scan_balanced_parens_from(tail, 0)?;
    let after = &tail[first_close + 1..];
    let after_trimmed = after.trim_start();
    let ident_len = after_trimmed
        .find(|c: char| !(c.is_alphanumeric() || c == '_'))
        .unwrap_or(after_trimmed.len());
    let looks_like_receiver =
        ident_len > 0 && after_trimmed[ident_len..].trim_start().starts_with('(');
    if looks_like_receiver {
        let skipped = after.len() - after_trimmed.len();
        let second_scan_start = first_close + 1 + skipped + ident_len;
        let (second_open, second_close) = scan_balanced_parens_from(tail, second_scan_start)?;
        return Some((second_open, second_close));
    }
    Some((first_open, first_close))
}

/// Deterministic OLD text for a signature rewrite: the exact source lines
/// from `from_line` (0-based, the function's `fn` line) through the line
/// containing the parameter list's balanced closing `)` — inclusive, even
/// when that line also carries the return type / opening brace (e.g.
/// `) -> AssembledContext {`), since callers replace whole lines. Returns
/// `None` if no balanced `(...)` is found (malformed input; caller falls
/// back to the model-transcribes-OLD path).
///
/// This exists so the model asked to rewrite a signature never has to
/// reproduce OLD itself — see `model_edit::ask_rewrite_validated`'s
/// `known_old` param. Reproducing multi-line OLD verbatim (byte-for-byte,
/// exact indentation) is exactly where small models were failing: observed
/// on Devstral and Gemma 4, exhausting all 3 retries and forcing callers to
/// abandon the atomic refactor for a manual single-file edit that skips
/// every callsite (2026-07-04 forensic trace of a 4/6 compaction-bench run).
pub fn signature_old_block(source: &str, from_line: usize) -> Option<String> {
    let tail: String = source
        .lines()
        .skip(from_line)
        .collect::<Vec<_>>()
        .join("\n");
    let (_, close) = balanced_parens(&tail)?;
    let close_line_offset = tail[..close].matches('\n').count();
    Some(
        source
            .lines()
            .skip(from_line)
            .take(close_line_offset + 1)
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

/// Given the text right after an opening `<` of a generic argument list
/// (e.g. Rust turbofish `::<u32>`), returns the text after the matching
/// `>` (simple bracket-depth counting — type parameters never contain
/// string literals or comments, so this doesn't need the lexer-aware
/// scan). Returns the whole input unchanged if there's no matching `>`.
fn skip_generic_args(after_lt: &str) -> &str {
    let mut depth = 1i32;
    let mut end = after_lt.len();
    for (i, c) in after_lt.char_indices() {
        match c {
            '<' => depth += 1,
            '>' => {
                depth -= 1;
                if depth == 0 {
                    end = i + 1;
                    break;
                }
            }
            _ => {}
        }
    }
    &after_lt[end..]
}

/// Deterministic OLD text for a CALLSITE rewrite: the exact source lines
/// from `line_0` (0-based, the line the call expression starts on, per the
/// LSP-resolved reference position) through the line containing the call's
/// own balanced closing `)` — inclusive, whole lines (matching how the
/// model's NEW text is expected to reproduce the full line, e.g. `let x =
/// foo(...)`, not just the call expression). `column_0` (byte offset, same
/// convention as the rest of this module) skips any text BEFORE the call on
/// its first line so an unrelated earlier `(` on that line can't be
/// mistaken for the call's own parens.
///
/// Uses `scan_balanced_parens_from` directly, not `balanced_parens` — the
/// Go-receiver detection there is specific to function DEFINITIONS and
/// doesn't apply to call expressions. Returns `None` if no balanced `(...)`
/// follows (malformed input; caller falls back to the model-transcribes-OLD
/// path).
///
/// Same motivation as `signature_old_block`: real callsites in this
/// codebase (and presumably many rustfmt'd ones) put each argument on its
/// own line, and a live replay of a historical bench failure confirmed
/// multi-line callsite rewrites hit the identical "OLD line N doesn't match
/// source" failure the signature fix was built for (2026-07-05).
///
/// Guards against a reference that ISN'T actually a call — e.g. the
/// function used as a plain value (`let f = foo;`, a function pointer) —
/// by requiring the identifier at `column_0` to be immediately followed by
/// `(`, after skipping any further `::segment` path continuations (each
/// with an optional turbofish `<...>`). The loop handles both column
/// conventions an LSP might report for a qualified call like
/// `context::assemble(...)`: pointing at the last segment (`assemble`,
/// standard `textDocument/references` behavior — nothing left to skip) or,
/// defensively, at an earlier segment (`context` — the loop walks forward
/// through the remaining `::` segments to reach the real parens). Without
/// this guard at all, the scan would walk forward past a non-call
/// reference and grab the next unrelated `(...)` in the file, wherever it
/// happened to be (confirmed live: `let f = foo;\nbar(1, 2);` matched
/// `bar`'s parens as if they belonged to `foo`). A false-negative here
/// (declining a genuine but oddly-formatted call, e.g. a comment between
/// the name and `(`) is safe — it just falls back to the
/// model-transcribes-OLD path; only a wrong-but-present span would be
/// dangerous, and this guard prevents that.
pub fn callsite_old_block(source: &str, line_0: u32, column_0: u32) -> Option<String> {
    let lines: Vec<&str> = source.lines().collect();
    let start_line = line_0 as usize;
    let first_line = *lines.get(start_line)?;
    let col = (column_0 as usize).min(first_line.len());
    let mut tail = first_line[col..].to_string();
    for l in lines.get(start_line + 1..).unwrap_or_default() {
        tail.push('\n');
        tail.push_str(l);
    }

    let after_ident = tail
        .find(|c: char| !(c.is_alphanumeric() || c == '_'))
        .unwrap_or(tail.len());
    let mut rest = tail[after_ident..].trim_start();
    while let Some(after_colons) = rest.strip_prefix("::") {
        let after_colons = after_colons.trim_start();
        // Turbofish directly after "::" — `foo::<u32>(...)`, no
        // identifier segment in between.
        if let Some(after_lt) = after_colons.strip_prefix('<') {
            rest = skip_generic_args(after_lt).trim_start();
            continue;
        }
        // Otherwise a namespace/path segment — `context::assemble`,
        // optionally with its OWN turbofish (`Vec::<u32>::new`).
        let seg_end = after_colons
            .find(|c: char| !(c.is_alphanumeric() || c == '_'))
            .unwrap_or(after_colons.len());
        if seg_end == 0 {
            break; // "::" not followed by an identifier or turbofish — stop.
        }
        let remainder = &after_colons[seg_end..];
        rest = match remainder.strip_prefix('<') {
            Some(after_lt) => skip_generic_args(after_lt).trim_start(),
            None => remainder.trim_start(),
        };
    }
    if !rest.starts_with('(') {
        return None;
    }

    let (_, close) = scan_balanced_parens_from(&tail, 0)?;
    let close_line_offset = tail[..close].matches('\n').count();
    let end_line = start_line + close_line_offset;
    if end_line >= lines.len() {
        return None;
    }
    Some(lines[start_line..=end_line].join("\n"))
}

#[cfg(test)]
mod signature_old_block_tests {
    use super::signature_old_block;

    #[test]
    fn single_line_signature() {
        let src = "fn foo(a: u32, b: u32) -> u32 {\n    a + b\n}\n";
        assert_eq!(
            signature_old_block(src, 0),
            Some("fn foo(a: u32, b: u32) -> u32 {".to_string())
        );
    }

    #[test]
    fn multi_line_signature_includes_trailing_return_type_line() {
        let src = "pub fn assemble(\n    config: &Config,\n    mcp_summary: Option<&str>,\n) -> AssembledContext {\n    todo!()\n}\n";
        assert_eq!(
            signature_old_block(src, 0),
            Some(
                "pub fn assemble(\n    config: &Config,\n    mcp_summary: Option<&str>,\n) -> AssembledContext {"
                    .to_string()
            )
        );
    }

    #[test]
    fn from_line_offset_into_the_middle_of_a_file() {
        let src = "use std::foo;\n\nfn bar(x: u32) -> u32 {\n    x\n}\n";
        assert_eq!(
            signature_old_block(src, 2),
            Some("fn bar(x: u32) -> u32 {".to_string())
        );
    }

    #[test]
    fn no_parens_returns_none() {
        let src = "const X: u32 = 5;\n";
        assert_eq!(signature_old_block(src, 0), None);
    }

    #[test]
    fn nested_parens_in_default_value_dont_confuse_the_balance() {
        let src = "fn foo(a: (u32, u32), b: u32) -> u32 {\n    b\n}\n";
        assert_eq!(
            signature_old_block(src, 0),
            Some("fn foo(a: (u32, u32), b: u32) -> u32 {".to_string())
        );
    }

    // ── cross-language sanity (the compile-gate in plan::actions supports
    // Rust, TypeScript/JS, Go, Python, Java/Gradle — signature_old_block
    // needs to find the real parameter list in all of them) ──────────────

    #[test]
    fn go_plain_function() {
        let src = "func foo(a int, b int) int {\n\treturn a + b\n}\n";
        assert_eq!(
            signature_old_block(src, 0),
            Some("func foo(a int, b int) int {".to_string())
        );
    }

    #[test]
    fn go_method_with_receiver_skips_the_receiver_clause() {
        // The receiver clause `(r *T)` is NOT the parameter list. On a
        // single line this wouldn't distinguish a buggy vs. fixed impl (the
        // whole-line span is identical either way), so the params span
        // MULTIPLE lines here — using the receiver's `)` (buggy) would
        // return just line 0, an incomplete, syntactically broken OLD
        // block; using the real params' `)` (fixed) returns all 4 lines.
        let src = "func (r *T) foo(\n\ta int,\n\tb int,\n) int {\n\treturn a + b\n}\n";
        assert_eq!(
            signature_old_block(src, 0),
            Some("func (r *T) foo(\n\ta int,\n\tb int,\n) int {".to_string())
        );
    }

    #[test]
    fn go_receiver_without_pointer() {
        let src = "func (t T) foo(a int, b string) bool {\n\treturn true\n}\n";
        assert_eq!(
            signature_old_block(src, 0),
            Some("func (t T) foo(a int, b string) bool {".to_string())
        );
    }

    #[test]
    fn rust_function_returning_a_closure_is_not_mistaken_for_a_go_receiver() {
        // First `(...)` is the real param list; what follows (`-> impl
        // Fn(i32) -> i32 {`) starts with `->`, not an identifier, so this
        // must NOT be treated as a receiver clause.
        let src = "pub fn make_adder(x: i32) -> impl Fn(i32) -> i32 {\n    todo!()\n}\n";
        assert_eq!(
            signature_old_block(src, 0),
            Some("pub fn make_adder(x: i32) -> impl Fn(i32) -> i32 {".to_string())
        );
    }

    #[test]
    fn python_function() {
        let src = "def foo(a, b):\n    return a + b\n";
        assert_eq!(
            signature_old_block(src, 0),
            Some("def foo(a, b):".to_string())
        );
    }

    #[test]
    fn python_method_self_is_a_plain_param_not_a_receiver() {
        let src = "def foo(self, a, b):\n    return a + b\n";
        assert_eq!(
            signature_old_block(src, 0),
            Some("def foo(self, a, b):".to_string())
        );
    }

    #[test]
    fn typescript_function() {
        let src = "function foo(a: number, b: number): number {\n    return a + b;\n}\n";
        assert_eq!(
            signature_old_block(src, 0),
            Some("function foo(a: number, b: number): number {".to_string())
        );
    }

    #[test]
    fn java_method() {
        let src = "public int foo(int a, int b) {\n    return a + b;\n}\n";
        assert_eq!(
            signature_old_block(src, 0),
            Some("public int foo(int a, int b) {".to_string())
        );
    }

    #[test]
    fn python_default_value_string_with_a_paren_doesnt_confuse_the_balance() {
        let src = "def foo(a, msg=\"(unbalanced\"):\n    pass\n";
        assert_eq!(
            signature_old_block(src, 0),
            Some("def foo(a, msg=\"(unbalanced\"):".to_string())
        );
    }
}

#[cfg(test)]
mod callsite_old_block_tests {
    use super::callsite_old_block;

    #[test]
    fn single_line_call() {
        let src = "    let x = foo(1, 2);\n";
        // column_0 = 12 → the byte offset of `foo` (after "    let x = ").
        assert_eq!(
            callsite_old_block(src, 0, 12),
            Some("    let x = foo(1, 2);".to_string())
        );
    }

    #[test]
    fn multi_line_call_one_arg_per_line() {
        // Matches the real rustfmt style seen in run.rs/repl.rs's
        // context::assemble(...) calls that historically failed.
        let src =
            "    let assembled = context::assemble(\n        &config,\n        message,\n    );\n";
        assert_eq!(
            callsite_old_block(src, 0, 25),
            Some(
                "    let assembled = context::assemble(\n        &config,\n        message,\n    );"
                    .to_string()
            )
        );
    }

    #[test]
    fn column_skips_an_earlier_unrelated_paren_on_the_same_line() {
        // An earlier call `bar(z)` on the same line must not be mistaken
        // for the target call `foo(...)` — column_0 anchors past it.
        let src = "    let x = bar(z) + foo(1, 2);\n";
        let col = src.lines().next().unwrap().find("foo").unwrap();
        assert_eq!(
            callsite_old_block(src, 0, col as u32),
            Some("    let x = bar(z) + foo(1, 2);".to_string())
        );
    }

    #[test]
    fn no_parens_after_column_returns_none() {
        let src = "    let x = 5;\n";
        assert_eq!(callsite_old_block(src, 0, 12), None);
    }

    // ── correctness gaps found by the user's edge-case review, 2026-07-05:
    // string arguments containing brackets, method calls as parameters,
    // and references that aren't calls at all ─────────────────────────

    #[test]
    fn string_argument_containing_unbalanced_paren() {
        // Previously this returned None (a clean, if wrong, failure).
        let src = r#"    foo("hello (world", x);"#;
        let col = src.find("foo").unwrap();
        assert_eq!(
            callsite_old_block(src, 0, col as u32),
            Some(r#"    foo("hello (world", x);"#.to_string())
        );
    }

    #[test]
    fn string_argument_containing_paren_that_looks_like_a_close() {
        // The dangerous case: `("a) b(c"` previously matched `("a)` — an
        // arbitrary WRONG span, silently, not an error. The real call's
        // close is the final `)` before `;`.
        let src = r#"    foo("a) b(c", x);"#;
        let col = src.find("foo").unwrap();
        assert_eq!(
            callsite_old_block(src, 0, col as u32),
            Some(r#"    foo("a) b(c", x);"#.to_string())
        );
    }

    #[test]
    fn method_call_as_a_parameter() {
        let src = "    foo(bar(x), y);\n";
        let col = src.find("foo").unwrap();
        assert_eq!(
            callsite_old_block(src, 0, col as u32),
            Some("    foo(bar(x), y);".to_string())
        );
    }

    #[test]
    fn reference_used_as_a_value_not_called_returns_none() {
        // `foo` here is a function pointer, not a call — must NOT walk
        // forward and grab `bar`'s parens on the next line.
        let src = "let f = foo;\nbar(1, 2);\n";
        let col = src.lines().next().unwrap().find("foo").unwrap();
        assert_eq!(callsite_old_block(src, 0, col as u32), None);
    }

    #[test]
    fn comment_containing_a_paren_is_not_mistaken_for_the_close() {
        let src = "    foo(a /* unbalanced ( in a comment */, b);\n";
        let col = src.find("foo").unwrap();
        assert_eq!(
            callsite_old_block(src, 0, col as u32),
            Some("    foo(a /* unbalanced ( in a comment */, b);".to_string())
        );
    }

    #[test]
    fn turbofish_generic_call_is_recognized() {
        let src = "    foo::<u32>(a, b);\n";
        let col = src.find("foo").unwrap();
        assert_eq!(
            callsite_old_block(src, 0, col as u32),
            Some("    foo::<u32>(a, b);".to_string())
        );
    }

    #[test]
    fn column_pointing_at_an_earlier_namespace_segment_still_finds_the_call() {
        // Defensive: some LSP might resolve the reference to the FIRST
        // segment of a qualified path rather than the last. Either way the
        // guard must walk forward through the `::segment` chain to reach
        // the real parens.
        let src = "    let x = context::assemble(\n        &config,\n    );\n";
        let col = src.find("context").unwrap();
        assert_eq!(
            callsite_old_block(src, 0, col as u32),
            Some("    let x = context::assemble(\n        &config,\n    );".to_string())
        );
    }

    #[test]
    fn turbofish_on_a_namespaced_segment() {
        let src = "    Vec::<u32>::new();\n";
        let col = src.find("Vec").unwrap();
        assert_eq!(
            callsite_old_block(src, 0, col as u32),
            Some("    Vec::<u32>::new();".to_string())
        );
    }

    #[test]
    fn macro_invocation_argument_containing_parens() {
        // `println!` isn't itself a target callsite (macros aren't plain
        // function calls), but a real call INSIDE one must still resolve
        // correctly — the macro's own `!(...)` delimiters mustn't confuse
        // the scan for `foo`'s parens.
        let src = "    println!(\"{}\", foo(a, b));\n";
        let col = src.find("foo").unwrap();
        assert_eq!(
            callsite_old_block(src, 0, col as u32),
            Some("    println!(\"{}\", foo(a, b));".to_string())
        );
    }
}

#[cfg(test)]
mod tricky_callsite_tests {
    //! Cross-language edge-case matrix for `callsite_old_block`'s
    //! lexer-aware paren scan, requested 2026-07-05 as a "how sure are
    //! you" sanity check. Every PASS here is real confidence; a FAIL
    //! (there was one — Go backtick strings, now fixed) is the actual
    //! measure of the hand-rolled scanner's honest limits, not a guess.
    use super::callsite_old_block;

    fn check(src: &str, expected: &str) {
        let col = src.find("foo").expect("fixture must contain `foo`");
        assert_eq!(
            callsite_old_block(src, 0, col as u32),
            Some(expected.to_string()),
            "src: {src:?}"
        );
    }

    #[test]
    fn rust_raw_string_with_regex_parens() {
        check(r#"foo(r"(\d+)", x);"#, r#"foo(r"(\d+)", x);"#);
    }

    #[test]
    fn rust_hash_delimited_raw_string_containing_a_literal_quote() {
        check(
            r##"foo(r#"a "quoted" (paren)"#, x);"##,
            r##"foo(r#"a "quoted" (paren)"#, x);"##,
        );
    }

    #[test]
    fn rust_byte_string() {
        check(r#"foo(b"(", x);"#, r#"foo(b"(", x);"#);
    }

    #[test]
    fn rust_char_literal_paren() {
        check("foo('(', x);", "foo('(', x);");
    }

    #[test]
    fn python_triple_quoted_string_with_internal_quote_and_paren() {
        check(
            r#"foo("""a "quoted" (paren)""", x);"#,
            r#"foo("""a "quoted" (paren)""", x);"#,
        );
    }

    #[test]
    fn python_fstring_with_embedded_call() {
        check(r#"foo(f"{bar(x)}", y);"#, r#"foo(f"{bar(x)}", y);"#);
    }

    #[test]
    fn go_raw_string_backtick() {
        check("foo(`(unbalanced`, x);", "foo(`(unbalanced`, x);");
    }

    #[test]
    fn typescript_template_literal_with_embedded_call() {
        check("foo(`${bar(\"(\")}`, y);", "foo(`${bar(\"(\")}`, y);");
    }

    #[test]
    fn javascript_regex_literal_with_parens() {
        check(r#"foo(/\(\d+\)/, x);"#, r#"foo(/\(\d+\)/, x);"#);
    }

    #[test]
    fn javascript_arrow_function_as_argument() {
        check("foo((x) => x + 1, y);", "foo((x) => x + 1, y);");
    }

    #[test]
    fn java_text_block() {
        check(
            "foo(\"\"\"a \"quoted\" (paren)\"\"\", x);",
            "foo(\"\"\"a \"quoted\" (paren)\"\"\", x);",
        );
    }

    #[test]
    fn java_lambda_as_argument() {
        check("foo(x -> x + 1, y);", "foo(x -> x + 1, y);");
    }

    #[test]
    fn nested_block_comment() {
        check(
            "foo(a /* outer /* inner */ still comment */, b);",
            "foo(a /* outer /* inner */ still comment */, b);",
        );
    }
}
