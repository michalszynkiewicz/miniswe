//! `rename`: a thin wrapper over LSP `textDocument/rename`.
//!
//! Renames the symbol at the given location and applies every
//! `WorkspaceEdit` the server returns. Works for any symbol (function,
//! type, variable, parameter) — the LSP figures out the correct semantic
//! scope.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use lsp_types::{TextEdit, WorkspaceEdit};
use serde_json::Value;

use crate::config::Config;
use crate::lsp::{LspClient, uri_to_path};
use crate::tools::ToolResult;
use crate::tools::args;

use super::sites::ensure_ready;

pub async fn execute(args: &Value, config: &Config, lsp: Option<&LspClient>) -> Result<ToolResult> {
    let path_str = match args::require_str(args, "path") {
        Ok(p) => p,
        Err(e) => return Ok(ToolResult::err(e)),
    };
    let line_1 = match args::require_u64(args, "line") {
        Ok(n) => n,
        Err(e) => return Ok(ToolResult::err(e)),
    };
    let name = match args::require_str(args, "name") {
        Ok(n) => n,
        Err(e) => return Ok(ToolResult::err(e)),
    };
    let new_name = match args::require_str(args, "new_name") {
        Ok(p) => p,
        Err(e) => return Ok(ToolResult::err(e)),
    };

    let Some(lsp) = lsp else {
        return Ok(ToolResult::err(
            "rename requires LSP support (no LSP client available for this project)".into(),
        ));
    };
    if let Err(e) = ensure_ready(lsp, Duration::from_secs(60)).await {
        return Ok(ToolResult::err(format!(
            "LSP not ready in time: {e}. Try again in a moment."
        )));
    }

    let abs_path = config.project_root.join(path_str);
    let line_0: u32 = (line_1.saturating_sub(1)) as u32;

    // Find the column where `name` appears on the given line. Rename works
    // for any symbol kind (variable, parameter, field, type, function), so
    // we don't lean on documentSymbol — just search the line text. If the
    // identifier isn't on that line, give the agent an actionable error
    // instead of letting the LSP guess at whatever symbol happens to be at
    // the cursor.
    let source = std::fs::read_to_string(&abs_path).with_context(|| format!("read {path_str}"))?;
    let column_0 = match find_identifier_column(&source, line_0, name) {
        Some(c) => c,
        None => {
            return Ok(ToolResult::err(format!(
                "✗ rename: identifier `{name}` not found on line {line_1} of {path_str}. \
                 Check the line number and spelling, or use code(find_references) to locate it."
            )));
        }
    };

    let edit = match lsp.rename(&abs_path, line_0, column_0, new_name).await {
        Ok(Some(e)) => e,
        Ok(None) => {
            return Ok(ToolResult::err(format!(
                "LSP rejected the rename (server returned null). \
                 The symbol `{name}` at {path_str}:{line_1} may not be renameable, \
                 or `{new_name}` is invalid in this context."
            )));
        }
        Err(e) => {
            return Ok(ToolResult::err(format!("LSP rename failed: {e}")));
        }
    };

    let summary = apply_workspace_edit(edit, config)?;
    Ok(ToolResult::ok(summary))
}

/// Locate the first occurrence of `name` on the given (0-based) line of
/// `source`. Returns the 0-based column of its first character, or None
/// when the line is too short or the identifier isn't there. We require
/// the surrounding chars to be non-identifier so `foo` doesn't match
/// `foobar`.
fn find_identifier_column(source: &str, line_0: u32, name: &str) -> Option<u32> {
    let line = source.lines().nth(line_0 as usize)?;
    let mut start = 0;
    while let Some(idx) = line[start..].find(name) {
        let abs = start + idx;
        let before_ok = abs == 0 || !line[..abs].chars().next_back().is_some_and(is_ident_char);
        let after_ok = !line[abs + name.len()..]
            .chars()
            .next()
            .is_some_and(is_ident_char);
        if before_ok && after_ok {
            // line is &str; column is char count of byte prefix. For ASCII
            // it's identical; this stays correct for non-ASCII identifiers.
            return Some(line[..abs].chars().count() as u32);
        }
        start = abs + name.len();
    }
    None
}

fn is_ident_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Apply a `WorkspaceEdit` from `textDocument/rename`. Both `changes`
/// (URI → list of edits) and `documentChanges` (list of edits with
/// optional version) are supported; we just map both into the same
/// per-file edit list.
fn apply_workspace_edit(edit: WorkspaceEdit, config: &Config) -> Result<String> {
    let mut per_file: BTreeMap<PathBuf, Vec<TextEdit>> = BTreeMap::new();

    if let Some(changes) = edit.changes {
        for (uri, edits) in changes {
            if let Some(path) = uri_to_path(&uri) {
                per_file.entry(path).or_default().extend(edits);
            }
        }
    }
    if let Some(doc_changes) = edit.document_changes {
        use lsp_types::DocumentChanges;
        match doc_changes {
            DocumentChanges::Edits(edits) => {
                for tde in edits {
                    if let Some(path) = uri_to_path(&tde.text_document.uri) {
                        let only_text_edits: Vec<TextEdit> = tde
                            .edits
                            .into_iter()
                            .map(|edit| {
                                use lsp_types::OneOf;
                                match edit {
                                    OneOf::Left(te) => te,
                                    // Annotated edits also carry a TextEdit.
                                    OneOf::Right(annotated) => annotated.text_edit,
                                }
                            })
                            .collect();
                        per_file.entry(path).or_default().extend(only_text_edits);
                    }
                }
            }
            DocumentChanges::Operations(_) => {
                // create/rename/delete file ops are not produced by simple
                // symbol renames in any of our LSPs; bail loudly so we
                // notice if a server starts emitting them.
                anyhow::bail!(
                    "LSP returned file-level operations for rename — not supported in v1"
                );
            }
        }
    }

    if per_file.is_empty() {
        return Ok("rename: LSP returned an empty WorkspaceEdit (nothing to apply)".into());
    }

    let mut total_edits = 0usize;
    let mut report_lines = Vec::new();
    for (path, mut edits) in per_file {
        // Apply edits bottom-up so earlier edits don't shift later positions.
        edits.sort_by(|a, b| {
            b.range
                .start
                .line
                .cmp(&a.range.start.line)
                .then_with(|| b.range.start.character.cmp(&a.range.start.character))
        });
        let original = std::fs::read_to_string(&path)
            .with_context(|| format!("read {} during rename", path.display()))?;
        let mut updated = original;
        for edit in &edits {
            updated = apply_text_edit(&updated, edit)
                .with_context(|| format!("apply edit in {}", path.display()))?;
        }
        std::fs::write(&path, &updated)
            .with_context(|| format!("write {} after rename", path.display()))?;
        total_edits += edits.len();
        let rel = path
            .strip_prefix(&config.project_root)
            .unwrap_or(&path)
            .display()
            .to_string();
        report_lines.push(format!("  • {rel}: {} edit(s)", edits.len()));
    }

    let mut out = format!(
        "✓ rename: {total_edits} edit(s) applied across {} file(s).\n",
        report_lines.len()
    );
    for line in &report_lines {
        out.push_str(line);
        out.push('\n');
    }
    out.push_str(
        "\nNext: run code(diagnostics) or your build to confirm the project still compiles.",
    );
    Ok(out)
}

/// Apply a single LSP `TextEdit` to `source` (which is assumed to be the
/// current contents of the target file). LSP positions use UTF-16 code
/// units in the spec; in practice, for ASCII identifiers the difference
/// from byte offsets and char offsets is moot. We treat positions as char
/// offsets which is what every server we use produces in practice for
/// pure-ASCII renames.
fn apply_text_edit(source: &str, edit: &TextEdit) -> Result<String> {
    let lines: Vec<&str> = source.split_inclusive('\n').collect();
    let start_byte = pos_to_byte(&lines, edit.range.start.line, edit.range.start.character)?;
    let end_byte = pos_to_byte(&lines, edit.range.end.line, edit.range.end.character)?;
    let mut out = String::with_capacity(source.len() + edit.new_text.len());
    out.push_str(&source[..start_byte]);
    out.push_str(&edit.new_text);
    out.push_str(&source[end_byte..]);
    Ok(out)
}

fn pos_to_byte(lines: &[&str], line: u32, character: u32) -> Result<usize> {
    let line_idx = line as usize;
    if line_idx > lines.len() {
        anyhow::bail!(
            "LSP position line {line} out of range (file has {} lines)",
            lines.len()
        );
    }
    let mut byte = 0usize;
    for (i, l) in lines.iter().enumerate() {
        if i == line_idx {
            // Walk char positions within this line.
            for (chars_seen, (off, _ch)) in l.char_indices().enumerate() {
                if chars_seen as u32 == character {
                    return Ok(byte + off);
                }
            }
            // Position past the end of the line content (e.g. end of last
            // line in a file with no trailing newline) — clamp to length.
            return Ok(byte + l.len());
        }
        byte += l.len();
    }
    // Position at the very end of the file (line == lines.len()).
    Ok(byte)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lsp_types::{Position, Range};

    fn te(start_line: u32, start_col: u32, end_line: u32, end_col: u32, text: &str) -> TextEdit {
        TextEdit {
            range: Range {
                start: Position {
                    line: start_line,
                    character: start_col,
                },
                end: Position {
                    line: end_line,
                    character: end_col,
                },
            },
            new_text: text.into(),
        }
    }

    #[test]
    fn apply_single_line_replace() {
        let src = "fn foo() {}\n";
        let out = apply_text_edit(src, &te(0, 3, 0, 6, "bar")).unwrap();
        assert_eq!(out, "fn bar() {}\n");
    }

    #[test]
    fn apply_multi_line_replace() {
        let src = "a\nbb\nccc\n";
        let out = apply_text_edit(src, &te(0, 1, 2, 1, "X")).unwrap();
        assert_eq!(out, "aXcc\n");
    }

    #[test]
    fn apply_at_end_of_file_no_newline() {
        let src = "fn foo()";
        let out = apply_text_edit(src, &te(0, 3, 0, 6, "bar")).unwrap();
        assert_eq!(out, "fn bar()");
    }
}
