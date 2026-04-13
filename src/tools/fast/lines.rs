//! Line-based string manipulation helpers for fast-mode edits.
//!
//! `replace_range` and `insert_at` both operate on 1-based inclusive
//! line numbers. These helpers centralize the split/rejoin logic and
//! keep trailing-newline handling consistent across tools.

/// Split `content` into lines preserving empty-trailing-line semantics.
/// Returns `(lines, had_trailing_newline)` — the flag lets the caller
/// re-emit the file in its original shape.
pub fn split_preserving_trailing_nl(content: &str) -> (Vec<&str>, bool) {
    let had_nl = content.ends_with('\n');
    let body = if had_nl { &content[..content.len() - 1] } else { content };
    // str::split on empty string yields `[""]`; that matches the intent
    // (a zero-byte file has one "line" that is empty).
    let lines: Vec<&str> = if body.is_empty() {
        vec![""]
    } else {
        body.split('\n').collect()
    };
    (lines, had_nl)
}

/// Re-join lines with `\n`, appending a trailing newline if the original
/// file had one.
pub fn join_with_trailing_nl(lines: &[String], had_trailing_nl: bool) -> String {
    let mut out = lines.join("\n");
    if had_trailing_nl {
        out.push('\n');
    }
    out
}

/// Split an incoming replacement string into logical lines, *without*
/// any trailing-newline fiddling: the user's content is inserted verbatim.
pub fn split_replacement(content: &str) -> Vec<String> {
    // Treat "" as a single empty line (consistent with what
    // `replace_range ... ""` should mean). `"\n"` → two lines ["", ""]
    // which would duplicate a blank. Callers that want pure deletion
    // should use the empty string.
    if content.is_empty() {
        return vec![String::new()];
    }
    // Don't treat a trailing newline as an extra line: `"foo\n"` means
    // one line `"foo"`. Without this, insertion content with a trailing
    // newline would silently add a blank row.
    let body = content.strip_suffix('\n').unwrap_or(content);
    body.split('\n').map(|s| s.to_string()).collect()
}

/// Validate 1-based inclusive range bounds against a line count.
/// Returns `Ok(())` or a user-facing error message.
pub fn validate_range(start: usize, end: usize, line_count: usize) -> Result<(), String> {
    if start == 0 {
        return Err("start must be >= 1 (line numbers are 1-based)".into());
    }
    if end < start {
        return Err(format!("end ({end}) < start ({start})"));
    }
    if start > line_count {
        return Err(format!(
            "start ({start}) is past end of file ({line_count} line(s))"
        ));
    }
    if end > line_count {
        return Err(format!(
            "end ({end}) is past end of file ({line_count} line(s))"
        ));
    }
    Ok(())
}

/// Validate 0-based-or-1-based insertion anchor: `0 <= after_line <= line_count`.
pub fn validate_insertion_anchor(after_line: usize, line_count: usize) -> Result<(), String> {
    if after_line > line_count {
        return Err(format!(
            "after_line ({after_line}) is past end of file ({line_count} line(s)); use after_line = {line_count} to append"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_preserves_trailing_newline() {
        let (lines, had) = split_preserving_trailing_nl("a\nb\nc\n");
        assert_eq!(lines, vec!["a", "b", "c"]);
        assert!(had);

        let (lines, had) = split_preserving_trailing_nl("a\nb\nc");
        assert_eq!(lines, vec!["a", "b", "c"]);
        assert!(!had);
    }

    #[test]
    fn split_empty_is_one_empty_line() {
        let (lines, had) = split_preserving_trailing_nl("");
        assert_eq!(lines, vec![""]);
        assert!(!had);
    }

    #[test]
    fn join_reinstates_trailing_newline() {
        let v = vec!["a".into(), "b".into()];
        assert_eq!(join_with_trailing_nl(&v, true), "a\nb\n");
        assert_eq!(join_with_trailing_nl(&v, false), "a\nb");
    }

    #[test]
    fn split_replacement_empty_is_single_empty_line() {
        assert_eq!(split_replacement(""), vec![""]);
    }

    #[test]
    fn split_replacement_trailing_newline_not_duplicated() {
        assert_eq!(split_replacement("x\n"), vec!["x"]);
        assert_eq!(split_replacement("x\ny\n"), vec!["x", "y"]);
    }

    #[test]
    fn validate_range_accepts_valid() {
        assert!(validate_range(1, 1, 10).is_ok());
        assert!(validate_range(1, 10, 10).is_ok());
        assert!(validate_range(5, 7, 10).is_ok());
    }

    #[test]
    fn validate_range_rejects_bad() {
        assert!(validate_range(0, 1, 10).is_err());
        assert!(validate_range(5, 4, 10).is_err());
        assert!(validate_range(11, 11, 10).is_err());
        assert!(validate_range(1, 11, 10).is_err());
    }

    #[test]
    fn validate_insertion_anchor_accepts_zero_through_line_count() {
        assert!(validate_insertion_anchor(0, 10).is_ok());
        assert!(validate_insertion_anchor(5, 10).is_ok());
        assert!(validate_insertion_anchor(10, 10).is_ok());
    }

    #[test]
    fn validate_insertion_anchor_rejects_past_end() {
        assert!(validate_insertion_anchor(11, 10).is_err());
    }
}
