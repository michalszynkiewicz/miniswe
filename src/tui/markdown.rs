//! Minimal line-based markdown styling for assistant transcript lines.
//!
//! Handled: fenced code blocks, `#` headings, thematic breaks, `*`/`-`/`+`
//! bullets, `` `code` `` and `**bold**`. Everything else (italics, links,
//! tables, numbered lists) is left literal on purpose. Runs every frame over
//! the whole transcript, so output borrows from the input text.

use std::borrow::Cow;

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

/// Carried between consecutive assistant lines.
#[derive(Default, Clone, Copy)]
pub(super) struct MdState {
    in_code: bool,
}

/// Colours used by the styler.
pub(super) struct MdTheme {
    pub text: Color,
    pub heading: Color,
    pub code: Color,
    pub muted: Color,
    pub bg: Color,
}

/// Style one assistant line. `None` means the line renders as nothing (a
/// fence marker without a language label).
pub(super) fn render_line<'a>(
    text: &'a str,
    state: &mut MdState,
    width: usize,
    theme: &MdTheme,
) -> Option<Line<'a>> {
    let base = Style::default().fg(theme.text).bg(theme.bg);
    let muted = Style::default().fg(theme.muted).bg(theme.bg);
    let trimmed = text.trim_start();

    // Fences.
    if let Some(rest) = trimmed.strip_prefix("```") {
        if state.in_code {
            if rest.trim_end().chars().all(|c| c == '`') {
                state.in_code = false;
                return None;
            }
        } else if !rest.contains('`') {
            state.in_code = true;
            let label = rest.trim();
            return if label.is_empty() {
                None
            } else {
                Some(Line::from(vec![
                    Span::styled("  ", muted),
                    Span::styled(label, muted),
                ]))
            };
        }
    }

    if state.in_code {
        let code = base.fg(theme.code);
        return Some(Line::from(vec![
            Span::styled("  ", code),
            Span::styled(expand_tabs(text), code),
        ]));
    }

    // Heading: 1-6 '#' then a space, at column 0.
    let hashes = text.bytes().take_while(|&b| b == b'#').count();
    if (1..=6).contains(&hashes) && text.as_bytes().get(hashes) == Some(&b' ') {
        let style = base.fg(theme.heading).add_modifier(Modifier::BOLD);
        let mut spans = Vec::new();
        inline(&text[hashes + 1..], style, theme, &mut spans);
        return Some(Line::from(spans));
    }

    // Thematic break.
    let t = text.trim();
    if t.len() >= 3
        && ["-", "*", "_"]
            .iter()
            .any(|m| t.chars().all(|c| m.starts_with(c)))
    {
        return Some(Line::from(Span::styled("─".repeat(width), muted)));
    }

    // Bullet.
    let indent = text.len() - trimmed.len();
    if text[..indent].bytes().all(|b| b == b' ')
        && let Some(rest) = ["* ", "- ", "+ "]
            .iter()
            .find_map(|m| trimmed.strip_prefix(m))
    {
        let mut spans = vec![
            Span::styled(&text[..indent], base),
            Span::styled("• ", base),
        ];
        inline(rest, base, theme, &mut spans);
        return Some(Line::from(spans));
    }

    let mut spans = Vec::new();
    inline(text, base, theme, &mut spans);
    if spans.is_empty() {
        spans.push(Span::styled(text, base));
    }
    Some(Line::from(spans))
}

/// Ratatui drops control characters, which glues tab-separated columns and
/// removes tab indentation, so code lines get 4-column tab stops instead.
fn expand_tabs(text: &str) -> Cow<'_, str> {
    if !text.contains('\t') {
        return Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(text.len() + 8);
    let mut col = 0;
    for ch in text.chars() {
        if ch == '\t' {
            let pad = 4 - col % 4;
            out.extend(std::iter::repeat_n(' ', pad));
            col += pad;
        } else {
            out.push(ch);
            col += 1;
        }
    }
    Cow::Owned(out)
}

/// Inline `code` and `**bold**` parsing, appended to `out`.
fn inline<'a>(text: &'a str, base: Style, theme: &MdTheme, out: &mut Vec<Span<'a>>) {
    let mut lit = 0; // start of the pending literal run
    let mut i = 0; // scan position
    while i < text.len() {
        let rest = &text[i..];
        if let Some(tail) = rest.strip_prefix('`') {
            if let Some(len) = tail.find('`') {
                if len > 0 {
                    flush(text, lit, i, base, out);
                    out.push(Span::styled(&text[i + 1..i + 1 + len], base.fg(theme.code)));
                    i += len + 2;
                    lit = i;
                } else {
                    i += 2; // empty pair: both stay literal
                }
            } else {
                i += 1;
            }
        } else if let Some(after) = rest.strip_prefix("**") {
            match after.find("**") {
                Some(len) if bold_ok(&after[..len]) => {
                    flush(text, lit, i, base, out);
                    inline(&after[..len], base.add_modifier(Modifier::BOLD), theme, out);
                    i += len + 4;
                    lit = i;
                }
                _ => i += 2,
            }
        } else {
            // Advance one char (all markers are ASCII, so this stays on a
            // char boundary).
            i += rest.chars().next().map_or(1, char::len_utf8);
        }
    }
    flush(text, lit, text.len(), base, out);
}

fn bold_ok(inner: &str) -> bool {
    !inner.is_empty() && !inner.starts_with(' ') && !inner.ends_with(' ')
}

fn flush<'a>(text: &'a str, from: usize, to: usize, base: Style, out: &mut Vec<Span<'a>>) {
    if to > from {
        out.push(Span::styled(&text[from..to], base));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const THEME: MdTheme = MdTheme {
        text: Color::White,
        heading: Color::Cyan,
        code: Color::Yellow,
        muted: Color::DarkGray,
        bg: Color::Black,
    };

    fn plain(line: &Option<Line>) -> String {
        line.as_ref()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .unwrap_or_else(|| "<none>".into())
    }

    fn r(text: &str, st: &mut MdState) -> Option<Line<'static>> {
        // Own the spans so tests can drop `text`.
        render_line(text, st, 10, &THEME).map(|l| {
            Line::from(
                l.spans
                    .into_iter()
                    .map(|s| Span::styled(s.content.into_owned(), s.style))
                    .collect::<Vec<_>>(),
            )
        })
    }

    #[test]
    fn fences_hidden_and_label_shown() {
        let mut st = MdState::default();
        assert_eq!(plain(&r("```", &mut st)), "<none>");
        assert_eq!(plain(&r("let x = 1;", &mut st)), "  let x = 1;");
        assert_eq!(plain(&r("```", &mut st)), "<none>");
        assert_eq!(plain(&r("```rust", &mut st)), "  rust");
        assert!(st.in_code);
        assert_eq!(plain(&r("```", &mut st)), "<none>");
        assert!(!st.in_code);
    }

    #[test]
    fn code_is_literal() {
        let mut st = MdState::default();
        r("```", &mut st);
        for s in ["**x**", "* y", "#[derive(Debug)]", "# not heading", "---"] {
            let l = r(s, &mut st);
            assert_eq!(plain(&l), format!("  {s}"));
            assert_eq!(l.unwrap().spans[1].style.fg, Some(THEME.code));
        }
        // A "```rust" line inside a block is code, not a new fence.
        assert_eq!(plain(&r("```rust", &mut st)), "  ```rust");
        assert!(st.in_code);
    }

    #[test]
    fn code_tabs_expand_to_tab_stops() {
        let mut st = MdState::default();
        r("```", &mut st);
        assert_eq!(
            plain(&r("2026-10-01\trent\t120000", &mut st)),
            "  2026-10-01  rent    120000"
        );
        assert_eq!(plain(&r("\tindented", &mut st)), "      indented");
        assert_eq!(plain(&r("ab\t", &mut st)), "  ab  ");
    }

    #[test]
    fn derive_attribute_is_not_a_heading() {
        let mut st = MdState::default();
        assert_eq!(plain(&r("#[derive(Debug)]", &mut st)), "#[derive(Debug)]");
        assert_eq!(plain(&r("#foo", &mut st)), "#foo");
    }

    #[test]
    fn heading_markers_dropped_and_styled() {
        let mut st = MdState::default();
        let l = r("### `Item` struct", &mut st);
        assert_eq!(plain(&l), "Item struct");
        let l = l.unwrap();
        assert_eq!(l.spans[0].style.fg, Some(THEME.code));
        assert!(l.spans[0].style.add_modifier.contains(Modifier::BOLD));
        assert_eq!(l.spans[1].style.fg, Some(THEME.heading));
        assert!(l.spans[1].style.add_modifier.contains(Modifier::BOLD));
        assert_eq!(plain(&r("####### seven", &mut st)), "####### seven");
    }

    #[test]
    fn bold_markers_dropped_with_bold_modifier() {
        let mut st = MdState::default();
        let l = r("a **bold** b", &mut st);
        assert_eq!(plain(&l), "a bold b");
        let l = l.unwrap();
        assert!(l.spans[1].style.add_modifier.contains(Modifier::BOLD));
        assert!(!l.spans[0].style.add_modifier.contains(Modifier::BOLD));
    }

    #[test]
    fn backticks_dropped_with_code_colour_and_bold_nesting() {
        let mut st = MdState::default();
        let l = r("use `foo` and **`bar`**", &mut st);
        assert_eq!(plain(&l), "use foo and bar");
        let l = l.unwrap();
        assert_eq!(l.spans[1].style.fg, Some(THEME.code));
        let last = l.spans.last().unwrap();
        assert_eq!(last.style.fg, Some(THEME.code));
        assert!(last.style.add_modifier.contains(Modifier::BOLD));
    }

    #[test]
    fn stars_with_spaces_stay_literal() {
        let mut st = MdState::default();
        assert_eq!(plain(&r("cents * 100", &mut st)), "cents * 100");
        assert_eq!(plain(&r("2 ** 3 ** 4", &mut st)), "2 ** 3 ** 4");
        assert_eq!(plain(&r("`**x**`", &mut st)), "**x**");
    }

    #[test]
    fn bullets_and_rules() {
        let mut st = MdState::default();
        assert_eq!(plain(&r("* item", &mut st)), "• item");
        assert_eq!(plain(&r("  - **sub** x", &mut st)), "  • sub x");
        assert_eq!(plain(&r("+ plus", &mut st)), "• plus");
        assert_eq!(plain(&r("---", &mut st)), "──────────");
        assert_eq!(plain(&r("***", &mut st)), "──────────");
        assert_eq!(plain(&r("-x", &mut st)), "-x");
    }

    #[test]
    fn multibyte_is_safe() {
        let mut st = MdState::default();
        assert_eq!(plain(&r("**żółć** – `x`", &mut st)), "żółć – x");
        assert_eq!(plain(&r("żółć ** ą", &mut st)), "żółć ** ą");
        assert_eq!(plain(&r("日本語 `語", &mut st)), "日本語 `語");
        assert_eq!(plain(&r("# żółć", &mut st)), "żółć");
    }

    #[test]
    fn unterminated_markers_stay_literal() {
        let mut st = MdState::default();
        assert_eq!(plain(&r("**foo", &mut st)), "**foo");
        assert_eq!(plain(&r("`foo", &mut st)), "`foo");
        assert_eq!(plain(&r("a `` b", &mut st)), "a `` b");
        assert_eq!(plain(&r("a ****", &mut st)), "a ****");
    }
}
