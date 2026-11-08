//! TUI rendering with ratatui.
//!
//! Draws the split-pane layout: scrollable output + input line.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, Clear, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState, Wrap,
};

use super::app::{App, AppMode, LineStyle};

const BG_MAIN: Color = Color::Rgb(18, 22, 28);
const BG_PANEL: Color = Color::Rgb(24, 29, 36);
const BG_INPUT: Color = Color::Rgb(20, 25, 32);
const BORDER: Color = Color::Rgb(45, 52, 62);
const TEXT_PRIMARY: Color = Color::Rgb(220, 226, 235);
const TEXT_SECONDARY: Color = Color::Rgb(150, 160, 175);
const TEXT_MUTED: Color = Color::Rgb(110, 120, 135);
const TEXT_SOFT: Color = Color::Rgb(185, 194, 208);
const ACCENT_CYAN: Color = Color::Rgb(80, 200, 255);
const ACCENT_PURPLE: Color = Color::Rgb(170, 120, 255);
const SUCCESS: Color = Color::Rgb(100, 210, 140);
const WARNING: Color = Color::Rgb(255, 190, 90);
const ERROR: Color = Color::Rgb(255, 95, 95);
const USER_LINE: Color = Color::Rgb(95, 165, 255);
const CMD_LINE: Color = Color::Rgb(140, 220, 170);

/// Render the full UI.
pub fn draw(frame: &mut Frame, app: &App) {
    match app.mode {
        AppMode::Normal => draw_normal(frame, app),
        AppMode::Detail => draw_detail(frame, app),
    }
}

/// Normal mode: optional plan panel + output pane + input line.
fn draw_normal(frame: &mut Frame, app: &App) {
    let area = frame.area();
    frame.render_widget(Block::default().style(Style::default().bg(BG_MAIN)), area);

    // The plan panel appears whenever the current turn has a task. Height
    // auto-sizes to the plan (task line + steps + borders), capped so the
    // output pane always keeps room; long plans window around the current step.
    let plan_height: u16 = if app.plan_task.is_some() {
        let steps = app.plan_steps.len().max(1) as u16; // ≥1 for the "preparing" line
        let want = steps + 1 /*task*/ + 2 /*borders*/;
        let cap = (area.height / 2).max(4); // never eat more than half the screen
        let cap = if area.height < 30 {
            cap.min(4)
        } else {
            cap.min(16)
        };
        want.clamp(4, cap)
    } else {
        0
    };

    let mut constraints = vec![Constraint::Length(3)]; // top bar
    if plan_height > 0 {
        constraints.push(Constraint::Length(plan_height)); // plan panel
    }
    constraints.push(Constraint::Min(5)); // output
    constraints.push(Constraint::Length(3)); // input
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(area);

    let mut i = 0;
    draw_topbar(frame, app, chunks[i]);
    i += 1;
    if plan_height > 0 {
        draw_plan_panel(frame, app, chunks[i]);
        i += 1;
    }
    draw_output(frame, app, chunks[i]);
    i += 1;
    draw_input(frame, app, chunks[i]);

    if app.pending_permission.is_some() {
        draw_permission_modal(frame, app, area);
    }
}

/// Live plan panel: a projection of `.buzztr/plan.md`. Done steps are green,
/// the current (first unchecked) step is yellow, upcoming steps are blue.
fn draw_plan_panel(frame: &mut Frame, app: &App, area: Rect) {
    let (done, total) = app.plan_progress();
    let current = app.current_plan_step();

    let header_right = if total > 0 {
        format!(" round {} · {}/{} ", app.round, done, total)
    } else {
        format!(" round {} ", app.round)
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(BORDER))
        .style(Style::default().bg(BG_PANEL))
        .title(Span::styled(
            " Plan ",
            Style::default()
                .fg(ACCENT_CYAN)
                .add_modifier(Modifier::BOLD),
        ))
        .title(
            Line::from(Span::styled(
                header_right,
                Style::default().fg(TEXT_SECONDARY),
            ))
            .right_aligned(),
        );

    let inner_h = area.height.saturating_sub(2) as usize; // minus borders
    let inner_w = area.width.saturating_sub(2) as usize;
    let mut lines: Vec<Line> = Vec::new();

    // Task line.
    if let Some(task) = &app.plan_task {
        lines.push(Line::from(vec![
            Span::styled("Task: ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                ellipsize(task, inner_w.saturating_sub(6)), // "Task: " = 6 cols
                Style::default().fg(TEXT_SOFT),
            ),
        ]));
    }

    if app.plan_steps.is_empty() {
        lines.push(Line::from(Span::styled(
            "(preparing the plan…)",
            Style::default()
                .fg(TEXT_MUTED)
                .add_modifier(Modifier::ITALIC),
        )));
    } else {
        // Window the steps around the current one so long plans still fit.
        let budget = inner_h.saturating_sub(1).max(1); // minus the task line
        let total_steps = app.plan_steps.len();
        let start = if total_steps <= budget {
            0
        } else {
            let cur = current.unwrap_or(total_steps - 1); // all done -> show the tail
            cur.saturating_sub(budget / 2).min(total_steps - budget)
        };
        let end = (start + budget).min(total_steps);
        for (idx, step) in app.plan_steps.iter().enumerate().take(end).skip(start) {
            let is_current = current == Some(idx);
            let (marker, color) = if step.checked {
                ("✓", SUCCESS) // done → green
            } else if is_current {
                ("▸", WARNING) // in progress → yellow
            } else {
                (" ", USER_LINE) // todo → blue
            };
            let num = format!("{:>2}", idx + 1);
            let suffix = if step.checked {
                step.checked_round
                    .map(|r| format!("  (r{r})"))
                    .unwrap_or_default()
            } else if is_current {
                "  ← current".to_string()
            } else {
                String::new()
            };
            // prefix " ✓  1 " = 6 cols; the suffix must always stay intact.
            let text_budget = inner_w.saturating_sub(6 + suffix.chars().count());
            lines.push(Line::from(vec![
                Span::styled(format!(" {marker} {num} "), Style::default().fg(color)),
                Span::styled(
                    ellipsize(&step.text, text_budget),
                    Style::default().fg(color),
                ),
                Span::styled(suffix, Style::default().fg(TEXT_MUTED)),
            ]));
        }
    }

    let widget = Paragraph::new(lines).block(block);
    frame.render_widget(widget, area);
}

fn draw_topbar(frame: &mut Frame, app: &App, area: Rect) {
    let status = if let Some(active_job) = &app.active_job {
        format!("working: {active_job}")
    } else if app.is_thinking {
        "working".to_string()
    } else {
        "ready".to_string()
    };

    let line = Line::from(vec![
        Span::styled(
            "buzztr",
            Style::default()
                .fg(TEXT_PRIMARY)
                .bg(BG_PANEL)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" ".repeat(area.width.saturating_sub(44) as usize)),
        Span::styled(
            format!(" {status} "),
            Style::default()
                .fg(ACCENT_PURPLE)
                .bg(BG_PANEL)
                .add_modifier(Modifier::BOLD),
        ),
    ]);

    let widget = Paragraph::new(line).block(
        Block::default()
            .borders(Borders::LEFT | Borders::RIGHT | Borders::TOP | Borders::BOTTOM)
            .border_style(Style::default().fg(BORDER))
            .style(Style::default().bg(BG_PANEL)),
    );
    frame.render_widget(widget, area);
}

fn char_cols(c: char) -> usize {
    Span::raw(c.to_string()).width()
}

fn str_cols(s: &str) -> usize {
    Span::raw(s).width()
}

/// Shorten `s` to at most `max` display columns, ending in a single `…`
/// when anything was cut.
fn ellipsize(s: &str, max: usize) -> String {
    if str_cols(s) <= max {
        return s.to_string();
    }
    if max == 0 {
        return String::new();
    }
    let mut out = String::new();
    let mut w = 0;
    for c in s.chars() {
        let cw = char_cols(c);
        if w + cw > max - 1 {
            break;
        }
        out.push(c);
        w += cw;
    }
    out.push('…');
    out
}

/// Word-wrap `text` to `width` columns; continuation rows get `indent`
/// spaces. Words longer than a row are hard-broken.
fn wrap_hanging(text: &str, width: usize, indent: usize) -> Vec<String> {
    if width == 0 || str_cols(text) <= width {
        return vec![text.to_string()];
    }
    let lead = text.len() - text.trim_start_matches(' ').len();
    let mut rows = Vec::new();
    let mut cur = " ".repeat(lead);
    let mut cur_w = lead;
    let mut base = lead; // columns of non-content prefix on this row
    for word in text.split(' ').filter(|w| !w.is_empty()) {
        let ww = str_cols(word);
        let sep = usize::from(cur_w > base);
        if cur_w + sep + ww <= width {
            if sep == 1 {
                cur.push(' ');
            }
            cur.push_str(word);
            cur_w += sep + ww;
            continue;
        }
        if cur_w > base {
            rows.push(std::mem::replace(&mut cur, " ".repeat(indent)));
            cur_w = indent;
            base = indent;
        }
        if cur_w + ww <= width {
            cur.push_str(word);
            cur_w += ww;
            continue;
        }
        for ch in word.chars() {
            let cw = char_cols(ch);
            if cur_w + cw > width && cur_w > base {
                rows.push(std::mem::replace(&mut cur, " ".repeat(indent)));
                cur_w = indent;
                base = indent;
            }
            cur.push(ch);
            cur_w += cw;
        }
    }
    rows.push(cur);
    rows
}

/// Draw the scrollable output pane.
fn draw_output(frame: &mut Frame, app: &App, area: Rect) {
    let inner_height = area.height.saturating_sub(2) as usize; // minus borders
    let inner_width = area.width.saturating_sub(2);

    // Convert output lines to ratatui Lines with styles
    let mut styled_lines: Vec<Line> = Vec::new();
    for ol in app.output.iter() {
        {
            let style = match ol.style {
                LineStyle::Normal => Style::default().fg(USER_LINE).bg(BG_PANEL),
                LineStyle::Assistant => Style::default().fg(TEXT_PRIMARY).bg(BG_PANEL),
                LineStyle::ToolCall => Style::default().fg(CMD_LINE).bg(BG_PANEL),
                LineStyle::ToolOk => Style::default().fg(SUCCESS).bg(BG_PANEL),
                LineStyle::ToolErr => Style::default().fg(ERROR).bg(BG_PANEL),
                LineStyle::Status => Style::default().fg(TEXT_MUTED).bg(BG_PANEL),
                LineStyle::Error => Style::default()
                    .fg(ERROR)
                    .bg(BG_PANEL)
                    .add_modifier(Modifier::BOLD),
                LineStyle::Separator => Style::default().fg(BORDER).bg(BG_PANEL),
                LineStyle::Thinking => Style::default()
                    .fg(TEXT_MUTED)
                    .bg(BG_PANEL)
                    .add_modifier(Modifier::ITALIC),
            };
            match ol.style {
                LineStyle::Separator => {
                    styled_lines.push(Line::from(Span::styled(
                        "─".repeat(inner_width as usize),
                        style,
                    )));
                }
                LineStyle::ToolCall | LineStyle::ToolOk | LineStyle::ToolErr => {
                    for row in wrap_hanging(&ol.text, inner_width as usize, 4) {
                        styled_lines.push(Line::from(Span::styled(row, style)));
                    }
                }
                _ => styled_lines.push(Line::from(Span::styled(&ol.text, style))),
            }
        }
    }

    // Add thinking indicator if active
    let mut lines = styled_lines;
    if app.is_thinking {
        let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        let frame_idx = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            / 80) as usize
            % frames.len();
        lines.push(Line::from(Span::styled(
            format!("{} thinking...", frames[frame_idx]),
            Style::default()
                .fg(TEXT_MUTED)
                .bg(BG_PANEL)
                .add_modifier(Modifier::ITALIC),
        )));
    }

    // Also show the token buffer if it has content
    if !app.token_buffer.is_empty() {
        lines.push(Line::from(Span::styled(
            &app.token_buffer,
            Style::default().fg(TEXT_PRIMARY).bg(BG_PANEL),
        )));
    }

    let title = if app.scroll_offset > 0 {
        " transcript (scrolled) ".to_string()
    } else if let Some(active_job) = &app.active_job {
        format!(" transcript ({active_job}) ")
    } else if app.is_thinking {
        " transcript (working) ".to_string()
    } else {
        " transcript ".to_string()
    };

    // Build the paragraph first so we can ask it for the true wrapped-row
    // count. Using `lines.len()` here would count *logical* lines — but with
    // `Wrap { trim: false }` a single long logical line spans multiple
    // visual rows, and `Paragraph::scroll()` operates on visual rows. Feeding
    // it a logical-line count pushes content below the viewport whenever any
    // line wraps (long LLM output, long tool-result first-line, user prompt
    // on a narrow pane).
    let paragraph = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(title)
                .title_style(Style::default().fg(TEXT_SOFT).bg(BG_PANEL))
                .border_style(Style::default().fg(BORDER))
                .style(Style::default().bg(BG_PANEL)),
        )
        .wrap(Wrap { trim: false })
        .style(Style::default().bg(BG_PANEL));

    // Ask ratatui how many visual rows the wrapped text actually occupies.
    // `line_count(width)` wraps at the content width (area.width − 2 for the
    // left/right borders) and adds top+bottom border space (+2) to the
    // result, so we subtract it back out to get pure content rows.
    let total = paragraph.line_count(inner_width).saturating_sub(2);

    let scroll = if app.scroll_offset == 0 {
        total.saturating_sub(inner_height)
    } else {
        total
            .saturating_sub(inner_height)
            .saturating_sub(app.scroll_offset as usize)
    };

    let output_widget = paragraph.scroll((scroll as u16, 0));
    frame.render_widget(output_widget, area);

    // Scrollbar
    if total > inner_height {
        // ratatui: content_length = number of scroll positions.
        let mut scrollbar_state = ScrollbarState::new(total - inner_height + 1)
            .position(scroll)
            .viewport_content_length(inner_height);
        frame.render_stateful_widget(
            Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .begin_symbol(Some("↑"))
                .end_symbol(Some("↓")),
            area,
            &mut scrollbar_state,
        );
    }
}

/// Draw the input line.
fn draw_input(frame: &mut Frame, app: &App, area: Rect) {
    let style = if app.is_thinking {
        Style::default().fg(TEXT_MUTED)
    } else {
        Style::default().fg(BORDER)
    };

    if app.is_thinking {
        // Greyed out, show working indicator
        let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        let frame_idx = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            / 80) as usize
            % frames.len();
        let input_text = format!("{} working...", frames[frame_idx]);
        let input_widget = Paragraph::new(input_text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(style)
                    .title(" working — Ctrl+C to interrupt "),
            )
            .style(Style::default().fg(TEXT_MUTED).bg(BG_INPUT));
        frame.render_widget(input_widget, area);
        // Hide cursor while working
    } else {
        let input_width = area.width.saturating_sub(2) as usize;
        let prefix = "you> ";
        let visible_width = input_width.saturating_sub(prefix.chars().count());
        let (visible_input, cursor_col) =
            visible_input_window(&app.input, app.cursor, visible_width);
        let input_text = format!("{prefix}{visible_input}");
        let input_widget = Paragraph::new(input_text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(style)
                    .title(
                        " ask the agent to inspect files, propose a patch, run tests, or explain a diff... ",
                    )
                    .title_style(Style::default().fg(TEXT_SOFT).bg(BG_INPUT)),
            )
            .style(Style::default().fg(TEXT_SECONDARY).bg(BG_INPUT));
        frame.render_widget(input_widget, area);

        // Show cursor only when input is active
        let cursor_x = area.x + 1 + prefix.chars().count() as u16 + cursor_col as u16;
        let cursor_y = area.y + 1;
        if cursor_x < area.x + area.width.saturating_sub(1) {
            frame.set_cursor_position((cursor_x, cursor_y));
        }
    }
}

/// Draw the permission prompt as a centered modal.
fn draw_permission_modal(frame: &mut Frame, app: &App, area: Rect) {
    let modal = centered_rect(area, 80, 7);
    frame.render_widget(Clear, modal);
    frame.render_widget(
        Block::default()
            .borders(Borders::ALL)
            .title(" Permission Required ")
            .border_style(Style::default().fg(WARNING).add_modifier(Modifier::BOLD))
            .style(Style::default().bg(BG_PANEL)),
        modal,
    );

    let inner = Rect {
        x: modal.x + 1,
        y: modal.y + 1,
        width: modal.width.saturating_sub(2),
        height: modal.height.saturating_sub(2),
    };
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(2), Constraint::Length(3)])
        .split(inner);

    let text = app.pending_permission.as_deref().unwrap_or("");
    let widget = Paragraph::new(text.to_string())
        .style(Style::default().fg(TEXT_PRIMARY).bg(BG_PANEL))
        .wrap(Wrap { trim: false });

    frame.render_widget(widget, chunks[0]);

    let input_width = chunks[1].width.saturating_sub(2) as usize;
    let prefix = "  ";
    let visible_width = input_width.saturating_sub(prefix.chars().count());
    let (visible_input, cursor_col) = visible_input_window(&app.input, app.cursor, visible_width);
    let input_text = format!("{prefix}{visible_input}");
    let widget = Paragraph::new(input_text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(BORDER))
                .title(" [y]es / [n]o / [a]lways ")
                .title_style(Style::default().fg(TEXT_SOFT).bg(BG_INPUT)),
        )
        .style(Style::default().fg(TEXT_PRIMARY).bg(BG_INPUT));

    frame.render_widget(widget, chunks[1]);

    // Show cursor
    let cursor_x = chunks[1].x + 1 + prefix.chars().count() as u16 + cursor_col as u16;
    let cursor_y = chunks[1].y + 1;
    if cursor_x < chunks[1].x + chunks[1].width.saturating_sub(1) {
        frame.set_cursor_position((cursor_x, cursor_y));
    }
}

fn centered_rect(area: Rect, width_percent: u16, height: u16) -> Rect {
    let requested_width = area.width.saturating_mul(width_percent).saturating_div(100);
    let width = requested_width.clamp(20, area.width.saturating_sub(2).max(1));
    let height = height.min(area.height.saturating_sub(2).max(1));
    Rect {
        x: area.x + area.width.saturating_sub(width) / 2,
        y: area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    }
}

fn visible_input_window(input: &str, cursor_byte: usize, max_chars: usize) -> (String, usize) {
    if max_chars == 0 {
        return (String::new(), 0);
    }

    let chars: Vec<char> = input.chars().collect();
    let total_chars = chars.len();
    let cursor_char = input[..cursor_byte.min(input.len())].chars().count();

    if total_chars <= max_chars {
        return (input.to_string(), cursor_char);
    }

    let mut start = cursor_char.saturating_add(1).saturating_sub(max_chars);
    if start + max_chars > total_chars {
        start = total_chars.saturating_sub(max_chars);
    }
    let end = (start + max_chars).min(total_chars);

    let visible: String = chars[start..end].iter().collect();
    let cursor_col = cursor_char
        .saturating_sub(start)
        .min(max_chars.saturating_sub(1));
    (visible, cursor_col)
}

#[cfg(test)]
mod tests {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::layout::Rect;

    use super::{centered_rect, draw, ellipsize, str_cols, visible_input_window, wrap_hanging};
    use crate::tui::app::{App, LineStyle};

    #[test]
    fn visible_input_window_keeps_short_input_unchanged() {
        let (visible, cursor) = visible_input_window("hello", 5, 10);
        assert_eq!(visible, "hello");
        assert_eq!(cursor, 5);
    }

    #[test]
    fn visible_input_window_scrolls_to_keep_cursor_visible() {
        let input = "abcdefghijklmnopqrstuvwxyz";
        let cursor_byte = input.len();
        let (visible, cursor) = visible_input_window(input, cursor_byte, 8);
        assert_eq!(visible, "stuvwxyz");
        assert_eq!(cursor, 7);
    }

    #[test]
    fn visible_input_window_handles_mid_string_cursor() {
        let input = "abcdefghijklmnopqrstuvwxyz";
        let cursor_byte = input.char_indices().nth(10).map(|(i, _)| i).unwrap();
        let (visible, cursor) = visible_input_window(input, cursor_byte, 8);
        assert_eq!(visible, "defghijk");
        assert_eq!(cursor, 7);
    }

    #[test]
    fn centered_rect_is_centered_and_bounded() {
        let modal = centered_rect(Rect::new(0, 0, 100, 30), 80, 7);
        assert_eq!(modal, Rect::new(10, 11, 80, 7));
    }

    #[test]
    fn wrapped_lines_do_not_push_tail_below_viewport() {
        // Narrow terminal: 30 cols total → ~28 cols content width.
        // Output pane gets ~14 rows in this layout (3 topbar, 3 input).
        // Push a mix of short lines and a long one that wraps. The last
        // logical line ("TAIL_MARKER") must stay visible at the bottom.
        let backend = TestBackend::new(30, 20);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = crate::tui::app::App::new();

        // Fill the output with enough short lines that the tail would sit at
        // the boundary, then add a long wrapped line, then the marker.
        for i in 0..10 {
            app.push_output(&format!("line {i}"), crate::tui::app::LineStyle::Normal);
        }
        // This line is ~90 chars → wraps to 3-4 visual rows at width ~28.
        // Under the old scroll math the marker gets pushed off-screen.
        app.push_output(
            "this is a very long logical line that definitely wraps across several visual rows when rendered in a narrow pane",
            crate::tui::app::LineStyle::Normal,
        );
        app.push_output("TAIL_MARKER", crate::tui::app::LineStyle::Normal);

        terminal.draw(|frame| draw(frame, &app)).unwrap();

        let buffer = terminal.backend().buffer().clone();
        let text = buffer_text(&buffer);
        assert!(
            text.contains("TAIL_MARKER"),
            "tail marker hidden below viewport — wrapped-line scroll regression:\n{text}"
        );
    }

    #[test]
    fn wide_terminal_no_wrap_tail_stays_visible() {
        // Reproduce user's 148-col session: no lines wrap, but many logical
        // lines accumulate and the tail should still be visible with
        // scroll_offset == 0. If this test fails, the bug isn't in wrap math.
        let backend = TestBackend::new(148, 40);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = crate::tui::app::App::new();

        // Simulate ~50 output lines — none wrap at 146-col content width.
        for i in 0..30 {
            app.push_output(
                &format!("  → file(shell step {i})"),
                crate::tui::app::LineStyle::ToolCall,
            );
            app.push_output(
                "  ✓ file: [shell: exit 0]",
                crate::tui::app::LineStyle::ToolOk,
            );
        }
        // Tail marker — if the math is right, this is the last thing on screen.
        app.push_output("FINAL_TAIL_MARKER", crate::tui::app::LineStyle::Normal);

        assert_eq!(app.scroll_offset, 0, "push_output should reset scroll");

        terminal.draw(|frame| draw(frame, &app)).unwrap();

        let buffer = terminal.backend().buffer().clone();
        let text = buffer_text(&buffer);
        assert!(
            text.contains("FINAL_TAIL_MARKER"),
            "tail marker hidden at 148-col/no-wrap — math is wrong even without wrapping:\n{text}"
        );
    }

    #[test]
    fn permission_modal_disappears_after_pending_permission_clears() {
        let backend = TestBackend::new(80, 20);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new();
        app.pending_permission = Some("Allow shell command?\n  $ cargo check".into());

        terminal.draw(|frame| draw(frame, &app)).unwrap();

        let first = terminal.backend().buffer().clone();
        let first_text = buffer_text(&first);
        assert!(first_text.contains("Permission Required"));
        assert!(first_text.contains("Allow shell command?"));

        app.pending_permission = None;
        app.input.clear();
        app.cursor = 0;

        terminal.draw(|frame| draw(frame, &app)).unwrap();

        let second = terminal.backend().buffer().clone();
        let second_text = buffer_text(&second);
        assert!(!second_text.contains("Permission Required"));
        assert!(!second_text.contains("Allow shell command?"));
    }

    fn row_text(buffer: &ratatui::buffer::Buffer, y: u16) -> String {
        (0..buffer.area.width)
            .map(|x| buffer[(x, y)].symbol().to_string())
            .collect()
    }

    fn render(app: &App, w: u16, h: u16) -> ratatui::buffer::Buffer {
        let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
        terminal.draw(|f| draw(f, app)).unwrap();
        terminal.backend().buffer().clone()
    }

    #[test]
    fn ellipsize_fits_and_cuts() {
        assert_eq!(ellipsize("abc", 3), "abc");
        assert_eq!(ellipsize("abcdef", 4), "abc…");
        assert_eq!(ellipsize("abcdef", 0), "");
        assert_eq!(ellipsize("abcdef", 1), "…");
    }

    #[test]
    fn plan_placeholder_says_preparing() {
        let mut app = App::new();
        app.plan_task = Some("t".into());
        let text = buffer_text(&render(&app, 80, 40));
        assert!(text.contains("(preparing the plan…)"));
        assert!(!text.contains("exploring"));
    }

    #[test]
    fn separator_spans_full_inner_width() {
        for w in [100u16, 40] {
            let mut app = App::new();
            app.push_output("x".repeat(48).as_str(), LineStyle::Separator);
            let buf = render(&app, w, 20);
            let rows: Vec<String> = (0..20).map(|y| row_text(&buf, y)).collect();
            let sep: Vec<&String> = rows
                .iter()
                .filter(|r| r.contains('─') && r.starts_with('│'))
                .collect();
            assert_eq!(sep.len(), 1, "w={w}: {rows:#?}");
            let inner: String = sep[0].chars().skip(1).take(w as usize - 2).collect();
            assert_eq!(inner, "─".repeat(w as usize - 2));
        }
    }

    #[test]
    fn scrollbar_thumb_touches_bottom_at_tail_and_top_when_scrolled() {
        let mut app = App::new();
        for i in 0..200 {
            app.push_output(&format!("line {i}"), LineStyle::Normal);
        }
        let find = |buf: &ratatui::buffer::Buffer, sym: &str| {
            (0..buf.area.height)
                .find(|&y| buf[(59, y)].symbol() == sym)
                .unwrap()
        };
        let buf = render(&app, 60, 24);
        let (up, down) = (find(&buf, "↑"), find(&buf, "↓"));
        assert_eq!(buf[(59, down - 1)].symbol(), "█", "tail: thumb must hug ↓");
        assert_ne!(buf[(59, up + 1)].symbol(), "█");

        app.scroll_offset = 10_000;
        let buf = render(&app, 60, 24);
        assert_eq!(buf[(59, up + 1)].symbol(), "█", "top: thumb must hug ↑");
        assert_ne!(buf[(59, down - 1)].symbol(), "█");
    }

    #[test]
    fn plan_panel_narrow_keeps_suffix_and_ellipsizes_task() {
        let mut app = App::new();
        app.plan_task = Some("t".repeat(120));
        app.plan_steps = vec![crate::tui::app::PlanStepView {
            checked: true,
            checked_round: Some(4),
            text: "s".repeat(120),
        }];
        let buf = render(&app, 60, 20);
        let rows: Vec<String> = (0..20).map(|y| row_text(&buf, y)).collect();
        let task = rows.iter().find(|r| r.contains("Task:")).unwrap();
        assert!(task.trim_end().ends_with("…│"), "{task}");
        let step = rows.iter().find(|r| r.contains("(r4)")).unwrap();
        assert!(step.contains("…  (r4)"), "{step}");
        // panel is 4 rows: top border, task, step, bottom border
        let top = rows.iter().position(|r| r.contains("Plan")).unwrap();
        assert!(rows[top + 3].starts_with('└'), "{rows:#?}");
    }

    #[test]
    fn plan_panel_all_done_shows_last_step() {
        let mut app = App::new();
        app.plan_task = Some("task".into());
        app.plan_steps = (1..=3)
            .map(|i| crate::tui::app::PlanStepView {
                checked: true,
                checked_round: Some(i),
                text: format!("step-{i}"),
            })
            .collect();
        let text = buffer_text(&render(&app, 60, 20));
        assert!(text.contains("step-3"));
        assert!(!text.contains("step-1"));
    }

    #[test]
    fn wrapped_tool_line_has_hanging_indent() {
        let mut app = App::new();
        app.push_output(
            "  ✓ insert_at: insert_at src/lib.rs after L50: rev_3 applied (+12 -0) and some more words to force wrapping",
            LineStyle::ToolOk,
        );
        let buf = render(&app, 60, 20);
        let rows: Vec<String> = (0..20).map(|y| row_text(&buf, y)).collect();
        let first = rows.iter().position(|r| r.contains("✓ insert_at")).unwrap();
        let cont: String = rows[first + 1].chars().skip(1).take(4).collect();
        assert_eq!(cont, "    ", "{rows:#?}");
        assert!(rows[first + 1].chars().nth(5).is_some_and(|c| c != ' '));
    }

    #[test]
    fn wrap_hanging_hard_breaks_long_words() {
        let rows = wrap_hanging(&format!("  → {}", "x".repeat(30)), 10, 4);
        assert!(rows.iter().all(|r| str_cols(r) <= 10), "{rows:?}");
        assert!(rows[1..].iter().all(|r| r.starts_with("    ")));
    }

    fn buffer_text(buffer: &ratatui::buffer::Buffer) -> String {
        let mut out = String::new();
        for y in 0..buffer.area.height {
            for x in 0..buffer.area.width {
                out.push_str(buffer[(x, y)].symbol());
            }
            out.push('\n');
        }
        out
    }
}

/// Detail viewer: full-screen view of tool result content.
fn draw_detail(frame: &mut Frame, app: &App) {
    let area = frame.area();
    frame.render_widget(Block::default().style(Style::default().bg(BG_MAIN)), area);

    let lines: Vec<Line> = app
        .detail_content
        .lines()
        .map(|l| Line::from(l.to_string()))
        .collect();

    let title = format!(" {} (Ctrl+O or Esc to close) ", app.detail_title);

    let detail_widget = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(title)
                .border_style(Style::default().fg(BORDER))
                .style(Style::default().bg(BG_PANEL)),
        )
        .wrap(Wrap { trim: false })
        .style(Style::default().fg(TEXT_PRIMARY).bg(BG_PANEL));

    frame.render_widget(detail_widget, area);
}
