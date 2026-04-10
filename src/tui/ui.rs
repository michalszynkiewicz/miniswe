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

/// Normal mode: output pane + input line + optional permission prompt.
fn draw_normal(frame: &mut Frame, app: &App) {
    let area = frame.area();
    frame.render_widget(Block::default().style(Style::default().bg(BG_MAIN)), area);
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // top bar
            Constraint::Min(5),    // output
            Constraint::Length(3), // input
        ])
        .split(area);

    draw_topbar(frame, app, chunks[0]);
    draw_output(frame, app, chunks[1]);
    draw_input(frame, app, chunks[2]);

    if app.pending_permission.is_some() {
        draw_permission_modal(frame, app, area);
    }
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
            "miniswe",
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

/// Draw the scrollable output pane.
fn draw_output(frame: &mut Frame, app: &App, area: Rect) {
    let inner_height = area.height.saturating_sub(2) as usize; // minus borders

    // Convert output lines to ratatui Lines with styles
    let styled_lines: Vec<Line> = app
        .output
        .iter()
        .map(|ol| {
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
            Line::from(Span::styled(&ol.text, style))
        })
        .collect();

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

    // Calculate scroll: we want to show the bottom of the output by default
    let total = lines.len();
    let scroll = if app.scroll_offset == 0 {
        total.saturating_sub(inner_height)
    } else {
        total
            .saturating_sub(inner_height)
            .saturating_sub(app.scroll_offset as usize)
    };

    let title = if app.scroll_offset > 0 {
        " transcript (scrolled) ".to_string()
    } else if let Some(active_job) = &app.active_job {
        format!(" transcript ({active_job}) ")
    } else if app.is_thinking {
        " transcript (working) ".to_string()
    } else {
        " transcript ".to_string()
    };

    let output_widget = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(title)
                .title_style(Style::default().fg(TEXT_SOFT).bg(BG_PANEL))
                .border_style(Style::default().fg(BORDER))
                .style(Style::default().bg(BG_PANEL)),
        )
        .wrap(Wrap { trim: false })
        .style(Style::default().bg(BG_PANEL))
        .scroll((scroll as u16, 0));

    frame.render_widget(output_widget, area);

    // Scrollbar
    if total > inner_height {
        let mut scrollbar_state = ScrollbarState::new(total).position(scroll);
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
            .border_style(
                Style::default()
                    .fg(WARNING)
                    .add_modifier(Modifier::BOLD),
            )
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

    use super::{centered_rect, draw, visible_input_window};
    use crate::tui::app::App;

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
