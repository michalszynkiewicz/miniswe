//! TUI rendering with ratatui.
//!
//! Draws the split-pane layout: scrollable output + input line.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap, Scrollbar, ScrollbarOrientation, ScrollbarState};

use super::app::{App, AppMode, LineStyle};

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

    if app.pending_permission.is_some() {
        // Three-way split: output, permission prompt, input
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(5),    // output
                Constraint::Length(4), // permission prompt
                Constraint::Length(3), // input (y/n/a)
            ])
            .split(area);

        draw_output(frame, app, chunks[0]);
        draw_permission(frame, app, chunks[1]);
        draw_permission_input(frame, app, chunks[2]);
    } else {
        // Normal two-way split: output + input
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(5),    // output
                Constraint::Length(3), // input
            ])
            .split(area);

        draw_output(frame, app, chunks[0]);
        draw_input(frame, app, chunks[1]);
    }
}

/// Draw the scrollable output pane.
fn draw_output(frame: &mut Frame, app: &App, area: Rect) {
    let inner_height = area.height.saturating_sub(2) as usize; // minus borders

    // Convert output lines to ratatui Lines with styles
    let styled_lines: Vec<Line> = app.output.iter().map(|ol| {
        let style = match ol.style {
            LineStyle::Normal => Style::default(),
            LineStyle::Assistant => Style::default().fg(Color::White),
            LineStyle::ToolCall => Style::default().fg(Color::Yellow),
            LineStyle::ToolOk => Style::default().fg(Color::Green),
            LineStyle::ToolErr => Style::default().fg(Color::Red),
            LineStyle::Status => Style::default().fg(Color::DarkGray),
            LineStyle::Error => Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            LineStyle::Separator => Style::default().fg(Color::DarkGray),
            LineStyle::Thinking => Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC),
        };
        Line::from(Span::styled(&ol.text, style))
    }).collect();

    // Add thinking indicator if active
    let mut lines = styled_lines;
    if app.is_thinking {
        let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        let frame_idx = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() / 80) as usize % frames.len();
        lines.push(Line::from(Span::styled(
            format!("{} thinking...", frames[frame_idx]),
            Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC),
        )));
    }

    // Also show the token buffer if it has content
    if !app.token_buffer.is_empty() {
        lines.push(Line::from(Span::styled(
            &app.token_buffer,
            Style::default().fg(Color::White),
        )));
    }

    // Calculate scroll: we want to show the bottom of the output by default
    let total = lines.len();
    let scroll = if app.scroll_offset == 0 {
        total.saturating_sub(inner_height)
    } else {
        total.saturating_sub(inner_height).saturating_sub(app.scroll_offset as usize)
    };

    let title = if app.is_thinking {
        " miniswe (working...) "
    } else {
        " miniswe "
    };

    let output_widget = Paragraph::new(lines)
        .block(Block::default()
            .borders(Borders::ALL)
            .title(title)
            .border_style(Style::default().fg(Color::Cyan)))
        .wrap(Wrap { trim: false })
        .scroll((scroll as u16, 0));

    frame.render_widget(output_widget, area);

    // Scrollbar
    if total > inner_height {
        let mut scrollbar_state = ScrollbarState::new(total)
            .position(scroll);
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
        Style::default().fg(Color::DarkGray)
    } else {
        Style::default().fg(Color::Magenta)
    };

    if app.is_thinking {
        // Greyed out, show working indicator
        let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        let frame_idx = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() / 80) as usize % frames.len();
        let input_text = format!("{} working...", frames[frame_idx]);
        let input_widget = Paragraph::new(input_text)
            .block(Block::default()
                .borders(Borders::ALL)
                .border_style(style)
                .title(" working — Ctrl+C to interrupt "))
            .style(Style::default().fg(Color::DarkGray));
        frame.render_widget(input_widget, area);
        // Hide cursor while working
    } else {
        let input_text = format!("you> {}", app.input);
        let input_widget = Paragraph::new(input_text)
            .block(Block::default()
                .borders(Borders::ALL)
                .border_style(style)
                .title(" Ctrl+O: details | ↑↓: history | PgUp/Dn: scroll "))
            .style(Style::default().fg(Color::White));
        frame.render_widget(input_widget, area);

        // Show cursor only when input is active
        let cursor_x = area.x + 1 + "you> ".len() as u16 + app.cursor as u16;
        let cursor_y = area.y + 1;
        if cursor_x < area.x + area.width - 1 {
            frame.set_cursor_position((cursor_x, cursor_y));
        }
    }
}

/// Draw the permission prompt bar.
fn draw_permission(frame: &mut Frame, app: &App, area: Rect) {
    let text = app.pending_permission.as_deref().unwrap_or("");
    // Show the prompt text (e.g. "Allow shell command?\n  $ rm src/tests.rs")
    let lines: Vec<Line> = text.lines()
        .take(3)
        .map(|l| Line::from(Span::styled(l, Style::default().fg(Color::Yellow))))
        .collect();

    let widget = Paragraph::new(lines)
        .block(Block::default()
            .borders(Borders::ALL)
            .title(" Permission Required ")
            .border_style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)));

    frame.render_widget(widget, area);
}

/// Draw the permission response input.
fn draw_permission_input(frame: &mut Frame, app: &App, area: Rect) {
    let input_text = format!("  {}", app.input);
    let widget = Paragraph::new(input_text)
        .block(Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Yellow))
            .title(" [y]es / [n]o / [a]lways "))
        .style(Style::default().fg(Color::White));

    frame.render_widget(widget, area);

    // Show cursor
    let cursor_x = area.x + 1 + "  ".len() as u16 + app.cursor as u16;
    let cursor_y = area.y + 1;
    if cursor_x < area.x + area.width - 1 {
        frame.set_cursor_position((cursor_x, cursor_y));
    }
}

/// Detail viewer: full-screen view of tool result content.
fn draw_detail(frame: &mut Frame, app: &App) {
    let area = frame.area();

    let lines: Vec<Line> = app.detail_content
        .lines()
        .map(|l| Line::from(l.to_string()))
        .collect();

    let title = format!(" {} (Ctrl+O or Esc to close) ", app.detail_title);

    let detail_widget = Paragraph::new(lines)
        .block(Block::default()
            .borders(Borders::ALL)
            .title(title)
            .border_style(Style::default().fg(Color::Yellow)))
        .wrap(Wrap { trim: false })
        .style(Style::default().fg(Color::White));

    frame.render_widget(detail_widget, area);
}
