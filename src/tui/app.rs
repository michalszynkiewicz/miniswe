//! TUI application state.
//!
//! Holds all UI state: output lines, input buffer, scroll position,
//! detail viewer content, and mode flags.

/// A styled output line in the main view.
#[derive(Debug, Clone)]
pub struct OutputLine {
    pub text: String,
    pub style: LineStyle,
}

/// Visual style for an output line.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LineStyle {
    Normal,
    /// LLM-generated text
    Assistant,
    /// Tool call: → tool_name(args)
    ToolCall,
    /// Tool result: ✓/✗ name: summary
    ToolOk,
    ToolErr,
    /// Status/dim text
    Status,
    /// Error message
    Error,
    /// Separator line
    Separator,
    /// Spinner / thinking indicator
    Thinking,
}

/// UI mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AppMode {
    /// Normal: output pane + input line
    Normal,
    /// Detail viewer: full-screen view of a tool result (Ctrl+O)
    Detail,
}

/// The TUI application state.
pub struct App {
    /// Output lines (scrollable)
    pub output: Vec<OutputLine>,
    /// Current input buffer
    pub input: String,
    /// Cursor position within input
    pub cursor: usize,
    /// Scroll offset in output (0 = bottom/latest)
    pub scroll_offset: u16,
    /// Current UI mode
    pub mode: AppMode,
    /// Detail viewer content (shown in Detail mode)
    pub detail_content: String,
    /// Detail viewer title
    pub detail_title: String,
    /// Input history
    pub history: Vec<String>,
    /// Current position in history (for up/down navigation)
    pub history_pos: Option<usize>,
    /// Saved input when navigating history
    pub saved_input: String,
    /// Whether the app should quit
    pub should_quit: bool,
    /// Whether LLM is currently generating
    pub is_thinking: bool,
    /// Current streaming token buffer (accumulated between newlines)
    pub token_buffer: String,
    /// Tool results for Ctrl+O detail viewer (tool_name → full content)
    pub tool_results: Vec<(String, String)>,
    /// Permission prompt waiting for user input (shown above input bar)
    pub pending_permission: Option<String>,
    /// User's response to the permission prompt
    pub permission_response: Option<String>,
}

impl App {
    pub fn new() -> Self {
        Self {
            output: Vec::new(),
            input: String::new(),
            cursor: 0,
            scroll_offset: 0,
            mode: AppMode::Normal,
            detail_content: String::new(),
            detail_title: String::new(),
            history: Vec::new(),
            history_pos: None,
            saved_input: String::new(),
            should_quit: false,
            is_thinking: false,
            token_buffer: String::new(),
            tool_results: Vec::new(),
            pending_permission: None,
            permission_response: None,
        }
    }

    /// Load input history from file.
    pub fn load_history(&mut self, path: &std::path::Path) {
        if let Ok(content) = std::fs::read_to_string(path) {
            self.history = content
                .lines()
                .map(|l| l.to_string())
                .collect();
        }
    }

    /// Save input history to file.
    pub fn save_history(&self, path: &std::path::Path) {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let content = self.history.iter()
            .rev()
            .take(200)
            .rev()
            .cloned()
            .collect::<Vec<_>>()
            .join("\n");
        let _ = std::fs::write(path, content);
    }

    /// Add a line to output.
    pub fn push_output(&mut self, text: &str, style: LineStyle) {
        self.output.push(OutputLine {
            text: text.to_string(),
            style,
        });
        // Auto-scroll to bottom when new content arrives
        self.scroll_offset = 0;
    }

    /// Append streaming token to the current line.
    pub fn push_token(&mut self, token: &str) {
        self.token_buffer.push_str(token);

        // Split on newlines — each completed line becomes an output line
        while let Some(pos) = self.token_buffer.find('\n') {
            let line = self.token_buffer[..pos].to_string();
            self.output.push(OutputLine {
                text: line,
                style: LineStyle::Assistant,
            });
            self.token_buffer = self.token_buffer[pos + 1..].to_string();
            self.scroll_offset = 0;
        }
    }

    /// Flush any remaining token buffer as a final line.
    pub fn flush_tokens(&mut self) {
        if !self.token_buffer.is_empty() {
            let text = std::mem::take(&mut self.token_buffer);
            self.output.push(OutputLine {
                text,
                style: LineStyle::Assistant,
            });
            self.scroll_offset = 0;
        }
    }

    /// Submit the current input, returning it and adding to history.
    pub fn submit_input(&mut self) -> String {
        let input = self.input.clone();
        if !input.is_empty() {
            self.history.push(input.clone());
        }
        self.input.clear();
        self.cursor = 0;
        self.history_pos = None;
        input
    }

    /// Navigate up in history.
    pub fn history_up(&mut self) {
        if self.history.is_empty() {
            return;
        }
        match self.history_pos {
            None => {
                self.saved_input = self.input.clone();
                self.history_pos = Some(self.history.len() - 1);
            }
            Some(pos) if pos > 0 => {
                self.history_pos = Some(pos - 1);
            }
            _ => return,
        }
        if let Some(pos) = self.history_pos {
            self.input = self.history[pos].clone();
            self.cursor = self.input.len();
        }
    }

    /// Navigate down in history.
    pub fn history_down(&mut self) {
        match self.history_pos {
            Some(pos) => {
                if pos + 1 < self.history.len() {
                    self.history_pos = Some(pos + 1);
                    self.input = self.history[pos + 1].clone();
                } else {
                    self.history_pos = None;
                    self.input = self.saved_input.clone();
                }
                self.cursor = self.input.len();
            }
            None => {}
        }
    }

    /// Insert a character at cursor position.
    pub fn insert_char(&mut self, c: char) {
        self.input.insert(self.cursor, c);
        self.cursor += c.len_utf8();
    }

    /// Delete character before cursor (backspace).
    pub fn delete_char(&mut self) {
        if self.cursor > 0 {
            let prev = self.input[..self.cursor]
                .chars()
                .last()
                .map(|c| c.len_utf8())
                .unwrap_or(0);
            self.input.remove(self.cursor - prev);
            self.cursor -= prev;
        }
    }

    /// Move cursor left.
    pub fn cursor_left(&mut self) {
        if self.cursor > 0 {
            let prev = self.input[..self.cursor]
                .chars()
                .last()
                .map(|c| c.len_utf8())
                .unwrap_or(0);
            self.cursor -= prev;
        }
    }

    /// Move cursor right.
    pub fn cursor_right(&mut self) {
        if self.cursor < self.input.len() {
            let next = self.input[self.cursor..]
                .chars()
                .next()
                .map(|c| c.len_utf8())
                .unwrap_or(0);
            self.cursor += next;
        }
    }

    /// Scroll output up.
    pub fn scroll_up(&mut self, amount: u16) {
        let max_scroll = self.output.len().saturating_sub(1) as u16;
        self.scroll_offset = (self.scroll_offset + amount).min(max_scroll);
    }

    /// Scroll output down.
    pub fn scroll_down(&mut self, amount: u16) {
        self.scroll_offset = self.scroll_offset.saturating_sub(amount);
    }

    /// Open detail viewer with content from the last tool result.
    pub fn open_detail(&mut self) {
        if let Some((name, content)) = self.tool_results.last() {
            self.detail_title = name.clone();
            self.detail_content = content.clone();
            self.mode = AppMode::Detail;
        }
    }

    /// Close detail viewer.
    pub fn close_detail(&mut self) {
        self.mode = AppMode::Normal;
    }

    /// Store a tool result for the detail viewer.
    pub fn store_tool_result(&mut self, name: &str, content: &str) {
        self.tool_results.push((name.to_string(), content.to_string()));
        // Keep last 50
        if self.tool_results.len() > 50 {
            self.tool_results.remove(0);
        }
    }
}
