//! Async event handling for the TUI.
//!
//! Multiplexes keyboard input, LLM streaming events, and tool results
//! into a single event stream that the main loop processes.

use std::time::Duration;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use tokio::sync::mpsc;

/// Events that the TUI main loop handles.
#[derive(Debug)]
pub enum AppEvent {
    /// User pressed a key
    Key(KeyEvent),
    /// Terminal tick (for spinner animation)
    Tick,
    /// LLM produced a token
    Token(String),
    /// LLM finished responding
    LlmDone,
    /// LLM error
    LlmError(String),
    /// Tool call started
    ToolCall(String, String), // (name, args_summary)
    /// Tool call result
    ToolResult(String, bool, String, String), // (name, success, summary, full_content)
    /// Status message
    Status(String),
    /// Agent loop finished for this user message
    AgentDone,
}

/// Spawn a keyboard event reader that sends events to the channel.
/// Runs in a dedicated thread (crossterm events are blocking).
/// `cancel_flag` is set directly on Ctrl+C so it works even when
/// the main event loop is blocked by the agent loop.
pub fn spawn_key_reader(
    tx: mpsc::UnboundedSender<AppEvent>,
    cancel_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    std::thread::spawn(move || {
        loop {
            if event::poll(Duration::from_millis(80)).unwrap_or(false) {
                if let Ok(Event::Key(key)) = event::read() {
                    // Ctrl+C: set cancel flag immediately (bypasses event queue)
                    if is_ctrl_c(&key) {
                        cancel_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                    if tx.send(AppEvent::Key(key)).is_err() {
                        break;
                    }
                }
            } else {
                if tx.send(AppEvent::Tick).is_err() {
                    break;
                }
            }
        }
    });
}

/// Check if a key event is Ctrl+C.
pub fn is_ctrl_c(key: &KeyEvent) -> bool {
    key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL)
}

/// Check if a key event is Ctrl+O (detail viewer toggle).
pub fn is_ctrl_o(key: &KeyEvent) -> bool {
    key.code == KeyCode::Char('o') && key.modifiers.contains(KeyModifiers::CONTROL)
}

/// Check if a key event is Ctrl+D.
pub fn is_ctrl_d(key: &KeyEvent) -> bool {
    key.code == KeyCode::Char('d') && key.modifiers.contains(KeyModifiers::CONTROL)
}
