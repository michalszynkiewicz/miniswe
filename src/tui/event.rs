//! Async event handling for the TUI.
//!
//! Multiplexes keyboard input, LLM streaming events, and tool results
//! into a single event stream that the main loop processes.

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers, MouseEvent};
use std::sync::mpsc as std_mpsc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// Events that the TUI main loop handles.
#[derive(Debug)]
pub enum AppEvent {
    /// User pressed a key
    Key(KeyEvent),
    /// Mouse event
    Mouse(MouseEvent),
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
    /// Worker requests a permission decision from the UI thread
    PermissionRequest(String, std_mpsc::Sender<String>), // (prompt, response_tx)
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
        enum EscapeState {
            None,
            SawEsc(KeyEvent, Instant),
            InSequence,
        }

        let mut escape_state = EscapeState::None;

        loop {
            if event::poll(Duration::from_millis(80)).unwrap_or(false) {
                if let Ok(evt) = event::read() {
                    match evt {
                        Event::Key(key) => {
                            match &mut escape_state {
                                EscapeState::None => {
                                    if key.code == KeyCode::Esc {
                                        escape_state = EscapeState::SawEsc(key, Instant::now());
                                        continue;
                                    }
                                }
                                EscapeState::SawEsc(saved_esc, _) => {
                                    if starts_escape_sequence(&key) {
                                        escape_state = EscapeState::InSequence;
                                        continue;
                                    }
                                    if tx.send(AppEvent::Key(*saved_esc)).is_err() {
                                        break;
                                    }
                                    escape_state = EscapeState::None;
                                }
                                EscapeState::InSequence => {
                                    if ends_escape_sequence(&key) {
                                        escape_state = EscapeState::None;
                                    }
                                    continue;
                                }
                            }

                            // Ctrl+C: set cancel flag immediately (bypasses event queue)
                            if is_ctrl_c(&key) {
                                cancel_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                            }
                            if tx.send(AppEvent::Key(key)).is_err() {
                                break;
                            }
                        }
                        Event::Mouse(mouse) => {
                            if tx.send(AppEvent::Mouse(mouse)).is_err() {
                                break;
                            }
                        }
                        _ => {}
                    }
                }
            } else {
                if let EscapeState::SawEsc(saved_esc, started_at) = &escape_state {
                    if started_at.elapsed() >= Duration::from_millis(25) {
                        if tx.send(AppEvent::Key(*saved_esc)).is_err() {
                            break;
                        }
                        escape_state = EscapeState::None;
                        continue;
                    }
                }
                if tx.send(AppEvent::Tick).is_err() {
                    break;
                }
            }
        }
    });
}

fn starts_escape_sequence(key: &KeyEvent) -> bool {
    matches!(
        key.code,
        KeyCode::Char('[') | KeyCode::Char('<') | KeyCode::Char('O')
    )
}

fn ends_escape_sequence(key: &KeyEvent) -> bool {
    match key.code {
        KeyCode::Char(c) => c.is_ascii_alphabetic() || matches!(c, '~' | 'm' | 'M'),
        _ => true,
    }
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
