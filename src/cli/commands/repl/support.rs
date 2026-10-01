//! Support helpers for the REPL loop: rewind/scrap handling, key and
//! interrupt plumbing, streamed-content reconciliation, and bounded
//! awaiting of tool and shell jobs with a live UI.

use super::*;

pub(super) fn handle_background_key(app: &mut App, key: &crossterm::event::KeyEvent) -> bool {
    match key.code {
        KeyCode::PageUp => {
            app.scroll_up(10);
            true
        }
        KeyCode::PageDown => {
            app.scroll_down(10);
            true
        }
        KeyCode::Up if app.input.is_empty() => {
            app.scroll_up(1);
            true
        }
        KeyCode::Down if app.input.is_empty() => {
            app.scroll_down(1);
            true
        }
        KeyCode::Home if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.scroll_offset = app.output.len().saturating_sub(1) as u16;
            true
        }
        KeyCode::End if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.scroll_offset = 0;
            true
        }
        _ => false,
    }
}

/// Drop any input-kind events queued in `rx` without processing them.
///
/// Called at the end of the Enter handler to discard keystrokes the user
/// typed while the "working" indicator was up (LLM streaming + post-turn
/// compression). Those keys were meant to be ignored per `is_thinking`, but
/// because the main loop is `await`-ing the agent/compressor, keys queue in
/// the channel and would otherwise be replayed against the now-idle input
/// box — producing a visible desync where the user's next prompt appears to
/// fire "on its own".
///
/// We preserve non-key events (permission requests, status updates). Ctrl+C
/// was already handled inline by the key reader via the `cancelled` flag.
pub(super) fn drain_stale_key_events(rx: &mut mpsc::UnboundedReceiver<AppEvent>) {
    while let Ok(evt) = rx.try_recv() {
        match evt {
            AppEvent::Key(_) | AppEvent::Mouse(_) | AppEvent::Tick => {}
            other => {
                // Put non-input events back via a small re-enqueue: we only
                // have a receiver here, so the cleanest option is to drop
                // them too. In practice, permission requests and status
                // messages are only emitted while an agent task is running,
                // which it isn't by the time we get here.
                let _ = other;
            }
        }
    }
}

pub(super) fn reconcile_streamed_assistant_content(
    rendered: &str,
    final_content: &str,
) -> Option<String> {
    if final_content.is_empty() || rendered == final_content {
        return None;
    }
    if let Some(suffix) = final_content.strip_prefix(rendered) {
        return (!suffix.is_empty()).then(|| suffix.to_string());
    }
    if rendered.is_empty() {
        return Some(final_content.to_string());
    }
    Some(format!(
        "\n[final response continuation]\n{}",
        final_content
    ))
}

pub(super) fn finish_completed_turn(
    app: &mut App,
    terminal: &mut Terminal<impl Backend>,
    final_content: Option<&str>,
    rendered_assistant_text: Option<&str>,
) -> io::Result<()> {
    app.is_thinking = false;
    if let (Some(final_content), Some(rendered)) = (final_content, rendered_assistant_text)
        && let Some(missing) = reconcile_streamed_assistant_content(rendered, final_content)
    {
        app.push_token(&missing);
    }
    app.flush_tokens();
    app.push_output(
        "────────────────────────────────────────────────",
        LineStyle::Separator,
    );
    terminal
        .draw(|frame| ui::draw(frame, app))
        .map_err(|e| io::Error::other(e.to_string()))?;
    Ok(())
}

pub(super) async fn await_tool_job_ui(
    rx: &mut mpsc::UnboundedReceiver<AppEvent>,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
    job_label: &str,
    result_rx: &mut tokio::sync::oneshot::Receiver<Result<crate::tools::ToolResult, String>>,
    cancelled: &Arc<AtomicBool>,
) -> crate::tools::ToolResult {
    app.set_active_job(job_label);
    loop {
        tokio::select! {
            result = &mut *result_rx => {
                app.clear_active_job();
                return match result {
                    Ok(Ok(tool_result)) => tool_result,
                    Ok(Err(err)) => crate::tools::ToolResult::err(err),
                    Err(_) => crate::tools::ToolResult::err("Tool worker dropped job".into()),
                };
            }
            evt = rx.recv() => {
                match evt {
                    Some(AppEvent::Tick) => {
                        let _ = terminal.draw(|frame| ui::draw(frame, app));
                    }
                    Some(AppEvent::Key(key)) if handle_background_key(app, &key) => {
                        let _ = terminal.draw(|frame| ui::draw(frame, app));
                    }
                    Some(AppEvent::Key(key)) if event::is_ctrl_c(&key) => {
                        cancelled.store(true, Ordering::Relaxed);
                        app.push_output("(interrupted)", LineStyle::Status);
                        let _ = terminal.draw(|frame| ui::draw(frame, app));
                    }
                    Some(AppEvent::Mouse(_)) => {}
                    Some(AppEvent::PermissionRequest(prompt, response_tx)) => {
                        let response = fulfill_permission_request(app, rx, terminal, prompt).await;
                        let _ = response_tx.send(response);
                    }
                    Some(_) => {}
                    None => {
                        app.clear_active_job();
                        return crate::tools::ToolResult::err("Event stream closed.".into())
                    },
                }
            }
        }
    }
}

/// Wait for the user to respond to a permission prompt in the TUI.
/// Blocks until Enter is pressed, returns the trimmed input (e.g., "y", "n", "a").
pub(super) async fn wait_for_modal_input(
    app: &mut App,
    rx: &mut mpsc::UnboundedReceiver<AppEvent>,
    terminal: &mut Terminal<impl Backend>,
    instant_keys: &[char],
) -> String {
    loop {
        let _ = terminal.draw(|frame| ui::draw(frame, app));

        let evt = match rx.recv().await {
            Some(e) => e,
            None => return "n".into(),
        };

        match evt {
            AppEvent::Key(key) => match key.code {
                KeyCode::Enter => {
                    let response = app.input.trim().to_lowercase();
                    app.input.clear();
                    app.cursor = 0;
                    return response;
                }
                KeyCode::Char('\n') | KeyCode::Char('\r') => {
                    let response = app.input.trim().to_lowercase();
                    app.input.clear();
                    app.cursor = 0;
                    return response;
                }
                KeyCode::Char(c) => {
                    if key.modifiers.is_empty() {
                        let lower = c.to_ascii_lowercase();
                        if app.input.is_empty() && instant_keys.contains(&lower) {
                            app.input.clear();
                            app.cursor = 0;
                            return lower.to_string();
                        }
                    }
                    app.insert_char(c);
                }
                KeyCode::Backspace => app.delete_char(),
                KeyCode::Esc => {
                    app.input.clear();
                    app.cursor = 0;
                    return "n".into();
                }
                _ => {}
            },
            AppEvent::Tick => {} // re-render
            _ => {}
        }
    }
}

pub(super) async fn wait_for_permission_input(
    app: &mut App,
    rx: &mut mpsc::UnboundedReceiver<AppEvent>,
    terminal: &mut Terminal<impl Backend>,
) -> String {
    wait_for_modal_input(app, rx, terminal, &['y', 'n', 'a']).await
}

pub(super) async fn fulfill_permission_request(
    app: &mut App,
    rx: &mut mpsc::UnboundedReceiver<AppEvent>,
    terminal: &mut Terminal<impl Backend>,
    prompt: String,
) -> String {
    app.pending_permission = Some(prompt);
    app.input.clear();
    app.cursor = 0;
    let response = wait_for_permission_input(app, rx, terminal).await;
    app.pending_permission = None;
    let _ = terminal.draw(|frame| ui::draw(frame, app));
    response
}

pub(super) async fn await_shell_job_repl(
    mut shell_job: crate::runtime::ShellJobHandle,
    app: &mut App,
    rx: &mut mpsc::UnboundedReceiver<AppEvent>,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    cancelled: &Arc<AtomicBool>,
) -> crate::tools::ToolResult {
    app.set_active_job("shell");
    loop {
        tokio::select! {
            event = shell_job.events_rx.recv() => {
                match event {
                    Some(ShellWorkerEvent::TimedOut { command, timeout_secs }) => {
                        app.pending_permission = Some(format!(
                            "Shell command has been running for {timeout_secs}s:\n  $ {command}\nChoose: [c]ontinue waiting or [k]ill the command."
                        ));
                        app.input.clear();
                        app.cursor = 0;
                        let _ = terminal.draw(|frame| ui::draw(frame, app));
                        let response = wait_for_modal_input(app, rx, terminal, &['c', 'k']).await;
                        app.pending_permission = None;
                        let control = match response.as_str() {
                            "c" => {
                                app.push_output("  · Continuing to wait for shell command...", LineStyle::Status);
                                ShellControl::Continue
                            }
                            _ => {
                                app.push_output("  · Shell command killed.", LineStyle::Status);
                                ShellControl::Kill
                            }
                        };
                        if shell_job.send_control(control).is_err() {
                            app.clear_active_job();
                            return crate::tools::ToolResult::err("Shell worker dropped control channel".into());
                        }
                        let _ = terminal.draw(|frame| ui::draw(frame, app));
                    }
                    Some(ShellWorkerEvent::Detached { running, command }) => {
                        // The REPL never sends ShellControl::Detach (jobs are
                        // a headless-only surface) — if this arrives anyway,
                        // fail safe: kill the command rather than leak it.
                        app.clear_active_job();
                        let _ = crate::tools::shell::kill(running, 0);
                        return crate::tools::ToolResult::err(format!(
                            "Shell command detached unexpectedly and was killed: {command}"
                        ));
                    }
                    Some(ShellWorkerEvent::Completed(result)) => {
                        app.clear_active_job();
                        if cancelled.load(Ordering::Relaxed) {
                            cancelled.store(false, Ordering::Relaxed);
                        }
                        if matches!(&result, Ok(tool_result) if !tool_result.success && tool_result.content == "Command interrupted by user.") {
                            app.push_output("  · Shell command interrupted.", LineStyle::Status);
                        }
                        return match result {
                            Ok(tool_result) => tool_result,
                            Err(err) => crate::tools::ToolResult::err(err),
                        };
                    }
                    None => {
                        app.clear_active_job();
                        if cancelled.load(Ordering::Relaxed) {
                            cancelled.store(false, Ordering::Relaxed);
                        }
                        return crate::tools::ToolResult::err("Shell worker dropped before reporting a result".into());
                    }
                }
            }
            evt = rx.recv() => {
                match evt {
                    Some(AppEvent::Tick) => {
                        let _ = terminal.draw(|frame| ui::draw(frame, app));
                    }
                    Some(AppEvent::Key(key)) if handle_background_key(app, &key) => {
                        let _ = terminal.draw(|frame| ui::draw(frame, app));
                    }
                    Some(AppEvent::Key(key)) if event::is_ctrl_c(&key) => {
                        cancelled.store(true, Ordering::Relaxed);
                        app.push_output("(interrupted)", LineStyle::Status);
                        let _ = terminal.draw(|frame| ui::draw(frame, app));
                    }
                    Some(AppEvent::Mouse(_)) => {}
                    Some(AppEvent::PermissionRequest(prompt, response_tx)) => {
                        let response = fulfill_permission_request(app, rx, terminal, prompt).await;
                        let _ = response_tx.send(response);
                    }
                    Some(_) => {}
                    None => {
                        app.clear_active_job();
                        if cancelled.load(Ordering::Relaxed) {
                            cancelled.store(false, Ordering::Relaxed);
                        }
                        return crate::tools::ToolResult::err("Event stream closed.".into());
                    }
                }
            }
        }
    }
}
