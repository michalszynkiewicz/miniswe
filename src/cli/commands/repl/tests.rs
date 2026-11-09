use super::*;
use crossterm::event::{KeyEvent, KeyEventKind, KeyEventState};
use ratatui::Terminal;
use ratatui::backend::TestBackend;
use serde_json::json;
use tokio::sync::mpsc;

#[test]
fn shell_read_only_allows_reads_blocks_writes() {
    for ok in [
        "ls -R",
        "cat src/main.rs",
        "grep -rn foo .",
        "rg pattern",
        "find . -name '*.rs'",
        "git status",
        "git log --oneline",
        "cd app && ls",
        "wc -l file",
        "grep x 2>/dev/null",
        "git diff | grep foo",
    ] {
        assert!(shell_is_read_only(ok), "should ALLOW read-only: {ok}");
    }
    for bad in [
        "echo hi > foo.rs",
        "rm -rf x",
        "mv a b",
        "cargo build",
        "cargo check",
        "git commit -m x",
        "git add .",
        "sed -i 's/a/b/' f",
        "touch new",
        "mkdir d",
        "find . -delete",
        "find . -exec rm {} \\;",
        "ls && rm x",
        "echo $(rm x)",
        "bash -c 'rm x'",
        "python -c 'import os'",
        "x > y",
    ] {
        assert!(!shell_is_read_only(bad), "should BLOCK mutating: {bad}");
    }
}

#[test]
fn explore_block_reason_blocks_mutations_allows_reads() {
    let empty = json!({});
    // edit/delete/spawn tools → blocked
    for t in [
        "write_file",
        "replace_range",
        "refactor",
        "revert",
        "delete_file",
        "spawn_agents",
    ] {
        assert!(explore_block_reason(t, "", &empty).is_some(), "block {t}");
    }
    // read/intel tools → allowed
    for t in ["read_symbol", "search", "code"] {
        assert!(
            explore_block_reason(t, "repo_map", &empty).is_none(),
            "allow {t}"
        );
    }
    // file: read ok, read-only shell ok, mutating shell blocked
    assert!(explore_block_reason("file", "read", &json!({"path": "x"})).is_none());
    assert!(
        explore_block_reason("file", "shell", &json!({"command": "ls -R"})).is_none(),
        "read-only shell allowed"
    );
    assert!(
        explore_block_reason("file", "shell", &json!({"command": "echo x > f"})).is_some(),
        "mutating shell blocked"
    );
}

#[test]
fn classifier_parse_is_fail_safe_to_coding() {
    // EXPLORE only on a clean leading EXPLORE.
    assert!(is_explore_reply("EXPLORE"));
    assert!(is_explore_reply("  explore \n"));
    assert!(is_explore_reply("EXPLORE."));
    assert!(is_explore_reply("EXPLORE - read only"));
    // Everything else → CODING (fail-safe), incl. prose-wrapped,
    // empty, the other label, garbage.
    assert!(!is_explore_reply("CODING"));
    assert!(!is_explore_reply(""));
    assert!(!is_explore_reply("I think this is EXPLORE"));
    assert!(!is_explore_reply("probably explore?"));
    assert!(!is_explore_reply("</think> blah"));
}

#[test]
fn read_only_filter_drops_writers_and_plan_keeps_readers() {
    use crate::llm::{FunctionDefinition, ToolDefinition};
    let td = |n: &str| ToolDefinition {
        r#type: "function".into(),
        function: FunctionDefinition {
            name: n.into(),
            description: String::new(),
            parameters: json!({}),
        },
    };
    let all = vec![
        td("file"),
        td("code"),
        td("web"),
        td("show_rev"),
        td("check"),
        td("write_file"),
        td("edit_file"),
        td("refactor"),
        td("add_function_param"),
        td("replace_range"),
        td("insert_at"),
        td("revert"),
        td("delete_file"),
        td("plan"),
        td("spawn_agents"),
    ];
    let ro: Vec<String> = read_only_tool_defs(&all)
        .iter()
        .map(|t| t.function.name.clone())
        .collect();
    assert_eq!(ro, ["file", "code", "web", "show_rev", "check"]);
}

#[test]
fn file_search_summary_uses_pattern_and_path() {
    let args = json!({
        "action": "search",
        "path": "src/context/mod.rs",
        "pattern": "pub fn assemble",
    });

    assert_eq!(
        summarize_args("file", &args),
        "search \"pub fn assemble\" in src/context/mod.rs"
    );
}

#[test]
fn plan_refine_summary_includes_step() {
    let args = json!({
        "action": "refine",
        "step": 2,
    });

    assert_eq!(summarize_args("plan", &args), "refine step 2");
}

#[test]
fn web_search_summary_includes_query() {
    let args = json!({
        "action": "search",
        "query": "Michał Szynkiewicz",
    });

    assert_eq!(
        summarize_args("web", &args),
        "search \"Michał Szynkiewicz\""
    );
}

#[test]
fn grouped_file_shell_maps_to_shell_permission_action() {
    let args = json!({
        "action": "shell",
        "command": "python -m http.server",
    });

    match permission_action("file", &args) {
        Some(Action::Shell(cmd)) => assert_eq!(cmd, "python -m http.server"),
        _ => panic!("expected grouped file shell to require shell permission"),
    }
}

#[test]
fn loop_hint_smart_mentions_edit_file() {
    let hint = loop_detected_hint(EditMode::Smart);
    assert!(hint.contains("edit_file"));
}

#[test]
fn loop_hint_fast_mentions_revision_table_tools() {
    let hint = loop_detected_hint(EditMode::Fast);
    assert!(hint.contains("show_rev"));
    assert!(hint.contains("revert"));
    // Fast mode now exposes edit_file, so the loop hint suggests it
    // as a structural-rewrite escape hatch.
    assert!(hint.contains("edit_file"));
}

#[test]
fn reconcile_streamed_assistant_content_appends_missing_suffix() {
    assert_eq!(
        reconcile_streamed_assistant_content("Hello", "Hello world"),
        Some(" world".into())
    );
}

#[test]
fn reconcile_streamed_assistant_content_returns_none_when_complete() {
    assert_eq!(
        reconcile_streamed_assistant_content("Hello world", "Hello world"),
        None
    );
}

#[test]
fn reconcile_streamed_assistant_content_uses_full_content_when_nothing_rendered() {
    assert_eq!(
        reconcile_streamed_assistant_content("", "Hello world"),
        Some("Hello world".into())
    );
}

#[test]
fn finish_completed_turn_draws_final_text_and_separator() {
    let backend = TestBackend::new(80, 20);
    let mut terminal = Terminal::new(backend).unwrap();
    let mut app = App::new();
    app.push_token("Final answer");

    finish_completed_turn(&mut app, &mut terminal, Some("Final answer"), Some("")).unwrap();

    let text = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|cell| cell.symbol())
        .collect::<String>();
    assert!(text.contains("Final answer"));
    assert!(text.contains("────────────────────────────────────────────────"));
    assert!(!app.is_thinking);
}

#[test]
fn finish_completed_turn_appends_missing_suffix_before_separator() {
    let backend = TestBackend::new(100, 20);
    let mut terminal = Terminal::new(backend).unwrap();
    let mut app = App::new();
    app.push_token("Hello");

    finish_completed_turn(&mut app, &mut terminal, Some("Hello world"), Some("Hello")).unwrap();

    let joined = app
        .output
        .iter()
        .map(|line| line.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(joined.contains("Hello world"));
    assert!(joined.ends_with("────────────────────────────────────────────────"));
}

#[tokio::test]
async fn redraw_while_ticks_the_timer_during_a_router_await() {
    let mut app = App::new();
    app.is_thinking = true;
    app.turn_started = Some(std::time::Instant::now() - std::time::Duration::from_secs(65));
    let backend = TestBackend::new(80, 20);
    let mut terminal = Terminal::new(backend).unwrap();

    let value = redraw_while(
        async {
            tokio::time::sleep(std::time::Duration::from_millis(120)).await;
            7
        },
        &mut terminal,
        &app,
    )
    .await;

    assert_eq!(value, 7);
    let buffer = terminal.backend().buffer();
    let top_row: String = (0..buffer.area.width)
        .map(|x| buffer[(x, 0)].symbol())
        .collect();
    assert!(top_row.contains("1:05"), "top row was: {top_row}");
}

#[tokio::test]
async fn permission_prompt_accepts_single_key_without_enter() {
    let mut app = App::new();
    app.pending_permission = Some("Allow shell command?".into());
    let backend = TestBackend::new(80, 20);
    let mut terminal = Terminal::new(backend).unwrap();
    let (tx, mut rx) = mpsc::unbounded_channel();

    tx.send(AppEvent::Key(KeyEvent {
        code: KeyCode::Char('y'),
        modifiers: KeyModifiers::empty(),
        kind: KeyEventKind::Press,
        state: KeyEventState::empty(),
    }))
    .unwrap();

    let response = wait_for_permission_input(&mut app, &mut rx, &mut terminal).await;
    assert_eq!(response, "y");
    assert!(app.input.is_empty());
    assert_eq!(app.cursor, 0);
}

#[tokio::test]
async fn permission_prompt_accepts_raw_carriage_return_as_enter() {
    let mut app = App::new();
    app.pending_permission = Some("Allow shell command?".into());
    app.input = "yes".into();
    app.cursor = app.input.len();
    let backend = TestBackend::new(80, 20);
    let mut terminal = Terminal::new(backend).unwrap();
    let (tx, mut rx) = mpsc::unbounded_channel();

    tx.send(AppEvent::Key(KeyEvent {
        code: KeyCode::Char('\r'),
        modifiers: KeyModifiers::empty(),
        kind: KeyEventKind::Press,
        state: KeyEventState::empty(),
    }))
    .unwrap();

    let response = wait_for_permission_input(&mut app, &mut rx, &mut terminal).await;
    assert_eq!(response, "yes");
    assert!(app.input.is_empty());
    assert_eq!(app.cursor, 0);
}

#[test]
fn help_lists_commands_then_keys() {
    let lines = help_lines();
    let pos = |needle: &str| lines.iter().position(|l| l.contains(needle));
    let quit = pos("quit   — exit").expect("commands present");
    let enter = pos("Enter: send · Ctrl+C: interrupt · Ctrl+D or quit: exit").unwrap();
    let ctrl_o = pos("Ctrl+O: tool-result details · PgUp/PgDn, Ctrl+Home/End: scroll").unwrap();
    let arrows = pos("Up/Down: input history").unwrap();
    assert!(quit < enter && enter < ctrl_o && ctrl_o < arrows);
}

#[test]
fn builtin_command_classification() {
    for c in [
        "/help",
        "/clear",
        "/new",
        "/skills",
        "/skills list",
        "/skills foo help",
    ] {
        assert!(is_builtin_command(c), "{c}");
    }
    for c in ["hello", "/foo", "/skills foo"] {
        assert!(!is_builtin_command(c), "{c}");
    }
    assert!(is_quit_command("quit") && is_quit_command("/quit") && is_quit_command("exit"));
    assert!(!is_quit_command("quitting"));
}

#[test]
fn submitting_quit_leaves_history_unchanged() {
    let mut app = App::new();
    app.history.push("earlier".into());
    app.input = "quit".into();
    let input = app.submit_input();
    if is_quit_command(&input) {
        app.forget_last_history();
    }
    assert_eq!(app.history, vec!["earlier".to_string()]);
}

#[test]
fn endpoint_hint_line_names_endpoint_and_config_key() {
    assert_eq!(
        endpoint_hint_line("http://localhost:8080"),
        "  endpoint: http://localhost:8080 (set [model].endpoint in .buzztr/config.toml)"
    );
}

fn plain_key(code: KeyCode) -> KeyEvent {
    KeyEvent {
        code,
        modifiers: KeyModifiers::empty(),
        kind: KeyEventKind::Press,
        state: KeyEventState::empty(),
    }
}

#[test]
fn up_on_empty_input_recalls_history_and_down_restores_it() {
    let mut app = App::new();
    app.history = vec!["first".into(), "second".into()];
    app.history_up();
    assert_eq!(app.input, "second");
    app.history_up();
    assert_eq!(app.input, "first");
    app.history_down();
    app.history_down();
    assert_eq!(app.input, "");
}

#[test]
fn arrows_do_not_scroll_mid_turn_but_page_keys_do() {
    let mut app = App::new();
    for i in 0..50 {
        app.push_output(&format!("line {i}"), LineStyle::Normal);
    }
    assert!(!handle_background_key(&mut app, &plain_key(KeyCode::Up)));
    assert!(!handle_background_key(&mut app, &plain_key(KeyCode::Down)));
    assert_eq!(app.scroll_offset, 0);
    assert!(handle_background_key(&mut app, &plain_key(KeyCode::PageUp)));
    assert!(app.scroll_offset > 0);
}

#[test]
fn tool_result_line_drops_repeated_name_and_icon() {
    use super::tui_ui::tool_result_line;
    assert_eq!(
        tool_result_line(
            "insert_at",
            true,
            "insert_at src/lib.rs after L50: rev_3 applied (+12 -0)"
        ),
        "  ✓ insert_at: src/lib.rs after L50: rev_3 applied (+12 -0)"
    );
    assert_eq!(
        tool_result_line("plan", true, "✓ Step 1 checked off at round 5"),
        "  ✓ plan: Step 1 checked off at round 5"
    );
    // Name and icon together, failure icon.
    assert_eq!(
        tool_result_line("shell", false, "shell ✗ exit 1"),
        "  ✗ shell: exit 1"
    );
    // The other result's icon, a trailing/inner icon, a longer tool name
    // sharing the prefix, and a bare name are all left alone.
    assert_eq!(tool_result_line("plan", true, "✗ nope"), "  ✓ plan: ✗ nope");
    assert_eq!(
        tool_result_line("plan", true, "done ✓ ok ✓"),
        "  ✓ plan: done ✓ ok ✓"
    );
    assert_eq!(
        tool_result_line("read", true, "read_file ok"),
        "  ✓ read: read_file ok"
    );
    assert_eq!(tool_result_line("plan", true, "plan"), "  ✓ plan: plan");
    assert_eq!(tool_result_line("x", false, "denied"), "  ✗ x: denied");
}
