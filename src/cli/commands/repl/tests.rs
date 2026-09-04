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
fn consume_interrupt_clears_flag_after_first_read() {
    let cancelled = AtomicBool::new(true);
    assert!(consume_interrupt(&cancelled));
    assert!(!consume_interrupt(&cancelled));
    assert!(!cancelled.load(Ordering::Relaxed));
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
