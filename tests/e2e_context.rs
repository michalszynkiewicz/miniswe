//! E2E tests for context assembly — verifies the real context pipeline
//! with actual .miniswe/ files.

mod helpers;

use std::fs;

use miniswe::context;
use miniswe::llm::Message;

// ── Basic assembly ──────────────────────────────────────────────────

#[test]
fn basic_assembly_includes_system_prompt() {
    let (_tmp, config) = helpers::create_test_project();

    let assembled = context::assemble(&config, "hello", &[], false, None);

    assert!(!assembled.messages.is_empty());
    // First message should be system
    assert_eq!(assembled.messages[0].role, "system");
    let system = assembled.messages[0].content.as_deref().unwrap();
    assert!(system.contains("miniswe"), "system prompt should mention miniswe");
    assert!(system.contains("[RULES]"), "system prompt should have rules");
    assert!(system.contains("[TOOLS]"), "system prompt should have tools");

    // Last message should be the user message
    let last = assembled.messages.last().unwrap();
    assert_eq!(last.role, "user");
    assert_eq!(last.content.as_deref().unwrap(), "hello");
}

#[test]
fn assembly_includes_profile() {
    let (_tmp, config) = helpers::create_test_project();

    // Write a profile (compress_profile converts "- Key: Value" to "key=value|")
    fs::write(
        config.miniswe_path("profile.md"),
        "# Test Project\n## Overview\n- Name: test-project\n- Language: Rust\n- Framework: tokio\n",
    )
    .unwrap();

    let assembled = context::assemble(&config, "test", &[], false, None);
    let system = assembled.messages[0].content.as_deref().unwrap();

    // Compressed profile converts "- Key: Value" → "key=value|..."
    assert!(
        system.contains("test-project") || system.contains("name=test-project"),
        "profile should be included in system context: {}",
        system
    );
}

#[test]
fn assembly_includes_guide() {
    let (_tmp, config) = helpers::create_test_project();

    fs::write(
        config.miniswe_path("guide.md"),
        "Always use snake_case for variables.\nPrefer iterators over loops.\n",
    )
    .unwrap();

    let assembled = context::assemble(&config, "test", &[], false, None);
    let system = assembled.messages[0].content.as_deref().unwrap();

    assert!(system.contains("[GUIDE]"), "should have guide section");
    assert!(system.contains("snake_case"), "guide content should be present");
}

#[test]
fn assembly_skips_template_guide() {
    let (_tmp, config) = helpers::create_test_project();

    // Write the template placeholder (should be skipped)
    fs::write(
        config.miniswe_path("guide.md"),
        "<!-- Add project-specific instructions here -->\n",
    )
    .unwrap();

    let assembled = context::assemble(&config, "test", &[], false, None);
    let system = assembled.messages[0].content.as_deref().unwrap();

    assert!(!system.contains("[GUIDE]"), "template guide should be skipped");
}

// ── Scratchpad ──────────────────────────────────────────────────────

#[test]
fn scratchpad_included_in_context() {
    let (_tmp, config) = helpers::create_test_project();

    fs::write(
        config.miniswe_path("scratchpad.md"),
        "## Current Task\nImplement auth\n\n## Plan\n1. Add middleware\n",
    )
    .unwrap();

    let assembled = context::assemble(&config, "continue", &[], false, None);
    let system = assembled.messages[0].content.as_deref().unwrap();

    assert!(system.contains("[SCRATCHPAD]"), "should have scratchpad section");
    assert!(system.contains("Implement auth"), "scratchpad content should be present");
}

// ── Plan mode ───────────────────────────────────────────────────────

#[test]
fn plan_mode_adds_readonly_marker() {
    let (_tmp, config) = helpers::create_test_project();

    let assembled = context::assemble(&config, "analyze this", &[], true, None);
    let system = assembled.messages[0].content.as_deref().unwrap();

    assert!(
        system.contains("[MODE:PLAN]"),
        "plan mode should add plan marker"
    );
    assert!(system.contains("Read-only"));
}

// ── AI README ───────────────────────────────────────────────────────

#[test]
fn ai_readme_included_in_context() {
    let (_tmp, config) = helpers::create_test_project();

    fs::create_dir_all(helpers::project_path(&config, ".ai")).unwrap();
    fs::write(
        helpers::project_path(&config, ".ai/README.md"),
        "# Architecture\nThis project uses a layered architecture.\n",
    )
    .unwrap();

    let assembled = context::assemble(&config, "test", &[], false, None);
    let system = assembled.messages[0].content.as_deref().unwrap();

    assert!(system.contains("[PROJECT NOTES]"), "should have project notes");
    assert!(system.contains("layered architecture"));
}

// ── MCP summary ─────────────────────────────────────────────────────

#[test]
fn mcp_summary_included_when_provided() {
    let (_tmp, config) = helpers::create_test_project();

    let mcp = "[MCP:github] 3 tools: create_issue, list_prs, review_pr";
    let assembled = context::assemble(&config, "test", &[], false, Some(mcp));
    let system = assembled.messages[0].content.as_deref().unwrap();

    assert!(system.contains("[MCP SERVERS]"));
    assert!(system.contains("github"));
    assert!(system.contains("create_issue"));
}

// ── Lessons ─────────────────────────────────────────────────────────

#[test]
fn relevant_lessons_included() {
    let (_tmp, config) = helpers::create_test_project();

    fs::write(
        config.miniswe_path("lessons.md"),
        "## Testing\nAlways run cargo test before committing.\n\n## Deployment\nUse docker compose.\n",
    )
    .unwrap();

    // "testing" keyword should match the Testing section
    let assembled = context::assemble(&config, "add testing for auth", &[], false, None);
    let system = assembled.messages[0].content.as_deref().unwrap();

    assert!(system.contains("[LESSONS]"));
    assert!(system.contains("cargo test"));
}

// ── History compression ─────────────────────────────────────────────

#[test]
fn compress_history_keeps_recent() {
    let history = vec![
        Message::user("first message"),
        Message::assistant("first response"),
        Message::user("second message"),
        Message::assistant("second response"),
        Message::user("third message"),
        Message::assistant("third response"),
    ];

    // Keep last 4 messages raw
    let compressed = context::compress_history(&history, 4);

    // Should have: summary user + summary assistant + last 4 raw messages
    assert!(compressed.len() >= 4, "should keep at least 4 raw messages, got {}", compressed.len());

    // Last message should be the third response
    let last = compressed.last().unwrap();
    assert_eq!(last.content.as_deref().unwrap(), "third response");
}

#[test]
fn compress_history_noop_when_short() {
    let history = vec![
        Message::user("only message"),
        Message::assistant("only response"),
    ];

    let compressed = context::compress_history(&history, 10);
    assert_eq!(compressed.len(), history.len());
}

#[test]
fn compress_history_summarizes_old_tool_results() {
    use miniswe::llm::{FunctionCall, ToolCall};

    let history = vec![
        Message::user("read main.rs"),
        Message::assistant_tool_calls(vec![ToolCall {
            id: "call_1".into(),
            r#type: "function".into(),
            function: FunctionCall {
                name: "read_file".into(),
                arguments: r#"{"path":"src/main.rs"}"#.into(),
            },
        }]),
        Message::tool_result("call_1", "[src/main.rs: 50 lines]\n   1│fn main() {\n..."),
        Message::assistant("I've read the file."),
        Message::user("now edit it"),
        Message::assistant("Editing..."),
    ];

    // Keep only last 2 messages raw
    let compressed = context::compress_history(&history, 2);

    // The compressed portion should include a summary
    let first_content = compressed[0].content.as_deref().unwrap();
    assert!(
        first_content.contains("summarized"),
        "should have summary header: {}",
        first_content
    );
}

// ── Message sanitization ────────────────────────────────────────────

#[test]
fn sanitize_merges_consecutive_user_messages() {
    let mut messages = vec![
        Message::system("sys"),
        Message::user("first"),
        Message::user("second"),
        Message::assistant("reply"),
    ];

    context::sanitize_messages(&mut messages);

    // Should have merged the two user messages
    let user_count = messages.iter().filter(|m| m.role == "user").count();
    assert_eq!(user_count, 1, "should have merged user messages");

    let user = messages.iter().find(|m| m.role == "user").unwrap();
    let content = user.content.as_deref().unwrap();
    assert!(content.contains("first"));
    assert!(content.contains("second"));
}

#[test]
fn sanitize_inserts_assistant_bridge_between_tool_and_user() {
    use miniswe::llm::{FunctionCall, ToolCall};

    let mut messages = vec![
        Message::system("sys"),
        Message::user("read file"),
        Message::assistant_tool_calls(vec![ToolCall {
            id: "call_1".into(),
            r#type: "function".into(),
            function: FunctionCall {
                name: "read_file".into(),
                arguments: "{}".into(),
            },
        }]),
        Message::tool_result("call_1", "file contents"),
        Message::user("now edit it"),
    ];

    context::sanitize_messages(&mut messages);

    // Find the tool message index
    let tool_idx = messages.iter().position(|m| m.role == "tool").unwrap();
    // Next should be an assistant bridge, not directly user
    if tool_idx + 1 < messages.len() {
        let next = &messages[tool_idx + 1];
        assert_eq!(
            next.role, "assistant",
            "should insert assistant bridge after tool, got: {}",
            next.role
        );
    }
}

#[test]
fn sanitize_removes_duplicate_system() {
    let mut messages = vec![
        Message::system("first system"),
        Message::user("hello"),
        Message::system("second system"),
        Message::assistant("reply"),
    ];

    context::sanitize_messages(&mut messages);

    let system_count = messages.iter().filter(|m| m.role == "system").count();
    assert_eq!(system_count, 1, "should have only one system message");
}

// ── Token estimation ────────────────────────────────────────────────

#[test]
fn token_estimate_roughly_correct() {
    // ~4 chars per token
    let tokens = context::estimate_tokens("hello world"); // 11 chars
    assert!(tokens >= 2 && tokens <= 4, "should be ~2-3 tokens, got {tokens}");

    let tokens = context::estimate_tokens(&"x".repeat(400)); // 400 chars
    assert_eq!(tokens, 100);
}

// ── Assembly token budget ───────────────────────────────────────────

#[test]
fn assembly_produces_reasonable_token_estimate() {
    let (_tmp, config) = helpers::create_test_project();

    let assembled = context::assemble(&config, "hello world", &[], false, None);

    // System prompt alone is ~1.2K tokens, plus the user message
    assert!(
        assembled.token_estimate > 100,
        "should have reasonable token count, got {}",
        assembled.token_estimate
    );
    assert!(
        assembled.token_estimate < config.model.context_window,
        "should be within context window"
    );
}
