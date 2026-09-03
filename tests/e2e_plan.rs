//! Tests for the structured planning tool.

mod helpers;

use miniswe::tools::plan;
use std::fs;

#[tokio::test]
async fn plan_set_creates_file() {
    let (_tmp, config) = helpers::create_test_project();

    let args = serde_json::json!({
        "action": "set",
        "content": "## Plan\n- [ ] Step one\n- [ ] Step two\n- [ ] Step three\n"
    });
    let result = plan::execute(&args, &config, 1).await.unwrap();

    assert!(result.success);
    assert!(result.content.contains("Before editing"));
    assert!(result.content.contains("plan(action='refine')"));
    let plan = fs::read_to_string(config.session_path("plan.md")).unwrap();
    assert!(plan.contains("Step one"));
    assert!(plan.contains("- [ ]"));
}

#[tokio::test]
async fn plan_check_marks_step() {
    let (_tmp, config) = helpers::create_test_project();

    // Create plan
    let plan_content = "## Plan\n- [ ] First\n- [ ] Second\n- [ ] Third\n";
    fs::create_dir_all(config.miniswe_dir()).ok();
    fs::write(config.session_path("plan.md"), plan_content).unwrap();

    // Check step 2
    let args = serde_json::json!({"action": "check", "step": 2});
    let result = plan::execute(&args, &config, 5).await.unwrap();

    assert!(result.success);
    assert!(result.content.contains("Step 2 checked"));

    let plan = fs::read_to_string(config.session_path("plan.md")).unwrap();
    assert!(plan.contains("- [ ] First"), "step 1 should be unchecked");
    assert!(
        plan.contains("- [x] (round 5) Second"),
        "step 2 should be checked with round"
    );
    assert!(plan.contains("- [ ] Third"), "step 3 should be unchecked");
}

#[tokio::test]
async fn plan_check_already_done() {
    let (_tmp, config) = helpers::create_test_project();

    let plan_content = "## Plan\n- [x] (round 3) Already done\n- [ ] Not done\n";
    fs::create_dir_all(config.miniswe_dir()).ok();
    fs::write(config.session_path("plan.md"), plan_content).unwrap();

    let args = serde_json::json!({"action": "check", "step": 1});
    let result = plan::execute(&args, &config, 5).await.unwrap();

    assert!(
        !result.success,
        "should fail on already checked: {}",
        result.content
    );
}

#[tokio::test]
async fn plan_show_includes_round() {
    let (_tmp, config) = helpers::create_test_project();

    let plan_content = "## Plan\n- [x] (round 2) Done\n- [ ] Pending\n";
    fs::create_dir_all(config.miniswe_dir()).ok();
    fs::write(config.session_path("plan.md"), plan_content).unwrap();

    let args = serde_json::json!({"action": "show"});
    let result = plan::execute(&args, &config, 10).await.unwrap();

    assert!(result.success);
    assert!(
        result.content.contains("[round 10]"),
        "should show current round"
    );
    assert!(result.content.contains("Pending"));
}

#[tokio::test]
async fn plan_show_empty() {
    let (_tmp, config) = helpers::create_test_project();

    let args = serde_json::json!({"action": "show"});
    let result = plan::execute(&args, &config, 1).await.unwrap();

    assert!(result.success);
    assert!(result.content.contains("No plan"));
}

#[tokio::test]
async fn plan_load_for_context() {
    let (_tmp, config) = helpers::create_test_project();

    // No plan yet
    assert!(plan::load_plan(&config).is_none());

    // Create plan
    fs::create_dir_all(config.miniswe_dir()).ok();
    fs::write(config.session_path("plan.md"), "## Plan\n- [ ] Do things\n").unwrap();

    let loaded = plan::load_plan(&config);
    assert!(loaded.is_some());
    assert!(loaded.unwrap().contains("Do things"));
}

#[tokio::test]
async fn plan_failure_hint_shows_progress_and_next_step() {
    let (_tmp, config) = helpers::create_test_project();
    fs::create_dir_all(config.miniswe_dir()).ok();
    fs::write(
        config.session_path("plan.md"),
        "- [x] (round 2) Add flag [compile]\n- [x] (round 4) Update assemble [compile]\n- [ ] Update call sites [compile]\n- [ ] Run tests [compile]\n",
    )
    .unwrap();

    let hint = plan::failure_hint(&config).unwrap();
    assert!(hint.contains("Plan: 2/4 done"));
    assert!(hint.contains("Done: 1 Add flag; 2 Update assemble"));
    assert!(hint.contains("Next: 3 Update call sites; 4 Run tests"));
    assert!(hint.contains("Before fixing"));
    assert!(hint.contains("plan(action='refine')"));
}

/// A second miniswe process in the same project must neither see nor
/// destroy the first one's plan.
///
/// Regression: session state used to live at a fixed `.miniswe/plan.md`,
/// and every non-`--continue` start deleted it. The benchmark task is
/// miniswe itself, so when the agent smoke-tested the binary it had just
/// built inside its own workspace, the nested run wiped its parent's plan
/// mid-run — which also stripped the parent's edit tools, since
/// `plan_exists` gates them.
#[tokio::test]
async fn nested_session_cannot_clobber_parent_plan() {
    let (_tmp, parent) = helpers::create_test_project();
    let args = serde_json::json!({
        "action": "set",
        "content": "## Plan\n- [ ] Parent step one\n- [ ] Parent step two\n"
    });
    plan::execute(&args, &parent, 1).await.unwrap();
    assert!(plan::plan_exists(&parent));

    // A nested run: same project root, fresh session.
    let mut nested = parent.clone();
    nested.session_id = "nested-run".to_string();
    nested.ensure_session_dir().unwrap();

    assert_ne!(
        parent.session_path("plan.md"),
        nested.session_path("plan.md")
    );
    assert!(
        !plan::plan_exists(&nested),
        "nested run must start with no plan of its own"
    );
    assert!(
        plan::plan_exists(&parent),
        "parent's plan must survive a nested run starting up"
    );

    // The nested run setting its own plan leaves the parent's untouched.
    let nested_args = serde_json::json!({
        "action": "set",
        "content": "## Plan\n- [ ] Nested step\n"
    });
    plan::execute(&nested_args, &nested, 1).await.unwrap();

    let parent_plan = fs::read_to_string(parent.session_path("plan.md")).unwrap();
    assert!(parent_plan.contains("Parent step one"), "{parent_plan}");
    assert!(!parent_plan.contains("Nested step"), "{parent_plan}");
}
