//! Tests for the structured planning tool.

mod helpers;

use std::fs;
use miniswe::tools::plan;

#[tokio::test]
async fn plan_set_creates_file() {
    let (_tmp, config) = helpers::create_test_project();

    let args = serde_json::json!({
        "action": "set",
        "content": "## Plan\n- [ ] Step one\n- [ ] Step two\n- [ ] Step three\n"
    });
    let result = plan::execute(&args, &config, 1).await.unwrap();

    assert!(result.success);
    let plan = fs::read_to_string(config.miniswe_dir().join("plan.md")).unwrap();
    assert!(plan.contains("Step one"));
    assert!(plan.contains("- [ ]"));
}

#[tokio::test]
async fn plan_check_marks_step() {
    let (_tmp, config) = helpers::create_test_project();

    // Create plan
    let plan_content = "## Plan\n- [ ] First\n- [ ] Second\n- [ ] Third\n";
    fs::create_dir_all(config.miniswe_dir()).ok();
    fs::write(config.miniswe_dir().join("plan.md"), plan_content).unwrap();

    // Check step 2
    let args = serde_json::json!({"action": "check", "step": 2});
    let result = plan::execute(&args, &config, 5).await.unwrap();

    assert!(result.success);
    assert!(result.content.contains("Step 2 checked"));

    let plan = fs::read_to_string(config.miniswe_dir().join("plan.md")).unwrap();
    assert!(plan.contains("- [ ] First"), "step 1 should be unchecked");
    assert!(plan.contains("- [x] (round 5) Second"), "step 2 should be checked with round");
    assert!(plan.contains("- [ ] Third"), "step 3 should be unchecked");
}

#[tokio::test]
async fn plan_check_already_done() {
    let (_tmp, config) = helpers::create_test_project();

    let plan_content = "## Plan\n- [x] (round 3) Already done\n- [ ] Not done\n";
    fs::create_dir_all(config.miniswe_dir()).ok();
    fs::write(config.miniswe_dir().join("plan.md"), plan_content).unwrap();

    let args = serde_json::json!({"action": "check", "step": 1});
    let result = plan::execute(&args, &config, 5).await.unwrap();

    assert!(!result.success, "should fail on already checked: {}", result.content);
}

#[tokio::test]
async fn plan_show_includes_round() {
    let (_tmp, config) = helpers::create_test_project();

    let plan_content = "## Plan\n- [x] (round 2) Done\n- [ ] Pending\n";
    fs::create_dir_all(config.miniswe_dir()).ok();
    fs::write(config.miniswe_dir().join("plan.md"), plan_content).unwrap();

    let args = serde_json::json!({"action": "show"});
    let result = plan::execute(&args, &config, 10).await.unwrap();

    assert!(result.success);
    assert!(result.content.contains("[round 10]"), "should show current round");
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
    fs::write(config.miniswe_dir().join("plan.md"), "## Plan\n- [ ] Do things\n").unwrap();

    let loaded = plan::load_plan(&config);
    assert!(loaded.is_some());
    assert!(loaded.unwrap().contains("Do things"));
}
