//! Structured planning tool — high-level plan with checkable steps.
//!
//! The model creates a plan at session start, then checks off steps
//! as it completes them. Each step can have sub-steps for the current
//! work. The plan persists in `.miniswe/plan.md` and is injected
//! into context each round.

use anyhow::Result;
use serde_json::Value;
use std::fs;

use crate::config::Config;
use super::ToolResult;

/// Update the structured plan.
///
/// The model can:
/// - Set the initial plan (full content)
/// - Check off a step (by number)
/// - Set sub-steps for the current step
pub async fn execute(args: &Value, config: &Config, current_round: usize) -> Result<ToolResult> {
    let action = args["action"].as_str().unwrap_or("set");
    let plan_path = config.miniswe_dir().join("plan.md");

    match action {
        "set" => {
            // Set or replace the full plan
            let content = args["content"].as_str().unwrap_or("");
            if content.is_empty() {
                return Ok(ToolResult::err("Missing 'content' for plan set.".into()));
            }
            fs::create_dir_all(config.miniswe_dir()).ok();
            fs::write(&plan_path, content)?;
            Ok(ToolResult::ok(format!("✓ Plan saved ({} lines)", content.lines().count())))
        }
        "check" => {
            // Check off a step by number
            let step = args["step"].as_u64().unwrap_or(0) as usize;
            if step == 0 {
                return Ok(ToolResult::err("Missing 'step' number to check off.".into()));
            }

            let content = fs::read_to_string(&plan_path).unwrap_or_default();
            if content.is_empty() {
                return Ok(ToolResult::err("No plan exists. Use action='set' first.".into()));
            }

            let mut lines: Vec<String> = content.lines().map(|l| l.to_string()).collect();
            let mut checked = false;
            let mut step_count = 0;

            for line in &mut lines {
                if line.trim_start().starts_with("- [ ]") || line.trim_start().starts_with("- [x]") {
                    step_count += 1;
                    if step_count == step && line.contains("- [ ]") {
                        *line = line.replace("- [ ]", &format!("- [x] (round {current_round})"));
                        checked = true;
                        break;
                    }
                }
            }

            if !checked {
                return Ok(ToolResult::err(format!("Step {step} not found or already checked.")));
            }

            let new_content = lines.join("\n");
            fs::write(&plan_path, &new_content)?;
            Ok(ToolResult::ok(format!("✓ Step {step} checked off at round {current_round}")))
        }
        "show" => {
            let content = fs::read_to_string(&plan_path).unwrap_or_default();
            if content.is_empty() {
                Ok(ToolResult::ok("No plan exists yet.".into()))
            } else {
                Ok(ToolResult::ok(format!("[round {current_round}]\n{content}")))
            }
        }
        _ => Ok(ToolResult::err(format!("Unknown action: {action}. Use 'set', 'check', or 'show'."))),
    }
}

/// Check if a plan has been created.
pub fn plan_exists(config: &Config) -> bool {
    let plan_path = config.miniswe_dir().join("plan.md");
    plan_path.exists() && fs::read_to_string(&plan_path).map(|c| !c.trim().is_empty()).unwrap_or(false)
}

/// Load the current plan for context injection.
pub fn load_plan(config: &Config) -> Option<String> {
    let plan_path = config.miniswe_dir().join("plan.md");
    let content = fs::read_to_string(plan_path).ok()?;
    if content.trim().is_empty() {
        None
    } else {
        Some(content)
    }
}
