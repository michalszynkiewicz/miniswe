//! Structured planning tool — high-level plan with checkable steps.
//!
//! The model creates a plan at session start, then checks off steps
//! as it completes them. Each step declares whether the project should
//! compile at its end (`compile: true`). The `check` action enforces
//! this by running `cargo check` (or the appropriate checker). If a
//! step proves too coarse, the model can `refine` it into substeps.
//!
//! The plan persists in `.miniswe/plan.md` and is injected into context
//! each round.

use anyhow::Result;
use serde_json::Value;
use std::fs;

use crate::config::Config;
use super::ToolResult;

/// A single plan step with compile gate metadata.
#[derive(Debug, Clone)]
struct Step {
    checked: bool,
    checked_round: Option<usize>,
    description: String,
    compile: bool,
    reason: Option<String>,
}

impl Step {
    fn to_markdown(&self) -> String {
        let check = if self.checked {
            format!("[x] (round {})", self.checked_round.unwrap_or(0))
        } else {
            "[ ]".to_string()
        };
        let compile_tag = if self.compile {
            " [compile]".to_string()
        } else {
            format!(" [no-compile: {}]", self.reason.as_deref().unwrap_or("?"))
        };
        format!("- {check} {}{compile_tag}", self.description)
    }

    fn from_markdown(line: &str) -> Option<Self> {
        let trimmed = line.trim_start();
        if !trimmed.starts_with("- [") {
            return None;
        }

        let (checked, checked_round, rest) = if trimmed.starts_with("- [x]") {
            // Parse optional round: "- [x] (round 5) description"
            let after_check = &trimmed[5..].trim_start();
            if let Some(rp) = after_check.strip_prefix("(round ") {
                if let Some(end) = rp.find(')') {
                    let round = rp[..end].parse().ok();
                    (true, round, rp[end + 1..].trim_start().to_string())
                } else {
                    (true, None, after_check.to_string())
                }
            } else {
                (true, None, after_check.to_string())
            }
        } else if trimmed.starts_with("- [ ]") {
            (false, None, trimmed[5..].trim_start().to_string())
        } else {
            return None;
        };

        // Parse compile tag from end
        let (description, compile, reason) = if let Some(idx) = rest.rfind(" [compile]") {
            (rest[..idx].to_string(), true, None)
        } else if let Some(idx) = rest.rfind(" [no-compile: ") {
            let reason_start = idx + " [no-compile: ".len();
            let reason_end = rest.len().saturating_sub(1); // strip trailing ]
            let reason = rest[reason_start..reason_end].to_string();
            (rest[..idx].to_string(), false, Some(reason))
        } else {
            // Legacy format: no compile tag, default to compile: true
            (rest, true, None)
        };

        Some(Step {
            checked,
            checked_round,
            description,
            compile,
            reason,
        })
    }
}

/// Parse all steps from plan markdown.
fn parse_steps(content: &str) -> Vec<Step> {
    content
        .lines()
        .filter_map(Step::from_markdown)
        .collect()
}

/// Serialize steps back to markdown.
fn steps_to_markdown(steps: &[Step]) -> String {
    steps.iter().map(|s| s.to_markdown()).collect::<Vec<_>>().join("\n")
}

/// Validate plan invariants:
/// - Last step must be compile: true
/// - Any compile: false must have a reason
/// - Any compile: false must be followed by a compile: true before end
fn validate_steps(steps: &[Step]) -> Result<(), String> {
    if steps.is_empty() {
        return Err("Plan must have at least one step.".into());
    }

    // Last step must compile
    if !steps.last().unwrap().compile {
        return Err("Last step must have compile: true. The plan cannot end with a broken tree.".into());
    }

    // Check compile: false steps
    for (i, step) in steps.iter().enumerate() {
        if !step.compile {
            // Must have reason
            if step.reason.as_ref().map_or(true, |r| r.trim().is_empty()) {
                return Err(format!(
                    "Step {} has compile: false but no reason. Explain why the tree will be broken.",
                    i + 1
                ));
            }
            // Must be followed by a compile: true before end
            let has_restore = steps[i + 1..].iter().any(|s| s.compile);
            if !has_restore {
                return Err(format!(
                    "Step {} has compile: false but no subsequent compile: true step to restore the invariant.",
                    i + 1
                ));
            }
        }
    }

    Ok(())
}

/// Max total plan steps.
const MAX_PLAN_STEPS: usize = 30;

/// Run the project-level compile check (`cargo check` or equivalent).
/// Returns (success, output).
async fn run_compile_check(config: &Config) -> (bool, String) {
    let project_root = config.project_root.clone();
    let result = tokio::task::spawn_blocking(move || {
        super::run_check_with_timeout(
            "cargo",
            &["check".into(), "--tests".into(), "--message-format=short".into()],
            &project_root,
            30,
        )
    })
    .await;

    match result {
        Ok(Some((success, output))) => (success, output),
        Ok(None) => (false, "Could not run cargo check (not available?)".into()),
        Err(e) => (false, format!("Check task failed: {e}")),
    }
}

/// Execute the plan tool.
pub async fn execute(args: &Value, config: &Config, current_round: usize) -> Result<ToolResult> {
    let action = args["action"].as_str().unwrap_or("set");
    let plan_path = config.miniswe_dir().join("plan.md");

    match action {
        "set" => {
            // Parse steps from structured input or raw content
            let steps = if let Some(steps_arr) = args["steps"].as_array() {
                // Structured input: array of step objects
                let mut parsed = Vec::new();
                for (i, step_val) in steps_arr.iter().enumerate() {
                    let desc = step_val["description"]
                        .as_str()
                        .or_else(|| step_val.as_str())
                        .unwrap_or("")
                        .to_string();
                    if desc.is_empty() {
                        return Ok(ToolResult::err(format!("Step {} has empty description.", i + 1)));
                    }
                    let compile = step_val["compile"].as_bool().unwrap_or(true);
                    let reason = step_val["reason"].as_str().map(|s| s.to_string());
                    parsed.push(Step {
                        checked: false,
                        checked_round: None,
                        description: desc,
                        compile,
                        reason,
                    });
                }
                parsed
            } else if let Some(content) = args["content"].as_str() {
                if content.is_empty() {
                    return Ok(ToolResult::err("Missing 'content' or 'steps' for plan set.".into()));
                }
                // Parse markdown content — add [compile] tag if not present
                let mut steps = Vec::new();
                for line in content.lines() {
                    if let Some(step) = Step::from_markdown(line) {
                        steps.push(step);
                    } else if line.trim_start().starts_with("- [ ]") || line.trim_start().starts_with("- [x]") {
                        // Bare checkbox without compile tag — parse as compile: true
                        let trimmed = line.trim_start();
                        let desc = if trimmed.starts_with("- [ ]") {
                            trimmed[5..].trim().to_string()
                        } else {
                            trimmed[5..].trim().to_string()
                        };
                        steps.push(Step {
                            checked: trimmed.starts_with("- [x]"),
                            checked_round: None,
                            description: desc,
                            compile: true,
                            reason: None,
                        });
                    }
                }
                if steps.is_empty() {
                    // Treat raw text lines as steps
                    for line in content.lines() {
                        let l = line.trim();
                        if !l.is_empty() {
                            steps.push(Step {
                                checked: false,
                                checked_round: None,
                                description: l.to_string(),
                                compile: true,
                                reason: None,
                            });
                        }
                    }
                }
                steps
            } else {
                return Ok(ToolResult::err("Missing 'content' or 'steps' for plan set.".into()));
            };

            // Validate
            if let Err(e) = validate_steps(&steps) {
                return Ok(ToolResult::err(format!("Invalid plan: {e}")));
            }

            if steps.len() > MAX_PLAN_STEPS {
                return Ok(ToolResult::err(format!(
                    "Plan has {} steps (max {MAX_PLAN_STEPS}). Break into fewer, coarser steps.",
                    steps.len()
                )));
            }

            fs::create_dir_all(config.miniswe_dir()).ok();
            let md = steps_to_markdown(&steps);
            fs::write(&plan_path, &md)?;

            let compile_count = steps.iter().filter(|s| s.compile).count();
            let nocompile_count = steps.len() - compile_count;
            Ok(ToolResult::ok(format!(
                "✓ Plan saved ({} steps, {} compile-gated, {} deferred)",
                steps.len(),
                compile_count,
                nocompile_count
            )))
        }

        "check" => {
            let step_num = args["step"].as_u64().unwrap_or(0) as usize;
            if step_num == 0 {
                return Ok(ToolResult::err("Missing 'step' number to check off.".into()));
            }

            let content = fs::read_to_string(&plan_path).unwrap_or_default();
            if content.is_empty() {
                return Ok(ToolResult::err("No plan exists. Use action='set' first.".into()));
            }

            let mut steps = parse_steps(&content);
            if step_num > steps.len() {
                return Ok(ToolResult::err(format!(
                    "Step {step_num} out of range (plan has {} steps).",
                    steps.len()
                )));
            }

            let is_checked = steps[step_num - 1].checked;
            let needs_compile = steps[step_num - 1].compile;

            if is_checked {
                return Ok(ToolResult::err(format!("Step {step_num} is already checked.")));
            }

            // If this step has compile: true, run the compile gate
            if needs_compile {
                let (success, output) = run_compile_check(config).await;
                if !success {
                    let error_lines: Vec<&str> = output
                        .lines()
                        .filter(|l| l.contains("error") || l.starts_with("  "))
                        .take(30)
                        .collect();

                    let mut msg = format!(
                        "✗ Step {step_num} compile gate FAILED. Cannot check off — project does not compile.\n\n"
                    );
                    msg.push_str("Errors:\n");
                    msg.push_str(&error_lines.join("\n"));
                    msg.push_str("\n\nFix the errors, or if the step is too coarse, use action='refine' to break it into substeps.");
                    return Ok(ToolResult::err(msg));
                }
            }

            // Mark checked
            steps[step_num - 1].checked = true;
            steps[step_num - 1].checked_round = Some(current_round);

            let md = steps_to_markdown(&steps);
            fs::write(&plan_path, &md)?;

            let compile_note = if needs_compile {
                " (compile gate passed ✓)"
            } else {
                ""
            };
            Ok(ToolResult::ok(format!(
                "✓ Step {step_num} checked off at round {current_round}{compile_note}"
            )))
        }

        "refine" => {
            let step_num = args["step"].as_u64().unwrap_or(0) as usize;
            if step_num == 0 {
                return Ok(ToolResult::err("Missing 'step' number to refine.".into()));
            }

            let content = fs::read_to_string(&plan_path).unwrap_or_default();
            if content.is_empty() {
                return Ok(ToolResult::err("No plan exists. Use action='set' first.".into()));
            }

            let mut steps = parse_steps(&content);
            if step_num > steps.len() {
                return Ok(ToolResult::err(format!(
                    "Step {step_num} out of range (plan has {} steps).",
                    steps.len()
                )));
            }

            if steps[step_num - 1].checked {
                return Ok(ToolResult::err(format!(
                    "Step {step_num} is already checked. Cannot refine a completed step."
                )));
            }

            // Parse substeps
            let substeps_arr = args["substeps"]
                .as_array()
                .ok_or_else(|| anyhow::anyhow!("Missing 'substeps' array for refine."))?;

            if substeps_arr.is_empty() {
                return Ok(ToolResult::err("substeps array is empty.".into()));
            }

            let parent_compile = steps[step_num - 1].compile;
            let mut new_substeps = Vec::new();
            for (i, sv) in substeps_arr.iter().enumerate() {
                let desc = sv["description"]
                    .as_str()
                    .or_else(|| sv.as_str())
                    .unwrap_or("")
                    .to_string();
                if desc.is_empty() {
                    return Ok(ToolResult::err(format!("Substep {} has empty description.", i + 1)));
                }
                let compile = sv["compile"].as_bool().unwrap_or(parent_compile);
                let reason = sv["reason"].as_str().map(|s| s.to_string());
                new_substeps.push(Step {
                    checked: false,
                    checked_round: None,
                    description: desc,
                    compile,
                    reason,
                });
            }

            // Replace step N with substeps
            let mut new_steps = Vec::new();
            new_steps.extend_from_slice(&steps[..step_num - 1]);
            new_steps.extend(new_substeps.iter().cloned());
            new_steps.extend_from_slice(&steps[step_num..]);

            // Validate the resulting plan
            if let Err(e) = validate_steps(&new_steps) {
                return Ok(ToolResult::err(format!("Refined plan invalid: {e}")));
            }

            if new_steps.len() > MAX_PLAN_STEPS {
                return Ok(ToolResult::err(format!(
                    "Refined plan would have {} steps (max {MAX_PLAN_STEPS}). Refine less aggressively.",
                    new_steps.len()
                )));
            }

            let md = steps_to_markdown(&new_steps);
            fs::write(&plan_path, &md)?;

            Ok(ToolResult::ok(format!(
                "✓ Step {step_num} refined into {} substeps. Plan now has {} steps.",
                new_substeps.len(),
                new_steps.len()
            )))
        }

        "show" => {
            let content = fs::read_to_string(&plan_path).unwrap_or_default();
            if content.is_empty() {
                Ok(ToolResult::ok("No plan exists yet.".into()))
            } else {
                Ok(ToolResult::ok(format!("[round {current_round}]\n{content}")))
            }
        }

        _ => Ok(ToolResult::err(format!(
            "Unknown action: {action}. Use 'set', 'check', 'refine', or 'show'."
        ))),
    }
}

/// Check if a plan has been created.
pub fn plan_exists(config: &Config) -> bool {
    let plan_path = config.miniswe_dir().join("plan.md");
    plan_path.exists()
        && fs::read_to_string(&plan_path)
            .map(|c| !c.trim().is_empty())
            .unwrap_or(false)
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
