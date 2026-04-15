//! Action dispatch for the `plan` tool (`set`, `check`, `refine`,
//! `show`, `scratchpad`, `help`), plus the language-agnostic compile
//! gate used by the `check` action.

use anyhow::Result;
use serde_json::Value;
use std::fs;

use crate::config::Config;
use crate::tools::ToolResult;

use super::hints::architecture_review_hint;
use super::step::{Step, parse_steps, plan_preview, steps_to_markdown};
use super::validate::{MAX_PLAN_STEPS, validate_steps};

/// Run the project-level compile check (language-agnostic).
/// Returns (success, output).
async fn run_compile_check(config: &Config) -> (bool, String) {
    let project_root = config.project_root.clone();

    // Detect language from project markers
    let (cmd, args): (&str, Vec<String>) = if project_root.join("Cargo.toml").exists() {
        (
            "cargo",
            vec![
                "check".into(),
                "--tests".into(),
                "--message-format=short".into(),
            ],
        )
    } else if project_root.join("tsconfig.json").exists()
        || project_root.join("package.json").exists()
    {
        (
            "npx",
            vec![
                "tsc".into(),
                "--noEmit".into(),
                "--pretty".into(),
                "false".into(),
            ],
        )
    } else if project_root.join("go.mod").exists() {
        ("go", vec!["build".into(), "./...".into()])
    } else if project_root.join("pyproject.toml").exists() || project_root.join("setup.py").exists()
    {
        ("python3", vec!["-m".into(), "py_compile".into()])
    } else if project_root.join("pom.xml").exists() {
        ("mvn", vec!["compile".into(), "-q".into()])
    } else if project_root.join("build.gradle").exists() {
        ("gradle", vec!["compileJava".into(), "-q".into()])
    } else {
        return (
            true,
            "No recognized build system — skipping compile check.".into(),
        );
    };

    let cmd = cmd.to_string();
    let result = tokio::task::spawn_blocking(move || {
        crate::tools::run_check_with_timeout(&cmd, &args, &project_root, 30)
    })
    .await;

    match result {
        Ok(Some((success, output))) => (success, output),
        Ok(None) => (false, "Compile check not available or timed out.".into()),
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
                        return Ok(ToolResult::err(format!(
                            "Step {} has empty description.",
                            i + 1
                        )));
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
                    return Ok(ToolResult::err(
                        "Missing 'content' or 'steps' for plan set.".into(),
                    ));
                }
                // Parse markdown content — add [compile] tag if not present
                let mut steps = Vec::new();
                for line in content.lines() {
                    if let Some(step) = Step::from_markdown(line) {
                        steps.push(step);
                    } else if line.trim_start().starts_with("- [ ]")
                        || line.trim_start().starts_with("- [x]")
                    {
                        // Bare checkbox without compile tag — parse as compile: true
                        let trimmed = line.trim_start();
                        let desc = trimmed
                            .strip_prefix("- [ ]")
                            .or_else(|| trimmed.strip_prefix("- [x]"))
                            .map(|rest| rest.trim().to_string())
                            .unwrap_or_default();
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
                return Ok(ToolResult::err(
                    "Missing 'content' or 'steps' for plan set.".into(),
                ));
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
            let preview = crate::truncate_chars(&plan_preview(&steps), 180);
            let hint = architecture_review_hint(config.tools.edit_mode);
            Ok(ToolResult::ok(format!(
                "✓ Plan saved ({} steps, {} compile-gated, {} deferred): {preview}\n{hint}\n\n{md}",
                steps.len(),
                compile_count,
                nocompile_count
            )))
        }

        "check" => {
            let step_num = args["step"].as_u64().unwrap_or(0) as usize;
            if step_num == 0 {
                return Ok(ToolResult::err(
                    "Missing 'step' number to check off.".into(),
                ));
            }

            let content = fs::read_to_string(&plan_path).unwrap_or_default();
            if content.is_empty() {
                return Ok(ToolResult::err(
                    "No plan exists. Use action='set' first.".into(),
                ));
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
                return Ok(ToolResult::err(format!(
                    "Step {step_num} is already checked."
                )));
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
                return Ok(ToolResult::err(
                    "No plan exists. Use action='set' first.".into(),
                ));
            }

            let steps = parse_steps(&content);
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
                    return Ok(ToolResult::err(format!(
                        "Substep {} has empty description.",
                        i + 1
                    )));
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
                Ok(ToolResult::ok(format!(
                    "[round {current_round}]\n{content}"
                )))
            }
        }

        "scratchpad" => crate::tools::task_update::execute(args, config).await,

        "help" => Ok(ToolResult::ok(
            crate::tools::definitions::plan_help().into(),
        )),

        _ => Ok(ToolResult::err(format!(
            "Unknown action: {action}. Use 'set', 'check', 'refine', 'show', or 'scratchpad'."
        ))),
    }
}
