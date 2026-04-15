//! `Step` — the unit of a plan, plus markdown round-tripping.

#[derive(Debug, Clone)]
pub struct Step {
    pub checked: bool,
    pub checked_round: Option<usize>,
    pub description: String,
    pub compile: bool,
    pub reason: Option<String>,
}

impl Step {
    pub fn to_markdown(&self) -> String {
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

    pub fn from_markdown(line: &str) -> Option<Self> {
        let trimmed = line.trim_start();
        if !trimmed.starts_with("- [") {
            return None;
        }

        let (checked, checked_round, rest) =
            if let Some(after_check) = trimmed.strip_prefix("- [x]") {
                // Parse optional round: "- [x] (round 5) description"
                let after_check = after_check.trim_start();
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
            } else if let Some(after_check) = trimmed.strip_prefix("- [ ]") {
                (false, None, after_check.trim_start().to_string())
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
pub fn parse_steps(content: &str) -> Vec<Step> {
    content.lines().filter_map(Step::from_markdown).collect()
}

/// Serialize steps back to markdown.
pub fn steps_to_markdown(steps: &[Step]) -> String {
    steps
        .iter()
        .map(|s| s.to_markdown())
        .collect::<Vec<_>>()
        .join("\n")
}

/// One-line preview of the first few steps.
pub fn plan_preview(steps: &[Step]) -> String {
    steps
        .iter()
        .take(4)
        .enumerate()
        .map(|(idx, step)| format!("{}. {}", idx + 1, step.description))
        .collect::<Vec<_>>()
        .join("; ")
}
