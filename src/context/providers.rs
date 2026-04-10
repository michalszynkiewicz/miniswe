//! Pluggable context providers — each injects a section into the system prompt.
//!
//! Add a new capability by implementing `ContextProvider`, then registering it
//! in `default_providers()`. Toggle via `[context.providers]` in config.toml.

use crate::config::Config;
use crate::context::compress;
use crate::knowledge::ProjectIndex;
use crate::knowledge::graph::DependencyGraph;
use crate::knowledge::repo_map;
use std::fs;

/// Input available to all providers for the current turn.
pub struct ProviderInput<'a> {
    pub config: &'a Config,
    pub user_message: &'a str,
    pub keywords: Vec<&'a str>,
    pub plan_only: bool,
    pub mcp_summary: Option<&'a str>,
}

/// A block of context to inject into the system prompt.
pub struct ContextBlock {
    /// Section tag, e.g. "[PROFILE]". Empty string means no header.
    pub header: &'static str,
    pub content: String,
}

/// Trait for pluggable context providers.
pub trait ContextProvider: Send + Sync {
    /// Unique name matching the config key in `[context.providers]`.
    fn name(&self) -> &'static str;

    /// Produce context for this turn, or `None` if nothing to contribute.
    fn provide(&self, input: &ProviderInput) -> Option<ContextBlock>;
}

// ---------------------------------------------------------------------------
// Implementations
// ---------------------------------------------------------------------------

/// Project profile from `.miniswe/profile.md` (language, structure, deps).
pub struct ProfileProvider;

impl ContextProvider for ProfileProvider {
    fn name(&self) -> &'static str {
        "profile"
    }

    fn provide(&self, input: &ProviderInput) -> Option<ContextBlock> {
        let path = input.config.miniswe_path("profile.md");
        let content = fs::read_to_string(path).ok()?;
        Some(ContextBlock {
            header: "",
            content: compress::compress_profile(&content),
        })
    }
}

/// User guide from `.miniswe/guide.md` (project-specific instructions).
pub struct GuideProvider;

impl ContextProvider for GuideProvider {
    fn name(&self) -> &'static str {
        "guide"
    }

    fn provide(&self, input: &ProviderInput) -> Option<ContextBlock> {
        let path = input.config.miniswe_path("guide.md");
        let content = fs::read_to_string(path).ok()?;
        if content.contains("<!-- Add project-specific instructions")
            && content.lines().count() <= 5
        {
            return None;
        }
        Some(ContextBlock {
            header: "[GUIDE]",
            content,
        })
    }
}

/// Architecture notes from `.ai/README.md`.
pub struct ProjectNotesProvider;

impl ContextProvider for ProjectNotesProvider {
    fn name(&self) -> &'static str {
        "project_notes"
    }

    fn provide(&self, input: &ProviderInput) -> Option<ContextBlock> {
        let path = input.config.project_root.join(".ai").join("README.md");
        let content = fs::read_to_string(path).ok()?;
        let budget = input.config.tool_output_budget_chars();
        Some(ContextBlock {
            header: "[PROJECT NOTES]",
            content: crate::truncate_chars(&content, budget),
        })
    }
}

/// Active plan from `.miniswe/plan.md`.
pub struct PlanProvider;

impl ContextProvider for PlanProvider {
    fn name(&self) -> &'static str {
        "plan"
    }

    fn provide(&self, input: &ProviderInput) -> Option<ContextBlock> {
        let path = input.config.miniswe_path("plan.md");
        let content = fs::read_to_string(path).ok()?;
        Some(ContextBlock {
            header: "[PLAN]",
            content,
        })
    }
}

/// Keyword-matched lessons from `.miniswe/lessons.md`.
pub struct LessonsProvider;

impl ContextProvider for LessonsProvider {
    fn name(&self) -> &'static str {
        "lessons"
    }

    fn provide(&self, input: &ProviderInput) -> Option<ContextBlock> {
        let path = input.config.miniswe_path("lessons.md");
        let content = fs::read_to_string(path).ok()?;
        if content.contains("<!-- Accumulated tips") && content.lines().count() <= 5 {
            return None;
        }
        // Short lessons files (< 2000 chars) are always injected in full —
        // they're cheap and the keyword filter often misses relevant sections
        // when section headings don't overlap with the user's phrasing.
        if input.keywords.is_empty() || content.len() < 2000 {
            return Some(ContextBlock {
                header: "[LESSONS]",
                content,
            });
        }
        // Extract only sections matching keywords
        let mut relevant = String::new();
        let mut in_section = false;
        for line in content.lines() {
            if line.starts_with("## ") {
                let heading_lower = line.to_lowercase();
                in_section = input
                    .keywords
                    .iter()
                    .any(|kw| kw.len() >= 3 && heading_lower.contains(&kw.to_lowercase()));
            }
            if in_section {
                relevant.push_str(line);
                relevant.push('\n');
            }
        }
        if relevant.is_empty() {
            None
        } else {
            Some(ContextBlock {
                header: "[LESSONS]",
                content: relevant,
            })
        }
    }
}

/// PageRank-scored repo map, personalized to the current task.
pub struct RepoMapProvider;

impl ContextProvider for RepoMapProvider {
    fn name(&self) -> &'static str {
        "repo_map"
    }

    fn provide(&self, input: &ProviderInput) -> Option<ContextBlock> {
        let miniswe_dir = input.config.miniswe_dir();
        let index = ProjectIndex::load(&miniswe_dir).ok()?;
        let graph = DependencyGraph::load(&miniswe_dir).ok()?;
        let map = repo_map::render(
            &index,
            &graph,
            input.config.context.repo_map_budget,
            &input.keywords,
            &input.config.project_root,
        );
        if map.is_empty() {
            None
        } else {
            Some(ContextBlock {
                header: "[REPO MAP]",
                content: map,
            })
        }
    }
}

/// MCP server summaries (one line each).
pub struct McpProvider;

impl ContextProvider for McpProvider {
    fn name(&self) -> &'static str {
        "mcp"
    }

    fn provide(&self, input: &ProviderInput) -> Option<ContextBlock> {
        let mcp = input.mcp_summary?;
        Some(ContextBlock {
            header: "[MCP SERVERS]",
            content: format!("{mcp}\nmcp_use(server,tool,args)→call MCP tool"),
        })
    }
}

/// Task scratchpad from `.miniswe/scratchpad.md`.
pub struct ScratchpadProvider;

impl ContextProvider for ScratchpadProvider {
    fn name(&self) -> &'static str {
        "scratchpad"
    }

    fn provide(&self, input: &ProviderInput) -> Option<ContextBlock> {
        let path = input.config.miniswe_path("scratchpad.md");
        let content = fs::read_to_string(path).ok()?;
        Some(ContextBlock {
            header: "[SCRATCHPAD]",
            content,
        })
    }
}

/// Embedded usage guide — injected when the user asks meta-questions.
pub struct UsageGuideProvider;

impl ContextProvider for UsageGuideProvider {
    fn name(&self) -> &'static str {
        "usage_guide"
    }

    fn provide(&self, input: &ProviderInput) -> Option<ContextBlock> {
        if !super::is_meta_question(input.user_message) {
            return None;
        }
        Some(ContextBlock {
            header: "[USAGE GUIDE]",
            content: super::USAGE_GUIDE.to_string(),
        })
    }
}

/// Plan-mode flag — marks read-only mode.
pub struct PlanModeProvider;

impl ContextProvider for PlanModeProvider {
    fn name(&self) -> &'static str {
        "plan_mode"
    }

    fn provide(&self, input: &ProviderInput) -> Option<ContextBlock> {
        if !input.plan_only {
            return None;
        }
        Some(ContextBlock {
            header: "[MODE:PLAN]",
            content: "Read-only.No edits/shell.Write plan via task_update.".into(),
        })
    }
}

/// Build the default ordered list of providers.
pub fn default_providers() -> Vec<Box<dyn ContextProvider>> {
    vec![
        Box::new(ProfileProvider),
        Box::new(GuideProvider),
        Box::new(ProjectNotesProvider),
        Box::new(PlanProvider),
        Box::new(LessonsProvider),
        Box::new(RepoMapProvider),
        Box::new(McpProvider),
        Box::new(ScratchpadProvider),
        Box::new(UsageGuideProvider),
        Box::new(PlanModeProvider),
    ]
}
