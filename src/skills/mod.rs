//! Skill support: load and render `.ai/skills/<name>/SKILL.md` files.
//!
//! Compatible with Claude Code's skill format: YAML frontmatter + markdown body,
//! `$ARGUMENTS` / `$0` / `$N` / named-arg substitution, and `!`cmd`` / ` ```! ` shell injection.
//!
//! Discovery order: project (`.ai/skills/`) shadows global (`~/.ai/skills/`).

mod discover;
mod display;
mod render;

use std::path::PathBuf;

pub use discover::{discover, load, load_by_name};
pub use display::{format_help, format_list_entry};
pub use render::render;

/// A fully loaded skill.
pub struct Skill {
    pub name: String,
    pub description: String,
    /// Named positional arguments declared in frontmatter `arguments:` field.
    pub arguments: Vec<String>,
    pub argument_hint: Option<String>,
    pub body: String,
    pub path: PathBuf,
}

impl Skill {
    /// Short display path: project-relative or `~/`-relative.
    pub fn display_path(&self, project_root: &std::path::Path) -> String {
        if let Ok(rel) = self.path.strip_prefix(project_root) {
            return rel.display().to_string();
        }
        if let Some(home) = dirs::home_dir()
            && let Ok(rel) = self.path.strip_prefix(&home)
        {
            return format!("~/{}", rel.display());
        }
        self.path.display().to_string()
    }
}

/// A discovered skill entry (name + path, not yet fully loaded).
pub struct SkillEntry {
    pub name: String,
    pub path: PathBuf,
}
