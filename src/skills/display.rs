use std::path::Path;

use super::Skill;

/// One-line entry for `/skills list`.
pub fn format_list_entry(skill: &Skill) -> String {
    let first_desc = skill
        .description
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .to_string();
    if first_desc.is_empty() {
        skill.name.clone()
    } else {
        let desc = crate::truncate_chars(&first_desc, 72);
        format!("{} — {desc}", skill.name)
    }
}

/// Multi-line detail for `/skills <name> help`.
pub fn format_help(skill: &Skill, project_root: &Path) -> Vec<String> {
    let mut lines = Vec::new();
    lines.push(format!("/{}", skill.name));
    lines.push(format!("  path: {}", skill.display_path(project_root)));
    if !skill.description.is_empty() {
        lines.push(format!("  description: {}", skill.description));
    }
    if !skill.arguments.is_empty() {
        lines.push(format!("  arguments: {}", skill.arguments.join(", ")));
    }
    if let Some(ref hint) = skill.argument_hint {
        lines.push(format!("  usage: /{} {hint}", skill.name));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::super::discover::parse;
    use super::*;
    use std::path::{Path, PathBuf};

    fn fake_path() -> PathBuf {
        PathBuf::from("/proj/.ai/skills/my-skill/SKILL.md")
    }

    #[test]
    fn list_entry_no_description() {
        let skill = parse(&fake_path(), "---\nname: my-skill\n---\nbody\n").unwrap();
        assert_eq!(format_list_entry(&skill), "my-skill");
    }

    #[test]
    fn list_entry_with_description() {
        let skill =
            parse(&fake_path(), "---\nname: my-skill\ndescription: Does things\n---\nbody\n")
                .unwrap();
        assert_eq!(format_list_entry(&skill), "my-skill — Does things");
    }

    #[test]
    fn list_entry_long_description_truncated() {
        let long = "A".repeat(100);
        let content = format!("---\nname: s\ndescription: {long}\n---\nbody\n");
        let skill = parse(&fake_path(), &content).unwrap();
        let entry = format_list_entry(&skill);
        assert!(entry.len() < 100, "should be truncated: {entry}");
        assert!(entry.contains("..."));
    }

    #[test]
    fn help_minimal() {
        let skill = parse(&fake_path(), "---\nname: my-skill\n---\nbody\n").unwrap();
        let lines = format_help(&skill, Path::new("/proj"));
        assert_eq!(lines[0], "/my-skill");
        assert!(lines[1].contains(".ai/skills/my-skill/SKILL.md"));
    }

    #[test]
    fn help_full() {
        let content = "---\nname: deploy\ndescription: Deploys the app\narguments: [env]\nargument-hint: <env>\n---\nbody\n";
        let skill = parse(&fake_path(), content).unwrap();
        let lines = format_help(&skill, Path::new("/proj"));
        assert!(lines.iter().any(|l| l.contains("Deploys the app")));
        assert!(lines.iter().any(|l| l.contains("env")));
        assert!(lines.iter().any(|l| l.contains("usage: /deploy <env>")));
    }

    #[test]
    fn display_path_project_relative() {
        let skill = parse(&fake_path(), "---\nname: s\n---\nbody\n").unwrap();
        assert_eq!(
            skill.display_path(Path::new("/proj")),
            ".ai/skills/my-skill/SKILL.md"
        );
    }
}
