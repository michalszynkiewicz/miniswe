use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::{Skill, SkillEntry};

/// Discover all skills. Project skills shadow global by name.
/// Returns entries sorted by name.
pub fn discover(project_root: &Path) -> Vec<SkillEntry> {
    let mut map: HashMap<String, PathBuf> = Default::default();
    if let Some(home) = dirs::home_dir() {
        collect_from(&home.join(".ai").join("skills"), &mut map);
    }
    collect_from(&project_root.join(".ai").join("skills"), &mut map);
    let mut entries: Vec<SkillEntry> = map
        .into_iter()
        .map(|(name, path)| SkillEntry { name, path })
        .collect();
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    entries
}

fn collect_from(dir: &Path, map: &mut HashMap<String, PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in rd.flatten() {
        let dir_path = entry.path();
        if !dir_path.is_dir() {
            continue;
        }
        let skill_md = dir_path.join("SKILL.md");
        if !skill_md.exists() {
            continue;
        }
        let name = dir_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        if !name.is_empty() {
            map.insert(name, skill_md);
        }
    }
}

/// Find and load a skill by name from project/global paths.
pub fn load_by_name(name: &str, project_root: &Path) -> Option<Skill> {
    discover(project_root)
        .into_iter()
        .find(|e| e.name == name)
        .and_then(|e| load(&e.path).ok())
}

/// Load and parse a SKILL.md file.
pub fn load(path: &Path) -> Result<Skill, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    parse(path, &content)
}

pub(super) fn parse(path: &Path, content: &str) -> Result<Skill, String> {
    let (frontmatter, body) = split_frontmatter(content);

    let name = extract_scalar(&frontmatter, "name").unwrap_or_else(|| {
        path.parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string()
    });

    Ok(Skill {
        name,
        description: extract_scalar(&frontmatter, "description").unwrap_or_default(),
        argument_hint: extract_scalar(&frontmatter, "argument-hint"),
        arguments: extract_list(&frontmatter, "arguments"),
        body: body.to_string(),
        path: path.to_path_buf(),
    })
}

fn split_frontmatter(content: &str) -> (String, &str) {
    let content = content.trim_start();
    let Some(rest) = content.strip_prefix("---") else {
        return (String::new(), content);
    };
    let rest = rest.trim_start_matches('\n');
    let end = rest.find("\n---").unwrap_or(rest.len());
    let fm = &rest[..end];
    let body = if end + 4 < rest.len() {
        rest[end + 4..].trim_start_matches('\n')
    } else {
        ""
    };
    (fm.to_string(), body)
}

fn extract_scalar(fm: &str, key: &str) -> Option<String> {
    let prefix = format!("{key}:");
    for line in fm.lines() {
        if let Some(rest) = line.strip_prefix(&prefix) {
            let val = rest.trim();
            if val.is_empty() {
                continue;
            }
            if val.len() >= 2
                && ((val.starts_with('"') && val.ends_with('"'))
                    || (val.starts_with('\'') && val.ends_with('\'')))
            {
                return Some(val[1..val.len() - 1].to_string());
            }
            return Some(val.to_string());
        }
    }
    None
}

fn extract_list(fm: &str, key: &str) -> Vec<String> {
    let prefix = format!("{key}:");
    let mut in_list = false;
    let mut items = Vec::new();

    for line in fm.lines() {
        if let Some(rest) = line.strip_prefix(&prefix) {
            let val = rest.trim();
            if val.is_empty() {
                in_list = true;
                continue;
            }
            if val.starts_with('[') {
                let inner = val.trim_matches(|c| c == '[' || c == ']');
                return inner
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
            return vec![val.to_string()];
        }
        if in_list {
            let trimmed = line.trim();
            if let Some(item) = trimmed.strip_prefix('-') {
                items.push(item.trim().to_string());
            } else if !trimmed.is_empty() {
                in_list = false;
            }
        }
    }
    items
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fake_path() -> PathBuf {
        PathBuf::from("/proj/.ai/skills/my-skill/SKILL.md")
    }

    #[test]
    fn parse_basic_frontmatter() {
        let content = "---\nname: my-skill\ndescription: Does things\n---\n\nBody text here.\n";
        let skill = parse(&fake_path(), content).unwrap();
        assert_eq!(skill.name, "my-skill");
        assert_eq!(skill.description, "Does things");
        assert_eq!(skill.body, "Body text here.\n");
    }

    #[test]
    fn parse_list_arguments() {
        let content = "---\nname: s\narguments:\n  - src\n  - dst\n---\nbody\n";
        let skill = parse(&fake_path(), content).unwrap();
        assert_eq!(skill.arguments, vec!["src", "dst"]);
    }

    #[test]
    fn parse_inline_list_arguments() {
        let content = "---\nname: s\narguments: [src, dst]\n---\nbody\n";
        let skill = parse(&fake_path(), content).unwrap();
        assert_eq!(skill.arguments, vec!["src", "dst"]);
    }

    #[test]
    fn parse_argument_hint() {
        let content = "---\nname: s\nargument-hint: <branch>\n---\nbody\n";
        let skill = parse(&fake_path(), content).unwrap();
        assert_eq!(skill.argument_hint.as_deref(), Some("<branch>"));
    }

    #[test]
    fn parse_no_frontmatter_falls_back_to_dir_name() {
        let content = "Just a body.";
        let skill = parse(&fake_path(), content).unwrap();
        assert_eq!(skill.name, "my-skill");
        assert_eq!(skill.body, "Just a body.");
    }

    #[test]
    fn parse_quoted_values() {
        let content = "---\nname: \"quoted-skill\"\ndescription: 'single quoted'\n---\nbody\n";
        let skill = parse(&fake_path(), content).unwrap();
        assert_eq!(skill.name, "quoted-skill");
        assert_eq!(skill.description, "single quoted");
    }
}
