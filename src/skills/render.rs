use std::process::Command;

use super::Skill;

/// Render the skill body: substitute arguments, run shell injections.
pub fn render(skill: &Skill, args: &str) -> Result<String, String> {
    let mut body = skill.body.clone();

    let arg_values: Vec<&str> = if args.is_empty() {
        Vec::new()
    } else if skill.arguments.is_empty() {
        vec![args]
    } else {
        args.splitn(skill.arguments.len(), ' ').collect()
    };

    // Named substitution ($name) and positional by index ($0, $1, ...)
    for (i, arg_name) in skill.arguments.iter().enumerate() {
        let val = arg_values.get(i).copied().unwrap_or("");
        body = body.replace(&format!("${arg_name}"), val);
        body = body.replace(&format!("${i}"), val);
    }
    // Bare positional for skills without declared arguments
    for (i, val) in arg_values.iter().enumerate() {
        body = body.replace(&format!("${i}"), val);
    }
    // $ARGUMENTS = entire args string
    body = body.replace("$ARGUMENTS", args);

    process_shell_injections(&body)
}

fn process_shell_injections(content: &str) -> Result<String, String> {
    process_inline_injections(&process_block_injections(content)?)
}

/// Replace ` ```!\ncmds\n``` ` blocks with their stdout.
fn process_block_injections(content: &str) -> Result<String, String> {
    let mut out = String::new();
    let mut rest = content;
    while let Some(start) = rest.find("```!\n") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 5..];
        if let Some(end) = after.find("\n```") {
            out.push_str(&run_shell(&after[..end])?);
            rest = &after[end + 4..];
        } else {
            out.push_str(&rest[start..]);
            return Ok(out);
        }
    }
    out.push_str(rest);
    Ok(out)
}

/// Replace `` !`cmd` `` inline with trimmed stdout.
fn process_inline_injections(content: &str) -> Result<String, String> {
    let mut out = String::new();
    let mut rest = content;
    while let Some(start) = rest.find("!`") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        if let Some(end) = after.find('`') {
            let cmd_out = run_shell(&after[..end])?;
            out.push_str(cmd_out.trim_end_matches('\n'));
            rest = &after[end + 1..];
        } else {
            out.push_str(&rest[start..]);
            return Ok(out);
        }
    }
    out.push_str(rest);
    Ok(out)
}

fn run_shell(cmd: &str) -> Result<String, String> {
    let output = Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .output()
        .map_err(|e| format!("shell injection failed: {e}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("shell injection `{cmd}` failed: {stderr}"));
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

#[cfg(test)]
mod tests {
    use super::super::discover::parse;
    use super::*;
    use std::path::PathBuf;

    fn fake_path() -> PathBuf {
        PathBuf::from("/proj/.ai/skills/s/SKILL.md")
    }

    fn skill(body: &str) -> Skill {
        parse(&fake_path(), &format!("---\nname: s\n---\n{body}")).unwrap()
    }

    fn skill_with_args(args_decl: &str, body: &str) -> Skill {
        parse(
            &fake_path(),
            &format!("---\nname: s\narguments: [{args_decl}]\n---\n{body}"),
        )
        .unwrap()
    }

    #[test]
    fn render_no_substitutions() {
        let s = skill("Plain body.\n");
        assert_eq!(render(&s, "").unwrap(), "Plain body.\n");
    }

    #[test]
    fn render_dollar_arguments() {
        let s = skill("Args: $ARGUMENTS\n");
        assert_eq!(render(&s, "hello world").unwrap(), "Args: hello world\n");
    }

    #[test]
    fn render_dollar_arguments_empty() {
        let s = skill("Args: $ARGUMENTS\n");
        assert_eq!(render(&s, "").unwrap(), "Args: \n");
    }

    #[test]
    fn render_named_args() {
        let s = skill_with_args("src, dst", "Copy $src to $dst.\n");
        assert_eq!(render(&s, "foo bar").unwrap(), "Copy foo to bar.\n");
    }

    #[test]
    fn render_positional_no_declared_args() {
        let s = skill("First: $0\n");
        assert_eq!(render(&s, "hello").unwrap(), "First: hello\n");
    }

    #[test]
    fn render_missing_arg_becomes_empty() {
        let s = skill_with_args("src, dst", "Copy $src to $dst.\n");
        // only one arg provided — dst becomes ""
        assert_eq!(render(&s, "foo").unwrap(), "Copy foo to .\n");
    }

    #[test]
    fn render_inline_shell_injection() {
        let s = skill("Date: !`echo hello`\n");
        assert_eq!(render(&s, "").unwrap(), "Date: hello\n");
    }

    #[test]
    fn render_block_shell_injection() {
        // The trailing \n after the closing ``` is kept as document whitespace
        let s = skill("```!\necho block\n```\n");
        assert_eq!(render(&s, "").unwrap(), "block\n\n");
    }

    #[test]
    fn render_shell_injection_failure_is_error() {
        let s = skill("!`exit 1`\n");
        assert!(render(&s, "").is_err());
    }

    #[test]
    fn render_no_shell_markers_unmodified() {
        let s = skill("No shell here.\n");
        assert_eq!(render(&s, "").unwrap(), "No shell here.\n");
    }
}
