use std::process::Command;

use super::Skill;

/// Render the skill body: substitute arguments, run shell injections.
///
/// `authorize_shell` is called at most once, lazily — only if the body
/// actually contains a shell-injection marker that would execute. Pass
/// `|| Ok(())` for tests or trusted contexts. Production callers should
/// route this through the permission manager so untrusted project-local
/// skills get prompted on first use.
pub fn render<F>(skill: &Skill, args: &str, authorize_shell: F) -> Result<String, String>
where
    F: FnOnce() -> Result<(), String>,
{
    let body = substitute_args(&skill.body, &skill.arguments, args);

    if has_real_injection(&body) {
        authorize_shell()?;
        process_injections(&body)
    } else {
        Ok(body)
    }
}

/// Substitute `$ARGUMENTS`, `$<name>`, and `$N` in `body`.
///
/// Named/positional substitution uses *word-boundary* matching so that
/// `$src` does not chew the prefix off `$srcdir` or `$src_path`.
fn substitute_args(body: &str, declared_args: &[String], args: &str) -> String {
    let arg_values: Vec<&str> = if args.is_empty() {
        Vec::new()
    } else if declared_args.is_empty() {
        vec![args]
    } else {
        args.splitn(declared_args.len(), ' ').collect()
    };

    let mut out = body.to_string();

    for (i, arg_name) in declared_args.iter().enumerate() {
        let val = arg_values.get(i).copied().unwrap_or("");
        out = replace_token(&out, arg_name, val);
        out = replace_token(&out, &i.to_string(), val);
    }
    for (i, val) in arg_values.iter().enumerate() {
        out = replace_token(&out, &i.to_string(), val);
    }
    out = replace_token(&out, "ARGUMENTS", args);
    out
}

/// Replace `$<token>` in `body` with `value`, only when the next character
/// is not part of an identifier (alphanumeric or `_`). So `$src` matches
/// in `Copy $src to ...` but not in `$srcdir` or `$src_path`.
fn replace_token(body: &str, token: &str, value: &str) -> String {
    let needle = format!("${token}");
    let mut out = String::with_capacity(body.len());
    let mut rest = body;
    while let Some(pos) = rest.find(&needle) {
        out.push_str(&rest[..pos]);
        let after = &rest[pos + needle.len()..];
        let is_ident_continuation =
            matches!(after.chars().next(), Some(c) if c.is_alphanumeric() || c == '_');
        if is_ident_continuation {
            out.push_str(&needle);
        } else {
            out.push_str(value);
        }
        rest = after;
    }
    out.push_str(rest);
    out
}

/// True if the body has any shell-injection marker that would actually
/// execute (i.e. not buried inside a fenced code block).
fn has_real_injection(content: &str) -> bool {
    let mut in_fence = false;
    let mut in_injection = false;
    for line in content.split_inclusive('\n') {
        let trimmed = line.trim_start();
        if in_injection {
            if is_fence_close(trimmed) {
                return true;
            }
            continue;
        }
        if in_fence {
            if is_fence_close(trimmed) {
                in_fence = false;
            }
            continue;
        }
        if is_injection_open(trimmed) {
            in_injection = true;
            continue;
        }
        if trimmed.starts_with("```") {
            in_fence = true;
            continue;
        }
        if line.contains("!`") {
            return true;
        }
    }
    false
}

/// Single-pass scanner that:
/// - Replaces ` ```!\n…\n``` ` blocks (outside any doc fence) with stdout.
/// - Replaces `` !`cmd` `` inline (outside any doc fence) with stdout.
/// - Leaves anything inside a regular ` ```…``` ` fence verbatim, so a
///   skill that *documents* the syntax does not execute it.
fn process_injections(content: &str) -> Result<String, String> {
    let mut out = String::new();
    let mut in_fence = false;
    let mut injection_buf: Option<String> = None;

    for line in content.split_inclusive('\n') {
        let trimmed = line.trim_start();

        if let Some(buf) = injection_buf.as_mut() {
            // Inside a `` ```!\n…\n``` `` block — collect until close.
            if is_fence_close(trimmed) {
                let stdout = run_shell(buf.trim_end_matches('\n'))?;
                out.push_str(&stdout);
                if let Some(after) = line.split_once("```").map(|(_, a)| a) {
                    out.push_str(after);
                }
                injection_buf = None;
            } else {
                buf.push_str(line);
            }
            continue;
        }

        if in_fence {
            out.push_str(line);
            if is_fence_close(trimmed) {
                in_fence = false;
            }
            continue;
        }

        if is_injection_open(trimmed) {
            injection_buf = Some(String::new());
            continue;
        }

        if trimmed.starts_with("```") {
            in_fence = true;
            out.push_str(line);
            continue;
        }

        out.push_str(&process_inline_in_line(line)?);
    }

    // Unterminated injection block — emit verbatim rather than crash.
    if let Some(buf) = injection_buf {
        out.push_str("```!\n");
        out.push_str(&buf);
    }

    Ok(out)
}

fn is_injection_open(trimmed_line: &str) -> bool {
    // Strict: line starts with ` ```! ` and the rest is whitespace/newline.
    // (e.g. ` ```!  \n ` or ` ```!\n `.) Anything else is a normal fence.
    if let Some(rest) = trimmed_line.strip_prefix("```!") {
        rest.trim().is_empty()
    } else {
        false
    }
}

fn is_fence_close(trimmed_line: &str) -> bool {
    if let Some(rest) = trimmed_line.strip_prefix("```") {
        // Strictly the close — no `!` info string.
        rest.trim().is_empty()
    } else {
        false
    }
}

/// Replace `` !`cmd` `` inline within a single line.
fn process_inline_in_line(line: &str) -> Result<String, String> {
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(start) = rest.find("!`") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        if let Some(end) = after.find('`') {
            let cmd_out = run_shell(&after[..end])?;
            out.push_str(cmd_out.trim_end_matches('\n'));
            rest = &after[end + 1..];
        } else {
            // No closing backtick on this line — emit verbatim.
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

    fn allow() -> impl FnOnce() -> Result<(), String> {
        || Ok(())
    }

    fn deny() -> impl FnOnce() -> Result<(), String> {
        || Err("denied".into())
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
        assert_eq!(render(&s, "", allow()).unwrap(), "Plain body.\n");
    }

    #[test]
    fn render_dollar_arguments() {
        let s = skill("Args: $ARGUMENTS\n");
        assert_eq!(
            render(&s, "hello world", allow()).unwrap(),
            "Args: hello world\n"
        );
    }

    #[test]
    fn render_dollar_arguments_empty() {
        let s = skill("Args: $ARGUMENTS\n");
        assert_eq!(render(&s, "", allow()).unwrap(), "Args: \n");
    }

    #[test]
    fn render_named_args() {
        let s = skill_with_args("src, dst", "Copy $src to $dst.\n");
        assert_eq!(
            render(&s, "foo bar", allow()).unwrap(),
            "Copy foo to bar.\n"
        );
    }

    #[test]
    fn render_positional_no_declared_args() {
        let s = skill("First: $0\n");
        assert_eq!(render(&s, "hello", allow()).unwrap(), "First: hello\n");
    }

    #[test]
    fn render_missing_arg_becomes_empty() {
        let s = skill_with_args("src, dst", "Copy $src to $dst.\n");
        assert_eq!(render(&s, "foo", allow()).unwrap(), "Copy foo to .\n");
    }

    #[test]
    fn render_inline_shell_injection() {
        let s = skill("Date: !`echo hello`\n");
        assert_eq!(render(&s, "", allow()).unwrap(), "Date: hello\n");
    }

    #[test]
    fn render_block_shell_injection() {
        let s = skill("```!\necho block\n```\n");
        assert_eq!(render(&s, "", allow()).unwrap(), "block\n\n");
    }

    #[test]
    fn render_shell_injection_failure_is_error() {
        let s = skill("!`exit 1`\n");
        assert!(render(&s, "", allow()).is_err());
    }

    #[test]
    fn render_no_shell_markers_unmodified() {
        let s = skill("No shell here.\n");
        assert_eq!(render(&s, "", allow()).unwrap(), "No shell here.\n");
    }

    #[test]
    fn authorizer_only_called_when_shell_present() {
        let s = skill("Plain body.\n");
        let mut called = false;
        let auth = || {
            called = true;
            Ok(())
        };
        render(&s, "", auth).unwrap();
        assert!(!called, "no shell markers — authorizer must not run");
    }

    #[test]
    fn authorizer_denial_blocks_execution() {
        let s = skill("Date: !`echo should-not-run`\n");
        let result = render(&s, "", deny());
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "denied");
    }

    #[test]
    fn named_arg_does_not_consume_longer_identifier() {
        let s = skill_with_args("src", "$src/$srcdir/$src_path\n");
        assert_eq!(
            render(&s, "FOO", allow()).unwrap(),
            "FOO/$srcdir/$src_path\n"
        );
    }

    #[test]
    fn named_arg_word_boundary_at_end_of_input() {
        let s = skill_with_args("src", "Copy $src");
        assert_eq!(render(&s, "FOO", allow()).unwrap(), "Copy FOO");
    }

    #[test]
    fn arguments_token_does_not_consume_longer_identifier() {
        let s = skill("$ARGUMENTS / $ARGUMENTSx\n");
        assert_eq!(render(&s, "FOO", allow()).unwrap(), "FOO / $ARGUMENTSx\n");
    }

    #[test]
    fn inline_injection_inside_doc_fence_is_not_executed() {
        let s = skill("Use !`echo hello` outside.\n```\nUse !`echo inside` here.\n```\n");
        let out = render(&s, "", allow()).unwrap();
        assert!(out.contains("Use hello outside."), "outside ran: {out}");
        assert!(
            out.contains("Use !`echo inside` here."),
            "inside fence preserved: {out}"
        );
    }

    #[test]
    fn fenced_code_only_skill_does_not_trigger_authorizer() {
        let s = skill("```\nUse !`echo doc` here.\n```\n");
        let mut called = false;
        let auth = || {
            called = true;
            Ok(())
        };
        let out = render(&s, "", auth).unwrap();
        assert!(!called, "no real injection — authorizer must not run");
        assert!(out.contains("!`echo doc`"));
    }

    #[test]
    fn block_injection_marker_inside_doc_fence_is_not_executed() {
        // The skill documents the `` ```! `` syntax inside a 4-backtick fence.
        // We don't support 4-backtick parsing, so use a regular fence with
        // `text` info string and an indented inner marker.
        let s = skill("```text\n```!\necho should-not-run\n```\n```\n");
        let out = render(&s, "", allow()).unwrap();
        assert!(
            out.contains("echo should-not-run"),
            "inside doc fence must be preserved verbatim: {out}"
        );
    }
}
