//! Pre-turn skill router: a dedicated no-tools classifier call that maps a
//! task onto exactly one installed skill (or none), then the caller rewrites
//! the task to an imperative "read the skill and follow it".
//!
//! Motivation (uds-mcp e2e, 2026-07-15, 6 runs across 2 sessions): the
//! [SKILLS] listing was delivered with a near-verbatim description match
//! and the model read ZERO SKILL.md files — advisory prose in the system
//! prompt is inert. A focused classifier + task rewrite flips first-action
//! skill adoption from 0/8 to 8/8 (tier1-skill-router-probe, stage 2), and
//! the classifier itself was 30/30 (correct picks on matching tasks, NONE
//! on non-matching) with zero salvage needed.
//!
//! Fail-safe by construction, mirroring the explore router: any parse
//! failure, LLM error, or exhausted retry → None → the turn proceeds with
//! the original task, i.e. today's behavior. The dangerous failure mode
//! (grafting an irrelevant skill onto a task) requires a confident wrong
//! pick twice in a row.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use crate::config::ModelRole;
use crate::llm::{ChatRequest, Message};
use crate::runtime::{LlmWorkerEvent, LlmWorkerHandle};

enum Pick {
    Skill(String),
    None,
    Invalid,
}

/// Classify `task` against the installed skills. Returns the matched skill
/// name, or None for "no skill / couldn't decide" (fail-safe).
pub async fn route_task_to_skill(
    llm_worker: &LlmWorkerHandle,
    project_root: &std::path::Path,
    task: &str,
    cancelled: &Arc<AtomicBool>,
) -> Option<String> {
    let entries = crate::skills::discover(project_root);
    if entries.is_empty() {
        return None;
    }
    let mut names: Vec<String> = Vec::new();
    let mut listing = String::new();
    for entry in entries {
        let Ok(skill) = crate::skills::load(&entry.path) else {
            continue;
        };
        let desc: String = skill
            .description
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        listing.push_str(&format!(
            "- {}: {}\n",
            skill.name,
            crate::truncate_chars(&desc, 800)
        ));
        names.push(skill.name);
    }
    if names.is_empty() {
        return None;
    }

    let sys = format!(
        "You route coding tasks to skills. Installed skills:\n{listing}\
         If exactly one skill clearly applies to the user's task, reply with that \
         skill's name and nothing else. If none clearly applies, reply NONE. \
         Reply with a single word only."
    );
    let mut messages = vec![Message::system(&sys), Message::user(task)];

    // One shot + one corrective retry, then fail-safe None.
    for _ in 0..2 {
        let out = ask(llm_worker, messages.clone(), cancelled).await;
        match parse_pick(&out, &names) {
            Pick::Skill(name) => return Some(name),
            Pick::None => return None,
            Pick::Invalid => {
                messages.push(Message::assistant(&out));
                messages.push(Message::user(&format!(
                    "Answer with exactly one of: {} or NONE.",
                    names.join(", ")
                )));
            }
        }
    }
    None
}

/// The imperative rewrite. Full FILE path (a directory read errors) and an
/// explicit "follow its instructions" — probe: 8/8 first-action skill reads
/// vs 0/8 for the plain task.
pub fn rewrite_task_for_skill(skill: &str, task: &str) -> String {
    format!(
        "Read .ai/skills/{skill}/SKILL.md and follow its instructions to handle \
         this request: {task}"
    )
}

async fn ask(
    llm_worker: &LlmWorkerHandle,
    messages: Vec<Message>,
    cancelled: &Arc<AtomicBool>,
) -> String {
    // Skill names are short; thinking disabled like the explore router.
    ask_with_budget(llm_worker, messages, cancelled, 24).await
}

async fn ask_with_budget(
    llm_worker: &LlmWorkerHandle,
    messages: Vec<Message>,
    cancelled: &Arc<AtomicBool>,
    max_tokens: u64,
) -> String {
    let request = ChatRequest {
        messages,
        tools: None,
        tool_choice: None,
        max_tokens_override: Some(max_tokens),
        chat_template_kwargs: Some(serde_json::json!({"enable_thinking": false})),
        cache_prompt: None,
    };
    let mut events = llm_worker.submit(ModelRole::Default, request, cancelled.clone());
    let mut out = String::new();
    while let Some(ev) = events.recv().await {
        match ev {
            LlmWorkerEvent::Completed(Ok(r)) => {
                out = r
                    .choices
                    .first()
                    .and_then(|c| c.message.content.clone())
                    .unwrap_or_default();
                break;
            }
            LlmWorkerEvent::Completed(Err(_)) => break, // fail-safe → empty
            _ => {}
        }
    }
    out
}

/// One actionable step extracted from a skill document, with a coarse
/// anchor (heading or short verbatim quote) locating it in the source so
/// the text can be re-read at execution time.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SkillStep {
    pub name: String,
    pub anchor: String,
}

/// Turn a skill document into an ordered, anchored execution checklist.
///
/// LLM-driven on purpose — deterministic parsing is impossible: of the
/// real UDS skills only 1 of 5 uses `### Step` headings, steps branch
/// (conditionals, "IFF unavailable" fallbacks, mid-prose handoffs), and
/// the actual content is spread across a doc tree that SKILL.md only
/// indexes. Probe 2026-07-15: 10/10 recovered all 17 build-skill steps in
/// order, every one anchored. Returns empty on any failure — the caller
/// degrades to plain read-and-follow.
pub async fn extract_skill_steps(
    llm_worker: &LlmWorkerHandle,
    skill_body: &str,
    cancelled: &Arc<AtomicBool>,
) -> Vec<SkillStep> {
    let sys = "You convert a skill document into an ordered execution checklist. \
        Output ONLY a JSON array of objects, each {\"step\":\"<ShortName>\",\"anchor\":\"<the \
        heading or a short verbatim quote locating this step in the document>\"}. One entry per \
        actionable step, in execution order. Do not invent steps; do not include preamble or \
        principles sections.";
    let messages = vec![Message::system(sys), Message::user(skill_body)];
    let out = ask_with_budget(llm_worker, messages, cancelled, 3000).await;
    parse_steps(&out)
}

/// Distill ONE skill step into self-contained, verbatim-faithful
/// instructions plus a completion criterion, from the skill material
/// (SKILL.md + its sub-files). This is the just-in-time step-cursor
/// mechanism — far more reliable than a file→anchor slice on the real
/// skills (prose steps, distributed sub-file content). Probe 2026-07-16:
/// 7/8 on the critical WriteZarfYaml step (kept the exact `.git@<ref>` tool
/// args live runs kept botching), 8/8 on a cross-file step. Empty on
/// failure (caller falls back to the raw step name).
pub async fn distill_step(
    llm_worker: &LlmWorkerHandle,
    material: &str,
    step_name: &str,
    cancelled: &Arc<AtomicBool>,
) -> String {
    let sys = "You are preparing focused, self-contained instructions for ONE step of a skill, \
        to hand to an executor who will do only that step and nothing else. You are given the \
        skill document and its referenced sub-files.\n\
        Output exactly two sections:\n\
        INSTRUCTIONS: the concrete actions for this step. COPY load-bearing specifics VERBATIM — \
        tool names, exact argument names and formats, commands, URLs, file paths. Do NOT \
        paraphrase them or leave them abstract. Inline anything from a referenced sub-file that \
        this step needs. Omit other steps.\n\
        DONE WHEN: a one-line, checkable completion criterion for this step.\n\
        Output only those two sections.";
    let user = format!("Skill material:\n\n{material}\n\nDistill the step: '{step_name}'.");
    let messages = vec![Message::system(sys), Message::user(&user)];
    ask_with_budget(llm_worker, messages, cancelled, 4000).await
}

/// Turn a step's `DONE WHEN` criterion into a cheap, READ-ONLY shell command
/// that exits 0 iff the step is complete — the per-step completion check.
/// While the step is active this becomes the effective validation command
/// (see `skill_cursor::current_check_command`), which lights up the
/// otherwise-dormant gate + debugger stack on projects with no configured
/// task-level check. Returns None when the step isn't shell-checkable (the
/// model answers NONE) or the command isn't provably read-only — the caller
/// then just relies on the model's self-attested `skill(done)`.
///
/// Probe 2026-07-16 (tier1-step-check-generator, real skills + fixtures):
/// 30/32 correct, ZERO false-passes, all read-only; the critical
/// ConfigureSSO case was 8/8 (every check targeted the Package CR template,
/// failing on the real `sso`-in-zarf.yaml mistake). The lone risk — an
/// over-specific check that false-FAILS (2/8 on one step, traceable to
/// DONE WHEN wording) — is bounded by the one-retry override on skill(done).
pub async fn generate_step_check(
    llm_worker: &LlmWorkerHandle,
    material: &str,
    step_name: &str,
    done_when: &str,
    cancelled: &Arc<AtomicBool>,
) -> Option<String> {
    let sys = "You write a completion CHECK for ONE step of a skill. Given the step's DONE WHEN \
        criterion, output a SINGLE shell command that exits 0 if and only if the step is \
        complete, and non-zero otherwise.\n\
        HARD RULES:\n\
        - READ-ONLY. Use only test / [ ] / grep / ls / find / cat / yq eval. NEVER modify, \
        create, delete, build, or deploy anything. No redirects to files, no sed -i, no \
        kubectl/helm/zarf/git/docker, no package create or deploy.\n\
        - Reference the EXACT file paths from the skill material.\n\
        - Cheap and objective: filesystem/text checks only, no cluster access.\n\
        - If the step's completion CANNOT be verified by such a command (it is investigative, \
        or produces no on-disk artifact), output exactly: NONE\n\
        Output ONLY the one-line command, or NONE. No explanation, no code fence.";
    let user = format!(
        "Skill material:\n\n{material}\n\nStep: '{step_name}'\nDONE WHEN: {done_when}\n\n\
         Write the CHECK command."
    );
    let messages = vec![Message::system(sys), Message::user(&user)];
    let out = ask_with_budget(llm_worker, messages, cancelled, 2000).await;
    parse_check(&out).filter(|c| is_read_only_check(c))
}

/// Out-of-band completion judge: asked mid-step whether the CURRENT step is
/// complete, given its definition and the model's recent activity. Drives
/// cursor advancement — the model almost never calls skill(done) on its own
/// (e2e 2026-07-16: 2 calls across 4 attempts), so a fixed safety valve was
/// dragging every step 20 rounds. Neutral, skeptical framing so it doesn't
/// rubber-stamp "done" for progress' sake.
///
/// Probe 2026-07-16 (tier1-ask-if-done, real states): 40/40 honest — 8/8
/// NOT DONE on scaffold stubs, 8/8 on a half-filled step, 8/8 DONE on
/// genuinely-complete, and it even flagged files written to the wrong
/// directory. Returns false (not done) on any ambiguity — advancement is the
/// side that must be earned.
pub async fn judge_step_done(
    llm_worker: &LlmWorkerHandle,
    step_name: &str,
    step_def: &str,
    recent_activity: &str,
    cancelled: &Arc<AtomicBool>,
) -> (bool, String) {
    let sys = "You are mid-task, executing ONE step of a UDS packaging skill. A routine status \
        check is running. Judge ONLY whether the CURRENT step is complete, based on the actual \
        recent activity shown — do not assume, do not be optimistic, do not credit work that \
        isn't evidenced. Reply with the first line EXACTLY 'DONE' or 'NOT DONE', then one line \
        stating the specific remaining work (or why it is complete).";
    let user = format!(
        "Current step: {step_name}\n{step_def}\n\nYour recent activity:\n{recent_activity}\n\n\
         Are you done with THIS step?"
    );
    let messages = vec![Message::system(sys), Message::user(&user)];
    let out = ask_with_budget(llm_worker, messages, cancelled, 4000).await;
    (parse_done_verdict(&out), extract_verdict_reason(&out))
}

/// The judge's one-line justification — the text after the DONE / NOT DONE
/// verdict. On a NOT DONE this is the concrete remaining work (e.g. "write
/// zarf.yaml to the package root"), which the caller surfaces to the model:
/// the judge's diagnosis was previously thrown away, and it's often the exact
/// steer the model needs (e2e 2026-07-16: the judge correctly flagged the
/// tmp_repo build, silently).
fn extract_verdict_reason(raw: &str) -> String {
    let t = raw.trim();
    let up = t.to_uppercase();
    let rest = if let Some(i) = up.find("NOT DONE") {
        &t[i + "NOT DONE".len()..]
    } else if let Some(i) = up.find("DONE") {
        &t[i + "DONE".len()..]
    } else {
        t
    };
    let rest = rest
        .trim_start_matches([':', '.', '-', ' ', '\n', '\t', '—'])
        .trim();
    crate::truncate_chars(rest, 240)
}

/// Parse a completion-judge reply to `true` (done) only on an unambiguous
/// DONE. "NOT DONE" and anything unclear → `false` (conservative — don't
/// advance on ambiguity). "NOT DONE" must be tested before "DONE" (substring).
fn parse_done_verdict(raw: &str) -> bool {
    let t = raw.to_uppercase();
    match (t.find("NOT DONE"), t.find("DONE")) {
        (Some(_), _) => false,
        (None, Some(_)) => true,
        _ => false,
    }
}

/// Extract the `DONE WHEN:` criterion from distilled step instructions.
pub fn extract_done_when(distilled: &str) -> Option<String> {
    let lower = distilled.to_lowercase();
    let pos = lower.find("done when")?;
    let after = &distilled[pos + "done when".len()..];
    let after = after.trim_start_matches([':', ' ', '\t']);
    let crit = after.trim();
    if crit.is_empty() {
        None
    } else {
        Some(crate::truncate_chars(crit, 600))
    }
}

/// Pull a single check command out of a possibly chatty/fenced response.
/// Returns None for an explicit NONE (or empty) — meaning "no check".
fn parse_check(raw: &str) -> Option<String> {
    let mut t = raw.trim().to_string();
    // Strip one ``` fence if present.
    if let Some(start) = t.find("```") {
        let rest = &t[start + 3..];
        // drop an optional language tag on the fence's first line
        let rest = rest
            .strip_prefix("bash")
            .or_else(|| rest.strip_prefix("sh"))
            .unwrap_or(rest);
        let rest = rest.trim_start_matches('\n');
        if let Some(end) = rest.find("```") {
            t = rest[..end].trim().to_string();
        }
    }
    let line = t
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("");
    let bare = line.trim_matches(|c: char| c == '`' || c == '"' || c == '\'' || c == '*');
    if bare.is_empty() || bare.trim_end_matches('.').eq_ignore_ascii_case("none") {
        return None;
    }
    Some(bare.to_string())
}

/// A check is read-only unless a *command-position* token is a mutator, or it
/// writes files (redirect / in-place edit). Command position = start of the
/// string or right after a shell operator; this is what stops `zarf.yaml` /
/// `uds-package.yaml` (mutators appearing only as PATH substrings) from being
/// mis-flagged — a valid check legitimately names those files. Mirrors the
/// probe's validated detector.
fn is_read_only_check(cmd: &str) -> bool {
    const MUTATORS: &[&str] = &[
        "rm", "mv", "cp", "dd", "truncate", "tee", "chmod", "chown", "mkdir", "touch", "curl",
        "wget", "kubectl", "helm", "zarf", "uds", "docker", "git", "ln", "install", "npm", "node",
        "python", "python3", "make", "apt", "sh", "bash",
    ];
    // In-place edits: sed/yq/jq with a -i flag before the next operator.
    for tool in ["sed", "yq", "jq"] {
        let mut hay = cmd;
        while let Some(i) = hay.find(tool) {
            let seg = &hay[i + tool.len()..];
            let seg = seg.split(['|', ';', '&']).next().unwrap_or(seg);
            if seg.contains(" -i") {
                return false;
            }
            hay = &hay[i + tool.len()..];
        }
    }
    // File-writing redirects (but tolerate 2>/dev/null, 2>&1, &>/dev/null).
    let bytes = cmd.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'>' {
            if i > 0 && bytes[i - 1] == b'>' {
                return false; // ">>"
            }
            if i + 1 < bytes.len() && bytes[i + 1] == b'>' {
                continue; // first char of ">>", handled next iteration
            }
            let after = cmd[i + 1..].trim_start();
            let after = after
                .strip_prefix('&')
                .map(str::trim_start)
                .unwrap_or(after);
            if !after.starts_with("/dev/null")
                && !after.chars().next().is_some_and(|c| c.is_ascii_digit())
            {
                return false;
            }
        }
    }
    for seg in cmd.split(['|', ';', '&', '\n']) {
        let seg = seg.trim().trim_start_matches(['!', '(']).trim();
        let Some(tok) = seg.split_whitespace().next() else {
            continue;
        };
        let tok = tok.trim_matches('(');
        if MUTATORS.contains(&tok) {
            return false;
        }
    }
    true
}

/// Extract the JSON array from a possibly chatty response and parse it into
/// steps, dropping malformed entries. Returns empty (not an error) so the
/// caller fails safe.
fn parse_steps(raw: &str) -> Vec<SkillStep> {
    let (Some(start), Some(end)) = (raw.find('['), raw.rfind(']')) else {
        return Vec::new();
    };
    if end <= start {
        return Vec::new();
    }
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&raw[start..=end]) else {
        return Vec::new();
    };
    let Some(arr) = value.as_array() else {
        return Vec::new();
    };
    arr.iter()
        .filter_map(|v| {
            let name = v.get("step")?.as_str()?.trim().to_string();
            let anchor = v.get("anchor")?.as_str()?.trim().to_string();
            if name.is_empty() {
                return None;
            }
            Some(SkillStep { name, anchor })
        })
        .collect()
}

/// Salvage-then-validate: strip decoration (backticks, quotes, trailing
/// period) before comparing case-insensitively; a unique containment match
/// also counts (small models sometimes wrap the name in a phrase).
fn parse_pick(raw: &str, names: &[String]) -> Pick {
    let t = raw
        .trim()
        .trim_matches(|c: char| c == '`' || c == '\'' || c == '"' || c == '*')
        .trim()
        .trim_end_matches('.')
        .trim();
    if t.is_empty() {
        return Pick::Invalid;
    }
    if t.eq_ignore_ascii_case("none") {
        return Pick::None;
    }
    for n in names {
        if t.eq_ignore_ascii_case(n) {
            return Pick::Skill(n.clone());
        }
    }
    let contained: Vec<&String> = names
        .iter()
        .filter(|n| t.to_lowercase().contains(&n.to_lowercase()))
        .collect();
    if let [single] = contained.as_slice() {
        return Pick::Skill((*single).clone());
    }
    Pick::Invalid
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names() -> Vec<String> {
        vec!["uds-package".into(), "uds-package-build".into()]
    }

    #[test]
    fn parse_steps_extracts_ordered_anchored_steps() {
        let raw = "Here you go:\n[\n \
            {\"step\":\"ChartUrl\",\"anchor\":\"### Step ChartUrl: Identify Helm Chart URL\"},\n \
            {\"step\":\"ScaffoldZarf\",\"anchor\":\"Call `scaffold-package` with targetDir\"}\n]";
        let steps = parse_steps(raw);
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].name, "ChartUrl");
        assert_eq!(steps[1].name, "ScaffoldZarf");
        assert!(steps[1].anchor.contains("scaffold-package"));
    }

    #[test]
    fn parse_steps_drops_malformed_entries_and_fails_safe() {
        // missing anchor / empty name entries are skipped, junk yields empty
        let mixed =
            "[{\"step\":\"A\",\"anchor\":\"x\"},{\"step\":\"\",\"anchor\":\"y\"},{\"step\":\"B\"}]";
        let steps = parse_steps(mixed);
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].name, "A");
        assert!(parse_steps("no json here").is_empty());
        assert!(parse_steps("").is_empty());
    }

    #[test]
    fn exact_and_decorated_names_parse() {
        assert!(
            matches!(parse_pick("uds-package-build", &names()), Pick::Skill(n) if n == "uds-package-build")
        );
        assert!(
            matches!(parse_pick("`uds-package-build`", &names()), Pick::Skill(n) if n == "uds-package-build")
        );
        assert!(
            matches!(parse_pick("UDS-Package-Build.", &names()), Pick::Skill(n) if n == "uds-package-build")
        );
    }

    #[test]
    fn none_variants_parse() {
        assert!(matches!(parse_pick("NONE", &names()), Pick::None));
        assert!(matches!(parse_pick(" none. ", &names()), Pick::None));
    }

    #[test]
    fn ambiguous_containment_is_invalid() {
        // "uds-package-build" contains "uds-package" too — a phrase matching
        // BOTH names must not resolve to either.
        assert!(matches!(
            parse_pick("use uds-package or uds-package-build", &names()),
            Pick::Invalid
        ));
        // But a phrase containing exactly one name salvages. Note
        // "uds-package-build" textually contains "uds-package", so unique
        // containment only works for the shorter name here when the longer
        // is absent.
        assert!(matches!(
            parse_pick("I would use the uds-package skill", &names()),
            Pick::Skill(n) if n == "uds-package"
        ));
    }

    #[test]
    fn junk_is_invalid() {
        assert!(matches!(parse_pick("", &names()), Pick::Invalid));
        assert!(matches!(parse_pick("who knows", &names()), Pick::Invalid));
    }

    #[test]
    fn parse_check_strips_fence_and_detects_none() {
        assert_eq!(
            parse_check("```bash\ntest -f zarf.yaml\n```").as_deref(),
            Some("test -f zarf.yaml")
        );
        assert_eq!(
            parse_check("grep -q 'sso:' chart/templates/uds-package.yaml").as_deref(),
            Some("grep -q 'sso:' chart/templates/uds-package.yaml")
        );
        assert_eq!(parse_check("NONE"), None);
        assert_eq!(parse_check(" none. "), None);
        assert_eq!(parse_check(""), None);
    }

    #[test]
    fn read_only_check_allows_paths_named_like_mutators() {
        // `zarf`/`uds` here are PATH substrings, not commands — must pass.
        assert!(is_read_only_check(
            "[ -f zarf.yaml ] && yq eval '.kind' zarf.yaml | grep -q ZarfPackageConfig"
        ));
        assert!(is_read_only_check(
            "grep -q 'sso:' chart/templates/uds-package.yaml"
        ));
        assert!(is_read_only_check(
            "cat zarf.yaml 2>/dev/null | grep -q kind"
        ));
        // real mutators in command position — must fail.
        assert!(!is_read_only_check("rm -rf zarf.yaml"));
        assert!(!is_read_only_check("zarf package create ."));
        assert!(!is_read_only_check("grep sso x > out.txt"));
        assert!(!is_read_only_check("sed -i s/a/b/ zarf.yaml"));
        assert!(!is_read_only_check("test -f x && kubectl apply -f y"));
    }

    #[test]
    fn done_verdict_parses_conservatively() {
        assert!(parse_done_verdict("DONE\nboth files are filled in"));
        assert!(parse_done_verdict("done — looks complete"));
        assert!(!parse_done_verdict(
            "NOT DONE\ncommon/zarf.yaml is still a stub"
        ));
        assert!(!parse_done_verdict("not done yet"));
        assert!(!parse_done_verdict("hmm, hard to say")); // ambiguous → not done
        // "NOT DONE" wins even if "DONE" also appears later
        assert!(!parse_done_verdict("NOT DONE. it is not yet DONE."));
    }

    #[test]
    fn verdict_reason_strips_the_verdict_token() {
        assert_eq!(
            extract_verdict_reason("NOT DONE\nwrite zarf.yaml to the package root"),
            "write zarf.yaml to the package root"
        );
        assert_eq!(
            extract_verdict_reason("NOT DONE: common/zarf.yaml is still a stub"),
            "common/zarf.yaml is still a stub"
        );
        assert_eq!(
            extract_verdict_reason("DONE — both files are filled in"),
            "both files are filled in"
        );
    }

    #[test]
    fn extract_done_when_pulls_criterion() {
        let d = "INSTRUCTIONS:\nDo the thing.\n\nDONE WHEN: zarf.yaml exists and is valid.";
        assert_eq!(
            extract_done_when(d).as_deref(),
            Some("zarf.yaml exists and is valid.")
        );
        assert_eq!(extract_done_when("INSTRUCTIONS: no criterion here"), None);
    }

    #[test]
    fn rewrite_shape() {
        let r = rewrite_task_for_skill("uds-package", "deploy the app");
        assert!(r.starts_with("Read .ai/skills/uds-package/SKILL.md"));
        assert!(r.contains("follow its instructions"));
        assert!(r.ends_with("deploy the app"));
    }
}
