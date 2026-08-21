//! Decoupled skill step-cursor.
//!
//! Replaces the plan-seeding approach (which fought the model's own
//! `plan(set)` and tangled the ordering — see the 2026-07-16 e2e
//! post-mortem). Instead of writing skill steps into `plan.md`, the harness
//! owns a private cursor: a STACK of frames `{skill, steps, idx}` (a stack
//! so an "invoke sub-skill" step DESCENDS into that skill and RETURNS when
//! it's exhausted). One step is active at a time; its instructions are
//! distilled just-in-time (LLM, see skill_router::distill_step) and
//! re-injected. The model works the step and signals `skill(done)` to
//! advance — with one step in flight there is no plan checklist to
//! rubber-stamp.
//!
//! State persists in `.miniswe/skill_cursor.json`. The model's `plan.md`
//! stays entirely its own — no coupling, no conflict.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::skill_router::SkillStep;
use crate::config::Config;

/// Total material fed to the distiller (SKILL.md + sub-files), capped so a
/// large skill tree can't blow the distiller's context.
const MAX_MATERIAL_CHARS: usize = 40_000;

#[derive(Serialize, Deserialize)]
struct Frame {
    skill: String,
    /// Directory of the skill's SKILL.md, for gathering sub-file material.
    dir: PathBuf,
    steps: Vec<SkillStep>,
    idx: usize,
}

#[derive(Serialize, Deserialize, Default)]
pub struct SkillCursor {
    /// Innermost frame is last. Empty = no skill in progress.
    stack: Vec<Frame>,
    /// Distilled instructions keyed by "<skill>#<idx>".
    distilled: HashMap<String, String>,
    /// Per-step completion check commands keyed by "<skill>#<idx>". Presence
    /// of a key means "generation attempted"; an empty value means "attempted,
    /// not shell-checkable" (so we don't re-generate every round).
    #[serde(default)]
    checks: HashMap<String, String>,
    /// Rounds spent on the current step (safety-valve counter). Reset
    /// whenever the current step changes (mark_done / descend).
    #[serde(default)]
    on_current: usize,
    /// Consecutive skill(done) calls on the current step. The completion
    /// check gates the FIRST call; a SECOND overrides it (see run.rs). Reset
    /// on every step change.
    #[serde(default)]
    done_attempts: usize,
}

fn path(config: &Config) -> PathBuf {
    config.miniswe_dir().join("skill_cursor.json")
}

pub fn load(config: &Config) -> SkillCursor {
    std::fs::read_to_string(path(config))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

pub fn save(config: &Config, cursor: &SkillCursor) {
    if let Ok(json) = serde_json::to_string(cursor) {
        let _ = std::fs::create_dir_all(config.miniswe_dir());
        let _ = std::fs::write(path(config), json);
    }
}

pub fn clear(config: &Config) {
    let _ = std::fs::remove_file(path(config));
}

impl SkillCursor {
    pub fn is_active(&self) -> bool {
        !self.stack.is_empty()
    }

    /// The skills currently on the stack (to avoid re-descending into one).
    fn on_stack(&self, name: &str) -> bool {
        self.stack.iter().any(|f| f.skill == name)
    }

    /// Push a skill's steps as the INITIAL frame (the routed root skill).
    /// No-op if the skill is already on the stack or has no steps.
    pub fn push_skill(&mut self, skill: &str, dir: &Path, steps: Vec<SkillStep>) {
        if steps.is_empty() || self.on_stack(skill) {
            return;
        }
        self.on_current = 0;
        self.done_attempts = 0;
        self.stack.push(Frame {
            skill: skill.to_string(),
            dir: dir.to_path_buf(),
            steps,
            idx: 0,
        });
    }

    /// Descend into a sub-skill at the current (invoking) step: consume the
    /// invoking step in the current frame FIRST, then push the sub-skill.
    /// Consuming up front means when the sub-skill exhausts, execution
    /// resumes at the step AFTER the invoke — the same handoff never fires
    /// twice. No-op if the sub-skill is already on the stack or has no steps.
    pub fn descend(&mut self, skill: &str, dir: &Path, steps: Vec<SkillStep>) {
        if steps.is_empty() || self.on_stack(skill) {
            return;
        }
        if let Some(frame) = self.stack.last_mut() {
            frame.idx += 1; // consume the invoking step
        }
        self.on_current = 0;
        self.done_attempts = 0;
        self.stack.push(Frame {
            skill: skill.to_string(),
            dir: dir.to_path_buf(),
            steps,
            idx: 0,
        });
    }

    /// The current (skill, step) — the top frame's active step.
    pub fn current(&self) -> Option<(&str, &SkillStep)> {
        let frame = self.stack.last()?;
        let step = frame.steps.get(frame.idx)?;
        Some((frame.skill.as_str(), step))
    }

    /// A `skill/step` tag for the active step, for disambiguating loop-detection
    /// keys (so the same call on different steps yields different keys). `None`
    /// when no step is active.
    pub fn step_tag(&self) -> Option<String> {
        let (skill, step) = self.current()?;
        Some(format!("{skill}/{}", step.name))
    }

    fn current_dir(&self) -> Option<&Path> {
        self.stack.last().map(|f| f.dir.as_path())
    }

    /// Complete the current step: advance the top frame, popping any frames
    /// whose steps are exhausted (ascend out of finished sub-skills).
    pub fn mark_done(&mut self) {
        if let Some(frame) = self.stack.last_mut() {
            frame.idx += 1;
        }
        while self.stack.last().is_some_and(|f| f.idx >= f.steps.len()) {
            self.stack.pop();
        }
        self.on_current = 0;
        self.done_attempts = 0;
    }

    /// Register a skill(done) call on the current step and return the running
    /// count (1 on the first call). The caller gates the first call on the
    /// step's completion check and lets the second override it.
    pub fn note_done_attempt(&mut self) -> usize {
        self.done_attempts += 1;
        self.done_attempts
    }

    /// Count one round against the current step and report the running
    /// total. The caller uses this as a safety valve: a step the model
    /// never signals `done` on would otherwise freeze the injected guidance
    /// forever, so past a cap the caller force-advances.
    pub fn note_round(&mut self) -> usize {
        self.on_current += 1;
        self.on_current
    }

    /// The installed skill named in `hay` (not already on the stack), longest
    /// match first so `x-build` beats its own substring `x`.
    fn match_installed(&self, hay: &str, installed: &[String]) -> Option<String> {
        let hay = hay.to_lowercase();
        let mut cands: Vec<&String> = installed
            .iter()
            .filter(|n| !self.on_stack(n) && hay.contains(&n.to_lowercase()))
            .collect();
        cands.sort_by_key(|n| std::cmp::Reverse(n.len()));
        cands.first().map(|s| (*s).clone())
    }

    /// If the current step's NAME delegates to an installed skill not already
    /// on the stack, return its name — the caller extracts it and descends.
    /// The step NAME is the cheap, unambiguous signal (umbrella steps are
    /// literally "Invoke <skill> skill"); checked before distillation.
    pub fn handoff_target(&self, installed: &[String]) -> Option<String> {
        let (_, step) = self.current()?;
        self.match_installed(&step.name, installed)
    }

    /// Handoff named only in the step's distilled BODY, not its name — e.g.
    /// the build skill's final step is named `Integrate` but its prose says
    /// "Continue with the uds-package-integrate skill". Restricted to the
    /// skill's LAST step: a phase handoff is structurally the final step,
    /// whereas a skill name mentioned mid-prose ("…that's the Integration
    /// Phase's job, invoke uds-package-integrate") is a passing reference, not
    /// a handoff — the last-step guard rejects that false-fire. Requires the
    /// body to already be distilled (`cached`).
    pub fn handoff_in_body(&self, installed: &[String]) -> Option<String> {
        let frame = self.stack.last()?;
        if frame.idx + 1 != frame.steps.len() {
            return None;
        }
        let body = self.cached()?;
        self.match_installed(body, installed)
    }

    fn cache_key(&self) -> Option<String> {
        let frame = self.stack.last()?;
        Some(format!("{}#{}", frame.skill, frame.idx))
    }

    /// Distilled instructions for the current step, if already computed.
    pub fn cached(&self) -> Option<&str> {
        let key = self.cache_key()?;
        self.distilled.get(&key).map(|s| s.as_str())
    }

    pub fn cache(&mut self, text: String) {
        if let Some(key) = self.cache_key() {
            self.distilled.insert(key, text);
        }
    }

    /// Whether check-generation has already been attempted for the current
    /// step (so we don't re-run it every round, even when it produced no
    /// check).
    pub fn check_attempted(&self) -> bool {
        self.cache_key()
            .is_some_and(|k| self.checks.contains_key(&k))
    }

    /// The current step's completion check command, if a non-empty one was
    /// generated.
    pub fn current_check(&self) -> Option<&str> {
        let key = self.cache_key()?;
        self.checks
            .get(&key)
            .map(|s| s.as_str())
            .filter(|s| !s.is_empty())
    }

    /// Record the current step's check (empty string = attempted, none).
    pub fn cache_check(&mut self, cmd: String) {
        if let Some(key) = self.cache_key() {
            self.checks.insert(key, cmd);
        }
    }

    /// Gather the distiller material for the current step: the skill's
    /// SKILL.md plus its sibling `.md` sub-files, capped.
    pub fn current_material(&self) -> Option<String> {
        gather_material(self.current_dir()?)
    }
}

/// SKILL.md + sibling `.md` files in the same directory, concatenated and
/// capped. The real skills spread instructions across a doc tree that
/// SKILL.md only indexes, so the distiller needs the siblings too.
pub fn gather_material(dir: &Path) -> Option<String> {
    let main = std::fs::read_to_string(dir.join("SKILL.md")).ok()?;
    let mut blob = format!("=== SKILL.md ===\n{main}\n");
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| e.file_name().to_str().map(str::to_string))
        .filter(|n| n.ends_with(".md") && n != "SKILL.md")
        .collect();
    names.sort();
    for name in names {
        if blob.len() >= MAX_MATERIAL_CHARS {
            break;
        }
        if let Ok(c) = std::fs::read_to_string(dir.join(&name)) {
            blob.push_str(&format!("\n=== {name} ===\n{c}\n"));
        }
    }
    if blob.len() > MAX_MATERIAL_CHARS {
        blob = crate::truncate_chars(&blob, MAX_MATERIAL_CHARS);
    }
    Some(blob)
}

/// The current active step's completion check command, if one was generated.
/// While a step is active this is the *effective validation command* — it
/// takes precedence over the configured task-level command, because it's the
/// concrete criterion for the work happening right now. Disk-backed so the
/// gate sites can consult it without threading cursor state through the loop.
pub fn current_check_command(config: &Config) -> Option<String> {
    load(config).current_check().map(str::to_string)
}

/// A `skill/step` tag identifying the active cursor step, for disambiguating
/// loop-detection keys. Folding this into the call key means the SAME tool
/// call on two DIFFERENT steps — notably `skill(action='done')`, which is
/// byte-identical on every step — produces distinct keys, so legitimately
/// advancing through consecutive steps is not misread as a repeated call.
/// A genuine within-step rut keeps the step constant, so it's still caught.
/// `None` when no skill is active. Disk-backed like the other gate accessors.
pub fn current_step_tag(config: &Config) -> Option<String> {
    load(config).step_tag()
}

/// The `[SKILL STEP]` block for the [CURRENT STATE] re-injection: the
/// current step's distilled instructions (if computed), else a minimal
/// placeholder until the loop distills it. `None` when no skill is active.
pub fn active_step_block(config: &Config) -> Option<String> {
    let cursor = load(config);
    let (skill, step) = cursor.current()?;
    let body = match cursor.cached() {
        Some(text) => text.to_string(),
        None => format!("(preparing instructions for step '{}'…)", step.name),
    };
    Some(format!(
        "[SKILL STEP] From the {skill} skill — do THIS step now, following it exactly. \
         When its DONE WHEN criterion is met, call skill(action='done') to get the next step; \
         do not skip ahead or improvise.\n{body}\n"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn steps(names: &[&str]) -> Vec<SkillStep> {
        names
            .iter()
            .map(|n| SkillStep {
                name: (*n).to_string(),
                anchor: String::new(),
            })
            .collect()
    }

    fn cur(names: &[&str]) -> SkillCursor {
        let mut c = SkillCursor::default();
        c.push_skill("root", Path::new("/skills/root"), steps(names));
        c
    }

    #[test]
    fn advances_in_order_then_deactivates() {
        let mut c = cur(&["A", "B"]);
        assert_eq!(c.current().unwrap().1.name, "A");
        c.mark_done();
        assert_eq!(c.current().unwrap().1.name, "B");
        c.mark_done();
        assert!(c.current().is_none());
        assert!(!c.is_active());
    }

    #[test]
    fn step_tag_is_distinct_per_step_so_done_keys_dont_collapse() {
        // The loop-key fix rests on this: an identical skill(done) call on
        // consecutive steps must yield distinct tags, so 4 legit advances
        // aren't misread as a 4x repeat. A finished cursor tags nothing.
        let mut c = cur(&["A", "B"]);
        assert_eq!(c.step_tag().as_deref(), Some("root/A"));
        c.mark_done();
        assert_eq!(c.step_tag().as_deref(), Some("root/B"));
        c.mark_done();
        assert_eq!(c.step_tag(), None);
    }

    #[test]
    fn empty_or_duplicate_push_is_noop() {
        let mut c = cur(&["A"]);
        c.push_skill("root", Path::new("/x"), steps(&["Z"])); // dup name
        c.push_skill("other", Path::new("/x"), steps(&[])); // empty
        assert_eq!(c.stack.len(), 1);
    }

    #[test]
    fn descends_into_subskill_and_returns() {
        // root: [A, Invoke child skill, C]; at the invoke step the harness
        // descends into child: [X, Y]; when child exhausts, execution
        // resumes at C — the invoke step was consumed by the descent, so
        // the same handoff never fires again.
        let mut c = cur(&["A", "Invoke child skill", "C"]);
        c.mark_done(); // A done -> active "Invoke child skill"
        assert_eq!(
            c.handoff_target(&["child".into(), "unrelated".into()])
                .as_deref(),
            Some("child")
        );
        c.descend("child", Path::new("/skills/child"), steps(&["X", "Y"]));
        assert_eq!(c.current().unwrap(), ("child", &steps(&["X"])[0]));
        // the consumed invoke step must not re-trigger a handoff
        assert_eq!(c.handoff_target(&["child".into()]), None);
        c.mark_done(); // X
        assert_eq!(c.current().unwrap().1.name, "Y");
        c.mark_done(); // Y -> child exhausted -> pop -> resume at C directly
        assert_eq!(c.current().unwrap(), ("root", &steps(&["C"])[0]));
        // still no re-descend at C
        assert_eq!(c.handoff_target(&["child".into()]), None);
    }

    #[test]
    fn descend_when_invoke_is_last_step_deactivates_after_subskill() {
        // root: [A, Invoke child skill]; descending consumes the trailing
        // invoke step, so when child exhausts the whole cursor deactivates.
        let mut c = cur(&["A", "Invoke child skill"]);
        c.mark_done(); // -> invoke step
        c.descend("child", Path::new("/skills/child"), steps(&["X"]));
        assert_eq!(c.current().unwrap().1.name, "X");
        c.mark_done(); // X done -> child pops -> root already past its end
        assert!(!c.is_active());
    }

    #[test]
    fn note_round_counts_and_resets_on_step_change() {
        let mut c = cur(&["A", "B"]);
        assert_eq!(c.note_round(), 1);
        assert_eq!(c.note_round(), 2);
        c.mark_done(); // step change resets the counter
        assert_eq!(c.note_round(), 1);
    }

    #[test]
    fn body_handoff_fires_only_on_last_step_and_needs_distillation() {
        let installed = vec!["uds-package-integrate".to_string()];
        // Build-like skill: final step named "Integrate" whose body names the
        // integrate skill; a mid step also mentions it in passing.
        let mut c = cur(&["ScaffoldZarf", "PinImages", "Integrate"]);
        // Mid step (PinImages) mentions the skill in prose — must NOT fire.
        c.mark_done(); // -> PinImages (idx 1, not last)
        c.cache(
            "…full SSO wiring is the Integration Phase's job, invoke uds-package-integrate.".into(),
        );
        assert_eq!(
            c.handoff_in_body(&installed),
            None,
            "mid-step mention must not descend"
        );
        // Final step (Integrate) — but not distilled yet → None.
        c.mark_done(); // -> Integrate (idx 2, last)
        assert_eq!(c.handoff_in_body(&installed), None, "no body yet");
        // Distill the final step's handoff prose → now it fires.
        c.cache("Continue with the uds-package-integrate skill.".into());
        assert_eq!(
            c.handoff_in_body(&installed).as_deref(),
            Some("uds-package-integrate")
        );
    }

    #[test]
    fn handoff_prefers_longest_installed_name() {
        let mut c = cur(&["Invoke uds-package-build skill"]);
        // both substrings present; the longer, more specific name wins,
        // and an already-loaded skill is excluded.
        let got = c.handoff_target(&["uds-package".into(), "uds-package-build".into()]);
        assert_eq!(got.as_deref(), Some("uds-package-build"));
        c.push_skill("uds-package-build", Path::new("/x"), steps(&["S"]));
        // now loaded -> no further handoff to it from that name
        assert_eq!(c.handoff_target(&["uds-package-build".into()]), None);
    }

    #[test]
    fn cache_is_keyed_per_step() {
        let mut c = cur(&["A", "B"]);
        c.cache("instructions for A".into());
        assert_eq!(c.cached(), Some("instructions for A"));
        c.mark_done();
        assert_eq!(c.cached(), None); // B not distilled yet
        c.cache("instructions for B".into());
        assert_eq!(c.cached(), Some("instructions for B"));
    }

    #[test]
    fn check_cache_and_attempt_tracking_per_step() {
        let mut c = cur(&["A", "B"]);
        assert!(!c.check_attempted());
        assert_eq!(c.current_check(), None);
        c.cache_check("test -f a.txt".into());
        assert!(c.check_attempted());
        assert_eq!(c.current_check(), Some("test -f a.txt"));
        // empty check = attempted-but-not-checkable
        c.mark_done();
        c.cache_check(String::new());
        assert!(c.check_attempted());
        assert_eq!(c.current_check(), None);
    }

    #[test]
    fn done_attempts_increment_and_reset_on_advance() {
        let mut c = cur(&["A", "B"]);
        assert_eq!(c.note_done_attempt(), 1);
        assert_eq!(c.note_done_attempt(), 2); // override threshold
        c.mark_done();
        assert_eq!(c.note_done_attempt(), 1); // fresh for step B
    }

    #[test]
    fn gather_material_includes_siblings_capped() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        std::fs::write(dir.join("SKILL.md"), "main body").unwrap();
        std::fs::write(dir.join("scaffold.md"), "scaffold detail").unwrap();
        std::fs::write(dir.join("notes.txt"), "ignored non-md").unwrap();
        let m = gather_material(dir).unwrap();
        assert!(m.contains("main body"));
        assert!(m.contains("scaffold detail"));
        assert!(m.contains("=== scaffold.md ==="));
        assert!(!m.contains("ignored non-md"));
    }

    #[test]
    fn active_block_uses_cache_else_placeholder() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join(".miniswe")).unwrap();
        let mut config = Config::default();
        config.project_root = tmp.path().to_path_buf();

        let mut c = cur(&["Scaffold"]);
        save(&config, &c);
        let block = active_step_block(&config).unwrap();
        assert!(block.contains("[SKILL STEP]"));
        assert!(block.contains("preparing instructions for step 'Scaffold'"));
        assert!(block.contains("skill(action='done')"));

        c.cache("call scaffold-package with targetDir".into());
        save(&config, &c);
        let block = active_step_block(&config).unwrap();
        assert!(block.contains("call scaffold-package"));
        assert!(!block.contains("preparing instructions"));
    }
}
