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
    /// Steps this frame moved past WITHOUT a completion verdict — the
    /// safety valves in run.rs (round cap, repeated stops, unresolvable
    /// handoff). Kept distinct from done: while any remain, the frame has
    /// not finished its skill.
    #[serde(default)]
    abandoned: Vec<usize>,
    /// Whether this frame has already rewound once to pick its abandoned
    /// steps back up. One revisit pass, so termination stays a local
    /// argument rather than resting on the model's behaviour.
    #[serde(default)]
    rewound: bool,
}

impl Frame {
    /// Where this frame goes after leaving `idx`. On the forward pass that
    /// is simply the next step; on a revisit pass only the steps still
    /// marked abandoned are worth re-presenting, so it skips to the next of
    /// those (the completed ones in between are done and stay done).
    fn step_after(&self, idx: usize) -> usize {
        if self.rewound {
            self.abandoned
                .iter()
                .copied()
                .filter(|&i| i > idx)
                .min()
                .unwrap_or(self.steps.len())
        } else {
            idx + 1
        }
    }

    fn first_abandoned(&self) -> Option<usize> {
        self.abandoned.iter().copied().min()
    }

    /// `skill/step` labels for everything this frame left unfinished.
    fn unfinished_labels(&self) -> Vec<String> {
        let mut at: Vec<usize> = self.abandoned.clone();
        at.sort_unstable();
        at.iter()
            .filter_map(|&i| self.steps.get(i))
            .map(|s| format!("{}/{}", self.skill, s.name))
            .collect()
    }
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
    /// Per-step handoff verdicts keyed by "<skill>#<idx>". Presence of a key
    /// means "decided"; an empty value means "decided: no handoff". Cached
    /// because the decision is a model call and must not flap round to round —
    /// and because `descend()` is irreversible, so a verdict that changed its
    /// mind after the fact would have nothing to undo.
    #[serde(default)]
    handoffs: HashMap<String, String>,
    /// Rounds spent on the current step (safety-valve counter). Reset
    /// whenever the current step changes (mark_done / descend).
    #[serde(default)]
    on_current: usize,
    /// Consecutive skill(done) calls on the current step. The completion
    /// check gates the FIRST call; a SECOND overrides it (see run.rs). Reset
    /// on every step change.
    #[serde(default)]
    done_attempts: usize,
    /// Consecutive failed handoff descends on the current invoke step.
    /// Step extraction runs an LLM, which can fail transiently — the caller
    /// retries next round and only consumes the invoke step (skipping the
    /// whole sub-skill) once this passes a threshold. Reset on step change.
    #[serde(default)]
    handoff_failures: usize,
    /// Set by the last advance when an exhausted frame rewound onto work it
    /// had abandoned. Diagnostic for the caller's status line only, so it is
    /// deliberately not persisted.
    #[serde(skip)]
    rewound_into: Option<String>,
    /// Set by the last advance for every frame it popped that still held
    /// abandoned steps — the honest "this skill ended incomplete" report.
    /// Diagnostic only, not persisted.
    #[serde(skip)]
    dropped_unfinished: Vec<String>,
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
        self.handoff_failures = 0;
        self.stack.push(Frame {
            skill: skill.to_string(),
            dir: dir.to_path_buf(),
            steps,
            idx: 0,
            abandoned: Vec::new(),
            rewound: false,
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
            // Consume the invoking step: descending IS doing its work, so any
            // abandoned mark on it is discharged.
            let at = frame.idx;
            frame.abandoned.retain(|&i| i != at);
            frame.idx = frame.step_after(at);
        }
        self.on_current = 0;
        self.done_attempts = 0;
        self.handoff_failures = 0;
        self.stack.push(Frame {
            skill: skill.to_string(),
            dir: dir.to_path_buf(),
            steps,
            idx: 0,
            abandoned: Vec::new(),
            rewound: false,
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

    /// Complete the current step: a real completion verdict — the model's own
    /// `skill(done)`, or the judge with the step's check passing. Discharges
    /// any abandoned mark, since this is the step actually getting finished.
    pub fn mark_done(&mut self) {
        if let Some(frame) = self.stack.last_mut() {
            let at = frame.idx;
            frame.abandoned.retain(|&i| i != at);
        }
        self.advance();
    }

    /// Give up on the current step WITHOUT a completion verdict — what
    /// run.rs does when the stuck-step judge votes ABANDON (a frozen step
    /// whose instructions this environment cannot satisfy), the model keeps
    /// stopping on a step, or its handoff will not resolve.
    ///
    /// Deliberately not `mark_done`. A timer knows a step was expensive, never
    /// that it was finished, and every downstream consumer trusts done: the
    /// DONE WHEN check stops being re-run, the judge moves on, the frame pops
    /// and reports the skill complete, and `--continue` re-routes from scratch
    /// because the cursor went inactive. The 2026-08-28 uds-mcp e2e is the
    /// full price of that conflation in one run: `ImplementConfigChart` was
    /// retired at the round cap mid-write and shipped a chart entry with
    /// `path:` where Zarf needs `localPath:`; then the umbrella step guarding
    /// the deploy was retired on repeated stops, the stack popped reporting
    /// "skill complete", and both retries re-routed into a skill with no
    /// deploy step at all — three attempts, none of which could have fixed the
    /// one-word bug the first one introduced.
    pub fn mark_abandoned(&mut self) {
        if let Some(frame) = self.stack.last_mut()
            && frame.idx < frame.steps.len()
            && !frame.abandoned.contains(&frame.idx)
        {
            frame.abandoned.push(frame.idx);
        }
        self.advance();
    }

    /// Move off the current step and settle the stack. An exhausted frame
    /// that still holds abandoned steps rewinds onto the first of them rather
    /// than popping — popping is what claims the skill is complete, and it
    /// plainly is not. The revisit happens at the end of the phase, which is
    /// when the run knows the most; capped at one pass per frame.
    fn advance(&mut self) {
        self.rewound_into = None;
        self.dropped_unfinished.clear();
        if let Some(frame) = self.stack.last_mut() {
            let at = frame.idx;
            frame.idx = frame.step_after(at);
        }
        while let Some(frame) = self.stack.last() {
            if frame.idx < frame.steps.len() {
                break;
            }
            match frame.first_abandoned() {
                Some(at) if !frame.rewound => {
                    let skill = frame.skill.clone();
                    let frame = self.stack.last_mut().expect("observed above");
                    frame.rewound = true;
                    frame.idx = at;
                    self.rewound_into = Some(skill);
                    break;
                }
                _ => {
                    let frame = self.stack.pop().expect("observed above");
                    self.dropped_unfinished.extend(frame.unfinished_labels());
                }
            }
        }
        self.on_current = 0;
        self.done_attempts = 0;
        self.handoff_failures = 0;
    }

    /// Whether a safety valve may consume the current step. False on a
    /// frame's final step: advancing there pops the frame, which ends the
    /// phase — and ending a phase is a completion claim, which belongs to the
    /// judge or the model, never to a round counter. Grinding on a terminal
    /// step costs rounds; advancing off it ends the skill with the work
    /// undone and nothing left to notice.
    pub fn may_auto_advance(&self) -> bool {
        self.stack.last().is_some_and(|f| f.idx + 1 < f.steps.len())
    }

    /// The skill whose frame just rewound onto work it had abandoned, if the
    /// last advance did that.
    pub fn rewound_into(&self) -> Option<&str> {
        self.rewound_into.as_deref()
    }

    /// `skill/step` labels the last advance dropped unfinished — a frame
    /// popped with abandoned steps it never got back to. Empty in the normal
    /// case, which is the point: a non-empty list means the skill ended
    /// incomplete and the run must not report otherwise.
    pub fn dropped_unfinished(&self) -> &[String] {
        &self.dropped_unfinished
    }

    /// Register a skill(done) call on the current step and return the running
    /// count (1 on the first call). The caller gates the first call on the
    /// step's completion check and lets the second override it.
    pub fn note_done_attempt(&mut self) -> usize {
        self.done_attempts += 1;
        self.done_attempts
    }

    /// Count one round against the current step and report the running
    /// total. No cap consumes the step (a fixed budget is a refuted
    /// rounds-only trigger — see run.rs); the count paces the completion
    /// judge and feeds the stuck-step judge's prompt.
    pub fn note_round(&mut self) -> usize {
        self.on_current += 1;
        self.on_current
    }

    /// Rounds spent on the current step (since entry or the last reset).
    pub fn rounds_on_current(&self) -> usize {
        self.on_current
    }

    /// RETRY verdict (stuck-step judge): the step stays current behind a
    /// forced compaction — zero its counters so the fresh attempt is
    /// measured from the retry.
    pub fn reset_current_rounds(&mut self) {
        self.on_current = 0;
        self.done_attempts = 0;
    }

    /// Register a failed handoff descend on the current invoke step and
    /// return the running count (1 on the first failure). The caller retries
    /// on later rounds and only consumes the invoke step once the count
    /// passes its threshold — a single transient LLM extraction failure must
    /// not silently skip an entire sub-skill.
    pub fn note_handoff_failure(&mut self) -> usize {
        self.handoff_failures += 1;
        self.handoff_failures
    }

    /// Every installed skill named in `hay` (not already on the stack), as
    /// (offset, name). Matches only at token boundaries: a mention inside a
    /// longer identifier or a file path (`chart/templates/uds-package.yaml`)
    /// is not a handoff.
    fn matches_installed<'a>(
        &self,
        hay: &str,
        installed: &'a [String],
    ) -> Vec<(usize, &'a String)> {
        let hay = hay.to_lowercase();
        installed
            .iter()
            .filter(|n| !self.on_stack(n))
            .filter_map(|n| find_token(&hay, &n.to_lowercase()).map(|at| (at, n)))
            .collect()
    }

    /// The installed skill named in `hay`, longest match first so `x-build`
    /// beats its own substring `x`. For short haystacks (a step NAME) where
    /// there is one intended target and specificity is the only signal.
    fn match_installed(&self, hay: &str, installed: &[String]) -> Option<String> {
        let mut cands = self.matches_installed(hay, installed);
        cands.sort_by_key(|(_, n)| std::cmp::Reverse(n.len()));
        cands.first().map(|(_, n)| (*n).clone())
    }

    /// Installed skills named at token boundaries in `prose` and not already
    /// on the stack, ordered by first mention — the CANDIDATE set for the
    /// handoff classifier, never a verdict on its own.
    ///
    /// Retrieval is the part this layer is genuinely good at: it will not
    /// invent a skill, it rejects a name inside a path
    /// (`chart/templates/uds-package.yaml`), and it refuses an ancestor
    /// already on the stack. What it cannot do is tell a transfer of control
    /// from a locative cross-reference — both are "an installed skill is
    /// named here" — so that call belongs to
    /// `skill_router::classify_handoff`, with this list as its menu.
    ///
    /// Ordering is first-mention because a step that names two skills means
    /// the first: the real `uds-package-build` final step reads "Continue with
    /// the uds-package-integrate skill. … the Integration Phase invokes
    /// `validate-package` through the `uds-package-validate` skill."
    pub fn handoff_candidates(&self, prose: &str, installed: &[String]) -> Vec<String> {
        let mut cands = self.matches_installed(prose, installed);
        cands.sort_by_key(|(at, n)| (*at, std::cmp::Reverse(n.len())));
        cands.into_iter().map(|(_, n)| n.clone()).collect()
    }

    /// The source material from the LAST step's anchor to the end of that
    /// step's own document — the step's verbatim prose, without the rest of
    /// the skill tree. `None` when the step carries no anchor, the material
    /// is unreadable, or the anchor cannot be located.
    fn last_step_tail(&self) -> Option<String> {
        let frame = self.stack.last()?;
        let anchor = frame.steps.last()?.anchor.trim();
        if anchor.is_empty() {
            return None;
        }
        let material = gather_material(self.current_dir()?)?;
        let at = locate_anchor(&material, anchor)?;
        let tail = &material[at..];
        // Stop at the next sibling-file banner: a handoff is named in the
        // step's own prose, and reference docs further down the tree name
        // sibling skills constantly.
        let end = tail.find("\n=== ").unwrap_or(tail.len());
        Some(tail[..end].to_string())
    }

    /// If the current step's NAME delegates to an installed skill not already
    /// on the stack, return its name — the caller extracts it and descends.
    /// The step NAME is the cheap, unambiguous signal (umbrella steps are
    /// literally "Invoke <skill> skill"); checked before distillation.
    pub fn handoff_target(&self, installed: &[String]) -> Option<String> {
        let (_, step) = self.current()?;
        self.match_installed(&step.name, installed)
    }

    /// The prose the handoff decision reads for the current step: its
    /// verbatim source section, falling back to the distilled body. `None`
    /// when there is nothing to judge yet — the caller then leaves the verdict
    /// OPEN rather than recording a NONE it never earned.
    ///
    /// Restricted to the frame's LAST step. A phase handoff is structurally
    /// the final step, whereas a skill named mid-prose ("…that's the
    /// Integration Phase's job, invoke uds-package-integrate") is a passing
    /// reference; the guard rejects that shape for free, before any model
    /// call.
    ///
    /// Reads the SOURCE at the step's anchor first and only falls back to the
    /// distilled body. Making the build → integrate lifecycle contingent on a
    /// distillation call having landed is how the live e2e lost it: the judge
    /// advanced onto this step after that round's distillation had already
    /// run, the model called `done` on the resulting empty step, and the
    /// entire integration phase was skipped.
    pub fn handoff_prose(&self) -> Option<String> {
        let frame = self.stack.last()?;
        if frame.idx + 1 != frame.steps.len() {
            return None;
        }
        self.last_step_tail()
            .or_else(|| self.cached().map(str::to_string))
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

    /// Whether the handoff question has already been settled for the current
    /// step (so the classifier runs at most once per step, not once a round).
    pub fn handoff_decided(&self) -> bool {
        self.cache_key()
            .is_some_and(|k| self.handoffs.contains_key(&k))
    }

    /// The current step's handoff target, if one was decided.
    pub fn cached_handoff(&self) -> Option<&str> {
        let key = self.cache_key()?;
        self.handoffs
            .get(&key)
            .map(|s| s.as_str())
            .filter(|s| !s.is_empty())
    }

    /// Record the current step's handoff verdict (empty string = decided, no
    /// handoff).
    pub fn cache_handoff(&mut self, target: String) {
        if let Some(key) = self.cache_key() {
            self.handoffs.insert(key, target);
        }
    }

    /// Gather the distiller material for the current step: the skill's
    /// SKILL.md plus its sibling `.md` sub-files, capped.
    pub fn current_material(&self) -> Option<String> {
        gather_material(self.current_dir()?)
    }
}

/// Byte offset of the first occurrence of `needle` in `hay` as a whole token:
/// not butted against a name character (alnum/`-`/`_`) on either side, and not
/// followed by a file extension (`.yaml` in `uds-package.yaml`) — a
/// sentence-ending period is fine because no alphanumeric follows it.
fn find_token(hay: &str, needle: &str) -> Option<usize> {
    let is_name_char = |c: char| c.is_alphanumeric() || c == '-' || c == '_';
    let mut search_from = 0;
    while let Some(rel) = hay[search_from..].find(needle) {
        let start = search_from + rel;
        let end = start + needle.len();
        let before_ok = !hay[..start].chars().next_back().is_some_and(is_name_char);
        let mut after = hay[end..].chars();
        let after_ok = match after.next() {
            Some(c) if is_name_char(c) => false,
            Some('.') => !after.next().is_some_and(|c| c.is_alphanumeric()),
            _ => true,
        };
        if before_ok && after_ok {
            return Some(start);
        }
        // Step one char, not the whole needle: the next viable occurrence
        // may overlap a rejected one.
        search_from = start + hay[start..].chars().next().map_or(1, char::len_utf8);
    }
    None
}

/// True if `needle` occurs in `hay` at token boundaries.
pub(crate) fn contains_token(hay: &str, needle: &str) -> bool {
    find_token(hay, needle).is_some()
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

/// Byte offset of a step's `anchor` inside gathered material. The extractor
/// quotes a heading or a short verbatim line, so an exact hit is the common
/// case; a lightly-reflowed or multi-line anchor falls back to its first line.
/// The 8-char floor keeps a stubby fallback ("## Step") from matching the
/// wrong section.
fn locate_anchor(material: &str, anchor: &str) -> Option<usize> {
    if let Some(at) = material.find(anchor) {
        return Some(at);
    }
    let first = anchor.lines().next()?.trim();
    if first.len() < 8 {
        return None;
    }
    material.find(first)
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

/// Marker opening the `[SKILL STEP]` section of the [CURRENT STATE] block.
/// Shared with `context::compressor`, which keys the current-state block's
/// re-anchoring policy on whether a skill step is present.
pub const SKILL_STEP_MARKER: &str = "[SKILL STEP]";

/// The `[SKILL STEP]` block for the [CURRENT STATE] re-injection: the
/// current step's distilled instructions. `None` when no skill is active OR
/// the step is not distilled yet.
pub fn active_step_block(config: &Config) -> Option<String> {
    let cursor = load(config);
    let (skill, _) = cursor.current()?;
    // Nothing at all until the step is distilled. A placeholder body renders
    // as a step with no work and no completion criterion, under a standing
    // instruction to call skill(action='done') once the criterion is met —
    // and a model complies: the live uds-mcp e2e marked the build skill's
    // final handoff step done off exactly that placeholder, popping the frame
    // before the handoff could fire and skipping the whole integration phase.
    // Showing no step is strictly safer than showing an empty one.
    let body = cursor.cached()?;
    Some(format!(
        "{SKILL_STEP_MARKER} From the {skill} skill — do THIS step now, following it exactly. \
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
    fn a_frames_last_step_is_never_auto_advanced() {
        // Fix 1. Leaving a frame's final step pops the frame, and a popped
        // frame reports its skill complete — so a round counter must not be
        // able to do it. Live cost: the 2026-08-28 e2e retired
        // `DeployTestEnvironment`, the integrate skill's terminal step, with
        // the deploy never attempted.
        let mut c = cur(&["A", "B"]);
        assert!(
            c.may_auto_advance(),
            "A has a successor; the valve may fire"
        );
        c.mark_done();
        assert!(
            !c.may_auto_advance(),
            "B is terminal — only the judge or the model may end the phase"
        );
        // A sub-skill's terminal step is equally protected: leaving it pops
        // that frame and ends the phase, even though the stack survives.
        let mut c = cur(&["A", "Invoke child skill", "C"]);
        c.mark_done();
        c.descend("child", Path::new("/skills/child"), steps(&["X", "Y"]));
        assert!(c.may_auto_advance());
        c.mark_done(); // X
        assert_eq!(c.current().unwrap().1.name, "Y");
        assert!(!c.may_auto_advance(), "Y is the child's terminal step");
    }

    #[test]
    fn abandoning_is_not_completing_so_the_frame_cannot_report_complete() {
        // Fix 2, the core of it. `mark_done` on an abandoned step is what let
        // the e2e pop the whole stack and clear the cursor while the deploy
        // had never run.
        let mut c = cur(&["A", "B"]);
        c.mark_abandoned(); // A retired by a safety valve, NOT finished
        assert_eq!(c.current().unwrap().1.name, "B");
        c.mark_done(); // B genuinely finished -> frame exhausted
        // ...and instead of popping, the frame returns to the step it skipped.
        assert!(
            c.is_active(),
            "a skill with unfinished work is not complete"
        );
        assert_eq!(c.current().unwrap().1.name, "A");
        assert_eq!(c.rewound_into(), Some("root"));
        assert!(c.dropped_unfinished().is_empty());
    }

    #[test]
    fn the_revisit_pass_visits_only_the_unfinished_steps() {
        // Steps already completed stay completed — the rewind is to pick up
        // what was skipped, not to re-run the phase.
        let mut c = cur(&["A", "B", "C", "D"]);
        c.mark_done(); // A
        c.mark_abandoned(); // B skipped
        c.mark_done(); // C
        c.mark_abandoned(); // D skipped
        assert_eq!(c.current().unwrap().1.name, "B", "rewound to the first gap");
        c.mark_done(); // B finished on the revisit
        assert_eq!(c.current().unwrap().1.name, "D", "skips C, already done");
        c.mark_done(); // D finished
        assert!(
            !c.is_active(),
            "nothing outstanding — now the skill is done"
        );
        assert!(c.dropped_unfinished().is_empty());
    }

    #[test]
    fn a_second_abandonment_consumes_the_step_and_reports_it_unfinished() {
        // The revisit is one pass. A step abandoned again is genuinely given
        // up on — but the frame pops saying so, rather than claiming success.
        let mut c = cur(&["A", "B"]);
        c.mark_abandoned(); // A
        c.mark_done(); // B -> rewind to A
        assert_eq!(c.current().unwrap().1.name, "A");
        c.mark_abandoned(); // A again
        assert!(!c.is_active());
        assert_eq!(c.dropped_unfinished(), ["root/A"]);
    }

    #[test]
    fn abandoned_marks_survive_the_save_load_roundtrip() {
        // The cursor is reloaded from disk every round, so an in-memory-only
        // mark would be forgotten before the frame ever exhausted.
        let mut c = cur(&["A", "B"]);
        c.mark_abandoned();
        let c: SkillCursor = serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
        let mut c = c;
        c.mark_done(); // B
        assert_eq!(
            c.current().map(|(_, s)| s.name.clone()),
            Some("A".to_string()),
            "the abandoned step must outlive a round boundary"
        );
    }

    #[test]
    fn descending_discharges_the_invoking_steps_abandoned_mark() {
        // An invoke step whose handoff failed gets abandoned; if a later round
        // resolves it after all, descending IS that step's work, so it must
        // not linger as a gap and drag the frame back.
        let mut c = cur(&["A", "Invoke child skill"]);
        c.mark_abandoned(); // A skipped
        c.descend("child", Path::new("/skills/child"), steps(&["X"]));
        c.mark_done(); // X -> child pops -> root exhausted -> rewind to A
        assert_eq!(c.current().unwrap(), ("root", &steps(&["A"])[0]));
        c.mark_done();
        assert!(!c.is_active());
        assert!(c.dropped_unfinished().is_empty());
    }

    #[test]
    fn the_20260828_e2e_chain_leaves_a_cursor_for_the_retry_to_resume() {
        // Replays the shape that lost the run, on both frames at once.
        // Umbrella `uds-package` invokes `integrate`; inside it,
        // ImplementConfigChart is retired by the round cap mid-write (that is
        // where `path:` shipped instead of `localPath:`), the terminal
        // DeployTestEnvironment never runs, and back on the umbrella the
        // step guarding the deploy is retired on repeated stops.
        //
        // What must hold at the end is not that the run succeeds — it is that
        // the cursor is still ACTIVE. `--continue` re-routes from scratch
        // whenever it is not (run.rs), which is how both retries landed in a
        // validate skill with no deploy step and reported success.
        let mut c = cur(&[
            "Plan",
            "Invoke integrate skill",
            "ExecuteAndVerify",
            "ReportGaps",
        ]);
        c.mark_done(); // Plan
        c.descend(
            "integrate",
            Path::new("/skills/integrate"),
            steps(&["ImplementConfigChart", "DeployTestEnvironment"]),
        );
        c.mark_abandoned(); // round cap retires ImplementConfigChart mid-write
        assert!(
            !c.may_auto_advance(),
            "DeployTestEnvironment is terminal — no valve may retire it"
        );
        c.mark_done(); // the model finally deploys
        // integrate is exhausted but skipped a step, so it does not pop.
        assert_eq!(
            c.current().map(|(sk, s)| (sk.to_string(), s.name.clone())),
            Some(("integrate".into(), "ImplementConfigChart".into())),
            "the skipped chart step comes back before the phase can end"
        );
        c.mark_abandoned(); // give up on it a second time: now it is consumed
        assert_eq!(c.dropped_unfinished(), ["integrate/ImplementConfigChart"]);
        assert_eq!(c.current().unwrap().1.name, "ExecuteAndVerify");
        c.mark_abandoned(); // umbrella step retired on repeated stops
        c.mark_done(); // ReportGaps
        assert!(
            c.is_active(),
            "the umbrella skipped a step, so the cursor must survive for --continue"
        );
        assert_eq!(c.current().unwrap().1.name, "ExecuteAndVerify");
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
    fn handoff_requires_token_boundaries_not_raw_containment() {
        // A skill name inside a file path or longer identifier is a passing
        // mention, not a handoff — `chart/templates/uds-package.yaml` must
        // not descend into uds-package.
        let installed = vec!["uds-package".to_string(), "uds".to_string()];
        let c = cur(&["Create chart/templates/uds-package.yaml from the template"]);
        assert_eq!(c.handoff_target(&installed), None);

        let c = cur(&["Invoke uds-package skill"]);
        assert_eq!(c.handoff_target(&installed).as_deref(), Some("uds-package"));
        // longest name still wins over its own substring
        let c = cur(&["Invoke the uds-package skill, part of uds"]);
        assert_eq!(c.handoff_target(&installed).as_deref(), Some("uds-package"));
        // sentence-ending period is a boundary, an extension is not
        let c = cur(&["Finish with uds-package."]);
        assert_eq!(c.handoff_target(&installed).as_deref(), Some("uds-package"));
        let c = cur(&["Edit my-uds-package now"]);
        assert_eq!(c.handoff_target(&installed), None);
    }

    #[test]
    fn contains_token_boundary_cases() {
        assert!(contains_token("invoke uds-package skill", "uds-package"));
        assert!(contains_token("uds-package", "uds-package"));
        assert!(contains_token("(uds-package)", "uds-package"));
        assert!(contains_token("run uds-package.", "uds-package"));
        assert!(!contains_token("chart/uds-package.yaml", "uds-package"));
        assert!(!contains_token("my-uds-package", "uds-package"));
        assert!(!contains_token("uds-packages", "uds-package"));
        assert!(!contains_token("uds-package_v2", "uds-package"));
        // rejected first occurrence must not mask a valid later one
        assert!(contains_token(
            "uds-package.yaml uses uds-package",
            "uds-package"
        ));
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

    /// What the deterministic layer now returns on its own: the FIRST
    /// candidate. In production a classifier picks from this list (see
    /// `skill_router::classify_handoff`) — these cases pin the retrieval half,
    /// which is what has to be right before the model is asked anything.
    fn first_candidate(c: &SkillCursor, installed: &[String]) -> Option<String> {
        let prose = c.handoff_prose()?;
        c.handoff_candidates(&prose, installed).into_iter().next()
    }

    #[test]
    fn handoff_failure_counter_counts_and_resets_on_step_change() {
        // Retry-before-consume: the caller only mark_done()s the invoke step
        // once this counter passes its threshold, so a transient extraction
        // failure can't silently skip a whole sub-skill.
        let mut c = cur(&["Invoke child skill", "B"]);
        assert_eq!(c.note_handoff_failure(), 1);
        assert_eq!(c.note_handoff_failure(), 2);
        c.mark_done(); // step change resets the counter
        assert_eq!(c.note_handoff_failure(), 1);
    }

    #[test]
    fn handoff_failure_counter_resets_on_successful_descend() {
        // A failure followed by a successful descend starts the sub-skill
        // with a clean slate — its own eventual handoffs get full retries.
        let mut c = cur(&["Invoke child skill", "B"]);
        assert_eq!(c.note_handoff_failure(), 1);
        c.descend("child", Path::new("/skills/child"), steps(&["X"]));
        assert_eq!(c.note_handoff_failure(), 1);
    }

    #[test]
    fn handoff_failure_counter_survives_save_load_roundtrip() {
        // The cursor is reloaded from disk every round, so the counter must
        // persist — otherwise every round sees "first failure" and the
        // threshold never trips.
        let tmp = tempfile::TempDir::new().unwrap();
        let mut config = crate::config::Config::default();
        config.project_root = tmp.path().to_path_buf();
        let mut c = cur(&["Invoke child skill"]);
        c.note_handoff_failure();
        c.note_handoff_failure();
        save(&config, &c);
        let mut reloaded = load(&config);
        assert_eq!(reloaded.note_handoff_failure(), 3);
    }

    #[test]
    fn body_handoff_falls_back_to_the_distilled_body() {
        // No readable skill dir here, so the source-anchor path yields nothing
        // and matching falls back to the distilled body — still gated on the
        // LAST step.
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
            first_candidate(&c, &installed),
            None,
            "mid-step mention must not descend"
        );
        // Final step (Integrate) — but not distilled yet → None.
        c.mark_done(); // -> Integrate (idx 2, last)
        assert_eq!(first_candidate(&c, &installed), None, "no body yet");
        // Distill the final step's handoff prose → now it fires.
        c.cache("Continue with the uds-package-integrate skill.".into());
        assert_eq!(
            first_candidate(&c, &installed).as_deref(),
            Some("uds-package-integrate")
        );
    }

    /// The real `uds-package-build` SKILL.md tail, verbatim.
    fn build_skill_dir() -> tempfile::TempDir {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("SKILL.md"),
            "# uds-package-build\n\n## Step CreateUDSBundle: Assemble the bundle\n\
             Write `bundle/uds-bundle.yaml` referencing chart/templates/uds-package.yaml.\n\n\
             ## Step EnterIntegrationPhase: Next phase\n\
             Continue with the uds-package-integrate skill. Do not treat `uds zarf dev lint` \
             as the Validation Phase; the Integration Phase invokes `validate-package` through \
             the `uds-package-validate` skill.\n",
        )
        .unwrap();
        tmp
    }

    fn build_cursor(dir: &Path) -> SkillCursor {
        let mut c = SkillCursor::default();
        c.push_skill(
            "uds-package-build",
            dir,
            vec![
                SkillStep {
                    name: "CreateUDSBundle".into(),
                    anchor: "## Step CreateUDSBundle: Assemble the bundle".into(),
                },
                SkillStep {
                    name: "EnterIntegrationPhase".into(),
                    anchor: "## Step EnterIntegrationPhase: Next phase".into(),
                },
            ],
        );
        c
    }

    #[test]
    fn body_handoff_reads_the_source_anchor_without_distillation() {
        // The regression that cost the live e2e its whole integration phase:
        // the judge advanced onto the final handoff step after that round's
        // distillation had run, so `cached()` was empty and the handoff never
        // fired. Reading the source at the step's anchor makes it independent
        // of any LLM call having landed.
        let tmp = build_skill_dir();
        let mut c = build_cursor(tmp.path());
        let installed = vec!["uds-package-integrate".to_string()];
        assert_eq!(first_candidate(&c, &installed), None, "not the last step");
        c.mark_done();
        assert!(c.cached().is_none(), "precondition: undistilled");
        assert_eq!(
            first_candidate(&c, &installed).as_deref(),
            Some("uds-package-integrate")
        );
    }

    #[test]
    fn body_handoff_takes_the_first_skill_named_not_the_longest() {
        // That same step names TWO installed skills: integrate (the handoff)
        // and validate (a "don't confuse this with…" aside). Longest-first
        // picked the right one by a single character (21 vs 20) — not a
        // signal worth depending on. Position is the real one.
        let tmp = build_skill_dir();
        let mut c = build_cursor(tmp.path());
        c.mark_done();
        let installed = vec![
            "uds-package-validate".to_string(),
            "uds-package-integrate".to_string(),
        ];
        assert_eq!(
            first_candidate(&c, &installed).as_deref(),
            Some("uds-package-integrate")
        );
    }

    #[test]
    fn body_handoff_ignores_skills_named_in_sibling_docs() {
        // gather_material appends the whole doc tree; reference docs name
        // sibling skills constantly. The tail must stop at the first sibling
        // banner so only the step's own prose can trigger a handoff.
        let tmp = build_skill_dir();
        std::fs::write(
            tmp.path().join("reference.md"),
            "See also the uds-package-publish skill for registry pushes.\n",
        )
        .unwrap();
        let mut c = build_cursor(tmp.path());
        c.mark_done();
        let installed = vec!["uds-package-publish".to_string()];
        assert_eq!(
            first_candidate(&c, &installed),
            None,
            "a mention in a sibling reference doc is not this step's handoff"
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
    fn handoff_verdict_is_cached_per_step_and_survives_a_reload() {
        // The cursor is reloaded from disk every round and `descend()` is
        // irreversible, so the decision must be taken ONCE and stay taken —
        // a verdict that re-flips mid-step would have nothing to undo.
        let tmp = tempfile::TempDir::new().unwrap();
        let mut config = crate::config::Config::default();
        config.project_root = tmp.path().to_path_buf();

        let mut c = cur(&["A", "B"]);
        assert!(!c.handoff_decided());
        assert_eq!(c.cached_handoff(), None);
        // "decided: no handoff" is a real answer, distinct from "not asked".
        c.cache_handoff(String::new());
        assert!(c.handoff_decided());
        assert_eq!(c.cached_handoff(), None);

        c.mark_done(); // -> B, a different key
        assert!(
            !c.handoff_decided(),
            "the verdict is per step, not per skill"
        );
        c.cache_handoff("child".into());
        assert_eq!(c.cached_handoff(), Some("child"));

        save(&config, &c);
        let reloaded = load(&config);
        assert_eq!(reloaded.cached_handoff(), Some("child"));
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
    fn active_block_is_absent_until_the_step_is_distilled() {
        // An undistilled step must produce NO block at all. The old
        // placeholder ("(preparing instructions…)") rendered as a step with no
        // work and no completion criterion under a standing "call
        // skill(action='done') when its DONE WHEN is met" instruction — the
        // live e2e model complied, popped the frame, and skipped the whole
        // integration phase.
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join(".miniswe")).unwrap();
        let mut config = Config::default();
        config.project_root = tmp.path().to_path_buf();

        let mut c = cur(&["Scaffold"]);
        save(&config, &c);
        assert!(
            active_step_block(&config).is_none(),
            "an undistilled step must not be rendered as actionable"
        );

        c.cache("call scaffold-package with targetDir".into());
        save(&config, &c);
        let block = active_step_block(&config).unwrap();
        assert!(block.contains("[SKILL STEP]"));
        assert!(block.contains("call scaffold-package"));
        assert!(block.contains("skill(action='done')"));
    }
}
