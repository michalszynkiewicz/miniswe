//! T2c frozen-signature stuck detection + note injection (gaps 9/10).
//!
//! Port of the winning trigger from the offline eval
//! (`scripts/moments/trigger-eval.py`, 2026-08-24): the compiler/test
//! *signal signature* — AST state, LSP project errors, last tool failure,
//! check/gate state, shell exit state — unchanged for `FROZEN_ROUNDS`
//! rounds AND `MIN_FROZEN_SECS` of wall clock. Both halves are load-bearing:
//! rounds-only fires on fast healthy models (Laguna, 1.7 s/round — 3/3
//! false positives), wall-only fires late on slow ones. T2c scored 6/6
//! labeled stuck segments, 0 fires on all three healthy Laguna runs.
//!
//! On fire, the round's last tool result gets a note appended (the
//! placement the warm-replay probes validated):
//!   - red signal → stuck-note: broke glimmer's 110-round read loop 8/10
//!     vs control 2/10 (`tier1-glimmer-stuck-probe.py`, K=10 warm);
//!   - green signal + every plan step checked → done-note: teaching "a
//!     reply with no tool call ends the task" finished the can't-stop
//!     dither 10/10 vs control 1/10. The system prompt says "Emit ONE tool
//!     call per response" and never explains how to finish — the note is
//!     where the model first learns the mechanism.
//!
//! Wall time comes in as caller-supplied seconds (not `Instant::now()`)
//! so tests replay recorded pacing deterministically.

use std::hash::{DefaultHasher, Hash, Hasher};

/// Rounds the signature must stay frozen before firing (and between
/// re-fires within one frozen episode).
pub const FROZEN_ROUNDS: usize = 15;
/// Minimum wall-clock seconds frozen — the pace-fairness floor that keeps
/// fast healthy models (Laguna: 15 rounds ≈ 26 s) from tripping the
/// round count.
pub const MIN_FROZEN_SECS: f64 = 240.0;
/// Arm at the first `plan(action='set')` or this round, whichever first
/// (pre-plan exploration legitimately produces no signal).
const ARM_FALLBACK_ROUND: usize = 20;
/// Hard cap on notes per session — this is a nudge, not a metronome.
const MAX_FIRES: u32 = 5;
/// A read streak this long earns the "do NOT read it again" sentence.
const READ_STREAK_FOR_NOTE: u32 = 3;

const EDIT_TOOLS: [&str; 5] = [
    "replace_range",
    "insert_at",
    "write_file",
    "edit_file",
    "refactor",
];

/// Which side of the signal the frozen episode is on. The caller picks the
/// note: `Green` + all plan steps checked → done-note, everything else →
/// stuck-note.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StuckKind {
    /// Signal red or unknown: the agent is grinding without moving it.
    Red,
    /// Signal green after at least one edit: likely done and dithering.
    Green,
}

pub struct StuckTracker {
    round: usize,
    now: f64,
    armed_round: Option<usize>,
    plan_set: bool,

    // signature components (mirrors trigger-eval's sig_a)
    ast: Option<String>,
    lsp_hash: Option<u64>,
    fail_hash: Option<u64>,
    check_state: Option<String>,
    shell_state: Option<u64>,

    green: Option<bool>,
    edit_happened: bool,

    sig: Option<u64>,
    sig_round: usize,
    sig_time: f64,

    read_path: Option<String>,
    read_streak: u32,

    fired_at_frozen: usize,
    fires: u32,
}

fn h<T: Hash + ?Sized>(v: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    v.hash(&mut hasher);
    hasher.finish()
}

fn first_line(content: &str) -> &str {
    content.lines().next().unwrap_or("")
}

impl Default for StuckTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl StuckTracker {
    pub fn new() -> Self {
        Self {
            round: 0,
            now: 0.0,
            armed_round: None,
            plan_set: false,
            ast: None,
            lsp_hash: None,
            fail_hash: None,
            check_state: None,
            shell_state: None,
            green: None,
            edit_happened: false,
            sig: None,
            sig_round: 0,
            sig_time: 0.0,
            read_path: None,
            read_streak: 0,
            fired_at_frozen: 0,
            fires: 0,
        }
    }

    /// Call at the top of each round. `now` = seconds since session start.
    pub fn on_round(&mut self, round: usize, now: f64) {
        self.round = round;
        self.now = now;
        if self.armed_round.is_none() && (self.plan_set || round >= ARM_FALLBACK_ROUND) {
            self.armed_round = Some(round);
            // Initialize the signature at arm time: "no signal at all"
            // (pure read drift, north-r2's 169 reads on a clean tree) must
            // be able to freeze too.
            self.resig();
        }
    }

    /// Feed one executed tool call. Mirrors trigger-eval's `on_tool` +
    /// `lspstate` handling: only events that change the *signal* move the
    /// signature; reads and plan(show) never touch it.
    pub fn on_tool(&mut self, name: &str, args: &serde_json::Value, ok: bool, content: &str) {
        let action = args.get("action").and_then(|a| a.as_str()).unwrap_or("");

        // Read streak — wording material for the note, not signature input.
        // Keyed by path only: glimmer's loop jittered line ranges (54-54,
        // 54-55, 52-56), which reset an exact-args key and starved the old
        // ladder.
        if name == "file" && action == "read" {
            let p = args.get("path").and_then(|p| p.as_str());
            if p.is_some() && p == self.read_path.as_deref() {
                self.read_streak += 1;
            } else {
                self.read_streak = 1;
                self.read_path = p.map(str::to_string);
            }
        } else {
            self.read_streak = 0;
            self.read_path = None;
        }

        if ok && EDIT_TOOLS.contains(&name) {
            let mutating =
                name != "refactor" || matches!(action, "add_param" | "drop_param" | "rename");
            if mutating {
                self.edit_happened = true;
                // NOTE: deliberately no direct resig here (that was T2b,
                // 4/6). An edit unfreezes only through the signal it
                // changes — the [ast]/[lsp project] markers in its own
                // result, scanned below. A no-op edit that moves nothing
                // stays frozen.
            }
        }

        if !ok {
            self.fail_hash = Some(h(&format!("{name}{}", first_line(content))));
            self.resig();
        } else if name == "check" {
            let failed = content.contains("FAILED");
            self.check_state = Some(format!("check:{}", if failed { "FAILED" } else { "OK" }));
            self.green = Some(!failed);
            self.resig();
        } else if name == "plan" {
            if action == "set" {
                self.plan_set = true;
            }
            if action == "check" {
                let head: String = first_line(content).chars().take(80).collect();
                self.check_state = Some(format!("gate:{}", h(&head)));
                if content.contains("compile gate passed") {
                    self.green = Some(true);
                } else if content.contains("FAILED") {
                    self.green = Some(false);
                }
                self.resig();
            }
        } else if name == "shell"
            && let Some(exit) = parse_shell_exit(content)
        {
            let cmd = args.get("command").and_then(|c| c.as_str()).unwrap_or("");
            self.shell_state = Some(h(&format!("{cmd}exit{exit}")));
            if cmd.contains("cargo test")
                || cmd.contains("cargo check")
                || cmd.contains("cargo build")
            {
                self.green = Some(exit == 0);
            }
            self.resig();
        }

        // [ast]/[lsp project] markers in the result tail — the same signal
        // the offline eval read from the last tool content per request.
        self.scan_markers(content);
    }

    fn scan_markers(&mut self, content: &str) {
        let tail_start = content
            .char_indices()
            .rev()
            .nth(3999)
            .map(|(i, _)| i)
            .unwrap_or(0);
        let tail = &content[tail_start..];

        let mut changed = false;
        let ast = tail.rfind("[ast] ").map(|pos| {
            tail[pos + 6..]
                .chars()
                .take_while(|c| *c != '\n')
                .take(80)
                .collect::<String>()
        });
        let lsp = tail.rfind("[lsp project] ").and_then(|pos| {
            let rest = &tail[pos + 14..];
            let n: u64 = rest.split_whitespace().next()?.parse().ok()?;
            // n + the next ~300 chars of detail (cut at [revisions]): "3
            // errors" freezing while the errors themselves rotate must NOT
            // read as frozen.
            let detail: String = rest.chars().take(300).collect();
            let detail = detail.split("[revisions]").next().unwrap_or("");
            Some((n, h(&format!("{n}|{detail}"))))
        });
        if let Some((n, lsp)) = lsp {
            self.green = Some(n == 0);
            if self.lsp_hash != Some(lsp) {
                self.lsp_hash = Some(lsp);
                changed = true;
            }
        }
        if let Some(ast) = ast.filter(|a| !a.is_empty()) {
            if !ast.starts_with("ok") {
                self.green = Some(false);
            }
            if self.ast.as_deref() != Some(ast.as_str()) {
                self.ast = Some(ast);
                changed = true;
            }
        }
        if changed {
            self.resig();
        }
    }

    fn resig(&mut self) {
        let sig = h(&format!(
            "{:?}|{:?}|{:?}|{:?}|{:?}",
            self.ast, self.lsp_hash, self.fail_hash, self.check_state, self.shell_state
        ));
        if self.sig != Some(sig) {
            self.sig = Some(sig);
            self.sig_round = self.round;
            self.sig_time = self.now;
            self.fired_at_frozen = 0;
        }
    }

    /// Call at the end of each round (after all tool results). Fires at
    /// 15/30/45… frozen rounds within one episode, `MAX_FIRES` per session.
    pub fn check_fire(&mut self, round: usize, now: f64) -> Option<StuckKind> {
        self.round = round;
        self.now = now;
        let armed = self.armed_round?;
        self.sig?;
        let frozen_rounds = round.saturating_sub(self.sig_round.max(armed));
        let frozen_secs = now - self.sig_time;
        if frozen_rounds >= FROZEN_ROUNDS
            && frozen_secs >= MIN_FROZEN_SECS
            && frozen_rounds >= self.fired_at_frozen + FROZEN_ROUNDS
            && self.fires < MAX_FIRES
        {
            self.fires += 1;
            self.fired_at_frozen = frozen_rounds;
            return Some(if self.green == Some(true) && self.edit_happened {
                StuckKind::Green
            } else {
                StuckKind::Red
            });
        }
        None
    }

    /// Rounds the signature has been frozen (for note wording).
    pub fn frozen_rounds(&self) -> usize {
        self.round
            .saturating_sub(self.sig_round.max(self.armed_round.unwrap_or(0)))
    }

    /// Minutes the signature has been frozen (for note wording).
    pub fn frozen_minutes(&self) -> u64 {
        ((self.now - self.sig_time) / 60.0) as u64
    }

    /// The path a sustained read loop keeps re-reading, if one is running.
    pub fn looping_read_path(&self) -> Option<&str> {
        (self.read_streak >= READ_STREAK_FOR_NOTE)
            .then_some(self.read_path.as_deref())
            .flatten()
    }
}

fn parse_shell_exit(content: &str) -> Option<u32> {
    let pos = content.find("[shell: exit ")?;
    content[pos + 13..]
        .split(|c: char| !c.is_ascii_digit())
        .next()?
        .parse()
        .ok()
}

/// Stuck-note: red-frozen, or green-frozen with unchecked plan steps.
/// Wording generalizes the probe-validated text (8/10 loop breaks); the
/// "do NOT read it again" sentence was the single load-bearing ingredient
/// of the post-revert hint replay (12/12 vs 12/12 read-loop control).
pub fn stuck_note(
    frozen_rounds: usize,
    frozen_minutes: u64,
    looping_read_path: Option<&str>,
    first_unchecked_step: Option<usize>,
) -> String {
    let read_sentence = looping_read_path
        .map(|p| format!(" Re-reading {p} will show the same bytes again — do NOT read it again."))
        .unwrap_or_default();
    let check_clause = first_unchecked_step
        .map(|k| {
            format!(", or run plan(action='check', step={k}) if that step is in fact already done")
        })
        .unwrap_or_default();
    format!(
        "[stuck-check] {frozen_rounds}+ rounds and {frozen_minutes}+ minutes with NO change in \
         the compiler/test signal.{read_sentence} Take ONE concrete action now: make the \
         smallest edit that advances the first unchecked plan step{check_clause}."
    )
}

/// Done-note: green-frozen with every plan step checked. The final clause
/// teaches the finish mechanism the system prompt never states; in the
/// warm-replay probe it alone flipped glimmer's can't-stop dither from
/// 1/10 to 10/10 clean finishes.
pub fn done_note() -> String {
    "[done-check] Every plan step is checked off and the compiler/test signal is green. The \
     task appears COMPLETE. Do not re-read files or re-show the plan. If something is genuinely \
     missing, name it and fix it with ONE edit; otherwise finish NOW: reply with a one-paragraph \
     summary and NO tool call — a reply without a tool call ends the task."
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// ~9 s/round pacing (glimmer-like). 15 rounds ≈ 135 s, so the 4-min
    /// wall matters and tests must advance both dimensions explicitly.
    fn read_result() -> &'static str {
        "54: let assembled = context::assemble(&config);\n[round 100/600]"
    }

    fn arm(t: &mut StuckTracker) {
        // plan(set) at round 1 arms the tracker immediately.
        t.on_round(1, 0.0);
        t.on_tool("plan", &json!({"action": "set"}), true, "plan saved");
        t.on_round(2, 9.0);
    }

    #[test]
    fn frozen_read_loop_fires_red_after_rounds_and_wall() {
        let mut t = StuckTracker::new();
        arm(&mut t);
        let mut fired = None;
        for r in 2..40 {
            let now = r as f64 * 20.0;
            t.on_round(r, now);
            t.on_tool(
                "file",
                &json!({"action": "read", "path": "tests/e2e_context.rs"}),
                true,
                read_result(),
            );
            if let Some(k) = t.check_fire(r, now) {
                fired = Some((r, k));
                break;
            }
        }
        // armed at round 2 (plan set at 1, arm evaluates on round 2's
        // on_round)… arm happens at round 1? plan_set flips after round 1's
        // on_round, so arming lands on round 2. Frozen 15 rounds → round 17,
        // wall 17*20 s = 340 s ≥ 240 s.
        let (r, k) = fired.expect("must fire");
        assert_eq!(k, StuckKind::Red);
        assert_eq!(r, 17);
        assert!(t.looping_read_path().is_some());
    }

    #[test]
    fn wall_clock_floor_blocks_fast_healthy_pacing() {
        // Laguna pacing: 1.7 s/round. 15 frozen rounds = 25 s — must NOT fire.
        let mut t = StuckTracker::new();
        arm(&mut t);
        for r in 2..120 {
            let now = r as f64 * 1.7;
            t.on_round(r, now);
            t.on_tool(
                "file",
                &json!({"action": "read", "path": "src/main.rs"}),
                true,
                read_result(),
            );
            if r as f64 * 1.7 < MIN_FROZEN_SECS {
                assert!(
                    t.check_fire(r, now).is_none(),
                    "fired at {r} before wall floor"
                );
            }
        }
    }

    #[test]
    fn changing_signal_never_fires() {
        // Healthy run: every round's edit moves the lsp signal.
        let mut t = StuckTracker::new();
        arm(&mut t);
        for r in 2..60 {
            let now = r as f64 * 30.0;
            t.on_round(r, now);
            t.on_tool(
                "replace_range",
                &json!({"path": "src/main.rs", "start": 1, "end": 2}),
                true,
                &format!("edited\n[ast] ok\n[lsp project] {} errors: E{}", 60 - r, r),
            );
            assert!(
                t.check_fire(r, now).is_none(),
                "fired at {r} on a moving signal"
            );
        }
    }

    #[test]
    fn green_frozen_after_edit_fires_green() {
        let mut t = StuckTracker::new();
        arm(&mut t);
        t.on_round(2, 20.0);
        t.on_tool(
            "replace_range",
            &json!({"path": "src/main.rs", "start": 1, "end": 2}),
            true,
            "edited\n[ast] ok\n[lsp project] 0 errors",
        );
        t.on_tool(
            "shell",
            &json!({"command": "cargo test"}),
            true,
            "ok\n[shell: exit 0]",
        );
        let mut fired = None;
        for r in 3..40 {
            let now = 20.0 + r as f64 * 25.0;
            t.on_round(r, now);
            t.on_tool(
                "file",
                &json!({"action": "read", "path": "tests/e2e_context.rs"}),
                true,
                read_result(),
            );
            if let Some(k) = t.check_fire(r, now) {
                fired = Some(k);
                break;
            }
        }
        assert_eq!(fired, Some(StuckKind::Green));
    }

    #[test]
    fn refire_needs_another_full_window_and_caps() {
        let mut t = StuckTracker::new();
        arm(&mut t);
        let mut fires = vec![];
        for r in 2..200 {
            let now = r as f64 * 20.0;
            t.on_round(r, now);
            t.on_tool(
                "file",
                &json!({"action": "read", "path": "a.rs"}),
                true,
                read_result(),
            );
            if t.check_fire(r, now).is_some() {
                fires.push(r);
            }
        }
        assert_eq!(fires.len(), MAX_FIRES as usize);
        for w in fires.windows(2) {
            assert!(w[1] - w[0] >= FROZEN_ROUNDS, "refire too early: {fires:?}");
        }
    }

    #[test]
    fn failure_churn_keeps_signature_moving() {
        // devstral-gap9 shape inverse: DIFFERENT failures each round → no fire;
        // the SAME failure re-issued forever → fire.
        let mut t = StuckTracker::new();
        arm(&mut t);
        for r in 2..30 {
            let now = r as f64 * 20.0;
            t.on_round(r, now);
            t.on_tool(
                "refactor",
                &json!({"action": "add_param"}),
                false,
                &format!("validator rejected: missing position (variant {r})"),
            );
            assert!(t.check_fire(r, now).is_none());
        }
        let mut t = StuckTracker::new();
        arm(&mut t);
        let mut fired_at = None;
        for r in 2..40 {
            let now = r as f64 * 20.0;
            t.on_round(r, now);
            t.on_tool(
                "refactor",
                &json!({"action": "add_param"}),
                false,
                "validator rejected: missing position",
            );
            if t.check_fire(r, now).is_some() {
                fired_at = Some(r);
                break;
            }
        }
        assert_eq!(fired_at, Some(17));
    }

    #[test]
    fn unarmed_never_fires() {
        let mut t = StuckTracker::new();
        for r in 1..15 {
            let now = r as f64 * 60.0;
            t.on_round(r, now);
            t.on_tool(
                "file",
                &json!({"action": "read", "path": "a.rs"}),
                true,
                read_result(),
            );
            assert!(t.check_fire(r, now).is_none(), "fired unarmed at {r}");
        }
    }

    #[test]
    fn arms_at_fallback_round_without_plan() {
        let mut t = StuckTracker::new();
        let mut fired = None;
        for r in 1..60 {
            let now = r as f64 * 20.0;
            t.on_round(r, now);
            t.on_tool(
                "file",
                &json!({"action": "read", "path": "a.rs"}),
                true,
                read_result(),
            );
            if t.check_fire(r, now).is_some() {
                fired = Some(r);
                break;
            }
        }
        // arms at round 20, fires 15 frozen rounds later
        assert_eq!(fired, Some(35));
    }

    #[test]
    fn note_wording() {
        let n = stuck_note(17, 5, Some("tests/e2e_context.rs"), Some(3));
        assert!(n.contains("17+ rounds and 5+ minutes"));
        assert!(n.contains("Re-reading tests/e2e_context.rs"));
        assert!(n.contains("do NOT read it again"));
        assert!(n.contains("plan(action='check', step=3)"));
        let bare = stuck_note(15, 4, None, None);
        assert!(!bare.contains("Re-reading"));
        assert!(!bare.contains("step="));
        assert!(done_note().contains("NO tool call"));
    }

    #[test]
    fn jittered_read_ranges_keep_the_streak() {
        let mut t = StuckTracker::new();
        arm(&mut t);
        for (i, range) in [(54, 54), (54, 55), (52, 56), (54, 54)].iter().enumerate() {
            t.on_round(2 + i, 20.0 * (2 + i) as f64);
            t.on_tool(
                "file",
                &json!({"action": "read", "path": "tests/e2e_context.rs",
                        "start": range.0, "end": range.1}),
                true,
                read_result(),
            );
        }
        assert_eq!(t.looping_read_path(), Some("tests/e2e_context.rs"));
    }
}
