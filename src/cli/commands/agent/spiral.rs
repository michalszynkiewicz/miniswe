//! Spiral-reset (experimental, opt-in via `tools.spiral_reset`).
//!
//! A small model can tail-chase: re-try the same failing edit, revert it, re-try
//! the same edit, revert again — cycling without progress. iter-1 on Gemma 4
//! reverted one file to its clean rev **4 times**. A bare file-revert doesn't
//! break this (the agent already reverts; its *context* drags it back into the
//! same approach), and the loop detector misses it (the calls aren't
//! byte-identical) and so does the AST-cascade guard (the AST stays ok).
//!
//! The fix isn't more reverting — it's a *cognitive* reset: when the agent has
//! reverted the same file `SPIRAL_REVERT_THRESHOLD` times in a turn, inject a
//! message that (1) names what it kept trying, (2) says that approach failed and
//! must NOT be repeated, and (3) forces a `plan` update with a concrete
//! redirection. An API probe on Gemma 4 (see scripts / commit msg) found a
//! silent revert produced 0/8 approach-switches, a generic "try differently"
//! 1/8, and this inform+replan+redirect framing ~8/8.

use crate::tools::RevisionStore;

/// Reverts of the same file in a turn before the spiral reset fires. iter-1's
/// signature was 4 reverts of one file to pristine; 3 catches it while leaving
/// a legitimate revert-then-fix (1–2) alone.
pub const SPIRAL_REVERT_THRESHOLD: usize = 3;

/// Max spiral resets per turn — bounds the mechanism so it can't itself loop.
pub const MAX_RESETS_PER_TURN: usize = 2;

/// Done-gate blocks before a context-reset replaces in-context grinding with a
/// fresh restart. The done-gate normally keeps the agent fixing in-context after
/// it fails the verification check — but that grinds in the polluted, failure-
/// primed context and thrashes (observed on qwen: attempt 1 ground 121 rounds
/// over 3 gate blocks and still failed; a FRESH attempt fixed it in 53). The
/// best-of-3 fresh-attempt mechanism works precisely because it resets context.
/// This brings that reset in-session: after this many blocks, drop the polluted
/// history and re-assemble a clean context (files persist). 2 = one in-context
/// retry, then restart clean.
pub const GATE_RESET_AFTER_BLOCKS: usize = 2;

/// Max gate-triggered context-resets per turn (don't loop on resets).
pub const MAX_GATE_RESETS: usize = 1;

/// Fresh-start user message for a gate-triggered context reset. The verbose
/// turn-by-turn history is dropped on re-assemble, but the agent's *deliberate*
/// memory — its PLAN (and scratchpad) — is re-injected from disk, so this isn't
/// blind amnesia. Rather than "discard everything," it names the failure and
/// makes the agent DISTILL the fix into a revised plan: the plan that led here
/// is shown above, so revise IT to address the specific failure, then execute.
/// (Keeps the structure, drops the noise — the agent decides what carries
/// forward via its plan, not the harness via a blind cut.)
pub fn build_gate_reset_prompt(task: &str, check_output: &str) -> String {
    format!(
        "{task}\n\n\
         [The verification check has failed repeatedly, so the conversation has been COMPACTED to \
         a clean slate. Your code changes ARE still on disk, and your current plan is shown above. \
         That plan and approach led to the repeated failure — do NOT just retry it. FIRST revise \
         your plan with plan(action='refine' or 'set') so it directly addresses the SPECIFIC \
         problem the check reports below; then execute the revised plan, re-reading the current \
         state of the files as needed. Latest check output:\n{check_output}]"
    )
}

/// Labels of the most-recent reverted (tombstoned) revisions for `path`, so the
/// reset can name what the agent kept trying. Best-effort; empty if unavailable.
pub fn tried_edit_labels(revisions: &RevisionStore, path: &str, max: usize) -> Vec<String> {
    let mut labels: Vec<String> = revisions
        .list(path)
        .into_iter()
        .filter(|r| r.reverted && r.number != 0)
        .map(|r| r.label)
        .collect();
    labels.reverse(); // most-recent first
    labels.truncate(max);
    labels
}

/// The reset message injected on a detected revert-loop. Framing chosen by API
/// probe (Gemma 4): NOT a prescriptive "use tool X" redirect — that misdirects
/// when X doesn't fit (a leading "use refactor" sent gemma to call refactor on a
/// *struct* it can't touch, giving 0/6). Instead it forces error-reading and
/// fit-assessment before replanning, which lets the model adapt to what it's
/// actually changing (refactor where it fits, a manual edit where it doesn't).
/// Probe: this avoided the misdirection while still steering to the right tool.
pub fn build_reset_message(path: &str, revert_count: usize, tried: &[String]) -> String {
    let tried_str = if tried.is_empty() {
        "the same edits".to_string()
    } else {
        tried.join(", ")
    };
    format!(
        "STOP — you are cycling. You have reverted {path} to a clean state {revert_count} times, \
         re-trying the same kind of edit ({tried_str}) and hitting the same failures. That approach \
         is NOT working — do NOT repeat it. {path} is reset to a clean state.\n\
         Before editing again: READ the specific error messages you kept getting — they tell you \
         what is wrong with this approach — and decide whether the tool and the edit you were using \
         actually fit what you are changing here. Then call plan(action='set' or 'refine') with a \
         genuinely different approach, and proceed."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_message_names_failure_forces_replan_and_redirects() {
        let msg = build_reset_message(
            "src/cli/mod.rs",
            3,
            &["replace_range L16-22".into(), "replace_range L26-30".into()],
        );
        // names the file + count
        assert!(msg.contains("src/cli/mod.rs"));
        assert!(msg.contains("3 times"));
        // names what was tried
        assert!(msg.contains("replace_range L16-22"));
        // frames as failed + don't-repeat
        assert!(msg.contains("NOT working") && msg.contains("do NOT repeat"));
        // forces a replan
        assert!(msg.contains("plan(action="));
        // active ingredient: error-reading + fit-assessment (NOT a leading tool
        // prescription — that misdirected gemma onto a struct refactor can't do)
        assert!(msg.contains("READ the specific error"));
        assert!(msg.to_lowercase().contains("fit"));
        assert!(!msg.contains("refactor")); // deliberately not prescribed
    }

    #[test]
    fn gate_reset_prompt_keeps_task_compacts_and_forces_replan() {
        let p = build_gate_reset_prompt("Add the --foo flag", "COMPILES but foo NOT consumed");
        assert!(p.contains("Add the --foo flag")); // original task preserved
        assert!(p.contains("COMPACTED")); // history dropped, clean slate
        assert!(p.contains("still on disk")); // files persist
        assert!(p.contains("plan(action=")); // forces a plan revision
        assert!(p.contains("do NOT just retry it")); // don't repeat the failed approach
        assert!(p.contains("COMPILES but foo NOT consumed")); // check output
    }

    #[test]
    fn reset_message_handles_no_known_edits() {
        let msg = build_reset_message("a.rs", 4, &[]);
        assert!(msg.contains("the same edits"));
        assert!(msg.contains("4 times"));
    }

    #[test]
    fn tried_labels_returns_reverted_only_recent_first() {
        let s = RevisionStore::with_cap(50);
        s.ensure_pristine("a.rs", "v0").unwrap();
        for (i, lbl) in [(1usize, "edit-A"), (2, "edit-B"), (3, "edit-C")] {
            s.record(
                "a.rs",
                &format!("v{i}"),
                crate::tools::fast::RecordArgs {
                    operation: "replace_range",
                    label: lbl,
                    range: None,
                    payload: None,
                    added: 0,
                    removed: 0,
                    ast_ok: true,
                    ast_error: None,
                    file_errors: 0,
                    project_errors: 0,
                },
            )
            .unwrap();
        }
        // revert to rev_0 → revs 1,2,3 become tombstones
        s.mark_reverted_to("a.rs", 0).unwrap();
        let got = tried_edit_labels(&s, "a.rs", 2);
        assert_eq!(got, vec!["edit-C", "edit-B"]); // most-recent first, capped at 2
    }
}
