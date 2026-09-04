//! Tool-surface configuration: which tool groups the LLM sees and the
//! experimental agent-behavior flags.

use serde::{Deserialize, Serialize};

/// Toggle which tool groups are available to the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ToolsConfig {
    /// Context tools: get_repo_map, get_project_info, get_architecture_notes
    pub context_tools: bool,
    /// LSP tools: goto_definition, find_references
    pub lsp_tools: bool,
    /// Web tools: web_search, web_fetch
    pub web_tools: bool,
    /// Structured plan tool
    pub plan: bool,
    /// Scratchpad (task_update)
    pub scratchpad: bool,
    /// Agent ceremony level. `"strict"` (DEFAULT): plan-first gating +
    /// phase-aware prompt + progress nudges — the proven-good behavior
    /// (Qwen 6/6, passes `smoke`). `"off"`: leaner/faster minimal
    /// prompt with no plan machinery, but the real docker bench proved
    /// it regresses the end-to-end `smoke` check (opt-in only). See
    /// `docs/tiered-agent-design.md` §Real-bench refutation.
    pub ceremony: CeremonyMode,
    /// Flat refactor tools. `false` (default): grouped
    /// `refactor{action,position,callsite_fill_in}` (proven 6/6).
    /// `true`: replace it with flat single-purpose
    /// `add_function_param`/`drop_function_param`/`rename_symbol` (no
    /// `position`/`callsite_fill_in` DSL — removes the documented
    /// deterministic Devstral mangling). Under A/B evaluation; see
    /// `docs/tiered-agent-design.md`.
    pub flat: bool,
    /// Edit-tool surface: `"fast"` (default) exposes the primitive
    /// `replace_range` / `insert_at` / `revert` / `check` surface from
    /// `src/tools/fast/`; `"smart"` replaces it with `edit_file`, which
    /// delegates to an inner-model planner. See `docs/fast-mode-design.md`.
    pub edit_mode: EditMode,
    /// EXPERIMENTAL (fast mode only). When `true`, after a structural edit
    /// (`replace_range` / `insert_at`) leaves the file's AST broken for
    /// `CASCADE_THRESHOLD` consecutive edits in a row, the file is forcibly
    /// reverted to the most recent AST-clean revision and the model is told
    /// to stop digging and make one balanced edit. Targets the observed
    /// brace-cascade loop (small models patch line-by-line into ever-deeper
    /// breakage). On by default: a 10-run Gemma-4 A/B showed it removes the
    /// catastrophic tail (ON ~5.8 with no 0/6 vs OFF ~3.75 incl. a 17-broken
    /// 0/6); set `false` for the pure fast-mode philosophy of tolerating
    /// transient broken AST. Triggers only on `CASCADE_THRESHOLD` *consecutive*
    /// broken-AST edits, so a deliberate 1–2-step broken intermediate is safe.
    pub auto_revert_ast_cascade: bool,
    /// EXPERIMENTAL. When `true`, after the behavioral done-gate
    /// (`[validation]`) blocks completion `DEBUGGER_TRIGGER_BLOCKS` times in a
    /// turn, spin up a fresh-context "debugger" sub-agent handed only the
    /// specific failing check output + the changed files and told to fix only
    /// that. The bet (see GitHub #40) is *attention reset / fresh eyes* on a
    /// "knows-it's-wrong-but-can't-recover" stall — not extra capability
    /// (same weights). `true` since 2026-07-15 (bench-unconditional since
    /// 07-06: helps or no-ops, never hurt). `false` keeps the gate's plain retry-nudge
    /// loop. Requires a `[validation]` command to do anything. A/B only.
    pub reactive_debugger: bool,
    /// EXPERIMENTAL. Requires `reactive_debugger`. When `true`, the debugger may
    /// re-fire WITHIN a turn — but only when the gate's failure SIGNATURE changes
    /// (e.g. compile error fixed, now a runtime/smoke failure), so it walks the
    /// failure chain one fresh diagnosis per distinct failure instead of
    /// re-diagnosing the same thing. `false` keeps single-fire; `true` since
    /// 2026-07-15 (bench ran it unconditionally since 07-06: helps or no-ops,
    /// never hurt). The
    /// blunt "fire ≤N×/turn" variant regressed before (scattered diagnoses the
    /// small model couldn't integrate); the distinct-signature gate is the
    /// difference. A/B only.
    pub debugger_multifire: bool,
    /// EXPERIMENTAL (fast mode). When `true`, detect a *revert-loop* spiral —
    /// the agent reverting the same file to a clean revision
    /// `SPIRAL_REVERT_THRESHOLD` times in a turn (it's cycling: re-trying the
    /// same failing edits and undoing them). On detection, inject a reset
    /// message that names what was tried, says it failed, and forces a
    /// `plan(action='set'/'refine')` with a concrete redirection (use
    /// `refactor` for signature/callsite changes; one balanced edit for a
    /// thrashed region). API-probe on Gemma 4: silent revert 0/8 vs this
    /// framing ~8/8 at making it switch approach. `false` (default). A/B only.
    pub spiral_reset: bool,
    /// EXPERIMENTAL. When `true`, after the done-gate (`[validation]`) blocks
    /// `GATE_RESET_AFTER_BLOCKS` times in a turn, replace its in-context retry
    /// grinding with a CONTEXT RESET: drop the polluted conversation history and
    /// re-assemble a clean context (files persist on disk) — the in-session
    /// equivalent of a best-of-3 fresh attempt. Motivated by: in-context
    /// grinding thrashes in the failure-primed context (qwen: 121 rounds over 3
    /// blocks, still failed) while a fresh attempt fixed it fast (53 rounds).
    /// `false` by default: a controlled gemma-4 A/B (2026-06-29, 3 runs each,
    /// auto_revert on, unified) showed OFF is strictly better — 6.0 vs 5.67 and
    /// ~1.6× faster (≈839s vs ≈1380s). The reset fired 2–4×/run and caused
    /// re-work churn (≈336 vs ≈199 rounds) with no reliability payoff. The qwen
    /// motivation above may still hold on harder/long-repo tasks; opt in there.
    pub gate_context_reset: bool,
    /// EXPERIMENTAL (fast mode + snapshots). When `true`, if the project's LSP
    /// error count stays ABOVE the session baseline for `REVERT_TO_GREEN_BLOCKS`
    /// consecutive rounds (the agent is stuck not converging), revert the ENTIRE
    /// working tree to the last round that was green (compiled ≤ baseline) and
    /// tell the agent to restart from that clean base. Unlike
    /// `auto_revert_ast_cascade` (per-file, AST-syntax only), this is tree-wide
    /// and triggers on SEMANTIC breaks (type errors, deleted methods) that parse
    /// fine. Motivated by run2 (deleted `is_enabled`, broke a caller, ground 100+
    /// rounds unable to untangle its own change). `false` (default). A/B only.
    pub revert_to_green: bool,
    /// EXPERIMENTAL. When `true`, the first time the behavioral done-gate
    /// (`[validation]`) blocks, inject the ORIGINAL task goal and force a fresh
    /// `plan(action='set')` re-derived from it. Motivated by the run2 recovery
    /// dissection: under compile-firefighting the agent's plan DEGRADES from
    /// "build the feature" to "make it compile", dropping the behavior step; it
    /// then repairs to "compiles" and stops, never implementing the missing
    /// consumption. Re-anchoring on the goal counters that drift. Fires once per
    /// turn; needs a `[validation]` command. `false` (default). A/B only.
    pub gate_replan: bool,
    /// EXPERIMENTAL. When `true`, the first time the done-gate blocks, ABANDON
    /// the current (possibly off-path/poisoned) attempt entirely: revert the
    /// whole working tree to the clean baseline (round 0) AND reset the context
    /// to a fresh from-scratch attempt at the task. Tests the detect-and-restart
    /// hypothesis — a stuck/off-path state (run2: 259 rounds, tree broken, edits
    /// misdirected into config/ instead of the consumption in context/) is worse
    /// than a clean start, so scrapping it dominates recovery. Unlike
    /// `gate_context_reset` (context only) this also reverts the TREE, which is
    /// the actual poison. Fires once per turn. `false` (default). A/B only.
    pub gate_restart: bool,
    /// EXPERIMENTAL. Debugger-as-judge: when the done-gate blocks, the fresh-
    /// context read-only debugger DECIDES `SCRAP` vs `CONTINUE` (given the goal).
    /// SCRAP → the loop reverts the tree to the clean baseline + resets context
    /// (the proven restart); CONTINUE → its diagnosis + anchored plan is injected
    /// for the main agent to apply. Unifies the restart trigger, the debugger,
    /// and goal re-anchoring into ONE fresh-eyes decision the loop executes (the
    /// stuck agent never has to decide). Fires once per turn; needs a
    /// `[validation]` command. `true` since 2026-07-15 (bench-unconditional).
    pub debugger_judge: bool,
    /// EXPERIMENTAL. Requires `debugger_judge`. Adds a third option next to
    /// SCRAP/CONTINUE: when a mechanical scan of the revision store finds one
    /// changed file that regressed from a near-clean earlier revision (see
    /// `tools::find_rewind_candidate`), the judge is offered REWIND — revert
    /// JUST that file to the proposed revision, leaving the rest of the tree
    /// untouched. A free-form version (ask the judge to notice AND name the
    /// file+revision itself) scored 0/24 in a tier-1 replay probe; computing
    /// the candidate mechanically and narrowing the ask to accept/reject it
    /// raised that to 13/24. `true` since 2026-07-15 (bench-unconditional).
    pub debugger_judge_rewind: bool,
    /// EXPERIMENTAL. Standalone — does NOT require `reactive_debugger` or
    /// `debugger_judge` (it fires the same underlying sub-agent, which already
    /// defaults to plain diagnostician mode when `debugger_judge` is off).
    /// When `true`, fires the debugger on a trigger point distinct from
    /// `reactive_debugger`'s (the behavioral done-gate): `DEBUGGER_TRIGGER_BLOCKS`
    /// consecutive `plan(action='check')` failures on the SAME step.
    /// Motivated by a 2026-07-04 forensic trace of two 4/6 compaction-bench
    /// runs: the plan-check gate correctly reported a real `unused variable`
    /// warning alongside 14 self-inflicted arity errors (a signature change
    /// left 14 old callers unmigrated), but the primary agent — in its own
    /// accumulated context — fixated on the numerous errors and never revisited
    /// the warning, which was the actual bug. A tier-1 replay probe found no
    /// safe text-formatting fix (declutter/reposition/dedup all failed, or only
    /// "worked" via unsafe silent deletion of real error information); a
    /// fresh-context debugger sub-agent handed the SAME unmodified error text
    /// plus the goal and last action (no accumulated momentum) correctly
    /// targeted the real bug 12/12. Deliberately independent of
    /// `reactive_debugger` (initial A/B ran them coupled — sharing one fire
    /// budget meant the OTHER trigger's condition was hit first in 3 of 4
    /// runs, so the coupled config never actually tested this trigger cleanly)
    /// so it can be A/B'd in isolation. `true` since 2026-07-15 (bench-unconditional).
    pub plan_gate_debugger: bool,
    /// EXPERIMENTAL. T2c frozen-signature stuck detection (gaps 9/10): when
    /// the compiler/test signal (AST state, LSP project errors, failures,
    /// check/gate/shell states) is unchanged for 15 rounds AND 4+ minutes,
    /// append a note to the round's last tool result. Red signal → stuck-note
    /// (broke glimmer's 110-round read loop 8/10 vs control 2/10); green +
    /// every plan step checked → done-note teaching that a reply with no tool
    /// call ends the task (finished the can't-stop dither 10/10 vs 1/10).
    /// Offline trigger eval: 6/6 labeled stuck segments, 0 fires on all three
    /// healthy Laguna runs (scripts/moments/trigger-eval.py, 2026-08-24).
    /// `true` since 2026-08-24: live A/B win on glimmer ({6/6 @ 751s, 6/6 @
    /// 800s} vs baseline {5/6 @ 3406s, 6/6 @ 2735s}), no regression on gemma
    /// (6/6 @ 1000s, 0 fires) or devstral (6/6 @ 2463s, 1 correct Red fire).
    pub stuck_check: bool,
}

/// Agent ceremony level — see `ToolsConfig::ceremony`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum CeremonyMode {
    /// No plan gate, no `PLAN CHECK`, no nudge epicycles, all edit
    /// tools visible, one minimal prompt. Leaner and faster, but the
    /// real docker bench proved it REGRESSES the end-to-end `smoke`
    /// check (Qwen3-Coder-Next: strict 6/6 vs off 5/6 smoke:FAIL, same
    /// HEAD/harness). Opt-in only — the synthetic probe that motivated
    /// it could not measure real multi-step value-threading. See
    /// docs/tiered-agent-design.md §Real-bench refutation.
    Off,
    /// Lean code path (no gate / nudges / phase, like Off) BUT the
    /// prompt strongly *advises* outlining the value-threading steps
    /// before editing. Tests whether decomposition *advice* (not gate
    /// *enforcement*) is the active ingredient for `smoke`. Opt-in,
    /// under evaluation — see docs/tiered-agent-design.md.
    Advise,
    /// Plan-first gating + phase-aware prompt + progress nudges. The
    /// proven-good default: matches Qwen's reliable historical 6/6 and
    /// passes `smoke` (the check that proves the feature works).
    #[default]
    Strict,
}

/// Which edit-tool surface to expose to the outer model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum EditMode {
    /// `edit_file` + inner-model planner.
    Smart,
    /// Fast-mode primitives: `replace_range`, `insert_at`, `revert`, `check`.
    #[default]
    Fast,
}

impl Default for ToolsConfig {
    fn default() -> Self {
        Self {
            context_tools: true,
            lsp_tools: true,
            web_tools: true,
            plan: true,
            scratchpad: true,
            ceremony: CeremonyMode::Strict,
            flat: false,
            edit_mode: EditMode::Fast,
            auto_revert_ast_cascade: true,
            reactive_debugger: true,
            debugger_multifire: true,
            spiral_reset: false,
            // Off: the controlled gemma A/B (2026-06-29) showed OFF is strictly
            // better (6.0 vs 5.67, ~1.6× faster) — the reset causes re-work churn
            // with no reliability gain on this task. See the field doc above.
            gate_context_reset: false,
            revert_to_green: false,
            gate_replan: false,
            gate_restart: false,
            debugger_judge: true,
            debugger_judge_rewind: true,
            plan_gate_debugger: true,
            stuck_check: true,
        }
    }
}
