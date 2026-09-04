//! Skill-step machinery: descending into handoff skills, distilling step
//! instructions, preparing the active step, and the step-judge escalation.

use super::*;

/// Fetch, extract, and descend into `next` (a skill named by a handoff step).
/// Returns true on success. `descend` consumes the invoking step, so on a
/// failed extraction the cursor is left untouched and the caller decides what
/// to do. Shared by name-based and body-based handoff detection.
pub(super) async fn descend_into_skill(
    cursor: &mut crate::cli::commands::agent::skill_cursor::SkillCursor,
    next: &str,
    config: &Config,
    llm_worker: &LlmWorkerHandle,
    cancelled: &Arc<AtomicBool>,
) -> bool {
    use crate::cli::commands::agent::skill_router;
    let Some(entry) = crate::skills::discover(&config.project_root)
        .into_iter()
        .find(|e| e.name == next)
    else {
        return false;
    };
    let Ok(loaded) = crate::skills::load(&entry.path) else {
        return false;
    };
    let steps = skill_router::extract_skill_steps(llm_worker, &loaded.body, cancelled).await;
    if steps.is_empty() {
        return false;
    }
    let dir = entry
        .path
        .parent()
        .unwrap_or(&config.project_root)
        .to_path_buf();
    cursor.descend(next, &dir, steps);
    tui::print_status(&format!("[skills] handoff — descending into '{next}'"));
    true
}

/// Resolve the current step's prose-level handoff: the token matcher proposes
/// candidates, the MODEL decides, the verdict is cached for the life of the
/// step. `None` = stay in the current skill.
///
/// The split is the point. Deterministic matching was asked to answer a
/// question it cannot represent — "does control transfer, or is this a
/// cross-reference?" — and on 2026-08-30 it read
/// `pkg-package-integrate`'s "following the Cluster Setup procedure in the
/// `pkg-package` skill" as a handoff, tore down the integration phase and
/// spent 55 rounds inside a reference document. Every guard it had (last step
/// only, not on the stack, token boundaries, first mention) did its job; the
/// rule simply never modelled the distinction. So retrieval stays here and
/// judgement moves to `classify_handoff`, which sees the same prose plus a
/// menu it cannot invent an answer outside of.
///
/// Leaves the verdict OPEN (uncached) when there is no prose to judge yet, so
/// an undistilled step whose anchor did not resolve gets decided on a later
/// pass instead of being silently recorded as "no handoff".
pub(super) async fn resolve_handoff(
    cursor: &mut crate::cli::commands::agent::skill_cursor::SkillCursor,
    installed: &[String],
    llm_worker: &LlmWorkerHandle,
    cancelled: &Arc<AtomicBool>,
) -> Option<String> {
    use crate::cli::commands::agent::skill_router;
    if cursor.handoff_decided() {
        return cursor.cached_handoff().map(str::to_string);
    }
    let prose = cursor.handoff_prose()?;
    let candidates = cursor.handoff_candidates(&prose, installed);
    if candidates.is_empty() {
        cursor.cache_handoff(String::new());
        return None;
    }
    let name = cursor
        .current()
        .map(|(_, s)| s.name.clone())
        .unwrap_or_default();
    let verdict =
        skill_router::classify_handoff(llm_worker, &name, &prose, &candidates, cancelled).await;
    let menu = candidates.join(", ");
    match &verdict {
        Some(target) => tui::print_status(&format!(
            "[skills] step '{name}' hands off to '{target}' (named: {menu})"
        )),
        None => tui::print_status(&format!(
            "[skills] step '{name}' only references {menu} — no handoff"
        )),
    }
    cursor.cache_handoff(verdict.clone().unwrap_or_default());
    verdict
}

/// Distil the current step's instructions if they aren't cached yet.
pub(super) async fn distill_current(
    cursor: &mut crate::cli::commands::agent::skill_cursor::SkillCursor,
    llm_worker: &LlmWorkerHandle,
    cancelled: &Arc<AtomicBool>,
) {
    use crate::cli::commands::agent::skill_router;
    if cursor.cached().is_some() {
        return;
    }
    let Some(material) = cursor.current_material() else {
        return;
    };
    let name = cursor
        .current()
        .map(|(_, s)| s.name.clone())
        .unwrap_or_default();
    let instructions = skill_router::distill_step(llm_worker, &material, &name, cancelled).await;
    if instructions.trim().is_empty() {
        // An empty distillation would leave the step uncached, and an uncached
        // step is INVISIBLE: no [SKILL STEP] block, no DONE WHEN check, no
        // judge. The raw section is verbose but real — always preferable to
        // parking the cursor on nothing.
        tui::print_status(&format!(
            "[skills] step '{name}' did not distil — using its raw section"
        ));
        cursor.cache(material);
    } else {
        cursor.cache(instructions);
        tui::print_status(&format!("[skills] distilled step '{name}'"));
    }
}

/// Ready the cursor's current step for the LLM call: distil its instructions,
/// resolve a prose-level handoff, generate its DONE WHEN check. Returns true
/// if it descended into a sub-skill — the current step then changed, so the
/// caller must not judge it this round.
///
/// The invariant it exists to hold: **a step the cursor is parked on is
/// visible to the model**. Returning true used to double as an early exit,
/// which broke that — descending left the sub-skill's FIRST step undistilled
/// for the rest of the round, so it rendered as no `[SKILL STEP]` block at
/// all, with no check and no judge. The pkg-mcp e2e run of 2026-08-28 lost
/// `pkg-package-integrate`'s `VerifyInputs` exactly that way: the model saw no
/// step, read that as nothing-to-do, called `skill(done)`, and the cursor
/// advanced past a step it had never shown. Signalling the caller and
/// finishing preparation are separate jobs; only the former is a `return`.
///
/// Called in the normal pre-round position AND again after every advance,
/// because every guard downstream keys on the distilled body: the
/// handoff decision, the check that vetoes a premature
/// `skill(done)`, the judge, and the `[SKILL STEP]` block itself. An advance
/// that left the new step unprepared therefore surfaced a content-free step
/// with every safety valve simultaneously disarmed. That is exactly how the
/// live pkg-mcp e2e lost its integration phase: the judge (which runs LAST
/// here, and is the primary advance driver) moved the cursor onto the build
/// skill's final handoff step after that round's distillation had already
/// run, so the step went out empty, the model dutifully called `done` on it,
/// and the frame popped before the handoff to `pkg-package-integrate` ever
/// got a round.
pub(super) async fn prepare_step(
    cursor: &mut crate::cli::commands::agent::skill_cursor::SkillCursor,
    installed: &[String],
    config: &Config,
    llm_worker: &LlmWorkerHandle,
    cancelled: &Arc<AtomicBool>,
) -> bool {
    use crate::cli::commands::agent::skill_router;
    if !cursor.is_active() {
        return false;
    }
    fn step_name(cursor: &crate::cli::commands::agent::skill_cursor::SkillCursor) -> String {
        cursor
            .current()
            .map(|(_, s)| s.name.clone())
            .unwrap_or_default()
    }
    // Descending lands the cursor on a DIFFERENT step, which then needs the
    // same preparation — so loop rather than return. The handoff decision
    // fires only on a frame's last step and a fresh frame starts at idx 0, so
    // in practice this goes round at most twice; the cap keeps termination a
    // local argument instead of resting on descend()'s on_stack() check.
    const MAX_DESCENTS: usize = 3;
    let mut descended = false;
    for _ in 0..MAX_DESCENTS {
        // Handoff decision BEFORE distillation. `resolve_handoff` reads the
        // step's verbatim source section, so it needs no distilled body — and
        // `descend()` CONSUMES the invoking step, so distilling first spends a
        // 4k-token generation over the whole skill tree on a step that will
        // never get a round. Live 2026-08-30: 'DeployReviewWorkspace' was
        // distilled and descended past in the same breath.
        if let Some(next) = resolve_handoff(cursor, installed, llm_worker, cancelled).await
            && descend_into_skill(cursor, &next, config, llm_worker, cancelled).await
        {
            descended = true;
            continue;
        }
        distill_current(cursor, llm_worker, cancelled).await;
        // Second pass: a step whose anchor could not be located in the source
        // had no prose above, so its verdict was deliberately left open. The
        // distilled body is the fallback.
        if let Some(next) = resolve_handoff(cursor, installed, llm_worker, cancelled).await
            && descend_into_skill(cursor, &next, config, llm_worker, cancelled).await
        {
            descended = true;
            continue;
        }
        break;
    }
    // Per-step completion check: turn the distilled step's DONE WHEN into a
    // read-only shell check (once per step). While the step is active this
    // becomes the effective validation command (see the done-gate below),
    // which fires the debugger stack on projects with no configured
    // task-level check. None when not shell-checkable.
    if !cursor.check_attempted()
        && let Some(instr) = cursor.cached().map(str::to_string)
        && let Some(material) = cursor.current_material()
    {
        let name = step_name(cursor);
        match skill_router::extract_done_when(&instr) {
            Some(done_when) => {
                let check = skill_router::generate_step_check(
                    llm_worker, &material, &name, &done_when, cancelled,
                )
                .await;
                match &check {
                    Some(c) => tui::print_status(&format!("[skills] step '{name}' check: {c}")),
                    None => {
                        tui::print_status(&format!("[skills] step '{name}' not shell-checkable"))
                    }
                }
                cursor.cache_check(check.unwrap_or_default());
            }
            None => cursor.cache_check(String::new()),
        }
    }
    descended
}

/// Surface what an advance left behind. Silent in the normal case; loud when
/// a step was abandoned rather than finished, because that is precisely the
/// thing the old `mark_done`-for-everything path made invisible.
pub(super) fn report_cursor_gaps(cursor: &crate::cli::commands::agent::skill_cursor::SkillCursor) {
    if let Some(skill) = cursor.rewound_into() {
        let step = cursor
            .current()
            .map(|(_, s)| s.name.clone())
            .unwrap_or_default();
        tui::print_status(&format!(
            "[skills] {skill} reached its last step with work outstanding — \
             returning to '{step}' rather than reporting complete"
        ));
    }
    let dropped = cursor.dropped_unfinished();
    if !dropped.is_empty() {
        tui::print_status(&format!(
            "[skills] INCOMPLETE — {} step(s) never finished: {}",
            dropped.len(),
            dropped.join(", ")
        ));
    }
}

/// Escalate the active skill step to the fresh-context step judge and apply
/// its verdict. Shared by the two escalation triggers — the stuck_check Red
/// fire (frozen signals) and the K-th blocked premature finish at the
/// stop-valve — with `trigger` the one preformatted sentence telling the
/// judge which signal tripped. Side effects per verdict: RETRY sets
/// `*force_compact` and resets the step's round counter; ABANDON marks the
/// step abandoned (NOT done). Returns the note to inject, or `None` when no
/// step is active or the judge produced nothing (caller falls back to its
/// plain nudge/note).
#[allow(clippy::too_many_arguments)]
pub(super) async fn step_judge_escalation(
    goal: &str,
    trigger: &str,
    config: &Config,
    llm_worker: &LlmWorkerHandle,
    tool_defs: &[crate::llm::ToolDefinition],
    perms: &Arc<PermissionManager>,
    lsp_client: &Option<Arc<LspClient>>,
    fast_revisions: &Option<Arc<tools::RevisionStore>>,
    fast_baseline_errors: usize,
    cancelled: &Arc<AtomicBool>,
    force_compact: &mut bool,
) -> Option<String> {
    use crate::cli::commands::agent::skill_cursor;
    let mut cursor = skill_cursor::load(config);
    let (skill_name, step_name) = cursor
        .current()
        .map(|(sk, st)| (sk.to_string(), st.name.clone()))?;
    let instructions = cursor
        .cached()
        .map(str::to_string)
        .unwrap_or_else(|| "(the step was never distilled)".to_string());
    let check = cursor.current_check().map(str::to_string);
    tui::print_status(&format!("[step-judge] asking on '{step_name}': {trigger}"));
    let verdict = debugger::run_step_judge(
        goal,
        &skill_name,
        &step_name,
        &instructions,
        check.as_deref(),
        cursor.rounds_on_current(),
        trigger,
        config,
        llm_worker,
        tool_defs,
        perms,
        lsp_client,
        fast_revisions,
        fast_baseline_errors,
        cancelled,
    )
    .await?;
    let (report, label, follow_up) = match verdict {
        debugger::StepVerdict::Continue(r) => (
            r,
            "CONTINUE",
            "The step is doable and the work is close — apply the fix and finish it.".to_string(),
        ),
        debugger::StepVerdict::Retry(r) => {
            *force_compact = true;
            cursor.reset_current_rounds();
            skill_cursor::save(config, &cursor);
            (
                r,
                "RETRY",
                format!(
                    "The current approach is a dead end; the conversation will be compacted next \
                     round. Re-approach '{step_name}' fresh, guided by the report."
                ),
            )
        }
        debugger::StepVerdict::Abandon(r) => {
            cursor.mark_abandoned();
            skill_cursor::save(config, &cursor);
            report_cursor_gaps(&cursor);
            // Abandoning a frame's LAST step rewinds back onto it (its own
            // first abandoned step) for one final pass; a second ABANDON
            // there pops the frame with the gap reported.
            let next = cursor.current().map(|(_, s)| s.name.clone());
            let follow = match next {
                Some(next) if next != step_name => format!(
                    "It is recorded as abandoned, NOT done. Stop working on it and proceed with \
                     the '{next}' step."
                ),
                _ => "It is recorded as abandoned, NOT done. Do not grind on it further — follow \
                      the [SKILL STEP] guidance shown next round, or the remaining task if the \
                      skill has ended."
                    .to_string(),
            };
            (r, "ABANDON", follow)
        }
    };
    tui::print_status(&format!("[step-judge] {label} on '{step_name}'"));
    Some(format!(
        "[Step judge: a fresh-context read-only analyst reviewed the stuck '{step_name}' step \
         and chose {label}. Its report:\n{report}\n{follow_up}]"
    ))
}

/// Regression cover for `prepare_step`'s core invariant: **the step the cursor
/// is parked on when this returns is one the model will actually see.** These
/// drive the real function against a canned LLM endpoint, because the only way
/// to prove the invariant is to let a real descend happen mid-call — that is
/// exactly the moment the 2026-08-28 pkg-mcp e2e lost a step at.
#[cfg(test)]
mod prepare_step_tests {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    use tempfile::TempDir;
    use wiremock::matchers::{body_string_contains, method, path as url_path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::prepare_step;
    use crate::cli::commands::agent::skill_cursor::SkillCursor;
    use crate::cli::commands::agent::skill_router::SkillStep;
    use crate::config::Config;
    use crate::llm::ModelRouter;
    use crate::runtime::LlmWorkerHandle;

    // `skills::discover` also scans the real `~/.ai/skills`, so fixture names
    // have to be ones no installed skill will answer to.
    const PARENT: &str = "probe-handoff-parent";
    const CHILD: &str = "probe-handoff-child";
    const PARENT_DISTILLATION: &str = "INSTRUCTIONS: hand off to the integration skill.";

    /// One-chunk SSE: `LlmWorkerHandle::submit` always sets `stream: true`.
    fn sse(content: &str) -> ResponseTemplate {
        let body = format!(
            "data: {}\n\ndata: [DONE]\n\n",
            serde_json::json!({
                "choices": [{"delta": {"content": content}, "finish_reason": null}]
            })
        );
        ResponseTemplate::new(200).set_body_raw(body, "text/event-stream")
    }

    fn write_skill(root: &Path, name: &str, body: &str) -> PathBuf {
        let dir = root.join(".ai").join("skills").join(name);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("SKILL.md"), body).unwrap();
        dir
    }

    /// Both skills on disk, with the cursor parked on the parent's final step
    /// — whose prose names the child. The shape of the real
    /// `pkg-package-build` → `pkg-package-integrate` moment.
    fn fixture(root: &Path) -> SkillCursor {
        let parent_dir = write_skill(
            root,
            PARENT,
            &format!(
                "# Parent\n\n## Assemble\nAssemble the package.\n\n\
                 ## EnterIntegrationPhase\nThe build phase is complete. \
                 Continue with the {CHILD} skill.\n"
            ),
        );
        write_skill(
            root,
            CHILD,
            "# Child\n\n## VerifyInputs\nConfirm the package directory exists.\n\n\
             ## ConfigureNetworking\nExpose the service.\n",
        );
        let step = |name: &str| SkillStep {
            name: name.to_string(),
            anchor: format!("## {name}"),
        };
        let mut cursor = SkillCursor::default();
        cursor.push_skill(
            PARENT,
            &parent_dir,
            vec![step("Assemble"), step("EnterIntegrationPhase")],
        );
        cursor.mark_done();
        assert_eq!(
            cursor.current().map(|(_, s)| s.name.as_str()),
            Some("EnterIntegrationPhase")
        );
        assert!(
            cursor.cached().is_none(),
            "the replayed moment is an undistilled final step"
        );
        cursor
    }

    /// Answers every LLM call this fixture provokes: the handoff classifier on
    /// the parent's final step, step extraction for the child, and the child's
    /// first distillation — `verify_inputs`, the value under test. That
    /// distillation carries no `DONE WHEN`, so no check is generated and the
    /// mocked surface stays exactly these three calls.
    ///
    /// `verdict` is what the classifier answers for the parent's final step —
    /// `CHILD` for a real handoff, `"NONE"` for a step that only references
    /// the other skill.
    ///
    /// The parent's own distillation is mounted with an exact call count,
    /// verified when the server drops at the end of `run`. On a handoff it is
    /// 0: the decision reads the step's verbatim source, so it runs BEFORE
    /// distillation, and `descend()` then consumes the invoking step —
    /// distilling it would be pure waste. On `NONE` it is 1: the step keeps
    /// its round, so it must be visible.
    async fn mock_llm(server: &MockServer, verdict: &str, verify_inputs: &str) {
        let mount = |matcher: &'static str, reply: ResponseTemplate| {
            Mock::given(method("POST"))
                .and(url_path("/v1/chat/completions"))
                .and(body_string_contains(matcher))
                .respond_with(reply)
        };
        mount("Handoff target?", sse(verdict)).mount(server).await;
        mount(
            "ordered execution checklist",
            sse(&serde_json::json!([
                {"step": "VerifyInputs", "anchor": "## VerifyInputs"},
                {"step": "ConfigureNetworking", "anchor": "## ConfigureNetworking"},
            ])
            .to_string()),
        )
        .mount(server)
        .await;
        mount(
            "Distill the step: 'EnterIntegrationPhase'",
            sse(PARENT_DISTILLATION),
        )
        .expect(u64::from(verdict.trim() != CHILD))
        .mount(server)
        .await;
        mount("Distill the step: 'VerifyInputs'", sse(verify_inputs))
            .mount(server)
            .await;
    }

    fn config_for(root: &Path, endpoint: &str) -> Config {
        let mut config = Config::default();
        config.project_root = root.to_path_buf();
        config.model.endpoint = endpoint.to_string();
        config.model.provider = "openai-compatible".to_string();
        config.model.max_retries = 1;
        config
    }

    /// Run the real `prepare_step` over the fixture, with the handoff
    /// classifier answered by `verdict` and the child's first distillation by
    /// `verify_inputs`.
    async fn run(verdict: &str, verify_inputs: &str) -> (TempDir, SkillCursor, bool) {
        let server = MockServer::start().await;
        mock_llm(&server, verdict, verify_inputs).await;
        let temp = TempDir::new().unwrap();
        let mut cursor = fixture(temp.path());
        let config = config_for(temp.path(), &server.uri());
        let worker = LlmWorkerHandle::new(Arc::new(ModelRouter::new(&config)), 1);
        let installed = [PARENT.to_string(), CHILD.to_string()];
        let descended = prepare_step(
            &mut cursor,
            &installed,
            &config,
            &worker,
            &Arc::new(AtomicBool::new(false)),
        )
        .await;
        (temp, cursor, descended)
    }

    #[tokio::test]
    async fn descending_leaves_the_sub_skills_first_step_distilled() {
        let (_temp, cursor, descended) =
            run(CHILD, "INSTRUCTIONS: confirm the package directory exists.").await;

        assert!(descended, "the parent's final step hands off to the child");
        assert_eq!(
            cursor
                .current()
                .map(|(sk, s)| (sk.to_string(), s.name.clone())),
            Some((CHILD.to_string(), "VerifyInputs".to_string())),
            "the cursor must land on the child's FIRST step"
        );
        // The regression. This was None: `return true` on a successful
        // descend doubled as an early exit, so the step the cursor now sat on
        // never got distilled. An undistilled step renders as no [SKILL STEP]
        // block, with no check and no judge — the model saw nothing to do,
        // called skill(done), and the step was consumed unseen.
        assert!(
            cursor
                .cached()
                .unwrap_or_default()
                .contains("confirm the package directory"),
            "the child's first step must be distilled before prepare_step returns, got {:?}",
            cursor.cached()
        );
    }

    #[tokio::test]
    async fn an_empty_distillation_falls_back_to_the_raw_section() {
        // `distill_step` returns "" on any LLM failure, and caching nothing
        // would reproduce the invisible-step bug by a second route. The raw
        // material is verbose but real, so it stands in.
        let (_temp, cursor, _) = run(CHILD, "").await;

        assert!(
            cursor
                .cached()
                .unwrap_or_default()
                .contains("Confirm the package directory exists"),
            "an empty distillation must fall back to the step's raw material, got {:?}",
            cursor.cached()
        );
    }

    #[tokio::test]
    async fn a_step_the_classifier_clears_keeps_its_round() {
        // The mirror failure, from the pkg-mcp e2e run of 2026-08-30. The
        // deterministic matcher read `pkg-package-integrate`'s
        // `DeployReviewWorkspace` — "following the Cluster Setup procedure in
        // the `pkg-package` skill" — as a handoff and descended into a
        // reference document with no steps. 55 of that run's 305 rounds went
        // there, and `DeployReviewWorkspace`, consumed by the descend, never
        // got a round or a DONE WHEN check. A `NONE` verdict has to leave the
        // step exactly where it is, distilled and checkable.
        let (_temp, cursor, descended) = run("NONE", "unreached").await;

        assert!(!descended, "a reference is not a handoff");
        assert_eq!(
            cursor
                .current()
                .map(|(sk, s)| (sk.to_string(), s.name.clone())),
            Some((PARENT.to_string(), "EnterIntegrationPhase".to_string())),
            "the cursor must stay on the step that only referenced the skill"
        );
        assert_eq!(
            cursor.cached(),
            Some(PARENT_DISTILLATION),
            "a step that keeps its round must be distilled"
        );
    }
}
