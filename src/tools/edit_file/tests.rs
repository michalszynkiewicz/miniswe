use super::apply::{
    find_all_exact_line_matches, find_all_fuzzy_line_matches, find_all_ws_tolerant_line_matches,
    pick_best_candidate,
};
use super::parse::{
    MAX_FAILED_REASON_CHARS, looks_like_complete, parse_edit_plan, parse_failed,
    parse_needs_clarification, parse_patch, partition_overlapping_steps, strip_code_fences,
};
use super::*;
#[test]
fn retry_feedback_adds_signature_guidance_for_lsp_arity_errors() {
    let feedback = build_retry_feedback(
        "LSP diagnostics worsened for src/main.rs: 0 -> 3 error(s)\nsrc/main.rs:31:79: error: expected 4 arguments, found 5",
        None,
    );

    assert!(feedback.contains("Do not repeat the same patch shape"));
    assert!(!feedback.contains("Current known callee signatures"));
}

fn smart_step(start: usize, end: usize) -> EditPlanStep {
    EditPlanStep::SmartEdit(EditRegion {
        start,
        end,
        task: format!("dummy task L{start}-L{end}"),
    })
}

#[test]
fn partition_overlapping_steps_keeps_first_in_source_order() {
    // Three steps in emission order: middle one overlaps the first.
    // The first (by source line) wins; the overlapper is reported as
    // dropped, the third non-overlapping step is kept.
    let steps = vec![
        smart_step(5, 15),
        smart_step(10, 20), // overlaps step at L5-L15
        smart_step(25, 35),
    ];
    let (kept, dropped) = partition_overlapping_steps(steps);
    assert_eq!(kept.len(), 2);
    assert_eq!(kept[0].start_line(), 5);
    assert_eq!(kept[1].start_line(), 25);
    assert_eq!(dropped.len(), 1);
    assert_eq!(dropped[0].step.start_line(), 10);
    assert!(
        dropped[0].reason.contains("L5-L15"),
        "reason should reference the conflicting kept step, got {:?}",
        dropped[0].reason
    );
}

#[test]
fn partition_overlapping_steps_treats_shared_endpoint_as_overlap() {
    // L10-L20 and L20-L30 share line 20 — that counts as overlap so
    // the planner can't smuggle two edits through a single shared
    // endpoint.
    let steps = vec![smart_step(10, 20), smart_step(20, 30)];
    let (kept, dropped) = partition_overlapping_steps(steps);
    assert_eq!(kept.len(), 1);
    assert_eq!(kept[0].start_line(), 10);
    assert_eq!(dropped.len(), 1);
    assert_eq!(dropped[0].step.start_line(), 20);
}

#[test]
fn partition_overlapping_steps_keeps_disjoint_steps() {
    let steps = vec![smart_step(1, 5), smart_step(10, 15), smart_step(20, 25)];
    let (kept, dropped) = partition_overlapping_steps(steps);
    assert_eq!(kept.len(), 3);
    assert!(dropped.is_empty());
}

#[test]
fn gate_truncation_allows_small_files() {
    // Below LARGE_TRUNCATION_MIN_LINES, the guard never fires — even a
    // "delete everything except one line" edit is allowed (the empty-file
    // check in validate_candidate still catches full deletion).
    let original: String = (0..20).map(|i| format!("line {i}\n")).collect();
    let candidate = "line 0\n";
    assert!(gate_truncation("small.rs", &original, candidate, None).is_ok());
}

#[test]
fn gate_truncation_allows_partial_shrink_above_half() {
    // Shrinking from 100 to 60 lines keeps us above the half threshold,
    // so the guard stays quiet.
    let original: String = (0..100).map(|i| format!("line {i}\n")).collect();
    let candidate: String = (0..60).map(|i| format!("line {i}\n")).collect();
    assert!(gate_truncation("big.rs", &original, &candidate, None).is_ok());
}

#[test]
fn gate_truncation_rejects_large_cut_without_perms() {
    // 100 -> 40 lines on a >50 line file is below half; with no perms
    // available, the guard rejects (preserving the pre-existing behavior
    // for tests and headless runs).
    let original: String = (0..100).map(|i| format!("line {i}\n")).collect();
    let candidate: String = (0..40).map(|i| format!("line {i}\n")).collect();
    let err = gate_truncation("big.rs", &original, &candidate, None).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("truncates"), "unexpected error: {msg}");
    assert!(msg.contains("100"), "should name old line count: {msg}");
    assert!(msg.contains("40"), "should name new line count: {msg}");
}

#[test]
fn gate_truncation_rejects_large_cut_in_headless_mode() {
    // With an auto-approve PermissionManager (headless), the guard
    // rejects without prompting — same as passing None.
    let config = Config::default();
    let perms = PermissionManager::headless(&config);
    let original: String = (0..100).map(|i| format!("line {i}\n")).collect();
    let candidate: String = (0..10).map(|i| format!("line {i}\n")).collect();
    let err = gate_truncation("big.rs", &original, &candidate, Some(&perms)).unwrap_err();
    assert!(err.to_string().contains("truncates"));
}

fn s(lines: &[&str]) -> Vec<String> {
    lines.iter().map(|l| l.to_string()).collect()
}

#[test]
fn find_all_exact_line_matches_finds_single_hit() {
    let content = "alpha\nbeta\ngamma\n";
    let hits = find_all_exact_line_matches(content, &s(&["beta", "gamma"]));
    assert_eq!(hits, vec![(2, 3)]);
}

#[test]
fn find_all_exact_line_matches_finds_multiple_hits() {
    let content = "foo\nfoo\nfoo\n";
    let hits = find_all_exact_line_matches(content, &s(&["foo"]));
    assert_eq!(hits, vec![(1, 1), (2, 2), (3, 3)]);
}

#[test]
fn find_all_exact_line_matches_returns_empty_when_needle_longer_than_haystack() {
    let content = "one\n";
    let hits = find_all_exact_line_matches(content, &s(&["one", "two"]));
    assert!(hits.is_empty());
}

#[test]
fn find_all_exact_line_matches_empty_needle_returns_empty() {
    let content = "one\ntwo\n";
    let hits = find_all_exact_line_matches(content, &[]);
    assert!(hits.is_empty());
}

#[test]
fn find_all_ws_tolerant_ignores_indentation_drift() {
    // Planner used spaces, file uses tabs.
    let content = "\tif x {\n\t\treturn 1;\n\t}\n";
    let hits =
        find_all_ws_tolerant_line_matches(content, &s(&["if x {", "    return 1;", "}"]), &[]);
    assert_eq!(hits, vec![(1, 3)]);
}

#[test]
fn find_all_ws_tolerant_skips_excluded_ranges() {
    // Byte-exact hit already present — whitespace search should not
    // re-add the same range.
    let content = "foo\nbar\n";
    let exclude = vec![(1, 1)];
    let hits = find_all_ws_tolerant_line_matches(content, &s(&["foo"]), &exclude);
    assert!(hits.is_empty());
}

#[test]
fn find_all_ws_tolerant_rejects_all_blank_old() {
    // An OLD block that squashes to nothing would otherwise match
    // every stretch of blank lines.
    let content = "\n\n\nfoo\n\n\n";
    let hits = find_all_ws_tolerant_line_matches(content, &s(&["   ", "\t"]), &[]);
    assert!(hits.is_empty());
}

#[test]
fn pick_best_candidate_prefers_overlap() {
    // Two candidates: (5, 10) overlaps declared (8, 12); (20, 25)
    // does not. Overlap wins even if it's further from the endpoints.
    let candidates = vec![(5, 10), (20, 25)];
    let picked = pick_best_candidate(&candidates, 8, 12).unwrap();
    assert_eq!(picked, (5, 10));
}

#[test]
fn pick_best_candidate_picks_nearest_when_no_overlap() {
    let candidates = vec![(1, 5), (30, 35), (100, 105)];
    let picked = pick_best_candidate(&candidates, 28, 34).unwrap();
    assert_eq!(picked, (30, 35));
}

#[test]
fn pick_best_candidate_tie_break_prefers_earlier_insertion() {
    // Two candidates equidistant from the scope — the first one
    // inserted wins. This lets the caller front-load byte-exact
    // hits ahead of whitespace-tolerant ones.
    let candidates = vec![(10, 14), (20, 24)];
    let picked = pick_best_candidate(&candidates, 15, 19).unwrap();
    assert_eq!(picked, (10, 14));
}

#[test]
fn pick_best_candidate_returns_none_for_empty_list() {
    assert_eq!(pick_best_candidate(&[], 1, 5), None);
}

#[test]
fn extract_line_ranges_parses_range_with_l_prefix() {
    let ranges = extract_line_ranges_from_note("L283-290 — assemble fn signature");
    assert_eq!(ranges, vec![(283, 290)]);
}

#[test]
fn extract_line_ranges_parses_bare_range() {
    let ranges = extract_line_ranges_from_note("132-138 — context::assemble call");
    assert_eq!(ranges, vec![(132, 138)]);
}

#[test]
fn extract_line_ranges_parses_single_line_to_window() {
    // Single line reference expands to ±2 window
    let ranges = extract_line_ranges_from_note("L41 — run function signature");
    assert_eq!(ranges, vec![(39, 43)]);
}

#[test]
fn extract_line_ranges_parses_multiple_ranges() {
    let ranges = extract_line_ranges_from_note("L10-20 foo, L30-40 bar");
    assert_eq!(ranges, vec![(10, 20), (30, 40)]);
}

#[test]
fn extract_line_ranges_ignores_mid_word_numbers() {
    // "v2" or "utf8" shouldn't parse as line references
    let ranges = extract_line_ranges_from_note("uses utf8 encoding and v2 protocol");
    assert!(ranges.is_empty());
}

#[test]
fn extract_line_ranges_single_line_near_start_clamps() {
    // Line 1 expanded ±2 should clamp start to 1
    let ranges = extract_line_ranges_from_note("L1 — first line");
    assert_eq!(ranges, vec![(1, 3)]);
}

#[test]
fn find_all_fuzzy_rescues_single_char_typo() {
    // Typo'd identifier: `callback` vs `calback`. Byte-exact and
    // whitespace-normalized both reject; fuzzy accepts.
    let content = "fn handler(callback: F) {\n    callback();\n}\n";
    let hits = find_all_fuzzy_line_matches(content, &s(&["    calback();"]), &[]);
    assert_eq!(hits, vec![(2, 2)]);
}

#[test]
fn find_all_fuzzy_rescues_multiline_near_match() {
    // Two lines where one has a small typo. Both lines clear the
    // per-line floor and the block average clears the block floor.
    let content =
        "let mut config = Config::default();\nconfig.timeout = 30;\nconfig.retries = 5;\n";
    let hits = find_all_fuzzy_line_matches(
        content,
        &s(&["let mut config = Config::defalt();", "config.timeout = 30;"]),
        &[],
    );
    assert_eq!(hits, vec![(1, 2)]);
}

#[test]
fn find_all_fuzzy_rejects_block_below_threshold() {
    // Block average similarity is too low — the second line is
    // entirely different. Fuzzy should decline rather than surface a
    // noisy match.
    let content = "fn main() {\n    println!(\"hello\");\n}\n";
    let hits = find_all_fuzzy_line_matches(
        content,
        &s(&["fn main() {", "    assert_eq!(foo, bar_baz);"]),
        &[],
    );
    assert!(hits.is_empty());
}

#[test]
fn find_all_fuzzy_rejects_trivially_short_old() {
    // Single-line OLDs shorter than 5 non-whitespace chars are too
    // noisy to fuzzy-match reliably; the function bails early.
    let content = "x = 1;\ny = 2;\nz = 3;\n";
    let hits = find_all_fuzzy_line_matches(content, &s(&["x=1"]), &[]);
    assert!(hits.is_empty());
}

#[test]
fn find_all_fuzzy_skips_excluded_ranges() {
    let content = "callback(foo);\ncallback(foo);\n";
    // Pretend the first line was already matched exactly elsewhere;
    // fuzzy should still return the second.
    let hits = find_all_fuzzy_line_matches(content, &s(&["calback(foo);"]), &[(1, 1)]);
    assert_eq!(hits, vec![(2, 2)]);
}

#[test]
fn parse_preplan_window_response_collects_note_lines() {
    let response = parse_preplan_window_response(
        "NOTE provider loop spans L10-L20\nNOTE: assemble fn signature at L283\n",
    );
    assert_eq!(
        response.notes,
        vec![
            "provider loop spans L10-L20".to_string(),
            "assemble fn signature at L283".to_string(),
        ],
    );
    assert!(response.commands.is_empty());
}

#[test]
fn parse_preplan_window_response_collects_search_and_read_commands() {
    // The windowed pre-plan pass no longer has a separate recon phase —
    // the model queues SEARCH/READ requests right alongside the notes
    // it emits, and they get batch-executed before finalize.
    let response = parse_preplan_window_response(
        "NOTE callers live at L10-L20\n\
             SEARCH: assemble(\n\
             READ: 280-300\n",
    );
    assert_eq!(response.notes, vec!["callers live at L10-L20".to_string()]);
    assert_eq!(
        response.commands,
        vec![
            InspectionCommand::Search("assemble(".into()),
            InspectionCommand::Read {
                start: 280,
                end: 300
            },
        ],
    );
}

#[test]
fn parse_preplan_window_response_drops_invalid_read_ranges() {
    // Malformed READ ranges (start=0, end<start, non-numeric) are
    // silently dropped rather than erroring out so the model doesn't
    // crash the pipeline with a typo.
    let response =
        parse_preplan_window_response("READ: 0-5\nREAD: 10-3\nREAD: bad\nNOTE still here\n");
    assert!(response.commands.is_empty());
    assert_eq!(response.notes, vec!["still here".to_string()]);
}

#[test]
fn parse_preplan_window_response_silently_ignores_unknown_lines() {
    // Stray control words, half-formed edit-plan blocks, and leftover
    // DONE terminators from the old recon phase all get dropped
    // without erroring out.
    let response = parse_preplan_window_response(
        "NOTE looks good\n\
             SMART_EDIT\n\
             REGION 1 5\n\
             TASK: do thing\n\
             END\n\
             NO_CHANGES\n\
             DONE\n",
    );
    assert_eq!(response.notes, vec!["looks good".to_string()]);
    assert!(response.commands.is_empty());
}

#[test]
fn parse_preplan_window_response_accepts_empty_response() {
    // The model may have nothing to add for a slice — that's fine.
    let response = parse_preplan_window_response("");
    assert!(response.notes.is_empty());
    assert!(response.commands.is_empty());
    assert!(response.clarification.is_none());
}

#[test]
fn parse_preplan_window_response_captures_clarification() {
    // A bare NEEDS_CLARIFICATION line sets the clarification field and
    // does not pollute notes/commands.
    let response = parse_preplan_window_response(
        "NEEDS_CLARIFICATION: which run() function should accept the override?\n",
    );
    assert_eq!(
        response.clarification,
        Some("which run() function should accept the override?".to_string())
    );
    assert!(response.notes.is_empty());
    assert!(response.commands.is_empty());
}

#[test]
fn parse_preplan_window_response_clarification_coexists_with_notes() {
    // A confused model might emit a NOTE and also hedge with a
    // clarification — we keep both, and the caller short-circuits on
    // the clarification regardless.
    let response = parse_preplan_window_response(
        "NOTE run() at L12\nNEEDS_CLARIFICATION: which module owns this?\n",
    );
    assert_eq!(response.notes, vec!["run() at L12".to_string()]);
    assert_eq!(
        response.clarification,
        Some("which module owns this?".to_string())
    );
}

#[test]
fn execute_inspection_commands_executes_search_and_read() {
    let content = "fn one() {}\nfn two() {}\nfn three() {}\n";
    let commands = vec![
        InspectionCommand::Search("two".into()),
        InspectionCommand::Read { start: 1, end: 2 },
    ];
    let mut counters = InspectionCounters {
        search_count: 0,
        read_count: 0,
        max_reads: 6,
    };
    let mut extra = String::new();
    execute_inspection_commands(
        content,
        &commands,
        &mut counters,
        &mut extra,
        "test.rs",
        None,
    )
    .unwrap();
    assert_eq!(counters.search_count, 1);
    assert_eq!(counters.read_count, 1);
    assert!(extra.contains("SEARCH_RESULT query=`two`"));
    assert!(extra.contains("READ_RESULT range=L1-L2"));
    assert!(extra.contains("fn one()"));
}

#[test]
fn execute_inspection_commands_respects_read_cap() {
    // The per-edit READ cap is enforced across all queued commands in
    // the single batch pass.
    let content = "x\n";
    let mut counters = InspectionCounters {
        search_count: 0,
        read_count: 0,
        max_reads: 1,
    };
    let mut extra = String::new();
    execute_inspection_commands(
        content,
        &[
            InspectionCommand::Read { start: 1, end: 1 },
            InspectionCommand::Read { start: 1, end: 1 },
        ],
        &mut counters,
        &mut extra,
        "test.rs",
        None,
    )
    .unwrap();
    assert_eq!(counters.read_count, 1, "second read should be capped");
    assert!(extra.contains("READ 1-1 skipped: per-edit limit"));
}

#[test]
fn execute_inspection_commands_drops_overflow_with_inline_note() {
    let content = "x\n";
    let mut commands = Vec::new();
    for _ in 0..(MAX_PREPLAN_SEARCHES + 2) {
        commands.push(InspectionCommand::Search("x".into()));
    }
    let mut counters = InspectionCounters {
        search_count: 0,
        read_count: 0,
        max_reads: 6,
    };
    let mut extra = String::new();
    execute_inspection_commands(
        content,
        &commands,
        &mut counters,
        &mut extra,
        "test.rs",
        None,
    )
    .unwrap();
    assert_eq!(counters.search_count, MAX_PREPLAN_SEARCHES);
    assert!(extra.contains("SEARCH `x` skipped: per-edit limit"));
}

#[test]
fn inspection_results_are_labeled() {
    let mut extra = String::new();
    append_inspection_result(
        &mut extra,
        "SEARCH_RESULT query=`foo`",
        "SEARCH RESULT for `foo`: 1 hit",
    );
    append_inspection_result(
        &mut extra,
        "READ_RESULT range=L10-L12",
        "READ RESULT L10-L12:\n  10│x",
    );

    assert!(extra.contains("SEARCH_RESULT query=`foo`"));
    assert!(extra.contains("READ_RESULT range=L10-L12"));
    assert!(!extra.contains("Inspection result:"));
}

#[test]
fn parse_needs_clarification_accepts_keyword_with_question() {
    let r = parse_needs_clarification(
        "NEEDS_CLARIFICATION: which of the two run() functions should accept the override?\n",
    );
    assert_eq!(
        r,
        Some("which of the two run() functions should accept the override?".to_string())
    );
}

#[test]
fn parse_needs_clarification_accepts_keyword_with_whitespace_question() {
    let r = parse_needs_clarification(
        "NEEDS_CLARIFICATION   what should the new parameter default to?\n",
    );
    assert_eq!(
        r,
        Some("what should the new parameter default to?".to_string())
    );
}

#[test]
fn parse_needs_clarification_accepts_keyword_alone() {
    let r = parse_needs_clarification("NEEDS_CLARIFICATION");
    assert_eq!(r, Some(String::new()));
}

#[test]
fn parse_needs_clarification_tolerates_leading_whitespace() {
    let r = parse_needs_clarification("  \n  NEEDS_CLARIFICATION: which field?\n");
    assert_eq!(r, Some("which field?".to_string()));
}

#[test]
fn parse_needs_clarification_rejects_lookalikes() {
    // Don't false-match on a word that just starts with NEEDS_CLARIFICATION.
    assert_eq!(parse_needs_clarification("NEEDS_CLARIFICATIONS"), None);
    assert_eq!(
        parse_needs_clarification("NEEDS_CLARIFICATIONAL: foo"),
        None
    );
}

#[test]
fn parse_needs_clarification_rejects_unrelated_text() {
    assert_eq!(parse_needs_clarification("NO_CHANGES"), None);
    assert_eq!(
        parse_needs_clarification("LITERAL_REPLACE\nSCOPE 1 1\n"),
        None
    );
    assert_eq!(parse_needs_clarification(""), None);
}

#[test]
fn parse_needs_clarification_keeps_only_first_line() {
    // The sentinel is single-line by contract — any trailing lines
    // (e.g. stray tokens from a confused model) are dropped.
    let r = parse_needs_clarification("NEEDS_CLARIFICATION: which file?\nNOTE stray");
    assert_eq!(r, Some("which file?".to_string()));
}

#[test]
fn looks_like_complete_accepts_bare_sentinel() {
    assert!(looks_like_complete("COMPLETE"));
    // Legacy spelling preserved for back-compat during rollout.
    assert!(looks_like_complete("NO_CHANGES"));
}

#[test]
fn looks_like_complete_tolerates_surrounding_whitespace() {
    assert!(looks_like_complete("  COMPLETE  "));
    assert!(looks_like_complete("\n\nCOMPLETE\n"));
    assert!(looks_like_complete("\tCOMPLETE\n\n"));
}

#[test]
fn looks_like_complete_tolerates_code_fences() {
    // Smaller models sometimes wrap everything in a fence, and
    // `strip_code_fences` already handles that shape — we just need
    // to make sure the sentinel recognizer composes with it.
    assert!(looks_like_complete("```\nCOMPLETE\n```"));
    assert!(looks_like_complete("```text\nCOMPLETE\n```"));
}

#[test]
fn looks_like_complete_rejects_empty() {
    // Empty response is the pathology case, not a verdict — it
    // must NOT be treated as COMPLETE.
    assert!(!looks_like_complete(""));
    assert!(!looks_like_complete("   "));
    assert!(!looks_like_complete("\n\n"));
}

#[test]
fn looks_like_complete_rejects_content_alongside_sentinel() {
    assert!(!looks_like_complete("COMPLETE\nLITERAL_REPLACE\nSCOPE 1 1"));
    assert!(!looks_like_complete("LITERAL_REPLACE\nSCOPE 1 1\nCOMPLETE"));
    assert!(!looks_like_complete("NEEDS_CLARIFICATION: which field?"));
}

#[test]
fn looks_like_complete_rejects_lookalikes() {
    assert!(!looks_like_complete("COMPLETED"));
    assert!(!looks_like_complete("COMPLETELY_DONE"));
    assert!(!looks_like_complete("complete")); // case-sensitive on purpose
}

#[test]
fn parse_failed_accepts_reason_with_colon() {
    let r = parse_failed("FAILED: signature change breaks 3 callers in tests/");
    assert_eq!(
        r,
        Some("signature change breaks 3 callers in tests/".to_string())
    );
}

#[test]
fn parse_failed_accepts_reason_with_whitespace() {
    let r = parse_failed("FAILED  region too large to replace verbatim");
    assert_eq!(r, Some("region too large to replace verbatim".to_string()));
}

#[test]
fn parse_failed_accepts_bare_keyword() {
    // Reasonless failure still parses, caller substitutes placeholder.
    assert_eq!(parse_failed("FAILED"), Some(String::new()));
}

#[test]
fn parse_failed_rejects_lookalikes() {
    assert!(parse_failed("FAILURE: whatever").is_none());
    assert!(parse_failed("failed: lower").is_none());
}

#[test]
fn parse_failed_keeps_only_first_line() {
    let r = parse_failed("FAILED: top line\nsecond line");
    assert_eq!(r, Some("top line".to_string()));
}

#[test]
fn parse_failed_caps_reason_length() {
    let long: String = "x".repeat(MAX_FAILED_REASON_CHARS + 50);
    let input = format!("FAILED: {long}");
    let r = parse_failed(&input).expect("should parse");
    // The returned reason is capped + ellipsis, so <= max+1 chars.
    assert!(r.chars().count() <= MAX_FAILED_REASON_CHARS + 1);
    assert!(r.ends_with('…'));
}

#[test]
fn format_repair_context_first_step_failed_renders_empty_completed() {
    let ctx = RepairContext {
        previous_plan: vec![EditPlanStep::SmartEdit(EditRegion {
            start: 10,
            end: 20,
            task: "rewrite header".into(),
        })],
        completed_steps: Vec::new(),
        failed_step: Some(EditPlanStep::SmartEdit(EditRegion {
            start: 10,
            end: 20,
            task: "rewrite header".into(),
        })),
        failure_reason: "region missing anchor".into(),
        lsp_regression: None,
        cleanly_applied: false,
    };

    let block = format_repair_context(&ctx);
    assert!(block.contains("The previous iteration failed."));
    assert!(block.contains("Previous edit plan (as tried):"));
    assert!(block.contains("rewrite header"));
    assert!(block.contains(
            "Steps that succeeded and have ALREADY been applied to the file shown below:\n(none — the first step failed, file is unchanged from the initial state)"
        ));
    assert!(block.contains("Step that FAILED:"));
    assert!(block.contains("Failure reason:\nregion missing anchor"));
    assert!(block.contains("`FAILED: <reason>`"));
}

#[test]
fn format_repair_context_validation_failure_has_no_failed_step() {
    let step_a = EditPlanStep::SmartEdit(EditRegion {
        start: 10,
        end: 20,
        task: "first step".into(),
    });
    let step_b = EditPlanStep::SmartEdit(EditRegion {
        start: 30,
        end: 40,
        task: "second step".into(),
    });
    let ctx = RepairContext {
        previous_plan: vec![step_a.clone(), step_b.clone()],
        completed_steps: vec![step_a, step_b],
        failed_step: None,
        failure_reason: "LSP diagnostics worsened: 0 -> 4 errors".into(),
        lsp_regression: None,
        cleanly_applied: false,
    };

    let block = format_repair_context(&ctx);
    // Both steps appear in completed.
    assert!(block.contains("first step"));
    assert!(block.contains("second step"));
    assert!(block.contains(
            "Step that FAILED:\n(no individual step failed; the plan executed in full but post-validation rejected the result"
        ));
    assert!(block.contains("LSP diagnostics worsened"));
}

#[test]
fn format_repair_context_partial_success_lists_completed_and_failed() {
    let completed_step = EditPlanStep::LiteralReplace {
        scope_start: 5,
        scope_end: 10,
        all: false,
        old: vec!["foo".into()],
        new: vec!["bar".into()],
    };
    let failed_step = EditPlanStep::SmartEdit(EditRegion {
        start: 30,
        end: 40,
        task: "rewrite second region".into(),
    });
    let ctx = RepairContext {
        previous_plan: vec![completed_step.clone(), failed_step.clone()],
        completed_steps: vec![completed_step],
        failed_step: Some(failed_step),
        failure_reason: "smart edit returned no diff".into(),
        lsp_regression: None,
        cleanly_applied: false,
    };

    let block = format_repair_context(&ctx);
    // Completed section names the literal replace.
    assert!(block.contains("LITERAL_REPLACE"));
    assert!(block.contains("OLD:\nfoo"));
    // Failed section names the smart edit.
    assert!(block.contains("rewrite second region"));
    assert!(block.contains("smart edit returned no diff"));
    // The "first step failed" stub must NOT appear when partial success exists.
    assert!(!block.contains("(none — the first step failed"));
}

#[test]
fn format_repair_context_cleanly_applied_uses_soft_language() {
    let step = EditPlanStep::LiteralReplace {
        scope_start: 5,
        scope_end: 10,
        all: false,
        old: vec!["old text".into()],
        new: vec!["new text".into()],
    };
    let ctx = RepairContext {
        previous_plan: vec![step.clone()],
        completed_steps: vec![step],
        failed_step: None,
        failure_reason: String::new(),
        lsp_regression: None,
        cleanly_applied: true,
    };

    let block = format_repair_context(&ctx);
    // Success framing — no "failed" language.
    assert!(!block.contains("failed"));
    assert!(!block.contains("FAILED") || block.contains("`FAILED:"));
    assert!(block.contains("applied the following steps cleanly"));
    assert!(block.contains("`COMPLETE`"));
    assert!(block.contains("LITERAL_REPLACE"));
}

#[test]
fn format_repair_steps_for_window_shows_overlapping_steps() {
    let completed = EditPlanStep::LiteralReplace {
        scope_start: 17,
        scope_end: 17,
        all: true,
        old: vec!["    let x = assemble(&config, None);".into()],
        new: vec!["    let x = assemble(&config, None, None);".into()],
    };
    let failed = EditPlanStep::LiteralReplace {
        scope_start: 54,
        scope_end: 54,
        all: true,
        old: vec!["    let y = assemble(&config, None);".into()],
        new: vec!["    let y = assemble(&config, None, None);".into()],
    };
    let ctx = RepairContext {
        previous_plan: vec![completed.clone(), failed.clone()],
        completed_steps: vec![completed],
        failed_step: Some(failed),
        failure_reason: "literal OLD block was not found".into(),
        lsp_regression: None,
        cleanly_applied: false,
    };

    // Window 1-30: only the completed step overlaps
    let block = format_repair_steps_for_window(&ctx, 1, 30);
    assert!(block.contains("✓ L17"));
    assert!(block.contains("LITERAL_REPLACE applied"));
    assert!(
        !block.contains("✗"),
        "failed step at L54 should not appear in window 1-30"
    );

    // Window 40-70: only the failed step overlaps
    let block = format_repair_steps_for_window(&ctx, 40, 70);
    assert!(block.contains("✗ L54"));
    assert!(block.contains("FAILED"));
    assert!(
        !block.contains("✓"),
        "completed step at L17 should not appear in window 40-70"
    );

    // Window 1-100: both steps overlap
    let block = format_repair_steps_for_window(&ctx, 1, 100);
    assert!(block.contains("✓ L17"));
    assert!(block.contains("✗ L54"));

    // Window 200-300: neither step overlaps
    let block = format_repair_steps_for_window(&ctx, 200, 300);
    assert!(block.is_empty());
}

#[test]
fn strip_code_fences_removes_full_wrap_with_language_tag() {
    let input = "```rust\nLITERAL_REPLACE\nSCOPE 1 1\nALL true\nOLD:\nfoo\nEND_OLD\nNEW:\nbar\nEND_NEW\nEND\n```";
    let stripped = strip_code_fences(input);
    assert!(!stripped.contains("```"));
    assert!(stripped.starts_with("LITERAL_REPLACE"));
    assert!(stripped.contains("END_NEW"));
}

#[test]
fn strip_code_fences_removes_full_wrap_without_language_tag() {
    let input = "```\nREPLACE_AT 5\nOLD:\nx\nEND_OLD\nNEW:\ny\nEND_NEW\n```";
    let stripped = strip_code_fences(input);
    assert!(!stripped.contains("```"));
    assert!(stripped.starts_with("REPLACE_AT 5"));
}

#[test]
fn strip_code_fences_tolerates_leading_whitespace() {
    let input = "\n  \n```rust\nSMART_EDIT\nREGION 1 5\nTASK: x\nEND\n```\n";
    let stripped = strip_code_fences(input);
    assert!(!stripped.contains("```"));
    assert!(stripped.starts_with("SMART_EDIT"));
}

#[test]
fn strip_code_fences_keeps_text_with_no_fence() {
    let input =
        "LITERAL_REPLACE\nSCOPE 1 1\nALL true\nOLD:\nfoo\nEND_OLD\nNEW:\nbar\nEND_NEW\nEND\n";
    let stripped = strip_code_fences(input);
    // Identical pass-through.
    assert_eq!(stripped, input);
}

#[test]
fn strip_code_fences_handles_truncated_unclosed_fence() {
    // Model started a ```rust block but ran out of tokens before
    // closing it. We should still strip the opener and keep the
    // body so the parser has a chance.
    let input = "```rust\nLITERAL_REPLACE\nSCOPE 1 1\nALL true\nOLD:\nfoo\nEND_OLD\nNEW:\nbar\nEND_NEW\nEND\n";
    let stripped = strip_code_fences(input);
    assert!(!stripped.starts_with("```"));
    assert!(stripped.starts_with("LITERAL_REPLACE"));
    assert!(stripped.contains("END_NEW"));
}

#[test]
fn strip_code_fences_does_not_touch_inline_backticks() {
    // Backticks inside an otherwise plain block must not be
    // misinterpreted as a closing fence by some greedy regex
    // somewhere — we only consider an outer leading ``` fence.
    let input = "REPLACE_AT 1\nOLD:\nlet `s` = 1;\nEND_OLD\nNEW:\nlet `t` = 2;\nEND_NEW\n";
    let stripped = strip_code_fences(input);
    assert_eq!(stripped, input);
}

#[test]
fn parse_edit_plan_accepts_markdown_fenced_response() {
    // Real failure mode from past benches: model wraps the entire
    // structured response in a ```rust ... ``` fence even though the
    // prompt forbids markdown. Parsing must succeed and return the
    // step inside.
    let input = "```rust\nLITERAL_REPLACE\nSCOPE 1 1\nALL true\nOLD:\nfoo\nEND_OLD\nNEW:\nbar\nEND_NEW\nEND\n```";
    let steps = parse_edit_plan(input).expect("fenced plan should parse");
    assert_eq!(steps.len(), 1);
    match &steps[0] {
        EditPlanStep::LiteralReplace {
            scope_start,
            scope_end,
            all,
            old,
            new,
        } => {
            assert_eq!(*scope_start, 1);
            assert_eq!(*scope_end, 1);
            assert!(*all);
            assert_eq!(old, &vec!["foo".to_string()]);
            assert_eq!(new, &vec!["bar".to_string()]);
        }
        other => panic!("unexpected step variant: {other:?}"),
    }
}

#[test]
fn parse_edit_plan_rejects_literal_replace_without_old() {
    // The OLD-less shortcut is gone — the parser must reject it
    // with a message that tells the planner to either provide OLD
    // verbatim or switch to SMART_EDIT. This is the primary guard
    // against hallucinated replacements.
    let input = "\
LITERAL_REPLACE
SCOPE 10 12
ALL true
NEW:
new line one
new line two
END_NEW
END
";
    let err = parse_edit_plan(input).expect_err("no-OLD form must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("OLD:"),
        "error should instruct the planner to include OLD:, got: {msg}"
    );
    assert!(
        msg.contains("SMART_EDIT"),
        "error should mention SMART_EDIT as the fallback, got: {msg}"
    );
}

#[test]
fn parse_edit_plan_rejects_no_old_form_even_when_all_false() {
    // Same rejection applies regardless of the ALL flag — OLD is
    // required unconditionally.
    let input = "\
LITERAL_REPLACE
SCOPE 1 3
ALL false
NEW:
hello
END_NEW
END
";
    let err = parse_edit_plan(input).expect_err("no-OLD form must be rejected");
    assert!(err.to_string().contains("OLD:"));
}

#[test]
fn parse_edit_plan_classic_literal_replace_still_parses() {
    // Don't regress the existing OLD-bearing form when adding the
    // peek-ahead branch.
    let input = "\
LITERAL_REPLACE
SCOPE 5 7
ALL false
OLD:
foo()
END_OLD
NEW:
bar()
END_NEW
END
";
    let steps = parse_edit_plan(input).expect("classic literal replace should still parse");
    assert_eq!(steps.len(), 1);
    match &steps[0] {
        EditPlanStep::LiteralReplace {
            scope_start,
            scope_end,
            all,
            old,
            new,
        } => {
            assert_eq!(*scope_start, 5);
            assert_eq!(*scope_end, 7);
            assert!(!*all);
            assert_eq!(old, &vec!["foo()".to_string()]);
            assert_eq!(new, &vec!["bar()".to_string()]);
        }
        other => panic!("expected LiteralReplace, got {other:?}"),
    }
}

#[test]
fn parse_patch_accepts_markdown_fenced_response() {
    let input = "```\nREPLACE_AT 5\nOLD:\nold line\nEND_OLD\nNEW:\nnew line\nEND_NEW\n```";
    let ops = parse_patch(input).expect("fenced patch should parse");
    assert_eq!(ops.len(), 1);
    match &ops[0] {
        PatchOp::ReplaceAt { start, old, new } => {
            assert_eq!(*start, 5);
            assert_eq!(old, &vec!["old line".to_string()]);
            assert_eq!(new, &vec!["new line".to_string()]);
        }
        other => panic!("unexpected op variant: {other:?}"),
    }
}

#[test]
fn search_and_read_helpers_report_current_file_content() {
    let content = "\
fn one() {\n\
    context::assemble(&config, \"a\", &[], false, None);\n\
}\n\
\n\
fn two() {\n\
    context::assemble(&config, \"b\", &[], false, None);\n\
}\n";
    let search = search_in_file(content, "context::assemble");
    assert!(search.contains("2 hit(s)"));
    assert!(search.contains("context::assemble(&config, \"a\""));
    assert!(search.contains("context::assemble(&config, \"b\""));

    let read = read_in_file(content, 2, 3).unwrap();
    assert!(read.contains("READ RESULT L2-L3"));
    assert!(read.contains("context::assemble(&config, \"a\""));
}
