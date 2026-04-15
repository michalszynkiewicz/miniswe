//! Parsing and formatting for the edit-plan DSL.
//!
//! This module owns:
//!   * the public plan/patch types (`PatchOp`, `EditRegion`, `EditPlanStep`),
//!   * all parsers for the textual DSL the inner model emits
//!     (`parse_edit_plan`, `parse_patch`, and the tiny line-level helpers),
//!   * small format helpers that render plan steps back into their textual
//!     form for repair prompts and logs,
//!   * sentinel detectors (`looks_like_complete`, `parse_failed`,
//!     `parse_needs_clarification`) and the `truncate_multiline` utility
//!     used for log rendering across this module tree.
//!
//! Kept strictly pure: no LLM calls, no I/O, no `ModelRouter`. Everything
//! in here is deterministic and cheap to unit-test.

use anyhow::{Result, bail};

use super::{DroppedStep, MAX_PREPLAN_LOG_CHARS, MAX_PREPLAN_STEPS};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatchOp {
    InsertBefore {
        line: usize,
        content: Vec<String>,
    },
    InsertAfter {
        line: usize,
        content: Vec<String>,
    },
    ReplaceAt {
        start: usize,
        old: Vec<String>,
        new: Vec<String>,
    },
    DeleteAt {
        start: usize,
        old: Vec<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditRegion {
    pub start: usize,
    pub end: usize,
    pub task: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditPlanStep {
    SmartEdit(EditRegion),
    LiteralReplace {
        scope_start: usize,
        scope_end: usize,
        all: bool,
        old: Vec<String>,
        new: Vec<String>,
    },
}

impl EditPlanStep {
    pub(super) fn start_line(&self) -> usize {
        match self {
            Self::SmartEdit(region) => region.start,
            Self::LiteralReplace { scope_start, .. } => *scope_start,
        }
    }

    pub(super) fn end_line(&self) -> usize {
        match self {
            Self::SmartEdit(region) => region.end,
            Self::LiteralReplace { scope_end, .. } => *scope_end,
        }
    }
}

pub(super) fn format_edit_plan_steps(steps: &[EditPlanStep]) -> String {
    let mut out = String::new();
    for step in steps {
        match step {
            EditPlanStep::SmartEdit(region) => {
                out.push_str("SMART_EDIT\n");
                out.push_str(&format!("REGION {} {}\n", region.start, region.end));
                out.push_str(&format!("TASK: {}\n\n", region.task));
            }
            EditPlanStep::LiteralReplace {
                scope_start,
                scope_end,
                all,
                old,
                new,
            } => {
                out.push_str("LITERAL_REPLACE\n");
                out.push_str(&format!("SCOPE {scope_start} {scope_end}\n"));
                out.push_str(&format!("ALL {all}\n"));
                out.push_str("OLD:\n");
                out.push_str(&old.join("\n"));
                out.push_str("\nEND_OLD\nNEW:\n");
                out.push_str(&new.join("\n"));
                out.push_str("\nEND_NEW\n\n");
            }
        }
    }
    out
}

/// Compact one-liner per step for repair context — shows what changed
/// without dumping full OLD/NEW blocks that overwhelm small models.
pub(super) fn format_completed_steps_compact(steps: &[EditPlanStep]) -> String {
    let mut out = String::new();
    for step in steps {
        match step {
            EditPlanStep::SmartEdit(region) => {
                let task_preview = if region.task.len() > 80 {
                    format!("{}…", &region.task[..77])
                } else {
                    region.task.clone()
                };
                out.push_str(&format!(
                    "  ✓ L{}-L{}: SMART_EDIT applied ({task_preview})\n",
                    region.start, region.end
                ));
            }
            EditPlanStep::LiteralReplace {
                scope_start,
                scope_end,
                old,
                new,
                ..
            } => {
                let old_preview: String = old
                    .first()
                    .map(|l| l.trim().to_string())
                    .unwrap_or_default();
                let old_preview = if old_preview.len() > 60 {
                    format!("{}…", &old_preview[..57])
                } else {
                    old_preview
                };
                let delta = new.len() as isize - old.len() as isize;
                let delta_str = if delta > 0 {
                    format!("+{delta} lines")
                } else if delta < 0 {
                    format!("{delta} lines")
                } else {
                    "same line count".to_string()
                };
                out.push_str(&format!(
                    "  ✓ L{scope_start}-L{scope_end}: LITERAL_REPLACE applied ({delta_str}): {old_preview}\n",
                ));
            }
        }
    }
    out
}

pub(super) fn format_preplan_log(label: &str, steps: &[EditPlanStep]) -> String {
    let plan = format_edit_plan_steps(steps);
    let plan = truncate_multiline(&plan, MAX_PREPLAN_LOG_CHARS);
    format!("Raw {label} ({} step(s), parsed):\n{plan}\n", steps.len())
}

/// Detect the `COMPLETE` verdict sentinel. Accepts the legacy
/// `NO_CHANGES` spelling too so mid-refactor logs/models still converge
/// cleanly — both signal the same "task is satisfied" outcome.
pub(super) fn looks_like_complete(text: &str) -> bool {
    let unfenced = strip_code_fences(text);
    let t = unfenced.trim();
    t == "COMPLETE" || t == "NO_CHANGES"
}

/// Detect a `FAILED: <reason>` verdict sentinel. Returns the trimmed
/// reason, capped at `MAX_FAILED_REASON_CHARS` so a runaway explanation
/// can't balloon the agent-facing output. The model was told to keep it
/// under 200 chars and one line; we enforce both.
pub(super) fn parse_failed(text: &str) -> Option<String> {
    let unfenced = strip_code_fences(text);
    let trimmed = unfenced.trim_start();
    let rest = trimmed.strip_prefix("FAILED")?;
    let after = match rest.chars().next() {
        None => "",
        Some(':') => &rest[1..],
        Some(c) if c.is_whitespace() => rest,
        Some(_) => return None,
    };
    let first_line = after.lines().next().unwrap_or("");
    let trimmed_reason = first_line.trim();
    let reason = if trimmed_reason.chars().count() > MAX_FAILED_REASON_CHARS {
        let truncated: String = trimmed_reason
            .chars()
            .take(MAX_FAILED_REASON_CHARS)
            .collect();
        format!("{truncated}…")
    } else {
        trimmed_reason.to_string()
    };
    Some(reason)
}

/// Upper bound on characters propagated from a `FAILED: <reason>`
/// sentinel into the outer agent-facing error message.
pub(super) const MAX_FAILED_REASON_CHARS: usize = 200;

/// Detect a `NEEDS_CLARIFICATION: <question>` sentinel. Returns
/// `Some(question)` if the response opens with `NEEDS_CLARIFICATION`
/// (optionally followed by `:` and a question), and `None` otherwise. The
/// question is trimmed and may be empty if the model omits one — the
/// caller substitutes a placeholder at render time.
///
/// Guards against lookalikes like `NEEDS_CLARIFICATIONS` or
/// `NEEDS_CLARIFICATIONAL` by requiring end-of-string, whitespace, or `:`
/// immediately after the keyword.
pub(super) fn parse_needs_clarification(text: &str) -> Option<String> {
    let trimmed = text.trim_start();
    let rest = trimmed.strip_prefix("NEEDS_CLARIFICATION")?;
    let after = match rest.chars().next() {
        None => "",
        Some(':') => &rest[1..],
        Some(c) if c.is_whitespace() => rest,
        Some(_) => return None,
    };
    // If there's a trailing newline after the question, keep only the
    // first line — the sentinel is single-line by contract.
    let first_line = after.lines().next().unwrap_or("");
    Some(first_line.trim().to_string())
}

pub(super) fn truncate_multiline(text: &str, max_chars: usize) -> String {
    let char_count = text.chars().count();
    if char_count <= max_chars {
        return text.to_string();
    }

    let truncated: String = text.chars().take(max_chars).collect();
    format!("{truncated}\n...({char_count} chars total, truncated)\n")
}

/// Strip a single outer ```...``` markdown fence from `text`, if present.
///
/// The finalize prompt explicitly says "no markdown", but smaller models
/// sometimes wrap their entire structured response in a ``` fence anyway
/// (often with a language tag like ```rust). When that happens the parser
/// fails on the leading backticks even though the body is otherwise
/// well-formed. This helper detects that one common shape and returns the
/// inner body so parsing can proceed.
///
/// Behaviour:
/// - If the leading non-whitespace characters are not ``` we return the
///   input unchanged (no fence to strip).
/// - If a leading ``` is present we drop everything from the start through
///   the first newline (i.e. the fence line, including any language tag).
/// - If there is *also* a trailing ``` (anywhere later in the text) we drop
///   it and everything after it. If the response was truncated mid-output
///   and never closed the fence, we still strip the opener and keep the
///   rest — that gives us the best shot at parsing a partial reply.
pub(super) fn strip_code_fences(text: &str) -> String {
    let trimmed = text.trim_start();
    let Some(rest) = trimmed.strip_prefix("```") else {
        return text.to_string();
    };
    let Some(newline_pos) = rest.find('\n') else {
        return text.to_string();
    };
    let body = &rest[newline_pos + 1..];
    let inner = match body.rfind("```") {
        Some(pos) => &body[..pos],
        None => body,
    };
    inner.to_string()
}

pub fn parse_edit_plan(text: &str) -> Result<Vec<EditPlanStep>> {
    // Empty input parses to zero steps. The finalize caller now detects
    // empty responses and `NO_CHANGES` up-front and routes them into
    // dedicated `PreplanOutcome` variants, so this code path is only
    // reached when some caller passes a body that turned out to be empty
    // (e.g. after fence stripping) or on a stray NO_CHANGES token mixed
    // in with real steps — we tolerate those below rather than failing
    // the parse.
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }

    let unfenced = strip_code_fences(text);
    let text = unfenced.as_str();
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }

    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;
    let mut steps = Vec::new();

    while i < lines.len() {
        let line = lines[i];
        if line.trim().is_empty() {
            i += 1;
            continue;
        }

        // Tolerate a stray NO_CHANGES token from older prompts or
        // confused models — treat it as a no-op separator.
        if line.trim() == "NO_CHANGES" {
            i += 1;
            continue;
        }

        // Tolerate a stray END terminator. The current DSL no longer
        // uses END as a step terminator (END_OLD/END_NEW are enough),
        // but older models trained on the prior format may still emit
        // it. Skip it instead of failing the parse.
        if line.trim() == "END" {
            i += 1;
            continue;
        }

        if line == "SMART_EDIT" {
            i += 1;
            let (region, next) = parse_region_at(&lines, i)?;
            i = next;
            steps.push(EditPlanStep::SmartEdit(region));
            continue;
        }

        if line.starts_with("REGION ") {
            let (region, next) = parse_region_at(&lines, i)?;
            i = next;
            steps.push(EditPlanStep::SmartEdit(region));
            continue;
        }

        if line == "LITERAL_REPLACE" {
            i += 1;
            let scope_line = lines
                .get(i)
                .ok_or_else(|| anyhow::anyhow!("missing SCOPE line for literal replace"))?;
            let Some(rest) = scope_line.strip_prefix("SCOPE ") else {
                bail!("expected SCOPE line but found '{scope_line}'");
            };
            let (scope_start, scope_end) = parse_two_line_numbers(rest, scope_line, "scope")?;
            i += 1;

            let all_line = lines
                .get(i)
                .ok_or_else(|| anyhow::anyhow!("missing ALL line for literal replace"))?;
            let all = match all_line.strip_prefix("ALL ") {
                Some("true") => true,
                Some("false") => false,
                Some(other) => bail!("invalid ALL value '{other}' in line '{all_line}'"),
                None => bail!("expected ALL line but found '{all_line}'"),
            };
            i += 1;

            // Reject the old "NEW: without OLD:" shortcut. It let the
            // model regenerate the replacement text without cross-checking
            // the scope it was editing, which was the root cause of most
            // hallucination failures (inventing `&self` on a free fn,
            // dropping `pub struct Foo {` off the start of a scope, etc).
            // OLD: is now unconditional so we can verify at parse/apply
            // time that the model knows what it's replacing.
            let mut peek = i;
            while peek < lines.len() && lines[peek].trim().is_empty() {
                peek += 1;
            }
            let head = lines.get(peek).copied().unwrap_or("");
            if head == "NEW:" {
                bail!(
                    "LITERAL_REPLACE now requires an OLD: block — copy the exact current text of L{scope_start}-L{scope_end} into OLD:, then put the replacement in NEW:. If the region is too large to echo verbatim, use SMART_EDIT instead (its execution phase will see the region content)."
                );
            }

            expect_line(&lines, i, "OLD:")?;
            i += 1;
            let (old, next) = collect_until(&lines, i, "END_OLD")?;
            if old.is_empty() {
                bail!("literal OLD block must not be empty");
            }
            if old.iter().all(|line| line.trim().is_empty()) {
                bail!("literal OLD block must contain non-whitespace text");
            }
            i = next + 1;

            expect_line(&lines, i, "NEW:")?;
            i += 1;
            let (new, next) = collect_until(&lines, i, "END_NEW")?;
            i = next + 1;

            steps.push(EditPlanStep::LiteralReplace {
                scope_start,
                scope_end,
                all,
                old,
                new,
            });
            continue;
        }

        bail!("unexpected text in edit plan: {line}");
    }

    if steps.len() > MAX_PREPLAN_STEPS {
        bail!(
            "edit plan returned {} steps, maximum is {MAX_PREPLAN_STEPS}",
            steps.len()
        );
    }
    // Overlap is *not* a parse error: the planner caller resolves overlaps
    // by keeping the first step in source order and reporting the rest as
    // dropped steps in the per-step output. See `partition_overlapping_steps`.
    Ok(steps)
}

pub(super) fn parse_region_at(lines: &[&str], idx: usize) -> Result<(EditRegion, usize)> {
    let line = lines
        .get(idx)
        .ok_or_else(|| anyhow::anyhow!("missing REGION header"))?;
    let Some(rest) = line.strip_prefix("REGION ") else {
        bail!("unexpected text in region plan: {line}");
    };
    let rest = rest.trim();
    let (start, end) = if let Some((start, end)) = rest.split_once('-') {
        let start = start.trim().parse::<usize>().map_err(|e| {
            anyhow::anyhow!("invalid region start '{start}' in header '{line}': {e}")
        })?;
        let end = end
            .trim()
            .parse::<usize>()
            .map_err(|e| anyhow::anyhow!("invalid region end '{end}' in header '{line}': {e}"))?;
        (start, end)
    } else {
        let parts: Vec<_> = rest.split_whitespace().collect();
        match parts.as_slice() {
            [single] => {
                let value = single.parse::<usize>().map_err(|e| {
                    anyhow::anyhow!("invalid region line '{single}' in header '{line}': {e}")
                })?;
                (value, value)
            }
            [_, _] => parse_two_line_numbers(rest, line, "region")?,
            _ => bail!("invalid REGION header: {line}"),
        }
    };

    let task_line = lines
        .get(idx + 1)
        .ok_or_else(|| anyhow::anyhow!("missing TASK line for region"))?;
    let task = task_line
        .strip_prefix("TASK:")
        .ok_or_else(|| anyhow::anyhow!("expected TASK line but found '{task_line}'"))?
        .trim()
        .to_string();
    if task.is_empty() {
        bail!("region task must not be empty");
    }

    Ok((EditRegion { start, end, task }, idx + 2))
}

pub(super) fn parse_two_line_numbers(
    rest: &str,
    header: &str,
    label: &str,
) -> Result<(usize, usize)> {
    let mut parts = rest.split_whitespace();
    let start_token = parts
        .next()
        .ok_or_else(|| anyhow::anyhow!("missing {label} start in header: {header}"))?;
    let start = start_token.parse::<usize>().map_err(|e| {
        anyhow::anyhow!("invalid {label} start '{start_token}' in header '{header}': {e}")
    })?;
    let end_token = parts
        .next()
        .ok_or_else(|| anyhow::anyhow!("missing {label} end in header: {header}"))?;
    let end = end_token.parse::<usize>().map_err(|e| {
        anyhow::anyhow!("invalid {label} end '{end_token}' in header '{header}': {e}")
    })?;
    if parts.next().is_some() {
        bail!("too many fields in {label} header: {header}");
    }
    if start == 0 || end < start {
        bail!("invalid {label} L{start}-L{end}");
    }
    Ok((start, end))
}

pub(super) fn reject_overlapping_regions(regions: &[EditRegion]) -> Result<()> {
    let mut sorted: Vec<&EditRegion> = regions.iter().collect();
    sorted.sort_unstable_by(|a, b| a.start.cmp(&b.start).then_with(|| a.end.cmp(&b.end)));

    for pair in sorted.windows(2) {
        let prev = pair[0];
        let next = pair[1];
        if next.start <= prev.end {
            bail!(
                "region plan has overlapping regions: L{}-L{} overlaps L{}-L{}",
                prev.start,
                prev.end,
                next.start,
                next.end
            );
        }
    }
    Ok(())
}

/// Partition planned steps into "kept" (first occurrence wins, by source
/// order) and "dropped" (overlaps an earlier step). The kept set keeps
/// original emission order so downstream logging stays intuitive. The
/// dropped set carries the original step plus a human-readable reason
/// pointing at the kept step that caused the conflict, so the executor
/// can report each one as a failed step in the per-step output.
pub(super) fn partition_overlapping_steps(
    steps: Vec<EditPlanStep>,
) -> (Vec<EditPlanStep>, Vec<DroppedStep>) {
    // Sort by source position so "first wins" is deterministic and matches
    // file order rather than emission order.
    let mut indexed: Vec<(usize, EditPlanStep)> = steps.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| {
        a.1.start_line()
            .cmp(&b.1.start_line())
            .then_with(|| a.1.end_line().cmp(&b.1.end_line()))
            .then_with(|| a.0.cmp(&b.0))
    });

    let mut kept: Vec<(usize, EditPlanStep)> = Vec::with_capacity(indexed.len());
    let mut dropped: Vec<(usize, DroppedStep)> = Vec::new();
    for (orig_idx, step) in indexed {
        let conflict = kept.iter().find(|(_, prev)| {
            // [a,b] overlaps [c,d] iff a <= d && c <= b
            step.start_line() <= prev.end_line() && prev.start_line() <= step.end_line()
        });
        if let Some((_, prev)) = conflict {
            let reason = format!(
                "overlaps earlier step L{}-L{}",
                prev.start_line(),
                prev.end_line()
            );
            dropped.push((orig_idx, DroppedStep { step, reason }));
        } else {
            kept.push((orig_idx, step));
        }
    }

    // Restore original emission order so downstream "in order" logging stays
    // intuitive for the agent.
    kept.sort_by_key(|(idx, _)| *idx);
    dropped.sort_by_key(|(idx, _)| *idx);
    (
        kept.into_iter().map(|(_, s)| s).collect(),
        dropped.into_iter().map(|(_, d)| d).collect(),
    )
}

/// Parse strict patch DSL blocks.
pub fn parse_patch(text: &str) -> Result<Vec<PatchOp>> {
    let unfenced = strip_code_fences(text);
    let text = unfenced.as_str();
    if text.trim() == "NO_CHANGES" {
        return Ok(Vec::new());
    }
    if text.trim().is_empty() {
        bail!("empty patch");
    }

    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;
    let mut ops = Vec::new();

    while i < lines.len() {
        let line = lines[i];
        if line.trim().is_empty() {
            i += 1;
            continue;
        }

        if let Some(rest) = line.strip_prefix("INSERT_BEFORE ") {
            let line_num = parse_line_number(rest)?;
            i += 1;
            expect_line(&lines, i, "CONTENT:")?;
            i += 1;
            let (content, next) = collect_until(&lines, i, "END")?;
            i = next + 1;
            ops.push(PatchOp::InsertBefore {
                line: line_num,
                content,
            });
        } else if let Some(rest) = line.strip_prefix("INSERT_AFTER ") {
            let line_num = parse_line_number(rest)?;
            i += 1;
            expect_line(&lines, i, "CONTENT:")?;
            i += 1;
            let (content, next) = collect_until(&lines, i, "END")?;
            i = next + 1;
            ops.push(PatchOp::InsertAfter {
                line: line_num,
                content,
            });
        } else if let Some(rest) = line.strip_prefix("REPLACE_AT ") {
            let start = parse_line_number(rest)?;
            i += 1;
            expect_line(&lines, i, "OLD:")?;
            i += 1;
            let (old, next) = collect_until(&lines, i, "END_OLD")?;
            i = next + 1;
            expect_line(&lines, i, "NEW:")?;
            i += 1;
            let (new, next) = collect_until(&lines, i, "END_NEW")?;
            i = next + 1;
            ops.push(PatchOp::ReplaceAt { start, old, new });
        } else if let Some(rest) = line.strip_prefix("DELETE_AT ") {
            let start = parse_line_number(rest)?;
            i += 1;
            expect_line(&lines, i, "OLD:")?;
            i += 1;
            let (old, next) = collect_until(&lines, i, "END_OLD")?;
            i = next + 1;
            ops.push(PatchOp::DeleteAt { start, old });
        } else {
            bail!("unexpected text in patch: {line}");
        }
    }

    Ok(ops)
}

pub(super) fn parse_line_number(text: &str) -> Result<usize> {
    let raw = text.trim();
    let line = raw
        .parse::<usize>()
        .map_err(|e| anyhow::anyhow!("invalid line number '{raw}': {e}"))?;
    if line == 0 {
        bail!("line numbers are 1-based");
    }
    Ok(line)
}

pub(super) fn expect_line(lines: &[&str], idx: usize, expected: &str) -> Result<()> {
    match lines.get(idx) {
        Some(line) if *line == expected => Ok(()),
        Some(line) => bail!("expected '{expected}' but found '{line}'"),
        None => bail!("expected '{expected}' but reached end of patch"),
    }
}

pub(super) fn collect_until(
    lines: &[&str],
    start: usize,
    sentinel: &str,
) -> Result<(Vec<String>, usize)> {
    let mut collected = Vec::new();
    for (idx, line) in lines.iter().enumerate().skip(start) {
        if *line == sentinel {
            return Ok((collected, idx));
        }
        collected.push((*line).to_string());
    }
    bail!("missing sentinel '{sentinel}'");
}
