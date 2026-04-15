//! Patch application — everything that mutates an in-memory file view.
//!
//! This module handles:
//!   * dry-run application of patch ops (`apply_patch_dry_run` and its
//!     region-constrained cousin),
//!   * LITERAL_REPLACE application inside a scoped byte range,
//!   * the OLD-relocation rescue that searches the whole file for a
//!     misplaced or whitespace-drifted OLD block and asks the planner
//!     whether the new line range is the intended target,
//!   * patch resolution (`resolve_ops`) which turns positional patch ops
//!     into concrete line-range edits with exact-match anchoring,
//!   * small preview helpers used to build actionable error messages.
//!
//! The LLM-facing rescue (`try_relocate_and_replace` +
//! `request_relocation_confirmation`) is the only code path in this
//! module that issues a model call. Everything else is pure.

use std::sync::atomic::AtomicBool;

use anyhow::{Result, bail};

use super::parse::{PatchOp, truncate_multiline};
use super::{ensure_not_cancelled, log_debug, log_stage};
use crate::config::ModelRole;
use crate::llm::{ChatRequest, Message, ModelRouter};
use crate::logging::SessionLog;

/// Apply all operations to memory only. If any operation fails, returns an
/// error and the original file on disk remains untouched.
pub fn apply_patch_dry_run(content: &str, ops: &[PatchOp]) -> Result<String> {
    let lines: Vec<String> = content.lines().map(str::to_string).collect();
    let resolved = resolve_ops(&lines, ops)?;
    apply_resolved_patch(content, resolved)
}

pub(super) fn apply_patch_dry_run_in_region(
    content: &str,
    ops: &[PatchOp],
    start_line: usize,
    end_line: usize,
) -> Result<String> {
    let lines: Vec<String> = content.lines().map(str::to_string).collect();
    if start_line == 0 || end_line < start_line || end_line > lines.len() {
        bail!(
            "invalid edit region L{start_line}-L{end_line} for {} line file",
            lines.len()
        );
    }

    let resolved = resolve_ops(&lines, ops)?;
    let allowed_start = start_line - 1;
    let allowed_end = end_line;

    for op in &resolved {
        if op.start < allowed_start || op.end > allowed_end {
            bail!(
                "{} resolves to {}, outside allowed region L{}-L{}",
                op.label,
                display_span(op.start, op.end),
                start_line,
                end_line
            );
        }
    }

    apply_resolved_patch(content, resolved)
}

pub fn apply_literal_replace_in_scope(
    content: &str,
    scope_start: usize,
    scope_end: usize,
    old: &[String],
    new: &[String],
    all: bool,
) -> Result<(String, usize)> {
    if scope_start == 0 || scope_end < scope_start {
        bail!("invalid literal scope L{scope_start}-L{scope_end}");
    }
    if old.is_empty() {
        bail!("literal OLD block must not be empty");
    }

    let line_count = content.lines().count();
    if scope_end > line_count {
        bail!("literal scope L{scope_start}-L{scope_end} outside {line_count} line file");
    }

    let parts: Vec<&str> = content.split_inclusive('\n').collect();
    let start_byte: usize = parts[..scope_start - 1].iter().map(|part| part.len()).sum();
    let end_byte: usize = parts[..scope_end].iter().map(|part| part.len()).sum();

    let old_text = old.join("\n");
    let new_text = new.join("\n");
    let scoped = &content[start_byte..end_byte];
    let count = scoped.matches(&old_text).count();

    if all {
        if count == 0 {
            bail!(
                "literal OLD block was not found in scope L{scope_start}-L{scope_end}\nOLD block:\n{}",
                preview_block(old, None)
            );
        }
    } else if count != 1 {
        bail!(
            "literal OLD block matched {count} occurrence(s) in scope L{scope_start}-L{scope_end}; expected exactly 1"
        );
    }

    let replaced_scope = if all {
        scoped.replace(&old_text, &new_text)
    } else {
        scoped.replacen(&old_text, &new_text, 1)
    };

    let mut out = String::with_capacity(content.len() + replaced_scope.len());
    out.push_str(&content[..start_byte]);
    out.push_str(&replaced_scope);
    out.push_str(&content[end_byte..]);
    Ok((out, count))
}

/// Outcome of the OLD-relocation rescue.
///
/// When a byte-exact LITERAL_REPLACE fails at the declared scope, the
/// rescue searches the whole file for a candidate location where the
/// OLD block fits (byte-exact first, then whitespace-tolerant). If it
/// finds one and the planner confirms via a YES/NO round trip, the
/// replacement is applied at the corrected line range.
pub(super) enum RelocateOutcome {
    /// Candidate located, planner confirmed, replacement applied. Carries
    /// the new file content plus the 1-based inclusive line range at
    /// which the match was found.
    Applied {
        new_content: String,
        located_at: (usize, usize),
    },
    /// Candidate(s) located but the planner rejected the proposal. The
    /// caller should bubble straight up to plan-level repair — we have
    /// no better information to offer.
    Rejected,
    /// No candidate found anywhere in the file, or the confirmation
    /// request itself failed.
    NoCandidate,
}

/// Collect all 1-based inclusive `(start, end)` line ranges where the
/// `old` block matches `content` byte-exact, line-by-line.
pub(super) fn find_all_exact_line_matches(content: &str, old: &[String]) -> Vec<(usize, usize)> {
    if old.is_empty() {
        return Vec::new();
    }
    let lines: Vec<&str> = content.lines().collect();
    let n = old.len();
    if n > lines.len() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for i in 0..=lines.len() - n {
        if (0..n).all(|j| lines[i + j] == old[j]) {
            out.push((i + 1, i + n));
        }
    }
    out
}

fn ws_squash(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

/// Minimum per-line similarity (1.0 = identical, 0.0 = fully disjoint)
/// for fuzzy relocation to consider a line a match. Conservative — we
/// would rather miss a real edit than pull the planner into a noisy
/// confirmation on a line it didn't actually want to touch.
const FUZZY_MIN_LINE_SIM: f64 = 0.70;
/// Minimum average similarity across all OLD lines for a fuzzy block
/// to qualify as a candidate.
const FUZZY_MIN_BLOCK_SIM: f64 = 0.85;

/// Character-level Levenshtein distance, iterative two-row DP.
/// Multibyte-safe (operates on `char`s).
fn edit_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0usize; n + 1];
    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

/// Fraction of characters shared between two lines, normalized by the
/// longer length. Two empty strings score as identical.
fn line_similarity(a: &str, b: &str) -> f64 {
    let len_a = a.chars().count();
    let len_b = b.chars().count();
    let max_len = len_a.max(len_b);
    if max_len == 0 {
        return 1.0;
    }
    1.0 - (edit_distance(a, b) as f64) / (max_len as f64)
}

/// Collect all 1-based inclusive `(start, end)` line ranges where the
/// `old` block approximately matches `content` under character-level
/// edit distance. Every line must clear `FUZZY_MIN_LINE_SIM` and the
/// block average must clear `FUZZY_MIN_BLOCK_SIM`.
///
/// Ranges already in `exclude` (byte-exact or whitespace-tolerant hits)
/// are skipped so the picker does not see the same range twice.
///
/// Rejects OLD blocks whose lines are all trivially short — fuzzy
/// scoring is noisy on very short strings and would otherwise match
/// half the file for a one-word OLD.
pub(super) fn find_all_fuzzy_line_matches(
    content: &str,
    old: &[String],
    exclude: &[(usize, usize)],
) -> Vec<(usize, usize)> {
    if old.is_empty() {
        return Vec::new();
    }
    let lines: Vec<&str> = content.lines().collect();
    let n = old.len();
    if n > lines.len() {
        return Vec::new();
    }
    // Require at least one OLD line with ≥5 non-whitespace chars.
    // Single-line "x=1" OLDs are too noisy to fuzzy-match reliably.
    if !old
        .iter()
        .any(|l| l.chars().filter(|c| !c.is_whitespace()).count() >= 5)
    {
        return Vec::new();
    }
    let mut out = Vec::new();
    'outer: for i in 0..=lines.len() - n {
        let range = (i + 1, i + n);
        if exclude.contains(&range) {
            continue;
        }
        let mut sum = 0.0f64;
        for j in 0..n {
            let sim = line_similarity(lines[i + j], &old[j]);
            if sim < FUZZY_MIN_LINE_SIM {
                continue 'outer;
            }
            sum += sim;
        }
        if sum / n as f64 >= FUZZY_MIN_BLOCK_SIM {
            out.push(range);
        }
    }
    out
}

/// Collect all 1-based inclusive `(start, end)` line ranges where the
/// `old` block matches `content` after per-line whitespace normalization.
/// Lines are compared after removing every whitespace character.
/// Ranges that already appear in `exclude` (e.g. byte-exact hits) are
/// skipped to avoid duplicates feeding into the candidate picker.
///
/// OLD blocks that squash to nothing on every line (all-whitespace
/// input) would match every stretch of blank lines, so we require at
/// least one squashed OLD line to be non-empty before returning hits.
pub(super) fn find_all_ws_tolerant_line_matches(
    content: &str,
    old: &[String],
    exclude: &[(usize, usize)],
) -> Vec<(usize, usize)> {
    if old.is_empty() {
        return Vec::new();
    }
    let lines: Vec<&str> = content.lines().collect();
    let n = old.len();
    if n > lines.len() {
        return Vec::new();
    }
    let sq_old: Vec<String> = old.iter().map(|l| ws_squash(l)).collect();
    if sq_old.iter().all(|l| l.is_empty()) {
        return Vec::new();
    }
    let sq_file: Vec<String> = lines.iter().map(|l| ws_squash(l)).collect();
    let mut out = Vec::new();
    for i in 0..=lines.len() - n {
        let range = (i + 1, i + n);
        if exclude.contains(&range) {
            continue;
        }
        if (0..n).all(|j| sq_file[i + j] == sq_old[j]) {
            out.push(range);
        }
    }
    out
}

/// Pick the best candidate line range, biased toward the declared scope.
///
/// 1. If any candidate overlaps the declared `[scope_start, scope_end]`
///    (range intersection, endpoints included), only overlapping
///    candidates are considered.
/// 2. Within that set (or the full set if nothing overlaps), pick the
///    one closest by `|start - scope_start| + |end - scope_end|`.
/// 3. Ties are broken by insertion order, which means the caller can
///    front-load higher-confidence hits (byte-exact before fuzzy).
pub(super) fn pick_best_candidate(
    candidates: &[(usize, usize)],
    scope_start: usize,
    scope_end: usize,
) -> Option<(usize, usize)> {
    if candidates.is_empty() {
        return None;
    }
    let overlaps = |&(x, y): &(usize, usize)| x <= scope_end && y >= scope_start;
    let distance = |&(x, y): &(usize, usize)| {
        (x as isize - scope_start as isize).abs() + (y as isize - scope_end as isize).abs()
    };
    let any_overlap = candidates.iter().any(overlaps);
    candidates
        .iter()
        .enumerate()
        .filter(|(_, c)| !any_overlap || overlaps(c))
        .min_by(|(ai, a), (bi, b)| distance(a).cmp(&distance(b)).then_with(|| ai.cmp(bi)))
        .map(|(_, c)| *c)
}

/// Try to rescue a failed LITERAL_REPLACE by searching the whole file
/// for a location where the OLD block fits (byte-exact, then
/// whitespace-tolerant), picking the best candidate with a locality
/// bias toward the declared scope, and asking the planner to confirm
/// the corrected line range in a single YES/NO round trip.
///
/// On confirmation, re-fires LITERAL_REPLACE at the corrected scope
/// using the actual file text at that range as the canonical OLD (so
/// whitespace-tolerant matches don't leave the original drifted OLD to
/// trip up the byte-exact matcher a second time).
#[allow(clippy::too_many_arguments)]
pub(super) async fn try_relocate_and_replace(
    path_str: &str,
    content: &str,
    scope_start: usize,
    scope_end: usize,
    old: &[String],
    new: &[String],
    all: bool,
    router: &ModelRouter,
    cancelled: Option<&AtomicBool>,
    log: Option<&SessionLog>,
) -> RelocateOutcome {
    let exact = find_all_exact_line_matches(content, old);
    let ws = find_all_ws_tolerant_line_matches(content, old, &exact);
    // Fuzzy hits only fire when no exact or whitespace-tolerant match
    // exists at all — fuzzy search is the last resort and we don't want
    // its approximate hits displacing a high-confidence match from the
    // picker even under the locality bias.
    let fuzzy = if exact.is_empty() && ws.is_empty() {
        find_all_fuzzy_line_matches(content, old, &[])
    } else {
        Vec::new()
    };

    // Byte-exact hits come first so ties in the locality-biased picker
    // resolve in favor of higher-confidence matches. Fuzzy hits come
    // last for the same reason.
    let mut candidates: Vec<(usize, usize)> =
        Vec::with_capacity(exact.len() + ws.len() + fuzzy.len());
    candidates.extend(exact.iter().copied());
    candidates.extend(ws.iter().copied());
    candidates.extend(fuzzy.iter().copied());

    let Some(located_at) = pick_best_candidate(&candidates, scope_start, scope_end) else {
        return RelocateOutcome::NoCandidate;
    };

    // If the only match is byte-exact at the declared scope, the outer
    // caller already tried that and failed — something upstream is wrong.
    // Bail instead of asking the planner a pointless question.
    if located_at == (scope_start, scope_end) && exact.contains(&located_at) {
        return RelocateOutcome::NoCandidate;
    }

    let (new_start, new_end) = located_at;
    let strategy = if exact.contains(&located_at) {
        "byte-exact"
    } else if ws.contains(&located_at) {
        "whitespace-normalized"
    } else {
        "fuzzy line similarity"
    };

    // Reconstruct OLD from the actual file lines at the located range.
    // For byte-exact matches this is identical to the planner's OLD.
    // For whitespace-tolerant matches this restores the canonical text
    // so a follow-up LITERAL_REPLACE call succeeds without extra logic.
    let file_old: Vec<String> = content
        .lines()
        .skip(new_start - 1)
        .take(new_end - new_start + 1)
        .map(String::from)
        .collect();

    log_debug(
        log,
        path_str,
        &format!(
            "literal:relocate candidate L{new_start}-L{new_end} ({strategy}), declared L{scope_start}-L{scope_end}; asking planner to confirm"
        ),
    );

    let confirmed = match request_relocation_confirmation(
        path_str,
        scope_start,
        scope_end,
        new_start,
        new_end,
        &old.join("\n"),
        &file_old.join("\n"),
        strategy,
        router,
        cancelled,
        log,
    )
    .await
    {
        Ok(yes) => yes,
        Err(e) => {
            // Confirmation request itself failed (network, timeout,
            // cancellation). Treat as NoCandidate so a transient
            // transport hiccup doesn't look like a planner rejection.
            log_debug(
                log,
                path_str,
                &format!("literal:relocate confirmation request failed: {e}"),
            );
            return RelocateOutcome::NoCandidate;
        }
    };

    if !confirmed {
        log_debug(
            log,
            path_str,
            &format!("literal:relocate L{new_start}-L{new_end} rejected by planner"),
        );
        return RelocateOutcome::Rejected;
    }

    match apply_literal_replace_in_scope(content, new_start, new_end, &file_old, new, all) {
        Ok((new_content, _count)) => {
            log_debug(
                log,
                path_str,
                &format!("literal:relocate L{new_start}-L{new_end} confirmed and applied"),
            );
            RelocateOutcome::Applied {
                new_content,
                located_at,
            }
        }
        Err(e) => {
            // Shouldn't happen: we just reconstructed OLD from the file
            // at this range, so a byte-exact match is guaranteed unless
            // the caller passed a contradictory `all` flag. Log and bail.
            log_debug(
                log,
                path_str,
                &format!("literal:relocate L{new_start}-L{new_end} apply failed: {e}"),
            );
            RelocateOutcome::NoCandidate
        }
    }
}

/// Ask the planner whether a located line range is the same edit target
/// it intended when it wrote the (now-misplaced) OLD block. Single
/// round-trip, single word reply expected (`YES` / `NO`).
#[allow(clippy::too_many_arguments)]
async fn request_relocation_confirmation(
    path_str: &str,
    declared_start: usize,
    declared_end: usize,
    new_start: usize,
    new_end: usize,
    planner_old: &str,
    file_text: &str,
    strategy: &str,
    router: &ModelRouter,
    cancelled: Option<&AtomicBool>,
    log: Option<&SessionLog>,
) -> Result<bool> {
    ensure_not_cancelled(cancelled)?;
    let prompt = format!(
        "An edit-plan step for {path_str} declared SCOPE L{declared_start}-L{declared_end}, \
         but the byte-exact LITERAL_REPLACE did not match there. Searching the whole file \
         found a {strategy} match at L{new_start}-L{new_end}. Confirm whether that new \
         range is the same edit target you intended.\n\n\
         OLD as written by the planner:\n\
         ----\n{planner_old}\n----\n\n\
         Actual file text at L{new_start}-L{new_end}:\n\
         ----\n{file_text}\n----\n\n\
         Reply with exactly `YES` if L{new_start}-L{new_end} is the intended target, \
         or `NO` if it isn't. Output only the single word, no other text."
    );

    let request = ChatRequest {
        messages: vec![
            Message::system("You answer with exactly `YES` or `NO`. No other output, no markdown."),
            Message::user(&prompt),
        ],
        tools: None,
        tool_choice: None,
    };

    log_stage(
        log,
        path_str,
        &format!("literal:relocate_confirm:L{new_start}-L{new_end}"),
    );
    let response = router
        .chat_with_cancel(ModelRole::Fast, &request, cancelled)
        .await?;
    let text = response
        .choices
        .first()
        .and_then(|c| c.message.content.as_deref())
        .unwrap_or("");
    log_debug(
        log,
        path_str,
        &format!(
            "literal:relocate_confirm:L{new_start}-L{new_end} reply: {}",
            truncate_multiline(text, 400)
        ),
    );

    // Tolerate trailing punctuation, mixed case, and a leading word like
    // "Yes." or "yes,". Reject anything that doesn't have "yes" as its
    // first alphabetic token.
    let first_word: String = text
        .trim()
        .chars()
        .take_while(|c| c.is_alphabetic())
        .flat_map(|c| c.to_lowercase())
        .collect();
    Ok(first_word == "yes")
}

fn apply_resolved_patch(content: &str, mut resolved: Vec<ResolvedOp>) -> Result<String> {
    let had_trailing_newline = content.ends_with('\n');
    let mut lines: Vec<String> = content.lines().map(str::to_string).collect();

    resolved.sort_by(|a, b| b.start.cmp(&a.start).then_with(|| b.end.cmp(&a.end)));

    for op in &resolved {
        match &op.kind {
            ResolvedKind::Insert { content } => {
                lines.splice(op.start..op.start, content.clone());
            }
            ResolvedKind::Replace { content } => {
                lines.splice(op.start..op.end, content.clone());
            }
            ResolvedKind::Delete => {
                lines.splice(op.start..op.end, Vec::<String>::new());
            }
        }
    }

    let mut out = lines.join("\n");
    if had_trailing_newline && !out.ends_with('\n') {
        out.push('\n');
    }
    Ok(out)
}

#[derive(Debug, Clone)]
struct ResolvedOp {
    label: String,
    start: usize,
    end: usize,
    kind: ResolvedKind,
}

#[derive(Debug, Clone)]
enum ResolvedKind {
    Insert { content: Vec<String> },
    Replace { content: Vec<String> },
    Delete,
}

fn resolve_ops(original: &[String], ops: &[PatchOp]) -> Result<Vec<ResolvedOp>> {
    let mut resolved = Vec::new();

    for (idx, op) in ops.iter().enumerate() {
        let label = op_label(idx + 1, op);
        match op {
            PatchOp::InsertBefore { line, content } => {
                validate_insert_line(*line, original.len())?;
                resolved.push(ResolvedOp {
                    label,
                    start: *line - 1,
                    end: *line - 1,
                    kind: ResolvedKind::Insert {
                        content: content.clone(),
                    },
                });
            }
            PatchOp::InsertAfter { line, content } => {
                validate_insert_line(*line, original.len())?;
                resolved.push(ResolvedOp {
                    label,
                    start: *line,
                    end: *line,
                    kind: ResolvedKind::Insert {
                        content: content.clone(),
                    },
                });
            }
            PatchOp::ReplaceAt { start, old, new } => {
                let start_idx = resolve_old_anchor(original, *start, old, "REPLACE_AT")?;
                resolved.push(ResolvedOp {
                    label,
                    start: start_idx,
                    end: start_idx + old.len(),
                    kind: ResolvedKind::Replace {
                        content: new.clone(),
                    },
                });
            }
            PatchOp::DeleteAt { start, old } => {
                let start_idx = resolve_old_anchor(original, *start, old, "DELETE_AT")?;
                resolved.push(ResolvedOp {
                    label,
                    start: start_idx,
                    end: start_idx + old.len(),
                    kind: ResolvedKind::Delete,
                });
            }
        }
    }

    reject_overlapping_spans(&resolved)?;
    Ok(resolved)
}

fn op_label(ordinal: usize, op: &PatchOp) -> String {
    match op {
        PatchOp::InsertBefore { line, .. } => format!("op {ordinal} INSERT_BEFORE {line}"),
        PatchOp::InsertAfter { line, .. } => format!("op {ordinal} INSERT_AFTER {line}"),
        PatchOp::ReplaceAt { start, old, .. } => {
            format!(
                "op {ordinal} REPLACE_AT {start} ({} OLD line(s))",
                old.len()
            )
        }
        PatchOp::DeleteAt { start, old } => {
            format!("op {ordinal} DELETE_AT {start} ({} OLD line(s))", old.len())
        }
    }
}

fn resolve_old_anchor(
    original: &[String],
    start_line: usize,
    old: &[String],
    op_name: &str,
) -> Result<usize> {
    if old.is_empty() {
        bail!("{op_name} OLD block must not be empty");
    }
    if start_line == 0 {
        bail!("line numbers are 1-based");
    }

    let hinted = start_line - 1;
    if hinted + old.len() <= original.len() && original[hinted..hinted + old.len()] == *old {
        return Ok(hinted);
    }

    let matches = find_exact_block_matches(original, old);
    match matches.as_slice() {
        [idx] => Ok(*idx),
        [] => {
            let mut msg = format!(
                "OLD mismatch for {op_name} {start_line}: OLD block was not found at the anchor or elsewhere"
            );
            msg.push_str(&format!(
                "\nOLD block ({} line(s)):\n{}",
                old.len(),
                preview_block(old, None)
            ));
            msg.push_str(&format!(
                "\nActual text at anchor:\n{}",
                preview_anchor(original, hinted, old.len())
            ));
            let trimmed_matches = find_trimmed_block_matches(original, old);
            match trimmed_matches.as_slice() {
                [] => {}
                [idx] => msg.push_str(&format!(
                    "\nWhitespace-trimmed OLD would match at L{}; preserve exact indentation/spacing in OLD.",
                    idx + 1
                )),
                many => msg.push_str(&format!(
                    "\nWhitespace-trimmed OLD would match {} locations: {}. Use a more specific OLD block.",
                    many.len(),
                    format_line_list(many)
                )),
            }
            bail!("{msg}");
        }
        _ => bail!(
            "OLD mismatch for {op_name} {start_line}: OLD block matched {} locations: {}. Use a more specific OLD block.\nOLD block:\n{}",
            matches.len(),
            format_line_list(&matches),
            preview_block(old, None)
        ),
    }
}

fn find_exact_block_matches(haystack: &[String], needle: &[String]) -> Vec<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return Vec::new();
    }

    let mut matches = Vec::new();
    for start in 0..=haystack.len() - needle.len() {
        if haystack[start..start + needle.len()] == *needle {
            matches.push(start);
        }
    }
    matches
}

fn find_trimmed_block_matches(haystack: &[String], needle: &[String]) -> Vec<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return Vec::new();
    }

    let mut matches = Vec::new();
    for start in 0..=haystack.len() - needle.len() {
        if haystack[start..start + needle.len()]
            .iter()
            .zip(needle)
            .all(|(left, right)| left.trim() == right.trim())
        {
            matches.push(start);
        }
    }
    matches
}

fn preview_anchor(original: &[String], start_idx: usize, desired_len: usize) -> String {
    if start_idx >= original.len() {
        return format!(
            "anchor L{} is beyond end of file ({} line(s))",
            start_idx + 1,
            original.len()
        );
    }

    let len = desired_len.max(1);
    let end = (start_idx + len).min(original.len());
    preview_block(&original[start_idx..end], Some(start_idx + 1))
}

fn preview_block(lines: &[String], first_line: Option<usize>) -> String {
    const MAX_PREVIEW_LINES: usize = 6;
    let mut out = String::new();
    for (idx, line) in lines.iter().take(MAX_PREVIEW_LINES).enumerate() {
        if !out.is_empty() {
            out.push('\n');
        }
        match first_line {
            Some(first) => out.push_str(&format!("L{}: {:?}", first + idx, line)),
            None => out.push_str(&format!("OLD{}: {:?}", idx + 1, line)),
        }
    }
    if lines.len() > MAX_PREVIEW_LINES {
        out.push_str(&format!(
            "\n... {} more line(s)",
            lines.len() - MAX_PREVIEW_LINES
        ));
    }
    out
}

fn format_line_list(indices: &[usize]) -> String {
    const MAX_LINES: usize = 8;
    let mut parts: Vec<String> = indices
        .iter()
        .take(MAX_LINES)
        .map(|idx| format!("L{}", idx + 1))
        .collect();
    if indices.len() > MAX_LINES {
        parts.push(format!("...{} more", indices.len() - MAX_LINES));
    }
    parts.join(", ")
}

fn reject_overlapping_spans(ops: &[ResolvedOp]) -> Result<()> {
    let mut spans: Vec<&ResolvedOp> = ops.iter().filter(|op| op.start != op.end).collect();
    spans.sort_unstable_by(|a, b| a.start.cmp(&b.start).then_with(|| a.end.cmp(&b.end)));

    for pair in spans.windows(2) {
        let prev = pair[0];
        let next = pair[1];
        if next.start < prev.end {
            bail!(
                "patch operations have overlapping replacement/delete spans: {} covers {}, overlaps {} covers {}. Use the smallest enclosing REPLACE_AT block for the overlap, split the patch into non-overlapping regions, or retry with a narrower edit_file task for one region/function.",
                prev.label,
                display_span(prev.start, prev.end),
                next.label,
                display_span(next.start, next.end),
            );
        }
    }

    Ok(())
}

fn display_span(start: usize, end: usize) -> String {
    if end <= start + 1 {
        format!("L{}", start + 1)
    } else {
        format!("L{}-L{}", start + 1, end)
    }
}

fn validate_insert_line(line: usize, total_lines: usize) -> Result<()> {
    if line == 0 || line > total_lines {
        bail!("insert line {line} out of range for {total_lines} line file");
    }
    Ok(())
}
