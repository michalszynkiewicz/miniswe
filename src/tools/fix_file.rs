//! fix_file tool — LLM generates a strict patch, miniswe applies it atomically.
//!
//! The model describes the task, miniswe sends file content to the LLM, and the
//! inner LLM returns a small patch DSL. Patches are dry-run validated before any
//! write. If the broad patch path fails, fix_file falls back to smaller,
//! non-overlapping line regions and validates the combined result before writing.

use anyhow::{Result, bail};
use serde_json::Value;

use super::ToolResult;
use crate::config::{Config, ModelRole};
use crate::llm::{ChatRequest, Message, ModelRouter};

/// Max lines per window for reliable LLM recall.
const WINDOW_SIZE: usize = 800;
/// Overlap between windows to catch edits at boundaries.
const WINDOW_OVERLAP: usize = 100;
const MAX_PATCH_ATTEMPTS: usize = 3;
const MAX_SPLIT_REGIONS: usize = 3;
const LARGE_TRUNCATION_MIN_LINES: usize = 50;

pub async fn execute(args: &Value, config: &Config, router: &ModelRouter) -> Result<ToolResult> {
    let path_str = args["path"].as_str().unwrap_or("");
    let task = args["task"].as_str().unwrap_or("");

    if path_str.is_empty() {
        return Ok(ToolResult::err("Missing required parameter: path".into()));
    }
    if task.is_empty() {
        return Ok(ToolResult::err("Missing required parameter: task".into()));
    }

    let path = config.project_root.join(path_str);
    if !path.exists() {
        return Ok(ToolResult::err(format!("File not found: {path_str}")));
    }

    let content = std::fs::read_to_string(&path)
        .map_err(|e| anyhow::anyhow!("Failed to read {path_str}: {e}"))?;

    let mut feedback: Option<String> = None;
    let mut last_error = String::new();

    for attempt in 1..=MAX_PATCH_ATTEMPTS {
        let (ops, mut output) =
            match request_patch(path_str, task, &content, router, feedback.as_deref()).await {
                Ok(r) => r,
                Err(e) => {
                    last_error = e.to_string();
                    if attempt < MAX_PATCH_ATTEMPTS {
                        feedback = Some(last_error.clone());
                        continue;
                    }
                    break;
                }
            };

        if ops.is_empty() {
            return Ok(ToolResult::ok(format!(
                "No changes needed in {path_str} for task: {task}"
            )));
        }

        let candidate = match apply_patch_dry_run(&content, &ops) {
            Ok(candidate) => candidate,
            Err(e) => {
                last_error = e.to_string();
                if attempt < MAX_PATCH_ATTEMPTS {
                    feedback = Some(last_error.clone());
                    continue;
                }
                break;
            }
        };

        if let Err(e) = validate_candidate(path_str, &content, &candidate) {
            last_error = e.to_string();
            if attempt < MAX_PATCH_ATTEMPTS {
                feedback = Some(last_error.clone());
                continue;
            }
            break;
        }

        std::fs::write(&path, &candidate)?;
        output.push_str(&format!(
            "✓ Applied {} operation(s) to {path_str} ({} lines)\n",
            ops.len(),
            candidate.lines().count()
        ));
        return Ok(ToolResult::ok(output));
    }

    match execute_split_fallback(path_str, task, &content, router, &last_error).await {
        Ok(Some(candidate_result)) => {
            std::fs::write(&path, &candidate_result.content)?;
            return Ok(ToolResult::ok(candidate_result.message));
        }
        Ok(None) => {}
        Err(split_error) => {
            return Ok(ToolResult::err(format!(
                "fix_file failed: patch was not applied.\nReason: {last_error}\nSplit fallback failed: {split_error}\n"
            )));
        }
    }

    Ok(ToolResult::err(format!(
        "fix_file failed: patch was not applied.\nReason: {last_error}\n"
    )))
}

struct SplitResult {
    content: String,
    message: String,
}

async fn execute_split_fallback(
    path_str: &str,
    task: &str,
    original: &str,
    router: &ModelRouter,
    broad_error: &str,
) -> Result<Option<SplitResult>> {
    let regions = request_region_plan(path_str, task, original, router, broad_error).await?;
    if regions.is_empty() {
        return Ok(None);
    }

    let mut current = original.to_string();
    let mut total_ops = 0usize;
    let mut message = format!(
        "Broad patch failed: {broad_error}\nSplit fallback: {} region(s) planned\n",
        regions.len()
    );

    let mut regions_desc = regions;
    regions_desc.sort_by(|a, b| b.start.cmp(&a.start).then_with(|| b.end.cmp(&a.end)));

    for (idx, region) in regions_desc.iter().enumerate() {
        let mut feedback: Option<String> = None;
        let mut last_region_error = String::new();
        let region_label = format!("region {} L{}-L{}", idx + 1, region.start, region.end);

        for attempt in 1..=MAX_PATCH_ATTEMPTS {
            let (ops, _) = match request_patch_for_region(
                path_str,
                &region.task,
                &current,
                router,
                region,
                feedback.as_deref(),
            )
            .await
            {
                Ok(result) => result,
                Err(e) => {
                    last_region_error = e.to_string();
                    if attempt < MAX_PATCH_ATTEMPTS {
                        feedback = Some(last_region_error.clone());
                        continue;
                    }
                    break;
                }
            };

            if ops.is_empty() {
                message.push_str(&format!("Split {region_label}: no changes\n"));
                last_region_error.clear();
                break;
            }

            let candidate =
                match apply_patch_dry_run_in_region(&current, &ops, region.start, region.end) {
                    Ok(candidate) => candidate,
                    Err(e) => {
                        last_region_error = e.to_string();
                        if attempt < MAX_PATCH_ATTEMPTS {
                            feedback = Some(last_region_error.clone());
                            continue;
                        }
                        break;
                    }
                };

            if let Err(e) = validate_candidate(path_str, &current, &candidate) {
                last_region_error = e.to_string();
                if attempt < MAX_PATCH_ATTEMPTS {
                    feedback = Some(last_region_error.clone());
                    continue;
                }
                break;
            }

            total_ops += ops.len();
            current = candidate;
            message.push_str(&format!(
                "Split {region_label}: applied {} operation(s)\n",
                ops.len()
            ));
            last_region_error.clear();
            break;
        }

        if !last_region_error.is_empty() {
            bail!("{region_label} failed: {last_region_error}");
        }
    }

    validate_candidate(path_str, original, &current)?;
    message.push_str(&format!(
        "✓ Applied {total_ops} operation(s) to {path_str} via split fallback ({} lines)\n",
        current.lines().count()
    ));

    Ok(Some(SplitResult {
        content: current,
        message,
    }))
}

async fn request_patch(
    path_str: &str,
    task: &str,
    content: &str,
    router: &ModelRouter,
    repair_feedback: Option<&str>,
) -> Result<(Vec<PatchOp>, String)> {
    let lines: Vec<&str> = content.lines().collect();
    let total_lines = lines.len();
    let windows = build_windows(total_lines, WINDOW_SIZE, 0);
    let mut all_ops = Vec::new();
    let mut output = String::new();

    for (win_idx, (start, end)) in windows.iter().enumerate() {
        let window_content = lines[*start..*end]
            .iter()
            .enumerate()
            .map(|(i, l)| format!("{:>4}│{}", start + i + 1, l))
            .collect::<Vec<_>>()
            .join("\n");

        let window_info = if windows.len() > 1 {
            format!(
                "(window {}/{}, lines {}-{} of {})",
                win_idx + 1,
                windows.len(),
                start + 1,
                end,
                total_lines
            )
        } else {
            format!("({total_lines} lines)")
        };

        let repair = repair_feedback
            .map(|f| {
                format!(
                    "\nPrevious patch was not applied.\nFailure: {f}\nReturn a corrected patch against the original file. If the failure mentions overlapping spans, use the smallest enclosing REPLACE_AT block that covers the overlap, or split the patch into separate non-overlapping regions. Do not rewrite a much larger block just to avoid overlap.\n"
                )
            })
            .unwrap_or_default();

        let prompt = format!(
            "You are editing one file: {path_str} {window_info}.\n\
             Task: {task}\n\
             {repair}\n\
             Return a complete patch for all changes needed in this file/window.\n\
             Use this patch DSL exactly:\n\n\
             INSERT_BEFORE <line>\n\
             CONTENT:\n\
             <lines to insert>\n\
             END\n\n\
             INSERT_AFTER <line>\n\
             CONTENT:\n\
             <lines to insert>\n\
             END\n\n\
             REPLACE_AT <start_line>\n\
             OLD:\n\
             <exact original lines>\n\
             END_OLD\n\
             NEW:\n\
             <replacement lines>\n\
             END_NEW\n\n\
             DELETE_AT <start_line>\n\
             OLD:\n\
             <exact original lines>\n\
             END_OLD\n\n\
             Rules:\n\
             - Output ONLY patch DSL blocks, no markdown or explanations.\n\
             - If no changes are needed, output exactly NO_CHANGES.\n\
             - Line numbers refer to the original file shown below, before any operations apply.\n\
             - For REPLACE_AT/DELETE_AT, OLD determines how many lines are changed.\n\
             - Prefer small, non-overlapping operations. Do not output overlapping REPLACE_AT/DELETE_AT operations.\n\
             - If two edits overlap, use the smallest enclosing REPLACE_AT block that covers the overlap; do not rewrite a much larger block.\n\
             - Preserve indentation and blank lines exactly inside CONTENT/OLD/NEW.\n\
             - Validation is atomic: if any operation fails, no changes are applied.\n\n\
             File content:\n{window_content}"
        );

        let request = ChatRequest {
            messages: vec![
                Message::system(
                    "You output only strict patch DSL blocks. No explanations, no markdown.",
                ),
                Message::user(&prompt),
            ],
            tools: None,
            tool_choice: None,
        };

        let response = router.chat(ModelRole::Fast, &request).await?;
        let text = response
            .choices
            .first()
            .and_then(|c| c.message.content.as_deref())
            .unwrap_or("");

        let ops = parse_patch(text)?;
        if !ops.is_empty() {
            output.push_str(&format!(
                "Window {}: {} operation(s) found\n",
                win_idx + 1,
                ops.len()
            ));
        }
        all_ops.extend(ops);
    }

    Ok((all_ops, output))
}

async fn request_patch_for_region(
    path_str: &str,
    task: &str,
    content: &str,
    router: &ModelRouter,
    region: &EditRegion,
    repair_feedback: Option<&str>,
) -> Result<(Vec<PatchOp>, String)> {
    let lines: Vec<&str> = content.lines().collect();
    if region.start == 0 || region.end < region.start || region.end > lines.len() {
        bail!(
            "invalid edit region L{}-L{} for {} line file",
            region.start,
            region.end,
            lines.len()
        );
    }

    let region_content = lines[region.start - 1..region.end]
        .iter()
        .enumerate()
        .map(|(i, line)| format!("{:>4}│{}", region.start + i, line))
        .collect::<Vec<_>>()
        .join("\n");

    let repair = repair_feedback
        .map(|f| {
            format!(
                "\nPrevious region patch was not applied.\nFailure: {f}\nReturn a corrected patch for this same line region only.\n"
            )
        })
        .unwrap_or_default();

    let prompt = format!(
        "You are editing one line region in {path_str}: lines {}-{}.\n\
         Task: {task}\n\
         {repair}\n\
         You may edit ONLY lines {}-{}. Do not target lines outside this region.\n\
         Return a complete patch for this region using the patch DSL exactly:\n\n\
         INSERT_BEFORE <line>\n\
         CONTENT:\n\
         <lines to insert>\n\
         END\n\n\
         INSERT_AFTER <line>\n\
         CONTENT:\n\
         <lines to insert>\n\
         END\n\n\
         REPLACE_AT <start_line>\n\
         OLD:\n\
         <exact original lines>\n\
         END_OLD\n\
         NEW:\n\
         <replacement lines>\n\
         END_NEW\n\n\
         DELETE_AT <start_line>\n\
         OLD:\n\
         <exact original lines>\n\
         END_OLD\n\n\
         Rules:\n\
         - Output ONLY patch DSL blocks, no markdown or explanations.\n\
         - If no changes are needed, output exactly NO_CHANGES.\n\
         - Preserve indentation and blank lines exactly inside CONTENT/OLD/NEW.\n\
         - Keep edits small and inside the allowed line region.\n\n\
         Region content:\n{region_content}",
        region.start, region.end, region.start, region.end
    );

    let request = ChatRequest {
        messages: vec![
            Message::system("You output only strict patch DSL blocks. No explanations, no markdown."),
            Message::user(&prompt),
        ],
        tools: None,
        tool_choice: None,
    };

    let response = router.chat(ModelRole::Fast, &request).await?;
    let text = response
        .choices
        .first()
        .and_then(|c| c.message.content.as_deref())
        .unwrap_or("");
    let ops = parse_patch(text)?;
    let output = if ops.is_empty() {
        String::new()
    } else {
        format!(
            "Region L{}-L{}: {} operation(s) found\n",
            region.start,
            region.end,
            ops.len()
        )
    };
    Ok((ops, output))
}

async fn request_region_plan(
    path_str: &str,
    task: &str,
    content: &str,
    router: &ModelRouter,
    broad_error: &str,
) -> Result<Vec<EditRegion>> {
    let lines: Vec<&str> = content.lines().collect();
    let total_lines = lines.len();
    let windows = build_windows(total_lines, WINDOW_SIZE, WINDOW_OVERLAP);
    let mut regions = Vec::new();

    for (win_idx, (start, end)) in windows.iter().enumerate() {
        if regions.len() >= MAX_SPLIT_REGIONS {
            break;
        }

        let window_content = lines[*start..*end]
            .iter()
            .enumerate()
            .map(|(i, line)| format!("{:>4}│{}", start + i + 1, line))
            .collect::<Vec<_>>()
            .join("\n");
        let remaining = MAX_SPLIT_REGIONS - regions.len();

        let prompt = format!(
            "A broad patch for {path_str} failed.\n\
             Failure: {broad_error}\n\
             Original task: {task}\n\n\
             Break the task into up to {remaining} small, non-overlapping line regions within this window.\n\
             Each region must be the smallest contiguous block that can be edited independently.\n\
             For code, prefer functions/classes/import blocks. For YAML/TOML/JSON/Markdown/config files, prefer logical sections or key blocks.\n\
             If the task needs only one region in this window, return one region. If no region is needed in this window, output exactly NO_REGIONS.\n\n\
             Output only this format:\n\
             REGION <start_line> <end_line>\n\
             TASK: <specific subtask for this region>\n\
             END\n\n\
             Window {}/{} lines {}-{} of {}:\n{window_content}",
            win_idx + 1,
            windows.len(),
            start + 1,
            end,
            total_lines
        );

        let request = ChatRequest {
            messages: vec![
                Message::system("You output only strict REGION blocks. No explanations, no markdown."),
                Message::user(&prompt),
            ],
            tools: None,
            tool_choice: None,
        };

        let response = router.chat(ModelRole::Fast, &request).await?;
        let text = response
            .choices
            .first()
            .and_then(|c| c.message.content.as_deref())
            .unwrap_or("");

        let mut planned = parse_region_plan(text)?;
        for region in &planned {
            if region.start < start + 1 || region.end > *end {
                bail!(
                    "planned region L{}-L{} falls outside window L{}-L{}",
                    region.start,
                    region.end,
                    start + 1,
                    end
                );
            }
        }
        regions.append(&mut planned);
        if regions.len() > MAX_SPLIT_REGIONS {
            regions.truncate(MAX_SPLIT_REGIONS);
        }
    }

    reject_overlapping_regions(&regions)?;
    Ok(regions)
}

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

pub fn parse_region_plan(text: &str) -> Result<Vec<EditRegion>> {
    if text.trim() == "NO_REGIONS" {
        return Ok(Vec::new());
    }
    if text.trim().is_empty() {
        bail!("empty region plan");
    }

    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;
    let mut regions = Vec::new();

    while i < lines.len() {
        let line = lines[i];
        if line.trim().is_empty() {
            i += 1;
            continue;
        }

        let Some(rest) = line.strip_prefix("REGION ") else {
            bail!("unexpected text in region plan: {line}");
        };
        let mut parts = rest.split_whitespace();
        let start = parts
            .next()
            .ok_or_else(|| anyhow::anyhow!("missing region start"))?
            .parse::<usize>()?;
        let end = parts
            .next()
            .ok_or_else(|| anyhow::anyhow!("missing region end"))?
            .parse::<usize>()?;
        if parts.next().is_some() {
            bail!("too many fields in region header: {line}");
        }
        if start == 0 || end < start {
            bail!("invalid region L{start}-L{end}");
        }

        i += 1;
        let task_line = lines
            .get(i)
            .ok_or_else(|| anyhow::anyhow!("missing TASK line for region"))?;
        let task = task_line
            .strip_prefix("TASK:")
            .ok_or_else(|| anyhow::anyhow!("expected TASK line but found '{task_line}'"))?
            .trim()
            .to_string();
        if task.is_empty() {
            bail!("region task must not be empty");
        }

        i += 1;
        expect_line(&lines, i, "END")?;
        i += 1;

        regions.push(EditRegion { start, end, task });
    }

    if regions.len() > MAX_SPLIT_REGIONS {
        bail!(
            "region plan returned {} regions, maximum is {MAX_SPLIT_REGIONS}",
            regions.len()
        );
    }
    reject_overlapping_regions(&regions)?;
    Ok(regions)
}

fn reject_overlapping_regions(regions: &[EditRegion]) -> Result<()> {
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

/// Parse strict patch DSL blocks.
pub fn parse_patch(text: &str) -> Result<Vec<PatchOp>> {
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

fn parse_line_number(text: &str) -> Result<usize> {
    let line = text.trim().parse::<usize>()?;
    if line == 0 {
        bail!("line numbers are 1-based");
    }
    Ok(line)
}

fn expect_line(lines: &[&str], idx: usize, expected: &str) -> Result<()> {
    match lines.get(idx) {
        Some(line) if *line == expected => Ok(()),
        Some(line) => bail!("expected '{expected}' but found '{line}'"),
        None => bail!("expected '{expected}' but reached end of patch"),
    }
}

fn collect_until(lines: &[&str], start: usize, sentinel: &str) -> Result<(Vec<String>, usize)> {
    let mut collected = Vec::new();
    for (idx, line) in lines.iter().enumerate().skip(start) {
        if *line == sentinel {
            return Ok((collected, idx));
        }
        collected.push((*line).to_string());
    }
    bail!("missing sentinel '{sentinel}'");
}

/// Apply all operations to memory only. If any operation fails, returns an
/// error and the original file on disk remains untouched.
pub fn apply_patch_dry_run(content: &str, ops: &[PatchOp]) -> Result<String> {
    let lines: Vec<String> = content.lines().map(str::to_string).collect();
    let resolved = resolve_ops(&lines, ops)?;
    apply_resolved_patch(content, resolved)
}

fn apply_patch_dry_run_in_region(
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
            format!("op {ordinal} REPLACE_AT {start} ({} OLD line(s))", old.len())
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
        [] => bail!(
            "OLD mismatch for {op_name} {start_line}: OLD block was not found at the anchor or elsewhere"
        ),
        _ => bail!(
            "OLD mismatch for {op_name} {start_line}: OLD block matched {} locations",
            matches.len()
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

fn reject_overlapping_spans(ops: &[ResolvedOp]) -> Result<()> {
    let mut spans: Vec<&ResolvedOp> = ops
        .iter()
        .filter(|op| op.start != op.end)
        .collect();
    spans.sort_unstable_by(|a, b| a.start.cmp(&b.start).then_with(|| a.end.cmp(&b.end)));

    for pair in spans.windows(2) {
        let prev = pair[0];
        let next = pair[1];
        if next.start < prev.end {
            bail!(
                "patch operations have overlapping replacement/delete spans: {} covers {}, overlaps {} covers {}. Use the smallest enclosing REPLACE_AT block for the overlap, split the patch into non-overlapping regions, or retry with a narrower fix_file task for one region/function.",
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

fn validate_candidate(path_str: &str, original: &str, candidate: &str) -> Result<()> {
    if !original.is_empty() && candidate.is_empty() {
        bail!("candidate output is empty for a non-empty file");
    }

    let old_lines = original.lines().count();
    let new_lines = candidate.lines().count();
    if old_lines > LARGE_TRUNCATION_MIN_LINES && new_lines < old_lines / 2 {
        bail!("candidate truncates file from {old_lines} to {new_lines} lines");
    }

    if is_brace_file(path_str) {
        let old_balance = delimiter_imbalance(original);
        let new_balance = delimiter_imbalance(candidate);
        if new_balance > old_balance {
            bail!("candidate worsens delimiter balance from {old_balance} to {new_balance}");
        }
    }

    Ok(())
}

fn is_brace_file(path: &str) -> bool {
    let brace_exts = [
        "rs", "js", "ts", "tsx", "jsx", "go", "java", "c", "cpp", "h", "hpp", "cs", "kt", "swift",
        "scala", "zig",
    ];
    brace_exts
        .iter()
        .any(|ext| path.ends_with(&format!(".{ext}")))
}

fn delimiter_imbalance(text: &str) -> i64 {
    let parens = text.matches('(').count() as i64 - text.matches(')').count() as i64;
    let braces = text.matches('{').count() as i64 - text.matches('}').count() as i64;
    let brackets = text.matches('[').count() as i64 - text.matches(']').count() as i64;
    parens.abs() + braces.abs() + brackets.abs()
}

/// Build window ranges for a file.
pub fn build_windows(
    total_lines: usize,
    window_size: usize,
    overlap: usize,
) -> Vec<(usize, usize)> {
    if total_lines <= window_size {
        return vec![(0, total_lines)];
    }

    let mut windows = Vec::new();
    let mut start = 0;
    while start < total_lines {
        let end = (start + window_size).min(total_lines);
        windows.push((start, end));
        if end >= total_lines {
            break;
        }
        start = end - overlap;
    }
    windows
}
