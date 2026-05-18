//! Per-snippet model-call helper.
//!
//! Given a window of source code and a description of how to rewrite a
//! specific call expression (or function signature) inside it, asks the
//! routed Fast model for a strict OLD/NEW block and applies the edit by
//! verbatim string replacement. The strict format is what makes this safe:
//! if the model paraphrases the OLD block, replacement won't match and we
//! report failure rather than mis-edit somewhere else.

use anyhow::{Result, anyhow, bail};

use crate::config::ModelRole;
use crate::llm::{ChatRequest, Message, ModelRouter};
use crate::logging::SessionLog;

/// One model-driven rewrite of a snippet.
#[derive(Debug, Clone)]
pub struct SnippetRewrite {
    pub old: String,
    pub new: String,
}

/// Prefix on errors that originated from a model SKIP. The retry loop
/// detects this prefix and stops retrying — SKIP is a deliberate refusal,
/// not a transient failure to escape via re-sampling.
const SKIP_PREFIX: &str = "model declined edit: ";

/// System prompt shared across all per-callsite / per-signature rewrites.
/// Keep it tight — the model only ever does one thing: emit OLD/NEW.
pub const SYSTEM_PROMPT: &str = "\
You apply a single localized code edit and output ONLY the result in this strict format:

OLD:
<exact lines from the input that you want to replace, byte-for-byte>
END_OLD
NEW:
<replacement lines>
END_NEW

Rules:
- The OLD block MUST appear verbatim in the input. Match indentation and whitespace exactly.
- Change ONLY what the instruction asks for. Leave all other code identical.
- No prose, no markdown, no code fences, no commentary outside the OLD/NEW block.
- If you cannot perform the edit safely, output exactly: SKIP\n<one-line reason>";

/// Ask the model to rewrite a snippet according to `instruction`.
/// Returns `Ok(Some(rewrite))` on a parsed OLD/NEW response, `Ok(None)` when
/// the model emits SKIP, or `Err` on transport / parse failure.
///
/// Retries up to 3 times for: empty response, parse failure. Server-side
/// KV-cache state varies between calls so a retry sometimes succeeds where
/// the previous attempt collapsed.
///
/// `tag` is a short identifier (e.g. `"signature"`, `"callsite:src/main.rs:42"`)
/// included in log entries so a benchmark trace can distinguish multiple calls.
pub async fn ask_rewrite(
    router: &ModelRouter,
    log: Option<&SessionLog>,
    tag: &str,
    instruction: &str,
    window: &str,
) -> Result<Option<SnippetRewrite>> {
    ask_rewrite_validated(router, log, tag, instruction, window, |_| {
        Ok::<(), anyhow::Error>(())
    })
    .await
}

/// Like [`ask_rewrite`] but takes a validator. After parsing OLD/NEW, the
/// validator is called with the parsed rewrite; if it returns an error
/// the call retries (up to 3 attempts total).
///
/// This is what callers use when "the model produced a parseable answer
/// but it doesn't actually fit the source" should also trigger a retry.
/// Common case: strict-anchor `apply_rewrite_at` rejecting a paraphrased
/// OLD — we want the model to try again, not just give up.
pub async fn ask_rewrite_validated<V, E>(
    router: &ModelRouter,
    log: Option<&SessionLog>,
    tag: &str,
    instruction: &str,
    window: &str,
    validate: V,
) -> Result<Option<SnippetRewrite>>
where
    V: Fn(&SnippetRewrite) -> std::result::Result<(), E>,
    E: std::fmt::Display,
{
    let base_prompt = format!(
        "Instruction:\n{instruction}\n\nSource snippet:\n```\n{window}\n```\n\n\
         Output the OLD/NEW block now."
    );
    // Prefill the start of OLD with the window's first line. The model
    // continues from there, so OLD's first line is *guaranteed* to equal
    // source[anchor_line] — this lets the caller use direct line-range
    // replacement instead of searching the file for OLD.
    //
    // The first line of `window` is the snippet's first line (we extract
    // windows starting at the LSP-resolved target line). If the window
    // is empty for some reason we skip prefill — the parser still works
    // on a model-generated full OLD/NEW block.
    let first_window_line = window.lines().next().unwrap_or("");
    let prefill = format!("OLD:\n{first_window_line}\n");
    if let Some(log) = log {
        log.tool_debug(
            "change_signature",
            &format!(
                "ask_rewrite[{tag}] request:\n--- system ---\n{SYSTEM_PROMPT}\n\
                 --- user ---\n{base_prompt}\n--- assistant prefill ---\n{prefill}"
            ),
        );
    }

    let mut last_error: Option<anyhow::Error> = None;
    for attempt in 1..=3u32 {
        // On retries, append a small format reminder. This both (a) gives
        // the model a hint when the previous attempt mis-formatted, and
        // (b) perturbs the prompt enough to shift the sampling
        // distribution away from whatever attractor caused the previous
        // failure. Matters at low temperatures where straight retries
        // tend to reproduce the same bad output.
        let prompt = if attempt == 1 {
            base_prompt.clone()
        } else {
            format!(
                "{base_prompt}\n\n(reminder: close OLD with END_OLD and NEW with END_NEW; output nothing outside the OLD/NEW block.)"
            )
        };
        match try_ask_once(router, log, tag, attempt, &prompt, &prefill).await {
            Ok(Some(rewrite)) => match validate(&rewrite) {
                Ok(()) => return Ok(Some(rewrite)),
                Err(e) => {
                    if let Some(log) = log {
                        log.tool_debug(
                            "change_signature",
                            &format!("ask_rewrite[{tag}] attempt={attempt} validate_failed: {e}"),
                        );
                    }
                    last_error = Some(anyhow!("validation failed: {e}"));
                }
            },
            Ok(None) => return Ok(None),
            Err(e) => {
                // SKIP is a deliberate refusal; retrying just produces
                // the same refusal with slightly different wording.
                // Surface it immediately so the agent can decide what
                // to do (most often the agent is calling refactor on an
                // already-modified function and should move on).
                if e.to_string().starts_with(SKIP_PREFIX) {
                    return Err(e);
                }
                last_error = Some(e);
            }
        }
        // No breather between attempts — server-side state doesn't change
        // meaningfully in milliseconds, retry variance comes from sampling.
    }
    Err(last_error.unwrap_or_else(|| anyhow!("ask_rewrite failed without error")))
}

/// Deterministic prefix llama-server's chat template prepends to a
/// continuation when we send a partial assistant message to Gemma 4 (and
/// likely related models). Empirically observed across 12 consecutive
/// runs: byte-identical, never varies. We strip it before assembling the
/// final OLD/NEW text. On servers that don't add this prefix the strip
/// is a no-op.
const CHAT_TEMPLATE_LEAK_PREFIX: &str = "<|channel>thought\n<channel|>";

/// One inference attempt. Logs the response and parse outcome individually
/// so the trace shows whether the retry was the one that landed.
///
/// `prefill` is sent as a partial assistant message; the model continues
/// from where it leaves off. Combined with stripping the chat-template
/// leak prefix, this gives us guaranteed control of OLD's first line
/// (the caller writes it) while letting the model fill in the rest.
async fn try_ask_once(
    router: &ModelRouter,
    log: Option<&SessionLog>,
    tag: &str,
    attempt: u32,
    prompt: &str,
    prefill: &str,
) -> Result<Option<SnippetRewrite>> {
    let request = ChatRequest {
        messages: vec![
            Message::system(SYSTEM_PROMPT),
            Message::user(prompt),
            Message::assistant(prefill),
        ],
        tools: None,
        tool_choice: None,
        max_tokens_override: None,
        // Disable reasoning-mode (Gemma 4 and similar models) for these
        // structured rewrite calls. The model previously spent thousands
        // of tokens of internal reasoning that went into `reasoning_content`
        // (which we don't read) before emitting the actual OLD/NEW; with
        // a small max_tokens that meant reasoning ate the whole budget
        // and content stayed empty. With reasoning off, the same prompts
        // return correct OLD/NEW in ~1-2 seconds. Servers that don't
        // recognise the field ignore it.
        chat_template_kwargs: Some(serde_json::json!({"enable_thinking": false})),
    };
    let response = match router.chat(ModelRole::Fast, &request).await {
        Ok(r) => r,
        Err(e) => {
            if let Some(log) = log {
                log.tool_debug(
                    "change_signature",
                    &format!("ask_rewrite[{tag}] attempt={attempt} transport_error: {e}"),
                );
            }
            return Err(e);
        }
    };
    let raw = response
        .choices
        .first()
        .and_then(|c| c.message.content.as_deref())
        .unwrap_or("");
    if let Some(log) = log {
        log.tool_debug(
            "change_signature",
            &format!(
                "ask_rewrite[{tag}] attempt={attempt} response (len={}):\n{raw}",
                raw.len()
            ),
        );
    }
    if raw.is_empty() {
        return Err(anyhow!("model returned no content"));
    }
    // Strip the chat-template leak if present, then prepend our prefill
    // so the assembled text is a complete OLD/NEW block the parser can
    // handle without special-casing.
    let stripped = raw.strip_prefix(CHAT_TEMPLATE_LEAK_PREFIX).unwrap_or(raw);
    let assembled = format!("{prefill}{stripped}");
    let parsed = parse_old_new(assembled.trim());
    if let Some(log) = log {
        match &parsed {
            Ok(Some(r)) => log.tool_debug(
                "change_signature",
                &format!(
                    "ask_rewrite[{tag}] attempt={attempt} parsed OLD={} bytes NEW={} bytes",
                    r.old.len(),
                    r.new.len()
                ),
            ),
            Ok(None) => log.tool_debug(
                "change_signature",
                &format!("ask_rewrite[{tag}] attempt={attempt} parsed: SKIP"),
            ),
            Err(e) => log.tool_debug(
                "change_signature",
                &format!("ask_rewrite[{tag}] attempt={attempt} parse_error: {e}"),
            ),
        }
    }
    parsed
}

/// Parse a strict OLD/NEW block. Tolerates leading/trailing whitespace and
/// optional surrounding code fences but rejects free-form prose around the
/// markers.
///
/// SKIP returns `Err` (not `Ok(None)`) because the caller expresses "the
/// model declined" through an error path that bubbles up to the agent.
/// `Ok(None)` is reserved for explicit SKIP-style outputs that should
/// halt the retry loop without surfacing as a failure — currently no
/// caller produces that, but the type leaves the door open.
pub fn parse_old_new(text: &str) -> Result<Option<SnippetRewrite>> {
    let stripped = strip_optional_fence(text.trim());
    if let Some(reason) = stripped.strip_prefix("SKIP") {
        // Use a stable error tag the retry loop can detect. We don't
        // retry SKIPs — the model deliberately refused, retrying just
        // makes it refuse again with slightly different wording.
        return Err(anyhow!(
            "{SKIP_PREFIX}{}",
            reason.trim().trim_start_matches('\n').trim()
        ));
    }
    let after_old = stripped
        .find("OLD:")
        .ok_or_else(|| anyhow!("model output missing OLD: marker"))?
        + "OLD:".len();
    let end_old_rel = stripped[after_old..]
        .find("END_OLD")
        .ok_or_else(|| anyhow!("model output missing END_OLD"))?;
    let old_raw = &stripped[after_old..after_old + end_old_rel];

    let after_end_old = after_old + end_old_rel + "END_OLD".len();
    let after_new = stripped[after_end_old..]
        .find("NEW:")
        .ok_or_else(|| anyhow!("model output missing NEW: marker"))?
        + "NEW:".len()
        + after_end_old;
    let end_new_rel = stripped[after_new..]
        .find("END_NEW")
        .ok_or_else(|| anyhow!("model output missing END_NEW"))?;
    let new_raw = &stripped[after_new..after_new + end_new_rel];

    let old = trim_block(old_raw);
    let new = trim_block(new_raw);
    if old.is_empty() {
        bail!("OLD block is empty");
    }
    Ok(Some(SnippetRewrite { old, new }))
}

fn strip_optional_fence(s: &str) -> &str {
    let s = s.strip_prefix("```").unwrap_or(s);
    // Drop a trailing closing fence if present.
    if let Some(end) = s.rfind("```") {
        // Only treat as a fence if it's at end-of-trimmed-string.
        if s[end..].trim() == "```" {
            return &s[..end];
        }
    }
    s
}

fn trim_block(s: &str) -> String {
    // Strip exactly one leading newline (the one right after `OLD:` or
    // `NEW:`) and one trailing newline (the one right before `END_*`),
    // but preserve all internal whitespace exactly.
    let s = s.strip_prefix('\n').unwrap_or(s);
    let s = s.strip_suffix('\n').unwrap_or(s);
    s.to_string()
}

/// Apply `rewrite` to `source` by replacing OLD with NEW, anchored at
/// `anchor_line` (0-based, file line numbers).
///
/// Assumption (enforced by the prefill design in `ask_rewrite`): OLD's
/// first line equals `source[anchor_line]`. We verify each subsequent
/// OLD line matches `source[anchor_line + k]` (whitespace-tolerant on
/// trailing edge: trailing spaces and `\r` are stripped before compare;
/// leading whitespace must match exactly because indentation is
/// structurally meaningful in code).
///
/// On any line mismatch we error loud — the retry loop in
/// `ask_rewrite_validated` will give the model another shot.
pub fn apply_rewrite(source: &str, rewrite: &SnippetRewrite, anchor_line: u32) -> Result<String> {
    let old_lines: Vec<&str> = rewrite.old.lines().collect();
    if old_lines.is_empty() {
        bail!("OLD block is empty");
    }
    let old_trimmed: Vec<String> = old_lines
        .iter()
        .map(|l| l.trim_end().trim_end_matches('\r').to_string())
        .collect();

    let mut line_starts = vec![0usize];
    for (i, c) in source.char_indices() {
        if c == '\n' {
            line_starts.push(i + 1);
        }
    }
    // Use `lines().count()` for the *content* line count (matches what
    // a user would call "the file has N lines"). `line_starts` may have
    // one extra phantom entry pointing at the byte after a trailing `\n`,
    // which we don't want to count as a real line.
    let line_count = source.lines().count();
    let line_at = |idx: usize| -> &str {
        let s = line_starts[idx];
        let e = line_starts.get(idx + 1).copied().unwrap_or(source.len());
        let raw = &source[s..e];
        let raw = raw.strip_suffix('\n').unwrap_or(raw);
        raw.strip_suffix('\r').unwrap_or(raw)
    };

    let anchor = anchor_line as usize;
    if anchor >= line_count {
        bail!(
            "anchor line {} past end of file ({} lines)",
            anchor_line,
            line_count
        );
    }
    if anchor + old_trimmed.len() > line_count {
        bail!(
            "OLD ({} lines) extends past end of file from anchor line {}",
            old_trimmed.len(),
            anchor_line + 1
        );
    }
    for (k, expected) in old_trimmed.iter().enumerate() {
        let src_line = line_at(anchor + k).trim_end().trim_end_matches('\r');
        if src_line != expected {
            bail!(
                "OLD line {} doesn't match source at line {} (anchor {anchor_line}). \
                 Expected {:?}, found {:?}.",
                k + 1,
                anchor + k + 1,
                expected,
                src_line,
            );
        }
    }

    let byte_start = line_starts[anchor];
    let byte_end = line_starts
        .get(anchor + old_trimmed.len())
        .copied()
        .unwrap_or(source.len());

    let mut out = String::with_capacity(source.len() + rewrite.new.len());
    out.push_str(&source[..byte_start]);
    out.push_str(&rewrite.new);
    if !rewrite.new.ends_with('\n') && byte_end < source.len() && !rewrite.new.is_empty() {
        out.push('\n');
    }
    out.push_str(&source[byte_end..]);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_round_trip() {
        let text = "OLD:\n  foo();\nEND_OLD\nNEW:\n  foo(None);\nEND_NEW";
        let r = parse_old_new(text).unwrap().unwrap();
        assert_eq!(r.old, "  foo();");
        assert_eq!(r.new, "  foo(None);");
    }

    #[test]
    fn parse_strips_fences() {
        let text = "```\nOLD:\nx\nEND_OLD\nNEW:\ny\nEND_NEW\n```";
        let r = parse_old_new(text).unwrap().unwrap();
        assert_eq!(r.old, "x");
        assert_eq!(r.new, "y");
    }

    #[test]
    fn parse_skip_returns_err() {
        let err = parse_old_new("SKIP\nambiguous arg slot").unwrap_err();
        assert!(err.to_string().contains("ambiguous arg slot"));
    }

    #[test]
    fn parse_missing_marker_errs() {
        assert!(parse_old_new("OLD:\nx\nEND_OLD\nNEW:\ny").is_err());
        assert!(parse_old_new("just some prose").is_err());
    }

    #[test]
    fn apply_rewrite_single_line() {
        // OLD's first line equals source[anchor_line=1]. Replace 1 line.
        let src = "fn main() {\n    foo();\n    bar();\n}\n";
        let rewrite = SnippetRewrite {
            old: "    foo();".into(),
            new: "    foo(None);".into(),
        };
        let out = apply_rewrite(src, &rewrite, 1).unwrap();
        assert_eq!(out, "fn main() {\n    foo(None);\n    bar();\n}\n");
    }

    #[test]
    fn apply_rewrite_multi_line() {
        // OLD covers 2 lines, NEW covers 2 lines; structure preserved.
        let src = "fn x() {\n    a;\n    b;\n    c;\n}\n";
        let rewrite = SnippetRewrite {
            old: "    a;\n    b;".into(),
            new: "    a;\n    b_changed;".into(),
        };
        let out = apply_rewrite(src, &rewrite, 1).unwrap();
        assert_eq!(out, "fn x() {\n    a;\n    b_changed;\n    c;\n}\n");
    }

    #[test]
    fn apply_rewrite_new_has_more_lines_than_old() {
        // NEW adds lines (typical add_param case).
        let src = "fn x() {\n    foo(\n        a,\n        b,\n    );\n}\n";
        let rewrite = SnippetRewrite {
            old: "    foo(\n        a,\n        b,\n    );".into(),
            new: "    foo(\n        a,\n        b,\n        None,\n    );".into(),
        };
        let out = apply_rewrite(src, &rewrite, 1).unwrap();
        assert_eq!(
            out,
            "fn x() {\n    foo(\n        a,\n        b,\n        None,\n    );\n}\n"
        );
    }

    #[test]
    fn apply_rewrite_rejects_when_first_line_differs() {
        // OLD's first line doesn't match source[anchor]; we don't search,
        // we fail loud — caller's retry loop gets another shot.
        let src = "fn main() {\n    foo();\n    bar();\n}\n";
        let rewrite = SnippetRewrite {
            old: "    baz();".into(),
            new: "    baz(None);".into(),
        };
        let err = apply_rewrite(src, &rewrite, 1).unwrap_err();
        assert!(
            err.to_string().contains("doesn't match"),
            "expected mismatch error, got: {err}"
        );
    }

    #[test]
    fn apply_rewrite_rejects_when_middle_line_differs() {
        // OLD line 2 doesn't match source[anchor+1]; partial match still fails.
        let src = "    a;\n    b;\n    c;\n";
        let rewrite = SnippetRewrite {
            old: "    a;\n    DIFFERENT;".into(),
            new: "    a;\n    new_b;".into(),
        };
        let err = apply_rewrite(src, &rewrite, 0).unwrap_err();
        assert!(
            err.to_string().contains("doesn't match"),
            "expected mismatch error, got: {err}"
        );
    }

    #[test]
    fn apply_rewrite_rejects_when_old_overflows_file() {
        let src = "    a;\n";
        let rewrite = SnippetRewrite {
            old: "    a;\n    b;".into(),
            new: "    new;".into(),
        };
        let err = apply_rewrite(src, &rewrite, 0).unwrap_err();
        assert!(
            err.to_string().contains("extends past"),
            "expected overflow error, got: {err}"
        );
    }

    #[test]
    fn apply_rewrite_rejects_anchor_past_eof() {
        let src = "    a;\n";
        let rewrite = SnippetRewrite {
            old: "    a;".into(),
            new: "    new;".into(),
        };
        let err = apply_rewrite(src, &rewrite, 99).unwrap_err();
        assert!(
            err.to_string().contains("past end of file"),
            "expected past-end error, got: {err}"
        );
    }

    #[test]
    fn apply_rewrite_tolerates_trailing_whitespace_in_old() {
        // Model emitted OLD with trailing space that source doesn't have.
        let src = "fn main() {\n    foo();\n    bar();\n}\n";
        let rewrite = SnippetRewrite {
            old: "    foo();   ".into(),
            new: "    foo(None);".into(),
        };
        let out = apply_rewrite(src, &rewrite, 1).unwrap();
        assert_eq!(out, "fn main() {\n    foo(None);\n    bar();\n}\n");
    }

    #[test]
    fn apply_rewrite_tolerates_trailing_whitespace_in_source() {
        let src = "fn main() {\n    foo();   \n    bar();\n}\n";
        let rewrite = SnippetRewrite {
            old: "    foo();".into(),
            new: "    foo(None);".into(),
        };
        let out = apply_rewrite(src, &rewrite, 1).unwrap();
        assert_eq!(out, "fn main() {\n    foo(None);\n    bar();\n}\n");
    }

    #[test]
    fn apply_rewrite_tolerates_crlf() {
        let src = "fn main() {\r\n    foo();\r\n    bar();\r\n}\r\n";
        let rewrite = SnippetRewrite {
            old: "    foo();".into(),
            new: "    foo(None);".into(),
        };
        let out = apply_rewrite(src, &rewrite, 1).unwrap();
        assert!(out.contains("foo(None);"));
        assert!(out.contains("bar();"));
    }
}
