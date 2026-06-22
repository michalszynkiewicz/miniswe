//! `add_param`: add a parameter to a function signature and update every
//! callsite to pass a literal at the new slot.
//!
//! Flow:
//! 1. Read the function file, grab a window around the definition line.
//! 2. Ask the model to update the signature (one `OLD:`/`NEW:` block).
//! 3. LSP `find_references` → callsite list.
//! 4. For each callsite, ask the model to insert `default` at the
//!    matching slot (based on `position`).
//! 5. Apply all edits in memory, then write each file once.
//! 6. Return a per-callsite report.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde_json::Value;

use crate::config::Config;
use crate::llm::ModelRouter;
use crate::logging::SessionLog;
use crate::lsp::LspClient;
use crate::tools::ToolResult;
use crate::tools::args;

use crate::tools::fast::RevisionStore;

use super::model_edit::{apply_rewrite, ask_rewrite_validated};
use super::sites::{
    StagedEdit, commit_staged, ensure_ready, extract_window, find_callsites,
    resolve_function_location,
};
use super::validation::{ArgSchema, validate};

const ADD_PARAM_EXAMPLE: &str = "refactor(action=\"add_param\", path=\"src/lib.rs\", name=\"assemble\", new_param=\"x: u32\", position=\"after:b\", callsite_fill_in=\"0\")";

const ADD_PARAM_SCHEMA: ArgSchema<'static> = ArgSchema {
    action: "add_param",
    required_strings: &["path", "name", "new_param", "position", "callsite_fill_in"],
    optional_strings: &[],
    optional_ints: &["line"],
    example: ADD_PARAM_EXAMPLE,
};

/// Where to insert the new parameter relative to the existing list.
#[derive(Debug, Clone)]
pub enum Position {
    Start,
    After(String),
    /// Append after all existing parameters. The footgun-free default
    /// used by the flat `add_function_param` tool (`tools.flat`) — no
    /// `after:<name>` anchor to mangle. Also a valid grouped value.
    End,
}

impl Position {
    fn parse(raw: &str) -> Result<Self> {
        let trimmed = raw.trim();
        if trimmed == "start" {
            return Ok(Self::Start);
        }
        if trimmed == "end" || trimmed == "append" {
            return Ok(Self::End);
        }
        if let Some(rest) = trimmed.strip_prefix("after:") {
            let name = rest.trim();
            if name.is_empty() {
                bail!(Self::malformed_position_error(raw));
            }
            // Strict identifier check. Devstral (and likely other small models)
            // get primed by source-file content in context and emit positions
            // like "after:plan_only: bool, mcp_summary: Option<&str>" — i.e.
            // the entire source-file parameter list. Empirically (probe in
            // /tmp/devstral-probe4.py) the model recovers reliably only when
            // the error message names the malformation explicitly. Don't
            // accept anything that isn't a single Rust identifier.
            if !is_rust_ident(name) {
                bail!(Self::malformed_position_error(raw));
            }
            return Ok(Self::After(name.to_string()));
        }
        bail!(Self::malformed_position_error(raw))
    }

    fn malformed_position_error(raw: &str) -> String {
        format!(
            "the 'position' value you sent ({raw:?}) is malformed. \
             The 'position' field accepts ONLY one of: 'start', 'end', \
             or 'after:<single_param_name>' (e.g. 'after:mcp_summary'). \
             Do NOT include parameter types (like ': u32'), doc comments, \
             commas, or multiple parameter names — just the literal anchor."
        )
    }

    fn human(&self) -> String {
        match self {
            Self::Start => {
                "at the start of the parameter list (before all existing parameters)".to_string()
            }
            Self::After(name) => format!("immediately after the existing parameter `{name}`"),
            Self::End => {
                "at the end of the parameter list (after all existing parameters)".to_string()
            }
        }
    }
}

fn is_rust_ident(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

pub async fn execute(
    args: &Value,
    config: &Config,
    router: &ModelRouter,
    lsp: Option<&LspClient>,
    log: Option<&SessionLog>,
    revisions: Option<&RevisionStore>,
    cancelled: Option<&AtomicBool>,
) -> Result<ToolResult> {
    // Run schema validation and position-format check together so the
    // model gets both problems in one error. Devstral has been seen to
    // omit name/new_param/default *and* malform position in the same
    // call — surfacing only the missing-keys part doesn't unstick it
    // because the malformed shape gets copied verbatim on retry.
    let basic = validate(args, &ADD_PARAM_SCHEMA);
    let pos_problem = match args.get("position").and_then(|v| v.as_str()) {
        Some(raw) => Position::parse(raw).err().map(|e| e.to_string()),
        None => None,
    };
    match (basic, pos_problem) {
        (Ok(()), None) => {}
        (Err(b), None) => return Ok(ToolResult::err(b)),
        (Ok(()), Some(p)) => {
            return Ok(ToolResult::err(format!(
                "✗ change_signature(add_param): {p}"
            )));
        }
        (Err(b), Some(p)) => return Ok(ToolResult::err(format!("{b}\n\nAlso: {p}"))),
    }

    let path_str = args::require_str(args, "path").expect("validated");
    let function_name = args::require_str(args, "name").expect("validated");
    let new_param = args::require_str(args, "new_param").expect("validated");
    let position_raw = args::require_str(args, "position").expect("validated");
    let default_value = args::require_str(args, "callsite_fill_in").expect("validated");
    let line_hint = args::opt_u64(args, "line").expect("validated");
    let position = Position::parse(position_raw).expect("validated");

    let Some(lsp) = lsp else {
        return Ok(ToolResult::err(
            "add_param requires LSP support (no LSP client available for this project)".into(),
        ));
    };
    if let Err(e) = ensure_ready(lsp, Duration::from_secs(60)).await {
        return Ok(ToolResult::err(format!(
            "LSP not ready in time: {e}. Try again in a moment, or call code(diagnostics) first to warm it up."
        )));
    }

    let abs_path = config.project_root.join(path_str);
    let line_hint_0 = line_hint.map(|n| (n.saturating_sub(1)) as u32);

    // Resolve the agent's `name` (+ optional `line` hint) to a canonical
    // definition position via LSP `textDocument/documentSymbol`. This is
    // the gate that catches "you pointed me at a call site, not a
    // signature" *before* we waste any model calls.
    let resolved = match resolve_function_location(lsp, &abs_path, function_name, line_hint_0).await
    {
        Ok(r) => r,
        Err(e) => {
            return Ok(ToolResult::err(format!("✗ add_param: {e}")));
        }
    };
    let line_0 = resolved.line_0;
    let column_0 = resolved.column_0;
    let resolved_line_1 = line_0 + 1;

    let original_signature_source = std::fs::read_to_string(&abs_path)
        .with_context(|| format!("read function file {path_str}"))?;

    // Idempotency guard: re-adding a parameter that already exists stacks a
    // duplicate argument at EVERY callsite. Observed churn (seeded bench): a
    // small model calls add_param on the same function repeatedly and the call
    // sites balloon to 8–12 args until the file won't compile. If the param is
    // already present, refuse and point at the actual fix — editing the one
    // callsite that should carry the real value, not re-adding the parameter.
    let new_param_name = new_param
        .split(':')
        .next()
        .unwrap_or(new_param)
        .trim()
        .trim_start_matches("mut ")
        .trim();
    if !new_param_name.is_empty()
        && signature_has_param(&original_signature_source, line_0 as usize, new_param_name)
    {
        return Ok(ToolResult::err(format!(
            "✗ add_param: `{function_name}` already has a parameter named `{new_param_name}` — \
             not adding a duplicate (that would stack another `{default_value}` argument at every \
             callsite and break the build). If a value is not being threaded through, the fix is \
             NOT to add the parameter again: EDIT the specific callsite that should pass the real \
             value (replace its `{default_value}` placeholder with the actual expression), then \
             re-run your check."
        )));
    }

    // 1. Update the signature itself. The snippet starts at the function's
    // own definition line so the model has zero ambiguity about which
    // construct to edit.
    let sig_window = extract_window(&original_signature_source, line_0, 12);
    let sig_instruction = format!(
        "Add a new parameter to the function whose signature starts at the FIRST line of the snippet below. \
         The parameter to add: `{new_param}`. \
         Insert it {pos}. \
         Change ONLY the function's parameter list — do not touch the body, return type, generics, where-clause, or any other code in the snippet.",
        pos = position.human(),
    );
    if let Some(log) = log {
        log.tool_debug(
            "change_signature",
            &format!(
                "add_param entry path={path_str} name={function_name} line_hint={line_hint:?} \
                 resolved=line_0={line_0} column_0={column_0} \
                 new_param={new_param:?} position={position:?} default={default_value:?}"
            ),
        );
    }
    // Validator-aware retries: if the model's OLD/NEW can't be applied at
    // the LSP-resolved anchor (e.g. dropped indentation off the prefill —
    // a real failure mode we've seen on Devstral), retry with a fresh
    // inference pass. Mirrors what we do for callsites below.
    let sig_rewrite = match ask_rewrite_validated(
        router,
        log,
        &format!("signature:{path_str}:{resolved_line_1}"),
        &sig_instruction,
        &sig_window.text,
        cancelled,
        |r| apply_rewrite(&original_signature_source, r, line_0).map(|_| ()),
    )
    .await
    {
        Ok(Some(r)) => r,
        Ok(None) => unreachable!("ask_rewrite_validated returns Some on success or Err otherwise"),
        Err(e) => {
            return Ok(ToolResult::err(format!(
                "signature rewrite failed: {e}. The model could not produce an OLD/NEW \
                 block matching the source. Re-run, or edit the signature manually."
            )));
        }
    };
    // Validator already verified apply succeeds; re-run to produce the
    // final updated source.
    let updated_signature_source = apply_rewrite(&original_signature_source, &sig_rewrite, line_0)
        .expect("validator verified apply succeeds");

    // Stage the signature edit in memory; we'll commit all files at the end.
    let mut staged: BTreeMap<PathBuf, StagedEdit> = BTreeMap::new();
    staged.insert(
        abs_path.clone(),
        StagedEdit {
            original: original_signature_source.clone(),
            updated: updated_signature_source,
        },
    );

    // 2. Find callsites and rewrite each one.
    let callsites = find_callsites(lsp, config, &abs_path, line_0, column_0)
        .await
        .context("find_callsites failed")?;

    if callsites.is_empty() {
        commit_staged(&staged, config, revisions, "change_signature.add_param")?;
        return Ok(ToolResult::ok(format!(
            "✓ add_param: signature updated. No callsites found via LSP — \
             nothing else to change. (If you expected callers, ensure the LSP \
             has indexed the project; calling code(diagnostics) first warms it up.)\n\
             - Edited: {} (signature)",
            path_str
        )));
    }

    let mut report = Vec::new();
    let mut callsite_failures = Vec::new();
    let pos_human = position.human();
    // Bookend the per-callsite prompt with the actual OLD/NEW signatures.
    // The model can compare them positionally to figure out where the new
    // argument belongs without us having to encode that as a numeric index.
    let old_signature = sig_rewrite.old.clone();
    let new_signature = sig_rewrite.new.clone();

    for site in &callsites {
        let rel = display_path(&site.path, config);
        let instruction = format!(
            "A function's signature was modified: a new parameter was added {pos_human}. \
             Update the call expression at the FIRST line of the snippet below to match the new \
             signature: insert the literal expression `{fill}` at the matching argument position. \
             Change ONLY that one call.\n\
             \n\
             Old signature:\n{old_signature}\n\
             \n\
             New signature:\n{new_signature}",
            pos_human = pos_human,
            fill = default_value,
        );

        // Resolve which source content we'll edit (staged copy if this
        // file has already been touched in this refactor, fresh read
        // otherwise). We need it both for validation during retries and
        // for the final apply.
        let (original, src) = match staged.get(&site.path) {
            Some(edit) => (edit.original.clone(), edit.updated.clone()),
            None => match std::fs::read_to_string(&site.path) {
                Ok(s) => (s.clone(), s),
                Err(e) => {
                    callsite_failures.push(format!(
                        "{}:{}: read failed: {}",
                        rel,
                        site.line + 1,
                        e
                    ));
                    continue;
                }
            },
        };
        // ask_rewrite_validated retries when the model produces a
        // syntactically-valid OLD/NEW that nonetheless can't be applied —
        // e.g. a paraphrased / lazy OLD that doesn't match at the LSP-
        // resolved anchor line. Without this the model gets exactly one
        // shot per callsite even on cases where retry would succeed.
        let rewrite = match ask_rewrite_validated(
            router,
            log,
            &format!("callsite:{rel}:{}", site.line + 1),
            &instruction,
            &site.window,
            cancelled,
            |r| apply_rewrite(&src, r, site.line).map(|_| ()),
        )
        .await
        {
            Ok(Some(r)) => r,
            Ok(None) => unreachable!(),
            Err(e) => {
                callsite_failures.push(format!("{}:{}: {}", rel, site.line + 1, e));
                continue;
            }
        };
        // Validator already verified apply succeeds; this re-runs to
        // produce the final updated source. (Cheap pure function.)
        match apply_rewrite(&src, &rewrite, site.line) {
            Ok(updated) => {
                staged.insert(site.path.clone(), StagedEdit { original, updated });
                report.push(format!(
                    "  • {}:{} now passes `{}`",
                    rel,
                    site.line + 1,
                    default_value
                ));
            }
            Err(e) => {
                callsite_failures.push(format!("{}:{}: {}", rel, site.line + 1, e));
            }
        }
    }

    // 3. Commit edits.
    commit_staged(&staged, config, revisions, "change_signature.add_param")?;

    // 4. Format the report.
    let total = callsites.len();
    let succeeded = report.len();
    let mut out = String::new();
    if callsite_failures.is_empty() {
        // A/B knob: MINISWE_ADDPARAM_LEGACY_MSG=1 restores the old "✓ COMPLETE"
        // wording so the honest-stub message can be benchmarked against it.
        let legacy_msg = matches!(
            std::env::var("MINISWE_ADDPARAM_LEGACY_MSG").as_deref(),
            Ok("1")
        );
        if legacy_msg {
            out.push_str(&format!(
                "✓ COMPLETE — definition and all {total} callsites are now consistent.\n",
            ));
        } else {
            out.push_str(&format!(
                "✓ Signature updated — definition and all {total} callsite(s) now compile. \
                 Each callsite was filled with the placeholder `{default_value}` you specified. \
                 That is a compile-correct STUB, not necessarily finished wiring: any callsite that \
                 should receive a real value (rather than `{default_value}`) still needs to be edited \
                 to pass it.\n",
            ));
        }
    } else if succeeded == 0 {
        out.push_str(&format!(
            "✗ add_param FAILED: signature was rewritten on disk, but 0 of {total} callsite(s) \
             could be updated. The project will NOT compile until callsites are fixed. \
             Either retry with corrected callsites OR roll back via \
             file(action=\"revert\", to_round=<this round>) to discard the signature change.\n\
             Reasons {failed} site(s) failed:\n",
            failed = callsite_failures.len(),
        ));
        for f in &callsite_failures {
            out.push_str(&format!("  • {f}\n"));
        }
        out.push('\n');
    } else {
        out.push_str(&format!(
            "✗ add_param PARTIAL: signature rewritten, only {succeeded}/{total} callsite(s) updated. \
             The project will NOT compile until the remaining {failed} callsite(s) are fixed manually \
             OR you roll back via file(action=\"revert\", to_round=<this round>).\n\
             Failures:\n",
            failed = callsite_failures.len(),
        ));
        for f in &callsite_failures {
            out.push_str(&format!("  • {f}\n"));
        }
        out.push('\n');
    }
    if !report.is_empty() {
        out.push_str("Callsites updated (each now passes the placeholder — edit any that should carry the real value):\n");
        for line in &report {
            out.push_str(line);
            out.push('\n');
        }
    }
    out.push_str(
        "\nNext: run code(diagnostics) or your build to confirm the project still compiles.",
    );

    // Note: we do NOT auto-revert on partial failure. The user (agent) gets
    // the per-callsite report and can decide whether to keep, fix, or
    // git-revert the changes — same trade-off discussed in the design.

    Ok(if callsite_failures.is_empty() {
        ToolResult::ok(out)
    } else {
        ToolResult::err(out)
    })
}

fn display_path(p: &std::path::Path, config: &Config) -> String {
    p.strip_prefix(&config.project_root)
        .unwrap_or(p)
        .display()
        .to_string()
}

/// Best-effort check: does the function whose signature starts at `from_line`
/// (0-based) already have a parameter named `param_name`? Used to make
/// add_param idempotent. Scans the first balanced `(...)` after the definition
/// line and compares the binding name (the text before `:`) of each parameter.
///
/// Conservative by design: splitting the parameter list on `,` can mis-split
/// generic types like `HashMap<K, V>`, but since we only match a fragment whose
/// pre-`:` text *equals* `param_name`, that can only cause a missed detection
/// (no-op, same as today) — never a false positive that wrongly blocks a add.
fn signature_has_param(source: &str, from_line: usize, param_name: &str) -> bool {
    let tail: String = source
        .lines()
        .skip(from_line)
        .collect::<Vec<_>>()
        .join("\n");
    let Some(open) = tail.find('(') else {
        return false;
    };
    let mut depth = 0i32;
    let mut close = None;
    for (i, c) in tail[open..].char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(open + i);
                    break;
                }
            }
            _ => {}
        }
    }
    let Some(close) = close else {
        return false;
    };
    let params = &tail[open + 1..close];
    params.split(',').any(|p| {
        let p = p.trim().trim_start_matches("mut ").trim();
        p.split(':').next().map(str::trim) == Some(param_name)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signature_has_param_detects_existing_and_ignores_absent() {
        // multi-line signature like assemble()/run() in the bench
        let src = "fn assemble(\n    config: &Config,\n    user_message: &str,\n    system_prompt_override: Option<String>,\n) -> X {\n    body\n}";
        assert!(signature_has_param(src, 0, "system_prompt_override"));
        assert!(signature_has_param(src, 0, "config"));
        assert!(!signature_has_param(src, 0, "headless")); // not present
    }

    #[test]
    fn signature_has_param_not_fooled_by_generic_commas() {
        // a generic type with an internal comma must not produce a false positive
        let src = "fn f(map: HashMap<K, V>, name: String) {}";
        assert!(signature_has_param(src, 0, "map"));
        assert!(signature_has_param(src, 0, "name"));
        assert!(!signature_has_param(src, 0, "V")); // generic param, not a binding
        assert!(!signature_has_param(src, 0, "K"));
    }

    #[test]
    fn position_parses_end_append() {
        assert!(matches!(Position::parse("end").unwrap(), Position::End));
        assert!(matches!(Position::parse("append").unwrap(), Position::End));
        assert!(matches!(Position::parse(" end ").unwrap(), Position::End));
        assert!(Position::End.human().contains("end of the parameter list"));
    }

    #[test]
    fn position_parses_valid_anchors() {
        assert!(matches!(Position::parse("start").unwrap(), Position::Start));
        assert!(matches!(
            Position::parse("after:b").unwrap(),
            Position::After(n) if n == "b"
        ));
        assert!(matches!(
            Position::parse("after:_internal_flag").unwrap(),
            Position::After(n) if n == "_internal_flag"
        ));
        assert!(matches!(
            Position::parse("after:mcp_summary").unwrap(),
            Position::After(n) if n == "mcp_summary"
        ));
        // Surrounding whitespace tolerated.
        assert!(matches!(
            Position::parse("  after:b  ").unwrap(),
            Position::After(n) if n == "b"
        ));
    }

    #[test]
    fn position_rejects_param_list_with_targeted_error() {
        let err = Position::parse("after:plan_only: bool, mcp_summary: Option<&str>")
            .unwrap_err()
            .to_string();
        // The targeted phrasing is what fixes Devstral on retry — keep it.
        assert!(err.contains("malformed"), "got: {err}");
        assert!(err.contains("'start'"), "got: {err}");
        assert!(err.contains("after:<single_param_name>"), "got: {err}");
        assert!(err.contains("Do NOT"), "got: {err}");
    }

    #[test]
    fn position_rejects_other_malformations() {
        for bad in [
            "after:",         // empty name
            "after:b: u32",   // type annotation
            "after:b, c",     // list
            "after:my-param", // hyphen
            "after:1param",   // starts with digit
            "before:b",       // wrong prefix
            "first",          // not 'start'
            "",
        ] {
            let res = Position::parse(bad);
            assert!(res.is_err(), "expected {bad:?} to be rejected");
        }
    }
}
