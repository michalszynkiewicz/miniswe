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
use crate::tools::args;
use crate::tools::fast::RevisionStore;
use crate::tools::{ToolDetail, ToolResult};

use super::ast_span;
use super::model_edit::{apply_rewrite, ask_rewrite_validated};
use super::sites::{
    CallSite, StagedEdit, callsite_window, commit_staged, ensure_ready, extract_window,
    find_callsites, reanchor_callsite, resolve_function_location,
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
        (Err(verr), None) => {
            return Ok(ToolResult::err_with_detail(
                verr.message,
                ToolDetail::InvalidArgs {
                    action: "add_param",
                    missing: verr.missing,
                    bad_type: verr.bad_type,
                    unknown: verr.unknown,
                },
            ));
        }
        (Ok(()), Some(p)) => {
            return Ok(ToolResult::err(format!(
                "✗ change_signature(add_param): {p}"
            )));
        }
        (Err(verr), Some(p)) => {
            let message = format!("{}\n\nAlso: {p}", verr.message);
            return Ok(ToolResult::err_with_detail(
                message,
                ToolDetail::InvalidArgs {
                    action: "add_param",
                    missing: verr.missing,
                    bad_type: verr.bad_type,
                    unknown: verr.unknown,
                },
            ));
        }
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
    // sites balloon to 8–12 args until the file won't compile.
    //
    // But "the signature already has it" is TWO states, not one, and only the
    // first deserves a refusal:
    //   a) done — every callsite already passes the argument. Re-adding stacks
    //      duplicates. Refuse.
    //   b) HALF-APPLIED — a previous add_param rewrote the signature and then
    //      failed on some callsites (PARTIAL). The tree does not compile, and
    //      the one tool that can repair it in bulk locks itself out on exactly
    //      the state it exists to fix. Observed live: that lockout is where the
    //      model gives up on the refactor and reaches for `sed -i` across the
    //      callsites, which is how a 6/6 run turns into a corrupted test file.
    // Distinguish them mechanically (arg count vs param count) and re-sync (b).
    let new_param_name = param_ident(new_param);
    if !new_param_name.is_empty()
        && ast_span::has_param(
            &original_signature_source,
            path_str,
            line_0 as usize,
            new_param_name,
        )
    {
        return resync_or_refuse(
            config,
            router,
            lsp,
            log,
            revisions,
            cancelled,
            &abs_path,
            path_str,
            &original_signature_source,
            line_0,
            column_0,
            function_name,
            new_param_name,
            default_value,
        )
        .await;
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
    // Deterministic OLD: the model only needs to write NEW (see
    // `ask_rewrite_validated`'s `known_old` doc — this is what closes the
    // "OLD line N doesn't match source" failure mode on multi-line
    // signatures, which previously forced callers to abandon the atomic
    // refactor and fall back to a manual edit that skips every callsite).
    let known_old = ast_span::signature_span(&original_signature_source, path_str, line_0 as usize);
    let sig_rewrite = match ask_rewrite_validated(
        router,
        log,
        &format!("signature:{path_str}:{resolved_line_1}"),
        &sig_instruction,
        &sig_window.text,
        known_old.as_deref(),
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

    let pos_human = position.human();
    // Bookend the per-callsite prompt with the actual OLD/NEW signatures.
    // The model can compare them positionally to figure out where the new
    // argument belongs without us having to encode that as a numeric index.
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
        old_signature = sig_rewrite.old,
        new_signature = sig_rewrite.new,
    );

    let (report, callsite_failures) = rewrite_callsites(
        router,
        log,
        config,
        cancelled,
        &callsites,
        function_name,
        &instruction,
        default_value,
        &mut staged,
    )
    .await;

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
    // Probe-validated wording (moment-replay tier-1, 2026-07-02): the only
    // variant that produced follow-up callsite edits (0%→14%) and cut the
    // re-call-add_param loop (17%→11%). Every shorter rewording tested worse —
    // the NOT-finished framing, explicit tool names, and the re-refactor
    // prohibition are all load-bearing. Do not compress without re-probing.
    out.push_str(
        "\nNext: the placeholder callsites above are NOT finished. For each callsite that \
         must pass a real value: read it, then edit it DIRECTLY with replace_range or \
         insert_at, replacing the placeholder argument with the real value. Do NOT call \
         refactor again for this — the parameter already exists and add_param will be \
         rejected; direct edits are the correct tool for this step.",
    );
    // New addition (not part of the probe-validated paragraph above — kept
    // as a separate sentence so it doesn't disturb that wording, and not
    // itself re-probed): the failure this targets is different from the
    // "forgot to wire the value" case the paragraph above already covers —
    // it's the model rewriting unrelated code nearby (e.g. the function
    // body while wiring the new param in) and silently dropping something
    // that was already there. A signature update alone can't cause that;
    // only a follow-up hand-edit can, which is exactly the step this
    // message sends the model to next.
    out.push_str(
        " Once every callsite has a real value, run the FULL test suite — not just a compile \
         check — to confirm nothing else nearby broke while you were wiring this in.",
    );

    // Note: we do NOT auto-revert on partial failure. The user (agent) gets
    // the per-callsite report and can decide whether to keep, fix, or
    // git-revert the changes — same trade-off discussed in the design.

    Ok(if callsite_failures.is_empty() {
        ToolResult::ok(out)
    } else {
        ToolResult::err_with_detail(
            out,
            ToolDetail::PartialSignatureChange {
                action: "add_param",
                total,
                succeeded,
                callsite_failures,
                callsite_report: report,
            },
        )
    })
}

/// Rewrite each callsite to match the new signature, staging the edits.
///
/// Returns `(report, failures)`, one line per callsite. Split out of
/// `execute` because it is driven from two places: the ordinary path, right
/// after the signature is rewritten, and the re-sync path, which repairs a
/// half-applied refactor where the signature already carries the parameter
/// but some callsites were never updated.
#[allow(clippy::too_many_arguments)]
async fn rewrite_callsites(
    router: &ModelRouter,
    log: Option<&SessionLog>,
    config: &Config,
    cancelled: Option<&AtomicBool>,
    callsites: &[CallSite],
    function_name: &str,
    instruction: &str,
    default_value: &str,
    staged: &mut BTreeMap<PathBuf, StagedEdit>,
) -> (Vec<String>, Vec<String>) {
    let mut report = Vec::new();
    let mut callsite_failures = Vec::new();

    for site in callsites {
        let rel = display_path(&site.path, config);
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
        // The LSP resolved this position against the file as its index saw
        // it, which is not necessarily the content we are about to edit.
        // Re-anchor before building the window or the OLD block.
        let Some((line, column)) = reanchor_callsite(&src, site.line, function_name) else {
            callsite_failures.push(format!(
                "{}:{}: no call to `{function_name}` found near this line in the current file \
                 — it was probably moved or removed since the LSP indexed it. Edit this \
                 callsite directly.",
                rel,
                site.line + 1
            ));
            continue;
        };
        // Re-cut the window whenever the anchor moved: the stored one came
        // from the indexed content at the stale line and can start partway
        // into the argument list, which leaves the model guessing at OLD.
        let window = if line == site.line {
            site.window.clone()
        } else {
            callsite_window(&src, line)
        };
        // Deterministic OLD (same rationale as the signature rewrite above):
        // a live replay of a historical bench failure confirmed multi-line
        // callsites — one argument per line, the common rustfmt style —
        // hit the identical "OLD line N doesn't match source" failure.
        let known_old = ast_span::callsite_span(&src, &rel, line, column);
        // ask_rewrite_validated retries when the model produces a
        // syntactically-valid OLD/NEW that nonetheless can't be applied —
        // e.g. a paraphrased / lazy OLD that doesn't match at the LSP-
        // resolved anchor line. Without this the model gets exactly one
        // shot per callsite even on cases where retry would succeed.
        let rewrite = match ask_rewrite_validated(
            router,
            log,
            &format!("callsite:{rel}:{}", line + 1),
            instruction,
            &window,
            known_old.as_deref(),
            cancelled,
            |r| apply_rewrite(&src, r, line).map(|_| ()),
        )
        .await
        {
            Ok(Some(r)) => r,
            Ok(None) => unreachable!(),
            Err(e) => {
                callsite_failures.push(format!("{}:{}: {}", rel, line + 1, e));
                continue;
            }
        };
        // Validator already verified apply succeeds; this re-runs to
        // produce the final updated source. (Cheap pure function.)
        match apply_rewrite(&src, &rewrite, line) {
            Ok(updated) => {
                staged.insert(site.path.clone(), StagedEdit { original, updated });
                report.push(format!(
                    "  • {}:{} now passes `{}`",
                    rel,
                    line + 1,
                    default_value
                ));
            }
            Err(e) => {
                callsite_failures.push(format!("{}:{}: {}", rel, line + 1, e));
            }
        }
    }

    (report, callsite_failures)
}

/// The bare identifier of a parameter declaration: `mut cfg: &Config` -> `cfg`.
fn param_ident(decl: &str) -> &str {
    decl.split(':')
        .next()
        .unwrap_or(decl)
        .trim()
        .trim_start_matches("mut ")
        .trim()
}

/// `add_param` on a function whose signature ALREADY declares the parameter.
///
/// Refuses when the refactor is genuinely done (every callsite passes the
/// argument), re-syncs when it is half-applied (signature updated, some
/// callsites never were). "Cannot tell" always falls to refusal: the guard
/// exists to stop argument stacking, and a wrong re-sync stacks arguments at
/// every site it touches, which is the exact damage it is meant to prevent.
/// Concretely, refusal is the answer when the parameter list cannot be parsed,
/// when callsites cannot be resolved, and — per callsite — whenever
/// `arg_count` returns `None` or a count that is not short.
#[allow(clippy::too_many_arguments)]
async fn resync_or_refuse(
    config: &Config,
    router: &ModelRouter,
    lsp: &LspClient,
    log: Option<&SessionLog>,
    revisions: Option<&RevisionStore>,
    cancelled: Option<&AtomicBool>,
    abs_path: &std::path::Path,
    path_str: &str,
    source: &str,
    line_0: u32,
    column_0: u32,
    function_name: &str,
    new_param_name: &str,
    default_value: &str,
) -> Result<ToolResult> {
    // The original refusal. Kept verbatim: it is the message the model sees on
    // every legitimately-redundant call, and it already points at the real fix
    // (edit the one callsite that should carry the value).
    let refuse = |checked: Option<usize>| {
        let checked_note = match checked {
            Some(n) => format!(" All {n} callsite(s) already pass it."),
            None => String::new(),
        };
        Ok(ToolResult::err(format!(
            "✗ add_param: `{function_name}` already has a parameter named `{new_param_name}` — \
             not adding a duplicate (that would stack another `{default_value}` argument at every \
             callsite and break the build).{checked_note} If a value is not being threaded through, \
             the fix is NOT to add the parameter again: EDIT the specific callsite that should pass \
             the real value (replace its `{default_value}` placeholder with the actual expression), \
             then re-run your check."
        )))
    };

    let Some(params) = ast_span::param_names(source, path_str, line_0 as usize) else {
        return refuse(None);
    };
    let expected = params.len();
    let ordinal = params.iter().position(|p| param_ident(p) == new_param_name);

    let Ok(callsites) = find_callsites(lsp, config, abs_path, line_0, column_0).await else {
        return refuse(None);
    };
    if callsites.is_empty() {
        return refuse(None);
    }

    // A callsite is out of date only if we can COUNT its arguments and the
    // count is short. Anything else — unparseable, a non-call reference, an
    // argument count at or above the parameter count — is left alone.
    let mut stale = Vec::new();
    let mut in_sync = 0usize;
    for site in callsites {
        let rel = display_path(&site.path, config);
        let Ok(src) = std::fs::read_to_string(&site.path) else {
            continue;
        };
        let Some((line, column)) = reanchor_callsite(&src, site.line, function_name) else {
            continue;
        };
        match ast_span::arg_count(&src, &rel, line, column) {
            Some(n) if n < expected => stale.push(site),
            Some(_) => in_sync += 1,
            None => {}
        }
    }

    if stale.is_empty() {
        return refuse(Some(in_sync));
    }

    if let Some(log) = log {
        log.tool_debug(
            "change_signature",
            &format!(
                "add_param re-sync path={path_str} name={function_name} \
                 param={new_param_name} expected_args={expected} \
                 stale={} in_sync={in_sync}",
                stale.len()
            ),
        );
    }

    let signature = ast_span::signature_span(source, path_str, line_0 as usize)
        .unwrap_or_else(|| function_name.to_string());
    // The ordinal is what a bare "add the argument" prompt cannot supply: with
    // a placeholder like `None` the model has no type information to infer the
    // position from, and appending it at the end silently reorders arguments.
    let ordinal_hint = match ordinal {
        Some(i) => format!(" It is argument {} of {expected}.", i + 1),
        None => String::new(),
    };
    let instruction = format!(
        "This call is out of date: the function's signature declares a parameter \
         `{new_param_name}` that this call does not pass.{ordinal_hint} Update the call \
         expression at the FIRST line of the snippet below to insert the literal expression \
         `{default_value}` at that argument position. Change ONLY that one call.\n\
         \n\
         Signature:\n{signature}"
    );

    let mut staged: BTreeMap<PathBuf, StagedEdit> = BTreeMap::new();
    let total = stale.len();
    let (report, callsite_failures) = rewrite_callsites(
        router,
        log,
        config,
        cancelled,
        &stale,
        function_name,
        &instruction,
        default_value,
        &mut staged,
    )
    .await;
    commit_staged(&staged, config, revisions, "change_signature.add_param")?;

    let succeeded = report.len();
    let mut out = String::new();
    if callsite_failures.is_empty() {
        out.push_str(&format!(
            "✓ add_param: `{function_name}` already declared `{new_param_name}`, so the signature \
             was left alone — but {total} callsite(s) had not been updated to pass it. Those are \
             now filled with the placeholder `{default_value}`.\n"
        ));
    } else {
        out.push_str(&format!(
            "✗ add_param PARTIAL re-sync: `{function_name}` already declared `{new_param_name}`; \
             of the {total} callsite(s) that were not passing it, {succeeded} were repaired and \
             {failed} were not. The project will NOT compile until the rest are fixed — edit them \
             directly.\nFailures:\n",
            failed = callsite_failures.len(),
        ));
        for f in &callsite_failures {
            out.push_str(&format!("  • {f}\n"));
        }
    }
    if !report.is_empty() {
        out.push_str(
            "Callsites updated (each now passes the placeholder — edit any that should carry \
             the real value):\n",
        );
        for line in &report {
            out.push_str(line);
            out.push('\n');
        }
    }
    if in_sync > 0 {
        out.push_str(&format!(
            "({in_sync} other callsite(s) were already passing it and were not touched.)\n"
        ));
    }
    out.push_str(
        "\nNext: the placeholder callsites above are NOT finished. For each callsite that \
         must pass a real value: read it, then edit it DIRECTLY with replace_range or \
         insert_at, replacing the placeholder argument with the real value. Do NOT call \
         refactor again for this — the parameter already exists and add_param will be \
         rejected; direct edits are the correct tool for this step.",
    );

    Ok(if callsite_failures.is_empty() {
        ToolResult::ok(out)
    } else {
        ToolResult::err_with_detail(
            out,
            ToolDetail::PartialSignatureChange {
                action: "add_param",
                total,
                succeeded,
                callsite_failures,
                callsite_report: report,
            },
        )
    })
}

fn display_path(p: &std::path::Path, config: &Config) -> String {
    p.strip_prefix(&config.project_root)
        .unwrap_or(p)
        .display()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

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
