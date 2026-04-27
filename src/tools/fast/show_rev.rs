//! `show_rev <path> <rev>` — inspect the full stored metadata + payload
//! for one revision of a file.
//!
//! Works for both live revs and tombstones. Payload is capped at
//! [`SHOW_REV_PAYLOAD_CAP`] bytes (2 KB) in the rendered output — the
//! tombstone table already holds the first few lines; this tool is the
//! model's escape hatch when that preview isn't enough to decide whether
//! a proposed edit is a byte-identical replay of a reverted one.
//!
//! Read-only. Does NOT create a revision, does not touch disk.

use anyhow::Result;
use serde_json::Value;

use super::super::ToolResult;
use super::super::permissions::PermissionManager;
use super::revisions::{Revision, RevisionStore};

/// Cap on payload bytes rendered by `show_rev`. Stored payloads may be
/// larger up to [`super::revisions::MAX_STORED_PAYLOAD_BYTES`]; this cap
/// keeps the single tool response bounded no matter what's in the store.
pub const SHOW_REV_PAYLOAD_CAP: usize = 2 * 1024;

pub async fn execute(
    args: &Value,
    perms: &PermissionManager,
    revisions: &RevisionStore,
) -> Result<ToolResult> {
    let path = match super::super::args::require_str(args, "path") {
        Ok(p) => p,
        Err(e) => return Ok(ToolResult::err(e)),
    };
    let rev_num = match super::super::args::require_u64(args, "rev") {
        Ok(n) => n as usize,
        Err(e) => return Ok(ToolResult::err(e)),
    };

    if let Err(e) = perms.resolve_and_check_path(path) {
        return Ok(ToolResult::err(e));
    }

    let Some(rev) = revisions.get(path, rev_num) else {
        // Provide a helpful listing of what IS available for this file.
        let list = revisions.list(path);
        if list.is_empty() {
            return Ok(ToolResult::err(format!(
                "show_rev: no revisions recorded for {path}"
            )));
        }
        let available: Vec<String> = list
            .iter()
            .map(|r| {
                let tag = if r.reverted { " (reverted)" } else { "" };
                format!("rev_{}{}", r.number, tag)
            })
            .collect();
        return Ok(ToolResult::err(format!(
            "show_rev: rev_{rev_num} not found for {path} (available: {})",
            available.join(", ")
        )));
    };

    Ok(ToolResult::ok(render(path, &rev)))
}

fn render(path: &str, r: &Revision) -> String {
    let status_tag = if r.reverted {
        " (reverted — tombstone)"
    } else {
        ""
    };
    let mut out = format!("rev_{} {} {}{}\n", r.number, r.operation, path, status_tag);

    // Operation-specific location line.
    match (r.operation.as_str(), r.range) {
        ("replace_range", Some((start, end))) => {
            out.push_str(&format!("  range: L{start}-{end}\n"));
        }
        ("insert_at", _) => {
            // For insert_at, the label already encodes "after Lxx" —
            // surface that here instead of an empty range line.
            out.push_str(&format!("  label: {}\n", r.label));
        }
        ("initial", _) => {
            out.push_str("  (pristine baseline — no tool call produced this)\n");
        }
        _ => {
            out.push_str(&format!("  label: {}\n", r.label));
        }
    }

    // Deltas / outcome.
    if r.added != 0 || r.removed != 0 {
        out.push_str(&format!("  delta: +{} -{}\n", r.added, r.removed));
    }
    if r.ast_ok {
        out.push_str("  ast: ok\n");
    } else if let Some(err) = &r.ast_error {
        out.push_str(&format!("  ast: broken at {err}\n"));
    } else {
        out.push_str("  ast: broken\n");
    }
    out.push_str(&format!(
        "  file_errors: {}   project_errors: {}\n",
        r.file_errors, r.project_errors
    ));

    // Payload (verbatim, capped).
    match (r.operation.as_str(), &r.payload) {
        ("write_file", _) => {
            let bytes = r.content.len();
            let lines = r.content.lines().count().max(1);
            out.push_str(&format!(
                "  (full rewrite, {lines} line(s), {bytes} bytes — use file(action='read') to see current content)\n"
            ));
        }
        (_, Some(payload)) => {
            let key = if r.operation == "insert_at" {
                "text"
            } else {
                "new_text"
            };
            let (shown, truncated_bytes) = cap_for_display(payload, SHOW_REV_PAYLOAD_CAP);
            out.push_str(&format!("  {key}: |\n"));
            for line in shown.lines() {
                out.push_str("    ");
                out.push_str(line);
                out.push('\n');
            }
            if truncated_bytes > 0 {
                out.push_str(&format!(
                    "  … (truncated — {truncated_bytes} more byte(s) not shown)\n"
                ));
            }
        }
        (_, None) => {
            // Pristine / unknown — nothing to show.
        }
    }

    // Drop trailing newline for consistent formatting.
    if out.ends_with('\n') {
        out.pop();
    }
    out
}

/// Cut `s` to at most `cap` bytes on a char boundary. Returns
/// `(shown, extra_bytes)` where `extra_bytes > 0` means truncation
/// happened.
fn cap_for_display(s: &str, cap: usize) -> (String, usize) {
    if s.len() <= cap {
        return (s.to_string(), 0);
    }
    let mut cut = cap;
    while cut > 0 && !s.is_char_boundary(cut) {
        cut -= 1;
    }
    (s[..cut].to_string(), s.len() - cut)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::tools::fast::revisions::RecordArgs;
    use crate::tools::permissions::PermissionManager;

    fn scratch_perms() -> PermissionManager {
        let mut cfg = Config::default();
        cfg.project_root = std::env::temp_dir();
        PermissionManager::new(&cfg)
    }

    fn args(op: &str, label: &str, payload: Option<&str>) -> RecordArgs<'static> {
        // Build with 'static strs so we don't fight lifetimes in tests.
        let op: &'static str = Box::leak(op.to_string().into_boxed_str());
        let label: &'static str = Box::leak(label.to_string().into_boxed_str());
        RecordArgs {
            operation: op,
            label,
            range: None,
            payload: payload.map(String::from),
            added: 1,
            removed: 1,
            ast_ok: true,
            ast_error: None,
            file_errors: 0,
            project_errors: 0,
        }
    }

    #[tokio::test]
    async fn shows_live_replace_range_with_payload() {
        let store = RevisionStore::with_cap(20);
        store.ensure_pristine("f.rs", "x").unwrap();
        let mut a = args(
            "replace_range",
            "replace_range L1-1",
            Some("println!(\"hi\");"),
        );
        a.range = Some((1, 1));
        store.record("f.rs", "y", a).unwrap();

        let perms = scratch_perms();
        let r = execute(
            &serde_json::json!({ "path": "f.rs", "rev": 1 }),
            &perms,
            &store,
        )
        .await
        .unwrap();
        assert!(r.success, "{}", r.content);
        assert!(r.content.contains("rev_1"));
        assert!(r.content.contains("replace_range"));
        assert!(r.content.contains("range: L1-1"));
        assert!(r.content.contains("ast: ok"));
        assert!(r.content.contains("new_text: |"));
        assert!(r.content.contains("println!(\"hi\");"));
        assert!(!r.content.contains("(reverted"));
    }

    #[tokio::test]
    async fn shows_tombstone_with_reverted_marker() {
        let store = RevisionStore::with_cap(20);
        store.ensure_pristine("f.rs", "x").unwrap();
        store
            .record(
                "f.rs",
                "y",
                args("replace_range", "replace_range L1-1", Some("BROKEN")),
            )
            .unwrap();
        store.mark_reverted_to("f.rs", 0).unwrap(); // rev_1 becomes tombstone

        let perms = scratch_perms();
        let r = execute(
            &serde_json::json!({ "path": "f.rs", "rev": 1 }),
            &perms,
            &store,
        )
        .await
        .unwrap();
        assert!(r.success, "{}", r.content);
        assert!(
            r.content.contains("(reverted — tombstone)"),
            "expected tombstone marker: {}",
            r.content
        );
        assert!(r.content.contains("BROKEN"));
    }

    #[tokio::test]
    async fn large_payload_is_capped_with_marker() {
        let store = RevisionStore::with_cap(20);
        store.ensure_pristine("f.rs", "x").unwrap();
        let big = "a".repeat(SHOW_REV_PAYLOAD_CAP + 500);
        store
            .record(
                "f.rs",
                "y",
                args("replace_range", "replace_range L1-1", Some(&big)),
            )
            .unwrap();

        let perms = scratch_perms();
        let r = execute(
            &serde_json::json!({ "path": "f.rs", "rev": 1 }),
            &perms,
            &store,
        )
        .await
        .unwrap();
        assert!(r.success);
        assert!(
            r.content.contains("truncated"),
            "should mention truncation: {}",
            r.content
        );
        // Should contain at least SHOW_REV_PAYLOAD_CAP bytes of the payload.
        assert!(r.content.len() >= SHOW_REV_PAYLOAD_CAP);
    }

    #[tokio::test]
    async fn unknown_rev_reports_available_list() {
        let store = RevisionStore::with_cap(20);
        store.ensure_pristine("f.rs", "x").unwrap();
        store
            .record(
                "f.rs",
                "y",
                args("replace_range", "replace_range L1-1", Some("foo")),
            )
            .unwrap();

        let perms = scratch_perms();
        let r = execute(
            &serde_json::json!({ "path": "f.rs", "rev": 99 }),
            &perms,
            &store,
        )
        .await
        .unwrap();
        assert!(!r.success);
        assert!(r.content.contains("rev_99"));
        assert!(r.content.contains("rev_0"));
        assert!(r.content.contains("rev_1"));
    }

    #[tokio::test]
    async fn missing_rev_arg_errors() {
        let store = RevisionStore::with_cap(20);
        let perms = scratch_perms();
        let r = execute(&serde_json::json!({ "path": "f.rs" }), &perms, &store)
            .await
            .unwrap();
        assert!(!r.success);
        assert!(r.content.contains("rev"));
    }

    #[tokio::test]
    async fn no_revisions_for_file_errors_helpfully() {
        let store = RevisionStore::with_cap(20);
        let perms = scratch_perms();
        let r = execute(
            &serde_json::json!({ "path": "never_seen.rs", "rev": 0 }),
            &perms,
            &store,
        )
        .await
        .unwrap();
        assert!(!r.success);
        assert!(r.content.contains("no revisions recorded"));
    }

    #[tokio::test]
    async fn insert_at_uses_text_key_not_new_text() {
        let store = RevisionStore::with_cap(20);
        store.ensure_pristine("f.rs", "x").unwrap();
        store
            .record(
                "f.rs",
                "y",
                args("insert_at", "insert_at after L0", Some("HELLO")),
            )
            .unwrap();
        let perms = scratch_perms();
        let r = execute(
            &serde_json::json!({ "path": "f.rs", "rev": 1 }),
            &perms,
            &store,
        )
        .await
        .unwrap();
        assert!(r.content.contains("text: |"));
        assert!(!r.content.contains("new_text: |"));
        assert!(r.content.contains("HELLO"));
    }
}
