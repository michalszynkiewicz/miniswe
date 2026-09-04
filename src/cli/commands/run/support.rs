//! Support helpers for the run loop: patch-path parsing and recent-activity
//! digests.

use super::*;

/// Project-relative paths touched by a unified-diff patch file (parsed from
/// its `+++ b/<path>` headers). Used by replay mode to notify the LSP of files
/// changed out-of-band by `--replay-apply`. Best-effort: returns `[]` on read error.
pub(super) fn changed_paths_in_patch(patch: &std::path::Path) -> Vec<String> {
    let Ok(text) = std::fs::read_to_string(patch) else {
        return Vec::new();
    };
    text.lines()
        .filter_map(|l| l.strip_prefix("+++ b/"))
        .map(|p| p.trim().to_string())
        .filter(|p| p != "/dev/null")
        .collect()
}

/// Render the tail of the conversation into a compact transcript — the
/// evidence of what the model just did and observed. Fed to the skill-step
/// completion judge (see `skill_router::judge_step_done`) so it decides from
/// actual recent activity, not the whole (possibly stale) history.
pub(super) fn recent_activity(messages: &[Message], n: usize, cap: usize) -> String {
    let tail: Vec<&Message> = messages.iter().rev().take(n).collect();
    let mut lines: Vec<String> = Vec::new();
    for m in tail.into_iter().rev() {
        if let Some(c) = &m.content {
            let c = c.trim();
            if !c.is_empty() {
                lines.push(format!("[{}] {}", m.role, crate::truncate_chars(c, 500)));
            }
        }
        for tc in m.tool_calls.iter().flatten() {
            lines.push(format!(
                "[{} call] {}({})",
                m.role,
                tc.function.name,
                crate::truncate_chars(&tc.function.arguments, 160)
            ));
        }
    }
    crate::truncate_chars(&lines.join("\n"), cap)
}
