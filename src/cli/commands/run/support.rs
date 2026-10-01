//! Support helpers for the run loop: patch-path parsing.

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
