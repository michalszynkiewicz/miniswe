//! Shared helpers used by every per-language symbol extractor.

/// Extract an identifier name immediately after `keyword` in `line`.
/// `keyword` may be empty, in which case the identifier is taken from the
/// start of `line`.
pub fn extract_name_after(line: &str, keyword: &str) -> Option<String> {
    let after = if keyword.is_empty() {
        line
    } else {
        line.split(keyword).nth(1)?
    };
    let name: String = after
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    if name.is_empty() { None } else { Some(name) }
}
