//! Compress a markdown project-profile to a dense
//! `[SECTION]key=value|key=value` notation.

/// Compress a project profile from prose to structured key-value format.
///
/// Input: markdown profile with headings and bullet points
/// Output: dense [SECTION]key=value|key=value notation
pub fn compress_profile(profile: &str) -> String {
    let mut result = String::new();
    for line in profile.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("# ") {
            // Skip the main heading
            continue;
        }

        if trimmed.starts_with("## ") {
            let section = trimmed
                .trim_start_matches("## ")
                .to_uppercase()
                .replace(' ', "_");
            result.push_str(&format!("[{section}]"));
            continue;
        }

        if trimmed.starts_with("- ") {
            let item = trimmed.trim_start_matches("- ");
            // Convert "Key: value" to "key=value"
            if let Some((key, value)) = item.split_once(": ") {
                let short_key = key
                    .to_lowercase()
                    .replace(' ', "_")
                    .chars()
                    .take(12)
                    .collect::<String>();
                result.push_str(&format!("{short_key}={value}|"));
            } else {
                result.push_str(item);
                result.push('|');
            }
        }
    }

    // Clean trailing pipe
    if result.ends_with('|') {
        result.pop();
    }

    result.push('\n');
    result
}
