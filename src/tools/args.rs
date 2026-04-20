//! Tool argument parsing helpers.
//!
//! Execution-path tools should use `require_*` so a missing or wrong-type
//! key produces a clear error the LLM can correct — the existing system
//! prompt already instructs it to retry with the exact required parameter
//! names. Display / compression paths use the `get_*_or` helpers for
//! graceful UI rendering instead.

use serde_json::Value;

fn kind_of(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Require a string-valued `key` in `args`. Returns an error string suitable
/// for wrapping in `ToolResult::err`.
pub fn require_str<'a>(args: &'a Value, key: &str) -> Result<&'a str, String> {
    match &args[key] {
        Value::String(s) => Ok(s.as_str()),
        Value::Null => Err(format!(
            "Missing required parameter: '{key}' (expected string)"
        )),
        other => Err(format!(
            "Parameter '{key}' must be a string, got {}",
            kind_of(other)
        )),
    }
}

/// Require a non-negative integer `key`.
pub fn require_u64(args: &Value, key: &str) -> Result<u64, String> {
    match &args[key] {
        Value::Number(n) => n
            .as_u64()
            .ok_or_else(|| format!("Parameter '{key}' must be a non-negative integer, got {n}")),
        Value::Null => Err(format!(
            "Missing required parameter: '{key}' (expected integer)"
        )),
        other => Err(format!(
            "Parameter '{key}' must be an integer, got {}",
            kind_of(other)
        )),
    }
}

/// Optional string: `None` if absent/null, `Some(s)` if present and string,
/// `Err` if present but wrong type.
pub fn opt_str<'a>(args: &'a Value, key: &str) -> Result<Option<&'a str>, String> {
    match &args[key] {
        Value::Null => Ok(None),
        Value::String(s) => Ok(Some(s.as_str())),
        other => Err(format!(
            "Parameter '{key}' must be a string if present, got {}",
            kind_of(other)
        )),
    }
}

/// Optional u64: `None` if absent/null, `Some(n)` if valid, `Err` otherwise.
pub fn opt_u64(args: &Value, key: &str) -> Result<Option<u64>, String> {
    match &args[key] {
        Value::Null => Ok(None),
        Value::Number(n) => n
            .as_u64()
            .map(Some)
            .ok_or_else(|| format!("Parameter '{key}' must be a non-negative integer, got {n}")),
        other => Err(format!(
            "Parameter '{key}' must be an integer if present, got {}",
            kind_of(other)
        )),
    }
}

/// Cosmetic: always returns a string, falling back to `default` for
/// missing or non-string values. Use when rendering labels, not when
/// executing logic.
pub fn get_str_or<'a>(args: &'a Value, key: &str, default: &'a str) -> &'a str {
    args.get(key).and_then(|v| v.as_str()).unwrap_or(default)
}

/// Cosmetic integer getter — same shape as `get_str_or`.
pub fn get_u64_or(args: &Value, key: &str, default: u64) -> u64 {
    args.get(key).and_then(|v| v.as_u64()).unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn require_str_missing() {
        let args = json!({});
        assert_eq!(
            require_str(&args, "path").unwrap_err(),
            "Missing required parameter: 'path' (expected string)"
        );
    }

    #[test]
    fn require_str_null() {
        let args = json!({ "path": null });
        assert!(
            require_str(&args, "path")
                .unwrap_err()
                .starts_with("Missing required parameter")
        );
    }

    #[test]
    fn require_str_wrong_type() {
        let args = json!({ "path": 42 });
        assert_eq!(
            require_str(&args, "path").unwrap_err(),
            "Parameter 'path' must be a string, got number"
        );
    }

    #[test]
    fn require_str_ok() {
        let args = json!({ "path": "src/foo.rs" });
        assert_eq!(require_str(&args, "path").unwrap(), "src/foo.rs");
    }

    #[test]
    fn require_u64_wrong_type() {
        let args = json!({ "timeout": "60" });
        assert_eq!(
            require_u64(&args, "timeout").unwrap_err(),
            "Parameter 'timeout' must be an integer, got string"
        );
    }

    #[test]
    fn opt_u64_absent_is_none() {
        let args = json!({});
        assert_eq!(opt_u64(&args, "timeout").unwrap(), None);
    }

    #[test]
    fn opt_u64_wrong_type_is_err() {
        let args = json!({ "timeout": "60" });
        assert!(opt_u64(&args, "timeout").is_err());
    }

    #[test]
    fn get_str_or_falls_back() {
        let args = json!({});
        assert_eq!(get_str_or(&args, "path", "?"), "?");
    }
}
