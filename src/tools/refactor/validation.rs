//! Argument validation for `refactor` actions.
//!
//! Returns a single multi-problem error per call (every missing key,
//! every wrong type, every unknown key with did-you-mean) plus the
//! action's required/optional list and a copy-pasteable example. The
//! previous one-key-at-a-time errors led to multi-round chases when
//! the model misnamed several fields at once.

use std::collections::BTreeSet;

use serde_json::Value;

pub struct ArgSchema<'a> {
    pub action: &'a str,
    pub required_strings: &'a [&'a str],
    pub optional_strings: &'a [&'a str],
    pub optional_ints: &'a [&'a str],
    pub example: &'a str,
}

pub fn validate(args: &Value, schema: &ArgSchema) -> Result<(), String> {
    let mut missing: Vec<&str> = Vec::new();
    let mut bad_type: Vec<String> = Vec::new();
    let mut unknown: Vec<String> = Vec::new();

    for key in schema.required_strings {
        match &args[*key] {
            Value::String(_) => {}
            Value::Null => missing.push(*key),
            other => bad_type.push(format!("'{key}' must be string, got {}", kind_of(other))),
        }
    }
    for key in schema.optional_strings {
        match &args[*key] {
            Value::String(_) | Value::Null => {}
            other => bad_type.push(format!(
                "'{key}' must be string if present, got {}",
                kind_of(other)
            )),
        }
    }
    for key in schema.optional_ints {
        match &args[*key] {
            Value::Number(_) | Value::Null => {}
            other => bad_type.push(format!(
                "'{key}' must be integer if present, got {}",
                kind_of(other)
            )),
        }
    }

    let allowed: BTreeSet<&str> = schema
        .required_strings
        .iter()
        .chain(schema.optional_strings.iter())
        .chain(schema.optional_ints.iter())
        .copied()
        .collect();
    if let Value::Object(map) = args {
        for key in map.keys() {
            // `action` is consumed by the dispatcher; always allow.
            if key == "action" || allowed.contains(key.as_str()) {
                continue;
            }
            match closest_match(key, &allowed) {
                Some(s) => unknown.push(format!("'{key}' (did you mean '{s}'?)")),
                None => unknown.push(format!("'{key}'")),
            }
        }
    }

    if missing.is_empty() && bad_type.is_empty() && unknown.is_empty() {
        return Ok(());
    }

    let mut parts: Vec<String> = Vec::new();
    if !missing.is_empty() {
        parts.push(format!(
            "missing required parameter(s): {}",
            missing.join(", ")
        ));
    }
    if !bad_type.is_empty() {
        parts.push(format!("type error(s): {}", bad_type.join("; ")));
    }
    if !unknown.is_empty() {
        parts.push(format!("unknown parameter(s): {}", unknown.join(", ")));
    }

    let optional: Vec<&str> = schema
        .optional_strings
        .iter()
        .chain(schema.optional_ints.iter())
        .copied()
        .collect();
    Err(format!(
        "✗ change_signature({action}): {problems}\nRequired: {required}\nOptional: {optional}\nExample: {example}",
        action = schema.action,
        problems = parts.join("\n"),
        required = schema.required_strings.join(", "),
        optional = if optional.is_empty() {
            "(none)".to_string()
        } else {
            optional.join(", ")
        },
        example = schema.example,
    ))
}

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

fn levenshtein(a: &str, b: &str) -> usize {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr = vec![0usize; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            curr[j + 1] = (curr[j] + 1).min(prev[j + 1] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

fn closest_match<'a>(key: &str, candidates: &BTreeSet<&'a str>) -> Option<&'a str> {
    // Substring containment first — catches `signature_default` → `default`,
    // which Levenshtein distance (10) would otherwise reject.
    if let Some(c) = candidates
        .iter()
        .copied()
        .filter(|c| key.contains(c) || c.contains(key))
        .max_by_key(|c| c.len())
    {
        return Some(c);
    }
    candidates
        .iter()
        .copied()
        .map(|c| (levenshtein(key, c), c))
        .min_by_key(|&(d, _)| d)
        .filter(|&(d, _)| d <= 5)
        .map(|(_, c)| c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn schema() -> ArgSchema<'static> {
        ArgSchema {
            action: "add_param",
            required_strings: &["path", "name", "new_param", "position", "callsite_fill_in"],
            optional_strings: &[],
            optional_ints: &["line"],
            example: "refactor(action=\"add_param\", path=\"src/lib.rs\", name=\"f\", new_param=\"x: u32\", position=\"after:b\", callsite_fill_in=\"0\")",
        }
    }

    #[test]
    fn ok_when_all_required_present() {
        let v = json!({
            "action": "add_param",
            "path": "p", "name": "n", "new_param": "x: u32",
            "position": "start", "callsite_fill_in": "0"
        });
        assert!(validate(&v, &schema()).is_ok());
    }

    #[test]
    fn lists_all_missing_in_one_error() {
        let v = json!({"action": "add_param", "path": "p"});
        let err = validate(&v, &schema()).unwrap_err();
        assert!(err.contains("name"));
        assert!(err.contains("new_param"));
        assert!(err.contains("position"));
        assert!(err.contains("callsite_fill_in"));
        assert!(err.contains("Example:"));
    }

    #[test]
    fn flags_unknown_key_with_suggestion() {
        let v = json!({
            "action": "add_param",
            "path": "p", "name": "n", "new_param": "x: u32",
            "position": "start", "fill_in": "0"
        });
        let err = validate(&v, &schema()).unwrap_err();
        assert!(err.contains("fill_in"));
        assert!(err.contains("did you mean"));
        // `callsite_fill_in` contains `fill_in` → wins by substring match.
        assert!(err.contains("'callsite_fill_in'"));
    }

    #[test]
    fn flags_wrong_type() {
        let v = json!({
            "action": "add_param",
            "path": 123, "name": "n", "new_param": "x: u32",
            "position": "start", "callsite_fill_in": "0"
        });
        let err = validate(&v, &schema()).unwrap_err();
        assert!(err.contains("'path' must be string"));
    }
}
