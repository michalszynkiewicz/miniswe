//! Canonical keying for tool-call loop detection.
//!
//! The agent loop tracks the last few `(tool_name, args)` pairs it has
//! dispatched. When the same key shows up 3× in a row, the
//! `loop_detected_hint` injection fires. Keys must be stable under
//! irrelevant JSON differences (object key ordering, insignificant
//! whitespace), which is what `canonical_json` provides.

pub fn loop_call_key(tool_name: &str, args: &serde_json::Value) -> String {
    format!("{tool_name}:{}", canonical_json(args))
}

pub fn canonical_json(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::String(s) => serde_json::to_string(s).unwrap_or_else(|_| "\"\"".into()),
        serde_json::Value::Array(items) => {
            let inner = items
                .iter()
                .map(canonical_json)
                .collect::<Vec<_>>()
                .join(",");
            format!("[{inner}]")
        }
        serde_json::Value::Object(map) => {
            let mut entries = map.iter().collect::<Vec<_>>();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            let inner = entries
                .into_iter()
                .map(|(k, v)| {
                    let key = serde_json::to_string(k).unwrap_or_else(|_| "\"\"".into());
                    format!("{key}:{}", canonical_json(v))
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("{{{inner}}}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn object_key_order_does_not_affect_canonical_form() {
        let a = canonical_json(&json!({ "b": 1, "a": 2 }));
        let b = canonical_json(&json!({ "a": 2, "b": 1 }));
        assert_eq!(a, b);
    }

    #[test]
    fn loop_key_combines_name_and_canonical_args() {
        let key = loop_call_key("write_file", &json!({ "path": "x.rs" }));
        assert_eq!(key, "write_file:{\"path\":\"x.rs\"}");
    }
}
