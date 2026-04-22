//! Tree-sitter AST parse check for fast-mode feedback.
//!
//! Tiny helper — takes a file path + proposed content, runs a syntactic
//! parse through tree-sitter, and returns either `Ok(())` (parses cleanly)
//! or `Err(first_error_line:col: msg)`. Used to give the model an
//! immediate signal when an edit introduces a bracket mismatch or other
//! structural break, independent of whether LSP is running.
//!
//! Supported languages match ts_extract's default set: rust, python,
//! javascript, typescript, go. Other extensions return `Ok(())` — the
//! check is best-effort and unsupported languages fall through as "no
//! AST feedback available".

/// Run a tree-sitter parse over `content` for the language inferred from
/// `rel_path`'s extension. Returns `Ok(())` on clean parse or unsupported
/// language. Returns `Err(msg)` if a syntactic error is found, with a
/// one-line description pointing at the first error node.
pub fn parse_check(rel_path: &str, content: &str) -> Result<(), String> {
    let ext = extension(rel_path);
    parse_check_ext(ext, content)
}

/// Same as `parse_check` but takes an explicit extension (useful for tests
/// and for callers that already know the language).
#[cfg(feature = "tree-sitter")]
pub fn parse_check_ext(ext: &str, content: &str) -> Result<(), String> {
    use tree_sitter::Parser;

    #[cfg(feature = "lang-yaml")]
    if ext == "yaml" || ext == "yml" {
        return yaml_check(content);
    }

    let Some(lang) = language_for(ext) else {
        return Ok(()); // unsupported — treat as ok
    };
    let mut parser = Parser::new();
    if parser.set_language(&lang).is_err() {
        return Ok(()); // parser init failed — skip check
    }
    let Some(tree) = parser.parse(content, None) else {
        return Err("tree-sitter parse returned no tree".into());
    };
    let root = tree.root_node();
    if !root.has_error() && !has_missing(root) {
        return Ok(());
    }

    // Walk to the first ERROR or MISSING node and describe it.
    let err = first_error(root).unwrap_or(root);
    let start = err.start_position();
    let line = start.row + 1;
    let col = start.column + 1;
    let kind = if err.is_missing() {
        format!("missing {}", err.kind())
    } else if err.is_error() {
        "syntax error".into()
    } else {
        // has_missing but the root itself reports no error: walk children
        // to find it. first_error already does this, so this branch
        // shouldn't trigger, but keep a safe fallback.
        "syntax error".into()
    };
    Err(format!("{line}:{col}: {kind}"))
}

#[cfg(not(feature = "tree-sitter"))]
pub fn parse_check_ext(_ext: &str, _content: &str) -> Result<(), String> {
    Ok(())
}

#[cfg(feature = "tree-sitter")]
fn has_missing(node: tree_sitter::Node) -> bool {
    if node.is_missing() {
        return true;
    }
    for i in 0..node.child_count() as u32 {
        if let Some(child) = node.child(i)
            && has_missing(child)
        {
            return true;
        }
    }
    false
}

#[cfg(feature = "tree-sitter")]
fn first_error(node: tree_sitter::Node) -> Option<tree_sitter::Node> {
    if node.is_error() || node.is_missing() {
        return Some(node);
    }
    for i in 0..node.child_count() as u32 {
        if let Some(child) = node.child(i)
            && let Some(err) = first_error(child)
        {
            return Some(err);
        }
    }
    None
}

/// YAML syntax + duplicate-key check using tree-sitter-yaml.
/// tree-sitter catches structural syntax errors; we walk the mapping nodes
/// ourselves to catch duplicate keys (technically valid YAML syntax but
/// almost always a bug — parsers silently last-value-wins).
#[cfg(feature = "lang-yaml")]
fn yaml_check(content: &str) -> Result<(), String> {
    use tree_sitter::Parser;

    let mut parser = Parser::new();
    if parser
        .set_language(&tree_sitter_yaml::LANGUAGE.into())
        .is_err()
    {
        return Ok(());
    }
    let Some(tree) = parser.parse(content, None) else {
        return Err("tree-sitter parse returned no tree".into());
    };
    let root = tree.root_node();
    if root.has_error() {
        let err = first_error(root).unwrap_or(root);
        let start = err.start_position();
        return Err(format!(
            "{}:{}: syntax error",
            start.row + 1,
            start.column + 1
        ));
    }
    yaml_check_dup_keys(root, content)
}

/// Recursively walk the tree and report the first duplicate key in any mapping.
#[cfg(feature = "lang-yaml")]
fn yaml_check_dup_keys(node: tree_sitter::Node, src: &str) -> Result<(), String> {
    if node.kind() == "block_mapping" || node.kind() == "flow_mapping" {
        let pair_kind = if node.kind() == "block_mapping" {
            "block_mapping_pair"
        } else {
            "flow_pair"
        };
        let mut seen: std::collections::HashMap<String, (usize, usize)> = Default::default();
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() != pair_kind {
                continue;
            }
            // Key is the first named child of the pair
            if let Some(key_node) = child.child(0) {
                let key_text = &src[key_node.start_byte()..key_node.end_byte()];
                let row = key_node.start_position().row + 1;
                let col = key_node.start_position().column + 1;
                if let Some(&(prev_row, prev_col)) = seen.get(key_text) {
                    return Err(format!(
                        "{row}:{col}: duplicate key '{key_text}' (first defined at {prev_row}:{prev_col})"
                    ));
                }
                seen.insert(key_text.to_string(), (row, col));
            }
        }
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        yaml_check_dup_keys(child, src)?;
    }
    Ok(())
}

#[cfg(feature = "tree-sitter")]
fn language_for(ext: &str) -> Option<tree_sitter::Language> {
    match ext {
        #[cfg(feature = "lang-rust")]
        "rs" => Some(tree_sitter_rust::LANGUAGE.into()),

        #[cfg(feature = "lang-python")]
        "py" => Some(tree_sitter_python::LANGUAGE.into()),

        #[cfg(feature = "lang-javascript")]
        "js" | "jsx" | "mjs" | "cjs" => Some(tree_sitter_javascript::LANGUAGE.into()),

        #[cfg(feature = "lang-typescript")]
        "ts" | "mts" => Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
        #[cfg(feature = "lang-typescript")]
        "tsx" => Some(tree_sitter_typescript::LANGUAGE_TSX.into()),

        #[cfg(feature = "lang-go")]
        "go" => Some(tree_sitter_go::LANGUAGE.into()),

        // YAML: handled separately (dup-key check), not via language_for
        _ => None,
    }
}

fn extension(path: &str) -> &str {
    std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_language_returns_ok() {
        assert!(parse_check("README.md", "## heading\n\nbody").is_ok());
        assert!(parse_check("data.xml", "<root>").is_ok());
    }

    #[cfg(feature = "lang-rust")]
    #[test]
    fn clean_rust_parses() {
        let src = "fn main() { println!(\"hi\"); }\n";
        assert!(parse_check("m.rs", src).is_ok());
    }

    #[cfg(feature = "lang-rust")]
    #[test]
    fn unclosed_brace_flagged() {
        let src = "fn main() { let x = 1;\n";
        let err = parse_check("m.rs", src).unwrap_err();
        assert!(
            err.contains(':'),
            "error should carry a line:col prefix: {err}"
        );
    }

    #[cfg(feature = "lang-rust")]
    #[test]
    fn missing_semicolon_in_fn_body_flagged() {
        // tree-sitter reports this as a missing node / error
        let src = "fn f() -> i32 { let x = 1 x }\n";
        let err = parse_check("m.rs", src).unwrap_err();
        assert!(err.contains(':'));
    }

    #[cfg(feature = "lang-python")]
    #[test]
    fn clean_python_parses() {
        assert!(parse_check("m.py", "def f():\n    return 1\n").is_ok());
    }

    #[cfg(feature = "lang-yaml")]
    #[test]
    fn clean_yaml_parses() {
        let src = "key: value\nother: 123\nnested:\n  a: 1\n  b: 2\n";
        assert!(parse_check("zarf.yaml", src).is_ok());
    }

    #[cfg(feature = "lang-yaml")]
    #[test]
    fn yaml_duplicate_key_caught() {
        let src = "charts:\n  - name: podinfo\n    valuesFiles:\n      - a.yaml\n    valuesFiles:\n      - b.yaml\n";
        let err = parse_check("common/zarf.yaml", src).unwrap_err();
        assert!(err.contains("duplicate key 'valuesFiles'"), "got: {err}");
        assert!(err.contains("first defined at"), "got: {err}");
    }

    #[cfg(feature = "lang-yaml")]
    #[test]
    fn yaml_top_level_duplicate_key_caught() {
        let src = "name: foo\nversion: 1\nname: bar\n";
        let err = parse_check("config.yml", src).unwrap_err();
        assert!(err.contains("duplicate key 'name'"), "got: {err}");
    }

    #[cfg(feature = "lang-yaml")]
    #[test]
    fn yaml_syntax_error_caught() {
        let src = "key: :\n  bad indent\n";
        assert!(parse_check("bad.yaml", src).is_err());
    }
}
