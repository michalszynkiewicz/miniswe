//! Tree-sitter-based extraction of the exact source-line span of a
//! function DEFINITION's parameter list, or a CALL's argument list, given
//! a known position (the LSP-resolved location of the function/reference
//! name). Replaces the hand-rolled lexer-aware paren scanner that used to
//! live in `sites.rs` — see git history around 2026-07-05 for the bug
//! trail (parens inside string arguments, non-call references mistaken
//! for calls, Go receiver clauses, backtick raw strings) that motivated
//! the switch to a real per-language parser instead of patching instances
//! of that bug class one at a time.
//!
//! Verified against the actual compiled grammars (not just assumed from
//! documentation) that `child_by_field_name("parameters")` /
//! `("arguments")` are consistent field names across every language here,
//! even though the underlying node *kind* names vary (Rust: `parameters`/
//! `arguments`; Python: `parameters`/`argument_list`; Go: `parameter_list`/
//! `argument_list`; JS/TS: `formal_parameters`/`arguments`; Java:
//! `formal_parameters`/`argument_list`). This is why only two small
//! per-language tables are needed (which node kinds count as "a call" and
//! "a definition") rather than a field-name table too. Go's
//! `method_declaration` also exposes the receiver clause as a SEPARATE
//! `"receiver"` field from `"parameters"`, so asking for `"parameters"`
//! specifically already returns the real parameter list, not the receiver
//! — the exact problem the old scanner needed a hand-written heuristic for.

#[cfg(feature = "tree-sitter")]
use tree_sitter::{Node, Parser};

/// Node kinds that count as "a function/method call" — the ancestor walk
/// stops at the first node whose kind is in this set. A reference used as
/// a plain value (e.g. a function pointer, `let f = foo;`) never has such
/// an ancestor, so the walk safely reaches the tree root and returns
/// `None` — no special-casing needed, unlike the old scanner's hand-built
/// "is this actually followed by `(`" guard.
#[cfg(feature = "tree-sitter")]
fn call_node_kinds(ext: &str) -> &'static [&'static str] {
    match ext {
        "rs" => &["call_expression"],
        "py" => &["call"],
        "go" => &["call_expression"],
        "js" | "jsx" | "mjs" | "cjs" | "ts" | "mts" | "tsx" => &["call_expression"],
        #[cfg(feature = "lang-java")]
        "java" => &["method_invocation"],
        _ => &[],
    }
}

/// Node kinds that count as "a function/method definition".
#[cfg(feature = "tree-sitter")]
fn definition_node_kinds(ext: &str) -> &'static [&'static str] {
    match ext {
        "rs" => &["function_item"],
        "py" => &["function_definition"],
        // Go: plain functions and receiver methods are different node kinds.
        "go" => &["function_declaration", "method_declaration"],
        // JS/TS: plain functions and class methods are different node kinds.
        "js" | "jsx" | "mjs" | "cjs" | "ts" | "mts" | "tsx" => {
            &["function_declaration", "method_definition"]
        }
        #[cfg(feature = "lang-java")]
        "java" => &["method_declaration", "constructor_declaration"],
        _ => &[],
    }
}

/// Extension → `tree_sitter::Language`. Deliberately a separate, small
/// match rather than sharing `tools::fast::ast`'s private `language_for`
/// — that one is scoped to fast-mode syntax-error feedback and doesn't
/// cover Java; this project's existing tree-sitter integrations already
/// keep per-purpose language tables rather than one shared one (see
/// `knowledge::ts_extract::get_tags_config` vs `tools::fast::ast`).
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
        #[cfg(feature = "lang-java")]
        "java" => Some(tree_sitter_java::LANGUAGE.into()),
        _ => None,
    }
}

#[cfg(feature = "tree-sitter")]
fn extension(path: &str) -> &str {
    std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
}

#[cfg(feature = "tree-sitter")]
fn parse(ext: &str, source: &str) -> Option<tree_sitter::Tree> {
    let lang = language_for(ext)?;
    let mut parser = Parser::new();
    parser.set_language(&lang).ok()?;
    parser.parse(source, None)
}

/// Byte offset of the first non-whitespace character on `line` (0-based),
/// or `None` if the line is out of range. Anchoring on whitespace would
/// make `descendant_for_byte_range` resolve to whatever encloses the gap
/// between tokens rather than the definition itself.
#[cfg(feature = "tree-sitter")]
fn first_non_ws_byte_on_line(source: &str, line: usize) -> Option<usize> {
    let mut byte = 0usize;
    for (i, l) in source.lines().enumerate() {
        if i == line {
            let offset = l.find(|c: char| !c.is_whitespace()).unwrap_or(0);
            return Some(byte + offset);
        }
        byte += l.len() + 1; // +1 for the '\n' `lines()` strips
    }
    None
}

/// Byte offset for 0-based (line, column) — same loose byte-vs-UTF-16
/// convention the rest of this module already uses for LSP positions
/// (see `find_identifier_column`), not a fully UTF-16-correct conversion.
#[cfg(feature = "tree-sitter")]
fn byte_offset_for(source: &str, line: usize, column: usize) -> Option<usize> {
    let mut byte = 0usize;
    for (i, l) in source.lines().enumerate() {
        if i == line {
            return Some(byte + column.min(l.len()));
        }
        byte += l.len() + 1;
    }
    None
}

/// Given a byte position, walk up from the smallest containing node until
/// one whose kind is in `kinds` is found.
#[cfg(feature = "tree-sitter")]
fn find_ancestor<'a>(root: Node<'a>, byte: usize, kinds: &[&str]) -> Option<Node<'a>> {
    let mut cur = root.descendant_for_byte_range(byte, byte)?;
    loop {
        if kinds.contains(&cur.kind()) {
            return Some(cur);
        }
        cur = cur.parent()?;
    }
}

/// Extend a byte range to whole source lines: from the line containing
/// `start_byte` through the line containing `end_byte`, inclusive —
/// matching the old scanner's convention (the model's NEW text is
/// expected to reproduce the full line, e.g. `let x = foo(...)`, not just
/// the call/parameter-list expression).
#[cfg(feature = "tree-sitter")]
fn whole_lines(source: &str, start_byte: usize, end_byte: usize) -> String {
    let start_line = source[..start_byte].matches('\n').count();
    let end_line = source[..end_byte].matches('\n').count();
    source
        .lines()
        .skip(start_line)
        .take(end_line - start_line + 1)
        .collect::<Vec<_>>()
        .join("\n")
}

/// Deterministic OLD text for a signature rewrite: the whole-line span
/// from `from_line` (0-based, the function's `fn`/`def`/`func` line)
/// through the line containing the parameter list's closing paren.
/// `path` is used only to infer the language from its extension. Returns
/// `None` on any uncertainty (unsupported language, parse failure, no
/// matching ancestor) — callers fall back to the model-transcribes-OLD
/// path, exactly as before.
#[cfg(feature = "tree-sitter")]
pub fn signature_span(source: &str, path: &str, from_line: usize) -> Option<String> {
    let ext = extension(path);
    let tree = parse(ext, source)?;
    let byte = first_non_ws_byte_on_line(source, from_line)?;
    let def = find_ancestor(tree.root_node(), byte, definition_node_kinds(ext))?;
    let params = def.child_by_field_name("parameters")?;
    Some(whole_lines(source, def.start_byte(), params.end_byte()))
}

#[cfg(not(feature = "tree-sitter"))]
pub fn signature_span(_source: &str, _path: &str, _from_line: usize) -> Option<String> {
    None
}

/// Deterministic OLD text for a callsite rewrite: the whole-line span
/// from `line_0` (0-based, the line the call expression starts on, per
/// the LSP-resolved reference position) through the line containing the
/// call's closing paren. `column_0` disambiguates an earlier unrelated
/// call on the same line. Returns `None` on any uncertainty, including
/// when the reference isn't actually called (used as a plain value) —
/// the ancestor walk simply never finds a call-kind node in that case, no
/// special-casing needed.
#[cfg(feature = "tree-sitter")]
pub fn callsite_span(source: &str, path: &str, line_0: u32, column_0: u32) -> Option<String> {
    let ext = extension(path);
    let tree = parse(ext, source)?;
    let byte = byte_offset_for(source, line_0 as usize, column_0 as usize)?;
    let call = find_ancestor(tree.root_node(), byte, call_node_kinds(ext))?;
    let args = call.child_by_field_name("arguments")?;
    Some(whole_lines(source, call.start_byte(), args.end_byte()))
}

#[cfg(not(feature = "tree-sitter"))]
pub fn callsite_span(_source: &str, _path: &str, _line_0: u32, _column_0: u32) -> Option<String> {
    None
}

/// Best-effort check: does the function whose signature starts at
/// `from_line` (0-based) already have a parameter named `param_name`?
/// Used to make `add_param` idempotent. False-negative-only by design
/// (matches the old hand-rolled version's safety contract): on any
/// uncertainty (unsupported language, parse failure, no matching
/// ancestor) this returns `false`, which only risks re-adding an existing
/// parameter — `add_param`'s own duplicate-name validation is the actual
/// backstop for that, this is just an early, friendlier rejection.
#[cfg(feature = "tree-sitter")]
pub fn has_param(source: &str, path: &str, from_line: usize, param_name: &str) -> bool {
    let ext = extension(path);
    let Some(tree) = parse(ext, source) else {
        return false;
    };
    let Some(byte) = first_non_ws_byte_on_line(source, from_line) else {
        return false;
    };
    let Some(def) = find_ancestor(tree.root_node(), byte, definition_node_kinds(ext)) else {
        return false;
    };
    let Some(params) = def.child_by_field_name("parameters") else {
        return false;
    };
    let mut cursor = params.walk();
    for child in params.children(&mut cursor) {
        if !child.is_named() {
            continue; // skip punctuation tokens: `(`, `)`, `,`
        }
        // The binding name is whatever `child_by_field_name("name")`/
        // ("pattern") resolves to for this language's parameter node; if
        // neither field exists, fall back to the first identifier-like
        // token in the parameter's own text (covers e.g. Rust's plain
        // `parameter` node, which doesn't expose a named field for the
        // binding, only for typed patterns).
        let name_node = child
            .child_by_field_name("name")
            .or_else(|| child.child_by_field_name("pattern"));
        let name_text = match name_node {
            Some(n) => &source[n.start_byte()..n.end_byte()],
            None => {
                let text = &source[child.start_byte()..child.end_byte()];
                let end = text
                    .find(|c: char| !(c.is_alphanumeric() || c == '_'))
                    .unwrap_or(text.len());
                text[..end].trim()
            }
        };
        if name_text == param_name {
            return true;
        }
    }
    false
}

#[cfg(not(feature = "tree-sitter"))]
pub fn has_param(_source: &str, _path: &str, _from_line: usize, _param_name: &str) -> bool {
    false
}

#[cfg(all(test, feature = "tree-sitter"))]
mod signature_span_tests {
    use super::signature_span;

    #[test]
    #[cfg(feature = "lang-rust")]
    fn rust_single_line_signature() {
        let src = "fn foo(a: u32, b: u32) -> u32 {\n    a + b\n}\n";
        assert_eq!(
            signature_span(src, "m.rs", 0),
            Some("fn foo(a: u32, b: u32) -> u32 {".to_string())
        );
    }

    #[test]
    #[cfg(feature = "lang-rust")]
    fn rust_multi_line_signature_includes_trailing_return_type_line() {
        let src = "pub fn assemble(\n    config: &Config,\n    mcp_summary: Option<&str>,\n) -> AssembledContext {\n    todo!()\n}\n";
        assert_eq!(
            signature_span(src, "m.rs", 0),
            Some(
                "pub fn assemble(\n    config: &Config,\n    mcp_summary: Option<&str>,\n) -> AssembledContext {"
                    .to_string()
            )
        );
    }

    #[test]
    #[cfg(feature = "lang-rust")]
    fn rust_from_line_offset_into_the_middle_of_a_file() {
        let src = "use std::foo;\n\nfn bar(x: u32) -> u32 {\n    x\n}\n";
        assert_eq!(
            signature_span(src, "m.rs", 2),
            Some("fn bar(x: u32) -> u32 {".to_string())
        );
    }

    #[test]
    #[cfg(feature = "lang-rust")]
    fn rust_no_function_at_that_line_returns_none() {
        let src = "const X: u32 = 5;\n";
        assert_eq!(signature_span(src, "m.rs", 0), None);
    }

    #[test]
    #[cfg(feature = "lang-rust")]
    fn rust_nested_generic_in_param_type_is_not_confused() {
        let src = "fn foo(a: Vec<(u32, u32)>, b: u32) -> u32 {\n    b\n}\n";
        assert_eq!(
            signature_span(src, "m.rs", 0),
            Some("fn foo(a: Vec<(u32, u32)>, b: u32) -> u32 {".to_string())
        );
    }

    #[test]
    #[cfg(feature = "lang-rust")]
    fn rust_function_returning_a_closure() {
        let src = "pub fn make_adder(x: i32) -> impl Fn(i32) -> i32 {\n    todo!()\n}\n";
        assert_eq!(
            signature_span(src, "m.rs", 0),
            Some("pub fn make_adder(x: i32) -> impl Fn(i32) -> i32 {".to_string())
        );
    }

    #[test]
    #[cfg(feature = "lang-go")]
    fn go_plain_function() {
        let src = "func foo(a int, b int) int {\n\treturn a + b\n}\n";
        assert_eq!(
            signature_span(src, "m.go", 0),
            Some("func foo(a int, b int) int {".to_string())
        );
    }

    #[test]
    #[cfg(feature = "lang-go")]
    fn go_method_with_receiver_returns_real_params_not_receiver() {
        let src = "func (r *T) foo(\n\ta int,\n\tb int,\n) int {\n\treturn a + b\n}\n";
        assert_eq!(
            signature_span(src, "m.go", 0),
            Some("func (r *T) foo(\n\ta int,\n\tb int,\n) int {".to_string())
        );
    }

    #[test]
    #[cfg(feature = "lang-go")]
    fn go_receiver_without_pointer() {
        let src = "func (t T) foo(a int, b string) bool {\n\treturn true\n}\n";
        assert_eq!(
            signature_span(src, "m.go", 0),
            Some("func (t T) foo(a int, b string) bool {".to_string())
        );
    }

    #[test]
    #[cfg(feature = "lang-python")]
    fn python_function() {
        let src = "def foo(a, b):\n    return a + b\n";
        assert_eq!(
            signature_span(src, "m.py", 0),
            Some("def foo(a, b):".to_string())
        );
    }

    #[test]
    #[cfg(feature = "lang-python")]
    fn python_method_self_is_a_plain_param_not_a_receiver() {
        let src = "class C:\n    def foo(self, a, b):\n        return a + b\n";
        assert_eq!(
            signature_span(src, "m.py", 1),
            Some("    def foo(self, a, b):".to_string())
        );
    }

    #[test]
    #[cfg(feature = "lang-python")]
    fn python_default_value_string_with_a_paren_doesnt_confuse_the_balance() {
        let src = "def foo(a, msg=\"(unbalanced\"):\n    pass\n";
        assert_eq!(
            signature_span(src, "m.py", 0),
            Some("def foo(a, msg=\"(unbalanced\"):".to_string())
        );
    }

    #[test]
    #[cfg(feature = "lang-typescript")]
    fn typescript_function() {
        let src = "function foo(a: number, b: number): number {\n    return a + b;\n}\n";
        assert_eq!(
            signature_span(src, "m.ts", 0),
            Some("function foo(a: number, b: number): number {".to_string())
        );
    }

    #[test]
    #[cfg(feature = "lang-javascript")]
    fn javascript_class_method() {
        let src = "class C {\n  foo(a, b) {\n    return a + b;\n  }\n}\n";
        assert_eq!(
            signature_span(src, "m.js", 1),
            Some("  foo(a, b) {".to_string())
        );
    }

    #[test]
    #[cfg(feature = "lang-java")]
    fn java_method() {
        let src = "class C {\n    int foo(int a, int b) {\n        return a + b;\n    }\n}\n";
        assert_eq!(
            signature_span(src, "m.java", 1),
            Some("    int foo(int a, int b) {".to_string())
        );
    }
}

#[cfg(all(test, feature = "tree-sitter"))]
mod callsite_span_tests {
    use super::callsite_span;

    fn check_ident(path: &str, src: &str, ident: &str, expected: &str) {
        let col = src
            .find(ident)
            .unwrap_or_else(|| panic!("fixture must contain `{ident}`"));
        let line = src[..col].matches('\n').count() as u32;
        let line_start = src[..col].rfind('\n').map(|i| i + 1).unwrap_or(0);
        let column = (col - line_start) as u32;
        assert_eq!(
            callsite_span(src, path, line, column),
            Some(expected.to_string()),
            "src: {src:?}"
        );
    }

    fn check(path: &str, src: &str, expected: &str) {
        check_ident(path, src, "foo", expected);
    }

    #[test]
    #[cfg(feature = "lang-rust")]
    fn rust_single_line_call() {
        check(
            "m.rs",
            "fn main() {\n    let x = foo(1, 2);\n}\n",
            "    let x = foo(1, 2);",
        );
    }

    #[test]
    #[cfg(feature = "lang-rust")]
    fn rust_multi_line_call_one_arg_per_line() {
        // "assemble" alone would first match inside "assembl**ed**" (the
        // local variable name) — search for the qualified call instead.
        check_ident(
            "m.rs",
            "fn main() {\n    let assembled = context::assemble(\n        &config,\n        message,\n    );\n}\n",
            "context::assemble",
            "    let assembled = context::assemble(\n        &config,\n        message,\n    );",
        );
    }

    #[test]
    #[cfg(feature = "lang-rust")]
    fn rust_column_skips_an_earlier_unrelated_call_on_the_same_line() {
        check(
            "m.rs",
            "fn main() {\n    let x = bar(z) + foo(1, 2);\n}\n",
            "    let x = bar(z) + foo(1, 2);",
        );
    }

    #[test]
    #[cfg(feature = "lang-rust")]
    fn rust_reference_used_as_a_value_not_called_returns_none() {
        let src = "fn main() {\n    let f = foo;\n    bar(1, 2);\n}\n";
        let col = src.find("foo").unwrap();
        let line = src[..col].matches('\n').count() as u32;
        let line_start = src[..col].rfind('\n').map(|i| i + 1).unwrap_or(0);
        let column = (col - line_start) as u32;
        assert_eq!(callsite_span(src, "m.rs", line, column), None);
    }

    #[test]
    #[cfg(feature = "lang-rust")]
    fn rust_method_call_as_a_parameter() {
        check(
            "m.rs",
            "fn main() {\n    foo(bar(x), y);\n}\n",
            "    foo(bar(x), y);",
        );
    }

    #[test]
    #[cfg(feature = "lang-rust")]
    fn rust_string_argument_containing_unbalanced_paren() {
        check(
            "m.rs",
            "fn main() {\n    foo(\"hello (world\", x);\n}\n",
            "    foo(\"hello (world\", x);",
        );
    }

    #[test]
    #[cfg(feature = "lang-rust")]
    fn rust_string_argument_containing_paren_that_looks_like_a_close() {
        check(
            "m.rs",
            "fn main() {\n    foo(\"a) b(c\", x);\n}\n",
            "    foo(\"a) b(c\", x);",
        );
    }

    #[test]
    #[cfg(feature = "lang-rust")]
    fn rust_raw_string_with_regex_parens() {
        check(
            "m.rs",
            "fn main() {\n    foo(r\"(\\d+)\", x);\n}\n",
            "    foo(r\"(\\d+)\", x);",
        );
    }

    #[test]
    #[cfg(feature = "lang-rust")]
    fn rust_hash_delimited_raw_string_containing_a_literal_quote() {
        check(
            "m.rs",
            "fn main() {\n    foo(r#\"a \"quoted\" (paren)\"#, x);\n}\n",
            "    foo(r#\"a \"quoted\" (paren)\"#, x);",
        );
    }

    #[test]
    #[cfg(feature = "lang-rust")]
    fn rust_byte_string() {
        check(
            "m.rs",
            "fn main() {\n    foo(b\"(\", x);\n}\n",
            "    foo(b\"(\", x);",
        );
    }

    #[test]
    #[cfg(feature = "lang-rust")]
    fn rust_char_literal_paren() {
        check(
            "m.rs",
            "fn main() {\n    foo('(', x);\n}\n",
            "    foo('(', x);",
        );
    }

    #[test]
    #[cfg(feature = "lang-rust")]
    fn rust_turbofish_generic_call() {
        check(
            "m.rs",
            "fn main() {\n    foo::<u32>(a, b);\n}\n",
            "    foo::<u32>(a, b);",
        );
    }

    #[test]
    #[cfg(feature = "lang-rust")]
    fn rust_nested_block_comment() {
        check(
            "m.rs",
            "fn main() {\n    foo(a /* outer /* inner */ still comment */, b);\n}\n",
            "    foo(a /* outer /* inner */ still comment */, b);",
        );
    }

    #[test]
    #[cfg(feature = "lang-rust")]
    fn rust_macro_argument_containing_a_real_call_returns_none() {
        // Genuine tree-sitter-rust limitation, verified against the real
        // grammar: a macro invocation's body parses as an opaque
        // `token_tree` (macros can have arbitrary custom syntax until
        // expansion, which a syntactic-only parser can't perform), so
        // `foo(a, b)` inside `println!(...)` never becomes a real
        // `call_expression` — `foo` shows up as a bare `identifier` with
        // no call ancestor. Returning `None` here is the correct, SAFE
        // behavior (falls back to the model-transcribes-OLD path) rather
        // than a wrong span — this is a real, narrow, honestly-documented
        // gap, not a bug. In practice `add_param`/`drop_param` callsites
        // are essentially never inside a macro's own arguments (they
        // rewrite direct calls to a renamed function), so this is a
        // low-risk edge case, not the common path.
        let src = "fn main() {\n    println!(\"{}\", foo(a, b));\n}\n";
        let col = src.find("foo").unwrap();
        let line = src[..col].matches('\n').count() as u32;
        let line_start = src[..col].rfind('\n').map(|i| i + 1).unwrap_or(0);
        let column = (col - line_start) as u32;
        assert_eq!(callsite_span(src, "m.rs", line, column), None);
    }

    #[test]
    #[cfg(feature = "lang-python")]
    fn python_fstring_with_embedded_call() {
        check(
            "m.py",
            "def main():\n    foo(f\"{bar(x)}\", y)\n",
            "    foo(f\"{bar(x)}\", y)",
        );
    }

    #[test]
    #[cfg(feature = "lang-python")]
    fn python_triple_quoted_string_with_internal_quote_and_paren() {
        check(
            "m.py",
            "def main():\n    foo(\"\"\"a \"quoted\" (paren)\"\"\", x)\n",
            "    foo(\"\"\"a \"quoted\" (paren)\"\"\", x)",
        );
    }

    #[test]
    #[cfg(feature = "lang-go")]
    fn go_raw_string_backtick() {
        check(
            "m.go",
            "func main() {\n\tfoo(`(unbalanced`, x)\n}\n",
            "\tfoo(`(unbalanced`, x)",
        );
    }

    #[test]
    #[cfg(feature = "lang-go")]
    fn go_call_via_selector_expression_on_a_receiver() {
        // Matches real Go usage: r.foo(...) — the call is a selector, not
        // a bare identifier; `foo` is the field being called.
        check("m.go", "func main() {\n\tr.foo(1, 2)\n}\n", "\tr.foo(1, 2)");
    }

    #[test]
    #[cfg(feature = "lang-typescript")]
    fn typescript_template_literal_with_embedded_call() {
        check(
            "m.ts",
            "function main() {\n    foo(`${bar(\"(\")}`, y);\n}\n",
            "    foo(`${bar(\"(\")}`, y);",
        );
    }

    #[test]
    #[cfg(feature = "lang-javascript")]
    fn javascript_regex_literal_with_parens() {
        check(
            "m.js",
            "function main() {\n    foo(/\\(\\d+\\)/, x);\n}\n",
            "    foo(/\\(\\d+\\)/, x);",
        );
    }

    #[test]
    #[cfg(feature = "lang-javascript")]
    fn javascript_arrow_function_as_argument() {
        check(
            "m.js",
            "function main() {\n    foo((x) => x + 1, y);\n}\n",
            "    foo((x) => x + 1, y);",
        );
    }

    #[test]
    #[cfg(feature = "lang-java")]
    fn java_text_block() {
        check(
            "m.java",
            "class C {\n  void m() {\n    foo(\"\"\"a \"quoted\" (paren)\"\"\", x);\n  }\n}\n",
            "    foo(\"\"\"a \"quoted\" (paren)\"\"\", x);",
        );
    }

    #[test]
    #[cfg(feature = "lang-java")]
    fn java_lambda_as_argument() {
        check(
            "m.java",
            "class C {\n  void m() {\n    foo(x -> x + 1, y);\n  }\n}\n",
            "    foo(x -> x + 1, y);",
        );
    }
}

#[cfg(all(test, feature = "tree-sitter", feature = "lang-rust"))]
mod has_param_tests {
    use super::has_param;

    #[test]
    fn detects_existing_and_ignores_absent() {
        let src = "fn assemble(\n    config: &Config,\n    user_message: &str,\n    system_prompt_override: Option<String>,\n) -> X {\n    body\n}";
        assert!(has_param(src, "m.rs", 0, "system_prompt_override"));
        assert!(has_param(src, "m.rs", 0, "config"));
        assert!(!has_param(src, "m.rs", 0, "headless"));
    }

    #[test]
    fn not_fooled_by_generic_commas() {
        let src = "fn f(map: HashMap<K, V>, name: String) {}";
        assert!(has_param(src, "m.rs", 0, "map"));
        assert!(has_param(src, "m.rs", 0, "name"));
        assert!(!has_param(src, "m.rs", 0, "V"));
        assert!(!has_param(src, "m.rs", 0, "K"));
    }
}
