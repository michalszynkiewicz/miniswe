//! Tree-sitter based symbol extraction.
//!
//! Uses tree-sitter-tags to extract definitions and references from source
//! files. This replaces the regex-based extractors with precise AST parsing.
//! Each language grammar is gated behind a feature flag to minimize build time.

#[cfg(feature = "tree-sitter")]
use tree_sitter_tags::{TagsConfiguration, TagsContext};

use super::Symbol;

/// A reference found in source code (symbol used but not defined here).
#[derive(Debug, Clone)]
pub struct SymbolRef {
    /// Name of the referenced symbol
    pub name: String,
    /// File where the reference occurs
    pub file: String,
}

/// Result of tree-sitter extraction for a single file.
#[derive(Debug, Default)]
pub struct ExtractionResult {
    pub symbols: Vec<Symbol>,
    pub references: Vec<SymbolRef>,
}

/// Get the tree-sitter tags configuration for a file extension.
/// Returns None if the language isn't supported or its feature isn't enabled.
#[cfg(feature = "tree-sitter")]
fn get_tags_config(ext: &str) -> Option<TagsConfiguration> {
    match ext {
        // Tier 1: default languages
        #[cfg(feature = "lang-rust")]
        "rs" => make_config(
            tree_sitter_rust::LANGUAGE.into(),
            tree_sitter_rust::TAGS_QUERY,
        ),

        #[cfg(feature = "lang-python")]
        "py" => make_config(
            tree_sitter_python::LANGUAGE.into(),
            tree_sitter_python::TAGS_QUERY,
        ),

        #[cfg(feature = "lang-javascript")]
        "js" | "jsx" | "mjs" | "cjs" => make_config(
            tree_sitter_javascript::LANGUAGE.into(),
            tree_sitter_javascript::TAGS_QUERY,
        ),

        #[cfg(feature = "lang-typescript")]
        "ts" | "tsx" | "mts" => {
            let lang = if ext == "tsx" {
                tree_sitter_typescript::LANGUAGE_TSX
            } else {
                tree_sitter_typescript::LANGUAGE_TYPESCRIPT
            };
            make_config(lang.into(), tree_sitter_typescript::TAGS_QUERY)
        }

        #[cfg(feature = "lang-go")]
        "go" => make_config(
            tree_sitter_go::LANGUAGE.into(),
            tree_sitter_go::TAGS_QUERY,
        ),

        // Tier 2: opt-in languages
        #[cfg(feature = "lang-java")]
        "java" => make_config(
            tree_sitter_java::LANGUAGE.into(),
            tree_sitter_java::TAGS_QUERY,
        ),

        #[cfg(feature = "lang-c")]
        "c" | "h" => make_config(
            tree_sitter_c::LANGUAGE.into(),
            tree_sitter_c::TAGS_QUERY,
        ),

        #[cfg(feature = "lang-cpp")]
        "cpp" | "hpp" | "cc" | "cxx" | "hh" => make_config(
            tree_sitter_cpp::LANGUAGE.into(),
            tree_sitter_cpp::TAGS_QUERY,
        ),

        #[cfg(feature = "lang-ruby")]
        "rb" | "rake" | "gemspec" => make_config(
            tree_sitter_ruby::LANGUAGE.into(),
            tree_sitter_ruby::TAGS_QUERY,
        ),

        #[cfg(feature = "lang-php")]
        "php" => make_config(
            tree_sitter_php::LANGUAGE_PHP.into(),
            tree_sitter_php::TAGS_QUERY,
        ),

        #[cfg(feature = "lang-csharp")]
        "cs" => make_config(
            tree_sitter_c_sharp::LANGUAGE.into(),
            tree_sitter_c_sharp::TAGS_QUERY,
        ),

        #[cfg(feature = "lang-kotlin")]
        "kt" | "kts" => make_config(
            tree_sitter_kotlin::LANGUAGE.into(),
            tree_sitter_kotlin::TAGS_QUERY,
        ),

        #[cfg(feature = "lang-swift")]
        "swift" => make_config(
            tree_sitter_swift::LANGUAGE.into(),
            tree_sitter_swift::TAGS_QUERY,
        ),

        #[cfg(feature = "lang-scala")]
        "scala" | "sc" => make_config(
            tree_sitter_scala::LANGUAGE.into(),
            tree_sitter_scala::TAGS_QUERY,
        ),

        #[cfg(feature = "lang-zig")]
        "zig" => make_config(
            tree_sitter_zig::LANGUAGE.into(),
            tree_sitter_zig::TAGS_QUERY,
        ),

        #[cfg(feature = "lang-elixir")]
        "ex" | "exs" => make_config(
            tree_sitter_elixir::LANGUAGE.into(),
            tree_sitter_elixir::TAGS_QUERY,
        ),

        #[cfg(feature = "lang-haskell")]
        "hs" => make_config(
            tree_sitter_haskell::LANGUAGE.into(),
            tree_sitter_haskell::TAGS_QUERY,
        ),

        #[cfg(feature = "lang-lua")]
        "lua" => make_config(
            tree_sitter_lua::LANGUAGE.into(),
            tree_sitter_lua::TAGS_QUERY,
        ),

        _ => None,
    }
}

/// Helper to build a TagsConfiguration, returning None on error.
#[cfg(feature = "tree-sitter")]
fn make_config(
    language: tree_sitter::Language,
    tags_query: &str,
) -> Option<TagsConfiguration> {
    TagsConfiguration::new(language, tags_query, "").ok()
}

/// Extract symbols and references from a source file using tree-sitter.
///
/// Returns None if tree-sitter isn't available for this language,
/// in which case the caller should fall back to regex extraction.
#[cfg(feature = "tree-sitter")]
pub fn extract(file: &str, content: &str, ext: &str) -> Option<ExtractionResult> {
    let config = get_tags_config(ext)?;
    let mut ctx = TagsContext::new();

    let source = content.as_bytes();
    let (tags_iter, _has_error) = ctx.generate_tags(&config, source, None).ok()?;

    let mut result = ExtractionResult::default();

    for tag_result in tags_iter {
        let tag = match tag_result {
            Ok(t) => t,
            Err(_) => continue,
        };

        // Extract the name from the source
        let name = match std::str::from_utf8(&source[tag.name_range.clone()]) {
            Ok(n) => n.to_string(),
            Err(_) => continue,
        };

        // Skip very short names (likely noise)
        if name.len() < 2 {
            continue;
        }

        let syntax_type = config.syntax_type_name(tag.syntax_type_id).to_string();

        // Calculate line number from byte offset
        let line = content[..tag.name_range.start]
            .chars()
            .filter(|&c| c == '\n')
            .count()
            + 1;

        if tag.is_definition {
            // Extract the full line as signature
            let sig_line = content
                .lines()
                .nth(line - 1)
                .unwrap_or("")
                .trim()
                .trim_end_matches('{')
                .trim_end_matches(':')
                .trim()
                .to_string();

            result.symbols.push(Symbol {
                name: name.clone(),
                file: file.into(),
                line,
                kind: map_syntax_type(&syntax_type),
                signature: sig_line,
                end_line: 0, // filled by compute_end_lines after extraction
                deps: Vec::new(),
            });
        } else {
            // It's a reference
            result.references.push(SymbolRef {
                name,
                file: file.into(),
            });
        }
    }

    Some(result)
}

/// Fallback: tree-sitter feature not enabled.
#[cfg(not(feature = "tree-sitter"))]
pub fn extract(_file: &str, _content: &str, _ext: &str) -> Option<ExtractionResult> {
    None
}

/// Check if tree-sitter extraction is available for a given extension.
pub fn is_supported(ext: &str) -> bool {
    match ext {
        #[cfg(feature = "lang-rust")]
        "rs" => true,
        #[cfg(feature = "lang-python")]
        "py" => true,
        #[cfg(feature = "lang-javascript")]
        "js" | "jsx" | "mjs" | "cjs" => true,
        #[cfg(feature = "lang-typescript")]
        "ts" | "tsx" | "mts" => true,
        #[cfg(feature = "lang-go")]
        "go" => true,
        #[cfg(feature = "lang-java")]
        "java" => true,
        #[cfg(feature = "lang-c")]
        "c" | "h" => true,
        #[cfg(feature = "lang-cpp")]
        "cpp" | "hpp" | "cc" | "cxx" | "hh" => true,
        #[cfg(feature = "lang-ruby")]
        "rb" | "rake" | "gemspec" => true,
        #[cfg(feature = "lang-php")]
        "php" => true,
        #[cfg(feature = "lang-csharp")]
        "cs" => true,
        #[cfg(feature = "lang-kotlin")]
        "kt" | "kts" => true,
        #[cfg(feature = "lang-swift")]
        "swift" => true,
        #[cfg(feature = "lang-scala")]
        "scala" | "sc" => true,
        #[cfg(feature = "lang-zig")]
        "zig" => true,
        #[cfg(feature = "lang-elixir")]
        "ex" | "exs" => true,
        #[cfg(feature = "lang-haskell")]
        "hs" => true,
        #[cfg(feature = "lang-lua")]
        "lua" => true,
        _ => false,
    }
}

/// List all enabled language features.
pub fn enabled_languages() -> Vec<&'static str> {
    let mut langs = Vec::new();
    #[cfg(feature = "lang-rust")]
    langs.push("Rust");
    #[cfg(feature = "lang-python")]
    langs.push("Python");
    #[cfg(feature = "lang-javascript")]
    langs.push("JavaScript");
    #[cfg(feature = "lang-typescript")]
    langs.push("TypeScript");
    #[cfg(feature = "lang-go")]
    langs.push("Go");
    #[cfg(feature = "lang-java")]
    langs.push("Java");
    #[cfg(feature = "lang-c")]
    langs.push("C");
    #[cfg(feature = "lang-cpp")]
    langs.push("C++");
    #[cfg(feature = "lang-ruby")]
    langs.push("Ruby");
    #[cfg(feature = "lang-php")]
    langs.push("PHP");
    #[cfg(feature = "lang-csharp")]
    langs.push("C#");
    #[cfg(feature = "lang-kotlin")]
    langs.push("Kotlin");
    #[cfg(feature = "lang-swift")]
    langs.push("Swift");
    #[cfg(feature = "lang-scala")]
    langs.push("Scala");
    #[cfg(feature = "lang-zig")]
    langs.push("Zig");
    #[cfg(feature = "lang-elixir")]
    langs.push("Elixir");
    #[cfg(feature = "lang-haskell")]
    langs.push("Haskell");
    #[cfg(feature = "lang-lua")]
    langs.push("Lua");
    langs
}

/// Map tree-sitter syntax type names to our normalized kind names.
fn map_syntax_type(syntax_type: &str) -> String {
    match syntax_type {
        "function" | "method" | "function.method" => "function".into(),
        "class" => "class".into(),
        "module" => "module".into(),
        "struct" => "struct".into(),
        "enum" => "enum".into(),
        "interface" => "interface".into(),
        "trait" => "trait".into(),
        "type" => "type".into(),
        "constant" | "const" => "const".into(),
        "variable" => "variable".into(),
        "property" | "field" => "field".into(),
        "constructor" => "constructor".into(),
        "implementation" => "impl".into(),
        other => other.to_string(),
    }
}
