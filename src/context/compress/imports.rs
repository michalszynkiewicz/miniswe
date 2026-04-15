//! Elide standard-library imports. Used by the compressor when building
//! a dense view of a file for the LLM — the model already knows what
//! `std::collections::HashMap` is.

/// Elide standard library imports (the model knows them).
/// Only keep project-internal and third-party imports.
pub fn elide_std_imports(code: &str, language: &str) -> String {
    let mut result = String::new();

    for line in code.lines() {
        let trimmed = line.trim();

        let should_skip = match language {
            "rs" => {
                trimmed.starts_with("use std::")
                    || trimmed.starts_with("use core::")
                    || trimmed.starts_with("use alloc::")
            }
            "py" => {
                let is_import = trimmed.starts_with("import ") || trimmed.starts_with("from ");
                if is_import {
                    // Skip standard library imports
                    let stdlib = [
                        "os",
                        "sys",
                        "re",
                        "json",
                        "math",
                        "datetime",
                        "collections",
                        "itertools",
                        "functools",
                        "typing",
                        "pathlib",
                        "io",
                        "abc",
                        "dataclasses",
                        "enum",
                        "copy",
                        "hashlib",
                        "logging",
                        "unittest",
                        "argparse",
                        "subprocess",
                        "threading",
                        "time",
                        "random",
                    ];
                    stdlib.iter().any(|lib| {
                        trimmed.starts_with(&format!("import {lib}"))
                            || trimmed.starts_with(&format!("from {lib}"))
                    })
                } else {
                    false
                }
            }
            "go" => false, // Go imports are essential for understanding code
            "js" | "ts" | "tsx" | "jsx" => {
                // Skip node builtins
                let builtins = [
                    "\"fs\"",
                    "\"path\"",
                    "\"os\"",
                    "\"url\"",
                    "\"http\"",
                    "\"https\"",
                    "\"crypto\"",
                    "\"util\"",
                    "\"events\"",
                    "'fs'",
                    "'path'",
                    "'os'",
                    "'url'",
                    "'http'",
                ];
                let is_import = trimmed.starts_with("import ") || trimmed.starts_with("const ");
                is_import && builtins.iter().any(|b| trimmed.contains(b))
            }
            _ => false,
        };

        if !should_skip {
            result.push_str(line);
            result.push('\n');
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elide_rust_std_imports() {
        let code =
            "use std::collections::HashMap;\nuse crate::config::Config;\nuse anyhow::Result;\n";
        let elided = elide_std_imports(code, "rs");
        assert!(!elided.contains("std::collections"));
        assert!(elided.contains("crate::config"));
        assert!(elided.contains("anyhow::Result"));
    }
}
