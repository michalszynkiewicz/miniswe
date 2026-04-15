; Vendored from tree-sitter-scala 0.25.0 queries/tags.scm.
; MIT License, Copyright (c) 2018 Max Brunsfeld and GitHub.
; Source: https://github.com/tree-sitter/tree-sitter-scala
;
; This file is vendored because tree-sitter-scala 0.25.0 ships tags.scm
; in the crate but does not expose it as a `pub const TAGS_QUERY: &str`.
; When a future release adds that constant, replace
;   include_str!("queries/scala_tags.scm")
; in src/knowledge/ts_extract.rs with
;   tree_sitter_scala::TAGS_QUERY
; and delete this file.

; Definitions

(package_clause
  name: (package_identifier) @name) @definition.module

(trait_definition
  name: (identifier) @name) @definition.interface

(enum_definition
  name: (identifier) @name) @definition.enum

(simple_enum_case
  name: (identifier) @name) @definition.class

(full_enum_case
  name: (identifier) @name) @definition.class

(class_definition
  name: (identifier) @name) @definition.class

(object_definition
  name: (identifier) @name) @definition.object

(function_definition
  name: (identifier) @name) @definition.function

(val_definition
  pattern: (identifier) @name) @definition.variable

(given_definition
  name: (identifier) @name) @definition.variable

(var_definition
  pattern: (identifier) @name) @definition.variable

(val_declaration
  name: (identifier) @name) @definition.variable

(var_declaration
  name: (identifier) @name) @definition.variable

(type_definition
  name: (type_identifier) @name) @definition.type

(class_parameter
  name: (identifier) @name) @definition.property

; References

(call_expression
  (identifier) @name) @reference.call

(instance_expression
  (type_identifier) @name) @reference.interface

(instance_expression
  (generic_type
    (type_identifier) @name)) @reference.interface

(extends_clause
  (type_identifier) @name) @reference.class

(extends_clause
  (generic_type
    (type_identifier) @name)) @reference.class
