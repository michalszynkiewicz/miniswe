# Workspace Awareness: Generalized Multi-Module Project Understanding

## Problem

Most non-trivial projects are composed of multiple modules/packages/crates with internal dependency relationships. The agent needs to know the dependency order between modules to edit them correctly (always edit dependencies before dependents, understand which module owns which types). Currently miniswe treats all files as a flat set.

## Core Abstraction

A **workspace** is a project containing multiple **modules**, where each module:
- Has a name and a root directory
- Declares dependencies on other modules in the same workspace
- May have its own build/test commands

This maps to concrete build systems:

| Ecosystem | Workspace declaration | Module unit | Dependency declaration |
|---|---|---|---|
| Rust/Cargo | `[workspace]` in root `Cargo.toml` | Crate (`Cargo.toml` with `[package]`) | `[dependencies]` with `path = "../sibling"` |
| Java/Gradle | `settings.gradle(.kts)` `include` directives | Subproject directory | `implementation(project(":sibling"))` |
| Java/Maven | `<modules>` in parent `pom.xml` | Child `pom.xml` with `<parent>` | `<dependency>` with same `groupId` + relative path |
| Go | `go.work` file | `go.mod` directory | `use ./sibling` in `go.work`, `replace` directives |
| TypeScript/JS | `workspaces` in root `package.json` (npm/yarn/pnpm) | Package directory with own `package.json` | `"@scope/sibling": "workspace:*"` |
| Python | `pyproject.toml` with hatch/PDM workspaces, or monorepo with multiple `pyproject.toml` | Directory with `pyproject.toml` or `setup.py` | Path dependencies, `{path = "../sibling"}` in deps |
| C#/.NET | `*.sln` solution file | `*.csproj` project file | `<ProjectReference Include="../Sibling/Sibling.csproj">` |
| Kotlin/Gradle | Same as Java/Gradle | Same as Java/Gradle | Same as Java/Gradle |

## Data Model

```rust
struct WorkspaceInfo {
    /// Detected build system (cargo, gradle, maven, go, npm, etc.)
    build_system: String,
    /// Ordered list of modules (topologically sorted by dependency)
    modules: Vec<ModuleInfo>,
}

struct ModuleInfo {
    /// Module name (crate name, gradle project name, package name)
    name: String,
    /// Root directory relative to project root
    path: String,
    /// Names of other workspace modules this depends on
    internal_deps: Vec<String>,
    /// Build command for this module alone (if available)
    build_cmd: Option<String>,
    /// Test command for this module alone (if available)
    test_cmd: Option<String>,
}
```

## Detection Logic

On `miniswe init`, probe in order (first match wins):

1. **Cargo**: root `Cargo.toml` contains `[workspace]` → parse `members` globs, resolve to directories, read each member's `Cargo.toml` for path dependencies
2. **Go**: `go.work` exists → parse `use` directives, read each module's `go.mod` for `replace` directives pointing to siblings
3. **Gradle**: `settings.gradle` or `settings.gradle.kts` exists → parse `include` statements, read each `build.gradle(.kts)` for `project(":")` dependencies
4. **Maven**: root `pom.xml` contains `<modules>` → parse module list, read each child `pom.xml` for intra-project `<dependency>` entries (match by groupId)
5. **npm/yarn/pnpm**: root `package.json` contains `"workspaces"` → resolve globs, read each package's `package.json` for `workspace:` protocol deps
6. **.NET**: `*.sln` exists → parse for `Project(` entries, read each `.csproj` for `<ProjectReference>` paths
7. **Python**: root `pyproject.toml` with hatch/PDM workspace config, or multiple `pyproject.toml` files with path dependencies between them

Each detector is a small function: `fn detect_X(root: &Path) -> Option<WorkspaceInfo>`. Easy to add new ones.

## Profile Integration

Append a `## Workspace` section to `.miniswe/profile.md`:

```markdown
## Workspace (cargo, 4 crates)
core/ → (no internal deps)
routing/ → core
middleware/ → core, routing
server/ → core, routing, middleware

Build order: core → routing → middleware → server
```

This costs ~50-80 tokens and gives the model the dependency DAG at a glance.

## Repo Map Integration

Boost PageRank scores for modules relevant to the current task. If the user says "add middleware support to the router", boost both `routing/` and `middleware/` files. The dependency info also lets us boost `core/` (since both depend on it — changes there might be needed).

In `graph.rs` `personalized_scores`: when a keyword matches a module name, also boost files in that module's `internal_deps` (transitive dependencies are important context).

## Diagnostics Integration

When auto-running compiler checks after a file edit, scope the check to the module that was edited rather than the whole workspace. For Cargo: `cargo check -p <crate_name>` instead of `cargo check`. This is faster and produces more focused error output.

## Implementation Priority

1. **Data model + Cargo detector** — covers the immediate use case (Rust REST framework), ~100 lines
2. **Profile integration** — append to existing profile generation, ~30 lines
3. **Go detector** — `go.work` is simple to parse, ~50 lines
4. **Gradle/Maven detector** — covers Java/Kotlin, ~80 lines each
5. **npm detector** — covers TS/JS monorepos, ~60 lines
6. **Scoped diagnostics** — use module info to run targeted checks, ~20 lines
7. **PageRank boost** — wire into dependency graph, ~30 lines
