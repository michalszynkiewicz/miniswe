# Lessons

## Rust
- Return Result<T, E> in library code, never unwrap
- Prefer impl Trait over dyn Trait in function arguments
- Use thiserror for library error types, anyhow for applications
- When borrow checker rejects code: try Clone, Arc, or restructure ownership before adding lifetimes
- Read compiler errors carefully — they usually contain the fix
- Keep files under 200 lines; split into modules early
- For async traits: use async-trait crate or return Pin<Box<dyn Future>>
- When writing trait impls: read the trait definition first to understand required methods

