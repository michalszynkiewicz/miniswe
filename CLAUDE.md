* do not disable tests to work around issues. If they are wrong, suggest dropping them. If they are not, keep them
* unless otherwise specified, do not change anything before an approval from the user
* always run `cargo test`, `cargo clippy` and `cargo fmt` before committing
* never push to the repository
* do not put implementation code in `mod.rs` — use it only for type definitions and `pub use` re-exports; put logic in named submodules
