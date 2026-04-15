//! Context compression pipeline.
//!
//! Five layers of deterministic, lossless-for-code-semantics compression:
//! 1. Code format stripping (`format`) — remove comments, collapse whitespace
//! 2. Structured profile format (`profile`) — key-value notation for system context
//! 3. Import elision (`imports`) — drop standard library imports
//! 4. Line-preserving reading compression (`reading`) — for the `read_file` tool
//! 5. Tool-result summarization (`tool_result`) — observation masking
//!
//! Total effective multiplier: ~1.6× — a 64K window carries ~100K of information.

mod format;
mod imports;
mod profile;
mod reading;
mod tool_result;

pub use format::strip_code_format;
pub use imports::elide_std_imports;
pub use profile::compress_profile;
pub use reading::compress_for_reading;
pub use tool_result::summarize_tool_result;
