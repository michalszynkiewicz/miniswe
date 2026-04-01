//! Shared test helpers for E2E tests.
//! Not every test file uses every helper, so suppress per-crate dead_code warnings.
#![allow(unused)]

use std::fs;
use std::path::PathBuf;

use miniswe::config::Config;
use tempfile::TempDir;
use wiremock::ResponseTemplate;

/// Create a temporary project directory with `.miniswe/` initialized.
/// Returns the TempDir (must be held alive) and a Config pointed at it.
pub fn create_test_project() -> (TempDir, Config) {
    let temp = TempDir::new().expect("failed to create temp dir");
    let project_root = temp.path().to_path_buf();

    // Create .miniswe directory structure
    let miniswe_dir = project_root.join(".miniswe");
    fs::create_dir_all(&miniswe_dir).unwrap();
    fs::create_dir_all(miniswe_dir.join("index")).unwrap();

    // Write a minimal config.toml
    fs::write(
        miniswe_dir.join("config.toml"),
        r#"[model]
provider = "llama-cpp"
endpoint = "http://localhost:9999"
model = "test-model"
context_window = 50000
temperature = 0.15
max_output_tokens = 16384

[context]
max_rounds = 10
pause_after_rounds = 100

[hardware]
vram_gb = 24.0

[web]
search_backend = "serper"
fetch_backend = "jina"
"#,
    )
    .unwrap();

    // Write empty index files so tools don't error on missing index
    fs::write(miniswe_dir.join("index/symbols.json"), "{}").unwrap();
    fs::write(miniswe_dir.join("index/summaries.json"), "{}").unwrap();
    fs::write(miniswe_dir.join("index/file_tree.txt"), "").unwrap();
    fs::write(miniswe_dir.join("index/mtimes.json"), "{}").unwrap();

    // Write empty .mcp.json so McpConfig::load doesn't fail
    fs::write(project_root.join(".mcp.json"), r#"{"servers":{}}"#).unwrap();

    let mut config = Config::default();
    config.project_root = project_root;

    (temp, config)
}

/// Create a Config pointed at a mock LLM server.
pub fn config_with_mock_endpoint(config: &mut Config, mock_uri: &str) {
    config.model.endpoint = mock_uri.to_string();
    config.model.provider = "openai-compatible".to_string();
}

/// Build a mock non-streaming LLM response with plain text content.
pub fn mock_text_response(content: &str) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(serde_json::json!({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
    }))
}

/// Build a mock non-streaming LLM response with a single tool call.
pub fn mock_tool_call_response(
    tool_name: &str,
    tool_args: serde_json::Value,
) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(serde_json::json!({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_test_1",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": serde_json::to_string(&tool_args).unwrap()
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    }))
}

/// Build a mock SSE streaming response with plain text content.
pub fn mock_sse_text_response(content: &str) -> ResponseTemplate {
    let body = format!(
        "data: {}\n\ndata: [DONE]\n\n",
        serde_json::json!({
            "choices": [{
                "delta": {"content": content},
                "finish_reason": null
            }]
        })
    );
    ResponseTemplate::new(200).set_body_raw(body, "text/event-stream")
}

/// Build a mock SSE streaming response with a tool call.
pub fn mock_sse_tool_call(tool_name: &str, tool_args: &str) -> ResponseTemplate {
    let body = format!(
        "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
        serde_json::json!({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_test_1",
                        "function": {"name": tool_name, "arguments": ""}
                    }]
                },
                "finish_reason": null
            }]
        }),
        serde_json::json!({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": tool_args}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }),
    );
    ResponseTemplate::new(200).set_body_raw(body, "text/event-stream")
}

/// Build a mock SSE streaming response that returns stop with no content.
pub fn mock_sse_stop() -> ResponseTemplate {
    let body = format!(
        "data: {}\n\ndata: [DONE]\n\n",
        serde_json::json!({
            "choices": [{
                "delta": {"content": "Done."},
                "finish_reason": "stop"
            }]
        })
    );
    ResponseTemplate::new(200).set_body_raw(body, "text/event-stream")
}

/// Get the absolute path for a file within the test project.
pub fn project_path(config: &Config, relative: &str) -> PathBuf {
    config.project_root.join(relative)
}
