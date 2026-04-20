//! Regression test for the BufReader-per-request leak in `mcp::client`.
//!
//! Uses a small Python script as a mock MCP server that deliberately
//! writes response 1 + a *partial* prefix of response 2 in one stdout
//! flush, then completes response 2 after request 2 arrives.
//!
//! Before the fix, `request()` built a fresh `BufReader` on every call.
//! The partial tail of response 2 that got buffered together with
//! response 1 was dropped when the first `request()` returned, so the
//! second `request()` would wait forever for bytes the OS had already
//! delivered.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use miniswe::mcp::client::McpClient;
use miniswe::mcp::config::McpServerConfig;

const MOCK_SERVER_SCRIPT: &str = r#"
import sys, json

def recv():
    line = sys.stdin.readline()
    return json.loads(line) if line else None

def send_line(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

# initialize handshake
req = recv()
send_line({"jsonrpc": "2.0", "id": req["id"], "result": {}})

# notifications/initialized — no response
recv()

# first tools/list: write response 1 + the opening of response 2
# in ONE flush, stopping before the id so the buffered tail can only
# be parsed after req2 is received.
req1 = recv()
resp1 = json.dumps({"jsonrpc": "2.0", "id": req1["id"], "result": {"tools": []}})
sys.stdout.write(resp1 + "\n" + '{"jsonrpc":"2.0","id":')
sys.stdout.flush()

# second tools/list: finish the previously-started response.
req2 = recv()
sys.stdout.write(str(req2["id"]) + ',"result":{"tools":[]}}' + "\n")
sys.stdout.flush()
"#;

#[test]
fn bufreader_persists_between_requests() {
    let python_available = std::process::Command::new("python3")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !python_available {
        eprintln!("python3 not available — skipping");
        return;
    }

    let dir = tempfile::tempdir().unwrap();
    let script_path = dir.path().join("mock_mcp.py");
    std::fs::write(&script_path, MOCK_SERVER_SCRIPT).unwrap();

    let cfg = McpServerConfig {
        command: "python3".to_string(),
        args: vec![script_path.to_string_lossy().into_owned()],
        env: HashMap::new(),
        timeout: 60_000,
    };

    let start = Instant::now();
    let mut client = McpClient::connect("mock", &cfg).expect("connect failed");
    client.list_tools().expect("first list_tools failed");
    client.list_tools().expect("second list_tools failed");
    let elapsed = start.elapsed();

    assert!(
        elapsed < Duration::from_secs(5),
        "two list_tools calls took {elapsed:?} — likely hung on dropped-BufReader bytes"
    );
}
