//! Drives the real miniswe REPL in a PTY against a local llama-cpp server
//! to catch the "hidden tail" rendering bug that only surfaces after
//! multiple roundtrips. Gated behind MINISWE_LIVE_MODEL=1.

#![cfg(test)]

use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use portable_pty::{CommandBuilder, PtySize, native_pty_system};

const COLS: u16 = 148;
const ROWS: u16 = 40;
const TURN_TIMEOUT: Duration = Duration::from_secs(120);

const PROMPTS: &[(&str, &str)] = &[
    ("print the word APPLE", "APPLE"),
    ("print the word BANANA", "BANANA"),
    ("print the word CHERRY", "CHERRY"),
    ("print the word DURIAN", "DURIAN"),
    ("print the word ELDERBERRY", "ELDERBERRY"),
    ("print the word FIG", "FIG"),
];

struct VtReader {
    vt: Arc<Mutex<avt::Vt>>,
}

impl VtReader {
    fn spawn(mut reader: Box<dyn std::io::Read + Send>, cols: usize, rows: usize) -> Self {
        let vt = Arc::new(Mutex::new(
            avt::Vt::builder().size(cols, rows).build(),
        ));
        let vt_clone = vt.clone();
        std::thread::spawn(move || {
            let mut buf = [0u8; 4096];
            loop {
                match reader.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        let chunk = String::from_utf8_lossy(&buf[..n]);
                        let mut vt = vt_clone.lock().unwrap();
                        vt.feed_str(&chunk);
                    }
                    Err(_) => break,
                }
            }
        });
        Self { vt }
    }

    fn screen_text(&self) -> String {
        let vt = self.vt.lock().unwrap();
        vt.view()
            .map(|line| line.text())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[test]
fn tui_hidden_output_across_multiple_turns() {
    if std::env::var("MINISWE_LIVE_MODEL").is_err() {
        eprintln!("skipped: set MINISWE_LIVE_MODEL=1 to run (needs llama-cpp on :8464)");
        return;
    }

    let temp = tempfile::tempdir().unwrap();
    write_config(temp.path());

    let pty_system = native_pty_system();
    let pair = pty_system
        .openpty(PtySize {
            rows: ROWS,
            cols: COLS,
            pixel_width: 0,
            pixel_height: 0,
        })
        .unwrap();

    let mut cmd = CommandBuilder::new(env!("CARGO_BIN_EXE_miniswe"));
    cmd.arg("-y");
    cmd.cwd(temp.path());
    cmd.env("TERM", "xterm-256color");
    cmd.env_remove("NO_COLOR");

    let mut child = pair.slave.spawn_command(cmd).unwrap();
    drop(pair.slave);

    let reader = pair.master.try_clone_reader().unwrap();
    let mut writer = pair.master.take_writer().unwrap();

    let vt_reader = VtReader::spawn(reader, COLS as usize, ROWS as usize);

    wait_for_idle(&vt_reader, TURN_TIMEOUT)
        .expect("initial banner never settled to idle");

    let mut hidden_count = 0;
    for (i, (prompt, marker)) in PROMPTS.iter().enumerate() {
        eprintln!(">>> turn {}: prompt={prompt:?}  marker={marker:?}", i + 1);

        for ch in prompt.bytes() {
            writer.write_all(&[ch]).unwrap();
            writer.flush().unwrap();
            std::thread::sleep(Duration::from_millis(5));
        }
        std::thread::sleep(Duration::from_millis(50));
        writer.write_all(b"\r").unwrap();
        writer.flush().unwrap();

        wait_for_working(&vt_reader, Duration::from_secs(5)).ok();

        wait_for_idle(&vt_reader, TURN_TIMEOUT).unwrap_or_else(|e| {
            let screen = vt_reader.screen_text();
            panic!(
                "turn {prompt:?} did not complete: {e}\n--- screen ---\n{screen}"
            )
        });

        std::thread::sleep(Duration::from_millis(300));

        let screen = vt_reader.screen_text();

        // Check that the marker appears in the TRANSCRIPT area (not just the
        // input box). The transcript contains the LLM's response text.
        // We look for the marker after a "│" prefix (transcript pane border)
        // but NOT in a "you>" line (user input).
        let marker_in_transcript = screen.lines().any(|l| {
            l.contains(marker)
                && !l.contains("you>")
                && !l.contains("print the word")
        });

        eprintln!(
            "  turn {} marker_in_transcript={marker_in_transcript}",
            i + 1
        );
        if !marker_in_transcript {
            eprintln!(
                "  *** HIDDEN OUTPUT: marker {marker:?} not in transcript after turn {}",
                i + 1
            );
            eprintln!("--- screen ({} lines) ---", screen.lines().count());
            for (li, line) in screen.lines().enumerate() {
                eprintln!("  [{li:02}] {line}");
            }
            // Dump session log immediately to capture what the LLM produced
            let log_dir = temp.path().join(".miniswe").join("logs");
            if let Ok(entries) = std::fs::read_dir(&log_dir) {
                for entry in entries.flatten() {
                    if entry.path().extension().map_or(false, |e| e == "log") {
                        let content = std::fs::read_to_string(entry.path()).unwrap_or_default();
                        eprintln!("\n=== SESSION LOG (turn {} failure) ===", i + 1);
                        for line in content.lines().rev().take(20).collect::<Vec<_>>().into_iter().rev() {
                            eprintln!("  {line}");
                        }
                    }
                }
            }
            hidden_count += 1;
        }
    }

    // Check session log to confirm LLM actually produced the text
    let log_dir = temp.path().join(".miniswe").join("logs");
    if let Ok(entries) = std::fs::read_dir(&log_dir) {
        for entry in entries.flatten() {
            if entry.path().extension().map_or(false, |e| e == "log") {
                let content = std::fs::read_to_string(entry.path()).unwrap_or_default();
                eprintln!("\n=== SESSION LOG ({}) ===", entry.path().display());
                for line in content.lines() {
                    if line.contains("llm_response") || line.contains("APPLE")
                        || line.contains("BANANA") || line.contains("CHERRY")
                        || line.contains("DURIAN") || line.contains("ELDERBERRY")
                        || line.contains("FIG")
                    {
                        eprintln!("  {line}");
                    }
                }
            }
        }
    }

    let _ = writer.write_all(b"quit\r");
    drop(writer);
    let _ = child.wait();

    assert_eq!(
        hidden_count, 0,
        "{hidden_count} turn(s) had hidden output — the TUI bug is reproduced"
    );
}

fn write_config(root: &std::path::Path) {
    let miniswe = root.join(".miniswe");
    std::fs::create_dir_all(&miniswe).unwrap();
    std::fs::write(
        miniswe.join("config.toml"),
        r#"
[model]
provider = "llama-cpp"
endpoint = "http://localhost:8464"
model = "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
context_window = 50000
temperature = 0.0
max_output_tokens = 1024
request_timeout_secs = 120
stream_idle_timeout_secs = 30
max_retries = 2

[context]
max_rounds = 10
pause_after_rounds = 100

[context.providers]
profile = false
guide = false
project_notes = false
plan = false
lessons = false
repo_map = false
mcp = false
scratchpad = false
usage_guide = false
plan_mode = false

[hardware]
vram_gb = 24.0

[web]
search_backend = "serper"
fetch_backend = "jina"

[tools]
edit_mode = "fast"
web_tools = false
plan = false
context_tools = false
lsp_tools = false

[lsp]
enabled = false

[logging]
level = "debug"
enabled = true
"#,
    )
    .unwrap();
    std::fs::write(root.join(".mcp.json"), r#"{"servers":{}}"#).unwrap();
}

fn wait_for_working(vt_reader: &VtReader, timeout: Duration) -> Result<(), String> {
    let deadline = Instant::now() + timeout;
    loop {
        if Instant::now() >= deadline {
            return Err("timeout waiting for working state".into());
        }
        std::thread::sleep(Duration::from_millis(100));
        let screen = vt_reader.screen_text();
        if screen.lines().take(3).any(|l| l.contains("working")) {
            return Ok(());
        }
    }
}

fn wait_for_idle(vt_reader: &VtReader, timeout: Duration) -> Result<(), String> {
    let deadline = Instant::now() + timeout;
    let mut idle_since: Option<Instant> = None;

    loop {
        if Instant::now() >= deadline {
            let screen = vt_reader.screen_text();
            return Err(format!("timeout\n--- screen ---\n{screen}"));
        }

        std::thread::sleep(Duration::from_millis(200));
        let screen = vt_reader.screen_text();

        let bottom_has_prompt = screen
            .lines()
            .rev()
            .take(3)
            .any(|l| l.contains("you>"));
        let topbar_ready = screen
            .lines()
            .take(3)
            .any(|l| l.contains(" ready "));

        if bottom_has_prompt && topbar_ready {
            if idle_since.is_none() {
                idle_since = Some(Instant::now());
            }
            if let Some(since) = idle_since {
                if since.elapsed() >= Duration::from_millis(500) {
                    return Ok(());
                }
            }
        } else {
            idle_since = None;
        }
    }
}
