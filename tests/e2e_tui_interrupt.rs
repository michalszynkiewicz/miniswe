//! Ctrl+C in the real REPL, driven through a PTY against a fake LLM server.
//!
//! The cancel flag is armed by the key reader thread and read deep inside the
//! turn, so only the whole binary shows whether the two agree.

mod helpers;

use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use portable_pty::{Child, CommandBuilder, MasterPty, PtySize, native_pty_system};
use tempfile::TempDir;
use wiremock::matchers::{body_string_contains, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const COLS: u16 = 120;
const ROWS: u16 = 40;
const WAIT: Duration = Duration::from_secs(30);
const CTRL_C: &[u8] = b"\x03";

struct Repl {
    vt: Arc<Mutex<avt::Vt>>,
    writer: Box<dyn Write + Send>,
    child: Box<dyn Child + Send + Sync>,
    project: TempDir,
    _master: Box<dyn MasterPty + Send>,
}

impl Drop for Repl {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

impl Repl {
    fn start(endpoint: &str) -> Self {
        let project = tempfile::tempdir().unwrap();
        write_config(&project, endpoint);

        let pair = native_pty_system()
            .openpty(PtySize {
                rows: ROWS,
                cols: COLS,
                pixel_width: 0,
                pixel_height: 0,
            })
            .unwrap();
        let mut cmd = CommandBuilder::new(env!("CARGO_BIN_EXE_buzztr"));
        cmd.arg("-y");
        cmd.cwd(project.path());
        cmd.env("TERM", "xterm-256color");
        let child = pair.slave.spawn_command(cmd).unwrap();
        drop(pair.slave);

        let vt = Arc::new(Mutex::new(
            avt::Vt::builder()
                .size(COLS as usize, ROWS as usize)
                .build(),
        ));
        let mut reader = pair.master.try_clone_reader().unwrap();
        let feed = vt.clone();
        std::thread::spawn(move || {
            let mut buf = [0u8; 4096];
            while let Ok(n) = reader.read(&mut buf) {
                if n == 0 {
                    break;
                }
                feed.lock()
                    .unwrap()
                    .feed_str(&String::from_utf8_lossy(&buf[..n]));
            }
        });
        let writer = pair.master.take_writer().unwrap();

        Self {
            vt,
            writer,
            child,
            project,
            _master: pair.master,
        }
    }

    fn screen(&self) -> String {
        let vt = self.vt.lock().unwrap();
        vt.view()
            .map(|line| line.text())
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn send(&mut self, bytes: &[u8]) {
        self.writer.write_all(bytes).unwrap();
        self.writer.flush().unwrap();
    }

    fn submit(&mut self, text: &str) {
        self.send(text.as_bytes());
        std::thread::sleep(Duration::from_millis(100));
        self.send(b"\r");
    }

    fn wait_for(&self, what: &str, done: impl Fn(&str) -> bool) {
        let deadline = Instant::now() + WAIT;
        loop {
            let screen = self.screen();
            if done(&screen) {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "timed out waiting for {what}\n--- screen ---\n{screen}\n--- log ---\n{}",
                self.log()
            );
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    fn wait_ready(&self) {
        self.wait_for("the idle prompt", |screen| {
            screen.lines().next().is_some_and(|l| l.contains(" ready "))
                && screen.lines().rev().take(3).any(|l| l.contains("you>"))
        });
    }

    fn wait_working(&self) {
        self.wait_for("a running turn", |screen| {
            screen.lines().next().is_some_and(|l| l.contains("working"))
        });
    }

    fn log(&self) -> String {
        let Ok(entries) = std::fs::read_dir(self.project.path().join(".buzztr/logs")) else {
            return String::new();
        };
        entries
            .flatten()
            .filter_map(|entry| std::fs::read_to_string(entry.path()).ok())
            .collect()
    }
}

fn write_config(project: &TempDir, endpoint: &str) {
    let dir = project.path().join(".buzztr");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("config.toml"),
        format!(
            r#"
[model]
provider = "llama-cpp"
endpoint = "{endpoint}"
model = "fake.gguf"
context_window = 50000
max_retries = 1

[context]
max_rounds = 3

[tools]
ceremony = "off"
plan = false
context_tools = false
lsp_tools = false
web_tools = false

[lsp]
enabled = false

[logging]
level = "debug"
enabled = true
"#
        ),
    )
    .unwrap();
    std::fs::write(project.path().join(".mcp.json"), r#"{"servers":{}}"#).unwrap();
}

async fn fake_llm(response: ResponseTemplate) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(response)
        .mount(&server)
        .await;
    server
}

/// F18: the key reader arms the flag on every Ctrl+C, even when no turn is
/// running to consume it, and the next message used to end at once.
#[tokio::test(flavor = "multi_thread")]
async fn ctrl_c_at_the_idle_prompt_does_not_kill_the_next_turn() {
    let server = fake_llm(helpers::mock_sse_text_response("PONG")).await;
    let mut repl = Repl::start(&server.uri());
    repl.wait_ready();

    repl.send(CTRL_C);
    // Longer than the key reader's 80 ms poll, so the flag is armed before the
    // message arrives.
    std::thread::sleep(Duration::from_millis(300));

    repl.submit("say pong");
    repl.wait_for("the reply", |screen| {
        screen
            .lines()
            .any(|l| l.contains("PONG") && !l.contains("you>"))
    });
    repl.wait_ready();

    assert!(
        !repl.screen().contains("(interrupted)"),
        "{}",
        repl.screen()
    );
    let log = repl.log();
    assert!(!log.contains("status=interrupted"), "{log}");
}

/// The other side of F18: disarming the flag must not swallow a real interrupt
/// that lands while the pre-turn routers are waiting on the server.
#[tokio::test(flavor = "multi_thread")]
async fn ctrl_c_during_routing_still_interrupts_the_turn() {
    let stalled = helpers::mock_sse_text_response("PONG").set_delay(Duration::from_secs(120));
    let server = fake_llm(stalled).await;
    let mut repl = Repl::start(&server.uri());
    repl.wait_ready();

    repl.submit("say pong");
    repl.wait_working();
    // Let the router's request reach the stalled server.
    std::thread::sleep(Duration::from_millis(300));
    repl.send(CTRL_C);

    repl.wait_ready();
    assert!(repl.screen().contains("(interrupted)"), "{}", repl.screen());
    let log = repl.log();
    assert!(log.contains("status=interrupted"), "{log}");
}

/// A prompt interrupted before the model answered used to stay in the history
/// with no reply, and the next message was merged into it, so the model
/// answered both.
#[tokio::test(flavor = "multi_thread")]
async fn an_interrupted_prompt_is_not_sent_with_the_next_one() {
    let server = fake_llm(helpers::mock_sse_text_response("PONG")).await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_string_contains("CANCELLED-ASK"))
        // Only the agent call carries tools; the routers answer at once, so the
        // interrupt lands on the main request.
        .and(body_string_contains("\"tools\""))
        .respond_with(helpers::mock_sse_text_response("ESSAY").set_delay(Duration::from_secs(120)))
        .with_priority(1)
        .mount(&server)
        .await;
    let mut repl = Repl::start(&server.uri());
    repl.wait_ready();

    repl.submit("write CANCELLED-ASK");
    repl.wait_working();
    std::thread::sleep(Duration::from_millis(300));
    repl.send(CTRL_C);
    repl.wait_ready();

    repl.submit("say pong");
    repl.wait_for("the reply", |screen| {
        screen
            .lines()
            .any(|l| l.contains("PONG") && !l.contains("you>"))
    });
    repl.wait_ready();

    let requests = server.received_requests().await.unwrap();
    let last = String::from_utf8_lossy(&requests.last().unwrap().body).into_owned();
    assert!(last.contains("say pong"), "{last}");
    assert!(!last.contains("CANCELLED-ASK"), "{last}");
    assert!(
        repl.log().contains("[interrupt] history withdrawn"),
        "{}",
        repl.log()
    );
}
