#!/usr/bin/env python3
"""
Simple-tier I/O modality probe.

Decision under test: for an aider-shaped "simple tier", is
  A) pure text, zero tools  (model drives reads + edits entirely in prose)
vs
  B) minimal flat 3-tool surface (read/search/shell as flat OpenAI tools;
     edits still via text SEARCH/REPLACE)
provably better on Gemma-4-26B-A4B for a multi-file value-threading task
(the exact shape of miniswe's documented 5-6/6 ceiling)?

Oracle: real `cargo check` + behavioral run of the produced binary.
"""
import json, os, re, shutil, subprocess, sys, time, urllib.request

ENDPOINT = "http://localhost:8464/v1/chat/completions"
WORK = "/tmp/simple-tier-probe"
TRIALS = int(os.environ.get("TRIALS", "8"))
MAX_ROUNDS = int(os.environ.get("MAX_ROUNDS", "8"))
TEMP = 0.2
MAX_TOKENS = 2048

TASK = (
    "Add a boolean CLI flag `--shout`. When `--shout` is passed, the program's "
    "greeting must be UPPERCASE. The flag's value must be threaded from argument "
    "parsing in main.rs into the `greet` function in lib.rs (add a parameter to "
    "`greet`; do not uppercase in main). Without `--shout`, behavior is unchanged."
)

# --- tiny representative crate -------------------------------------------------
CRATE = {
    "Cargo.toml": (
        "[package]\nname = \"greeter\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n"
        "[[bin]]\nname = \"greeter\"\npath = \"src/main.rs\"\n"
    ),
    "src/lib.rs": (
        "/// Build a greeting for `name`.\n"
        "pub fn greet(name: &str) -> String {\n"
        "    format!(\"Hello, {}!\", name)\n"
        "}\n"
    ),
    "src/main.rs": (
        "use greeter::greet;\n\n"
        "fn main() {\n"
        "    let args: Vec<String> = std::env::args().collect();\n"
        "    let name = args.get(1).cloned().unwrap_or_else(|| \"world\".to_string());\n"
        "    println!(\"{}\", greet(&name));\n"
        "}\n"
    ),
}

REPO_MAP = (
    "[REPO MAP]\n"
    "Cargo.toml\n"
    "src/lib.rs:\n"
    "  pub fn greet(name: &str) -> String\n"
    "src/main.rs:\n"
    "  fn main()  // parses args, calls greet, prints\n"
)

# --- prompts ------------------------------------------------------------------
EDIT_FORMAT = (
    "To edit a file, output one or more *edit blocks* in this EXACT format, "
    "each in its own fenced code block:\n"
    "```\n"
    "<relative/path>\n"
    "<<<<<<< SEARCH\n"
    "<exact lines currently in the file>\n"
    "=======\n"
    "<replacement lines>\n"
    ">>>>>>> REPLACE\n"
    "```\n"
    "The SEARCH text must match the current file content EXACTLY (whitespace included). "
    "Keep SEARCH minimal but unique. To create a new file, use an empty SEARCH section. "
    "When the task is fully done and the project compiles, output the single token DONE."
)

SYS_A = (
    "You are a coding agent working on a small Rust crate. You CANNOT call tools. "
    "Drive everything with plain text using this protocol.\n\n"
    "To read a file before editing it, output exactly one line:\n"
    "READ: <relative/path>\n"
    "(you may request several READ lines at once; you will get the contents back).\n\n"
    + EDIT_FORMAT
    + "\n\nAfter each batch of edits the project is compiled with `cargo check` and the "
    "result is fed back to you. Fix any errors with more edit blocks. Work in small steps."
)

SYS_B = (
    "You are a coding agent working on a small Rust crate. You have three tools: "
    "read_file, search, shell. Use them to inspect the project. "
    "You do NOT have an edit tool — apply code changes by emitting text edit blocks.\n\n"
    + EDIT_FORMAT
    + "\n\nAfter each batch of edits the project is compiled with `cargo check` and the "
    "result is fed back to you. Fix any errors with more edit blocks. Work in small steps."
)

TOOLS_B = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file's full contents.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "relative path"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "search", "description": "Search the crate for literal text.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "shell", "description": "Run a shell command in the crate root.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}}, "required": ["command"]}}},
]


def scaffold(d):
    if os.path.isdir(d):
        shutil.rmtree(d)
    for rel, content in CRATE.items():
        p = os.path.join(d, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(content)


def sanitize_messages(msgs):
    """Port of miniswe's context::sanitize_messages — strict chat-template
    hygiene for Mistral/Devstral. Returns a NEW list (wire-only; the
    harness keeps its own raw conversation state). Rules: single leading
    system; merge consecutive same-role (user/assistant); drop empty
    assistant msgs with no tool_calls; bridge tool->user with an
    assistant 'Understood.'; drop orphan tool msgs."""
    out = []
    seen_system = False
    for m in msgs:
        m = dict(m)
        role = m.get("role")
        if role == "system":
            if seen_system:
                continue
            seen_system = True
            out.append(m); continue
        # drop empty assistant with no tool_calls (breaks alternation)
        if role == "assistant" and not (m.get("content") or "").strip() \
                and not m.get("tool_calls"):
            continue
        if out:
            prev = out[-1]
            pr = prev.get("role")
            # merge consecutive same-role user/assistant
            if role == pr and role in ("user", "assistant"):
                prev["content"] = ((prev.get("content") or "")
                                   + "\n" + (m.get("content") or "")).strip()
                if m.get("tool_calls") and not prev.get("tool_calls"):
                    prev["tool_calls"] = m["tool_calls"]
                continue
            # tool result must follow an assistant that had tool_calls
            if role == "tool" and not (pr == "assistant" and prev.get("tool_calls")) \
                    and pr != "tool":
                m = {"role": "user", "content": m.get("content", "")}
            # tool -> user needs an assistant bridge
            if role == "user" and pr == "tool":
                out.append({"role": "assistant", "content": "Understood."})
        out.append(m)
    return out


def llm(messages, tools=None):
    body = {
        "model": "local", "messages": sanitize_messages(messages),
        "temperature": TEMP,
        "max_tokens": MAX_TOKENS,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(
        ENDPOINT, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r)
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2)


def strip_think(t):
    return re.sub(r"<think>.*?</think>", "", t or "", flags=re.S).strip()


EDIT_RE = re.compile(
    r"([^\n`]+?)\s*\n<{5,7}\s*SEARCH\s*\n(.*?)\n?={5,7}\s*\n(.*?)\n?>{5,7}\s*REPLACE",
    re.S)


def apply_edits(d, text):
    """Returns (n_applied, [errors]). Deterministic exact-match apply."""
    applied, errors = 0, []
    for m in EDIT_RE.finditer(text):
        path = m.group(1).strip().strip("`").strip()
        search = m.group(2)
        replace = m.group(3)
        fp = os.path.join(d, path)
        if search.strip() == "":
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            with open(fp, "w") as f:
                f.write(replace if replace.endswith("\n") else replace + "\n")
            applied += 1
            continue
        if not os.path.isfile(fp):
            errors.append(f"{path}: no such file for SEARCH")
            continue
        cur = open(fp).read()
        if search in cur:
            with open(fp, "w") as f:
                f.write(cur.replace(search, replace, 1))
            applied += 1
        else:
            # try whitespace-flexible match on stripped lines
            norm = lambda s: "\n".join(l.rstrip() for l in s.strip("\n").split("\n"))
            cn, sn = norm(cur), norm(search)
            if sn in cn:
                with open(fp, "w") as f:
                    f.write(cn.replace(sn, norm(replace), 1) + "\n")
                applied += 1
            else:
                errors.append(f"{path}: SEARCH block did not match current content")
    return applied, errors


def cargo_check(d):
    p = subprocess.run(["cargo", "check", "--quiet"], cwd=d,
                        capture_output=True, text=True, timeout=120)
    return p.returncode == 0, (p.stderr or "")[-1500:]


def behaves(d):
    """Build once, run with and without --shout, verify thread-through."""
    b = subprocess.run(["cargo", "build", "--quiet"], cwd=d,
                        capture_output=True, text=True, timeout=180)
    if b.returncode != 0:
        return False
    binp = os.path.join(d, "target/debug/greeter")
    try:
        plain = subprocess.run([binp, "Ada"], capture_output=True, text=True, timeout=10).stdout
        shout = subprocess.run([binp, "Ada", "--shout"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return False
    return ("Ada" in plain and plain.strip() != plain.strip().upper()
            and "ADA" in shout.upper() and shout.strip() == shout.strip().upper())


def read_file_safe(d, path):
    fp = os.path.join(d, path.strip())
    if os.path.isfile(fp):
        return open(fp).read()
    return f"(no such file: {path})"


def run_trial(arm, idx):
    d = os.path.join(WORK, f"{arm}_{idx}")
    scaffold(d)
    sys_p = SYS_A if arm == "A" else SYS_B
    tools = None if arm == "A" else TOOLS_B
    messages = [
        {"role": "system", "content": sys_p},
        {"role": "user", "content": f"TASK: {TASK}\n\n{REPO_MAP}\n\nBegin."},
    ]
    fmt_violations = 0
    log = []
    for rnd in range(1, MAX_ROUNDS + 1):
        try:
            resp = llm(messages, tools)
        except Exception as e:
            log.append(f"r{rnd} LLM ERROR {e}")
            break
        msg = resp["choices"][0]["message"]
        content = strip_think(msg.get("content") or "")
        tcs = msg.get("tool_calls") or []
        messages.append({"role": "assistant",
                         "content": msg.get("content") or "",
                         **({"tool_calls": tcs} if tcs else {})})

        progressed = False

        # Arm B: handle tool calls (reads/search/shell)
        if tcs:
            for tc in tcs:
                fn = tc["function"]["name"]
                try:
                    a = json.loads(tc["function"]["arguments"] or "{}")
                except Exception:
                    a = {}
                if fn == "read_file":
                    out = read_file_safe(d, a.get("path", ""))
                elif fn == "search":
                    q = a.get("query", "")
                    hits = subprocess.run(["grep", "-rn", q, "src"], cwd=d,
                                          capture_output=True, text=True).stdout
                    out = hits or "(no matches)"
                elif fn == "shell":
                    pr = subprocess.run(a.get("command", "true"), cwd=d, shell=True,
                                        capture_output=True, text=True, timeout=120)
                    out = (pr.stdout + pr.stderr)[-1500:]
                else:
                    out = f"(unknown tool {fn})"
                    fmt_violations += 1
                messages.append({"role": "tool", "tool_call_id": tc.get("id", "x"),
                                 "content": out[:4000]})
                progressed = True
            log.append(f"r{rnd} tool_calls={[t['function']['name'] for t in tcs]}")

        # Arm A: parse READ: directives from prose
        if arm == "A" and not tcs:
            reads = re.findall(r"^\s*READ:\s*(.+)$", content, re.M)
            if reads:
                fb = "\n\n".join(
                    f"=== {r.strip()} ===\n{read_file_safe(d, r)}" for r in reads)
                messages.append({"role": "user", "content": fb})
                progressed = True
                log.append(f"r{rnd} READ {reads}")

        # Edits (both arms): SEARCH/REPLACE blocks in content
        if "SEARCH" in content and "REPLACE" in content:
            n, errs = apply_edits(d, content)
            if n == 0 and errs:
                fmt_violations += 1
            ok, cargo_out = cargo_check(d) if n else (False, "no edits applied")
            fb = f"Applied {n} edit(s)."
            if errs:
                fb += " ERRORS: " + "; ".join(errs)
            fb += f"\n\ncargo check: {'PASS' if ok else 'FAIL'}\n{'' if ok else cargo_out}"
            messages.append({"role": "user", "content": fb})
            progressed = True
            log.append(f"r{rnd} edits={n} errs={len(errs)} cargo={'OK' if ok else 'FAIL'}")
            if ok and behaves(d):
                log.append(f"r{rnd} SUCCESS")
                return {"arm": arm, "idx": idx, "success": True, "rounds": rnd,
                        "fmt_violations": fmt_violations, "log": log}

        if "DONE" in content and not progressed:
            ok, _ = cargo_check(d)
            success = ok and behaves(d)
            log.append(f"r{rnd} DONE success={success}")
            return {"arm": arm, "idx": idx, "success": success, "rounds": rnd,
                    "fmt_violations": fmt_violations, "log": log}

        if not progressed:
            # model produced neither a recognized action nor edits
            fmt_violations += 1
            messages.append({"role": "user", "content":
                "No actionable READ/tool call or edit block detected. "
                "Follow the protocol exactly: request files, then emit edit blocks."})
            log.append(f"r{rnd} NO-ACTION (fmt violation)")

    return {"arm": arm, "idx": idx, "success": False, "rounds": MAX_ROUNDS,
            "fmt_violations": fmt_violations, "log": log}


def main():
    os.makedirs(WORK, exist_ok=True)
    results = []
    for arm in ("A", "B"):
        for i in range(TRIALS):
            t0 = time.time()
            r = run_trial(arm, i)
            r["secs"] = round(time.time() - t0, 1)
            results.append(r)
            print(f"[{arm} {i}] success={r['success']} rounds={r['rounds']} "
                  f"fmt_viol={r['fmt_violations']} {r['secs']}s",
                  flush=True)
            with open(os.path.join(WORK, "results.jsonl"), "a") as f:
                f.write(json.dumps(r) + "\n")
    # summary
    print("\n=== SUMMARY ===")
    for arm in ("A", "B"):
        a = [r for r in results if r["arm"] == arm]
        s = sum(r["success"] for r in a)
        succ = [r for r in a if r["success"]]
        ar = (sum(r["rounds"] for r in succ) / len(succ)) if succ else float("nan")
        fv = sum(r["fmt_violations"] for r in a)
        label = "pure-text (A)" if arm == "A" else "min-3-tool (B)"
        print(f"{label}: success {s}/{len(a)}  "
              f"avg_rounds_on_success={ar:.1f}  total_fmt_violations={fv}")


if __name__ == "__main__":
    main()
