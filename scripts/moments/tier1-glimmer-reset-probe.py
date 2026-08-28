#!/usr/bin/env python3
"""tier1 — does a CONTEXT RESET escape Glimmer's terminal read loop?

The wording probe (tier1-glimmer-readloop-probe.py) returned 0 escapes in
96/96 samples across 6 arms. That leaves one question open: is the model
unable to do this plan step, or is the CONTEXT the thing that is stuck?

Arms (all at moment 130, the deepest point of the loop):
  control   the real 160-message prompt, untouched            (calibration)
  reset     system + original task + [CURRENT STATE] + file tail, nothing else
  reset+tail  same, plus the last 40 lines of the target file inlined
  reset+step  reset, plus one concrete sentence naming the action

Scoring identical to the wording probe: EDIT = escape.
"""
import json, os, sys, time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
P = import_module("tier1-glimmer-readloop-probe".replace("-", "_")) if False else None

import urllib.request
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN = os.path.join(REPO,
    "benchmark_results/docker_20260826_163007__home_michal_models_Muse-Glimmer-30B-GGU",
    "00_baseline/llm_dumps")
ENDPOINT = os.environ.get("LLAMA_ENDPOINT", "http://localhost:8464")
EDIT_TOOLS = {"insert_at", "replace_range", "write_file", "edit_file", "refactor"}

def call(payload, timeout=900):
    req = urllib.request.Request(f"{ENDPOINT}/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def classify(resp):
    m = resp["choices"][0]["message"]
    tcs = m.get("tool_calls") or []
    if not tcs:
        return "STALL", (m.get("content") or "").strip()[:80]
    name = tcs[0]["function"]["name"]
    try: args = json.loads(tcs[0]["function"]["arguments"])
    except Exception: args = {}
    if name in EDIT_TOOLS:
        return "EDIT", "%s(%s:%s)" % (name, args.get("path",""), args.get("line", args.get("start","")))
    if name == "file" and args.get("action") == "read":
        return "READ", "read %s:%s-%s" % (args.get("path",""), args.get("start",""), args.get("end",""))
    return "OTHER", "%s(%s)" % (name, args.get("action",""))

d = json.load(open(os.path.join(RUN, sorted(os.listdir(RUN))[130])))
MSGS, TOOLS = d["messages"], d["tools"]

SYS  = MSGS[0]
TASK = next(m for m in MSGS if m["role"] == "user")

# the [CURRENT STATE] block as the harness itself renders it
state = ""
for m in reversed(MSGS):
    c = m.get("content") or ""
    if isinstance(c, str) and "[CURRENT STATE]" in c:
        state = c[c.index("[CURRENT STATE]"):]
        break
assert state, "no [CURRENT STATE] block found"

FILE_TAIL = """[tests/e2e_context.rs: 441 lines, showing L420-441]
 420│    #[tokio::test]
 421│    async fn meta_question_detection_is_conservative() {
 ...
 435│        let system = assembled.messages[0].content.as_deref().unwrap();
 436│        assert!(
 437│            !system.contains("[USAGE GUIDE]"),
 438│            "should NOT detect as meta question: {q}"
 439│        );
 440│    }
 441│}"""

STEP = ("The only remaining plan step is the test. The file ends at line 441 "
        "with the closing brace of the test module. Append the new test there.")

def build(arm):
    if arm == "control":
        return [dict(m) for m in MSGS]
    body = state
    if arm in ("reset+tail", "reset+step"):
        body += "\n\n" + FILE_TAIL
    if arm == "reset+step":
        body += "\n\n" + STEP
    return [SYS, TASK, {"role": "user", "content": body}]

ARMS = ["control", "reset", "reset+tail", "reset+step"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
tot = {}
for arm in ARMS:
    msgs = build(arm)
    c, t0 = Counter(), time.time()
    for k in range(N):
        payload = {"messages": msgs, "tools": TOOLS, "tool_choice": "auto",
                   "temperature": 0.7, "top_p": 0.95, "seed": 1000 + k, "max_tokens": 700}
        try:
            v, det = classify(call(payload))
        except Exception as e:
            v, det = "ERR", str(e)[:80]
        c[v] += 1
        print("  m130 %-12s %d/%d  %-6s %s" % (arm, k + 1, N, v, det)); sys.stdout.flush()
    tot[arm] = c
    print("m130 %-12s EDIT=%d READ=%d OTHER=%d STALL=%d ERR=%d (%ds) msgs=%d"
          % (arm, c["EDIT"], c["READ"], c["OTHER"], c["STALL"], c["ERR"],
             time.time() - t0, len(msgs)))
    sys.stdout.flush()

print("\n" + "=" * 60)
for arm in ARMS:
    c = tot[arm]
    print("%-12s  escape %d/%d   E%d/R%d/O%d/S%d/X%d"
          % (arm, c["EDIT"], N, c["EDIT"], c["READ"], c["OTHER"], c["STALL"], c["ERR"]))
