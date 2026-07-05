#!/usr/bin/env python3
"""tier1-fresh-diagnosis — instead of text-surgery on the SAME poisoned
context (tier1-signal-visibility.py's approach, which only "worked" via
unsafe silent deletion — see memory), spawn a fresh-context sub-agent that
sees ONLY the goal, the last action taken, and the raw blocking errors —
none of the 20+ rounds of momentum that led the primary agent to fixate on
`tests/e2e_context.rs`. Mirrors the existing debugger sub-agent design
(read-only, minimal system prompt) but applied to the plan-tool's compile
gate specifically, which currently has no fresh-eyes escalation at all.

Moment: run3, round 12 (same as tier1-signal-visibility.py's moment_b) —
`plan(check step=1)` fails with a real `warning: unused variable:
system_prompt_override` sitting alongside 14 self-inflicted E0061 arity
errors (pre-existing test callsites broken by the model's own signature
change, confirmed against the clean fixture). The REAL run's primary agent,
seeing all this in its OWN accumulated context, chased only the arity
errors for the next ~70 rounds. Question: does a FRESH mind, given nothing
but this snapshot, correctly identify the warning as the actual root cause
instead?

Nothing is hidden from the sub-agent — it gets the complete, unedited error
text. The lever here is WHO looks and WHAT ELSE is in their context, not
what's removed from the message.
"""

import argparse
import json
import os
import urllib.request

REPO = "/home/michal/dev/miniswe"
ENDPOINT = os.environ.get("LLAMA_ENDPOINT", "http://localhost:8464")

DUMP_B = os.path.join(
    REPO,
    "benchmark_results/compaction_20260704_164112__home_michal_models_gemma-4-26B-A4B-it-G/"
    "unified/run3/llm_dumps/req-1783179507-36182-000012.json",
)

GOAL = (
    "Add a CLI flag --system-prompt-override (short: -s) that takes a string and replaces "
    "the default system prompt with the provided text. When this flag is set, skip all "
    "context providers and just use the override text as the system message. Make sure it "
    "works for both single-shot and interactive modes."
)

LAST_ACTION = (
    "replace_range src/cli/commands/run.rs L132-138: rev_1 applied (+9 -7)\n"
    "[ast] ok\n[lsp file] 0 errors\n[lsp project] 0 error(s) (no change from baseline)\n"
    "Then a `check` tool call reported: [cargo check] OK — no errors."
)

RAW_ERRORS = (
    "Errors:\n"
    "  --> src/cli/commands/run.rs:41:82\n"
    "   |\n"
    "   |                                                                                  "
    "^^^^^^^^^^^^^^^^^^^^^^ help: if this is intentional, prefix it with an underscore: "
    "`_system_prompt_override`\n"
    "   |\n"
    "   = note: `#[warn(unused_variables)]` on by default\n"
    "    Checking miniswe v0.1.0 (/work)\n"
    "tests/e2e_context.rs:17:21: error[E0061]: this function takes 6 arguments but 5 arguments were supplied\n"
    "tests/e2e_context.rs:54:21: error[E0061]: this function takes 6 arguments but 5 arguments were supplied\n"
    "tests/e2e_context.rs:75:21: error[E0061]: this function takes 6 arguments but 5 arguments were supplied\n"
    "tests/e2e_context.rs:93:21: error[E0061]: this function takes 6 arguments but 5 arguments were supplied\n"
    "tests/e2e_context.rs:111:21: error[E0061]: this function takes 6 arguments but 5 arguments were supplied\n"
    "tests/e2e_context.rs:124:21: error[E0061]: this function takes 6 arguments but 5 arguments were supplied\n"
    "tests/e2e_context.rs:147:21: error[E0061]: this function takes 6 arguments but 5 arguments were supplied\n"
    "tests/e2e_context.rs:161:21: error[E0061]: this function takes 6 arguments but 5 arguments were supplied\n"
    "tests/e2e_context.rs:182:21: error[E0061]: this function takes 6 arguments but 5 arguments were supplied\n"
    "tests/e2e_context.rs:346:21: error[E0061]: this function takes 6 arguments but 5 arguments were supplied\n"
    "tests/e2e_context.rs:367:21: error[E0061]: this function takes 6 arguments but 5 arguments were supplied\n"
    "tests/e2e_context.rs:388:21: error[E0061]: this function takes 6 arguments but 5 arguments were supplied\n"
    "tests/e2e_context.rs:416:25: error[E0061]: this function takes 6 arguments but 5 arguments were supplied\n"
    "tests/e2e_context.rs:432:25: error[E0061]: this function takes 6 arguments but 5 arguments were supplied\n"
    "error: could not compile `miniswe` (test \"e2e_context\") due to 14 previous errors"
)

SYSTEM_PROMPT = (
    "You are a READ-ONLY debugging analyst with fresh eyes on a stuck task. You have ONLY "
    "read/search/inspect tools — you CANNOT edit files, run shell, set a plan, or use a "
    "scratchpad, and you must not try (you have no such tools). Do NOT plan; just investigate "
    "and report.\n"
    "Your sole deliverable is a DIAGNOSIS: what is the SINGLE most important problem blocking "
    "progress right now, and what should be fixed FIRST? A wall of errors may contain several "
    "distinct issues of very different importance — do not assume the most NUMEROUS class of "
    "error is the most important one. Read the exact locations named. Then output:\n"
    "PRIORITY ISSUE: <the one thing to fix first, and exactly where>\n"
    "WHY: <why this one, not the others>\n"
    "FIX: <conceptually what must change>"
)


def load_tools():
    d = json.load(open(DUMP_B))
    all_tools = d.get("tools", [])
    keep = {"file", "code", "check", "show_rev"}
    return [t for t in all_tools if t["function"]["name"] in keep], d["model"]


def build_messages():
    user = (
        f"GOAL (the task):\n{GOAL}\n\n"
        f"LAST ACTION TAKEN (this just succeeded):\n{LAST_ACTION}\n\n"
        f"The next verification attempt then produced these BLOCKING ERRORS:\n{RAW_ERRORS}\n\n"
        "Investigate and report your diagnosis per your instructions."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def call_llm(payload, timeout=180):
    req = urllib.request.Request(
        f"{ENDPOINT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def classify(resp):
    msg = resp["choices"][0]["message"]
    tcs = msg.get("tool_calls") or []
    if tcs:
        tc = tcs[0]["function"]
        try:
            args = json.loads(tc["arguments"])
        except Exception:
            args = {}
        path = str(args.get("path", ""))
        on_target = "run.rs" in path or "repl.rs" in path
        on_test = "e2e_context" in path or "test" in path.lower()
        cat = "READ-TARGET" if on_target else ("READ-TEST" if on_test else "READ-OTHER")
        return {"cat": cat, "detail": f"{tc['name']} {path}"}
    text = msg.get("content") or ""
    lower = text.lower()
    mentions_warning = "unused variable" in lower or "system_prompt_override" in lower and (
        "warn" in lower or "run.rs" in lower or "repl.rs" in lower
    )
    mentions_priority_line = "priority issue" in lower
    if mentions_priority_line and ("run.rs" in text or "repl.rs" in text) and "unused" in lower:
        cat = "CORRECT-DIAGNOSIS"
    elif "e2e_context" in text and "unused" not in lower and "run.rs" not in text:
        cat = "WRONG-DIAGNOSIS-TESTS"
    elif mentions_warning:
        cat = "PARTIAL-MENTIONS-WARNING"
    else:
        cat = "OTHER-PROSE"
    return {"cat": cat, "detail": text[:150].replace("\n", " ")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(REPO, "benchmark_results/_moments/fresh-diagnosis-v1"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tools, model = load_tools()
    messages = build_messages()
    samples = []
    for i in range(args.k):
        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "temperature": 0.2,
            "max_tokens": 2000,
            "stream": False,
        }
        try:
            resp = call_llm(payload)
            c = classify(resp)
        except Exception as e:
            c = {"cat": "ERROR", "detail": str(e)[:100]}
        samples.append(c)
        print(f"[fresh_diagnosis] {i+1}/{args.k}: {c['cat']}  {c.get('detail','')}", flush=True)
    json.dump(samples, open(os.path.join(args.out, "results.json"), "w"), indent=1)

    print("\n=== SUMMARY ===")
    from collections import Counter

    cnt = Counter(s["cat"] for s in samples)
    print(dict(cnt))


if __name__ == "__main__":
    main()
