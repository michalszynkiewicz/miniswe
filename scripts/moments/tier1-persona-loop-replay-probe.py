#!/usr/bin/env python3
"""tier1-persona-loop-replay-probe — does prompt framing break gemma's
re-verification loop?

Replays a REAL captured request from the point where gemma is stuck
re-reading the chart (it already has the repo cloned, Chart.yaml read, version
0.1.0 and the git tags all in ~7k tokens of context — 11% of the window — yet
keeps re-running `cat Chart.yaml` / `git ls-remote`). We diagnosed this as a
DISPOSITION failure (won't commit "I have enough, move on"), not context size.

Tests whether a system-prompt persona / decisiveness cue shifts the next
action from LOOP (re-verify the chart) to PROCEED (scaffold / write / advance).
Arms:
  A baseline        — the exact captured system prompt.
  B senior-persona  — prepend "You are a senior software engineer." (upfront).
  C decisive-system — + a "commit once info is sufficient, don't re-verify" line
                      in the SYSTEM prompt (distant).
  D decisive-recency— same cue but appended as the FINAL message (recency —
                      where this codebase's evidence says guidance lands).

Grades the emitted tool call. Usage: [--k 8] [--dump PATH]
"""

import argparse
import json
import os
import re
import urllib.request

ENDPOINT = os.environ.get("LLAMA_ENDPOINT", "http://localhost:8464")
DEFAULT_DUMP = (
    "/home/michal/work/uds-mcp/tests/e2e/results/"
    "app-with-deps-todo-skills-miniswe-20260716-174941/llm_dumps-0/"
    "req-1784216981-4008005-000029.json"
)

PERSONA = "You are a senior software engineer. "
DECISIVE = (
    "Work decisively: once you have the information a step needs, COMMIT and move on. "
    "Do NOT re-run commands to re-verify facts you already have (you already cloned the repo, "
    "read Chart.yaml, and know the chart version and tags). Take the next build action."
)

# The model is stuck re-reading/verifying the chart — any of these = LOOP.
LOOP_RX = re.compile(
    r"ls-remote|chart\.yaml|helm show chart|ls\s+-d.*chart|# the request|git clone",
    re.I,
)


def call_llm(payload, timeout=300):
    req = urllib.request.Request(
        f"{ENDPOINT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def classify(resp_msg):
    """LOOP (re-verifies chart) / PROCEED (forward tool call) / NO_CALL."""
    tcs = resp_msg.get("tool_calls") or []
    if not tcs:
        return "NO_CALL", (resp_msg.get("content") or "").strip().replace("\n", " ")[:70]
    first = tcs[0]["function"]
    blob = first["name"] + " " + str(first.get("arguments", ""))
    if LOOP_RX.search(blob):
        return "LOOP", blob[:70]
    return "PROCEED", blob[:70]


def build_messages(base_msgs, arm):
    msgs = [dict(m) for m in base_msgs]
    # locate the system message
    sys_i = next((i for i, m in enumerate(msgs) if m.get("role") == "system"), 0)
    sys = msgs[sys_i].get("content") or ""
    if arm == "A-baseline":
        pass
    elif arm == "B-senior":
        msgs[sys_i]["content"] = PERSONA + sys
    elif arm == "C-decisive-system":
        msgs[sys_i]["content"] = sys + "\n\n" + DECISIVE
    elif arm == "D-decisive-recency":
        msgs.append({"role": "user", "content": "[" + DECISIVE + "]"})
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--dump", default=DEFAULT_DUMP)
    args = ap.parse_args()

    dump = json.load(open(args.dump))
    base_msgs = dump["messages"]
    tools = dump.get("tools")
    model = dump["model"]
    temp = dump.get("temperature", 0.2)
    max_tokens = dump.get("max_tokens") or 4000
    kwargs = dump.get("chat_template_kwargs")
    print(f"replay: {os.path.basename(args.dump)}  ({len(base_msgs)} msgs, temp={temp}, "
          f"max_tokens={max_tokens})\n")

    for arm in ["A-baseline", "B-senior", "C-decisive-system", "D-decisive-recency"]:
        msgs = build_messages(base_msgs, arm)
        counts = {"LOOP": 0, "PROCEED": 0, "NO_CALL": 0}
        sample = None
        for i in range(args.k):
            payload = {"model": model, "messages": msgs, "tools": tools,
                       "temperature": temp, "max_tokens": max_tokens, "stream": False}
            if kwargs:
                payload["chat_template_kwargs"] = kwargs
            try:
                out = call_llm(payload)["choices"][0]["message"]
                verdict, detail = classify(out)
            except Exception as e:
                verdict, detail = "NO_CALL", f"(err {str(e)[:40]})"
            counts[verdict] += 1
            if sample is None or verdict == "PROCEED":
                sample = f"{verdict}: {detail}"
        print(f"[{arm:<20}] PROCEED {counts['PROCEED']}/{args.k}  "
              f"LOOP {counts['LOOP']}/{args.k}  NO_CALL {counts['NO_CALL']}/{args.k}")
        print(f"   e.g. {sample}")


if __name__ == "__main__":
    main()
