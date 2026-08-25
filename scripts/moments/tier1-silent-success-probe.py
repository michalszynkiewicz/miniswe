#!/usr/bin/env python3
"""tier1-silent-success-probe — does a bare "[shell: exit 0]" make the model
re-run a silent generator instead of accepting success?

The uds e2e looped on `uds zarf dev generate`: it writes a zarf.yaml but
prints ~nothing, so the shell tool returns a bare "[shell: exit 0]" — no
confirmation of WHAT happened or which file appeared. Hypothesis: the model
reads "no output" as "unclear / maybe didn't work" and re-runs (often with
different flags), churning until the loop detector kills the turn.

Arms (only the tool result of the generate call differs):
  bare       "[shell: exit 0]"                          (today's behavior)
  annotated  "[shell: exit 0 — completed, no output. Any files it writes are
             now on disk; read them to verify instead of re-running.]"

Next-action classes:
  RE-RUN      runs zarf dev generate again  <- the loop
  VERIFY      reads the generated file / lists it / checks  <- desired
  PROCEED     plan check / moves to next step  <- desired
  OTHER

Usage: tier1-silent-success-probe.py [--k 12]
"""

import argparse
import json
import os
import urllib.request

REPO = "/home/michal/dev/miniswe"
ENDPOINT = os.environ.get("LLAMA_ENDPOINT", "http://localhost:8464")
REPRO_DUMPS = "/tmp/claude-1000/-home-michal-dev-miniswe/91153bbc-3489-42aa-88d1-4ad66657da3b/scratchpad/skills-repro/dumps"

GEN_CMD = (
    "uds zarf dev generate app-with-deps --url https://github.com/defenseunicorns/uds-mcp.git "
    "--version main --output-directory app-with-deps-package"
)
TASK = (
    "Build the UDS package for app-with-deps. You are on the GenerateZarf step: generate the "
    "initial Zarf package structure, then the next steps review and edit the generated files."
)

RESULTS = {
    "bare": "[shell: exit 0]",
    "annotated": (
        "[shell: exit 0 — command completed successfully with no output. Any files it "
        "writes are now on disk; read them to verify instead of re-running.]"
    ),
}


def base():
    d = json.load(open(os.path.join(REPRO_DUMPS, sorted(os.listdir(REPRO_DUMPS))[0])))
    return d["messages"][0], d["tools"], d["model"], d.get("temperature", 0.2)


def build(system, arm):
    return [
        system,
        {"role": "user", "content": TASK},
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "g1", "type": "function",
                "function": {"name": "shell", "arguments": json.dumps(
                    {"action": "run", "command": GEN_CMD})},
            }],
        },
        {"role": "tool", "tool_call_id": "g1", "content": RESULTS[arm]},
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
    m = resp["choices"][0]["message"]
    tcs = m.get("tool_calls") or []
    if not tcs:
        return "PROCEED(no-tool/text)"
    fn = tcs[0]["function"]
    name = fn.get("name", "")
    args = fn.get("arguments", "")
    blob = (name + " " + args).lower()
    if "zarf dev generate" in blob:
        return "RE-RUN"
    if name == "file" and ('"action": "read"' in args or '"action":"read"' in args):
        return "VERIFY(read)"
    if name in ("file", "shell") and ("ls" in blob or "cat" in blob or "find" in blob):
        return "VERIFY(list)"
    if name == "plan":
        return "PROCEED(plan)"
    return f"OTHER({name})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12)
    args = ap.parse_args()
    from collections import Counter

    system, tools, model, temp = base()
    for arm in ("bare", "annotated"):
        msgs = build(system, arm)
        cats = []
        for i in range(args.k):
            payload = {"model": model, "messages": msgs, "tools": tools,
                       "temperature": temp, "max_tokens": 2000, "stream": False}
            try:
                cats.append(classify(call_llm(payload)))
            except Exception as e:
                cats.append(f"ERROR:{str(e)[:40]}")
            print(f"[{arm}] {i + 1}/{args.k}: {cats[-1]}", flush=True)
        print(f"== {arm}: {dict(Counter(cats))}\n")


if __name__ == "__main__":
    main()
