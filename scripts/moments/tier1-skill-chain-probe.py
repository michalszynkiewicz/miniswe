#!/usr/bin/env python3
"""tier1-skill-chain-probe — can gemma resolve a skill NAME to its path?

Skills reference each other by name only ("continue with the
uds-package-integrate skill") because bodies are shared across agents with
different install locations. miniswe's name->path mapping exists in the
[SKILLS] listing (system prompt) which shows each skill's path.

Moment: mid-task, build skill already read (full real body), a feedback
message says SSO is missing and points at the uds-package-integrate skill
BY NAME. Sample the next tool call.

Arms:
  lookup   pure gemma-driven resolution via the [SKILLS] listing (preferred)
  hinted   the routed task additionally carries the path template sentence
           ("If the skill refers you to another skill, read
            .ai/skills/<that-skill-name>/SKILL.md too.")

Usage: tier1-skill-chain-probe.py [--k 10]
"""

import argparse
import json
import os
import urllib.request

REPO = "/home/michal/dev/miniswe"
ENDPOINT = os.environ.get("LLAMA_ENDPOINT", "http://localhost:8464")
SKILLS_DIR = "/tmp/tmp.tII2UGxURO/.ai/skills"
REPRO_DUMPS = "/tmp/claude-1000/-home-michal-dev-miniswe/91153bbc-3489-42aa-88d1-4ad66657da3b/scratchpad/skills-repro/dumps"

UDS_TASK = (
    "Create a UDS package for app-with-dependencies using the Helm chart at "
    "https://github.com/defenseunicorns/uds-mcp (in the `tests/e2e/app-with-deps/src/chart/` "
    "directory, branch/ref: `main`). The container image is at "
    "ghcr.io/michalszynkiewicz/app-with-dependencies:0.1.0. Expose the app at "
    "app-with-deps.uds.dev. Deploy it to a k3d test cluster and verify that the application "
    "is running and healthy."
)

BASE_REWRITE = (
    "Read .ai/skills/uds-package-build/SKILL.md and follow its instructions to handle "
    f"this request: {UDS_TASK}"
)
HINT = (
    " If the skill refers you to another skill, read "
    ".ai/skills/<that-skill-name>/SKILL.md and follow it too."
)

NUDGE = (
    "[validate-package: FAIL — no SSO client found in Keycloak for app-with-deps.uds.dev. "
    "The UDS Package CR's 'sso' section was never written. The build skill notes that SSO "
    "wiring is the Integration Phase's job — continue with the uds-package-integrate skill.]"
)


def build(arm):
    files = sorted(os.listdir(REPRO_DUMPS))
    dump = json.load(open(os.path.join(REPRO_DUMPS, files[0])))
    system = dump["messages"][0]
    tools = dump["tools"]

    skill_body = open(os.path.join(SKILLS_DIR, "uds-package-build", "SKILL.md")).read()
    task = BASE_REWRITE + (HINT if arm == "hinted" else "")

    msgs = [
        system,
        {"role": "user", "content": task},
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "file", "arguments": json.dumps(
                    {"action": "read", "path": ".ai/skills/uds-package-build/SKILL.md"})},
            }],
        },
        {"role": "tool", "tool_call_id": "c1",
         "content": f"[.ai/skills/uds-package-build/SKILL.md: 248 lines]\n{skill_body}"},
        {"role": "user", "content": NUDGE},
    ]
    return msgs, tools, dump["model"], dump.get("temperature", 0.2)


def call_llm(payload, timeout=240):
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
        return "NO-TOOL-CALL"
    fn = tcs[0]["function"]
    try:
        args = json.loads(fn.get("arguments", "{}"))
    except Exception:
        return "MALFORMED"
    path = str(args.get("path", ""))
    if path == ".ai/skills/uds-package-integrate/SKILL.md":
        return "CORRECT-PATH"
    if "uds-package-integrate" in path:
        return f"WRONG-PATH({path})"
    if "uds-package-integrate" in json.dumps(args):
        return f"NAME-ELSEWHERE({fn['name']})"
    return f"OTHER:{fn['name']}({json.dumps(args)[:60]})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    from collections import Counter

    for arm in ("lookup", "hinted"):
        msgs, tools, model, temp = build(arm)
        cats = []
        for i in range(args.k):
            payload = {"model": model, "messages": msgs, "tools": tools,
                       "temperature": temp, "max_tokens": 2500, "stream": False}
            try:
                cats.append(classify(call_llm(payload)))
            except Exception as e:
                cats.append(f"ERROR:{str(e)[:40]}")
            print(f"[{arm}] {i + 1}/{args.k}: {cats[-1]}", flush=True)
        print(f"== {arm}: {dict(Counter(cats))}\n")


if __name__ == "__main__":
    main()
