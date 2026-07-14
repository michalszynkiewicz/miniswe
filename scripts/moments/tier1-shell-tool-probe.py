#!/usr/bin/env python3
"""tier1-shell-tool-probe — A/B the tool surface for command execution.

Arm A (current):      file(action='shell', background=true) + separate jobs tool
Arm B (consolidated): top-level shell tool owning the whole lifecycle:
                      actions run / wait / status / kill (file loses shell)

Three moments, each seeded with arm-appropriate history:
  start    task only → how does the model launch a known-long deploy?
  monitor  after the background-start message → first monitoring call?
  poll     after start + ONE direct status poll → poll again (unpaced, the
           failure the redirect nudge patches today) or switch to wait?

Hypothesis under test: consolidation removes the tool-switch at the
monitoring step, so arm B should show fewer direct re-polls at the `poll`
moment and equal-or-better wait adoption elsewhere.

Usage: tier1-shell-tool-probe.py [--k 8]
"""

import argparse
import copy
import json
import os
import urllib.request

REPO = "/home/michal/dev/miniswe"
ENDPOINT = os.environ.get("LLAMA_ENDPOINT", "http://localhost:8464")
DONOR = os.path.join(
    REPO,
    "benchmark_results/compaction_20260713_212036__home_michal_models_gemma-4-26B-A4B-it-G/lazy/run1/llm_dumps",
)

TASK = (
    "Deploy the app by running ./fake-deploy.sh and confirm the deployment completes "
    "successfully. It takes a few minutes. Monitor progress with ./check-deploy.sh — done "
    "when it reports READY 5/5."
)

GUIDE_A = (
    "Long-running commands (deploys, builds, servers): run with file(action='shell', "
    "background=true) and manage with jobs(action='wait'|'status'|'kill'); wait takes "
    "check='<status cmd>' to probe progress. NEVER self-background with '&' or nohup."
)
GUIDE_B = (
    "Long-running commands (deploys, builds, servers): run with shell(action='run', "
    "background=true), then monitor with shell(action='wait', id=N, secs=60, "
    "check='<status cmd>') and stop with shell(action='kill'). NEVER self-background "
    "with '&' or nohup."
)

JOBS_TOOL = {
    "type": "function",
    "function": {
        "name": "jobs",
        "description": "Manage background jobs (started with file shell background=true, or auto-promoted long commands). Actions: 'wait' (block up to `secs`, return new output; pass `check` = a quick status command to probe progress after the wait), 'status' (new output + state, id optional), 'kill'. A finished job reports its full result once.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["status", "wait", "kill"]},
                "id": {"type": "integer"},
                "secs": {"type": "integer"},
                "check": {"type": "string"},
            },
            "required": ["action"],
        },
    },
}

SHELL_TOOL = {
    "type": "function",
    "function": {
        "name": "shell",
        "description": "Run shell commands and manage long-running ones. Actions: 'run' (execute `command`; pass background=true for long-running commands — deploys, builds, servers), 'wait' (block up to `secs` for background job `id`, return new output; pass `check` = a quick status command probed after the wait), 'status' (new output + state), 'kill'. Long foreground commands are auto-promoted to background jobs.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["run", "wait", "status", "kill"]},
                "command": {"type": "string", "description": "for run"},
                "timeout": {"type": "integer", "description": "for run"},
                "background": {"type": "boolean", "description": "run: start as a background job"},
                "id": {"type": "integer"},
                "secs": {"type": "integer"},
                "check": {"type": "string"},
            },
            "required": ["action"],
        },
    },
}


def start_message(arm):
    mgmt = (
        "jobs(action='wait', id=1, secs=60, check='<status command>')"
        if arm == "A"
        else "shell(action='wait', id=1, secs=60, check='<status command>')"
    )
    kill = "jobs(action='kill', id=1)" if arm == "A" else "shell(action='kill', id=1)"
    return (
        "[shell: started as background job 1]\n  $ ./fake-deploy.sh\n"
        "It runs in the background. Wait and monitor with "
        f"{mgmt} — use the progress-check command your task guidance recommends. "
        f"{kill} stops it."
    )


def deploy_call(arm, call_id):
    if arm == "A":
        fn = {"name": "file", "arguments": json.dumps({"action": "shell", "command": "./fake-deploy.sh", "background": True})}
    else:
        fn = {"name": "shell", "arguments": json.dumps({"action": "run", "command": "./fake-deploy.sh", "background": True})}
    return {"role": "assistant", "tool_calls": [{"id": call_id, "type": "function", "function": fn}]}


def poll_call(arm, call_id):
    if arm == "A":
        fn = {"name": "file", "arguments": json.dumps({"action": "shell", "command": "./check-deploy.sh"})}
    else:
        fn = {"name": "shell", "arguments": json.dumps({"action": "run", "command": "./check-deploy.sh"})}
    return {"role": "assistant", "tool_calls": [{"id": call_id, "type": "function", "function": fn}]}


def build(arm, moment):
    files = sorted(os.listdir(DONOR))
    dump = json.load(open(os.path.join(DONOR, files[-1])))
    system = copy.deepcopy(dump["messages"][0])
    guide = GUIDE_A if arm == "A" else GUIDE_B
    system["content"] += "\n" + guide
    if arm == "B":
        # The donor prompt's tool-contract section teaches file-shell by
        # example — with the action removed from the schema, that line is
        # a leak that contaminated the first probe run (6/6 near-verbatim
        # imitations of its `ls` example). Swap it for the new surface.
        system["content"] = system["content"].replace(
            'file shell: {"action":"shell","command":"ls","timeout":60}',
            'shell run: {"action":"run","command":"ls","timeout":60}',
        )

    tools = copy.deepcopy(dump["tools"])
    if arm == "A":
        tools.append(JOBS_TOOL)
    else:
        for t in tools:
            if t["function"]["name"] == "file":
                t["function"]["description"] = t["function"]["description"].replace("shell, ", "")
                d = t["function"]["parameters"]["properties"]
                d.pop("command", None)
                d.pop("timeout", None)
                if "action" in d:
                    d["action"]["description"] = "One of: read, delete, search, revert, help"
        tools.append(SHELL_TOOL)

    msgs = [system, {"role": "user", "content": TASK}]
    if moment in ("monitor", "poll"):
        msgs.append(deploy_call(arm, "c1"))
        msgs.append({"role": "tool", "tool_call_id": "c1", "content": start_message(arm)})
    if moment == "poll":
        msgs.append(poll_call(arm, "c2"))
        msgs.append({"role": "tool", "tool_call_id": "c2", "content": "[shell: exit 0]\nREADY 1/5 — pulling images"})
    return msgs, tools, dump["model"], dump.get("temperature", 0.2), dump.get("max_tokens", 8000)


def call_llm(payload, timeout=180):
    req = urllib.request.Request(
        f"{ENDPOINT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def classify(arm, moment, resp):
    msg = resp["choices"][0]["message"]
    tcs = msg.get("tool_calls") or []
    if not tcs:
        return "NO-TOOL-CALL"
    fn = tcs[0]["function"]
    name = fn.get("name", "")
    try:
        args = json.loads(fn.get("arguments", "{}"))
    except Exception:
        return "MALFORMED"
    cmd = str(args.get("command", ""))
    is_shell_run = (arm == "A" and name == "file" and args.get("action") == "shell") or (
        arm == "B" and name == "shell" and args.get("action") == "run"
    )
    wait_tool = "jobs" if arm == "A" else "shell"
    is_wait = name == wait_tool and args.get("action") == "wait"
    is_mgmt = name == wait_tool and args.get("action") in ("wait", "status", "kill")

    if moment == "start":
        if is_shell_run and "fake-deploy" in cmd:
            if "&" in cmd:
                return "START-WITH-AMP"
            return "START-BG" if args.get("background") else "START-FG"
        if name == "plan" or (name == "file" and args.get("action") in ("read", "search")) or name == "code":
            return "EXPLORE-FIRST"
        return f"OTHER:{name}"
    if is_wait:
        return "WAIT-CHECK" if args.get("check") else "WAIT"
    if is_mgmt:
        return f"MGMT-{args.get('action').upper()}"
    if is_shell_run and "check-deploy" in cmd:
        return "DIRECT-POLL"
    if is_shell_run and "fake-deploy" in cmd:
        return "RERUN-DEPLOY"
    return f"OTHER:{name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(REPO, "benchmark_results/_moments/shell-tool-probe-v1"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from collections import Counter

    results = {}
    for arm in ("A", "B"):
        for moment in ("start", "monitor", "poll"):
            msgs, tools, model, temp, max_tok = build(arm, moment)
            cats = []
            for i in range(args.k):
                payload = {"model": model, "messages": msgs, "tools": tools,
                           "temperature": temp, "max_tokens": max_tok, "stream": False}
                try:
                    cats.append(classify(arm, moment, call_llm(payload)))
                except Exception as e:
                    cats.append(f"ERROR:{str(e)[:60]}")
                print(f"[{arm}/{moment}] {i + 1}/{args.k}: {cats[-1]}", flush=True)
            results[f"{arm}/{moment}"] = dict(Counter(cats))
            json.dump(results, open(os.path.join(args.out, "results.json"), "w"), indent=1)

    print("\n=== SUMMARY ===")
    for key, cnt in results.items():
        print(f"{key}: {cnt}")


if __name__ == "__main__":
    main()
