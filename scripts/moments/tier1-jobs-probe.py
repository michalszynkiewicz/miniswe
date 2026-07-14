#!/usr/bin/env python3
"""tier1-jobs-probe — can the model drive the new `jobs` tool?

Synthetic-but-faithful moment: a real bench dump donates the genuine system
prompt + tool list (so the probe context matches production shape); we
append the `jobs` tool definition (copied verbatim from
tools/definitions.rs), then construct a deploy turn:

  user:      deploy task (uds-mcp flavored)
  assistant: file(shell zarf package deploy ...)
  tool:      the REAL promotion message (tools::jobs::promotion_message)

and sample k completions. Classify the next call:
  JOBS-WAIT-CHECK  jobs(action=wait, ...) with a check probe  <- ideal
  JOBS-WAIT        jobs(action=wait) without check
  JOBS-STATUS/KILL other well-formed jobs calls
  JOBS-MALFORMED   jobs call that would error (bad action/args)
  BAD-RERUN        re-runs the promoted shell command (explicitly forbidden)
  OTHER            anything else (reads, plan, prose)

Usage: tier1-jobs-probe.py [--k 15]
"""

import argparse
import json
import os
import urllib.request

REPO = "/home/michal/dev/miniswe"
ENDPOINT = os.environ.get("LLAMA_ENDPOINT", "http://localhost:8464")

DONOR_DUMP_DIR = os.path.join(
    REPO,
    "benchmark_results/compaction_20260713_212036__home_michal_models_gemma-4-26B-A4B-it-G/lazy/run1/llm_dumps",
)

DEPLOY_CMD = (
    "zarf package deploy app-with-deps-package/zarf-package-app-with-deps-amd64-0.1.0.tar.zst --confirm"
)

TASK = (
    "Build and deploy the app-with-deps UDS package to the local cluster, then verify the "
    "deployment is healthy (pods Running in namespace app-with-deps)."
)

# Verbatim from tools/definitions.rs jobs_tool_definition().
JOBS_TOOL = {
    "type": "function",
    "function": {
        "name": "jobs",
        "description": "Manage background jobs (long-running shell commands get promoted automatically). Actions: 'wait' (block up to `secs`, return new output; pass `check` = a quick status command to probe progress after the wait, e.g. kubectl get pods), 'status' (new output + state, id optional), 'kill'. A finished job reports its full result once.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["status", "wait", "kill"], "description": "What to do"},
                "id": {"type": "integer", "description": "Job id (from the promotion message; optional when only one job is live)"},
                "secs": {"type": "integer", "description": "wait: max seconds to block (default 60, cap 300)"},
                "check": {"type": "string", "description": "wait: shell command run after the wait to probe progress; its output is returned"},
            },
            "required": ["action"],
        },
    },
}

# Mirrors tools::jobs::promotion_message(1, DEPLOY_CMD, 60, <zarf-ish tail>).
PROMOTION_RESULT = (
    "[shell: still running after 60s — promoted to background job 1]\n"
    f"  $ {DEPLOY_CMD}\n"
    "Output so far:\n"
    "  Loading Zarf Package app-with-deps\n"
    "  Deploying component 'app'\n"
    "  Waiting for deployment app-with-deps/app to be ready\n"
    "The command keeps running. Do NOT re-run it. "
    "Wait and monitor with jobs(action='wait', id=1, secs=60, check='<status command>') — "
    "use the progress-check command your task guidance recommends (e.g. kubectl get pods). "
    "jobs(action='kill', id=1) stops it."
)


def load_donor():
    files = sorted(os.listdir(DONOR_DUMP_DIR))
    dump = json.load(open(os.path.join(DONOR_DUMP_DIR, files[-1])))
    return dump


def build_context(dump):
    system = dump["messages"][0]
    tools = [t for t in dump["tools"]] + [JOBS_TOOL]
    call_id = "probe-shell-call-1"
    messages = [
        system,
        {"role": "user", "content": TASK},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "file",
                        "arguments": json.dumps({"action": "shell", "command": DEPLOY_CMD}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": PROMOTION_RESULT},
    ]
    return messages, tools, dump["model"], dump.get("temperature", 0.2), dump.get("max_tokens", 8000)


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
    if not tcs:
        return {"cat": "OTHER-PROSE", "detail": (msg.get("content") or "")[:120].replace("\n", " ")}
    tc = tcs[0]["function"]
    name = tc.get("name", "")
    try:
        args = json.loads(tc.get("arguments", "{}"))
    except Exception:
        return {"cat": "MALFORMED-JSON", "detail": str(tc)[:120]}
    if name == "jobs":
        action = args.get("action")
        if action not in ("status", "wait", "kill"):
            return {"cat": "JOBS-MALFORMED", "detail": json.dumps(args)[:150]}
        wrong_id = "id" in args and args.get("id") != 1
        if wrong_id:
            return {"cat": "JOBS-MALFORMED", "detail": f"wrong id: {json.dumps(args)[:130]}"}
        if action == "wait":
            cat = "JOBS-WAIT-CHECK" if args.get("check") else "JOBS-WAIT"
            return {"cat": cat, "detail": json.dumps(args)[:150]}
        return {"cat": f"JOBS-{action.upper()}", "detail": json.dumps(args)[:150]}
    if name == "file":
        cmd = str(args.get("command", ""))
        if "zarf package deploy" in cmd:
            return {"cat": "BAD-RERUN", "detail": cmd[:120]}
        return {"cat": "OTHER-SHELL", "detail": cmd[:120]}
    return {"cat": f"OTHER:{name}", "detail": json.dumps(args)[:120]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--out", default=os.path.join(REPO, "benchmark_results/_moments/jobs-probe-v1"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dump = load_donor()
    messages, tools, model, temperature, max_tokens = build_context(dump)

    samples = []
    for i in range(args.k):
        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        try:
            resp = call_llm(payload)
            c = classify(resp)
        except Exception as e:
            c = {"cat": "ERROR", "detail": str(e)[:150]}
        samples.append(c)
        print(f"[jobs-probe] {i + 1}/{args.k}: {c['cat']}  {c.get('detail', '')}", flush=True)
    json.dump(samples, open(os.path.join(args.out, "results.json"), "w"), indent=1)

    print("\n=== SUMMARY ===")
    from collections import Counter

    print(dict(Counter(s["cat"] for s in samples)))


if __name__ == "__main__":
    main()
