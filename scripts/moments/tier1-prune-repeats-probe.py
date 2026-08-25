#!/usr/bin/env python3
"""tier1-prune-repeats-probe — is dropping the repeated tool-call/result
pairs SAFE, and does it help at all?

Replays the real loop context (dump 000029: the model has the chart cloned,
Chart.yaml read, version 0.1.0 and tags in context, yet its last 3 actions are
the identical `cat Chart.yaml` narration). Compares:
  A unpruned  — the context as-is.
  B pruned    — collapse each repeated tool-call signature to its FIRST
                occurrence (drop the redundant assistant call + its result).

Grades each next action:
  PROCEED   — a forward build action (scaffold / write / plan / mcp_use).
  LOOP      — re-runs a chart read/verify (the stuck behavior).
  RE_GATHER — re-issues the EXACT command we pruned (amnesia signal — pruning
              made it forget it already has the info; this is the risk).

CAVEAT: this is a COLD replay. It cannot reproduce the warm-KV-cache state that
drives the LIVE loop (that's built incrementally over 30 rounds). So it tests
SAFETY (does pruning break correctness?) and any cold-regime signal; the
loop-breaking itself must be confirmed in a live e2e. Usage: [--k 16]
"""

import argparse
import json
import os
import re
import urllib.request
from collections import Counter

ENDPOINT = os.environ.get("LLAMA_ENDPOINT", "http://localhost:8464")
DUMP = (
    "/home/michal/work/uds-mcp/tests/e2e/results/"
    "app-with-deps-todo-skills-miniswe-20260716-174941/llm_dumps-0/"
    "req-1784216981-4008005-000029.json"
)
LOOP_RX = re.compile(r"ls-remote|chart\.yaml|helm show chart|ls\s+-d.*chart|# the request", re.I)


def prune_repeated(msgs):
    """Collapse each tool-call arg-signature that occurs >=2x to its first
    occurrence; drop later duplicate (assistant call + following tool result).
    Returns (pruned_messages, set_of_pruned_signatures)."""
    sigs = Counter()
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            sigs[tc["function"].get("arguments", "")] += 1
    repeated = {s for s, c in sigs.items() if c >= 2 and s}
    out, kept_once, drop_next_result = [], set(), False
    for m in msgs:
        tcs = m.get("tool_calls") or []
        sig = tcs[0]["function"].get("arguments", "") if tcs else None
        if sig in repeated:
            if sig in kept_once:
                drop_next_result = True
                continue  # drop this duplicate call
            kept_once.add(sig)
            out.append(m)
            drop_next_result = False
            continue
        if m.get("role") == "tool" and drop_next_result:
            drop_next_result = False
            continue  # drop the dropped call's result
        drop_next_result = False
        out.append(m)
    return out, repeated


def call_llm(payload, timeout=300):
    req = urllib.request.Request(
        f"{ENDPOINT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def classify(resp_msg, pruned_sigs):
    tcs = resp_msg.get("tool_calls") or []
    if not tcs:
        return "NO_CALL", (resp_msg.get("content") or "").strip().replace("\n", " ")[:60]
    fn = tcs[0]["function"]
    args = str(fn.get("arguments", ""))
    if args in pruned_sigs:
        return "RE_GATHER", (fn["name"] + " " + args)[:60]
    if LOOP_RX.search(fn["name"] + " " + args):
        return "LOOP", (fn["name"] + " " + args)[:60]
    return "PROCEED", (fn["name"] + " " + args)[:60]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=16)
    args = ap.parse_args()
    dump = json.load(open(DUMP))
    base = dump["messages"]
    pruned, pruned_sigs = prune_repeated(base)
    model = dump["model"]
    temp = dump.get("temperature", 0.15)
    max_tokens = dump.get("max_tokens") or 4000
    kwargs = dump.get("chat_template_kwargs")
    print(f"unpruned: {len(base)} msgs → pruned: {len(pruned)} msgs "
          f"(dropped {len(base) - len(pruned)}; {len(pruned_sigs)} repeated sigs)\n")

    for name, msgs in [("A-unpruned", base), ("B-pruned", pruned)]:
        c = {"PROCEED": 0, "LOOP": 0, "RE_GATHER": 0, "NO_CALL": 0}
        sample = None
        for _ in range(args.k):
            payload = {"model": model, "messages": msgs, "tools": dump.get("tools"),
                       "temperature": temp, "max_tokens": max_tokens, "stream": False}
            if kwargs:
                payload["chat_template_kwargs"] = kwargs
            try:
                out = call_llm(payload)["choices"][0]["message"]
                v, detail = classify(out, pruned_sigs)
            except Exception as e:
                v, detail = "NO_CALL", f"(err {str(e)[:35]})"
            c[v] += 1
            if sample is None or v == "PROCEED":
                sample = f"{v}: {detail}"
        print(f"[{name:<11}] PROCEED {c['PROCEED']:2d}/{args.k}  LOOP {c['LOOP']:2d}  "
              f"RE_GATHER {c['RE_GATHER']:2d}  NO_CALL {c['NO_CALL']:2d}")
        print(f"   e.g. {sample}")


if __name__ == "__main__":
    main()
