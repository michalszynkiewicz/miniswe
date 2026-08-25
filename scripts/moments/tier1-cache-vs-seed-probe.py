#!/usr/bin/env python3
"""tier1-cache-vs-seed-probe — is the warm/cold difference the KV CACHE (q4)
or just SAMPLING VARIANCE (seed)?

Two tests on the real loop context (dump 000029):
  T1  Fix the seed, toggle cache_prompt (cold prefill vs warm reuse). Same
      seed + same prompt: if the emitted action DIFFERS, the (q4) cache path
      changes numerics → it IS the cache. If IDENTICAL, the cache is
      behaviorally exact → the warm/cold behavioral gap was seed/sampling.
  T2  Cold across many seeds: how often does the loop appear at all? If it's
      reachable cold under some seeds, sampling variance is a real driver
      (the live loop = an unlucky streak the 3-repeat detector caught).

Usage: [--seeds 12]
"""

import argparse
import json
import re
import urllib.request

ENDPOINT = "http://localhost:8464"
DUMP = (
    "/home/michal/work/uds-mcp/tests/e2e/results/"
    "app-with-deps-todo-skills-miniswe-20260716-174941/llm_dumps-0/"
    "req-1784216981-4008005-000029.json"
)
LOOP_RX = re.compile(r"ls-remote|chart\.yaml|helm show chart|ls\s+-d.*chart|# the request", re.I)


def call(msgs, tools, model, seed, cache_prompt, max_tokens, kwargs):
    payload = {"model": model, "messages": msgs, "tools": tools, "temperature": 0.15,
               "seed": seed, "cache_prompt": cache_prompt, "max_tokens": max_tokens,
               "stream": False}
    if kwargs:
        payload["chat_template_kwargs"] = kwargs
    req = urllib.request.Request(
        f"{ENDPOINT}/v1/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["choices"][0]["message"]


def action_sig(m):
    tcs = m.get("tool_calls") or []
    if not tcs:
        return "TXT:" + (m.get("content") or "").strip()[:40]
    fn = tcs[0]["function"]
    return f"{fn['name']}({str(fn.get('arguments',''))[:50]})"


def verdict(m):
    tcs = m.get("tool_calls") or []
    if not tcs:
        return "NO_CALL"
    fn = tcs[0]["function"]
    return "LOOP" if LOOP_RX.search(fn["name"] + " " + str(fn.get("arguments", ""))) else "PROCEED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    args = ap.parse_args()
    d = json.load(open(DUMP))
    msgs, tools, model = d["messages"], d.get("tools"), d["model"]
    mt = d.get("max_tokens") or 4000
    kw = d.get("chat_template_kwargs")

    print("=== T1: same seed, cold (cache_prompt=false) vs warm (=true) ===")
    same = diff = 0
    for seed in range(1, 5):
        cold = call(msgs, tools, model, seed, False, mt, kw)   # fresh prefill
        warm = call(msgs, tools, model, seed, True, mt, kw)    # reuses the slot just filled
        sc, sw = action_sig(cold), action_sig(warm)
        match = sc == sw
        same += match
        diff += not match
        print(f"  seed={seed}: {'SAME' if match else 'DIFFER'}")
        if not match:
            print(f"     cold: {sc}")
            print(f"     warm: {sw}")
    print(f"  => {same} same / {diff} differ  "
          f"({'CACHE changes fixed-seed output' if diff else 'cache behaviorally exact'})\n")

    print(f"=== T2: cold across {args.seeds} seeds — is the loop reachable? ===")
    c = {"PROCEED": 0, "LOOP": 0, "NO_CALL": 0}
    for seed in range(1, args.seeds + 1):
        c[verdict(call(msgs, tools, model, seed, False, mt, kw))] += 1
    print(f"  PROCEED {c['PROCEED']}  LOOP {c['LOOP']}  NO_CALL {c['NO_CALL']}  (of {args.seeds})")


if __name__ == "__main__":
    main()
