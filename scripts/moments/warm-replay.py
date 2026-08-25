#!/usr/bin/env python3
"""warm-replay — test whether server-side KV-cache state flips the readloop moment.

Motivation: the live e2e run looped (re-read values.yaml at dumps 36..41) while
tier1-readloop.py replays of the byte-identical request-36 escaped 48/48 on a
freshly restarted server, same temperature 0.15. Same tokens in, different
behavior out => the difference lives in server state, not context content.

Modes:
  --mode cold : restart-fresh server assumed; sample the moment request K times
                with logprobs, print close-call token margins.
  --mode warm : first replay dumps 0..N-1 sequentially with max_tokens=1 so the
                slot cache is laid incrementally (chunked pp, approximating the
                live session's growth), THEN sample the moment request K times.
                The server must NOT have seen the moment request before warmup
                (a prior full-prompt sample would pre-lay the whole prefix in
                one shot and make warmup a no-op cache hit).

Fidelity caveat: live, the model's own generated tokens were laid into the
cache by token-generation (tg) kernels; sequential replay lays them via prompt
processing (pp) in request-sized chunks. So warm-replay reproduces
*incremental chunked pp* cache state, not the exact live tg-laid state. A flip
here is positive evidence for cache-state sensitivity; a non-flip is
inconclusive (tg-laid entries might still be needed).
"""

import argparse
import glob
import importlib.util
import json
import os
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FIX = os.path.join(REPO, "benchmark_results/_fixtures/uds-readloop-121347")
ENDPOINT = os.environ.get("LLAMA_ENDPOINT", "http://localhost:8464")

spec = importlib.util.spec_from_file_location(
    "probe", os.path.join(HERE, "tier1-readloop.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def call(payload, timeout=600):
    req = urllib.request.Request(
        f"{ENDPOINT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def close_calls(resp, margin_lt=2.0, max_tokens=120):
    """Tokens among the first max_tokens generated where top1-top2 logprob
    margin is below margin_lt — the points where sampling could plausibly
    branch."""
    lp = (resp["choices"][0].get("logprobs") or {}).get("content")
    if not lp:
        return None
    out = []
    for pos, t in enumerate(lp[:max_tokens]):
        tops = t.get("top_logprobs") or []
        if len(tops) < 2:
            continue
        m = tops[0]["logprob"] - tops[1]["logprob"]
        if m < margin_lt:
            out.append({"pos": pos, "top1": tops[0]["token"],
                        "top2": tops[1]["token"],
                        "margin": round(m, 3),
                        "p1": round(2.718281828 ** tops[0]["logprob"], 4),
                        "p2": round(2.718281828 ** tops[1]["logprob"], 4)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cold", "warm"], required=True)
    ap.add_argument("--variant", default="control",
                    choices=["control", "inband", "compact", "both"],
                    help="transform applied to the MOMENT request only; "
                         "warmup always replays live history verbatim")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1,
                    help=">1: roll continuable step-1 actions forward by "
                         "synthesizing their results (see tier1-readloop.py)")
    ap.add_argument("--moment-dump", default="llm_dumps2/req-1784119403-3306131-000036.json")
    ap.add_argument("--dumps-dir", default="llm_dumps2")
    ap.add_argument("--ref-dir", default="workspace-capture2")
    ap.add_argument("--kind", default="readloop")
    ap.add_argument("--out", default=os.path.join(REPO, "benchmark_results/_moments/readloop-v1"))
    args = ap.parse_args()

    moment_path = os.path.join(FIX, args.moment_dump)
    dump = json.load(open(moment_path))
    sig = probe.moment_signature(dump, args.kind)
    ref_dir = os.path.join(FIX, args.ref_dir)
    print(f"[{args.mode}] sig={sig}")

    if args.mode == "warm":
        all_dumps = sorted(glob.glob(os.path.join(FIX, args.dumps_dir, "*.json")))
        warmup = [p for p in all_dumps if p < moment_path]
        print(f"[warm] replaying {len(warmup)} requests sequentially (max_tokens=1)")
        t0 = time.time()
        for i, p in enumerate(warmup):
            d = json.load(open(p))
            payload = {"model": d["model"], "messages": d["messages"],
                       "temperature": d.get("temperature", 0.15),
                       "max_tokens": 1, "stream": False}
            if "tools" in d:
                payload["tools"] = d["tools"]
            if "chat_template_kwargs" in d:
                payload["chat_template_kwargs"] = d["chat_template_kwargs"]
            call(payload)
            if (i + 1) % 6 == 0:
                print(f"[warm] {i+1}/{len(warmup)} ({time.time()-t0:.0f}s)", flush=True)
        print(f"[warm] warmup done in {time.time()-t0:.0f}s")

    msgs = probe.transform(dump["messages"], args.variant, args.kind)
    payload = {"model": dump["model"], "messages": msgs,
               "tools": dump["tools"], "temperature": dump.get("temperature", 0.15),
               "max_tokens": dump.get("max_tokens", 16384), "stream": False,
               "logprobs": True, "top_logprobs": 5}
    if "chat_template_kwargs" in dump:
        payload["chat_template_kwargs"] = dump["chat_template_kwargs"]

    result_index = probe.build_result_index(dump["messages"])
    samples = []
    for i in range(args.k):
        steps = []
        cur_msgs = msgs
        first_cc = None
        for step in range(args.steps):
            p = dict(payload, messages=cur_msgs)
            resp = call(p)
            c = probe.classify(resp, args.kind, sig, ref_dir)
            if step == 0:
                first_cc = close_calls(resp)
            steps.append(c)
            if step + 1 >= args.steps or c["cat"] not in probe.ROLLOUT_CONTINUABLE:
                break
            nxt = probe.rollout_step(p, cur_msgs, resp)
            if nxt is None:
                break
            cur_msgs, tname, targs = nxt
            obs = probe.synth_result(tname, targs, ref_dir, result_index)
            if obs is None:
                break
            tcid = cur_msgs[-1]["tool_calls"][0]["id"]
            cur_msgs = cur_msgs + [{"role": "tool", "tool_call_id": tcid,
                                    "content": obs}]
        c = dict(steps[-1])
        if len(steps) > 1:
            c["steps"] = steps
        c["close_calls"] = first_cc
        samples.append(c)
        trail = " → ".join(s["cat"] for s in steps)
        ccs = f" close-calls={len(first_cc)}" if first_cc is not None else " (no logprobs)"
        print(f"[{args.mode}/{args.variant}] {i+1}/{args.k}: {trail}  {c.get('detail','')}{ccs}", flush=True)
        if first_cc:
            for x in first_cc[:8]:
                print(f"    pos={x['pos']:>3} {x['top1']!r} (p={x['p1']}) vs {x['top2']!r} (p={x['p2']}) margin={x['margin']}")

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"warmcold-{args.kind}-{args.mode}-{args.variant}.json")
    json.dump(samples, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")
    cats = {}
    for s in samples:
        cats[s["cat"]] = cats.get(s["cat"], 0) + 1
    print(f"[{args.mode}] summary: {cats}")


if __name__ == "__main__":
    main()
