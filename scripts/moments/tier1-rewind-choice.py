#!/usr/bin/env python3
"""tier1-rewind-choice — v2 of the rewind probe. v1 asked the debugger to
notice an abandoned-clean revision AND name it: 0/24 hits, plus 7/24
malformed responses. This version does the noticing/naming MECHANICALLY
(same anchor-selection logic as the proven `auto_revert_ast_cascade` guard,
extended from "ast broken streak" to "file_errors climbed past a margin"),
and asks the model a narrowed 3-way judgment call: RESET (whole tree) /
REWIND (accept the proposed single-file revert) / CONTINUE (fix forward,
ignore the proposal).

Candidate rule per changed file (mirrors auto_revert.rs's anchor logic):
  latest non-initial, ast_ok revision N such that file_errors(N) <= 1 AND
  file_errors(current) - file_errors(N) >= 3. Same two ground-truth moments
  as tier1-rewind.py; here only ONE candidate survives the rule per moment
  (run.rs rev_4 for postfix_run2, main.rs rev_2 for prefix_run3) — the other
  changed files in each moment correctly produce no candidate.
"""

import argparse
import json
import os
import re
import urllib.request

REPO = "/home/michal/dev/miniswe"
ENDPOINT = os.environ.get("LLAMA_ENDPOINT", "http://localhost:8464")

CHOICE_PROMPT = """You are a READ-ONLY analyst with fresh eyes on a STUCK coding task. You have ONLY read/search/inspect tools — you CANNOT edit files, run shell, set a plan, or use a scratchpad. Do NOT plan and do NOT try to edit.
Investigate the failure and the changes made so far. A mechanical scan of the edit history has already found ONE candidate: a specific file that regressed from a much cleaner earlier revision to its current, more broken state (shown below, under CANDIDATE REWIND POINT). You do not need to find it yourself — decide whether taking it is the right move.
Choose EXACTLY ONE:
(a) RESET — the damage is not limited to that one file; the whole attempt is misdirected or damaged everywhere, and reverting the ENTIRE tree to the clean original and starting fresh would be faster and more reliable. IGNORE effort already spent.
(b) REWIND — the candidate is correct: reverting JUST that file to the proposed revision recovers real progress, and the rest of the tree (outside that file) is fine or close to fine as it stands.
(c) CONTINUE — no revert is needed anywhere; the current state (including that file as-is) is on the path to the GOAL and only a focused forward fix remains.
Output your choice on the FIRST line, exactly one of:
CHOICE: (a)
CHOICE: (b)
CHOICE: (c)
Then one line — REASON: <the single most important reason>.
If (c): also add — FIX: <what must change, conceptually> and PLAN: <the concrete remaining steps>."""

MOMENTS = {
    "postfix_run2": {
        "dump": "/home/michal/dev/miniswe/benchmark_results/replaymatrix_20260703_000135_gemma-4-26B-A4B-it-UD-Q4_K_M/judge_mf/run2/llm_dumps/req-1783030593-00052-000181.json",
        "candidate": "CANDIDATE REWIND POINT: src/cli/commands/run.rs rev_4 (ast=ok, file_errors=1) — the file's CURRENT state (rev_5, ast=ok, file_errors=31) is a large regression from rev_4, reached via a 75-line replace_range that was never reverted.",
        "expected": "b",
    },
    "prefix_run3": {
        "dump": "/home/michal/dev/miniswe/benchmark_results/replaymatrix_20260702_224555_unknown/judge_mf/run3/llm_dumps/req-1783027248-00052-000110.json",
        "candidate": "CANDIDATE REWIND POINT: src/main.rs rev_2 (ast=ok, file_errors=0) — the file's CURRENT state (rev_5, ast=broken, file_errors=3) is a regression from rev_2, reached via two subsequent edits that were never reverted.",
        "expected": "b",
    },
}

CHOICE_RE = re.compile(r"CHOICE:\s*\(?([abc])\)?", re.IGNORECASE)


def build_messages(moment_key):
    m = MOMENTS[moment_key]
    dump = json.load(open(m["dump"]))
    user = dump["messages"][1]["content"] + f"\n\n=== {m['candidate']} ==="
    return [{"role": "system", "content": CHOICE_PROMPT}, {"role": "user", "content": user}], dump


def call_llm(messages, model, tools, temperature, max_tokens, timeout=180):
    payload = {
        "model": model, "messages": messages, "tools": tools,
        "temperature": temperature, "max_tokens": max_tokens, "stream": False,
    }
    req = urllib.request.Request(
        f"{ENDPOINT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def classify(text):
    m = CHOICE_RE.search(text)
    if not m:
        return {"choice": "NONE", "detail": text[:120].replace("\n", " ")}
    return {"choice": m.group(1).lower(), "detail": text[:150].replace("\n", " ")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--moments", default="postfix_run2,prefix_run3")
    ap.add_argument("--out", default=os.path.join(REPO, "benchmark_results/_moments/rewind-choice-v1"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    results = {}
    for mk in args.moments.split(","):
        messages, dump = build_messages(mk)
        samples = []
        for i in range(args.k):
            try:
                resp = call_llm(messages, dump["model"], dump.get("tools", []),
                                 dump.get("temperature", 0.2), dump.get("max_tokens", 8000))
                text = resp["choices"][0]["message"].get("content") or ""
                c = classify(text)
            except Exception as e:
                c = {"choice": "ERROR", "detail": str(e)[:100]}
            samples.append(c)
            expected = MOMENTS[mk]["expected"]
            hit = "HIT" if c["choice"] == expected else ""
            print(f"[{mk}] {i+1}/{args.k}: {c['choice']} {hit}  {c.get('detail','')}", flush=True)
        results[mk] = samples
        json.dump(results, open(os.path.join(args.out, "results.json"), "w"), indent=1)

    print("\n=== SUMMARY ===")
    for mk, samples in results.items():
        from collections import Counter
        cnt = Counter(s["choice"] for s in samples)
        print(f"{mk} (expected={MOMENTS[mk]['expected']}): {dict(cnt)}")


if __name__ == "__main__":
    main()
