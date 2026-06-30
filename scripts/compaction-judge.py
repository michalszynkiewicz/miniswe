#!/usr/bin/env python3
"""compaction-judge.py — score retention of injected needles in dumped contexts.

Reads the per-strategy compacted contexts dumped by the offline probe
(`probe_ctx_<Strategy>.txt`) and, for each planted needle, reports BOTH:
  - exact : the exact token still appears verbatim in the context
  - semantic : an LLM judge, given ONLY that context, can still answer the
               question (fair to summaries, which paraphrase)

Usage:
  LLAMA_ENDPOINT=http://localhost:8464 scripts/compaction-judge.py <dump_dir>

The needles must match what scripts/replay/reconstruct-trajectory.py injected.
"""
import glob
import json
import os
import sys
import urllib.request

ENDPOINT = os.environ.get("LLAMA_ENDPOINT", "http://localhost:8464")
MODEL = os.environ.get("PROBE_MODEL", "gemma-4-26B-A4B-it")

# (label, exact_substring, question, answer_keywords). Unique tokens that can't
# appear in real content or be guessed; 2 verbatim + 2 semantic. Must match
# scripts/replay/reconstruct-trajectory.py.
NEEDLES = [
    ("sentinel", "GORP_5521",
     "What literal token does the integration test pin?", ["GORP_5521"]),
    ("variable", "ovr_slot_88",
     "What is the name of the local variable that carries the override inside assemble()?",
     ["ovr_slot_88"]),
    ("scratchpad", "DELETE the .miniswe/scratchpad.md",
     "When the override is active, what must the agent do to the scratchpad file?",
     ["delete", "remove"]),
    ("loglevel", "logged at WARN level",
     "At what log level must the chosen override text be logged on startup?", ["warn"]),
]


def ask(context, question):
    body = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Answer the QUESTION using ONLY the CONTEXT. "
             "Reply with just the value/phrase from the context, nothing else. "
             "If it is not present in the context, reply with exactly: UNKNOWN."},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}\nAnswer:"},
        ],
        # Suppress gemma's reasoning (#39) so `content` isn't eaten by the
        # reasoning budget — the compressor relies on this same kwarg.
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": 512, "temperature": 0,
    }).encode()
    req = urllib.request.Request(f"{ENDPOINT}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
    return (d["choices"][0]["message"].get("content") or "").strip()


def main():
    dump_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    files = sorted(glob.glob(os.path.join(dump_dir, "probe_ctx_*.txt")))
    if not files:
        sys.exit(f"no probe_ctx_*.txt in {dump_dir}")
    print(f"\n=== Retention judge ({len(NEEDLES)} needles, endpoint={ENDPOINT}) ===")
    summary = {}
    for f in files:
        strat = os.path.basename(f)[len("probe_ctx_"):-len(".txt")]
        ctx = open(f, errors="replace").read()
        exact_hits = semantic_hits = 0
        rows = []
        for label, token, q, keys in NEEDLES:
            exact = token.lower() in ctx.lower()
            ans = ask(ctx, q)
            up = ans.upper()
            semantic = ("UNKNOWN" not in up) and any(k.lower() in ans.lower() for k in keys)
            exact_hits += exact
            semantic_hits += semantic
            rows.append(f"    {label:<11} exact={'Y' if exact else '.'} "
                        f"semantic={'Y' if semantic else '.'}  ans={ans[:60]!r}")
        summary[strat] = (exact_hits, semantic_hits)
        print(f"\n{strat}:  exact={exact_hits}/{len(NEEDLES)}  semantic={semantic_hits}/{len(NEEDLES)}")
        print("\n".join(rows))
    print("\n=== summary (needle retention) ===")
    print(f"{'strategy':<22} {'exact':>7} {'semantic':>9}")
    for strat, (e, s) in summary.items():
        print(f"{strat:<22} {e:>5}/{len(NEEDLES)} {s:>7}/{len(NEEDLES)}")
    print()


if __name__ == "__main__":
    main()
