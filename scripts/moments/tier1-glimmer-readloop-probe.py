#!/usr/bin/env python3
"""tier1 — Glimmer's terminal read loop: which intervention actually breaks it?

The moment: docker_20260826_163007 (G3, 6/6 in 3431s). Six of seven plan steps
are checked, the tree is GREEN, and the last step is "Add tests for override
behavior". From round 74 to round 144+ the model re-reads
`tests/e2e_context.rs:435-441` — the last 7 lines of the file, i.e. exactly
the insertion point it needs — and never emits the insert.

Everything the harness threw at it is in here as an arm, plus the two
mechanical arms it does not currently have. Scoring is on the FIRST tool call
of the reply:

  EDIT   insert_at / replace_range / write_file / edit_file / refactor  -> escape
  READ   file(action=read)                                             -> loop
  OTHER  any other tool                                                -> partial
  STALL  no tool call at all                                           -> the
         "Understood." pathology (32 of 188 rounds in the live run)

Arms are run grouped so llama.cpp keeps the ~40k-token prefix cached; the
tail-only arms cost one re-prefill each, the tool-list arms cost one per arm
(the tools block sits at the front of the prompt).

Usage:
  python3 tier1-glimmer-readloop-probe.py [--n 8] [--moments 101,130]
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN = os.path.join(
    REPO,
    "benchmark_results/docker_20260826_163007__home_michal_models_Muse-Glimmer-30B-GGU",
    "00_baseline/llm_dumps",
)
ENDPOINT = os.environ.get("LLAMA_ENDPOINT", "http://localhost:8464")

EDIT_TOOLS = {"insert_at", "replace_range", "write_file", "edit_file", "refactor"}

# ── what the harness actually sends today ────────────────────────────────
NUDGE = (
    "You just made this same read/inspection call 3 times in a row. The result "
    "hasn't changed. What specifically are you looking for? Try a narrower "
    "search, a different range, or move on to making an edit."
)
ESCALATION = (
    "You made this same read/inspection call 3 times in a row AGAIN — the result "
    "STILL has not changed, and the answer is not in this output. Older history "
    "has been compacted so you can re-approach. Re-orient from the plan and "
    "current state, then take a DIFFERENT action now: make an edit, run a check "
    "or validator, or consult the docs."
)
STUCK_NOTE = (
    "[stuck-check] 18+ rounds and 4+ minutes with NO change in the compiler/test "
    "signal. You have re-read tests/e2e_context.rs 3+ times in a row; the answer "
    "is not in it — do NOT read it again. Take ONE concrete action now: make the "
    "smallest edit that advances the first unchecked plan step."
)
# ── candidate: prune the repeats out of the PREFIX, then say so ─────────
PRUNE_NOTE = (
    "[pruned] Repeat reads of tests/e2e_context.rs were dropped from this "
    "conversation. You have the file tail above. Write the edit now."
)


def prune_repeats(msgs, keep_first=1):
    """Drop every (assistant read-call, tool result) pair for the looping read
    except the first `keep_first` and the LAST one.

    This is the surgical opposite of what `force_compress` does today:
    compaction summarizes OLD history, but the loop lives in the NEWEST
    messages, so each forced compaction leaves the repeats untouched and
    RAISES their share of the prompt (7% -> 24% between dump 101 and 130).
    """
    def is_loop_call(m):
        tcs = m.get("tool_calls") or []
        if not tcs or m["role"] != "assistant":
            return False
        fn = tcs[0]["function"]
        if fn["name"] != "file":
            return False
        try:
            a = json.loads(fn["arguments"])
        except Exception:
            return False
        return a.get("action") == "read" and "e2e_context.rs" in (a.get("path") or "")

    idxs = [i for i, m in enumerate(msgs) if is_loop_call(m)]
    if len(idxs) <= keep_first + 1:
        return msgs
    drop = set()
    for i in idxs[keep_first:-1]:
        drop.add(i)
        if i + 1 < len(msgs) and msgs[i + 1]["role"] == "tool":
            drop.add(i + 1)
    return [m for j, m in enumerate(msgs) if j not in drop]


def call(payload, timeout=900):
    req = urllib.request.Request(
        f"{ENDPOINT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def load(idx):
    f = sorted(os.listdir(RUN))[idx]
    d = json.load(open(os.path.join(RUN, f)))
    assert d.get("tools"), f"dump {idx} has no tools (sub-model call?)"
    return d


def last_tool_msg(msgs):
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i]["role"] == "tool":
            return i
    return None


def load_rust(idx, rust_dir):
    """Messages as the SHIPPED Rust pruner rewrote them.

    Generated by `tests/prune_reads_realdata.rs` with PRUNE_OUT set, so this
    arm replays `agent::prune_reads` itself — note wording, MIN_REPEATS, guard
    exemption and all — rather than the Python sketch in `prune_repeats`.
    """
    name = sorted(os.listdir(RUN))[idx]
    path = os.path.join(rust_dir, "pruned-" + name.rsplit("-", 1)[-1])
    if not os.path.exists(path):
        raise SystemExit(
            f"missing {path} — regenerate with:\n"
            f"  PRUNE_FIXTURE={os.path.join(RUN, name)} PRUNE_OUT={path} "
            f"cargo test --test prune_reads_realdata"
        )
    return json.load(open(path))["messages"]


def build(base, arm, idx=None, rust_dir=None):
    """Return (messages, tools) for one arm."""
    msgs = [dict(m) for m in base["messages"]]
    tools = base["tools"]
    i = last_tool_msg(msgs)
    assert i is not None, "moment has no tool result to attach to"

    if arm == "control":
        pass
    elif arm == "nudge":  # harness replaces the read result with the nudge
        msgs[i] = dict(msgs[i], content=NUDGE)
    elif arm == "escalate":
        msgs[i] = dict(msgs[i], content=ESCALATION)
    elif arm == "stuck":  # harness appends the note to the result
        msgs[i] = dict(msgs[i], content=(msgs[i]["content"] or "") + "\n" + STUCK_NOTE)
    elif arm == "prune":  # mechanical: the repeats leave the prompt prefix
        msgs = prune_repeats(msgs)
    elif arm == "rust":  # the shipped pruner's own output
        msgs = load_rust(idx, rust_dir)
    elif arm == "rust+escalate":  # what the live harness actually does on the
        # second loop detection: the pruner runs every round, and the escalation
        # replaces the read result. Neither lever is tested by the other arms.
        msgs = load_rust(idx, rust_dir)
        j = last_tool_msg(msgs)
        msgs[j] = dict(msgs[j], content=ESCALATION)
    elif arm == "prune+note":
        msgs = prune_repeats(msgs)
        j = last_tool_msg(msgs)
        msgs[j] = dict(msgs[j], content=(msgs[j]["content"] or "") + "\n" + PRUNE_NOTE)
    else:
        raise SystemExit(f"unknown arm {arm}")
    return msgs, tools


def classify(resp):
    m = resp["choices"][0]["message"]
    tcs = m.get("tool_calls") or []
    if not tcs:
        return "STALL", (m.get("content") or "").strip()[:80]
    name = tcs[0]["function"]["name"]
    try:
        args = json.loads(tcs[0]["function"]["arguments"])
    except Exception:
        args = {}
    if name in EDIT_TOOLS:
        return "EDIT", f"{name}({args.get('path','')}:{args.get('line',args.get('start',''))})"
    if name == "file" and args.get("action") == "read":
        return "READ", f"read {args.get('path','')}:{args.get('start','')}-{args.get('end','')}"
    return "OTHER", f"{name}({args.get('action','')})"


ARMS = ["control", "nudge", "escalate", "stuck", "prune", "prune+note", "rust", "rust+escalate"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--moments", default="101,130")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument(
        "--rust-dir",
        default=os.environ.get("PRUNE_RUST_DIR", ""),
        help="directory of PRUNE_OUT dumps for the `rust` arm",
    )
    args = ap.parse_args()

    moments = [int(x) for x in args.moments.split(",")]
    arms = args.arms.split(",")
    results = {}

    for mi in moments:
        base = load(mi)
        for arm in arms:
            msgs, tools = build(base, arm, idx=mi, rust_dir=args.rust_dir)
            key = (mi, arm)
            results[key] = []
            t0 = time.time()
            for k in range(args.n):
                payload = {
                    "model": base["model"],
                    "messages": msgs,
                    "tools": tools,
                    "temperature": base.get("temperature", 0.2),
                    "max_tokens": args.max_tokens,
                    "stream": False,
                    "chat_template_kwargs": base.get("chat_template_kwargs", {}),
                }
                try:
                    r = call(payload)
                    verdict, detail = classify(r)
                except Exception as e:  # noqa: BLE001
                    verdict, detail = "ERR", str(e)[:80]
                results[key].append((verdict, detail))
                print(f"  m{mi} {arm:<13} {k+1}/{args.n}  {verdict:<6} {detail}", flush=True)
            c = Counter(v for v, _ in results[key])
            print(
                f"m{mi} {arm:<13} EDIT={c['EDIT']} READ={c['READ']} "
                f"OTHER={c['OTHER']} STALL={c['STALL']} ERR={c['ERR']} "
                f"({time.time()-t0:.0f}s)",
                flush=True,
            )

    print("\n" + "=" * 74)
    header = f"{'arm':<14}" + "".join(f"{'m' + str(m):>20}" for m in moments)
    print(header + f"{'TOTAL escape':>16}")
    for arm in arms:
        row = f"{arm:<14}"
        esc = 0
        tot = 0
        for mi in moments:
            c = Counter(v for v, _ in results.get((mi, arm), []))
            cell = "E%d/R%d/O%d/S%d" % (c["EDIT"], c["READ"], c["OTHER"], c["STALL"])
            row += f"{cell:>20}"
            esc += c["EDIT"]
            tot += sum(c.values())
        row += f"{f'{esc}/{tot}':>16}"
        print(row)
    print("E=edit (escape)  R=read (loop)  O=other tool  S=stall (no tool call)")


if __name__ == "__main__":
    main()
