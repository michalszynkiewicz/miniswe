#!/usr/bin/env python3
"""Extract decision-point "moments" from real bench-run llm_dumps.

A moment = a captured request context cut at an interesting decision point,
plus generic metadata for scoring. Detectors are PATTERN-based (never
run-specific) so the corpus grows with every future bench run.

Moment types:
  post-stub      last msg is an add_param stub result ("compile-correct STUB")
  stub-loop      last msg is the add_param duplicate-guard refusal
  step-checkoff  last msg is a plan check-off with "(compile gate passed)"
  sig-decision   context cut right before the run's FIRST refactor(add_param)
                 response (regression metric: is refactor still chosen?)

Usage: extract-moments.py <bench_results_glob>... -o <out_dir> [--cap-per-type N]
"""

import argparse
import glob
import json
import os
import re
import sys

STUB_PAT = re.compile(r"compile-correct STUB|now passes `")
DUP_PAT = re.compile(r"already has a parameter named")
CHECKOFF_PAT = re.compile(r"checked off at round \d+ \(compile gate passed")
CALLSITE_LINE = re.compile(r"•\s+(\S+):(\d+) now passes")


def content_of(msg):
    c = msg.get("content")
    return c if isinstance(c, str) else (json.dumps(c) if c else "")


def load_dumps(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "llm_dumps", "*.json")))
    for f in files:
        try:
            yield f, json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue


def callsite_files_in(msgs):
    """Files named in the most recent add_param callsite report in context."""
    for m in reversed(msgs):
        c = content_of(m)
        if STUB_PAT.search(c):
            return sorted({p for p, _ in CALLSITE_LINE.findall(c)})
    return []


def refactor_target_in(msgs):
    """Function name of the most recent add_param call in context."""
    for m in reversed(msgs):
        for t in m.get("tool_calls") or []:
            a = str(t.get("function", {}).get("arguments", ""))
            if '"add_param"' in a:
                mm = re.search(r'"name"\s*:\s*"(\w+)"', a)
                return mm.group(1) if mm else None
    return None


def save_moment(out_dir, mtype, run_tag, seq, dump, cut, extra):
    mdir = os.path.join(out_dir, f"{mtype}__{run_tag}__{seq}")
    os.makedirs(mdir, exist_ok=True)
    msgs = dump["messages"][:cut] if cut is not None else dump["messages"]
    ctx = {"model": dump["model"], "messages": msgs, "tools": dump["tools"]}
    json.dump(ctx, open(os.path.join(mdir, "context.json"), "w"))
    meta = {
        "type": mtype,
        "source_run": run_tag,
        "n_messages": len(msgs),
        "callsite_files": callsite_files_in(msgs),
        "refactor_target": refactor_target_in(msgs),
    }
    meta.update(extra)
    json.dump(meta, open(os.path.join(mdir, "meta.json"), "w"), indent=1)
    return mdir


def extract_run(run_dir, run_tag, out_dir, seen_counts, cap):
    found = []
    got_sig = False
    got = set()  # one moment per type per run: first occurrence wins
    for f, dump in load_dumps(run_dir):
        msgs = dump.get("messages", [])
        if not msgs or "tools" not in dump:
            continue
        last = content_of(msgs[-1])

        def want(mtype):
            return (
                mtype not in got
                and seen_counts.get(mtype, 0) < cap
            )

        if want("post-stub") and msgs[-1].get("role") == "tool" and STUB_PAT.search(last) \
                and "already has a parameter" not in last:
            found.append(save_moment(out_dir, "post-stub", run_tag,
                                     seen_counts.get("post-stub", 0), dump, None,
                                     {"src": os.path.basename(f)}))
            got.add("post-stub")
            seen_counts["post-stub"] = seen_counts.get("post-stub", 0) + 1

        if want("stub-loop") and msgs[-1].get("role") == "tool" and DUP_PAT.search(last):
            found.append(save_moment(out_dir, "stub-loop", run_tag,
                                     seen_counts.get("stub-loop", 0), dump, None,
                                     {"src": os.path.basename(f)}))
            got.add("stub-loop")
            seen_counts["stub-loop"] = seen_counts.get("stub-loop", 0) + 1

        if want("step-checkoff") and msgs[-1].get("role") == "tool" and CHECKOFF_PAT.search(last):
            found.append(save_moment(out_dir, "step-checkoff", run_tag,
                                     seen_counts.get("step-checkoff", 0), dump, None,
                                     {"src": os.path.basename(f)}))
            got.add("step-checkoff")
            seen_counts["step-checkoff"] = seen_counts.get("step-checkoff", 0) + 1

        if not got_sig and want("sig-decision"):
            # cut right before the assistant msg holding the FIRST add_param call
            for i, m in enumerate(msgs):
                if m.get("role") == "assistant" and any(
                    '"add_param"' in str(t.get("function", {}).get("arguments", ""))
                    for t in (m.get("tool_calls") or [])
                ):
                    found.append(save_moment(out_dir, "sig-decision", run_tag,
                                             seen_counts.get("sig-decision", 0), dump, i,
                                             {"src": os.path.basename(f)}))
                    got.add("sig-decision")
                    seen_counts["sig-decision"] = seen_counts.get("sig-decision", 0) + 1
                    got_sig = True
                    break
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("globs", nargs="+", help="bench run dirs (each containing llm_dumps/)")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--cap-per-type", type=int, default=10)
    args = ap.parse_args()

    run_dirs = []
    for g in args.globs:
        for d in sorted(glob.glob(g)):
            if os.path.isdir(os.path.join(d, "llm_dumps")):
                run_dirs.append(d)
    if not run_dirs:
        sys.exit("no run dirs with llm_dumps/ matched")

    os.makedirs(args.out, exist_ok=True)
    seen = {}
    total = []
    for d in run_dirs:
        # tag: <matrix>/<arm>/<run>
        parts = os.path.normpath(d).split(os.sep)
        tag = "_".join(parts[-3:]).replace("replaymatrix_", "m")
        total += extract_run(d, tag, args.out, seen, args.cap_per_type)

    print(f"extracted {len(total)} moments from {len(run_dirs)} runs:")
    for t, n in sorted(seen.items()):
        print(f"  {t}: {n}")


if __name__ == "__main__":
    main()
