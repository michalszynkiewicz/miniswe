#!/usr/bin/env python3
"""Tier-1 moment probe: K next-action samples per (moment x variant).

Variants are pure TEXT transforms on the captured context (no rebuilds):
  baseline         normalize to the ORIGINAL instruction strings
  desc             new plan-first refactor tool description
  next-line        add_param result "Next:" points at follow-up edits (#1)
  checkoff         neutral step check-off wording (#2)
  no-contradiction system prompt hand-edit ban qualified (#3)
  all              all of the above

Classification is generic — predicates derive from the moment's own context
(callsite files parsed from the recorded stub result), nothing task-specific.

Usage: tier1-probe.py <moments_dir> --types post-stub,stub-loop \
         --variants baseline,no-contradiction --k 8 [--url http://localhost:8464]
"""

import argparse
import glob
import json
import os
import re
import urllib.request
from collections import Counter

OLD_DESC_SNIPPET = "DO NOT enumerate or edit callsites yourself with replace_range/insert_at"
NEW_DESC = (
    "ATOMIC multi-file refactor — updates the definition AND every callsite in one call. "
    "Use it to add/remove a parameter (e.g. 'add a flag', 'extend signature with X') or rename "
    "a function/method/type/variable across the codebase — do NOT hand-edit callsites one by one "
    "with replace_range/insert_at for these tasks.\n\n"
    "add_param fills every existing callsite with the `callsite_fill_in` placeholder — a "
    "compile-correct STUB, not finished wiring. Any callsite on the path from where the real "
    "value originates must be edited afterwards to pass it. Workflow: (1) BEFORE calling, add a "
    "plan step for each callsite that must carry a real value; (2) call refactor; (3) execute "
    "those follow-up edits — the feature does not work until the value is threaded end to end.\n\n"
    "Actions: add_param, drop_param, rename; action='help' shows worked examples."
)
OLD_DESC = (
    "ATOMIC multi-file refactor — updates the definition AND every callsite in one call. "
    "Use this WHENEVER the task is:\n"
    "• adding a parameter (e.g. 'add a flag', 'add a context arg', 'extend signature with X')\n"
    "• removing a parameter\n"
    "• renaming a function, method, type, or variable across the codebase\n\n"
    "DO NOT enumerate or edit callsites yourself with replace_range/insert_at for these tasks — "
    "that's manual, error-prone, and the exact thing this tool exists to avoid. One refactor call "
    "is faster and atomic.\n\n"
    "Target is resolved by `name` via LSP, so exact line/column isn't needed. Actions: add_param, "
    "drop_param, rename. Use action='help' for parameter details and worked examples."
)

NEXT_OLD = "Next: run code(diagnostics) or your build to confirm the project still compiles."
NEXT_NEW = ("Next: make the follow-up edits — every callsite on the path from where the real "
            "value originates must pass it instead of the placeholder (see the list above). "
            "The feature does not work until then. Build after.")
# short-direct: user's minimal version — tool names, no loop prohibition.
NEXT_SHORT = ("Callsites were updated with placeholder values. Update the callsites that must "
              "pass a real value using insert_at or replace_range.")
# short-plus: user's minimal version + one-line loop prohibition.
NEXT_SHORT_PLUS = ("Callsites were updated with placeholder values. Update the callsites that must "
                   "pass a real value using insert_at or replace_range — do not call refactor "
                   "again for this.")
# short-v3: one sentence, all three ingredients (NOT-finished + names + prohibition).
NEXT_SHORT_V3 = ("The placeholder callsites are NOT finished — update each callsite that must "
                 "pass a real value using insert_at or replace_range (not another refactor call).")
# direct-edits: name the exact tools AND forbid the re-refactor loop.
NEXT_DIRECT = ("Next: the placeholder callsites above are NOT finished. For each callsite that "
               "must pass a real value: read it, then edit it DIRECTLY with replace_range or "
               "insert_at, replacing the placeholder argument with the real value. Do NOT call "
               "refactor again for this — the parameter already exists and add_param will be "
               "rejected; direct edits are the correct tool for this step.")

# ── ship cells: the to-ship COMBINATION (desc variant + direct-edits Next + scoped ban) ──
BAN_SCOPED = "do NOT hand-edit callsites for the signature change itself."
DESC_HEAD = (
    "ATOMIC multi-file refactor — updates the definition AND every callsite in one call. "
    "Use it to add/remove a parameter (e.g. 'add a flag', 'extend signature with X') or rename "
    "a function/method/type/variable across the codebase — do NOT hand-edit callsites one by one "
    "with replace_range/insert_at for these tasks.\n\n")
DESC_TAIL = "\n\nActions: add_param, drop_param, rename; action='help' shows worked examples."
MID_FULL = (
    "add_param fills every existing callsite with the `callsite_fill_in` placeholder — a "
    "compile-correct STUB, not finished wiring. Any callsite on the path from where the real "
    "value originates must be edited afterwards to pass it. Workflow: (1) BEFORE calling, add a "
    "plan step for each callsite that must carry a real value; (2) call refactor; (3) execute "
    "those follow-up edits — the feature does not work until the value is threaded end to end.")
MID_WORKFLOW = (
    "For add_param, `callsite_fill_in` is a placeholder — callsites that must pass a real "
    "value need a follow-up edit after the call. Workflow: (1) BEFORE calling, add a plan step "
    "for each such callsite; (2) call refactor; (3) execute those follow-up edits.")
MID_LEAN = (
    "For add_param, `callsite_fill_in` is a placeholder — callsites that must pass a real "
    "value need a follow-up edit after the call.")
SHIP_DESCS = {
    "ship-full": DESC_HEAD + MID_FULL + DESC_TAIL,
    "ship-workflow": DESC_HEAD + MID_WORKFLOW + DESC_TAIL,
    "ship-lean": DESC_HEAD + MID_LEAN + DESC_TAIL,
}

CHECKOFF_RE = re.compile(r"(checked off at round \d+) \(compile gate passed ✓\)")
CHECKOFF_NEW = r"\1 (compiles ✓ — but compiling alone does not complete a step: confirm the step's BEHAVIOR is actually done before moving on)"

BAN_OLD_RE = re.compile(
    r"do NOT hand-edit callsites yourself(, that is exactly what it exists to avoid)?\.")
BAN_NEW = ("do NOT hand-edit callsites for the mechanical signature change itself — but a "
           "callsite that must pass a REAL value (not the placeholder) DOES need your follow-up "
           "replace_range/insert_at edit afterwards.")


def transform(ctx, variant):
    """Return a deep-copied context with the variant's strings applied."""
    c = json.loads(json.dumps(ctx))

    def map_msgs(fn):
        for m in c["messages"]:
            if isinstance(m.get("content"), str):
                m["content"] = fn(m["content"])

    # normalize the refactor description first (corpus mixes old/new eras)
    for t in c["tools"]:
        if t.get("function", {}).get("name") == "refactor":
            t["function"]["description"] = OLD_DESC
    map_msgs(lambda s: s)  # no-op placeholder for symmetry

    if variant in ("desc", "all"):
        for t in c["tools"]:
            if t.get("function", {}).get("name") == "refactor":
                t["function"]["description"] = NEW_DESC
    if variant in ("next-line", "all"):
        map_msgs(lambda s: s.replace(NEXT_OLD, NEXT_NEW))
    if variant == "short-direct":
        map_msgs(lambda s: s.replace(NEXT_OLD, NEXT_SHORT))
    if variant == "short-plus":
        map_msgs(lambda s: s.replace(NEXT_OLD, NEXT_SHORT_PLUS))
    if variant == "short-v3":
        map_msgs(lambda s: s.replace(NEXT_OLD, NEXT_SHORT_V3))
    if variant in ("direct-edits", "direct-edits+ban"):
        map_msgs(lambda s: s.replace(NEXT_OLD, NEXT_DIRECT))
    if variant == "direct-edits+ban":
        map_msgs(lambda s: BAN_OLD_RE.sub(BAN_NEW, s))
    if variant in SHIP_DESCS:
        for t in c["tools"]:
            if t.get("function", {}).get("name") == "refactor":
                t["function"]["description"] = SHIP_DESCS[variant]
        map_msgs(lambda s: s.replace(NEXT_OLD, NEXT_DIRECT))
        map_msgs(lambda s: BAN_OLD_RE.sub(BAN_SCOPED, s))
    if variant in ("checkoff", "all"):
        map_msgs(lambda s: CHECKOFF_RE.sub(CHECKOFF_NEW, s))
    if variant in ("no-contradiction", "all"):
        map_msgs(lambda s: BAN_OLD_RE.sub(BAN_NEW, s))
    return c


def call(url, ctx, temperature):
    body = json.dumps({
        "model": ctx["model"], "messages": ctx["messages"], "tools": ctx["tools"],
        "temperature": temperature, "max_tokens": 1600, "stream": False,
    }).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        m = json.load(r)["choices"][0]["message"]
    tcs = m.get("tool_calls") or []
    if not tcs:
        return "(no-tool)", (m.get("content") or ""), len(m.get("content") or "")
    fn = tcs[0]["function"]
    return fn.get("name", "?"), str(fn.get("arguments", "")), len(str(fn.get("arguments", "")))


THREAD_WORDS = re.compile(r"pass|thread|wire|propagat|real value|actual value|callsite", re.I)


def classify(name, args, meta):
    """Generic action classes. callsite predicates come from the moment's own
    recorded stub report — nothing hardcoded to a task."""
    sites = meta.get("callsite_files") or []
    target = meta.get("refactor_target")
    path = ""
    mm = re.search(r'"path"\s*:\s*"([^"]+)"', args)
    if mm:
        path = mm.group(1)

    if name == "refactor":
        if target and f'"{target}"' in args and '"add_param"' in args:
            return "RE-REFACTOR-same-fn"          # the run5 loop
        return "refactor"
    if name in ("replace_range", "insert_at", "edit_file", "write_file"):
        return "EDIT-callsite" if any(path.endswith(s) or s.endswith(path) or path == s
                                      for s in sites) else f"edit-other"
    if name == "file":
        act = "read" if '"read"' in args or '"search"' in args else "file"
        return "READ-callsite" if any(path.endswith(s) or s.endswith(path) or path == s
                                      for s in sites) else f"{act}-other"
    if name == "plan":
        if '"check"' in args:
            return "plan-CHECKOFF"
        if '"set"' in args or '"refine"' in args:
            return "plan-write+THREAD" if THREAD_WORDS.search(args) else "plan-write"
        return "plan-other"
    if name in ("check", "code"):
        return "build/diag"
    if name == "(no-tool)":
        return "EXIT"
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("moments_dir")
    ap.add_argument("--types", default="post-stub,stub-loop")
    ap.add_argument("--variants", default="baseline,no-contradiction")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--url", default="http://localhost:8464")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--out", default=None, help="write raw samples jsonl here")
    args = ap.parse_args()

    types = args.types.split(",")
    variants = args.variants.split(",")
    moments = []
    for d in sorted(glob.glob(os.path.join(args.moments_dir, "*"))):
        meta_p = os.path.join(d, "meta.json")
        if not os.path.isfile(meta_p):
            continue
        meta = json.load(open(meta_p))
        if meta["type"] in types:
            moments.append((d, meta))
    print(f"{len(moments)} moments x {len(variants)} variants x K={args.k} "
          f"= {len(moments) * len(variants) * args.k} calls\n", flush=True)

    raw = open(args.out, "a") if args.out else None
    agg = {v: Counter() for v in variants}
    for d, meta in moments:
        ctx = json.load(open(os.path.join(d, "context.json")))
        name_short = os.path.basename(d).replace("gemma-4-26B-A4B-it-UD-Q4_K_M_", "")
        for v in variants:
            tctx = transform(ctx, v)
            picks = Counter()
            for _ in range(args.k):
                try:
                    n, a, ln = call(args.url, tctx, args.temperature)
                except Exception as e:  # server hiccup: count separately
                    picks["(err)"] += 1
                    continue
                cls = classify(n, a, meta)
                picks[cls] += 1
                agg[v][cls] += 1
                if raw:
                    raw.write(json.dumps({"moment": name_short, "variant": v,
                                          "class": cls, "tool": n, "args": a[:300],
                                          "resp_len": ln}) + "\n")
                    raw.flush()
            print(f"[{meta['type']}] {name_short[:52]:52s} {v:16s} {dict(picks)}", flush=True)

    print("\n=== AGGREGATE (per variant) ===")
    for v in variants:
        total = sum(agg[v].values()) or 1
        tops = ", ".join(f"{k}:{n} ({100 * n // total}%)" for k, n in agg[v].most_common(8))
        print(f"{v:16s} {tops}")


if __name__ == "__main__":
    main()
