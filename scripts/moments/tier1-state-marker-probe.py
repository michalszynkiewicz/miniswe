#!/usr/bin/env python3
"""tier1-state-marker-probe — under a never-move state block, does a versioned
marker keep the model on the NEWEST copy?

Never-move stops relocating the `[CURRENT STATE]` block (the relocation is what
blows the KV cache past the model's sliding window). The cost is that stale
copies accumulate in the transcript until compaction sweeps them. Every copy is
literally labelled "[CURRENT STATE]", which is false for all but the last.

This probes the rename. The transcript is the real never-move shape: N copies
attached to successive tool messages, each copy one plan-step further along,
each step naming a UNIQUE file. Whichever file the model's next tool call names
identifies which copy it acted on.

Arms (only the marker text differs; plan bytes are identical):
  single   ONE copy (the newest) — the ceiling. This is what the shipped
           size-conditional policy and sticky-move produce. Same transcript
           length, stale blocks simply omitted.
  control  every copy "[CURRENT STATE]"           <- never-move as it stands today
  A        "[STATE v3]"                            <- ordinal only
  B        "[STATE v3 - replaces v1-v2]"           <- ordinal + explicit kill
  D        "[STATE]" + "update 3, replaces updates 1-2" on the next line
           <- same information as B, one line lower; keeps the marker constant
              so find/strip needs no prefix search

Read `single` first: if control already matches it, stale copies cost nothing
at that depth and the rename is cosmetic. The gap between single and control is
the damage; the gap an arm closes is its value.

Usage: tier1-state-marker-probe.py [--k 12] [--depths 3,12]
"""

import argparse
import json
import os
import re
import sys
import urllib.request
from collections import Counter

REPO = "/home/michal/dev/miniswe"
ENDPOINT = os.environ.get("LLAMA_ENDPOINT", "http://localhost:8464")
DUMPS = os.path.join(
    REPO,
    "benchmark_results/docker_20260825_152440__home_michal_models_Laguna-XS-2.1-GGUF_L",
    "00_baseline/llm_dumps",
)

# Unique module names — no name is a substring of another, so classification
# by filename is exact.
MODULES = [
    "writer", "adapter", "stream_sink", "codec", "registry", "dispatch",
    "envelope", "framing", "retry_gate", "backoff", "telemetry", "cursor",
    "batcher", "flusher",
]
NSTEPS = len(MODULES)

TASK = (
    "Add an --emit-format flag that selects the output encoding (json|msgpack) and "
    "thread it through the whole emit pipeline so every sink honours it. "
    "[Before making changes, explore the codebase and use the plan tool to outline your "
    "approach. Each step has compile: true (default) — the compiler must pass to check it off.]"
)


def step_line(i, done, round_no):
    """One plan line. i is 1-based."""
    mod = MODULES[i - 1]
    body = f"Thread the `--emit-format` selector through `src/emit/{mod}.rs` [compile]"
    return f"- [x] (round {round_no}) {body}" if done else f"- [ ] {body}"


REWRITE = False


def plan_text_rewrite(version):
    """Copy `version` is a self-contained plan revision. Every copy has the
    same shape and the same number of checked steps, so nothing in the CONTENT
    says which is newer — the marker (and position) is the only signal."""
    mod = MODULES[version - 1]
    return (
        f"Revised approach: route the selector through the `{mod}` layer.\n"
        f"- [x] Audit `src/emit/{mod}.rs` for the encode seam [compile]\n"
        f"- [ ] Thread the `--emit-format` selector through `src/emit/{mod}.rs` [compile]\n"
        f"- [ ] Add the format arg to the `{mod}` constructor [compile]\n"
        f"- [ ] Update the `{mod}` unit tests [compile]"
    )


def plan_text(version):
    """Plan as of copy `version`: steps 1..version-1 checked, step `version` next."""
    return "\n".join(
        step_line(i, i < version, 4 + 3 * i) for i in range(1, NSTEPS + 1)
    )


def marker(arm, version):
    if arm in ("control", "single"):
        return "[CURRENT STATE]"
    if arm == "A":
        return f"[STATE v{version}]"
    if arm == "B":
        if version == 1:
            return "[STATE v1]"
        prior = "v1" if version == 2 else f"v1-v{version - 1}"
        return f"[STATE v{version} - replaces {prior}]"
    if arm == "D":
        if version == 1:
            return "[STATE]\nupdate 1"
        prior = "update 1" if version == 2 else f"updates 1-{version - 1}"
        return f"[STATE]\nupdate {version}, replaces {prior}"
    if arm == "E":
        # Imperative claim of authority. Note EVERY copy claims it — we cannot
        # edit older copies, so this tests whether the claim survives repetition.
        if version == 1:
            return "[STATE v1]"
        return (f"[STATE v{version} - AUTHORITATIVE. Ignore every earlier STATE "
                f"block above; they are superseded.]")
    if arm == "F":
        # States the SELECTION RULE instead of claiming to be the latest. This
        # sentence is true in every copy and points at the max.
        return (f"[STATE v{version} - follow ONLY the highest-numbered STATE "
                f"block in this conversation]")
    raise ValueError(arm)


def block(arm, version):
    body = plan_text_rewrite(version) if REWRITE else plan_text(version)
    return f"\n\n{marker(arm, version)}\n[PLAN]\n{body}\n"


def build(system, arm, depth, tail=0):
    """Never-move transcript: `depth` copies on successive tool messages,
    then `tail` block-free messages so the newest copy sits buried in history
    the way it does in real prompts (run 11: block at msg 47 of 148)."""
    msgs = [system, {"role": "user", "content": TASK}]

    def push(call_name, call_args, result, version):
        cid = f"c{version}"
        msgs.append({
            "role": "assistant",
            "tool_calls": [{
                "id": cid, "type": "function",
                "function": {"name": call_name, "arguments": json.dumps(call_args)},
            }],
        })
        # `single` carries a block only on the newest tool message.
        carry = (version == depth) if arm == "single" else True
        msgs.append({
            "role": "tool", "tool_call_id": cid,
            "content": result + (block(arm, version) if carry else ""),
        })

    # Round 1: exploration, then the first plan lands.
    push("file", {"action": "read", "path": "src/emit/mod.rs"},
         "pub mod writer;\npub mod adapter;\npub mod stream_sink;\npub mod codec;\n"
         "pub mod registry;\npub mod dispatch;\n", 1)
    # Rounds 2..depth: each completes the previous step, so the new copy shows
    # it checked off.
    for v in range(2, depth + 1):
        if REWRITE:
            push("file",
                 {"action": "read", "path": "docs/emit-design.md",
                  "start": 1 + 30 * v, "end": 30 + 30 * v},
                 "The encoding seam is chosen at construction time; each sink\n"
                 "receives an already-configured codec handle.\n", v)
        else:
            done = MODULES[v - 2]
            push("file",
                 {"action": "replace_range", "path": f"src/emit/{done}.rs",
                  "start": 40, "end": 52},
                 f"[file: wrote src/emit/{done}.rs, 12 lines replaced]\n"
                 "[check: cargo check ok, 0 errors]", v)
    # Block-free filler: real work that names none of MODULES, so it neither
    # leaks progress nor implies a next step. Applied identically to every arm.
    for j in range(tail):
        cid = f"t{j}"
        if j % 2:
            name, cargs = "check", {}
            res = "[check: cargo check ok, 0 errors]"
        else:
            name = "file"
            cargs = {"action": "read", "path": "tests/emit_roundtrip.rs",
                     "start": 1 + 20 * j, "end": 20 + 20 * j}
            res = (f"    assert_eq!(decode(encode(&case_{j}))?, case_{j});\n"
                   f"    // roundtrip case {j}\n")
        msgs.append({"role": "assistant", "tool_calls": [{
            "id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(cargs)}}]})
        msgs.append({"role": "tool", "tool_call_id": cid, "content": res})
    return msgs


ASK = ("Which single step of the plan is next? Answer with ONLY the file path "
       "from that step, e.g. src/emit/foo.rs — no other words.")


def classify_text(resp, depth):
    m = resp["choices"][0]["message"]
    txt = (m.get("content") or "")
    tcs = m.get("tool_calls") or []
    if not txt and tcs:
        txt = json.dumps(tcs[0]["function"].get("arguments", ""))
    hits = [i + 1 for i, mod in enumerate(MODULES) if f"emit/{mod}.rs" in txt]
    if not hits:
        return f"NO_PATH({txt[:50]!r})"
    v = hits[0]
    if v == depth:
        return "NEWEST"
    return f"STALE(v{v})" if v < depth else f"AHEAD(v{v})"


def call_llm(payload, timeout=300):
    req = urllib.request.Request(
        f"{ENDPOINT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def classify(resp, depth):
    m = resp["choices"][0]["message"]
    tcs = m.get("tool_calls") or []
    if not tcs:
        txt = (m.get("content") or "")[:60].replace("\n", " ")
        return f"NO_TOOL({txt})"
    fn = tcs[0]["function"]
    name = fn.get("name", "")
    args = fn.get("arguments", "") or ""
    if not isinstance(args, str):
        args = json.dumps(args)
    hits = [i + 1 for i, mod in enumerate(MODULES) if f"emit/{mod}.rs" in args]
    if hits:
        v = hits[0]
        if v == depth:
            return "NEWEST"
        if v < depth:
            return f"STALE(v{v})"
        return f"AHEAD(v{v})"
    if name == "plan":
        return "PLAN_CALL"
    blob = args.lower()
    if any(w in blob for w in ("find ", "ls ", "grep", "rg ", "tree", '"search"', '"list"')):
        return "EXPLORE"
    if name == "file" and '"read"' in blob.replace(" ", ""):
        return "EXPLORE"
    return f"OTHER({name})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--depths", default="3,12")
    ap.add_argument("--arms", default="single,control,A,B,D")
    ap.add_argument("--rotate", type=int, default=0,
                    help="rotate MODULES left by N — controls for the newest "
                         "copy's module name being intrinsically plausible")
    ap.add_argument("--rewrite", action="store_true",
                    help="each copy is a full plan revision with no intrinsic "
                         "recency cue (marker is the only signal)")
    ap.add_argument("--ask", action="store_true",
                    help="forced-choice readout: ask which step is next, in text")
    ap.add_argument("--tail", type=int, default=0,
                    help="block-free messages appended after the newest copy")
    args = ap.parse_args()

    fs = sorted(os.listdir(DUMPS), key=lambda f: os.path.getmtime(os.path.join(DUMPS, f)))
    d = json.load(open(os.path.join(DUMPS, fs[0])))
    system, tools = d["messages"][0], d["tools"]
    model, temp = d["model"], d.get("temperature", 0.2)

    global REWRITE, MODULES
    REWRITE = args.rewrite
    if args.rotate:
        r = args.rotate % len(MODULES)
        MODULES = MODULES[r:] + MODULES[:r]
        print(f"[rotate {args.rotate}] newest module at depth 12 = {MODULES[11]}; "
              f"v1 = {MODULES[0]}")
    depths = [int(x) for x in args.depths.split(",")]
    arms = args.arms.split(",")
    table = {}
    for depth in depths:
        if depth > NSTEPS:
            sys.exit(f"depth {depth} > {NSTEPS} plan steps")
        for arm in arms:
            msgs = build(system, arm, depth, args.tail)
            if args.ask:
                msgs = msgs + [{"role": "user", "content": ASK}]
            nbytes = sum(len(m.get("content") or "") for m in msgs)
            cats = []
            for i in range(args.k):
                payload = {"model": model, "messages": msgs, "tools": tools,
                           "temperature": temp, "max_tokens": 1200, "stream": False}
                try:
                    fn_cls = classify_text if args.ask else classify
                    cats.append(fn_cls(call_llm(payload), depth))
                except Exception as e:
                    cats.append(f"ERROR:{str(e)[:40]}")
                print(f"[d{depth}/{arm}] {i + 1}/{args.k}: {cats[-1]}", flush=True)
            c = Counter(cats)
            table[(depth, arm)] = c
            newest = c["NEWEST"]
            print(f"== d{depth}t{args.tail} {arm}: NEWEST {newest}/{args.k}  ctx~{nbytes // 1024}KB  {dict(c)}\n",
                  flush=True)

    print("\n===== SUMMARY (NEWEST / k) =====")
    hdr = "depth  " + "".join(f"{a:>10}" for a in arms)
    print(hdr)
    for depth in depths:
        row = f"{depth:<7}"
        for arm in arms:
            c = table[(depth, arm)]
            row += f"{c['NEWEST']:>7}/{args.k:<3}"
        print(row)
    print("\nstale pulls (any STALE(vN)):")
    for depth in depths:
        for arm in arms:
            c = table[(depth, arm)]
            st = sum(v for k, v in c.items() if k.startswith("STALE"))
            det = {k: v for k, v in c.items() if k.startswith("STALE")}
            print(f"  d{depth} {arm:<8} {st:>3}/{args.k}  {det}")
    print("\nnon-executing (EXPLORE / PLAN_CALL / NO_TOOL / OTHER):")
    for depth in depths:
        for arm in arms:
            c = table[(depth, arm)]
            ne = sum(v for k, v in c.items()
                     if not k.startswith("STALE") and k != "NEWEST")
            det = {k: v for k, v in c.items()
                   if not k.startswith("STALE") and k != "NEWEST"}
            print(f"  d{depth} {arm:<8} {ne:>3}/{args.k}  {det}")


if __name__ == "__main__":
    main()
