#!/usr/bin/env python3
"""tier1-glimmer-stuck-probe.py — intervention probe on the two Glimmer stuck
moments from the 2026-08-23 overnight bench, simulating what the winning
stuck-trigger (trigger-eval.py: T2c frozen-sig-15+4min red, T4 green-noedit-15)
would INJECT at fire time.

Moment A (red-frozen, read loop) — glimmer r1 attempt 1, round 110:
  docker_20260823_225151 / req-1787518354-00055-000125.json (120 msgs,
  post-forced-compaction, still reading tests/e2e_context.rs L54-54; loop
  began ~21:17:30, T2c would have fired right about here).
  Arms: control / stucknote (T2c fire text appended to last tool result —
  the proven post-revert-hint delivery channel) / compact (masking mirror)
  / both.

Moment B (green-frozen, can't-stop) — glimmer r2, round 78:
  docker_20260823_235146 / req-1787521948-00055-000100.json (162 msgs,
  plan(show) all steps [x], cargo test exit 0 since 22:02:34; the model
  dithered 35 more minutes live).  NOTE: the system prompt never tells the
  model HOW to finish (no tool call = finish); the donegate arm teaches it.
  Arms: control / donegate (T4 fire text appended to the plan(show) result).

Scoring: FIRST tool call of the response (no rollout).
  A: LOOP-READ (any read of tests/e2e_context.rs — T3 ignores ranges) is the
     loop; EDIT/REFACTOR/CHECK/SHELL/PLAN-check break it.
  B: FINISH (no tool call) is the goal; SHELL-verify acceptable; reads /
     plan(show) / fresh edits are the dither.

Warm mode (default, per readloop-warm-cache-probe: cold read-loop replays
mislead): replay the last WINDOW main-loop dumps of the same session with
max_tokens=1 to lay the slot KV cache incrementally, then sample.  A cold
control pass runs first for comparison.

Usage: python3 scripts/moments/tier1-glimmer-stuck-probe.py [--k 10] [--cold-k 6]
       (llama-server for muse-glimmer-30b must be up on LLAMA_ENDPOINT)
"""
import argparse
import copy
import json
import os
import re
import time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENDPOINT = os.environ.get("LLAMA_ENDPOINT", "http://localhost:8464")

BR = os.path.join(REPO, "benchmark_results")
DD_A = os.path.join(BR, "docker_20260823_225151__home_michal_models_Muse-Glimmer-30B-GGU/00_baseline/llm_dumps")
DD_B = os.path.join(BR, "docker_20260823_235146__home_michal_models_Muse-Glimmer-30B-GGU/00_baseline/llm_dumps")

MOMENTS = {
    "A": dict(dumps=DD_A, prefix="1787518354", seq=125, kind="readloop",
              loop_path="tests/e2e_context.rs",
              variants=["control", "stucknote", "compact", "both"]),
    "B": dict(dumps=DD_B, prefix="1787521948", seq=100, kind="cantstop",
              loop_path=None,
              variants=["control", "donegate"]),
}

# ── injection texts (terse; harness can compute every fact referenced) ──
STUCK_NOTE = (
    "[stuck-check] 15+ rounds and 4+ minutes with NO edit and NO change in the "
    "compiler/test signal. Re-reading tests/e2e_context.rs will show the same "
    "bytes again — do NOT read it again. Take ONE concrete action now: make the "
    "smallest edit that advances the first unchecked plan step, or run "
    "plan(action='check', step=3) if that step is in fact already done."
)
DONE_NOTE = (
    "[done-check] Every plan step is checked off and the last full test run "
    "passed (cargo test exit 0). The task appears COMPLETE. Do not re-read "
    "files or re-show the plan. If something is genuinely missing, name it and "
    "fix it with ONE edit; otherwise finish NOW: reply with a one-paragraph "
    "summary and NO tool call — a reply without a tool call ends the task."
)

# ── faithful mirror of compressor.rs masking constants ──────────────────
OBS_PLACEHOLDER = "[earlier tool output elided to save context]"
KEEP_RAW_OBS = 3
GUARD_MARKERS = [
    "[auto-revert]",
    "[hint]",
    "You are in a loop",
    "You are in an edit↔revert loop",
    "same read/inspection call",
]
GUARD_MAX_CHARS = 4000


def is_guard_observation(content):
    return len(content) <= GUARD_MAX_CHARS and any(m in content for m in GUARD_MARKERS)


def transform_compact(messages):
    out = copy.deepcopy(messages)
    tool_idxs = [i for i, m in enumerate(out) if m.get("role") == "tool"]
    if len(tool_idxs) <= KEEP_RAW_OBS:
        return out
    for i in tool_idxs[: len(tool_idxs) - KEEP_RAW_OBS]:
        c = out[i].get("content")
        if not isinstance(c, str) or c == OBS_PLACEHOLDER or is_guard_observation(c):
            continue
        out[i]["content"] = OBS_PLACEHOLDER
    return out


def append_note(messages, note):
    out = copy.deepcopy(messages)
    for m in reversed(out):
        if m.get("role") == "tool":
            m["content"] = (m.get("content") or "") + "\n\n" + note
            return out
    raise AssertionError("no tool message to append to")


def transform(messages, variant, kind):
    note = STUCK_NOTE if kind == "readloop" else DONE_NOTE
    if variant == "control":
        return messages
    if variant in ("stucknote", "donegate"):
        return append_note(messages, note)
    if variant == "compact":
        return transform_compact(messages)
    if variant == "both":
        return append_note(transform_compact(messages), note)
    raise ValueError(variant)


# ── dumps ────────────────────────────────────────────────────────────────
def seq_dumps(dd, prefix):
    out = []
    for f in os.listdir(dd):
        m = re.match(rf"req-{prefix}-\d+-(\d+)\.json$", f)
        if m:
            out.append((int(m.group(1)), os.path.join(dd, f)))
    return sorted(out)


def call_llm(payload, timeout=600):
    req = urllib.request.Request(
        f"{ENDPOINT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def warm(spec, window):
    """Replay the last `window` main-loop dumps before the moment (max_tokens=1)
    to lay the slot KV cache incrementally, mirroring live growth. Summarizer
    dumps (2 msgs) are skipped — live they hit a different context entirely."""
    pre = [(s, p) for s, p in seq_dumps(spec["dumps"], spec["prefix"]) if s < spec["seq"]]
    main = [(s, p) for s, p in pre if len(json.load(open(p))["messages"]) > 2]
    picked = main[-window:]
    print(f"[warm] replaying {len(picked)} main-loop dumps "
          f"(seq {picked[0][0]}..{picked[-1][0]}), max_tokens=1", flush=True)
    t0 = time.time()
    for i, (s, p) in enumerate(picked):
        d = json.load(open(p))
        payload = {"model": d["model"], "messages": d["messages"],
                   "temperature": d.get("temperature", 0.6),
                   "max_tokens": 1, "stream": False}
        if "tools" in d:
            payload["tools"] = d["tools"]
        if "chat_template_kwargs" in d:
            payload["chat_template_kwargs"] = d["chat_template_kwargs"]
        call_llm(payload)
        if (i + 1) % 4 == 0:
            print(f"[warm] {i+1}/{len(picked)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[warm] done in {time.time()-t0:.0f}s", flush=True)


# ── scoring ──────────────────────────────────────────────────────────────
def classify(resp, kind, loop_path):
    msg = resp["choices"][0]["message"]
    tcs = msg.get("tool_calls") or []
    if not tcs:
        cat = "FINISH" if kind == "cantstop" else "PROSE"
        return {"cat": cat, "detail": (msg.get("content") or "")[:160]}
    tc = tcs[0]["function"]
    name = tc["name"]
    try:
        args = json.loads(tc["arguments"])
    except Exception:
        return {"cat": "MALFORMED", "detail": name}

    if name == "file":
        action = args.get("action", "")
        path = str(args.get("path", ""))
        detail = f"file({action} {path} {args.get('start')}-{args.get('end')})"
        if action == "read" and loop_path and path == loop_path:
            return {"cat": "LOOP-READ", "detail": detail}
        if action == "read":
            return {"cat": "READ-OTHER", "detail": detail}
        return {"cat": "FILE-OTHER", "detail": detail}
    if name in ("replace_range", "insert_at", "write_file"):
        detail = (f"{name} {args.get('path')} "
                  f"L{args.get('start', args.get('after_line'))}-{args.get('end', '')}")
        return {"cat": "EDIT", "detail": detail,
                "content": str(args.get("content", ""))[:800]}
    if name == "refactor":
        return {"cat": "REFACTOR", "detail": json.dumps(args)[:120]}
    if name == "plan":
        a = str(args.get("action", ""))
        d = f"plan({a}" + (f", step={args.get('step')}" if "step" in args else "") + ")"
        return {"cat": f"PLAN-{a.upper()}", "detail": d}
    if name == "check":
        return {"cat": "CHECK", "detail": "check()"}
    if name == "shell":
        return {"cat": "SHELL", "detail": str(args.get("command", ""))[:80]}
    if name == "revert":
        return {"cat": "REVERT", "detail": json.dumps(args)[:80]}
    return {"cat": name.upper(), "detail": json.dumps(args)[:80]}


BREAKS = {"readloop": {"EDIT", "REFACTOR", "CHECK", "SHELL", "PLAN-CHECK", "REVERT"},
          "cantstop": {"FINISH"}}
LOOPS = {"readloop": {"LOOP-READ"},
         "cantstop": {"LOOP-READ", "READ-OTHER", "PLAN-SHOW", "EDIT", "REFACTOR",
                      "FILE-OTHER"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--cold-k", type=int, default=6)
    ap.add_argument("--window", type=int, default=14)
    ap.add_argument("--moments", default="A,B")
    ap.add_argument("--out", default=os.path.join(BR, "_moments/glimmer-stuck-v1"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    results_path = os.path.join(args.out, "results.json")
    results = json.load(open(results_path)) if os.path.exists(results_path) else {}

    def save():
        json.dump(results, open(results_path, "w"), indent=1)

    for mname in args.moments.split(","):
        spec = MOMENTS[mname]
        moment_path = dict(seq_dumps(spec["dumps"], spec["prefix"]))[spec["seq"]]
        dump = json.load(open(moment_path))
        base_payload = {"model": dump["model"], "tools": dump["tools"],
                        "temperature": dump.get("temperature", 0.6),
                        "max_tokens": dump.get("max_tokens", 8000), "stream": False}
        if "chat_template_kwargs" in dump:
            base_payload["chat_template_kwargs"] = dump["chat_template_kwargs"]
        print(f"\n=== moment {mname} ({spec['kind']}) — {os.path.basename(moment_path)}, "
              f"{len(dump['messages'])} msgs ===", flush=True)

        runs = [("cold", ["control"], args.cold_k)] if args.cold_k else []
        runs.append(("warm", spec["variants"], args.k))
        warmed = False
        for mode, variants, k in runs:
            if mode == "warm" and not warmed:
                warm(spec, args.window)
                warmed = True
            for variant in variants:
                key = f"{mname}/{mode}/{variant}"
                samples = results.get(key, [])
                if len(samples) >= k:
                    print(f"[{key}] cached ({len(samples)}), skipping", flush=True)
                    continue
                msgs = transform(dump["messages"], variant, spec["kind"])
                payload = dict(base_payload, messages=msgs)
                t0 = time.time()
                while len(samples) < k:
                    resp = call_llm(payload)
                    c = classify(resp, spec["kind"], spec["loop_path"])
                    samples.append(c)
                    results[key] = samples
                    save()
                    print(f"[{key}] {len(samples)}/{k}: {c['cat']}  {c['detail'][:90]}",
                          flush=True)
                cats = [s["cat"] for s in samples]
                br = sum(c in BREAKS[spec["kind"]] for c in cats)
                lp = sum(c in LOOPS[spec["kind"]] for c in cats)
                print(f"[{key}] break {br}/{len(cats)}  loop {lp}/{len(cats)}  "
                      f"({time.time()-t0:.0f}s)", flush=True)

    # ── summary table ────────────────────────────────────────────────────
    print("\n===== SUMMARY =====")
    print(f"{'arm':32} {'N':>3} {'break':>6} {'loop':>5}  top categories")
    for key, samples in results.items():
        mname = key.split("/")[0]
        kind = MOMENTS[mname]["kind"]
        cats = [s["cat"] for s in samples]
        br = sum(c in BREAKS[kind] for c in cats)
        lp = sum(c in LOOPS[kind] for c in cats)
        from collections import Counter
        top = ", ".join(f"{c}×{n}" for c, n in Counter(cats).most_common(4))
        print(f"{key:32} {len(cats):>3} {br:>6} {lp:>5}  {top}")


if __name__ == "__main__":
    main()
