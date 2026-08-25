#!/usr/bin/env python3
"""tier1-finish-tool-probe.py — does an explicit `finish` tool fix the
can't-stop pathology without inducing premature exits?

Today a miniswe run ends when the model replies with NO tool call, but the
system prompt never says so (and says "Emit ONE tool call per response",
which actively fights it) — glimmer-r2 sat 35 min on a green tree.  The
donegate-note probe (tier1-glimmer-stuck-probe.py) fixed that decision point
10/10, but only fires after 15 no-edit rounds.  This probe tests the
structural alternative: a 13th tool `finish(summary)` + one terse
contract-example line in the system prompt.

Per model (gemma / devstral / laguna / glimmer), decision points:
  DONE  (shared, glimmer-r2 round 78: all plan steps [x], cargo test green)
        control K=6 (does the can't-stop generalize?) + finishtool K=8
        (good = FINISH-TOOL; FINISH-TEXT also ends the run today)
  MID   (each model's OWN healthy mid-task moment from its 6/6 run: plan
        mostly unchecked, mid-edit) finishtool K=8
        (ANY finish press = PREMATURE — the failure mode that would kill
        the idea; contract-examples-dominate says the added line can warp
        tool choice in unrelated states, so this is the cell that matters)
  STUCK (shared, glimmer-r1-a1 round 110 read loop) finishtool K=6
        (observational: a finish press here would hit the behavioral
        done-gate and get the failing-test output fed back — record only)

KV-cache note: the finish tool changes the SYSTEM PROMPT + tool schemas,
i.e. the very top of the rendered prompt, so live runs would have the whole
cache built over the modified prefix.  Warm-up therefore replays the last
WINDOW main-loop dumps WITH the arm's transform applied (simulating "the
tool was there all along"), per arm.

Usage:  llama-server for <model> up on LLAMA_ENDPOINT, then
        python3 scripts/moments/tier1-finish-tool-probe.py --model gemma
"""
import argparse
import copy
import json
import os
import re
import time
import urllib.request
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENDPOINT = os.environ.get("LLAMA_ENDPOINT", "http://localhost:8464")
BR = os.path.join(REPO, "benchmark_results")

def dd(run):
    return os.path.join(BR, run, "00_baseline/llm_dumps")

# shared glimmer decision points (same as tier1-glimmer-stuck-probe.py)
DONE_MOMENT = dict(dumps=dd("docker_20260823_235146__home_michal_models_Muse-Glimmer-30B-GGU"),
                   prefix="1787521948", pos=100, kind="done", loop_path=None)
STUCK_MOMENT = dict(dumps=dd("docker_20260823_225151__home_michal_models_Muse-Glimmer-30B-GGU"),
                    prefix="1787518354", pos=125, kind="stuck",
                    loop_path="tests/e2e_context.rs")

MODELS = {
    "gemma": dict(mid=dict(dumps=dd("docker_20260823_160906__home_michal_models_gemma-4-26B-A4B-it-G"),
                           prefix="1787494189", pos=48, kind="mid", loop_path=None),
                  temperature=0.2, kwargs={"enable_thinking": False}, name="gemma4"),
    "devstral": dict(mid=dict(dumps=dd("docker_20260823_161629__home_michal_models_devstral-small-2_Dev"),
                              prefix="1787494635", pos=55, kind="mid", loop_path=None),
                     temperature=0.2, kwargs={"enable_thinking": False}, name="devstral-small-2"),
    "laguna": dict(mid=dict(dumps=dd("docker_20260823_172958__home_michal_models_Laguna-XS-2.1-GGUF_L"),
                            prefix="1787499041", pos=60, kind="mid", loop_path=None),
                   temperature=0.2, kwargs={"enable_thinking": False}, name="laguna-xs-2.1"),
    "glimmer": dict(mid=dict(dumps=dd("docker_20260823_112750__home_michal_models_Muse-Glimmer-30B-GGU"),
                             prefix="1787477313", pos=60, kind="mid", loop_path=None),
                    temperature=0.6, kwargs={"enable_thinking": True}, name="muse-glimmer-30b"),
}

# ── the intervention ─────────────────────────────────────────────────────
FINISH_TOOL = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": "End the task. Call ONLY when every plan step is "
                       "complete and the change is verified by the FULL test "
                       "suite. Do not call while any step is unchecked or any "
                       "check is failing.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "One paragraph: what changed and how it was verified.",
                }
            },
            "required": ["summary"],
        },
    },
}
CONTRACT_ANCHOR = 'shell run: {"action":"run","command":"ls","timeout":60}'
CONTRACT_LINE = 'finish ends the task: {"summary":"what changed + how it was verified"}'
FINISH_SENT_ANCHOR = "don't finish until it's fixed."
FINISH_SENT_ADD = " When everything is done and verified, call finish."


def transform_finishtool(dump):
    msgs = copy.deepcopy(dump["messages"])
    sys = msgs[0]
    assert sys["role"] == "system", sys["role"]
    c = sys["content"]
    assert CONTRACT_ANCHOR in c, "contract anchor missing"
    c = c.replace(CONTRACT_ANCHOR, CONTRACT_ANCHOR + "\n" + CONTRACT_LINE)
    if FINISH_SENT_ANCHOR in c:
        c = c.replace(FINISH_SENT_ANCHOR, FINISH_SENT_ANCHOR + FINISH_SENT_ADD)
    sys["content"] = c
    tools = copy.deepcopy(dump["tools"]) + [FINISH_TOOL]
    return msgs, tools


def transform(dump, arm):
    if arm == "control":
        return dump["messages"], dump["tools"]
    if arm == "finishtool":
        return transform_finishtool(dump)
    raise ValueError(arm)


# ── dumps / http ─────────────────────────────────────────────────────────
def seq_dumps(spec):
    out = []
    for f in os.listdir(spec["dumps"]):
        m = re.match(rf"req-{spec['prefix']}-\d+-(\d+)\.json$", f)
        if m:
            out.append((int(m.group(1)), os.path.join(spec["dumps"], f)))
    return [p for _, p in sorted(out)]


def call_llm(payload, timeout=600, retries=3):
    req = urllib.request.Request(
        f"{ENDPOINT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            # 500 = model emitted tool-call output the server's parser rejects
            # (glimmer temp-0.6 format flake); resample.
            if e.code == 500 and attempt < retries:
                print(f"[retry] HTTP 500, model output parse flake ({attempt+1}/{retries})", flush=True)
                continue
            raise


def base_payload(model_spec):
    return {"model": model_spec["name"], "temperature": model_spec["temperature"],
            "chat_template_kwargs": model_spec["kwargs"],
            "max_tokens": 8000, "stream": False}


def warm(spec, arm, model_spec, window=10):
    """Replay the last `window` MAIN-LOOP dumps before the moment, with the
    arm's transform applied, max_tokens=1 — lays the slot KV cache over the
    (possibly modified) prefix.  Main-loop = same system prompt head as the
    moment dump (excludes debugger/model-edit/summarizer contexts)."""
    files = seq_dumps(spec)
    moment = files[spec["pos"]]
    head = json.load(open(moment))["messages"][0]["content"][:80]
    picked = []
    for p in files[: spec["pos"]]:
        d = json.load(open(p))
        m = d["messages"]
        if len(m) > 2 and m[0]["role"] == "system" and m[0]["content"][:80] == head:
            picked.append((p, d))
    picked = picked[-window:]
    print(f"[warm {arm}] replaying {len(picked)} dumps", flush=True)
    t0 = time.time()
    for p, d in picked:
        msgs, tools = transform(d, arm)
        payload = dict(base_payload(model_spec), messages=msgs, tools=tools, max_tokens=1)
        call_llm(payload)
    print(f"[warm {arm}] done in {time.time()-t0:.0f}s", flush=True)


# ── scoring ──────────────────────────────────────────────────────────────
def classify(resp, loop_path):
    msg = resp["choices"][0]["message"]
    tcs = msg.get("tool_calls") or []
    if not tcs:
        return {"cat": "FINISH-TEXT", "detail": (msg.get("content") or "")[:160]}
    tc = tcs[0]["function"]
    name = tc["name"]
    try:
        args = json.loads(tc["arguments"])
    except Exception:
        return {"cat": "MALFORMED", "detail": name}
    if name == "finish":
        return {"cat": "FINISH-TOOL", "detail": str(args.get("summary", ""))[:160]}
    if name == "file":
        a, path = args.get("action", ""), str(args.get("path", ""))
        d = f"file({a} {path} {args.get('start')}-{args.get('end')})"
        if a == "read" and loop_path and path == loop_path:
            return {"cat": "LOOP-READ", "detail": d}
        return {"cat": "READ" if a in ("read", "search") else "FILE-OTHER", "detail": d}
    if name in ("replace_range", "insert_at", "write_file"):
        return {"cat": "EDIT", "detail": f"{name} {args.get('path')} "
                f"L{args.get('start', args.get('after_line'))}-{args.get('end', '')}",
                "content": str(args.get("content", ""))[:400]}
    if name == "refactor":
        return {"cat": "REFACTOR", "detail": json.dumps(args)[:120]}
    if name == "plan":
        return {"cat": f"PLAN-{str(args.get('action','')).upper()}", "detail": ""}
    if name == "shell":
        return {"cat": "SHELL", "detail": str(args.get("command", ""))[:80]}
    return {"cat": name.upper(), "detail": json.dumps(args)[:80]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--out", default=os.path.join(BR, "_moments/finish-tool-v1"))
    args = ap.parse_args()
    mspec = MODELS[args.model]

    cells = [
        ("done", DONE_MOMENT, "control", 6),
        ("done", DONE_MOMENT, "finishtool", 8),
        ("mid", mspec["mid"], "finishtool", 8),
        ("stuck", STUCK_MOMENT, "finishtool", 6),
    ]
    os.makedirs(args.out, exist_ok=True)
    results_path = os.path.join(args.out, "results.json")
    results = json.load(open(results_path)) if os.path.exists(results_path) else {}

    for mkind, spec, arm, k in cells:
        key = f"{args.model}/{mkind}/{arm}"
        samples = results.get(key, [])
        if samples and samples[0]["cat"] == "SKIP-CTX":
            print(f"[{key}] previously skipped (ctx overflow)", flush=True)
            continue
        if len(samples) >= k:
            print(f"[{key}] cached ({len(samples)}), skipping", flush=True)
            continue
        dump = json.load(open(seq_dumps(spec)[spec["pos"]]))
        try:
            warm(spec, arm, mspec)
            msgs, tools = transform(dump, arm)
            payload = dict(base_payload(mspec), messages=msgs, tools=tools)
            t0 = time.time()
            while len(samples) < k:
                try:
                    c = classify(call_llm(payload), spec["loop_path"])
                except urllib.error.HTTPError as e:
                    if e.code != 500:
                        raise
                    c = {"cat": "MALFORMED-500",
                         "detail": "unparseable tool-call output after retries"}
                samples.append(c)
                results[key] = samples
                json.dump(results, open(results_path, "w"), indent=1)
                print(f"[{key}] {len(samples)}/{k}: {c['cat']}  {c['detail'][:90]}", flush=True)
        except urllib.error.HTTPError as e:
            if e.code == 400:
                try:
                    body = e.read().decode()[:200]
                except Exception:
                    body = ""
                results[key] = [{"cat": "SKIP-CTX", "detail": body}]
                json.dump(results, open(results_path, "w"), indent=1)
                print(f"[{key}] SKIPPED (HTTP 400, likely ctx overflow): {body[:140]}", flush=True)
                continue
            raise
        print(f"[{key}] {Counter(s['cat'] for s in samples)}  ({time.time()-t0:.0f}s)",
              flush=True)

    print(f"\n===== {args.model} =====")
    for key, samples in sorted(results.items()):
        if key.startswith(args.model + "/"):
            top = ", ".join(f"{c}×{n}" for c, n in
                            Counter(s["cat"] for s in samples).most_common(5))
            print(f"{key:28} N={len(samples):>2}  {top}")


if __name__ == "__main__":
    main()
