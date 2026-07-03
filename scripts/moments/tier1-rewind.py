#!/usr/bin/env python3
"""tier1-rewind — probe whether the debugger, given a third REWIND decision
and per-file revision tables, can identify an abandoned-clean earlier
revision instead of only SCRAP (whole tree) or CONTINUE (no revert).

Moments (real debugger fires, revision tables reconstructed AS KNOWABLE AT
FIRE TIME — no hindsight from later revisions):

  postfix_run2 (judge_mf-postfix/run2, 3rd fire, req 181): src/cli/commands/
    run.rs spiraled from rev_1/rev_4 (ast=ok, 0-1 errors) to the then-current
    rev_5 (ast=ok, 31 project_errors) — never reverted. Clear, unambiguous
    ground truth: REWIND run.rs to rev_1 or rev_4.

  prefix_run3 (judge_mf-prefix/run3, only fire, req 110): THREE files changed.
    src/main.rs regressed from rev_1/rev_2 (ast=ok, 0 errors) to rev_5
    (ast=broken, 3 errors) — good REWIND target. src/cli/mod.rs and
    src/context/mod.rs are each at their best-available state already
    (monotonically improving / near-done) — should NOT be targeted. Tests
    PRECISION: does it single out main.rs, or over-revert working files?

Variants:
  control : verbatim DEBUGGER_JUDGE_PROMPT (SCRAP/CONTINUE only), original
            user message (git diff only, no revision tables) — reproduces
            what actually happened (sanity check).
  rewind  : system prompt gains a third DECISION: REWIND option with a
            TARGET: <path> rev_<N> field; user message gains a REVISION
            HISTORY section (the real per-file tables, as reconstructed).

Scoring: parse first line for DECISION; if REWIND, parse TARGET file+rev and
compare against ground truth for that moment.
"""

import argparse
import json
import os
import re
import urllib.request

REPO = "/home/michal/dev/miniswe"
ENDPOINT = os.environ.get("LLAMA_ENDPOINT", "http://localhost:8464")

DEBUGGER_JUDGE_PROMPT = """You are a READ-ONLY analyst with fresh eyes on a STUCK coding task. You have ONLY read/search/inspect tools — you CANNOT edit files, run shell, set a plan, or use a scratchpad. Do NOT plan and do NOT try to edit.
Investigate the failure and the changes made so far (the failing location, the relevant definitions/callsites, and whether the changes are even in the right place for the GOAL). Then DECIDE whether this attempt is worth continuing:
- SCRAP: the changes are misdirected, damaged, or off-path — editing the wrong places for the GOAL, or broken in ways the GOAL did not require. Reverting everything to the clean original and starting fresh would be faster and more reliable. IGNORE effort already spent.
- CONTINUE: the changes are on the path to the GOAL and nearly working; only a focused fix remains.
Output your decision on the FIRST line, exactly one of:
DECISION: SCRAP
DECISION: CONTINUE
If SCRAP: add one line — REASON: <the single most important reason> — and STOP.
If CONTINUE: produce the recovery report the main agent will apply —
ROOT CAUSE: <the precise reason the check fails>
FIX: <where and what must change, described conceptually — NOT verbatim code you cannot compile-check>
PLAN: <the concrete remaining steps to finish the GOAL, including the step that makes the feature actually work at runtime, not merely compile>"""

REWIND_PROMPT = """You are a READ-ONLY analyst with fresh eyes on a STUCK coding task. You have ONLY read/search/inspect tools — you CANNOT edit files, run shell, set a plan, or use a scratchpad. Do NOT plan and do NOT try to edit.
Investigate the failure and the changes made so far (the failing location, the relevant definitions/callsites, and whether the changes are even in the right place for the GOAL). Below each changed file's diff you are also given its REVISION HISTORY — every edit made to that file, in order, with whether it parsed (ast=ok/broken) and its error count at that point. Use it. Then DECIDE:
- SCRAP: ALL the changes across the whole tree are misdirected or damaged — reverting EVERYTHING to the clean original and starting fresh would be faster and more reliable. IGNORE effort already spent.
- REWIND: ONE SPECIFIC FILE has a materially better earlier revision in its history (lower error count, ast=ok) that was abandoned by later edits which made things worse — reverting JUST that file recovers real progress instead of fighting the current broken state in it. Do NOT choose this for a file that is already at its best-available revision, or one that is steadily improving — only when a clearly better PAST revision of that file was abandoned.
- CONTINUE: the changes are on the path to the GOAL and nearly working as they stand; only a focused forward fix remains, no revert needed anywhere.
Output your decision on the FIRST line, exactly one of:
DECISION: SCRAP
DECISION: REWIND
DECISION: CONTINUE
If SCRAP: add one line — REASON: <the single most important reason> — and STOP.
If REWIND: add one line — TARGET: <path> rev_<N> — naming the EXACT file and revision number to restore, then REASON: <why that revision is better and what remains after restoring it> — and STOP.
If CONTINUE: produce the recovery report the main agent will apply —
ROOT CAUSE: <the precise reason the check fails>
FIX: <where and what must change, described conceptually — NOT verbatim code you cannot compile-check>
PLAN: <the concrete remaining steps to finish the GOAL, including the step that makes the feature actually work at runtime, not merely compile>"""

MOMENTS = {
    "postfix_run2": {
        "dump": "/home/michal/dev/miniswe/benchmark_results/replaymatrix_20260703_000135_gemma-4-26B-A4B-it-UD-Q4_K_M/judge_mf/run2/llm_dumps/req-1783030593-00052-000181.json",
        "revtables": {
            "src/cli/commands/run.rs": """rev_0  initial                             ast=ok      file_errors=0  project_errors=0
  rev_1  change_signature.add_param (+1 -0)  ast=ok      file_errors=0  project_errors=0
  rev_4  replace_range L131-140 (+10 -10)    ast=ok      file_errors=1  project_errors=1
* rev_5  replace_range L41-150 (+75 -110)    ast=ok      file_errors=31  project_errors=31  <- current
  rev_2  change_signature.add_param          [reverted, no errors]
  rev_3  replace_range L125-135 (+52 -11)    [reverted, ast=broken at 1:1: syntax error]""",
            "src/cli/mod.rs": """rev_0  initial                      ast=ok      file_errors=0  project_errors=0
* rev_2  insert_at after L18 (+4 -0)  ast=ok      file_errors=0  project_errors=2  <- current
  rev_1  replace_range L16-22 (+10 -7)  [reverted, ast=broken at 28:5: syntax error]""",
            "src/context/mod.rs": """rev_0  initial                             ast=ok      file_errors=0  project_errors=0
  rev_1  change_signature.add_param (+1 -0)  ast=ok      file_errors=0  project_errors=0
  rev_2  replace_range L299-337 (+46 -39)    ast=ok      file_errors=0  project_errors=4
  rev_13 replace_range L298-345 (+46 -48)    ast=ok      file_errors=1  project_errors=5
* rev_14 replace_range L298-345 (+46 -48)    ast=ok      file_errors=0  project_errors=4  <- current
  rev_3..rev_10  replace_range L299-337 (+46 -39)  [reverted, ast=broken at 347:9: syntax error] (x8)""",
        },
        "ground_truth": {"file_contains": "run.rs", "rev_in": {1, 4}},
    },
    "prefix_run3": {
        "dump": "/home/michal/dev/miniswe/benchmark_results/replaymatrix_20260702_224555_unknown/judge_mf/run3/llm_dumps/req-1783027248-00052-000110.json",
        "revtables": {
            "src/main.rs": """rev_0  initial                         ast=ok      file_errors=0  project_errors=0
  rev_1  change_signature.add_param      ast=ok      file_errors=0  project_errors=0
  rev_2  change_signature.add_param      ast=ok      file_errors=0  project_errors=0
  rev_3  replace_range L38-44 (+6 -7)    ast=broken  file_errors=8  project_errors=38
  rev_4  replace_range L29-44 (+6 -16)   ast=broken  file_errors=26  project_errors=56
* rev_5  replace_range L29-38 (+17 -10)  ast=broken  file_errors=3  project_errors=3  <- current""",
            "src/cli/mod.rs": """rev_0  initial                         ast=ok      file_errors=0  project_errors=0
  rev_1  replace_range L16-22 (+10 -7)   ast=broken  file_errors=29  project_errors=29
  rev_2  replace_range L24-26 (+6 -3)    ast=broken  file_errors=51  project_errors=51
* rev_3  replace_range L24-62 (+40 -39)  ast=broken  file_errors=4  project_errors=4  <- current""",
            "src/context/mod.rs": """rev_0  initial                             ast=ok      file_errors=0  project_errors=0
  rev_1  change_signature.add_param (+1 -0)  ast=ok      file_errors=0  project_errors=0
* rev_2  replace_range L298-334 (+46 -37)    ast=ok      file_errors=1  project_errors=1  <- current""",
        },
        "ground_truth": {"file_contains": "main.rs", "rev_in": {1, 2}},
        "trap_files": ["cli/mod.rs", "context/mod.rs"],
    },
}

DEC_RE = re.compile(r"DECISION:\s*(SCRAP|CONTINUE|REWIND)", re.IGNORECASE)
TARGET_RE = re.compile(r"TARGET:\s*(\S+)\s+rev[_ ]?(\d+)", re.IGNORECASE)


def build_messages(moment_key, variant):
    m = MOMENTS[moment_key]
    dump = json.load(open(m["dump"]))
    sysmsg = REWIND_PROMPT if variant == "rewind" else DEBUGGER_JUDGE_PROMPT
    user = dump["messages"][1]["content"]
    if variant == "rewind":
        sections = ["\n\n=== REVISION HISTORY (per changed file, from the fast-edit revision store) ==="]
        for fname, table in m["revtables"].items():
            sections.append(f"\n[revisions] {fname}\n{table}")
        user = user + "\n".join(sections)
    return [{"role": "system", "content": sysmsg}, {"role": "user", "content": user}], dump


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


def classify(text, moment_key):
    dm = DEC_RE.search(text)
    dec = dm.group(1).upper() if dm else "NONE"
    if dec != "REWIND":
        return {"decision": dec, "detail": text[:100].replace("\n", " ")}
    tm = TARGET_RE.search(text)
    if not tm:
        return {"decision": "REWIND", "detail": "no TARGET parsed: " + text[:100]}
    file, rev = tm.group(1), int(tm.group(2))
    gt = MOMENTS[moment_key]["ground_truth"]
    hit = gt["file_contains"] in file and rev in gt["rev_in"]
    trap = any(t in file for t in MOMENTS[moment_key].get("trap_files", []))
    verdict = "HIT" if hit else ("TRAP" if trap else "WRONG-TARGET")
    return {"decision": "REWIND", "detail": f"{verdict} target={file} rev_{rev}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--moments", default="postfix_run2,prefix_run3")
    ap.add_argument("--variants", default="control,rewind")
    ap.add_argument("--out", default=os.path.join(REPO, "benchmark_results/_moments/rewind-v1"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    results = {}
    for mk in args.moments.split(","):
        for variant in args.variants.split(","):
            key = f"{mk}/{variant}"
            messages, dump = build_messages(mk, variant)
            samples = []
            for i in range(args.k):
                try:
                    resp = call_llm(messages, dump["model"], dump.get("tools", []),
                                     dump.get("temperature", 0.2), dump.get("max_tokens", 8000))
                    text = resp["choices"][0]["message"].get("content") or ""
                    c = classify(text, mk)
                except Exception as e:
                    c = {"decision": "ERROR", "detail": str(e)[:100]}
                samples.append(c)
                print(f"[{key}] {i+1}/{args.k}: {c['decision']}  {c.get('detail','')}", flush=True)
            results[key] = samples
            json.dump(results, open(os.path.join(args.out, "results.json"), "w"), indent=1)

    print("\n=== SUMMARY ===")
    for key, samples in results.items():
        from collections import Counter
        cnt = Counter(f"{s['decision']}:{s['detail'].split()[0] if s['decision']=='REWIND' else ''}" for s in samples)
        print(f"{key}: {dict(cnt)}")


if __name__ == "__main__":
    main()
