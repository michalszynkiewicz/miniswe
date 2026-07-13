#!/usr/bin/env python3
"""tier1-replace-tightness — does rewording `replace_range`'s description
reduce whole-function reproduce-from-memory rewrites?

Moment: the recurring "dropped provider-header" bug (seen 3x across the lazy
compaction bench: batch1 run1, batch1 run2, batch2 run3 attempts 1-2). Root
cause confirmed via diff: the model's OWN plan says "replace the entire
function body" for `assemble()` in src/context/mod.rs, then it calls
replace_range(start=289, end=403) — a 115-line range covering the whole
function — reproduces it from memory, and silently drops the
`if !block.header.is_empty() { push_str(block.header) }` branch.

Captured moment: benchmark_results/compaction_20260713_152948_.../lazy/run1/
llm_dumps/req-1783951111-48805-000062.json, message index 60 (the assistant's
replace_range call). messages[:60] is exactly what the model saw — including
its own just-saved plan committing to "replacing the entire function body" —
right before making that call. Replaying this validates whether the tool
description alone (last line of defense, AFTER the model has already
committed to a whole-function-rewrite plan) can still pull it toward a
narrower edit.

The eventual correct fix (attempt 3 of that same run) used replace_range on
just L357-361 — a 5-line range — to reinstate the dropped header push. That's
the empirical "GOOD" range size to calibrate against the original 115-line
"BAD" one.

Arms:
  control   — current wording ("Keep the range TIGHT — anything in the
              range that isn't in `content` is gone.")
  treatment — reworded ("Replace ONLY the lines that actually need to
              change... If you're changing one line inside a bigger block,
              set start=end to that exact line.")

Moments:
  late  (default) — message index 60, described above: the "give up and
          rewrite everything" moment, 25 rounds into a struggle, executed
          against the model's own "replace the entire function body" plan.
  early — message index 26, ~round 13. IMPORTANT CAVEAT found while
          preparing this: this dump's own message[1] shows the conversation
          already starts from a COMPACTED SUMMARY ("[Your earlier work in
          this session]: updated `assemble` to include context providers...
          no signature change yet") — meaning the `system_prompt_override
          .is_none()` wrapper (and the missing header push) was ALREADY
          present in the file before this dump's visible history begins.
          This is NOT a pristine "first attempt" — it's the model re-editing
          code that a still-earlier (now compacted-away, unrecoverable)
          round had already broken. The REAL historical response here
          (messages[26]) was itself already broken: a 42-line replace_range
          that came back `ast broken: 369:43: missing ;`. Use --moment early
          --control-only first to check whether that failure reproduces
          under resampling before spending a treatment arm on it — if
          control mostly succeeds here, this cut is too noisy/lucky-punch to
          be a useful test and the finding should be reported as such rather
          than forced into a wording comparison.

Usage: tier1-replace-tightness.py [--k 12] [--moment late|early] [--control-only]
"""

import argparse
import json
import os
import re
import urllib.request

REPO = "/home/michal/dev/miniswe"
ENDPOINT = os.environ.get("LLAMA_ENDPOINT", "http://localhost:8464")

RUN_DIR = os.path.join(
    REPO,
    "benchmark_results/compaction_20260713_152948__home_michal_models_gemma-4-26B-A4B-it-G/"
    "lazy/run1/llm_dumps",
)

MOMENTS = {
    "late": {
        "dump": os.path.join(RUN_DIR, "req-1783951111-48805-000062.json"),
        "cut": 60,
    },
    "early": {
        "dump": os.path.join(RUN_DIR, "req-1783951111-48805-000062.json"),
        "cut": 26,
    },
    "early-nudged": {
        "dump": os.path.join(RUN_DIR, "req-1783951111-48805-000062.json"),
        "cut": 26,
        # Baseline-check (--moment early --control-only) showed 12/12
        # resamples choose a 3rd identical read (messages[22] and [24] are
        # already 2 identical `file(read src/context/mod.rs L305-352)`
        # calls) instead of editing. The live loop-detector would fire
        # REPEATED_READ_NUDGE on that 3rd identical call (same_call_streak
        # >= 3) and replace its result with the nudge text, not the file
        # content. Splice that in mechanically so the replay reaches the
        # same fork-in-the-road the live run would have.
        "append_nudge": True,
    },
}

# Verbatim src/cli/commands/agent/hints.rs REPEATED_READ_NUDGE.
REPEATED_READ_NUDGE = (
    "You just made this same read/inspection call 3 times in a row. The "
    "result hasn't changed. What specifically are you looking for? Try a "
    "narrower search, a different range, or move on to making an edit."
)

CONTROL_DESC = (
    "Replace lines [start..=end] (1-based, inclusive) with `content`. Empty "
    "content deletes. Keep the range TIGHT — anything in the range that "
    "isn't in `content` is gone. To ADD new lines, use insert_at. For "
    "signature changes / renames, use refactor. Per-edit AST+LSP feedback "
    "comes back in the response; if you see a regression, call `revert`."
)

TREATMENT_DESC = (
    "Replace lines [start..=end] (1-based, inclusive) with `content`. Empty "
    "content deletes. Replace ONLY the lines that actually need to change — "
    "anything in the range that isn't in `content` is gone, so a wide range "
    "forces you to retype unchanged code from memory and risks silently "
    "dropping lines. If you're changing one line inside a bigger block, set "
    "start=end to that exact line rather than re-supplying the whole block. "
    "To ADD new lines, use insert_at. For signature changes / renames, use "
    "refactor. Per-edit AST+LSP feedback comes back in the response; if you "
    "see a regression, call `revert`."
)

# Ceiling + split guidance: gives a concrete number instead of the vague
# "tight"/"only what's needed", and offers the actual alternative (several
# small calls) for the case that made the original bug (a multi-part
# function-body restructure).
SMALL_CEILING_DESC = (
    "Replace lines [start..=end] (1-based, inclusive) with `content`. Empty "
    "content deletes. Keep ranges SMALL — ideally under 10 lines. Anything "
    "in the range that isn't in `content` is gone: a large range forces you "
    "to retype surrounding code from memory, and any line you mistype, "
    "omit, or forget is silently deleted with no error. If a change touches "
    "several separate spots (e.g. restructuring a function), make SEVERAL "
    "small replace_range calls, one per spot, rather than one big call "
    "covering everything in between. To ADD new lines, use insert_at. For "
    "signature changes / renames, use refactor. Per-edit AST+LSP feedback "
    "comes back in the response; if you see a regression, call `revert`."
)

# Count-the-unchanged-lines rule: an explicit yes/no check to run before
# calling, rather than an adjective ("tight"/"small") that's easy to
# rationalize past.
COUNT_CHECK_DESC = (
    "Replace lines [start..=end] (1-based, inclusive) with `content`. Empty "
    "content deletes. Before calling, check: does EVERY line in [start,end] "
    "actually need to change? If any line in the range is one you intend to "
    "leave alone, shrink the range so it's excluded — reproducing unchanged "
    "lines from memory risks silently dropping or altering them, with no "
    "error. To ADD new lines without touching existing ones, use insert_at. "
    "For signature changes / renames, use refactor. Per-edit AST+LSP "
    "feedback comes back in the response; if you see a regression, call "
    "`revert`."
)

# Vivid consequence-first framing: leads with DANGER instead of burying the
# risk after the mechanics, on the theory that a duller advisory ("keep it
# tight") is easy to skim past once the model has already decided to do a
# big rewrite.
DANGER_FIRST_DESC = (
    "Replace lines [start..=end] (1-based, inclusive) with `content`. "
    "DANGER: anything in the range that isn't in `content` is gone — no "
    "error, no warning, just silently deleted. The wider the range, the "
    "more existing code you must retype from memory, and the higher the "
    "chance you drop something (a line, a branch, an import) without "
    "noticing. Default to the SMALLEST range that covers your actual "
    "change — often just one line. To ADD new lines, use insert_at. For "
    "signature changes / renames, use refactor. Per-edit AST+LSP feedback "
    "comes back in the response; if you see a regression, call `revert`."
)

# "Not a rewrite tool" framing: targets the actual mechanism found in the
# late-moment replay — the model's OWN PLAN said "replace the entire
# function body," and replace_range was then used to execute that plan
# literally. This variant tries to pre-empt that framing at the tool-choice
# level, independent of whatever the plan says.
NOT_REWRITE_DESC = (
    "Replace lines [start..=end] (1-based, inclusive) with `content`. Empty "
    "content deletes. This is a surgical tool, not a rewrite tool — do NOT "
    "use it to reproduce a whole function or block from memory, even if "
    "your plan describes the change as \"rewrite/replace the whole "
    "function\". Break a multi-part restructuring into several small "
    "replace_range/insert_at calls targeting only the specific lines that "
    "change. Anything in the range that isn't in `content` is gone. For "
    "signature changes / renames, use refactor. Per-edit AST+LSP feedback "
    "comes back in the response; if you see a regression, call `revert`."
)

# User-proposed minimal variant (verbatim, incl. the missing "that") —
# feedback was that both CONTROL_DESC and TREATMENT_DESC are too long.
SHORT_DESC = (
    "Replace lines [start..=end] (1-based, inclusive) with `content`. "
    "Replace ONLY the lines actually need to change"
)

ARMS = {
    "control": CONTROL_DESC,
    "treatment": TREATMENT_DESC,
    "small_ceiling": SMALL_CEILING_DESC,
    "count_check": COUNT_CHECK_DESC,
    "danger_first": DANGER_FIRST_DESC,
    "not_rewrite": NOT_REWRITE_DESC,
    "short": SHORT_DESC,
}


def load_moment(moment_key):
    spec = MOMENTS[moment_key]
    dump = json.load(open(spec["dump"]))
    messages = dump["messages"][: spec["cut"]]
    if spec.get("append_nudge"):
        # The 3rd identical read the model wants to make (matches
        # messages[22]/[24] in this same dump) — injected mechanically,
        # not sampled, since its args are already established by the
        # preceding two identical calls.
        synth_id = "replay-synth-repeat-read"
        messages = messages + [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "arguments": json.dumps(
                                {"action": "read", "path": "src/context/mod.rs", "start": 305, "end": 352}
                            ),
                            "name": "file",
                        },
                        "id": synth_id,
                        "type": "function",
                    }
                ],
            },
            {"role": "tool", "tool_call_id": synth_id, "content": REPEATED_READ_NUDGE},
        ]
    tools = dump["tools"]
    return messages, tools, dump["model"], dump.get("temperature", 0.2), dump.get("max_tokens", 8000)


def tools_with_desc(tools, desc):
    out = []
    for t in tools:
        if t["function"]["name"] == "replace_range":
            t = json.loads(json.dumps(t))  # deep copy
            t["function"]["description"] = desc
        out.append(t)
    return out


def call_llm(payload, timeout=180):
    req = urllib.request.Request(
        f"{ENDPOINT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def classify(resp):
    msg = resp["choices"][0]["message"]
    tcs = msg.get("tool_calls") or []
    if not tcs:
        text = (msg.get("content") or "")[:150].replace("\n", " ")
        return {"cat": "NO-TOOL-CALL", "detail": text}
    tc = tcs[0]["function"]
    name = tc.get("name", "")
    try:
        args = json.loads(tc.get("arguments", "{}"))
    except Exception:
        return {"cat": "MALFORMED-ARGS", "detail": str(tc)[:150]}
    path = str(args.get("path", ""))
    if name == "insert_at":
        return {"cat": "INSERT-AT", "detail": f"insert_at {path} after_line={args.get('after_line')}"}
    if name != "replace_range":
        return {"cat": f"OTHER-TOOL:{name}", "detail": path}
    start, end = args.get("start"), args.get("end")
    content = args.get("content", "") or ""
    try:
        size = int(end) - int(start) + 1
    except Exception:
        size = -1
    has_header = bool(re.search(r"block\.header|push_str\(header", content, re.IGNORECASE))
    brace_balance = content.count("{") - content.count("}")
    paren_balance = content.count("(") - content.count(")")
    balanced = brace_balance == 0 and paren_balance == 0
    if size < 0:
        cat = "BAD-RANGE"
    elif size <= 10:
        cat = "TINY"
    elif size <= 30:
        cat = "NARROW"
    else:
        cat = "WIDE"
    if not balanced:
        cat += "+UNBALANCED"
    return {
        "cat": cat,
        "detail": (
            f"{path} L{start}-{end} (size={size}) header_kept={has_header} "
            f"braces={brace_balance:+d} parens={paren_balance:+d} content_lines={len(content.splitlines())}"
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--moment", choices=sorted(MOMENTS), default="late")
    ap.add_argument("--control-only", action="store_true", help="skip treatment arm (baseline-failure check)")
    ap.add_argument("--arms", default=None, help="comma-separated subset of arm names (default: all)")
    ap.add_argument("--out", default=os.path.join(REPO, "benchmark_results/_moments/replace-tightness-v1"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    messages, tools, model, temperature, max_tokens = load_moment(args.moment)

    if args.control_only:
        arms = {"control": ARMS["control"]}
    elif args.arms:
        arms = {a: ARMS[a] for a in args.arms.split(",")}
    else:
        arms = ARMS
    results = {}
    for arm, desc in arms.items():
        arm_tools = tools_with_desc(tools, desc)
        samples = []
        for i in range(args.k):
            payload = {
                "model": model,
                "messages": messages,
                "tools": arm_tools,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            }
            try:
                resp = call_llm(payload)
                c = classify(resp)
            except Exception as e:
                c = {"cat": "ERROR", "detail": str(e)[:150]}
            samples.append(c)
            print(f"[{arm}] {i + 1}/{args.k}: {c['cat']}  {c.get('detail', '')}", flush=True)
        results[arm] = samples
        json.dump(results, open(os.path.join(args.out, "results.json"), "w"), indent=1)

    print("\n=== SUMMARY ===")
    from collections import Counter

    for arm, samples in results.items():
        cnt = Counter(s["cat"] for s in samples)
        print(f"{arm}: {dict(cnt)}")


if __name__ == "__main__":
    main()
