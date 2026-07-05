#!/usr/bin/env python3
"""tier1-signal-visibility — does making corrective guidance more prominent
(vs. shorter) change whether the model acts on it?

Three real moments from the unified-arm 4/6 forensics (2026-07-04,
compaction_20260704_164112, run3 + run4), all variants of the SAME bug:
`context::assemble()`'s call site hardcodes `None` instead of threading
`system_prompt_override`, silently no-opping the feature with zero compiler
warning under the bench's RUSTFLAGS=-A warnings.

  moment_a (run3, dump8, 18 msgs): a refactor-tool rejection gives near-
    verbatim correct guidance ("EDIT the specific callsite... replace its
    None placeholder with the actual expression"). The REAL run ignored it
    outright, writing None with zero reasoning, at only 18 messages of
    context — already short. Tests whether repeating/emphasizing a signal
    that's already maximally prominent (last message, on-topic) helps at all,
    since "shorten it" isn't available as a lever here (it's already short).

  moment_b (run3, dump12, 26 msgs): a genuine `warning: unused variable:
    system_prompt_override` sits in the SAME tool result as 14 hard E0061
    test-arity errors. The real run chased only the hard errors. Tests BOTH
    hypotheses: decluttered (remove the competing noise) vs repositioned
    (keep everything, move+label the warning prominently).

  moment_c (run4, attempt 3 start, 2 msgs): a REAL bench-harness bug —
    `head -40` truncation merges the test-failure listing and the smoke-
    failure message with no separator, so "SMOKE TEST FAILED... override is
    being silently ignored" visually reads as a continuation of one test's
    stdout rather than a separate, primary signal. Tests whether a clean
    fix (clear section break + priority label, no other content change)
    redirects the model to investigate the actual callsites instead of
    assemble()'s internals (which is what the real run did instead).

Variants are pure text transforms on the captured request JSON — no product
code involved. Ground truth per moment = does the sampled response correctly
target BOTH callsites (run.rs AND repl.rs) with the real value, one of them,
or neither.

Usage: tier1-signal-visibility.py --k 12 [--moments a,b,c] [--variants ...]
"""

import argparse
import copy
import json
import os
import re
import urllib.request

REPO = "/home/michal/dev/miniswe"
ENDPOINT = os.environ.get("LLAMA_ENDPOINT", "http://localhost:8464")

BASE = os.path.join(
    REPO, "benchmark_results/compaction_20260704_164112__home_michal_models_gemma-4-26B-A4B-it-G"
)
DUMP_A = os.path.join(BASE, "unified/run3/llm_dumps/req-1783179507-36182-000008.json")
DUMP_B = os.path.join(BASE, "unified/run3/llm_dumps/req-1783179507-36182-000012.json")
DUMP_C = os.path.join(BASE, "unified/run4/llm_dumps/req-1783182209-47357-000000.json")

REJECTION_TEXT = (
    "✗ add_param: `assemble` already has a parameter named `system_prompt_override` "
    "— not adding a duplicate (that would stack another `None` argument at every "
    "callsite and break the build). If a value is not being threaded through, the fix is "
    "NOT to add the parameter again: EDIT the specific callsite that should pass the real "
    "value (replace its `None` placeholder with the actual expression), then re-run your check."
)
REJECTION_EMPHASIZED = (
    "✗ add_param: `assemble` already has a parameter named `system_prompt_override` "
    "— not adding a duplicate (that would stack another `None` argument at every "
    "callsite and break the build).\n"
    "IMPORTANT — READ THIS BEFORE YOUR NEXT TOOL CALL: the fix is NOT to add the "
    "parameter again. You must EDIT the specific callsite(s) that should pass the real "
    "value: find every place that calls `context::assemble(...)` and replace the literal "
    "`None` placeholder argument with the actual `system_prompt_override` variable. Do this "
    "in BOTH `src/cli/commands/run.rs` AND `src/cli/commands/repl.rs` — there are TWO "
    "callsites, not one. Then re-run your check.\n"
    "REMINDER: the fix is NOT to add the parameter again — EDIT the callsite(s)."
)


def load(path):
    return json.load(open(path))


def variant_a(dump, variant):
    d = copy.deepcopy(dump)
    if variant == "control":
        return d
    if variant == "emphasized":
        msgs = d["messages"]
        msgs[-1]["content"] = msgs[-1]["content"].replace(REJECTION_TEXT, REJECTION_EMPHASIZED)
        return d
    raise ValueError(variant)


def general_dedup_lines(text, min_repeat=3, keep_example=True, fold_matching_totals=False):
    """Language-agnostic: collapse digit runs (line/col numbers, error codes,
    counts) so lines differing only in POSITION map to the same template;
    any template with >= min_repeat occurrences collapses down. No knowledge
    of any specific language or error format.
    keep_example=True: first occurrence verbatim + a count of the rest.
    keep_example=False: no verbatim text survives, just a bare count note.
    fold_matching_totals: after clustering, any OTHER line containing one of
    the collapsed cluster sizes as a standalone number is treated as a
    redundant restatement of that same cluster (e.g. a compiler's trailing
    "N errors" tally) and is dropped too — no knowledge of "error"/"compile"
    wording, purely "this number already accounted for a cluster I removed"."""
    lines = text.split("\n")
    counts = {}
    for l in lines:
        key = re.sub(r"\d+", "#", l.strip())
        counts[key] = counts.get(key, 0) + 1
    out, collapsed = [], set()
    cluster_sizes = {v for v in counts.values() if v >= min_repeat}
    for l in lines:
        key = re.sub(r"\d+", "#", l.strip())
        if counts[key] >= min_repeat and key:
            if key in collapsed:
                continue
            collapsed.add(key)
            if keep_example:
                out.append(l)
                out.append(f"... (+{counts[key]-1} more lines matching this same pattern, elsewhere)")
            else:
                out.append(f"[{counts[key]} lines of the same repeated pattern collapsed here]")
        elif (
            fold_matching_totals
            and cluster_sizes
            and any(int(n) in cluster_sizes for n in re.findall(r"\d+", l))
        ):
            continue  # redundant restatement of an already-collapsed cluster's size
        else:
            out.append(l)
    return "\n".join(out)


def variant_b(dump, variant):
    d = copy.deepcopy(dump)
    if variant == "control":
        return d
    content = d["messages"][-1]["content"]
    if variant == "general_dedup":
        d["messages"][-1]["content"] = general_dedup_lines(content)
        return d
    if variant == "general_dedup_noexample":
        # Pure general transform: no Rust-specific phrase matching, just the
        # same digit-collapse clustering as general_dedup, but the collapsed
        # clusters carry no verbatim repeated-error text at all.
        d["messages"][-1]["content"] = general_dedup_lines(content, keep_example=False)
        return d
    if variant == "general_dedup_fold_totals":
        d["messages"][-1]["content"] = general_dedup_lines(
            content, keep_example=True, fold_matching_totals=True
        )
        return d
    if variant == "general_dedup_full_removal":
        d["messages"][-1]["content"] = general_dedup_lines(
            content, keep_example=False, fold_matching_totals=True
        )
        return d
    warning_re = re.compile(
        r"warning: unused variable: `system_prompt_override`.*?on by default\n", re.DOTALL
    )
    m = warning_re.search(content)
    warning_block = m.group(0) if m else None
    if variant == "decluttered":
        # Remove the 14 hard E0061 lines and the surrounding compile noise,
        # keep the unused-variable warning as the sole, prominent signal.
        lines = content.splitlines()
        kept = []
        skip_arity = False
        for ln in lines:
            if "this function takes 6 arguments but 5 arguments were supplied" in ln:
                skip_arity = True
                continue
            if skip_arity and ln.strip() == "":
                skip_arity = False
                continue
            if skip_arity:
                continue
            kept.append(ln)
        d["messages"][-1]["content"] = "\n".join(kept)
        return d
    if variant == "repositioned":
        if not warning_block:
            return d
        rest = content[: m.start()] + content[m.end() :]
        labeled = (
            "\n=== HIGH-PRIORITY: UNUSED VARIABLE WARNING (check this first) ===\n"
            + warning_block.strip()
            + "\nThis warning means `system_prompt_override` is dead code somewhere — "
            "accepted as a parameter but never read. Find where it should be used and isn't.\n"
            "=== END HIGH-PRIORITY WARNING ===\n"
        )
        d["messages"][-1]["content"] = rest.rstrip() + "\n" + labeled
        return d
    raise ValueError(variant)


def variant_c(dump, variant):
    d = copy.deepcopy(dump)
    content = d["messages"][-1]["content"]
    if variant == "control":
        return d
    if variant == "fixed_formatting":
        marker = "---- assembly_includes_guide stdout ----\nSMOKE TEST FAILED."
        if marker not in content:
            return d
        before, _, after_marker = content.partition(marker)
        # before ends with the truncated "---- assembly_includes_guide stdout ----" header
        # with no panic body (the head -40 cutoff). Cleanly close that test's block, then
        # give the smoke failure its own clearly separated, labeled section.
        before_clean = before.rsplit("---- assembly_includes_guide stdout ----", 1)[0].rstrip()
        smoke_part = "SMOKE TEST FAILED." + after_marker
        new_content = (
            before_clean
            + "\n(... other test output truncated ...)\n\n"
            + "=== SEPARATE CHECK — THE ACTUAL FEATURE BEHAVIOR (distinct from the unit "
            "tests above; this is the primary signal) ===\n"
            + smoke_part.strip()
            + "\n=== END BEHAVIORAL CHECK ==="
        )
        d["messages"][-1]["content"] = new_content
        return d
    raise ValueError(variant)


MOMENTS = {
    "a": (DUMP_A, variant_a, ["control", "emphasized"]),
    "b": (DUMP_B, variant_b, ["general_dedup_full_removal"]),
    "c": (DUMP_C, variant_c, ["control", "fixed_formatting"]),
}


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
        return {"cat": "PROSE", "detail": (msg.get("content") or "")[:100]}
    tc = tcs[0]["function"]
    name = tc["name"]
    try:
        args = json.loads(tc["arguments"])
    except Exception:
        return {"cat": "MALFORMED", "detail": name}
    path = str(args.get("path", ""))
    content = str(args.get("content", ""))
    if name == "refactor" and args.get("action") == "add_param":
        return {"cat": "REPEAT-ADD-PARAM", "detail": "refactor add_param again (the rejected move)"}
    if name in ("replace_range", "insert_at", "write_file"):
        targets_run = "run.rs" in path
        targets_repl = "repl.rs" in path
        fixes_it = "system_prompt_override" in content and "None" not in content.split(
            "system_prompt_override"
        )[0][-40:]
        on_target = targets_run or targets_repl
        if on_target and "None" in content and "system_prompt_override," not in content.replace(
            " ", ""
        ).replace("\n", ""):
            cat = "WRITES-NONE-AGAIN" if "None" in content else "EDIT-TARGET"
        elif on_target:
            cat = "EDIT-TARGET-run" if targets_run else "EDIT-TARGET-repl"
        else:
            cat = "EDIT-OTHER-FILE"
        return {"cat": cat, "detail": f"{name} {path} w={len(content)}"}
    if name == "file":
        p = str(args.get("path", ""))
        on_target = "run.rs" in p or "repl.rs" in p
        return {"cat": "READ-TARGET" if on_target else "READ-OTHER", "detail": p}
    return {"cat": name.upper(), "detail": json.dumps(args)[:80]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--moments", default="a,b,c")
    ap.add_argument("--out", default=os.path.join(REPO, "benchmark_results/_moments/signal-visibility-v1"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    results = {}
    for mkey in args.moments.split(","):
        dump_path, variant_fn, variants = MOMENTS[mkey]
        dump = load(dump_path)
        for variant in variants:
            key = f"moment_{mkey}/{variant}"
            d = variant_fn(dump, variant)
            samples = []
            for i in range(args.k):
                payload = {
                    "model": d["model"],
                    "messages": d["messages"],
                    "tools": d.get("tools", []),
                    "temperature": d.get("temperature", 0.2),
                    "max_tokens": d.get("max_tokens", 8000),
                    "stream": False,
                }
                if "chat_template_kwargs" in d:
                    payload["chat_template_kwargs"] = d["chat_template_kwargs"]
                try:
                    resp = call_llm(payload)
                    c = classify(resp)
                except Exception as e:
                    c = {"cat": "ERROR", "detail": str(e)[:100]}
                samples.append(c)
                print(f"[{key}] {i+1}/{args.k}: {c['cat']}  {c.get('detail','')}", flush=True)
            results[key] = samples
            json.dump(results, open(os.path.join(args.out, "results.json"), "w"), indent=1)

    print("\n=== SUMMARY ===")
    for key, samples in results.items():
        from collections import Counter

        cnt = Counter(s["cat"] for s in samples)
        print(f"{key}: {dict(cnt)}")


if __name__ == "__main__":
    main()
