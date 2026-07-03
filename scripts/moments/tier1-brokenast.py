#!/usr/bin/env python3
"""tier1-brokenast — K-sample next-action probe on run4's broken-AST edit loop.

Moments (real decision points from replaymatrix_20260702_173202 judge/run4):
  early : dump 000025 — src/cli/mod.rs L12-22 thrash, 3 consecutive broken-AST
          block rewrites applied WITHOUT reverting (state compounding).
  regen : dump 000110 — src/cli/commands/run.rs restored to rev_3 (ast ok),
          post-debugger-diagnosis; the real run regenerated the same broken
          31-line block next. 8 broken-AST results survive in context.

Variants (pure text transforms of the request JSON — no product code):
  control        : verbatim, temp from dump (0.2)
  asthint        : every broken-AST tool result whose first syntax error lies
                   BELOW the edited range gets an explanatory hint (unbalanced
                   delimiters in the submitted replacement).
  asthint_narrow : asthint + recovery recipe (revert to last ast=ok, then make
                   the smallest possible 1-3 line edit); also appends a
                   smallest-edit hint to revert-restored results.
  temp035        : verbatim, temperature 0.35.

Scoring: first tool call of each sample. For edits on the thrash file we
compute range width, overlap with the thrash range, and (regen only) apply the
proposed content to the known rev_3 reference file and parse-check with
rustfmt (rc 0/1 = parses, else broken).
"""

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RUN4_DIR = os.path.join(
    REPO,
    "benchmark_results/replaymatrix_20260702_173202_gemma-4-26B-A4B-it-UD-Q4_K_M/judge/run4",
)
DUMPBASE = os.path.join(RUN4_DIR, "llm_dumps/req-1783009040-00052-")
# The "regen" moment's reference file — run.rs at the rev_3 (ast=ok) checkpoint
# just before the real run regenerated the broken 31-line block. Extracted
# on demand from that run's shadow-git rather than a pre-baked scratch path,
# so this script stays runnable as long as benchmark_results/ still has the
# run (gitignored — regenerate by re-running the same bench if it's gone).
REF_RUN_RS = os.path.join(HERE, ".cache_run4_r74_run_rs")


def _extract_ref_run_rs():
    if os.path.exists(REF_RUN_RS):
        return
    shadow_git = os.path.join(RUN4_DIR, "miniswe_state/shadow-git")
    # The container ran as root, so the shadow-git dir is root-owned — bypass
    # git's ownership check explicitly rather than relying on ambient global
    # config (`safe.directory`) being set on whatever machine runs this.
    safe = ["-c", "safe.directory=*"]
    out = subprocess.run(
        ["git", *safe, "log", "--oneline", "--all", "--grep=before round 74$"],
        cwd=shadow_git, capture_output=True, text=True,
    ).stdout
    commit = out.split()[0] if out.split() else None
    if not commit:
        raise RuntimeError(f"could not find round-74 commit in {shadow_git}")
    blob = subprocess.run(
        ["git", *safe, "--git-dir", shadow_git, "cat-file", "-p",
         f"{commit}:src/cli/commands/run.rs"],
        capture_output=True, text=True,
    ).stdout
    with open(REF_RUN_RS, "w") as f:
        f.write(blob)


ENDPOINT = os.environ.get("LLAMA_ENDPOINT", "http://localhost:8464")

MOMENTS = {
    "early": {
        "dump": "000025",
        "file": "src/cli/mod.rs",
        "thrash_range": (12, 25),
        "block_width": 8,
        "reference": None,  # current state is broken; no parse simulation
    },
    "regen": {
        "dump": "000110",
        "file": "src/cli/commands/run.rs",
        "thrash_range": (131, 160),
        "block_width": 20,
        "reference": REF_RUN_RS,
    },
}

HDR_RE = re.compile(r"replace_range (\S+) L(\d+)-(\d+): rev_\d+ applied \(\+(\d+) -\d+\)")
AST_RE = re.compile(r"\[ast\] broken: (\d+):\d+")
REVERT_RE = re.compile(r"revert (\S+) → rev_\d+: restored")


def hint_a(file, start, new_end, err):
    return (
        f"[hint] This file parsed cleanly BEFORE this edit; the first syntax error "
        f"(L{err}) is BELOW the range you replaced (L{start}-L{new_end}), so the "
        f"replacement text you submitted has unbalanced braces — an extra '}}' or a "
        f"missing '{{'. Resubmitting the same block will break the file the same way."
    )


def hint_b(file):
    return (
        f" Recover in two steps: (1) revert(path='{file}') to the last ast=ok revision "
        f"shown in the table; (2) then fix the ORIGINAL problem with the smallest "
        f"possible edit — change only the 1-3 lines that must differ (replace_range on "
        f"a narrow range, or insert_at). Do NOT rewrite the whole block."
    )


REVERT_HINT = (
    "[hint] Restored to a parsing state. Previous whole-block rewrites of this file "
    "kept breaking it — now make the SMALLEST possible edit: change only the 1-3 "
    "lines that must differ (replace_range on a narrow range, or insert_at), keeping "
    "braces balanced. Do NOT resubmit the whole block."
)


def transform(messages, variant):
    """Apply the variant's text transform to every qualifying tool result."""
    if variant in ("control", "temp035"):
        return messages
    out = copy.deepcopy(messages)
    for m in out:
        c = m.get("content")
        if m.get("role") != "tool" or not isinstance(c, str):
            continue
        if variant == "reverthint":
            # isolate the post-revert hint: no changes to broken-AST results
            if "restored" in c and "[ast] ok" in c and REVERT_RE.search(c):
                m["content"] = c.rstrip() + "\n" + REVERT_HINT
            continue
        if "[ast] broken" in c:
            h = HDR_RE.search(c)
            a = AST_RE.search(c)
            if h and a:
                file, start, _end, plus = h.group(1), int(h.group(2)), int(h.group(3)), int(h.group(4))
                new_end = start + plus - 1
                err = int(a.group(1))
                if err > new_end:
                    hint = hint_a(file, start, new_end, err)
                    if variant == "asthint_narrow":
                        hint += hint_b(file)
                    lines = c.splitlines()
                    for i, ln in enumerate(lines):
                        if ln.startswith("[ast] broken"):
                            lines.insert(i + 1, hint)
                            break
                    m["content"] = "\n".join(lines)
        elif variant == "asthint_narrow" and "restored" in c and "[ast] ok" in c and REVERT_RE.search(c):
            m["content"] = c.rstrip() + "\n" + REVERT_HINT
    return out


def call_llm(payload, timeout=180):
    req = urllib.request.Request(
        f"{ENDPOINT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def get_arg(args, *names, default=None):
    for n in names:
        if n in args:
            return args[n]
    return default


def parse_check(reference_path, start, end, content):
    """Apply replace_range(start, end, content) to the reference file, rustfmt-parse."""
    ref = open(reference_path).read().splitlines()
    if not (1 <= start <= len(ref)):
        return None
    new = ref[: start - 1] + content.splitlines() + ref[end:]
    with tempfile.NamedTemporaryFile("w", suffix=".rs", delete=False) as f:
        f.write("\n".join(new) + "\n")
        path = f.name
    try:
        rc = subprocess.run(
            ["rustfmt", "--edition", "2021", "--check", path],
            capture_output=True, timeout=30,
        ).returncode
        return rc in (0, 1)
    except Exception:
        return None
    finally:
        os.unlink(path)


def classify(resp, moment):
    msg = resp["choices"][0]["message"]
    tcs = msg.get("tool_calls") or []
    if not tcs:
        return {"cat": "PROSE", "detail": (msg.get("content") or "")[:80]}
    tc = tcs[0]["function"]
    name = tc["name"]
    try:
        args = json.loads(tc["arguments"])
    except Exception:
        return {"cat": "MALFORMED", "detail": name}
    if name == "revert":
        return {"cat": "REVERT", "detail": str(get_arg(args, "path", "file"))}
    if name in ("replace_range", "insert_at", "write_file"):
        path = str(get_arg(args, "path", "file", default=""))
        start = get_arg(args, "start", "start_line")
        end = get_arg(args, "end", "end_line", default=start)
        content = str(get_arg(args, "content", "text", "new_content", default=""))
        on_target = path.endswith(os.path.basename(moment["file"])) and moment["file"].split("/")[-2] in path
        width = 1
        if name == "replace_range" and isinstance(start, int) and isinstance(end, int):
            width = end - start + 1
        t0, t1 = moment["thrash_range"]
        overlaps = isinstance(start, int) and not (end < t0 or start > t1) if isinstance(end, int) else False
        block = on_target and overlaps and width >= moment["block_width"]
        pok = None
        if moment["reference"] and on_target and name == "replace_range" \
                and isinstance(start, int) and isinstance(end, int) and content:
            pok = parse_check(moment["reference"], start, end, content)
        cat = "BLOCK-REWRITE" if block else ("EDIT-NARROW" if on_target and width <= 10 else
                                             ("EDIT-TARGET" if on_target else "EDIT-OTHERFILE"))
        return {"cat": cat, "detail": f"{name} {path} L{start}-{end} w={width}",
                "width": width, "parse_ok": pok, "start": start, "end": end,
                "content": content[:4000]}
    if name == "file":
        return {"cat": "READ", "detail": str(get_arg(args, "action", default=""))[:60]}
    if name == "plan":
        return {"cat": "PLAN", "detail": str(get_arg(args, "action", default=""))}
    return {"cat": name.upper(), "detail": json.dumps(args)[:60]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--moments", default="early,regen")
    ap.add_argument("--variants", default="control,asthint,asthint_narrow,temp035")
    ap.add_argument("--out", default=os.path.join(REPO, "benchmark_results/_moments/brokenast-v1"))
    args = ap.parse_args()

    if "regen" in args.moments.split(","):
        _extract_ref_run_rs()

    os.makedirs(args.out, exist_ok=True)
    results = {}
    for mname in args.moments.split(","):
        m = MOMENTS[mname]
        dump = json.load(open(DUMPBASE + m["dump"] + ".json"))
        for variant in args.variants.split(","):
            key = f"{mname}/{variant}"
            samples = []
            msgs = transform(dump["messages"], variant)
            payload = {
                "model": dump["model"],
                "messages": msgs,
                "tools": dump["tools"],
                "temperature": 0.35 if variant == "temp035" else dump.get("temperature", 0.2),
                "max_tokens": dump.get("max_tokens", 8000),
                "stream": False,
            }
            if "chat_template_kwargs" in dump:
                payload["chat_template_kwargs"] = dump["chat_template_kwargs"]
            for i in range(args.k):
                try:
                    resp = call_llm(payload)
                    c = classify(resp, m)
                except Exception as e:
                    c = {"cat": "ERROR", "detail": str(e)[:80]}
                samples.append(c)
                print(f"[{key}] {i+1}/{args.k}: {c['cat']}  {c.get('detail','')}", flush=True)
            results[key] = samples
            json.dump(results, open(os.path.join(args.out, "results.json"), "w"), indent=1)

    # summary table
    print("\n=== SUMMARY ===")
    cats = ["BLOCK-REWRITE", "EDIT-NARROW", "EDIT-TARGET", "EDIT-OTHERFILE",
            "REVERT", "READ", "PLAN", "PROSE"]
    hdr = f"{'moment/variant':<28}" + "".join(f"{c:>15}" for c in cats) + f"{'parse-ok/edits':>16}"
    print(hdr)
    for key, samples in results.items():
        row = f"{key:<28}"
        n = len(samples)
        for c in cats:
            k = sum(1 for s in samples if s["cat"] == c)
            row += f"{(f'{k}/{n}' if k else '-'):>15}"
        edits = [s for s in samples if s.get("parse_ok") is not None]
        pok = sum(1 for s in edits if s["parse_ok"])
        row += f"{(f'{pok}/{len(edits)}' if edits else '-'):>16}"
        print(row)


if __name__ == "__main__":
    main()
