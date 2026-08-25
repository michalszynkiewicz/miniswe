#!/usr/bin/env python3
"""tier1-debugger-zarf-fail-probe — would a fresh-context debugger unstick the
zarf-create failure that ground down the 144442 e2e?

The model looped on `zarf package create` / lint failing with:
    WRN flavor not used in package
    ERR package validation failed: package does not contain any compatible components
because the root zarf.yaml gates its component to `only: {flavor: upstream}`,
so create/lint WITHOUT `--flavor upstream` matches zero components. Ground
truth (verified): `zarf dev lint . --flavor upstream` → clean. The fix is to
pass `--flavor upstream` (the skill even says so; the model dropped the flag).

The proposed harness lever is a LOOP debugger for FAILING loops: when the same
command keeps failing, spin up a fresh-context sub-agent with the error to
diagnose + fix, instead of cold-prefilling (which can't fix a real error).
This probes the core assumption: given the failure + the zarf.yaml, does a
fresh-context agent correctly diagnose "add --flavor upstream"?

Grades DONE-style: CORRECT (recommends passing/setting flavor upstream),
OTHER (a different plausible fix — e.g. remove the `only.flavor` gate), MISS
(doesn't identify the flavor/compatible-components root cause). Usage: [--k 10]
"""

import argparse
import json
import os
import re
import urllib.request

ENDPOINT = os.environ.get("LLAMA_ENDPOINT", "http://localhost:8464")

ZARF_YAML = """# yaml-language-server: $schema=https://.../zarf.schema.json
kind: ZarfPackageConfig
metadata:
  name: app-with-deps
  version: v0.6.0
components:
  - name: app-with-deps
    import:
      path: common
    only:
      flavor: upstream
"""

FAILURE = """The following command has now FAILED several times in a row; the agent is stuck retrying it:

$ uds zarf package create . --skip-sbom --confirm
WRN  flavor not used in package  flavor=
ERR  package validation failed: package does not contain any compatible components
[exit 1]

The package's root zarf.yaml:
""" + ZARF_YAML + """
(common/zarf.yaml defines the actual chart components, each also gated behind the same flavor.)"""

SYS = (
    "You are a debugger with fresh eyes brought in to unstick an agent that has repeated a "
    "failing command several times and cannot fix it. Diagnose the ROOT CAUSE of the failure "
    "and give the ONE specific, concrete fix (an exact command or edit). Be decisive and brief; "
    "do not restate the whole situation."
)


def call_llm(payload, timeout=300):
    req = urllib.request.Request(
        f"{ENDPOINT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def model():
    with urllib.request.urlopen(f"{ENDPOINT}/v1/models", timeout=30) as r:
        return json.load(r)["data"][0]["id"]


FLAVOR_FIX = re.compile(r"--flavor\s+upstream|flavor[ =:]+upstream|pass(ing)?\s+the\s+flavor|set\s+the\s+flavor", re.I)
REMOVE_GATE = re.compile(r"remove.*flavor|delete.*only|drop.*flavor|without the .*flavor|only:.*remov", re.I)


def grade(text):
    low = text.lower()
    mentions_flavor = "flavor" in low
    if FLAVOR_FIX.search(text):
        return "CORRECT"
    if REMOVE_GATE.search(text) or (mentions_flavor and "remove" in low):
        return "OTHER"
    if mentions_flavor and "compatible component" in low:
        return "OTHER"  # identified the cause but no clear actionable flavor fix
    return "MISS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    m = model()
    print(f"model: {m}\n")
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": FAILURE}]
    counts = {"CORRECT": 0, "OTHER": 0, "MISS": 0}
    sample = None
    for i in range(args.k):
        payload = {"model": m, "messages": msgs, "temperature": 0.2, "max_tokens": 4000, "stream": False}
        try:
            out = call_llm(payload)["choices"][0]["message"].get("content") or ""
        except Exception as e:
            out = f"(err {str(e)[:40]})"
        v = grade(out)
        counts[v] += 1
        if sample is None or v == "CORRECT":
            sample = f"{v}: {out.strip()[:220]}"
        print(f"  {i+1}/{args.k}: {v}", flush=True)
    print(f"\n== debugger diagnosis: CORRECT {counts['CORRECT']}/{args.k}  "
          f"OTHER {counts['OTHER']}  MISS {counts['MISS']}")
    print(f"   sample: {sample}")


if __name__ == "__main__":
    main()
