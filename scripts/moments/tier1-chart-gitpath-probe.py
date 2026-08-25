#!/usr/bin/env python3
"""tier1-chart-gitpath-probe — the run-222918 deploy blocker + the "scoped
sub-agent" idea, in one 3-arm experiment.

Ground truth (verified): prompt gives repo https://github.com/defenseunicorns/uds-mcp
(public), chart in subdir tests/e2e/app-with-deps/src/chart, ref `main` (a real
branch), chart version 0.1.0 (Chart.yaml). The skill (uds-package-build SKILL.md:88)
states the zarf git-chart convention: url = `<repo>.git@<ref>` PLUS a SEPARATE
`gitPath` for the subdir. The model concatenated the subdir INTO the url and
deploy failed:
    unable to pull the chart ... failed to locate tag main: tag not found
    failed to fetch ...uds-mcp.git@main/tests/e2e/app-with-deps/src/chart/ : 404
So the correct fix: url = https://github.com/defenseunicorns/uds-mcp.git@main ,
gitPath = tests/e2e/app-with-deps/src/chart (separate), version = 0.1.0.

Arms (all fresh context, k each):
  A  reactive debugger, ERROR ONLY            — what the current harness debugger sees
  B  reactive debugger, ERROR + skill snippet — an improved reactive debugger
  C  clean scoped sub-agent, DISTILLED TASK + skill snippet, no failure framing
     — the user's idea: separate agent, no shared/poisoned context, just "find the
       correct chart source for X"

CORRECT = keeps the subdir in a SEPARATE gitPath and does NOT concatenate it into
the url. Grades CORRECT / OTHER / MISS. Usage: [--k 12]
"""

import argparse
import json
import os
import re
import urllib.request

ENDPOINT = os.environ.get("LLAMA_ENDPOINT", "http://localhost:8464")

WRONG_ZARF = """  - name: app-with-dependencies
    charts:
      - name: app-with-dependencies
        url: https://github.com/defenseunicorns/uds-mcp.git@main/tests/e2e/app-with-deps/src/chart
        version: 0.1.0
        namespace: app-with-deps"""

FAILURE = """The deploy keeps failing. `uds run dev` (which runs `uds zarf package create` then deploys) fails when it tries to pull the chart:

$ uds run dev
...
unable to pull the chart "app-with-dependencies" from git: failed to locate tag main in repository: tag not found
Error: failed to fetch https://github.com/defenseunicorns/uds-mcp.git@main/tests/e2e/app-with-deps/src/chart/ : 404 Not Found
[exit 1]

The current chart entry in common/zarf.yaml:
""" + WRONG_ZARF + """

Facts: the repo https://github.com/defenseunicorns/uds-mcp is public and `main` is a valid branch; the chart really is at tests/e2e/app-with-deps/src/chart; its Chart.yaml version is 0.1.0."""

# convention only — does NOT hand over the literal url/gitPath values; model must apply it
SNIPPET = """
Reference (zarf git-chart convention): for a Helm chart stored inside a git repo, the `url:` must be the git repo URL ending in `.git`, optionally with the ref appended as `.git@<ref>` (e.g. `@v1.2.3` or `@main`). The chart's SUBDIRECTORY within the repo must go in a SEPARATE `gitPath:` field — it must NOT be appended onto the url. `version:` is the chart version from Chart.yaml."""

SYS_DEBUG = (
    "You are a debugger with fresh eyes brought in to unstick an agent that has repeated a "
    "failing deploy several times and cannot fix it. Diagnose the ROOT CAUSE and give the ONE "
    "specific, concrete fix (the exact corrected chart entry). Be decisive and brief."
)
SYS_SCOPED = (
    "You are a focused helper answering ONE narrow question for another agent. You share no "
    "context with it. Answer only what is asked, concretely and briefly: give the exact YAML."
)

SCOPED_TASK = """Give me the correct zarf.yaml `charts:` entry for a Helm chart that lives inside a git repository.

- Git repo: https://github.com/defenseunicorns/uds-mcp
- Chart subdirectory in the repo: tests/e2e/app-with-deps/src/chart
- Git ref (branch): main
- Chart name: app-with-dependencies
- Chart version (from Chart.yaml): 0.1.0
""" + SNIPPET + "\n\nOutput the exact `charts:` entry (url, gitPath, version, name)."

ARMS = {
    "A_debug_bare": [
        {"role": "system", "content": SYS_DEBUG},
        {"role": "user", "content": FAILURE},
    ],
    "B_debug_snippet": [
        {"role": "system", "content": SYS_DEBUG},
        {"role": "user", "content": FAILURE + "\n" + SNIPPET},
    ],
    "C_scoped_subagent": [
        {"role": "system", "content": SYS_SCOPED},
        {"role": "user", "content": SCOPED_TASK},
    ],
}


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


SUBDIR = "tests/e2e/app-with-deps/src/chart"
# the bug: subdir concatenated onto the url (after .git or .git@ref)
URL_CONCAT = re.compile(r"\.git(@[\w./+-]+)?/tests/e2e/app-with-deps/src/chart", re.I)
# a gitPath field carrying the subdir, on its own
GITPATH_OK = re.compile(r"gitPath\s*:?\s*\.?/?" + re.escape(SUBDIR), re.I)


def grade(text):
    low = text.lower()
    concat = URL_CONCAT.search(text) is not None
    gitpath = GITPATH_OK.search(text) is not None
    mentions_gitpath = "gitpath" in low
    if gitpath and not concat:
        return "CORRECT"
    # identified the separation issue but didn't produce a clean entry,
    # or told them to regenerate cleanly with dev generate + gitPath
    if mentions_gitpath and not concat:
        return "OTHER"
    if ("gitpath" in low or "separate" in low or "dev generate" in low) and concat:
        return "OTHER"  # names the fix but still shows the concatenated url
    return "MISS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12)
    args = ap.parse_args()
    m = model()
    print(f"model: {m}\n")
    for arm, msgs in ARMS.items():
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
            if sample is None or (v == "CORRECT" and "CORRECT" not in (sample or "")):
                sample = f"{v}: {out.strip()[:260]}"
            print(f"  {arm} {i+1}/{args.k}: {v}", flush=True)
        print(f"\n== {arm}: CORRECT {counts['CORRECT']}/{args.k}  OTHER {counts['OTHER']}  MISS {counts['MISS']}")
        print(f"   sample: {sample}\n")


if __name__ == "__main__":
    main()
