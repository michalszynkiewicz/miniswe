#!/usr/bin/env python3
"""tier1-scaffold-targetdir-probe — does the model build the package in the
WORKSPACE ROOT, or bury it inside the cloned source repo?

Pinpointed failure (uds e2e 2026-07-16, run app-with-deps): the model clones
the chart repo to `tmp_repo/` (per SKILL.md line 69 "(clone the repo to
read)"), then anchors its whole build there — it called scaffold/zarf-generate
with `targetDir=tmp_repo/tests/e2e/app-with-deps/zarf` and
`chartRepoURL=file://tmp_repo/...`, so zarf.yaml landed 4-5 levels deep and
validation (which finds zarf.yaml only at workspace root or one level deep)
never saw it. The ScaffoldZarf / WriteZarfYaml steps tell the model to pass
`targetDir` but NEVER say it should be `.` (the workspace root), and the
"clone to read" line never says the clone is read-only.

Hypothesis: an explicit working-directory anchor on those steps fixes where
the model builds. Arms:
  A (baseline)  — current skill wording.
  B (anchored)  — + "targetDir MUST be `.` (workspace root); the clone is
                  READ-ONLY, never build inside it; chartRepoURL is the REMOTE
                  url, not file://<clone>".

Scenario given to the model: repo already cloned to tmp_repo/ to read the
chart (version 0.1.0); it is on the ScaffoldZarf step; emit the
scaffold-package call as JSON {targetDir, applicationName, chartRepoURL}.
Grade the targetDir (root vs buried) and chartRepoURL (remote vs file/clone).

Usage: [--k 10]
"""

import argparse
import json
import os
import re
import urllib.request

ENDPOINT = os.environ.get("LLAMA_ENDPOINT", "http://localhost:8464")
SKILL = "/home/michal/work/uds-mcp/src/todo-skills/uds-package-build/SKILL.md"

# FAITHFUL scenario: reproduce the real trap — a nearly-empty workspace root
# and a freshly-cloned repo whose internal `tests/e2e/app-with-deps/` looks
# exactly like the app's package home. Crucially it does NOT tell the model
# which directory is the package root (naming it makes the model 10/10 even at
# baseline — the first, too-clean version of this probe). The model must
# decide targetDir against the pull of the seductive clone structure.
SCENARIO = (
    "Task: create a UDS package for \"app-with-dependencies\". The Helm chart is at "
    "https://github.com/defenseunicorns/uds-mcp (ref: main), in the repo subdirectory "
    "tests/e2e/app-with-deps/src/chart/. Container image: "
    "ghcr.io/michalszynkiewicz/app-with-dependencies:0.1.0.\n\n"
    "You have explored the workspace. Layout (`ls -R`, abbreviated):\n"
    "./\n"
    "  CLAUDE.md\n"
    "  TODO.md\n"
    "  tmp_repo/            # you cloned https://github.com/defenseunicorns/uds-mcp here\n"
    "    README.md\n"
    "    src/ ...\n"
    "    tests/e2e/app-with-deps/\n"
    "      validate.sh\n"
    "      src/chart/        # Chart.yaml (v0.1.0), values.yaml, templates/\n\n"
    "You are now on the ScaffoldZarf step. Emit ONLY the scaffold-package call as a JSON "
    'object with exactly the keys {"targetDir": ..., "applicationName": ..., '
    '"chartRepoURL": ...}. No prose, no code fence.'
)

# Arm C: a harness-injected current-state line (the miniswe-side lever — this
# would live in the [CURRENT STATE] block whenever a skill cursor is active).
DELIVERABLE = (
    "[CURRENT STATE]\n[DELIVERABLE] Build the UDS package at the workspace root — the current "
    "working directory `.`. `tmp_repo/` is a READ-ONLY clone (only for reading the chart); "
    "never create package files inside it.\n\n"
)

ANCHOR = (
    "\n\n### CRITICAL — where to build\n"
    "`targetDir` MUST be `.` (the current working directory = the package root), or a NEW "
    "subdirectory directly under it. NEVER build inside a cloned source repo. Any clone "
    "(e.g. `tmp_repo/`) is READ-ONLY — it exists only to read Chart.yaml. `chartRepoURL` is "
    "the REMOTE git/OCI URL, never a local `file://` path into a clone.\n"
)


def read_skill():
    return open(SKILL).read()


def scaffold_section(body):
    # Feed the model the ChartURL..WriteZarfYaml span (the relevant steps).
    i = body.find("### Step ValidateChart")
    j = body.find("### Step ", body.find("### Step WriteZarfYaml") + 10)
    j = j if j > 0 else len(body)
    return body[i:j] if i >= 0 else body[:6000]


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


def parse_obj(raw):
    t = (raw or "").strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    s = t.find("{")
    e = t.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        return json.loads(t[s : e + 1])
    except Exception:
        return None


def grade(obj):
    if not obj:
        return {"parsed": False}
    td = str(obj.get("targetDir", "")).strip().strip("/")
    url = str(obj.get("chartRepoURL", "")).strip()
    # root = ".", "", or a single non-clone segment (validation finds zarf.yaml
    # at workspace root or one level deep).
    segs = [p for p in td.split("/") if p and p != "."]
    root_target = ("tmp_repo" not in td) and (len(segs) <= 1)
    remote_url = (
        "file://" not in url
        and "tmp_repo" not in url
        and (url.startswith(("http", "oci://")) or url.endswith(".git") or "github.com" in url)
    )
    return {
        "parsed": True,
        "targetDir": td or ".",
        "url": url[:60],
        "root_target": root_target,
        "remote_url": remote_url,
        "both": root_target and remote_url,
    }


def run_arm(name, sys_material, m, k, user=SCENARIO):
    print(f"\n===== ARM {name} =====")
    msgs = [
        {"role": "system", "content": "You are executing ONE step of a UDS packaging skill. "
         "Follow the skill material exactly.\n\n" + sys_material},
        {"role": "user", "content": user},
    ]
    root = remote = both = parsed = 0
    sample = None
    for i in range(k):
        payload = {"model": m, "messages": msgs, "temperature": 0.2,
                   "max_tokens": 4000, "stream": False}
        try:
            out = call_llm(payload)["choices"][0]["message"].get("content") or ""
        except Exception as e:
            out = f"(err {str(e)[:30]})"
        g = grade(parse_obj(out))
        if sample is None:
            sample = out.strip()[:160]
        if not g["parsed"]:
            print(f"  {i+1}/{k}: unparsed")
            continue
        parsed += 1
        root += g["root_target"]
        remote += g["remote_url"]
        both += g["both"]
        tag = "ROOT-OK" if g["root_target"] else "BURIED"
        print(f"  {i+1}/{k}: {tag}  targetDir={g['targetDir']!r}  url_remote={g['remote_url']}")
    print(f"== ARM {name}: root-target {root}/{k}, remote-url {remote}/{k}, both-correct {both}/{k} "
          f"(parsed {parsed}/{k})")
    print(f"   sample: {sample}")
    return both


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    m = model()
    print(f"model: {m}")
    base = scaffold_section(read_skill())
    run_arm("A-baseline", base, m, args.k)
    run_arm("B-skill-anchor", base + ANCHOR, m, args.k)
    run_arm("C-harness-state", base, m, args.k, user=DELIVERABLE + SCENARIO)


if __name__ == "__main__":
    main()
