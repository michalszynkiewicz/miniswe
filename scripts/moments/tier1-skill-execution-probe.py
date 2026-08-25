#!/usr/bin/env python3
"""tier1-skill-execution-probe — three probes for the anchored-plan design.

The skill router gets the model to READ a skill (0/8 -> 8/8), but the e2e
showed it then reads once and drifts to priors: skipped ScaffoldZarf,
improvised, never chained to uds-package-integrate. Proposed fix:
LLM-extract the skill's steps into an anchored plan, re-inject the step
text at execution time, and mechanically detect skill handoffs.

Three probes:

  EXTRACT  — can an LLM turn the (branchy, sub-file-referencing) build
             skill into a clean ordered step list with anchors? Grade:
             does it recover the 17 canonical steps incl. ScaffoldZarf,
             in order, without hallucinating?

  A (core) — at the ScaffoldZarf moment, with the step's exact text
             re-injected, does the model call scaffold-package (CORRECT)
             or improvise mkdir/zarf-dev-generate (DRIFT)? This is the
             load-bearing premise — if it drifts even with the text in
             front of it, the whole design is moot.

  B (chain)— at the integrate handoff, if miniswe has ALREADY injected
             integrate's steps into the plan (miniswe owns the fetch,
             the model just executes), does the model execute an integrate
             step vs improvise? Sidesteps the refuted name-lookup.

Usage: tier1-skill-execution-probe.py [--k 10]
"""

import argparse
import json
import os
import urllib.request

REPO = "/home/michal/dev/miniswe"
ENDPOINT = os.environ.get("LLAMA_ENDPOINT", "http://localhost:8464")
SKILLS_DIR = "/tmp/tmp.tII2UGxURO/.ai/skills"
REPRO_DUMPS = "/tmp/claude-1000/-home-michal-dev-miniswe/91153bbc-3489-42aa-88d1-4ad66657da3b/scratchpad/skills-repro/dumps"

CANONICAL_STEPS = [
    "ChartUrl", "ValidateChart", "ScaffoldZarf", "GenerateZarf", "ReviewFiles",
    "AnalyzeValues", "DetermineDependencies", "ReadValues", "LoadReferences",
    "PinImages", "VerifyHelmChart", "CreateZarfPackage", "CreateUDSBundleConfig",
    "ConfigureTaskRunner", "CreateUDSBundle", "SetupTestScaffold", "CIFiles",
]


def read_skill(name):
    return open(os.path.join(SKILLS_DIR, name, "SKILL.md")).read()


def dump0():
    files = sorted(os.listdir(REPRO_DUMPS))
    return json.load(open(os.path.join(REPRO_DUMPS, files[0])))


def call_llm(payload, timeout=300):
    req = urllib.request.Request(
        f"{ENDPOINT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


# ── EXTRACT ────────────────────────────────────────────────────────────────

def probe_extract(model, k):
    print("=== EXTRACT: LLM step extraction from the build skill ===")
    body = read_skill("uds-package-build")
    sys = (
        "You convert a skill document into an ordered execution checklist. "
        "Output ONLY a JSON array of objects, each {\"step\": \"<ShortName>\", "
        "\"anchor\": \"<the heading or a short verbatim quote locating this step "
        "in the document>\"}. One entry per actionable step, in execution order. "
        "Do not invent steps; do not include preamble or principles sections."
    )
    msgs = [{"role": "system", "content": sys},
            {"role": "user", "content": f"Skill document:\n\n{body}"}]
    outcomes = []
    for i in range(k):
        payload = {"model": model, "messages": msgs, "temperature": 0.2,
                   "max_tokens": 3000, "stream": False}
        try:
            txt = call_llm(payload)["choices"][0]["message"].get("content") or ""
            start, end = txt.find("["), txt.rfind("]")
            steps = json.loads(txt[start:end + 1]) if start >= 0 else []
            names = [str(s.get("step", "")) for s in steps]
            has_scaffold = any("scaffold" in n.lower() and "zarf" in n.lower()
                               or n == "ScaffoldZarf" for n in names)
            # in-order recall of canonical steps (subsequence match, case-insens)
            low = [n.lower() for n in names]
            ci, hit = 0, 0
            for canon in CANONICAL_STEPS:
                cl = canon.lower()
                for j in range(ci, len(low)):
                    if cl in low[j] or low[j] in cl:
                        hit += 1
                        ci = j + 1
                        break
            anchors_ok = all(s.get("anchor") for s in steps) if steps else False
            cat = (f"n={len(names)} recall={hit}/17 scaffold={'Y' if has_scaffold else 'N'} "
                   f"anchored={'Y' if anchors_ok else 'N'}")
        except Exception as e:
            cat = f"ERROR:{str(e)[:50]}"
        outcomes.append(cat)
        print(f"[extract] {i + 1}/{k}: {cat}", flush=True)
    return outcomes


# ── shared: real system prompt + tools ──────────────────────────────────────

def base_ctx():
    d = dump0()
    return d["messages"][0], d["tools"], d["model"], d.get("temperature", 0.2)


UDS_TASK = (
    "Create a UDS package for app-with-dependencies from the Helm chart in "
    "tests/e2e/app-with-deps/src/chart/. Deploy it and verify it is healthy."
)

SCAFFOLD_STEP = (
    "### Step ScaffoldZarf: Scaffold Zarf Package\n"
    "Only for new packages. This IS a new package.\n"
    "Call the `scaffold-package` tool with `targetDir`, `applicationName`, and "
    "`chartRepoURL`. Tolerates a non-empty target directory; supports `force: true`. "
    "IFF unavailable, manually copy the reference-package layout."
)

INTEGRATE_STEP = (
    "### Step ConfigureSSO: Register the Keycloak client (Integration Phase)\n"
    "Add an `sso:` block to the UDS Package CR that registers a Keycloak client: "
    "set clientId, redirectUris for app-with-deps.uds.dev, and the protocol mappers. "
    "This is what makes the app reachable and authenticated."
)


def probe_a(model, temp, system, tools, k):
    print("\n=== A (core): re-injected ScaffoldZarf step at execution time ===")
    msgs = [
        system,
        {"role": "user", "content": UDS_TASK},
        {"role": "assistant", "content": "Starting the build. First step from the skill:"},
        {"role": "user", "content":
            f"[Now execute this plan step. Re-read it and do exactly what it says:]\n{SCAFFOLD_STEP}"},
    ]
    outcomes = []
    for i in range(k):
        payload = {"model": model, "messages": msgs, "tools": tools,
                   "temperature": temp, "max_tokens": 2000, "stream": False}
        try:
            m = call_llm(payload)["choices"][0]["message"]
            tcs = m.get("tool_calls") or []
            if not tcs:
                cat = "NO-TOOL-CALL"
            else:
                fn = tcs[0]["function"]
                args = fn.get("arguments", "")
                blob = (fn.get("name", "") + " " + args).lower()
                if "scaffold-package" in blob or "scaffold_package" in blob:
                    cat = "CORRECT(scaffold-package)"
                elif "mkdir" in blob or "zarf dev generate" in blob or "git clone" in blob:
                    cat = f"DRIFT({fn.get('name')}:{args[:40]})"
                else:
                    cat = f"OTHER({fn.get('name')}:{args[:40]})"
        except Exception as e:
            cat = f"ERROR:{str(e)[:40]}"
        outcomes.append(cat)
        print(f"[A] {i + 1}/{k}: {cat}", flush=True)
    return outcomes


def probe_b(model, temp, system, tools, k):
    print("\n=== B (chain): integrate step pre-injected into the plan ===")
    msgs = [
        system,
        {"role": "user", "content": UDS_TASK},
        {"role": "assistant", "content":
            "Build phase done: package created and deployed. The plan now continues "
            "with the Integration Phase steps."},
        {"role": "user", "content":
            "[The validation reports the app is deployed but has no SSO client and is "
            "unreachable. Execute the next plan step. Re-read it and do exactly what it "
            f"says:]\n{INTEGRATE_STEP}"},
    ]
    outcomes = []
    for i in range(k):
        payload = {"model": model, "messages": msgs, "tools": tools,
                   "temperature": temp, "max_tokens": 2000, "stream": False}
        try:
            m = call_llm(payload)["choices"][0]["message"]
            tcs = m.get("tool_calls") or []
            if not tcs:
                cat = "NO-TOOL-CALL"
            else:
                fn = tcs[0]["function"]
                blob = (fn.get("name", "") + " " + fn.get("arguments", "")).lower()
                # executing = editing the Package CR / uds.yaml with sso, or reading it
                if "sso" in blob or "package" in blob and ("uds.yaml" in blob or "cr" in blob):
                    cat = f"EXECUTES({fn.get('name')})"
                elif fn.get("name") in ("file", "code", "shell") and (
                        "uds.yaml" in blob or "zarf.yaml" in blob or "keycloak" in blob):
                    cat = f"ON-TARGET({fn.get('name')}:{fn.get('arguments','')[:40]})"
                else:
                    cat = f"OTHER({fn.get('name')}:{fn.get('arguments','')[:40]})"
        except Exception as e:
            cat = f"ERROR:{str(e)[:40]}"
        outcomes.append(cat)
        print(f"[B] {i + 1}/{k}: {cat}", flush=True)
    return outcomes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    from collections import Counter

    system, tools, model, temp = base_ctx()
    ex = probe_extract(model, args.k)
    a = probe_a(model, temp, system, tools, args.k)
    b = probe_b(model, temp, system, tools, args.k)

    print("\n=== SUMMARY ===")
    print("EXTRACT:")
    for line in ex:
        print("  ", line)
    print("A:", dict(Counter(a)))
    print("B:", dict(Counter(b)))


if __name__ == "__main__":
    main()
