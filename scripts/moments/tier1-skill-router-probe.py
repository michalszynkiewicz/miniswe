#!/usr/bin/env python3
"""tier1-skill-router-probe — validate the pre-turn skill router design.

Motivation: two uds-mcp e2e sessions (6 runs) delivered a correct [SKILLS]
listing whose top description matches the task nearly verbatim — and the
model read zero SKILL.md files. Advisory prose is inert; the proposed fix
is a dedicated no-tools classifier call + task rewrite.

Stage 1 (classifier): messages = [classifier system prompt + skills list,
task]. NO tools. Expected output: exactly a skill name, or NONE. Measures
pick accuracy on matching tasks, NONE reliability on non-matching ones,
and how often salvage (strip backticks/quotes) or retry would be needed.

Stage 2 (first actions): real assembled system prompt from the uds
workspace (captured via a live repro dump) + either the REWRITTEN task
("Read .ai/skills/<skill>/SKILL.md and follow its instructions to handle
this request: ...") or the plain task (baseline = today's behavior).
Classifies the first tool call: does the model actually read the skill?

Usage: tier1-skill-router-probe.py [--k1 10] [--k2 8]
"""

import argparse
import json
import os
import re
import urllib.request

REPO = "/home/michal/dev/miniswe"
ENDPOINT = os.environ.get("LLAMA_ENDPOINT", "http://localhost:8464")
SKILLS_DIR = "/tmp/tmp.tII2UGxURO/.ai/skills"
REPRO_DUMPS = "/tmp/claude-1000/-home-michal-dev-miniswe/91153bbc-3489-42aa-88d1-4ad66657da3b/scratchpad/skills-repro/dumps"

UDS_TASK = (
    "Create a UDS package for app-with-dependencies using the Helm chart at "
    "https://github.com/defenseunicorns/uds-mcp (in the `tests/e2e/app-with-deps/src/chart/` "
    "directory, branch/ref: `main`). The container image is at "
    "ghcr.io/michalszynkiewicz/app-with-dependencies:0.1.0. Expose the app at "
    "app-with-deps.uds.dev. Deploy it to a k3d test cluster and verify that the application "
    "is running and healthy."
)
NONMATCH_TASK = (
    "Add a CLI flag --system-prompt-override (short: -s) that takes a string and replaces "
    "the default system prompt with the provided text. Make sure it works for both "
    "single-shot and interactive modes."
)
TROUBLESHOOT_TASK = (
    "The app-with-deps deployment in my UDS cluster has pods stuck in CrashLoopBackOff — "
    "figure out why and fix it."
)

STAGE1_CASES = [
    ("uds-task", UDS_TASK, {"uds-package", "uds-package-build"}),
    ("nonmatch", NONMATCH_TASK, {"NONE"}),
    ("troubleshoot", TROUBLESHOOT_TASK, {"uds-troubleshoot", "uds-support"}),
]


def load_skills():
    out = []
    for name in sorted(os.listdir(SKILLS_DIR)):
        p = os.path.join(SKILLS_DIR, name, "SKILL.md")
        if not os.path.exists(p):
            continue
        text = open(p).read()
        m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
        desc = ""
        if m:
            dm = re.search(r"^description:\s*(.+?)(?=\n\w+:|\Z)", m.group(1), re.DOTALL | re.MULTILINE)
            if dm:
                desc = " ".join(dm.group(1).replace(">", " ").split())
        out.append((name, desc))
    return out


def classifier_messages(skills, task):
    listing = "\n".join(f"- {n}: {d}" for n, d in skills)
    system = (
        "You route coding tasks to skills. Below is the list of installed skills.\n"
        f"{listing}\n"
        "If exactly one skill clearly applies to the user's task, reply with that "
        "skill's name and nothing else. If none clearly applies, reply NONE. "
        "Reply with a single word only."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": task}]


def call_llm(payload, timeout=180):
    req = urllib.request.Request(
        f"{ENDPOINT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def salvage(raw, names):
    t = raw.strip().strip("`'\".* ").rstrip(".").strip()
    if t.upper() == "NONE":
        return "NONE", t == raw.strip()
    for n in names:
        if t.lower() == n.lower():
            return n, t == raw.strip()
    # last resort: unique containment
    hits = [n for n in names if n.lower() in t.lower()]
    if len(hits) == 1:
        return hits[0], False
    return None, False


def stage1(skills, model, k):
    names = [n for n, _ in skills]
    print("=== stage 1: classifier (no tools) ===")
    results = {}
    for label, task, expected in STAGE1_CASES:
        msgs = classifier_messages(skills, task)
        outcomes = []
        for i in range(k):
            payload = {"model": model, "messages": msgs, "temperature": 0.2,
                       "max_tokens": 2500, "stream": False}
            try:
                resp = call_llm(payload)
                raw = (resp["choices"][0]["message"].get("content") or "").strip()
                pick, clean = salvage(raw, names)
                if pick is None:
                    cat = f"UNPARSEABLE({raw[:30]})"
                elif pick in expected:
                    cat = "HIT" if clean else "HIT-SALVAGED"
                else:
                    cat = f"WRONG({pick})"
            except Exception as e:
                cat = f"ERROR:{str(e)[:40]}"
            outcomes.append(cat)
            print(f"[{label}] {i + 1}/{k}: {cat}", flush=True)
        results[label] = outcomes
    return results


def stage2(model, k):
    print("\n=== stage 2: first actions after rewrite (real system prompt) ===")
    files = sorted(os.listdir(REPRO_DUMPS))
    dump = json.load(open(os.path.join(REPRO_DUMPS, files[0])))
    system = dump["messages"][0]
    tools = dump["tools"]

    rewritten = (
        "Read .ai/skills/uds-package-build/SKILL.md and follow its instructions "
        f"to handle this request: {UDS_TASK}"
    )
    arms = {"baseline": UDS_TASK, "rewritten": rewritten}
    results = {}
    for arm, task in arms.items():
        msgs = [system, {"role": "user", "content": task}]
        outcomes = []
        for i in range(k):
            payload = {"model": model, "messages": msgs, "tools": tools,
                       "temperature": dump.get("temperature", 0.2),
                       "max_tokens": 2000, "stream": False}
            try:
                resp = call_llm(payload)
                m = resp["choices"][0]["message"]
                tcs = m.get("tool_calls") or []
                if not tcs:
                    cat = "NO-TOOL-CALL"
                else:
                    fn = tcs[0]["function"]
                    args = json.loads(fn.get("arguments", "{}"))
                    path = str(args.get("path", ""))
                    if ".ai/skills/uds-package-build/SKILL.md" in path:
                        cat = "READS-SKILL"
                    elif ".ai/skills" in path:
                        cat = f"READS-OTHER-SKILL({path[-40:]})"
                    else:
                        cat = f"OTHER:{fn['name']}({json.dumps(args)[:50]})"
            except Exception as e:
                cat = f"ERROR:{str(e)[:40]}"
            outcomes.append(cat)
            print(f"[{arm}] {i + 1}/{k}: {cat}", flush=True)
        results[arm] = outcomes
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k1", type=int, default=10)
    ap.add_argument("--k2", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(REPO, "benchmark_results/_moments/skill-router-v1"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    skills = load_skills()
    assert skills, "no skills found"
    files = sorted(os.listdir(REPRO_DUMPS))
    model = json.load(open(os.path.join(REPRO_DUMPS, files[0])))["model"]

    r1 = stage1(skills, model, args.k1)
    r2 = stage2(model, args.k2)
    json.dump({"stage1": r1, "stage2": r2}, open(os.path.join(args.out, "results.json"), "w"), indent=1)

    from collections import Counter
    print("\n=== SUMMARY ===")
    for label, o in r1.items():
        print(f"stage1/{label}: {dict(Counter(o))}")
    for arm, o in r2.items():
        print(f"stage2/{arm}: {dict(Counter(o))}")


if __name__ == "__main__":
    main()
