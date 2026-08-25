#!/usr/bin/env python3
"""tier1-chart-retrieval-probe — the untested link: can a fresh helper, given
read-only shell against the real skill docs, FIND the zarf git-chart convention
itself (rather than being handed the snippet)?

Context: tier1-chart-gitpath-probe showed a fresh helper HANDED the convention
hits 12/12 (arms B/C), but with only the error it's 0/12 and confidently wrong
(arm A — anchors on the misleading "failed to locate tag main"). The hybrid
design (deterministic harness trigger + helper fetches its own reference) rests
on the helper being able to retrieve the convention. This probes exactly that.

A minimal ReAct tool-loop: the model may issue read-only shell (grep/cat/ls/
find/head/rg/sed/awk/wc) against the real skill tree, gets stdout back, up to
MAX_ROUNDS, then must emit `ANSWER:` with the chart YAML. cwd = the todo-skills
dir (where uds-package-build/SKILL.md carries the convention at :88).

Arms (fresh context, k each):
  D  starts from the FAILURE (misleading "tag not found") + tools
     — the harness-triggered / deterministic path (no gemma4 choice)
  E  starts from a clean SCOPED QUESTION + tools
     — the investigate() path (model must have chosen to delegate)

CORRECT = final YAML keeps the subdir in a SEPARATE gitPath, url NOT concatenated.
Compare against the handed-snippet ceiling (12/12) and the no-tools floor (0/12).
Usage: [--k 10] [--rounds 8]
"""

import argparse
import json
import os
import re
import subprocess
import urllib.request

ENDPOINT = os.environ.get("LLAMA_ENDPOINT", "http://localhost:8464")
SKILL_ROOT = os.environ.get(
    "SKILL_ROOT", "/home/michal/work/uds-mcp/src/todo-skills"
)

WRONG_ZARF = """  - name: app-with-dependencies
    charts:
      - name: app-with-dependencies
        url: https://github.com/defenseunicorns/uds-mcp.git@main/tests/e2e/app-with-deps/src/chart
        version: 0.1.0
        namespace: app-with-deps"""

FAILURE_TASK = """The deploy keeps failing. `uds run dev` fails pulling the chart:

unable to pull the chart "app-with-dependencies" from git: failed to locate tag main in repository: tag not found
Error: failed to fetch https://github.com/defenseunicorns/uds-mcp.git@main/tests/e2e/app-with-deps/src/chart/ : 404 Not Found
[exit 1]

Current chart entry in common/zarf.yaml:
""" + WRONG_ZARF + """

Facts: the repo https://github.com/defenseunicorns/uds-mcp is public, `main` is a valid branch, the chart is at tests/e2e/app-with-deps/src/chart, Chart.yaml version is 0.1.0.
Find the root cause and give the corrected chart entry."""

SCOPED_TASK = """Give me the correct zarf.yaml `charts:` entry for a Helm chart that lives inside a git repository.

- Git repo: https://github.com/defenseunicorns/uds-mcp
- Chart subdirectory: tests/e2e/app-with-deps/src/chart
- Git ref (branch): main
- Chart name: app-with-dependencies
- Chart version (Chart.yaml): 0.1.0

Consult the skill docs to get the exact zarf syntax right, then output the `charts:` entry."""

SYS = (
    "You are a focused helper with fresh eyes and no shared context. The uds packaging skill "
    "docs are in your current directory (subdirs like uds-package-build/). You may consult them "
    "with READ-ONLY shell.\n\n"
    "Each turn, output EXACTLY ONE of:\n"
    "  SHELL: <a single read-only command>   (grep/cat/ls/find/head/tail/rg/sed/awk/wc — to search the docs)\n"
    "  ANSWER: <the final zarf.yaml charts: entry>\n"
    "Do your reasoning first, then the one directive on its own line. Prefer to check the docs "
    "before answering. Keep going until you can give a concrete ANSWER."
)

ALLOWED = {"grep", "rg", "cat", "ls", "find", "head", "tail", "sed", "awk", "wc", "cut", "sort", "uniq"}


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


def run_shell(cmd):
    # safety: read-only whitelist, no redirection/chaining
    if any(tok in cmd for tok in [">", "<", ">>", "|&", "`", "$("]):
        return "(rejected: no redirection/subshells; use a plain grep/cat/find)"
    head = cmd.strip().split()
    if not head or head[0] not in ALLOWED:
        return f"(rejected: '{head[0] if head else ''}' not allowed; use {sorted(ALLOWED)})"
    # split on pipes; every stage must be whitelisted
    for stage in cmd.split("|"):
        t = stage.strip().split()
        if t and t[0] not in ALLOWED:
            return f"(rejected: '{t[0]}' not allowed in pipeline)"
    try:
        out = subprocess.run(
            cmd, shell=True, cwd=SKILL_ROOT, capture_output=True, text=True, timeout=20
        )
        s = (out.stdout or "") + (("\n[stderr] " + out.stderr) if out.stderr.strip() else "")
        s = s.strip()
        if len(s) > 2000:
            s = s[:2000] + "\n…(truncated)"
        return s or "(no output)"
    except subprocess.TimeoutExpired:
        return "(command timed out)"
    except Exception as e:
        return f"(error: {str(e)[:80]})"


SUBDIR = "tests/e2e/app-with-deps/src/chart"
URL_CONCAT = re.compile(r"\.git(@[\w./+-]+)?/tests/e2e/app-with-deps/src/chart", re.I)
GITPATH_OK = re.compile(r"gitPath\s*:?\s*\.?/?" + re.escape(SUBDIR), re.I)
SHELL_RE = re.compile(r"^\s*SHELL:\s*(.+)$", re.I | re.M)
ANSWER_RE = re.compile(r"ANSWER:\s*(.+)", re.I | re.S)


def grade(text):
    low = text.lower()
    concat = URL_CONCAT.search(text) is not None
    gitpath = GITPATH_OK.search(text) is not None
    if gitpath and not concat:
        return "CORRECT"
    if "gitpath" in low and not concat:
        return "OTHER"
    if ("gitpath" in low or "separate" in low or "dev generate" in low) and concat:
        return "OTHER"
    return "MISS"


def run_episode(m, task, max_rounds):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": task}]
    used_shell = 0
    for _ in range(max_rounds):
        payload = {"model": m, "messages": msgs, "temperature": 0.2, "max_tokens": 4000, "stream": False}
        try:
            out = call_llm(payload)["choices"][0]["message"].get("content") or ""
        except Exception as e:
            return f"(err {str(e)[:40]})", used_shell
        # answer takes priority if present
        am = ANSWER_RE.search(out)
        sm = SHELL_RE.findall(out)
        if am and not (sm and out.rfind("SHELL:") > out.rfind("ANSWER:")):
            return out, used_shell
        if sm:
            cmd = sm[-1].strip().strip("`")
            res = run_shell(cmd)
            used_shell += 1
            msgs.append({"role": "assistant", "content": out})
            msgs.append({"role": "user", "content": f"[shell output]\n{res}"})
            continue
        # no directive — nudge once
        msgs.append({"role": "assistant", "content": out})
        msgs.append({"role": "user", "content": "Emit either `SHELL: <cmd>` to check the docs or `ANSWER: <chart entry>`."})
    # last resort: grade whatever the final message was
    return out, used_shell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=8)
    args = ap.parse_args()
    m = model()
    print(f"model: {m}\nskill_root: {SKILL_ROOT}\n")
    arms = {"D_failure+tools": FAILURE_TASK, "E_question+tools": SCOPED_TASK}
    for arm, task in arms.items():
        counts = {"CORRECT": 0, "OTHER": 0, "MISS": 0}
        shell_used = []
        sample = None
        for i in range(args.k):
            out, ns = run_episode(m, task, args.rounds)
            v = grade(out)
            counts[v] += 1
            shell_used.append(ns)
            if sample is None or (v == "CORRECT" and "CORRECT" not in (sample or "")):
                a = ANSWER_RE.search(out)
                sample = f"{v} ({ns} shell): {(a.group(1) if a else out).strip()[:220]}"
            print(f"  {arm} {i+1}/{args.k}: {v}  (shell used: {ns})", flush=True)
        avg = sum(shell_used) / len(shell_used) if shell_used else 0
        print(f"\n== {arm}: CORRECT {counts['CORRECT']}/{args.k}  OTHER {counts['OTHER']}  MISS {counts['MISS']}  (avg shell calls {avg:.1f})")
        print(f"   sample: {sample}\n")


if __name__ == "__main__":
    main()
