#!/usr/bin/env python3
"""tier1-step-check-generator-probe — can the model turn a step's DONE WHEN
into a CORRECT, READ-ONLY completion check?

This is the load-bearing assumption of the "per-step validation" design:
each distilled step already carries a `DONE WHEN:` criterion; if we can
generate a cheap read-only shell command that exits 0 iff the step is done,
that command becomes the *effective validation command* while the step is
active — which lights up the whole (currently dormant, because
`[validation] command = ""`) gate + reactive_debugger stack per-step, and
would surface errors like "SSO put in zarf.yaml instead of the Package CR"
as a hard, debugger-triggering failure.

The danger the probe measures directly, by running each generated check
against TWO real fixtures:
  - false-PASS: check exits 0 on a NOT-done state -> useless (silent pass).
  - false-FAIL: check exits non-zero on a genuinely-done state -> manufactures
    grinding (the failure mode we're trying to prevent).

Targets (canonical DONE WHENs stand in for distiller output; material is the
real skill so the model can ground exact file paths):
  WriteZarfYaml    (build)     checkable — zarf.yaml artifact
  ImplementConfigChart (integrate) checkable — Package CR template artifact
  ConfigureSSO     (integrate)  checkable — THE discriminator: not-done
                   fixture puts `spec.sso` in zarf.yaml (the real mistake)
                   with a Package CR template that has NO sso, so a lazy
                   `grep -r sso .` false-passes and a correct targeted grep
                   fails.
  ChartUrl         (build)      abstention — investigative, no on-disk
                   artifact; the RIGHT answer is NONE.

Usage: [--k 8]
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request

ENDPOINT = os.environ.get("LLAMA_ENDPOINT", "http://localhost:8464")
SKILLS = "/home/michal/work/uds-mcp/src/todo-skills"

SYS = (
    "You write a completion CHECK for ONE step of a skill. Given the step's DONE WHEN "
    "criterion, output a SINGLE shell command that exits 0 if and only if the step is "
    "complete, and non-zero otherwise.\n"
    "HARD RULES:\n"
    "- READ-ONLY. Use only test / [ ] / grep / ls / find / cat / yq eval. NEVER modify, "
    "create, delete, build, or deploy anything. No redirects to files, no sed -i, no "
    "kubectl/helm/zarf/git/docker, no package create or deploy.\n"
    "- Reference the EXACT file paths from the skill material.\n"
    "- Cheap and objective: filesystem/text checks only, no cluster access.\n"
    "- If the step's completion CANNOT be verified by such a command (it is investigative, "
    "or produces no on-disk artifact), output exactly: NONE\n"
    "Output ONLY the one-line command, or NONE. No explanation, no code fence."
)

# Fixtures: relative-path -> file content. Fresh temp dir per run.
ZARF_DONE = {
    "zarf.yaml": "kind: ZarfPackageConfig\nmetadata:\n  name: nginx-app\ncomponents:\n  - name: nginx\n"
}
ZARF_MISSING = {"README.md": "app source, no package yet\n"}

CR_DONE = {
    "chart/templates/uds-package.yaml": (
        "apiVersion: uds.dev/v1alpha1\nkind: Package\nmetadata:\n  name: nginx-app\n"
        "spec:\n  network:\n    expose:\n      - service: nginx\n"
    )
}
CR_MISSING = {"chart/values.yaml": "replicaCount: 1\n"}

# The real mistake: sso lives in zarf.yaml; the Package CR template exists but
# has NO sso. A correct check greps the CR template (fails); a lazy
# `grep -r sso .` finds it in zarf.yaml and false-passes.
SSO_DONE = {
    "zarf.yaml": "kind: ZarfPackageConfig\nmetadata:\n  name: nginx-app\n",
    "chart/templates/uds-package.yaml": (
        "apiVersion: uds.dev/v1alpha1\nkind: Package\nmetadata:\n  name: nginx-app\n"
        "spec:\n  sso:\n    - clientId: uds-nginx\n      name: Nginx\n"
    ),
}
SSO_NOTDONE = {
    "zarf.yaml": "kind: ZarfPackageConfig\nmetadata:\n  name: nginx-app\nspec:\n  sso:\n    - clientId: uds-nginx\n",
    "chart/templates/uds-package.yaml": (
        "apiVersion: uds.dev/v1alpha1\nkind: Package\nmetadata:\n  name: nginx-app\n"
        "spec:\n  network:\n    expose:\n      - service: nginx\n"
    ),
}

TARGETS = [
    {
        "name": "WriteZarfYaml",
        "skill": "uds-package-build",
        "refs": ["scaffold.md"],
        "done_when": "zarf.yaml exists at the package root and is a ZarfPackageConfig "
        "(has `kind: ZarfPackageConfig` and a metadata.name).",
        "kind": "check",
        "done": ZARF_DONE,
        "notdone": ZARF_MISSING,
    },
    {
        "name": "ImplementConfigChart",
        "skill": "uds-package-integrate",
        "refs": [],
        "done_when": "chart/templates/uds-package.yaml exists and contains a "
        "packages.uds.dev Package CR (`kind: Package`).",
        "kind": "check",
        "done": CR_DONE,
        "notdone": CR_MISSING,
    },
    {
        "name": "ConfigureSSO",
        "skill": "uds-package-integrate",
        "refs": ["sso-integration.md"],
        "done_when": "The UDS Package CR at chart/templates/uds-package.yaml registers "
        "the SSO client (there is an `sso:` entry under its spec). SSO config in "
        "zarf.yaml does NOT count — it belongs in the Package CR template.",
        "kind": "check",
        "done": SSO_DONE,
        "notdone": SSO_NOTDONE,
    },
    {
        "name": "ChartUrl",
        "skill": "uds-package-build",
        "refs": [],
        "done_when": "The application's Helm chart URL/OCI reference and version have "
        "been identified (e.g. via search-docs or the upstream repo).",
        "kind": "abstain",
    },
]

# A check is read-only unless a *command-position* token is a mutator, or it
# writes files (redirect / in-place edit). Command position = start of the
# string or right after a shell operator; this is what stops `zarf.yaml` /
# `uds-package.yaml` (mutators appearing only as PATH substrings) from being
# mis-flagged — the whole point being that a valid check names those files.
MUTATORS = {
    "rm", "mv", "cp", "dd", "truncate", "tee", "chmod", "chown", "mkdir",
    "touch", "curl", "wget", "kubectl", "helm", "zarf", "uds", "docker",
    "git", "ln", "install", "npm", "node", "python", "python3", "make", "apt",
}
_INPLACE = re.compile(r"\b(sed|yq|jq)\b[^|;&]*\s-i\b")
_BAD_REDIR = re.compile(r">>|>\s*(?!/dev/null|&\s*\d)\S")


def is_read_only(cmd):
    if _INPLACE.search(cmd) or _BAD_REDIR.search(cmd):
        return False
    # first token of each &&/||/|/;-separated segment
    for seg in re.split(r"&&|\|\||[|;\n]", cmd):
        seg = seg.strip().lstrip("!(").strip()
        if not seg:
            continue
        tok = seg.split(None, 1)[0].strip("(")
        if tok in MUTATORS:
            return False
    return True


def read(skill, name):
    p = os.path.join(SKILLS, skill, name)
    return open(p).read() if os.path.exists(p) else ""


def material(t):
    blob = f"=== SKILL.md ===\n{read(t['skill'], 'SKILL.md')}\n"
    for r in t["refs"]:
        c = read(t["skill"], r)
        if c:
            blob += f"\n=== {r} ===\n{c}\n"
    return blob


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


def parse_check(raw):
    """-> None (abstain), '' (junk), or the command string."""
    t = (raw or "").strip()
    m = re.search(r"```(?:bash|sh)?\s*\n(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    line = next((ln.strip() for ln in t.splitlines() if ln.strip()), "")
    if re.fullmatch(r"['\"`*]*none['\"`.*]*", line, re.I):
        return None
    return line


def run_check(cmd, files):
    d = tempfile.mkdtemp(prefix="chkprobe-")
    try:
        for rel, content in files.items():
            p = os.path.join(d, rel)
            os.makedirs(os.path.dirname(p) or d, exist_ok=True)
            with open(p, "w") as f:
                f.write(content)
        try:
            r = subprocess.run(
                ["bash", "-c", cmd], cwd=d, timeout=10, capture_output=True
            )
            return r.returncode
        except Exception:
            return 999
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()
    m = model()
    print(f"model: {m}\n")

    for t in TARGETS:
        blob = material(t)
        user = (
            f"Skill material:\n\n{blob}\n\nStep: '{t['name']}'\n"
            f"DONE WHEN: {t['done_when']}\n\nWrite the CHECK command."
        )
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]

        correct = false_pass = false_fail = not_ro = abstained = junk = 0
        sample = None
        for i in range(args.k):
            payload = {"model": m, "messages": msgs, "temperature": 0.2,
                       "max_tokens": 4000, "stream": False}
            try:
                out = call_llm(payload)["choices"][0]["message"].get("content") or ""
            except Exception as e:
                out = f"(error {str(e)[:40]})"
            cmd = parse_check(out)
            if sample is None:
                sample = cmd if cmd is not None else "NONE"

            if t["kind"] == "abstain":
                ok = cmd is None
                correct += ok
                print(f"[{t['name']}] {i+1}/{args.k}: "
                      f"{'PASS(abstain)' if ok else 'fail(wrote check): ' + (cmd or '')[:60]}",
                      flush=True)
                continue

            if cmd is None:
                abstained += 1
                print(f"[{t['name']}] {i+1}/{args.k}: fail(abstained on checkable step)", flush=True)
                continue
            if not cmd:
                junk += 1
                print(f"[{t['name']}] {i+1}/{args.k}: fail(junk)", flush=True)
                continue
            if not is_read_only(cmd):
                not_ro += 1
                print(f"[{t['name']}] {i+1}/{args.k}: fail(NOT read-only): {cmd[:70]}", flush=True)
                continue
            dc = run_check(cmd, t["done"])
            nc = run_check(cmd, t["notdone"])
            fp = dc == 0 and nc == 0
            ff = dc != 0
            ok = dc == 0 and nc != 0
            correct += ok
            false_pass += fp
            false_fail += ff
            tag = ("PASS" if ok else
                   "fail(FALSE-PASS)" if fp else
                   "fail(FALSE-FAIL)" if ff else "fail")
            print(f"[{t['name']}] {i+1}/{args.k}: {tag}  done_exit={dc} notdone_exit={nc}  {cmd[:60]}",
                  flush=True)

        if t["kind"] == "abstain":
            print(f"== {t['name']} (abstain expected): {correct}/{args.k} correctly abstained")
        else:
            print(f"== {t['name']}: {correct}/{args.k} correct  "
                  f"[false-pass {false_pass}, false-FAIL {false_fail}, "
                  f"not-read-only {not_ro}, wrongly-abstained {abstained}, junk {junk}]")
        print(f"   sample: {sample}\n")


if __name__ == "__main__":
    main()
