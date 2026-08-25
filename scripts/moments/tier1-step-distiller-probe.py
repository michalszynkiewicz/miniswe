#!/usr/bin/env python3
"""tier1-step-distiller-probe — can a just-in-time sub-agent distill ONE
skill step into correct, verbatim-faithful, self-contained instructions
(with a completion criterion)?

This is the load-bearing assumption of the decoupled step-cursor design:
instead of file->anchor (which resolves to garbage spans on prose skills and
misses the ~13 referenced sub-files), a focused agent reads the skill + its
sub-files when a step becomes active and produces exactly that step's
instructions. Probe grades whether the distilled output keeps the
load-bearing specifics.

Targets (real uds-package-build skill):
  WriteZarfYaml  — must carry the `zarf-generate` tool + git url `.git@<ref>`
                   format + packageName/version/gitPath args (the exact
                   things the model kept getting wrong live), and a
                   completion criterion.
  SetupTestScaffold — references testing.md; tests cross-file resolution
                   (does the distiller pull the sub-file's specifics?).

The distiller is given the full SKILL.md + the referenced sub-files inline
(a probe stand-in for the real agent's read tools). Usage: [--k 8]
"""

import argparse
import json
import os
import re
import urllib.request

REPO = "/home/michal/dev/miniswe"
ENDPOINT = os.environ.get("LLAMA_ENDPOINT", "http://localhost:8464")
SKILL_DIR = "/home/michal/work/uds-mcp/src/todo-skills/uds-package-build"
REPRO_DUMPS = "/tmp/claude-1000/-home-michal-dev-miniswe/91153bbc-3489-42aa-88d1-4ad66657da3b/scratchpad/skills-repro/dumps"

DISTILLER_SYS = (
    "You are preparing focused, self-contained instructions for ONE step of a skill, to hand "
    "to an executor who will do only that step and nothing else. You are given the skill "
    "document and its referenced sub-files.\n"
    "Output exactly two sections:\n"
    "INSTRUCTIONS: the concrete actions for this step. COPY load-bearing specifics VERBATIM — "
    "tool names, exact argument names and formats, commands, URLs, file paths. Do NOT paraphrase "
    "them or leave them abstract. Inline anything from a referenced sub-file that this step needs. "
    "Omit other steps.\n"
    "DONE WHEN: a one-line, checkable completion criterion for this step.\n"
    "Output only those two sections."
)


def read(name):
    p = os.path.join(SKILL_DIR, name)
    return open(p).read() if os.path.exists(p) else ""


def material(refs):
    body = read("SKILL.md")
    blob = f"=== SKILL.md ===\n{body}\n"
    for r in refs:
        c = read(r)
        if c:
            blob += f"\n=== {r} ===\n{c}\n"
    return blob


TARGETS = {
    "WriteZarfYaml": {
        "refs": [],
        # must survive distillation, verbatim-ish:
        "must": ["zarf-generate", ".git", "@"],
        "want_any": ["packageName", "gitPath", "version", "url"],
    },
    "SetupTestScaffold": {
        "refs": ["testing.md", "testing-overview.md", "test-templates.md"],
        "must": ["test"],
        # something concrete pulled from the sub-files:
        "want_any": ["playwright", "kubectl", "http", "scaffold", "tasks.yaml", "validate"],
    },
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
    d = json.load(open(os.path.join(REPRO_DUMPS, sorted(os.listdir(REPRO_DUMPS))[0])))
    return d["model"]


def grade(text, spec):
    low = text.lower()
    must_ok = all(m.lower() in low for m in spec["must"])
    want_ok = any(w.lower() in low for w in spec["want_any"])
    has_done = "done when" in low
    # hallucination smell: mentions the CLI as the primary path (should prefer the tool)
    verbatim_git = bool(re.search(r"\.git\s*@|@\s*<?(tag|ref|v\d|main)", low))
    return {
        "must": must_ok,
        "want": want_ok,
        "done": has_done,
        "git_fmt": verbatim_git,
        "pass": must_ok and want_ok and has_done,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()
    m = model()

    for step, spec in TARGETS.items():
        blob = material(spec["refs"])
        msgs = [
            {"role": "system", "content": DISTILLER_SYS},
            {"role": "user", "content": f"Skill material:\n\n{blob}\n\nDistill the step: '{step}'."},
        ]
        rows = []
        sample_text = None
        for i in range(args.k):
            payload = {"model": m, "messages": msgs, "temperature": 0.2,
                       "max_tokens": 4000, "stream": False}
            try:
                out = call_llm(payload)["choices"][0]["message"].get("content") or ""
                g = grade(out, spec)
                if sample_text is None:
                    sample_text = out
            except Exception as e:
                g = {"error": str(e)[:50]}
            rows.append(g)
            tag = "PASS" if g.get("pass") else "fail"
            print(f"[{step}] {i + 1}/{args.k}: {tag} must={g.get('must')} want={g.get('want')} "
                  f"done={g.get('done')} git_fmt={g.get('git_fmt')}", flush=True)
        passes = sum(1 for r in rows if r.get("pass"))
        gitf = sum(1 for r in rows if r.get("git_fmt"))
        print(f"== {step}: {passes}/{args.k} pass; git-format-kept {gitf}/{args.k}")
        if sample_text:
            print(f"--- sample distilled '{step}' (head) ---\n{sample_text[:600]}\n")


if __name__ == "__main__":
    main()
