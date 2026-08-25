#!/usr/bin/env python3
"""Diagnostic re-run of arm E (clean question + tools) with a larger round
budget and FULL transcript dumps, to tell apart:
  (a) ran out of rounds mid-search (round-cap artifact), vs
  (b) found the gitPath convention (SKILL.md:88) and still applied it wrong.
Records, per episode: converged? found_convention? (did any shell output
contain 'gitPath'), searched_gitpath? (did it ever grep the word), verdict.
"""
import json, os, re, subprocess, urllib.request, sys

sys.path.insert(0, os.path.dirname(__file__))
import importlib.util
spec = importlib.util.spec_from_file_location("rp", os.path.join(os.path.dirname(__file__), "tier1-chart-retrieval-probe.py"))
rp = importlib.util.module_from_spec(spec); spec.loader.exec_module(rp)

K = int(os.environ.get("K", "6"))
ROUNDS = int(os.environ.get("ROUNDS", "14"))


def episode(m, task, max_rounds):
    msgs = [{"role": "system", "content": rp.SYS}, {"role": "user", "content": task}]
    transcript = []
    saw_gitpath_in_output = False
    searched_gitpath = False
    converged = False
    final = ""
    for _ in range(max_rounds):
        pay = {"model": m, "messages": msgs, "temperature": 0.2, "max_tokens": 4000, "stream": False}
        try:
            out = rp.call_llm(pay)["choices"][0]["message"].get("content") or ""
        except Exception as e:
            final = f"(err {e})"; break
        transcript.append(("ASSISTANT", out))
        am = rp.ANSWER_RE.search(out)
        sm = rp.SHELL_RE.findall(out)
        if am and not (sm and out.rfind("SHELL:") > out.rfind("ANSWER:")):
            final = out; converged = True; break
        if sm:
            cmd = sm[-1].strip().strip("`")
            if "gitpath" in cmd.lower():
                searched_gitpath = True
            res = rp.run_shell(cmd)
            if "gitpath" in res.lower():
                saw_gitpath_in_output = True
            transcript.append(("SHELL", cmd + "\n→ " + res[:400]))
            msgs.append({"role": "assistant", "content": out})
            msgs.append({"role": "user", "content": f"[shell output]\n{res}"})
            continue
        msgs.append({"role": "assistant", "content": out})
        msgs.append({"role": "user", "content": "Emit either `SHELL: <cmd>` or `ANSWER: <chart entry>`."})
        final = out
    else:
        final = transcript[-1][1] if transcript else ""
    return final, converged, saw_gitpath_in_output, searched_gitpath, transcript


def main():
    m = rp.model()
    print(f"model: {m}  rounds={ROUNDS} k={K}\n")
    dumpdir = "/tmp/claude-1000/-home-michal-dev-miniswe/91153bbc-3489-42aa-88d1-4ad66657da3b/scratchpad"
    for i in range(K):
        final, conv, saw, searched, tr = episode(m, rp.SCOPED_TASK, ROUNDS)
        v = rp.grade(final)
        a = rp.ANSWER_RE.search(final)
        ans = (a.group(1) if a else final).strip()
        print(f"ep {i+1}: {v}  converged={conv}  saw_gitpath_in_docs={saw}  searched_gitpath={searched}")
        print(f"   final: {ans[:200]}")
        with open(f"{dumpdir}/E-ep{i+1}.txt", "w") as fh:
            for role, txt in tr:
                fh.write(f"===== {role} =====\n{txt}\n\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
