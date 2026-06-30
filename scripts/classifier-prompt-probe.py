#!/usr/bin/env python3
"""Find the shortest classifier prompt that still separates questions (EXPLORE)
from edit requests (CODING) on the live gemma server."""
import json, urllib.request

URL = "http://localhost:8464/v1/chat/completions"

CANDIDATES = {
    "current(long)": (
        "Classify the user's request. Reply with EXACTLY one word, nothing else: CODING or EXPLORE.\n"
        "EXPLORE = the user wants information or understanding about the existing code/project, with NO "
        "instruction to change anything. This covers explaining, summarizing, describing, reviewing, "
        "assessing, or locating code, and any question: \"what does X do\", \"how does X work\", \"why "
        "does X happen\", \"where is X\", \"walk me through X\", \"tell me about X\", \"summarize X\", "
        "\"explain X\", \"is X correct?\", \"does X have bugs?\". Questions are EXPLORE.\n"
        "CODING = the user instructs you to CHANGE code or files: any imperative to add, create, "
        "implement, write, fix, change, update, remove, delete, rename, refactor, or otherwise modify, "
        "or an explicit request to make/apply a change. \"add X\", \"fix the bug in X\", \"refactor X\", "
        "\"make it do Y\", \"implement X\" are CODING.\n"
        "Decide by intent: a request to UNDERSTAND or EVALUATE is EXPLORE; a request to MODIFY is "
        "CODING. If there is no clear instruction to change code, answer EXPLORE."
    ),
    "medium": (
        "Reply with ONE word: CODING or EXPLORE.\n"
        "CODING = the user instructs you to change code (add, fix, implement, refactor, rename, remove, modify).\n"
        "EXPLORE = the user asks a question or wants to understand, review, evaluate, or locate code — no change requested.\n"
        "If there is no clear instruction to change code, answer EXPLORE."
    ),
    "short": (
        "Reply with one word: CODING or EXPLORE.\n"
        "CODING = an instruction to change code (add/fix/implement/refactor/remove).\n"
        "EXPLORE = a question about code, or a request to understand/review/locate it.\n"
        "No clear change instruction -> EXPLORE."
    ),
    "tiny": (
        "Reply one word: CODING or EXPLORE. "
        "CODING = the user tells you to change/add/fix/refactor code. "
        "EXPLORE = the user asks about or wants to understand/review code. "
        "Default EXPLORE."
    ),
    "tiniest": (
        "One word: CODING (user asks to change/add/fix code) or "
        "EXPLORE (user asks a question or to review/understand code). Default EXPLORE."
    ),
}

# (message, expected)
BATTERY = [
    ("explain the recent changes", "EXPLORE"),
    ("summarize what this module does", "EXPLORE"),
    ("why is the build slow?", "EXPLORE"),
    ("is this approach reasonable?", "EXPLORE"),
    ("does the plan panel have bugs?", "EXPLORE"),
    ("what does the main function do?", "EXPLORE"),
    ("review the repl code", "EXPLORE"),
    ("where is the config loaded?", "EXPLORE"),
    ("add a --version flag", "CODING"),
    ("fix the bug in the parser", "CODING"),
    ("refactor the plan module", "CODING"),
    ("make the panel collapsible", "CODING"),
    ("remove the dead code", "CODING"),
    ("rename foo to bar", "CODING"),
    ("implement caching for the repo map", "CODING"),
    ("wire the override through assemble", "CODING"),
]

def classify(sys_p, msg):
    body = json.dumps({
        "model": "gemma",
        "messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": msg}],
        "max_tokens": 8, "temperature": 0.15,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"].strip().upper()

for name, sys_p in CANDIDATES.items():
    wrong = []
    safety_fail = 0  # CODING misrouted to EXPLORE (the dangerous direction)
    for msg, exp in BATTERY:
        got = "EXPLORE" if classify(sys_p, msg).startswith("EXPLORE") else "CODING"
        if got != exp:
            wrong.append(f"{msg!r} want {exp} got {got}")
            if exp == "CODING":
                safety_fail += 1
    n = len(BATTERY)
    print(f"\n### {name}  (len={len(sys_p)} chars)  {n-len(wrong)}/{n} correct, safety-fails={safety_fail}")
    for w in wrong:
        print(f"    MISS: {w}")
