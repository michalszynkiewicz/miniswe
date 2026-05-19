#!/usr/bin/env python3
import json, urllib.request
EP = "http://localhost:8464/v1/chat/completions"
# MUST mirror src/cli/commands/repl.rs classify_is_explore() sys prompt.
SYS = (
    "Classify the user's request. Reply with EXACTLY one word, nothing else: "
    "CODING or EXPLORE.\n"
    "Answer CODING for anything that could lead to changing code/files, "
    "INCLUDING: reviewing, auditing, or critiquing code; checking whether "
    "something is correct/buggy/safe; finding or diagnosing bugs; suggesting "
    "or making improvements/fixes/refactors; or any imperative to "
    "write/modify/add/remove. \"Is X correct?\", \"review X\", \"does X have "
    "bugs?\", \"can you improve X?\" are all CODING.\n"
    "Answer EXPLORE ONLY for pure information requests with zero evaluation "
    "and zero change intent — explaining or locating how the existing code "
    "works: \"how does X work\", \"where is X\", \"what does X do\", \"walk "
    "me through X\".\n"
    "If it is not unambiguously a pure how/what/where explanation, answer "
    "CODING."
)
CASES = [
    ("add a --verbose flag to the CLI", "CODING", "clear-code"),
    ("fix the panic in src/lsp/client.rs", "CODING", "clear-code"),
    ("refactor assemble() to take a config struct", "CODING", "clear-code"),
    ("implement caching for the repo map", "CODING", "clear-code"),
    ("how does the plan gate work?", "EXPLORE", "clear-q"),
    ("which file parses CLI args?", "EXPLORE", "clear-q"),
    ("explain the ceremony modes", "EXPLORE", "clear-q"),
    ("what does sanitize_messages do and why", "EXPLORE", "clear-q"),
    ("walk me through how compression works", "EXPLORE", "clear-q"),
    ("why is Devstral mangling the position field", "EXPLORE", "clear-q"),
    ("is the token refresh logic correct?", "CODING", "ambiguous->safe"),
    ("look at auth and tell me if it has bugs", "CODING", "ambiguous->safe"),
    ("review error handling in run.rs", "CODING", "ambiguous->safe"),
    ("can you improve the repo map?", "CODING", "ambiguous->safe"),
    ("is there a race condition in the shell worker?", "CODING", "ambiguous->safe"),
    ("actually, just explain it, don't change anything", "EXPLORE", "correction->q"),
    ("scrap that, only describe the design", "EXPLORE", "correction->q"),
    ("no — I want you to actually fix it", "CODING", "correction->code"),
    ("nvm, go ahead and edit the file", "CODING", "correction->code"),
]


def cls(p):
    body = {
        "model": "local",
        "messages": [{"role": "system", "content": SYS},
                     {"role": "user", "content": p}],
        "temperature": 0.0, "max_tokens": 8,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    r = urllib.request.Request(EP, data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    try:
        d = json.load(urllib.request.urlopen(r, timeout=60))
        out = d["choices"][0]["message"].get("content") or ""
    except Exception as e:
        out = f"<ERR {e}>"
    is_explore = out.strip().upper().startswith("EXPLORE")  # mirrors is_explore_reply
    return ("EXPLORE" if is_explore else "CODING"), out.strip()[:30]


mid = json.load(urllib.request.urlopen("http://localhost:8464/v1/models"))
mid = (mid.get("data") or [{}])[0].get("id", "?")
print("SERVING:", mid)
ok = danger = 0
print(f"{'cat':18} {'exp':7} {'got':7} raw")
for p, exp, cat in CASES:
    got, raw = cls(p)
    dang = (exp == "CODING" and got == "EXPLORE")
    if got == exp:
        ok += 1
    if dang:
        danger += 1
    mark = "OK" if got == exp else "MISS"
    print(f"{cat:18} {exp:7} {got:7} {mark:5}"
          f"{'  <<< DANGEROUS' if dang else ''}  {raw!r}  | {p[:45]}")
print(f"\nmodel={mid}  accuracy {ok}/{len(CASES)}  | "
      f"DANGEROUS coding->explore: {danger} (MUST be 0)")
