#!/usr/bin/env python3
"""
Arms C and D: miniswe's ACTUAL rich surface (grouped tools + plan gate),
holding handlers identical, varying ONLY the invocation modality.

  C — bare-verb text grammar  (READ / PLAN SET / REFACTOR / REPLACE_RANGE ...)
  D — OpenAI tool_calls over the same grouped schema (≈ real miniswe)

Same task / crate / cargo-check+behavioral oracle as arms A/B (imported
from harness.py) so all four arms are directly comparable.

C vs D  -> does text-invocation alone help, holding the tool surface fixed?
A vs C  -> does the rich/gated ceremony hurt even in text modality?
"""
import importlib.util, json, os, re, subprocess, sys, time

spec = importlib.util.spec_from_file_location("h", "/tmp/simple-tier-probe/harness.py")
H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)

WORK = "/tmp/simple-tier-probe"
TRIALS = int(os.environ.get("TRIALS", "6"))
MAX_ROUNDS = int(os.environ.get("MAX_ROUNDS", "8"))
NO_GATE = bool(os.environ.get("NO_GATE"))  # arms F/G: drop the plan gate

# ── shared rich-surface handlers ───────────────────────────────────────────
# Every C-directive and every D-tool_call normalizes to one (op, args) and
# runs through dispatch(). Only the PARSER differs between arms.

GATE_MSG = ("Create a plan first: use plan set with your step-by-step "
            "approach before making changes. (edit tools are locked until then)")

def _safe(d, p):
    """Sandbox a model-supplied path inside d. Strips absolute prefixes
    (qwen emits SWE-bench-style /testbed/...) and clamps escapes."""
    p = (p or "").strip().lstrip("/")
    full = os.path.normpath(os.path.join(d, p))
    rd = os.path.realpath(d)
    rf = os.path.realpath(full)
    if rf != rd and not rf.startswith(rd + os.sep):
        full = os.path.join(d, os.path.basename(p) or "x")
    return full

def _plan_path(d): return os.path.join(d, ".plan")

def plan_exists(d): return os.path.isfile(_plan_path(d))

def dispatch(op, a, d):
    """Returns (text_result, mutated_bool). Identical for C and D."""
    if op == "READ":
        return H.read_file_safe(d, a.get("path", "")), False
    if op == "SEARCH":
        out = subprocess.run(["grep", "-rn", a.get("query", ""), "src"], cwd=d,
                             capture_output=True, text=True).stdout
        return (out or "(no matches)"), False
    if op == "SHELL":
        pr = subprocess.run(a.get("command", "true"), cwd=d, shell=True,
                            capture_output=True, text=True, timeout=120)
        return (pr.stdout + pr.stderr)[-1500:], False
    if op == "PLAN_SET":
        open(_plan_path(d), "w").write(a.get("content", "plan"))
        return "Plan set. Edit tools unlocked: refactor, replace_range, insert_at, write_file.", False
    if op == "PLAN_CHECK":
        ok, out = H.cargo_check(d)
        return (f"step {a.get('step','?')}: cargo check {'PASS' if ok else 'FAIL'}\n"
                + ("" if ok else out)), False
    # ---- edit ops: plan-gated, exactly like real miniswe ----
    if op in ("REFACTOR", "REPLACE_RANGE", "INSERT_AT", "WRITE_FILE"):
        if not NO_GATE and not plan_exists(d):
            return GATE_MSG, False
    if op == "REFACTOR":
        # add_param, resolved by name (mirrors real refactor: def + every
        # callsite atomically). Deterministic for this task shape.
        if a.get("action", "add_param") != "add_param":
            return f"refactor: unsupported action {a.get('action')}", False
        name = a.get("name", "")
        np = a.get("new_param", "")
        fill = a.get("callsite_fill_in", a.get("fill", ""))
        lib = os.path.join(d, "src/lib.rs"); mn = os.path.join(d, "src/main.rs")
        changed = 0
        if os.path.isfile(lib):
            s = open(lib).read()
            s2 = re.sub(rf"(fn\s+{re.escape(name)}\s*\([^)]*)\)",
                        lambda m: m.group(1) + (", " if m.group(1).rstrip().endswith(")") is False and not m.group(1).rstrip().endswith("(") else "") + np + ")",
                        s, count=1)
            if s2 != s: open(lib, "w").write(s2); changed += 1
        if os.path.isfile(mn) and fill:
            s = open(mn).read()
            s2 = re.sub(rf"({re.escape(name)}\s*\([^)]*)\)",
                        lambda m: m.group(1) + ", " + fill + ")", s, count=1)
            if s2 != s: open(mn, "w").write(s2); changed += 1
        ok, out = H.cargo_check(d)
        return (f"refactor add_param {name}: updated {changed} site(s) "
                f"(definition + callsite). cargo check {'PASS' if ok else 'FAIL'}\n"
                + ("" if ok else out)), True
    def _int(v, what):
        try:
            return int(str(v).strip())
        except Exception:
            raise ValueError(what)
    if op == "REPLACE_RANGE":
        fp = _safe(d, a.get("path", ""))
        if not os.path.isfile(fp): return f"no such file {a.get('path')}", False
        try:
            st, en = _int(a.get("start", 1), "start"), _int(a.get("end", 1), "end")
        except ValueError as e:
            return (f"replace_range: {e} must be an integer line number "
                    f"(got {a.get('start')!r}/{a.get('end')!r}). Read the file "
                    f"first to get real line numbers."), False
        lines = open(fp).read().split("\n")
        new = a.get("content", "").split("\n")
        lines[st-1:en] = new
        open(fp, "w").write("\n".join(lines))
        ok, out = H.cargo_check(d)
        return f"replaced L{st}-{en}. cargo check {'PASS' if ok else 'FAIL'}\n" + ("" if ok else out), True
    if op == "INSERT_AT":
        fp = _safe(d, a.get("path", ""))
        if not os.path.isfile(fp): return f"no such file {a.get('path')}", False
        try:
            al = _int(a.get("after_line", 0), "after_line")
        except ValueError as e:
            return (f"insert_at: {e} must be an integer line number "
                    f"(got {a.get('after_line')!r})."), False
        lines = open(fp).read().split("\n")
        lines[al:al] = a.get("content", "").split("\n")
        open(fp, "w").write("\n".join(lines))
        ok, out = H.cargo_check(d)
        return f"inserted after L{al}. cargo check {'PASS' if ok else 'FAIL'}\n" + ("" if ok else out), True
    if op == "WRITE_FILE":
        fp = _safe(d, a.get("path", ""))
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        open(fp, "w").write(a.get("content", ""))
        ok, out = H.cargo_check(d)
        return f"wrote {a.get('path')}. cargo check {'PASS' if ok else 'FAIL'}\n" + ("" if ok else out), True
    if op == "DONE":
        return "__DONE__", False
    return f"(unknown op {op})", False

# ── arm C: bare-verb text parser ───────────────────────────────────────────
BODY_VERBS = {"REPLACE_RANGE", "INSERT_AT", "WRITE_FILE"}

# Control-token leakage hygiene — mirrors the spirit of miniswe's
# strip_xml_tool_blocks / has_tool_call_leak. Gemma, handed a tool-shaped
# surface in prose, leaks `<|channel>` / `<|tool_call>call:VERB ...<tool_call|>`
# and fakes tool-call JSON. A real agent sanitizes this; so must a fair probe.
_CTRL = re.compile(r"<\|?[A-Za-z0-9_]+\|?>")
def desanitize(text):
    t = _CTRL.sub("", text)
    t = re.sub(r"(?im)^[\W_]*call:\s*", "", t)        # faked `call:`/`_call:`
    t = re.sub(r"(?im)\bcall:\s*(?=[A-Z_]{3,}\b)", "", t)
    # trim leading junk before a recognized verb on a line
    t = re.sub(r"(?im)^[\W_]+(?=(?:READ|SEARCH-REPO|SHELL|PLAN|REFACTOR|"
               r"REPLACE_RANGE|INSERT_AT|WRITE_FILE|DONE)\b)", "", t)
    return t

def parse_C(text):
    """Line-anchored, tolerant: leading spaces / bullets / backticks / leaked
    control tokens / faked `call:`/JSON-ish arg forms all accepted."""
    text = desanitize(text)
    ops = []
    lines = text.split("\n")
    i = 0
    def clean(l): return l.strip().lstrip("-*` ").strip()
    while i < len(lines):
        raw = lines[i]; ln = clean(raw); up = ln.upper()
        i += 1
        if up.startswith("READ "):
            ops.append(("READ", {"path": ln[5:].strip()}))
        elif up.startswith("SEARCH-REPO "):
            ops.append(("SEARCH", {"query": ln[12:].strip()}))
        elif up.startswith("SHELL "):
            ops.append(("SHELL", {"command": ln[6:].strip()}))
        elif up.startswith("PLAN SET"):
            body = []
            while i < len(lines) and clean(lines[i]) and not _is_verb(clean(lines[i])):
                body.append(lines[i]); i += 1
            ops.append(("PLAN_SET", {"content": "\n".join(body) or "plan"}))
        elif up.startswith("PLAN CHECK"):
            ops.append(("PLAN_CHECK", {"step": ln.split()[-1] if ln.split() else "1"}))
        elif up.startswith("REFACTOR"):
            ops.append(("REFACTOR", _refactor_args(ln[8:])))
        elif up.split(" ")[0] in BODY_VERBS:
            parts = ln.split()
            verb = parts[0].upper()
            a = {}
            if verb == "REPLACE_RANGE" and len(parts) >= 4:
                a = {"path": parts[1], "start": parts[2], "end": parts[3]}
            elif verb == "INSERT_AT" and len(parts) >= 3:
                a = {"path": parts[1], "after_line": parts[2]}
            elif verb == "WRITE_FILE" and len(parts) >= 2:
                a = {"path": parts[1]}
            body, i = _read_fenced(lines, i)
            a["content"] = body
            ops.append((verb, a))
        elif up == "DONE":
            ops.append(("DONE", {}))
    return ops

VERB_PREFIXES = ("READ ", "SEARCH-REPO ", "SHELL ", "PLAN SET", "PLAN CHECK",
                 "REFACTOR", "REPLACE_RANGE", "INSERT_AT", "WRITE_FILE", "DONE")
def _is_verb(l): u = l.upper(); return any(u.startswith(p) for p in VERB_PREFIXES)

def _kv(s):
    """Parse  key=val | key: val | key:"quoted"  pairs (tolerant)."""
    out = {}
    for m in re.finditer(r'(\w+)\s*[=:]\s*("([^"]*)"|[^,\s}]+)', s):
        out[m.group(1)] = m.group(3) if m.group(3) is not None else m.group(2).strip('"')
    return out

def _refactor_args(s):
    """Accept `add_param path=.. name=..` AND json-ish
    `{add_param: "src/lib.rs", name: "greet", ...}` (Gemma's faked form)."""
    s = s.strip().lstrip("{").rstrip("}")
    kv = _kv(s)
    a = {}
    for act in ("add_param", "drop_param", "rename"):
        if act in kv:                       # json-ish: action key -> path value
            a["action"] = act
            if kv[act] not in ("", "true", "True"):
                a["path"] = kv[act]
        elif re.search(rf"\b{act}\b", s):   # bare leading word form
            a["action"] = act
    a.setdefault("action", "add_param")
    for k in ("path", "name", "new_param", "position", "callsite_fill_in", "fill"):
        if k in kv:
            a[k] = kv[k]
    return a

def _read_fenced(lines, i):
    """Body between <<< and >>> (or a ``` fence). Returns (body, new_i)."""
    while i < len(lines) and lines[i].strip() == "":
        i += 1
    if i < len(lines) and (lines[i].strip().startswith("<<<") or lines[i].strip().startswith("```")):
        i += 1
    body = []
    while i < len(lines) and not (lines[i].strip().startswith(">>>") or lines[i].strip() == "```"):
        body.append(lines[i]); i += 1
    if i < len(lines): i += 1
    return "\n".join(body), i

# ── arm D: OpenAI grouped schema (mirrors definitions.rs) ──────────────────
TOOLS_D = [
 {"type":"function","function":{"name":"file","description":"File ops. action: read|search|shell.",
  "parameters":{"type":"object","properties":{"action":{"type":"string"},"path":{"type":"string"},
   "query":{"type":"string"},"command":{"type":"string"}},"required":["action"]}}},
 {"type":"function","function":{"name":"plan","description":"action: set (UNLOCKS edit tools) | check.",
  "parameters":{"type":"object","properties":{"action":{"type":"string"},"content":{"type":"string"},
   "step":{"type":"integer"}},"required":["action"]}}},
 {"type":"function","function":{"name":"refactor","description":"Atomic def+callsites. action: add_param.",
  "parameters":{"type":"object","properties":{"action":{"type":"string"},"path":{"type":"string"},
   "name":{"type":"string"},"new_param":{"type":"string"},"position":{"type":"string"},
   "callsite_fill_in":{"type":"string"}},"required":["action"]}}},
 {"type":"function","function":{"name":"replace_range","description":"Replace lines [start..=end].",
  "parameters":{"type":"object","properties":{"path":{"type":"string"},"start":{"type":"integer"},
   "end":{"type":"integer"},"content":{"type":"string"}},"required":["path","start","end","content"]}}},
 {"type":"function","function":{"name":"insert_at","description":"Insert after a line.",
  "parameters":{"type":"object","properties":{"path":{"type":"string"},"after_line":{"type":"integer"},
   "content":{"type":"string"}},"required":["path","after_line","content"]}}},
 {"type":"function","function":{"name":"write_file","description":"Overwrite whole file.",
  "parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},
   "required":["path","content"]}}},
]
def tc_to_op(name, a):
    if name == "file":
        act = a.get("action", "")
        return {"read":"READ","search":"SEARCH","shell":"SHELL"}.get(act, "READ"), a
    if name == "plan":
        return ("PLAN_SET" if a.get("action") == "set" else "PLAN_CHECK"), a
    if name == "refactor": return "REFACTOR", a
    if name == "replace_range": return "REPLACE_RANGE", a
    if name == "insert_at": return "INSERT_AT", a
    if name == "write_file": return "WRITE_FILE", a
    return "?", a

# ── prompts ────────────────────────────────────────────────────────────────
RICH = ("You are working in a code repository. Tools: file(read/search/shell), "
        "plan(set/check), refactor(add_param — atomically updates a function "
        "definition AND its callsites), replace_range, insert_at, write_file. "
        "Edit tools (refactor/replace_range/insert_at/write_file) are LOCKED "
        "until you set a plan. After each edit the project is compiled and the "
        "result fed back; fix errors. Output the single token DONE when the "
        "task is complete and compiles.")
SYS_C = (RICH + "\n\nYou CANNOT call tools. Invoke by writing the verb at the "
 "start of a line:\n"
 "READ <path>\nSEARCH-REPO <text>\nSHELL <cmd>\n"
 "PLAN SET\n<plan lines...>\nPLAN CHECK <step>\n"
 "REFACTOR add_param path=src/lib.rs name=greet new_param=\"shout: bool\" "
 "position=after:name callsite_fill_in=shout\n"
 "REPLACE_RANGE <path> <start> <end>\n<<<\n<new lines>\n>>>\n"
 "INSERT_AT <path> <after_line>\n<<<\n<lines>\n>>>\n"
 "WRITE_FILE <path>\n<<<\n<contents>\n>>>\nDONE")
SYS_D = RICH + "\n\nUse the provided tools (function calls)."

# ── arm E: FLAT single-purpose tools, OpenAI tool-calls ────────────────────
# Same handlers/gate/oracle as D — only the schema SHAPE changes: no `action`
# discriminator, no position/callsite_fill_in DSL, one tool per intent,
# self-evident param names, all required. Tests "too many / wrongly described
# tools" directly: E vs D isolates schema shape, holding modality + power.
TOOLS_E = [
 {"type":"function","function":{"name":"read_file","description":"Read a file.",
  "parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
 {"type":"function","function":{"name":"search","description":"Search the crate for literal text.",
  "parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},
 {"type":"function","function":{"name":"shell","description":"Run a shell command in the crate root.",
  "parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}},
 {"type":"function","function":{"name":"set_plan","description":"Record your step-by-step plan. UNLOCKS the edit tools.",
  "parameters":{"type":"object","properties":{"plan":{"type":"string"}},"required":["plan"]}}},
 {"type":"function","function":{"name":"add_function_param",
  "description":"Add a parameter to a function and update every callsite, atomically.",
  "parameters":{"type":"object","properties":{
    "path":{"type":"string","description":"file containing the function definition"},
    "function":{"type":"string","description":"function name"},
    "param":{"type":"string","description":"full param declaration, e.g. 'shout: bool'"},
    "call_value":{"type":"string","description":"expression to pass at every callsite, e.g. 'shout'"}},
   "required":["path","function","param","call_value"]}}},
 {"type":"function","function":{"name":"drop_function_param",
  "description":"Remove a parameter from a function and every callsite.",
  "parameters":{"type":"object","properties":{"path":{"type":"string"},
    "function":{"type":"string"},"param":{"type":"string"}},"required":["path","function","param"]}}},
 {"type":"function","function":{"name":"rename_symbol",
  "description":"Rename a function/type/variable across the crate.",
  "parameters":{"type":"object","properties":{"path":{"type":"string"},
    "old_name":{"type":"string"},"new_name":{"type":"string"}},"required":["path","old_name","new_name"]}}},
 {"type":"function","function":{"name":"replace_range","description":"Replace lines [start..=end] with content.",
  "parameters":{"type":"object","properties":{"path":{"type":"string"},"start":{"type":"integer"},
   "end":{"type":"integer"},"content":{"type":"string"}},"required":["path","start","end","content"]}}},
 {"type":"function","function":{"name":"insert_at","description":"Insert content after a line (0=top).",
  "parameters":{"type":"object","properties":{"path":{"type":"string"},"after_line":{"type":"integer"},
   "content":{"type":"string"}},"required":["path","after_line","content"]}}},
 {"type":"function","function":{"name":"write_file","description":"Overwrite a whole file.",
  "parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},
   "required":["path","content"]}}},
]
def tc_to_op_E(name, a):
    if name == "read_file":  return "READ", {"path": a.get("path", "")}
    if name == "search":     return "SEARCH", {"query": a.get("query", "")}
    if name == "shell":      return "SHELL", {"command": a.get("command", "")}
    if name == "set_plan":   return "PLAN_SET", {"content": a.get("plan", "plan")}
    if name == "add_function_param":
        return "REFACTOR", {"action": "add_param", "path": a.get("path", ""),
                            "name": a.get("function", ""), "new_param": a.get("param", ""),
                            "callsite_fill_in": a.get("call_value", "")}
    if name == "drop_function_param":
        return "REFACTOR", {"action": "drop_param", "path": a.get("path", ""),
                            "name": a.get("function", ""), "param": a.get("param", "")}
    if name == "rename_symbol":
        return "REFACTOR", {"action": "rename", "path": a.get("path", ""),
                            "name": a.get("old_name", ""), "new_name": a.get("new_name", "")}
    if name == "replace_range": return "REPLACE_RANGE", a
    if name == "insert_at":     return "INSERT_AT", a
    if name == "write_file":    return "WRITE_FILE", a
    return "?", a
SYS_E = (RICH.replace(
    "Tools: file(read/search/shell), plan(set/check), refactor(add_param — "
    "atomically updates a function definition AND its callsites), replace_range, "
    "insert_at, write_file.",
    "You have flat single-purpose tools: read_file, search, shell, set_plan, "
    "add_function_param, drop_function_param, rename_symbol, replace_range, "
    "insert_at, write_file.")
 + "\n\nUse the provided tools (function calls).")

# ── Reframed, NO-GATE prompt (the fix validated by prompt-probe.py): no
# "edit tools LOCKED until plan", one action per message + STOP, PLAN CHECK
# only AFTER a step is verified. Arms F (free-text) and G (flat tools)
# share this philosophy; only the invocation channel differs.
REFRAME = (
 "You are working in a code repository. Sketch a brief numbered plan, then EXECUTE "
 "it. Do exactly ONE action per message, then STOP and wait for its result — "
 "do not continue past one action. After a step's work is done and the "
 "compiler confirms it, mark that one step with a single line "
 "`PLAN CHECK <step>` then STOP. When the whole task is complete and the "
 "project compiles, output DONE.")
SYS_F = (REFRAME + "\n\nYou CANNOT call tools. Invoke by writing ONE verb at "
 "the start of a line, then STOP:\n"
 "READ <path>\nSEARCH-REPO <text>\nSHELL <cmd>\nPLAN SET\n<plan lines>\n"
 "PLAN CHECK <step>\nREPLACE_RANGE <path> <start> <end>\n<<<\n<new lines>\n>>>\n"
 "INSERT_AT <path> <after_line>\n<<<\n<lines>\n>>>\n"
 "WRITE_FILE <path>\n<<<\n<contents>\n>>>\nDONE")
SYS_G = (REFRAME + "\n\nYou have flat single-purpose tools: read_file, search, "
 "shell, set_plan, add_function_param, drop_function_param, rename_symbol, "
 "replace_range, insert_at, write_file. Make exactly ONE tool call per "
 "message.")

def run_trial(arm, idx):
    d = os.path.join(WORK, f"{arm}_{idx}")
    H.scaffold(d)
    sys_p = {"C": SYS_C, "D": SYS_D, "E": SYS_E, "F": SYS_F, "G": SYS_G}[arm]
    tools = None if arm in ("C", "F") else (TOOLS_E if arm in ("E", "G") else TOOLS_D)
    op_map = tc_to_op_E if arm in ("E", "G") else tc_to_op
    msgs = [{"role": "system", "content": sys_p},
            {"role": "user", "content": f"TASK: {H.TASK}\n\n{H.REPO_MAP}\n\nBegin."}]
    log, fmt_viol = [], 0
    reached_plan = reached_edit = False
    for rnd in range(1, MAX_ROUNDS + 1):
        try:
            resp = H.llm(msgs, tools)
        except Exception as e:
            log.append(f"r{rnd} LLM ERR {e}"); break
        m = resp["choices"][0]["message"]
        content = H.strip_think(m.get("content") or "")
        tcs = m.get("tool_calls") or []
        msgs.append({"role": "assistant", "content": m.get("content") or "",
                     **({"tool_calls": tcs} if tcs else {})})
        # normalize this turn's invocations into ops
        ops = []
        if arm in ("C", "F"):
            ops = parse_C(content)
        if arm in ("D", "E", "G") and tcs:
            for tc in tcs:
                try: aa = json.loads(tc["function"]["arguments"] or "{}")
                except Exception: aa = {}
                ops.append(op_map(tc["function"]["name"], aa))
        if not ops:
            fmt_viol += 1
            nudge = ("No recognized invocation. Use the verb grammar exactly."
                     if arm in ("C", "F") else "No tool call. Use the provided tools.")
            msgs.append({"role": "user", "content": nudge})
            log.append(f"r{rnd} NO-OP")
            continue
        names = []
        done = False
        for op, a in ops:
            names.append(op)
            if op == "PLAN_SET": reached_plan = True
            res, mutated = dispatch(op, a, d)
            if mutated: reached_edit = True
            if res == "__DONE__": done = True; continue
            if arm in ("D", "E", "G") and tcs:
                # answer each tool call (any valid id; llama.cpp is lenient)
                msgs.append({"role": "tool", "tool_call_id": tcs[0]["id"],
                             "content": res[:3500]})
            else:
                msgs.append({"role": "user", "content": res[:3500]})
        log.append(f"r{rnd} {names}")
        # success check after any mutation
        if reached_edit:
            ok, _ = H.cargo_check(d)
            if ok and H.behaves(d):
                log.append(f"r{rnd} SUCCESS")
                return dict(arm=arm, idx=idx, success=True, rounds=rnd,
                            reached_plan=reached_plan, reached_edit=reached_edit,
                            fmt_viol=fmt_viol, log=log)
        if done:
            ok, _ = H.cargo_check(d); s = ok and H.behaves(d)
            log.append(f"r{rnd} DONE success={s}")
            return dict(arm=arm, idx=idx, success=s, rounds=rnd,
                        reached_plan=reached_plan, reached_edit=reached_edit,
                        fmt_viol=fmt_viol, log=log)
    return dict(arm=arm, idx=idx, success=False, rounds=MAX_ROUNDS,
                reached_plan=reached_plan, reached_edit=reached_edit,
                fmt_viol=fmt_viol, log=log)

LABELS = {"C": "C rich+text  ", "D": "D rich+openai",
          "E": "E flat+openai", "F": "F reframe+text", "G": "G reframe+tool"}
def main():
    os.makedirs(WORK, exist_ok=True)
    arms = tuple(os.environ.get("ARMS", "C,D").split(","))
    res = []
    for arm in arms:
        for i in range(TRIALS):
            t0 = time.time()
            r = run_trial(arm, i); r["secs"] = round(time.time() - t0, 1)
            res.append(r)
            print(f"[{arm} {i}] success={r['success']} rounds={r['rounds']} "
                  f"plan={r['reached_plan']} edit={r['reached_edit']} "
                  f"fmt_viol={r['fmt_viol']} {r['secs']}s", flush=True)
            open(os.path.join(WORK, "results2.jsonl"), "a").write(json.dumps(r) + "\n")
    print(f"\n=== SUMMARY (arms={','.join(arms)}) ===")
    for arm in arms:
        a = [r for r in res if r["arm"] == arm]
        s = sum(r["success"] for r in a)
        pl = sum(r["reached_plan"] for r in a); ed = sum(r["reached_edit"] for r in a)
        print(f"{LABELS.get(arm, arm)}: success {s}/{len(a)}  reached_plan {pl}/{len(a)}  "
              f"reached_edit {ed}/{len(a)}  fmt_viol {sum(r['fmt_viol'] for r in a)}")

if __name__ == "__main__":
    main()
