#!/usr/bin/env python3
"""Offline stuck-trigger evaluation over recorded bench run logs (gap 9/10).

Parses miniswe session logs from benchmark_results/docker_*/00_baseline/
miniswe_state/logs/*.log and simulates candidate "the agent is stuck"
triggers round-by-round, then scores them against hand-labeled ground-truth
windows (from .plan/findings/* of the 08-23/24 overnight queue).

No LLM calls. Pure log parsing. Run: python3 scripts/moments/trigger-eval.py

Trigger families:
  T1a zero-edit-N     : N rounds without ANY successful write-taxonomy call
  T1b zero-srcedit-N  : same but write_file/edits to *.md don't count
  T2a frozensig-N     : N rounds with the failure signature unchanged
                        (sig = last-fail-line hash + [ast] + [lsp project]
                         errors + check/gate/shell states; reads/plan don't
                         touch it, successful edits update ast/lsp parts)
  T2b frozensig+rev-N : T2a but any successful edit (new rev) also unfreezes
  T3  filereadloop-N  : N consecutive file(read) calls of the SAME path,
                        ignoring line ranges (jitter-tolerant loop key)
  T4  green-noedit-N  : lsp-project errors == 0 AND >=1 edit happened AND
                        N rounds since the last edit (the done-gate case)

All triggers arm only after the first plan(set) (or round 20, whichever
comes first). First fire per labeled segment is what is scored.
"""
import calendar, hashlib, json, os, re, sys
from glob import glob

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BR = os.path.join(REPO, "benchmark_results")

def H(s):
    return hashlib.md5(s.encode("utf-8", "replace")).hexdigest()[:10]

# ---------------------------------------------------------------- labels ---
# segments: (kind, start, end) with times "HH:MM:SS" (log-local UTC) or None
# kind: 'healthy' | 'stuck' ; gaps between segments are unscored buffer.
# note: what the stuck window is, per findings files.
LABELED = [
    ("north-r1-instr", "docker_20260823_200353*", "20260823_180434.log",
     [("healthy", None, "18:05:50"), ("stuck", "18:06:00", None)],
     "broken-AST read loop, 235 reads / 55 min"),
    ("north-r2-instr", "docker_20260823_210331*", "20260823_190414.log",
     [("healthy", None, "19:06:00"), ("stuck", "19:14:00", None)],
     "zero-edit read drift on clean tree, killed at 42 min"),
    ("north-r3-think", "docker_20260823_215011*", "20260823_195055.log",
     [("healthy", None, None)],
     "healthy to clock death (one call site short)"),
    ("glimmer-r1-a1", "docker_20260823_225151*", "20260823_205232.log",
     [("healthy", None, "21:17:30"), ("stuck", "21:19:00", None)],
     "jittered read loop on e2e_context.rs (14 notes, 0 escalations)"),
    ("glimmer-r1-a2", "docker_20260823_225151*", "20260823_213312.log",
     [("healthy", None, "21:34:00"), ("stuck", "21:35:00", None)],
     "zero-edit attempt, ends 'Understood.'"),
    ("glimmer-r1-a3", "docker_20260823_225151*", "20260823_214005.log",
     [("healthy", None, None)],
     "healthy repair to clock death"),
    ("glimmer-r2", "docker_20260823_235146*", "20260823_215227.log",
     [("healthy", None, "22:02:40"), ("stuck", "22:05:00", None)],
     "green at min 10, then 35-min can't-stop dither"),
    ("devstral-gap9", "docker_20260823_164918*", "20260823_145001.log",
     [("healthy", None, "14:53:00"), ("stuck", "14:54:00", "15:00:00")],
     "identical malformed refactor re-issued 21x (validator loop)"),
    ("laguna-1", "docker_20260823_172958*", "20260823_153040.log",
     [("healthy", None, None)], "6/6 in 640s"),
    ("laguna-2", "docker_20260824_004000*", "20260823_224044.log",
     [("healthy", None, None)], "6/6 in 566s"),
    ("laguna-3", "docker_20260824_005124*", "20260823_225209.log",
     [("healthy", None, None)], "6/6 in 792s"),
]

EDIT_TOOLS = {"replace_range", "insert_at", "write_file", "edit_file", "refactor"}

TS_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})\.(\d{3}) \[([^\]]+)\] ?(.*)$", re.S)

def tsec(h, m, s, ms="000"):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

def parse_hms(t):
    h, m, s = t.split(":")
    return tsec(h, m, s)

def fmt_t(sec):
    sec = int(sec) % 86400
    return f"{sec//3600:02d}:{sec%3600//60:02d}:{sec%60:02d}"

AST_RE = re.compile(r"\[ast\] (ok|broken[^\\\"]{0,80})")
LSP_RE = re.compile(r"\[lsp project\] (\d+) error")

def session_dumps(path):
    """Full request JSONs for this log's session, ordered by sequence number.

    Dump files are req-<session-epoch>-<n>-<seq>.json; the log filename is the
    session start (UTC). Match the prefix whose epoch is within 120s of it.
    """
    logs_dir = os.path.dirname(path)
    dumps_dir = os.path.normpath(os.path.join(logs_dir, "..", "..", "llm_dumps"))
    m = re.match(r"(\d{8})_(\d{6})", os.path.basename(path))
    if not m or not os.path.isdir(dumps_dir):
        return []
    d, t = m.group(1), m.group(2)
    epoch = calendar.timegm((int(d[:4]), int(d[4:6]), int(d[6:]),
                             int(t[:2]), int(t[2:4]), int(t[4:])))
    by_prefix = {}
    for f in os.listdir(dumps_dir):
        fm = re.match(r"req-(\d+)-\d+-(\d+)\.json$", f)
        if fm:
            by_prefix.setdefault(int(fm.group(1)), []).append((int(fm.group(2)), f))
    best = min(by_prefix, key=lambda e: abs(e - epoch), default=None)
    if best is None or abs(best - epoch) > 120:
        return []
    return [os.path.join(dumps_dir, f) for _, f in sorted(by_prefix[best])]

def last_tool_content(dump_path):
    try:
        msgs = json.load(open(dump_path, errors="replace")).get("messages", [])
    except Exception:
        return None
    for msg in reversed(msgs):
        if msg.get("role") == "tool":
            c = msg.get("content", "")
            if isinstance(c, list):
                c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
            return c
    return None

def parse_log(path):
    """Yield event dicts with absolute seconds (midnight-wrap corrected)."""
    events = []
    last = None
    day = 0
    pending_call = None
    dumps = session_dumps(path)
    req_i = 0
    with open(path, errors="replace") as f:
        for line in f:
            m = TS_RE.match(line)
            if not m:
                continue
            t = tsec(m.group(1), m.group(2), m.group(3), m.group(4))
            if last is not None and t < last - 6 * 3600:
                day += 86400
            last = t
            t += day
            tag, rest = m.group(5), m.group(6).rstrip("\n")
            if tag.startswith("round "):
                events.append({"t": t, "kind": "round", "n": int(tag.split()[1])})
            elif tag == "tool:call":
                name, _, argstr = rest.partition(" ")
                try:
                    args = json.loads(argstr) if argstr.startswith("{") else {}
                except Exception:
                    args = {}
                pending_call = (name, args)
            elif tag == "tool":
                ok = rest.startswith("✓")
                name, args = pending_call if pending_call else ("?", {})
                pending_call = None
                summary = rest.split("→", 1)[1].strip() if "→" in rest else rest
                events.append({"t": t, "kind": "tool", "name": name, "args": args,
                               "ok": ok, "summary": summary[:400]})
            elif tag == "llm:request":
                content = None
                if req_i < len(dumps):
                    content = last_tool_content(dumps[req_i])
                req_i += 1
                if content:
                    chunk = content[-4000:]
                    asts = AST_RE.findall(chunk)
                    lsp = None
                    for m2 in LSP_RE.finditer(chunk):
                        lsp = m2  # keep last match
                    if asts or lsp:
                        lsp_n, lsp_hash = None, None
                        if lsp:
                            lsp_n = int(lsp.group(1))
                            detail = chunk[lsp.end():lsp.end() + 300]
                            detail = detail.split("[revisions]")[0]
                            lsp_hash = H(f"{lsp_n}|{detail}")
                        events.append({"t": t, "kind": "lspstate",
                                       "ast": asts[-1] if asts else None,
                                       "lsp_n": lsp_n, "lsp_hash": lsp_hash})
    return events

class Sim:
    """Round-by-round trigger simulation over one log's events."""
    def __init__(self, events):
        self.fires = {}   # trigger -> list of (round, t)
        st = {
            "round": 0, "armed_round": None, "armed_t": None, "plan_set": False,
            "last_edit_any": None, "last_edit_src": None, "edit_happened": False,
            "last_edit_t": None,
            "ast": None, "lsp_n": None, "lsp_hash": None, "green": None,
            "fail_hash": None, "check_state": None, "shell_state": None,
            "sig_a_round": 0, "sig_b_round": 0, "sig_a": None, "sig_b": None,
            "sig_a_t": None, "sig_b_t": None, "t": 0.0,
            "read_streak": 0, "read_path": None, "edit_id": None,
        }
        self.max_round = 0
        for ev in events:
            if ev["kind"] == "round":
                st["round"] = ev["n"]
                self.max_round = max(self.max_round, ev["n"])
                st["t"] = ev["t"]
                if st["armed_round"] is None and (st["plan_set"] or ev["n"] >= 20):
                    st["armed_round"] = ev["n"]
                    st["armed_t"] = ev["t"]
                    self.resig(st)  # init sig: "no signal at all" can freeze too
                self.eval_round(st, ev)
            elif ev["kind"] == "tool":
                self.on_tool(st, ev)
            elif ev["kind"] == "lspstate":
                st["t"] = ev["t"]
                changed = False
                for k in ("ast", "lsp_n", "lsp_hash"):
                    if ev[k] is not None and ev[k] != st[k]:
                        st[k] = ev[k]; changed = True
                if ev["lsp_n"] is not None:
                    st["green"] = ev["lsp_n"] == 0
                if ev["ast"] is not None and ev["ast"] != "ok":
                    st["green"] = False
                if changed:
                    self.resig(st)

    def resig(self, st):
        a = H("|".join(str(st[k]) for k in
                       ("ast", "lsp_hash", "fail_hash", "check_state", "shell_state")))
        b = H(a + str(st["edit_id"]))
        if a != st["sig_a"]:
            st["sig_a"], st["sig_a_round"], st["sig_a_t"] = a, st["round"], st["t"]
        if b != st["sig_b"]:
            st["sig_b"], st["sig_b_round"], st["sig_b_t"] = b, st["round"], st["t"]

    def on_tool(self, st, ev):
        name, args, ok = ev["name"], ev["args"], ev["ok"]
        action = args.get("action", "")
        st["t"] = ev["t"]
        # read streak (jitter-tolerant: keyed by path only)
        if name == "file" and action == "read":
            p = args.get("path")
            st["read_streak"] = st["read_streak"] + 1 if p == st["read_path"] else 1
            st["read_path"] = p
        else:
            st["read_streak"], st["read_path"] = 0, None
        # edits
        if ok and name in EDIT_TOOLS:
            if name == "refactor" and action not in ("add_param", "drop_param", "rename"):
                pass
            else:
                st["last_edit_any"] = st["round"]
                st["last_edit_t"] = ev["t"]
                st["edit_happened"] = True
                path = str(args.get("path", ""))
                if not (name == "write_file" and path.endswith(".md")) and \
                   not path.endswith(".md"):
                    st["last_edit_src"] = st["round"]
                st["edit_id"] = H(name + ev["summary"])
                self.resig(st)
        # failure / check / shell state
        if not ok:
            st["fail_hash"] = H(name + ev["summary"]); self.resig(st)
        elif name == "check":
            failed = "FAILED" in ev["summary"]
            st["check_state"] = "check:" + ("FAILED" if failed else "OK")
            st["green"] = not failed
            self.resig(st)
        elif name == "plan":
            if action == "set":
                st["plan_set"] = True
            if action == "check":
                st["check_state"] = "gate:" + H(ev["summary"][:80])
                if "compile gate passed" in ev["summary"]:
                    st["green"] = True
                elif "FAILED" in ev["summary"]:
                    st["green"] = False
                self.resig(st)
        elif name == "shell":
            m = re.search(r"\[shell: exit (\d+)", ev["summary"])
            if m:
                st["shell_state"] = H(str(args.get("command", "")) + "exit" + m.group(1))
                cmd = str(args.get("command", ""))
                if "cargo test" in cmd or "cargo check" in cmd or "cargo build" in cmd:
                    st["green"] = m.group(1) == "0"
                self.resig(st)

    def fire(self, key, st, ev):
        self.fires.setdefault(key, []).append((st["round"], ev["t"], bool(st["green"])))

    def eval_round(self, st, ev):
        if st["armed_round"] is None:
            return
        r, t = st["round"], ev["t"]
        base = st["armed_round"]
        edit_dt = t - (st["last_edit_t"] if st["last_edit_t"] is not None else st["armed_t"])
        sig_dt = t - (st["sig_a_t"] if st["sig_a_t"] is not None else st["armed_t"])
        for n in (10, 15, 20, 30):
            if r - (st["last_edit_any"] if st["last_edit_any"] is not None else base) >= n:
                self.fire(f"T1a-zeroedit-{n}", st, ev)
                if edit_dt >= 240:
                    self.fire(f"T1c-zeroedit-{n}+4m", st, ev)
            if r - (st["last_edit_src"] if st["last_edit_src"] is not None else base) >= n:
                self.fire(f"T1b-zerosrc-{n}", st, ev)
            if st["sig_a"] is not None and r - max(st["sig_a_round"], base) >= n:
                self.fire(f"T2a-frozensig-{n}", st, ev)
                if sig_dt >= 240:
                    self.fire(f"T2c-frozensig-{n}+4m", st, ev)
            if st["sig_b"] is not None and r - max(st["sig_b_round"], base) >= n:
                self.fire(f"T2b-frozensig+rev-{n}", st, ev)
            if st["edit_happened"] and st["green"] and \
               st["last_edit_any"] is not None and r - st["last_edit_any"] >= n:
                self.fire(f"T4-greennoedit-{n}", st, ev)
        for n in (4, 6, 10):
            if st["read_streak"] >= n:
                self.fire(f"T3-readloop-{n}", st, ev)

def segments_abs(segs, t0, t1):
    out = []
    for kind, a, b in segs:
        sa = t0 if a is None else parse_hms(a) + (86400 if parse_hms(a) < t0 - 6 * 3600 else 0)
        sb = t1 if b is None else parse_hms(b) + (86400 if parse_hms(b) < t0 - 6 * 3600 else 0)
        out.append((kind, sa, sb))
    return out

def main():
    runs = []
    for label, dglob, logname, segs, note in LABELED:
        dirs = glob(os.path.join(BR, dglob))
        if not dirs:
            print(f"!! missing dir for {label}", file=sys.stderr); continue
        path = os.path.join(dirs[0], "00_baseline", "miniswe_state", "logs", logname)
        events = parse_log(path)
        rounds = [e for e in events if e["kind"] == "round"]
        if not rounds:
            print(f"!! no rounds in {label}", file=sys.stderr); continue
        t0, t1 = rounds[0]["t"], events[-1]["t"]
        sim = Sim(events)
        runs.append((label, sim, segments_abs(segs, t0, t1), note, t0))

    triggers = sorted({k for _, s, _, _, _ in runs for k in s.fires} |
                      {f"T1a-zeroedit-{n}" for n in (10, 15, 20, 30)} |
                      {f"T1b-zerosrc-{n}" for n in (10, 15, 20, 30)} |
                      {f"T2a-frozensig-{n}" for n in (10, 15, 20, 30)} |
                      {f"T2b-frozensig+rev-{n}" for n in (10, 15, 20, 30)} |
                      {f"T1c-zeroedit-{n}+4m" for n in (10, 15, 20, 30)} |
                      {f"T2c-frozensig-{n}+4m" for n in (10, 15, 20, 30)} |
                      {f"T3-readloop-{n}" for n in (4, 6, 10)} |
                      {f"T4-greennoedit-{n}" for n in (10, 15, 20, 30)},
                      key=lambda k: (k.split("-")[0],
                                     int(re.sub(r"[^0-9].*$", "", k.rsplit("-", 1)[1]))))

    stuck_segs = [(lbl, sa, sb) for lbl, _, segs, _, _ in runs
                  for kind, sa, sb in segs if kind == "stuck"]
    print(f"corpus: {len(runs)} logs, {len(stuck_segs)} stuck segments, "
          f"{sum(1 for _,_,segs,_,_ in runs for k,_,_ in segs if k=='healthy')} healthy segments")
    print()
    hdr = f"{'trigger':26} {'stuck-hit':9} {'med-late':16} {'FPr':5} {'FPg':5} detail"
    print(hdr); print("-" * len(hdr))
    for trig in triggers:
        hits, lates, fp_red, fp_grn, fp_ex, miss = 0, [], 0, 0, [], []
        for label, sim, segs, note, t0 in runs:
            fires = sim.fires.get(trig, [])
            for kind, sa, sb in segs:
                seg_fires = [f for f in fires if sa <= f[1] <= sb]
                if kind == "stuck":
                    if seg_fires:
                        hits += 1
                        lates.append((seg_fires[0][1] - sa) / 60.0)
                    else:
                        miss.append(label)
                else:
                    # ignore fires within 3 min before a stuck segment (label slack)
                    real = [f for f in seg_fires
                            if not any(0 <= ssa - f[1] <= 180 for k2, ssa, _ in segs
                                       if k2 == "stuck")]
                    if real:
                        if real[0][2]:
                            fp_grn += 1
                        else:
                            fp_red += 1
                        fp_ex.append(f"{label}@{fmt_t(real[0][1])}({'G' if real[0][2] else 'R'})")
        lates.sort()
        med = f"{lates[len(lates)//2]:.1f}min" if lates else "-"
        detail = []
        if miss:
            detail.append("miss:" + ",".join(miss))
        if fp_ex:
            detail.append("fp:" + ",".join(fp_ex[:4]))
        print(f"{trig:26} {hits}/{len(stuck_segs):<7} {med:16} {fp_red:<5} {fp_grn:<5} {'; '.join(detail)}")
    print()
    print("stuck segments:", ", ".join(f"{l}[{fmt_t(a)}-{fmt_t(b)}]" for l, a, b in stuck_segs))
    print("\nper-run first fires (labeled corpus):")
    show = ["T1c-zeroedit-15+4m", "T2c-frozensig-15+4m", "T2b-frozensig+rev-15",
            "T3-readloop-6", "T4-greennoedit-15"]
    print(f"{'run':16}" + "".join(f"{t.split('-')[0]+'-'+t.rsplit('-',1)[1]:>10}" for t in show))
    for label, sim, segs, note, t0 in runs:
        row = f"{label:16}"
        for t in show:
            f = sim.fires.get(t, [])
            row += f"{fmt_t(f[0][1])[0:8]:>10}" if f else f"{'-':>10}"
        print(row + f"   ({note})")

if __name__ == "__main__":
    main()
