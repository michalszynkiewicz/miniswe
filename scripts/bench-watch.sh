#!/usr/bin/env bash
# Read-only status snapshot of a miniswe benchmark run. Safe to run
# repeatedly. Usage: bench-watch.sh [RESULTS_DIR]
# Default targets the in-flight validation run.
set -uo pipefail

DIR="${1:-/home/michal/dev/miniswe/benchmark_results/docker_20260515_225155_Devstral-Small-2-24B-Instruct-2512-UD-Q4}"
BL="$DIR/00_baseline"

if [ ! -d "$BL" ]; then
    echo "not present yet: $BL"
    exit 0
fi

echo "=== $DIR"
echo "--- result:"
grep -E "ATTEMPT.*RESULT|=== ATTEMPT|=== FINAL" "$BL/container.log" 2>/dev/null | tail -8
echo "--- checks (per attempt):"
grep -E "(compile|build|help|parse|test|smoke):(PASS|FAIL)" "$BL/container.log" 2>/dev/null | tail -12
echo "--- wall_s: $(cat "$BL/wall_s.txt" 2>/dev/null || echo n/a)   dumps: $(ls "$BL/llm_dumps" 2>/dev/null | wc -l)"

python3 - "$BL" <<'PY'
import json, glob, collections, sys, re
bl = sys.argv[1]
fs = sorted(glob.glob(f"{bl}/llm_dumps/*.json"))
if not fs:
    print("--- (no llm dumps yet)"); sys.exit()
phase = collections.Counter()
plan_set_seen = False
flip_idx = None
prev = None
for i, f in enumerate(fs):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    s = d["messages"][0]["content"] if d.get("messages") else ""
    if "you are in the EDITING phase" in s:
        ph = "POST"
    elif "WORKFLOW: explore" in s:
        ph = "PRE"
    else:
        ph = "other"
    phase[ph] += 1
    if prev == "PRE" and ph == "POST" and flip_idx is None:
        flip_idx = i
    prev = ph
last = json.load(open(fs[-1]))
tools = [t["function"]["name"] for t in last.get("tools", [])]
print(f"--- prompt phase across {len(fs)} dumps: {dict(phase)}")
print(f"    PRE->POST flip at dump idx: {flip_idx if flip_idx is not None else 'NOT YET'}")
print(f"    latest dump tools: refactor={'refactor' in tools} edit_file={'edit_file' in tools}")

# refactor adoption + arg cleanliness
calls = collections.Counter()
rf_clean = rf_total = 0
sample = None
for f in fs:
    try:
        d = json.load(open(f))
    except Exception:
        continue
    for m in d.get("messages", []):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                n = tc["function"]["name"]
                calls[n] += 1
                if n == "refactor":
                    rf_total += 1
                    try:
                        a = json.loads(tc["function"]["arguments"])
                        pos = a.get("position", "")
                        ok = (pos in ("start", "end")
                              or (isinstance(pos, str) and pos.startswith("after:")
                                  and re.fullmatch(r"\w+", pos[6:] or "")))
                        if ok and a.get("name"):
                            rf_clean += 1
                    except Exception:
                        pass
                    if sample is None:
                        sample = tc["function"]["arguments"][:200]
top = ", ".join(f"{k}:{v}" for k, v in calls.most_common(8))
print(f"--- tool calls (cumulative across dumps): {top}")
print(f"--- refactor: {rf_total} calls, well-formed(name+clean position): {rf_clean}/{rf_total or 1}")
if sample:
    print(f"    sample refactor args: {sample}")
PY

echo "--- last 8 actions:"
grep -E "(→|✗|✓)" "$BL"/stderr_attempt*.txt 2>/dev/null | tail -8 | sed 's/\x1b\[[0-9;]*m//g'
