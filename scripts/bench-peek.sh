#!/usr/bin/env bash
# bench-peek — read-only inspection helpers for a miniswe bench run.
#
# All subcommands are pure reads (cat, grep, ls, python3 on json). Safe to
# permission once and reuse. No side effects, no network.
#
# Usage:
#   scripts/bench-peek.sh status     [BENCH_DIR]      — score + counters
#   scripts/bench-peek.sh trajectory [BENCH_DIR]      — plan + edit tool-call timeline
#   scripts/bench-peek.sh tools      [BENCH_DIR] [N]  — tool list at dump #N (default: first + last)
#   scripts/bench-peek.sh thread     [BENCH_DIR] [N]  — last N messages from latest dump (default 10)
#   scripts/bench-peek.sh nudges     [BENCH_DIR]      — surface any plan/stall nudges that fired
#   scripts/bench-peek.sh all        [BENCH_DIR]      — status + trajectory + nudges (most common)
#
# BENCH_DIR is the result dir like benchmark_results/docker_2026..._<MODEL>/.
# Omit to auto-pick the most recent docker_* under benchmark_results/.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BENCH_ROOT="${REPO_DIR}/benchmark_results"

# Resolve BENCH_DIR (full path to a "docker_*" result root, NOT 00_baseline).
resolve_bench_dir() {
    local arg="${1:-}"
    if [ -n "$arg" ]; then
        # Strip trailing slash, strip 00_baseline suffix if user passed it
        arg="${arg%/}"
        arg="${arg%/00_baseline}"
        if [ -d "$arg" ]; then
            echo "$arg"
            return
        fi
        # Try interpreting as a name under benchmark_results
        if [ -d "${BENCH_ROOT}/${arg}" ]; then
            echo "${BENCH_ROOT}/${arg}"
            return
        fi
        echo "bench dir not found: $arg" >&2
        exit 1
    fi
    # Auto-pick most recent
    local latest
    latest=$(ls -td "${BENCH_ROOT}"/docker_* 2>/dev/null | head -1 || true)
    if [ -z "$latest" ]; then
        echo "no docker_* benches under ${BENCH_ROOT}" >&2
        exit 1
    fi
    echo "$latest"
}

# Print a header — gives the result dir name so output is self-labeling.
header() {
    local bench="$1"
    echo "=== $(basename "$bench") ==="
}

cmd_status() {
    local bench
    bench="$(resolve_bench_dir "${1:-}")"
    local res="${bench}/00_baseline"
    header "$bench"
    if [ -f "${res}/container.log" ]; then
        # Most useful slice: any 'compile:/build:/help:/test:/smoke:/parse:' check lines + FINAL.
        echo "--- checks ---"
        grep -E '^(compile|build|help|parse|test|smoke):|^=== ATTEMPT [0-9]+ RESULT|^=== FINAL:' \
            "${res}/container.log" || echo "(no check lines yet)"
    else
        echo "container.log not present yet"
    fi
    echo "--- counters ---"
    local n_req leaks wall
    # `set -euo pipefail` is active. grep with no matches returns non-zero
    # which triggers pipefail and double-prints the `|| echo 0` fallback —
    # wrap in a `|| true` group so wc only ever sees one source of input.
    n_req=$({ ls "${res}/llm_dumps" 2>/dev/null || true; } | wc -l)
    leaks=$({ grep -h 'tool-call leak' "${res}"/stderr_attempt*.txt 2>/dev/null || true; } | wc -l)
    wall=$(cat "${res}/wall_s.txt" 2>/dev/null || echo "?")
    echo "requests=${n_req} leaks=${leaks} wall=${wall}s"
}

cmd_trajectory() {
    local bench
    bench="$(resolve_bench_dir "${1:-}")"
    local res="${bench}/00_baseline"
    header "$bench"
    echo "--- plan + edit + change_signature tool calls (deduped by dump) ---"
    if ! ls "${res}/llm_dumps"/*.json >/dev/null 2>&1; then
        echo "(no dumps yet)"
        return
    fi
    for f in "${res}"/llm_dumps/*.json; do
        python3 -c "
import json, sys
try: b = json.load(open(sys.argv[1]))
except: sys.exit(0)
short = sys.argv[1].split('/')[-1].split('-')[-1]
TARGETS = {'plan','refactor','change_signature','edit_file','replace_range','insert_at','write_file','rename'}
# Only emit for the LAST 3 messages so we capture the CURRENT request's new action.
for m in b.get('messages',[])[-3:]:
    if m.get('role') != 'assistant':
        continue
    for tc in (m.get('tool_calls') or []):
        name = tc['function']['name']
        if name not in TARGETS:
            continue
        args = tc['function']['arguments']
        detail = ''
        try:
            a = json.loads(args)
            if name == 'plan':
                detail = a.get('action','')
            elif 'path' in a:
                detail = a.get('path','')
        except: pass
        print(f'{short}  {name:<18} {detail}')
" "$f" 2>/dev/null
    done | sort -u
}

cmd_tools() {
    local bench
    bench="$(resolve_bench_dir "${1:-}")"
    local res="${bench}/00_baseline"
    local idx="${2:-}"
    header "$bench"
    if ! ls "${res}/llm_dumps"/*.json >/dev/null 2>&1; then
        echo "(no dumps yet)"
        return
    fi
    local files
    if [ -n "$idx" ]; then
        files=$(ls "${res}/llm_dumps"/*.json | sed -n "$((idx+1))p")
        [ -z "$files" ] && { echo "no dump at index $idx"; return; }
    else
        # First + last
        files="$(ls "${res}/llm_dumps"/*.json | head -1)
$(ls "${res}/llm_dumps"/*.json | tail -1)"
    fi
    while IFS= read -r f; do
        [ -z "$f" ] && continue
        python3 -c "
import json, sys
b = json.load(open(sys.argv[1]))
tools = [t['function']['name'] for t in b.get('tools',[])]
label = sys.argv[1].split('/')[-1].split('-')[-1]
print(f'{label}  ({len(tools)} tools): {tools}')
" "$f"
    done <<< "$files"
}

cmd_thread() {
    local bench
    bench="$(resolve_bench_dir "${1:-}")"
    local res="${bench}/00_baseline"
    local n="${2:-10}"
    header "$bench"
    local last
    last=$(ls "${res}"/llm_dumps/*.json 2>/dev/null | tail -1 || true)
    if [ -z "$last" ]; then
        echo "(no dumps yet)"
        return
    fi
    python3 -c "
import json, sys
b = json.load(open(sys.argv[1]))
n = int(sys.argv[2])
msgs = b.get('messages', [])
print(f'(showing last {n} of {len(msgs)} messages)')
print()
for m in msgs[-n:]:
    role = m.get('role','?')
    tcs = m.get('tool_calls') or []
    if tcs:
        for tc in tcs:
            args = tc['function']['arguments']
            print(f'[{role}] CALL {tc[\"function\"][\"name\"]}: {args[:180]}')
    else:
        c = (m.get('content') or '')[:240]
        if c.strip():
            print(f'[{role}] {c}')
    print()
" "$last" "$n"
}

cmd_nudges() {
    local bench
    bench="$(resolve_bench_dir "${1:-}")"
    local res="${bench}/00_baseline"
    header "$bench"
    local last
    last=$(ls "${res}"/llm_dumps/*.json 2>/dev/null | tail -1 || true)
    if [ -z "$last" ]; then
        echo "(no dumps yet)"
        return
    fi
    python3 -c "
import json, sys
b = json.load(open(sys.argv[1]))
hits = []
for m in b.get('messages', []):
    if m.get('role') != 'user':
        continue
    c = m.get('content') or ''
    if '[Reminder' in c or '[WARNING' in c:
        # First line up to 200 chars is enough to identify which nudge.
        hits.append(c.split(']')[0][:200] + ']')
if not hits:
    print('(no nudges fired)')
else:
    for h in hits:
        print(h)
" "$last"
}

cmd_all() {
    local arg="${1:-}"
    cmd_status "$arg"
    echo
    cmd_trajectory "$arg"
    echo
    cmd_nudges "$arg"
}

case "${1:-}" in
    status|s)       shift; cmd_status "${1:-}";;
    trajectory|t)   shift; cmd_trajectory "${1:-}";;
    tools|T)        shift; cmd_tools "${1:-}" "${2:-}";;
    thread|m)       shift; cmd_thread "${1:-}" "${2:-}";;
    nudges|n)       shift; cmd_nudges "${1:-}";;
    all|"")         shift || true; cmd_all "${1:-}";;
    -h|--help|help)
        sed -n '2,16p' "$0"
        ;;
    *)
        echo "unknown subcommand: $1" >&2
        echo "see: $0 --help" >&2
        exit 1
        ;;
esac
