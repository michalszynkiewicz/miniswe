#!/usr/bin/env bash
# run-compaction-bench.sh — A/B miniswe's conversation-compaction strategies.
#
# Compares four compaction strategies on the SAME task as run-benchmark-docker.sh
# (the --system-prompt-override feature, from scratch), each in its own fresh
# Docker container, N runs per strategy:
#
#   unified              miniswe production: rolling LLM summary + plan-anchor
#                        + disk archive, keep recent raw
#   sliding_window       pure truncation: drop oldest, keep recent within budget
#   rolling_summary      textbook rolling LLM summary (no archive, no plan-anchor,
#                        neutral prompt), keep recent raw
#   observation_masking  keep the trajectory; elide old tool outputs, keep last K
#
# All four fire at the SAME raw_budget trigger — only the action differs. To
# isolate the compaction variable, every OTHER experimental context-management
# knob is forced OFF (notably gate_context_reset, which the regular bench leaves
# on). The strategy is selected via config `[context] compaction`.
#
# The image is built ONCE (it bakes in the current-tree miniswe binary, which
# carries the new compaction code), then reused for all runs.
#
# Usage:
#   ./scripts/run-compaction-bench.sh [--model gemma-4-26B-A4B-it] [--runs 3] \
#                                     [--timeout 1800] [--max-rounds 50] \
#                                     [--strategies "unified sliding_window ..."]
#
# LLM server must already be running on $LLAMA_ENDPOINT (default localhost:8464).
set -uo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE_NAME="miniswe-bench"
BASELINE_SHA="cc34d2626faf32c1b6dd1b8b33af693fb936b098"
LLAMA_ENDPOINT="${LLAMA_ENDPOINT:-http://localhost:8464}"

# Defaults
MODEL="gemma-4-26B-A4B-it"
RUNS=3
TIMEOUT=1800
MAX_ROUNDS=50
MAX_ATTEMPTS=3
TEMPERATURE=0.2
STRATEGIES="unified sliding_window rolling_summary observation_masking"
RESULTS_DIR_OVERRIDE=""
TASK="Add a CLI flag --system-prompt-override (short: -s) that takes a string and replaces the default system prompt with the provided text. When this flag is set, skip all context providers and just use the override text as the system message. Make sure it works for both single-shot and interactive modes."

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL="$2";      shift 2 ;;
        --runs)       RUNS="$2";       shift 2 ;;
        --timeout)    TIMEOUT="$2";    shift 2 ;;
        --max-rounds) MAX_ROUNDS="$2"; shift 2 ;;
        --strategies) STRATEGIES="$2"; shift 2 ;;
        --results-dir) RESULTS_DIR_OVERRIDE="$2"; shift 2 ;;  # resume into an existing matrix dir
        --task)       TASK="$2";       shift 2 ;;
        -*) echo "Unknown: $1" >&2; exit 1 ;;
        *)  echo "Unexpected arg: $1" >&2; exit 1 ;;
    esac
done

MODEL_TAG="$(
    curl -fsS --max-time 3 "${LLAMA_ENDPOINT}/v1/models" 2>/dev/null \
    | python3 -c "import json,sys; r=json.load(sys.stdin); print((r.get('data') or [{}])[0].get('id','?'))" 2>/dev/null \
    | sed -E 's/\.gguf$//; s/[^A-Za-z0-9._-]/_/g' | cut -c1-40
)"
MODEL_TAG="${MODEL_TAG:-unknown}"
if [[ -n "${RESULTS_DIR_OVERRIDE}" ]]; then
    [[ "${RESULTS_DIR_OVERRIDE}" = /* ]] || RESULTS_DIR_OVERRIDE="${REPO_DIR}/${RESULTS_DIR_OVERRIDE}"
    RESULTS_DIR="${RESULTS_DIR_OVERRIDE}"
else
    RESULTS_DIR="${REPO_DIR}/benchmark_results/compaction_$(date +%Y%m%d_%H%M%S)_${MODEL_TAG}"
fi
ACTIVE_CONTAINER=""

cleanup() {
    set +e
    [[ -n "${ACTIVE_CONTAINER}" ]] && docker rm -f "${ACTIVE_CONTAINER}" >/dev/null 2>&1
    # NOTE: deliberately do NOT remove the image here — it is built once and
    # reused across every run in this matrix.
}
trap cleanup EXIT INT TERM

mkdir -p "${RESULTS_DIR}"

echo "=== Conversation-compaction strategy benchmark ==="
echo "Model:      ${MODEL}  (server: ${MODEL_TAG})"
echo "Endpoint:   ${LLAMA_ENDPOINT}"
echo "Strategies: ${STRATEGIES}"
echo "Runs each:  ${RUNS}   Timeout: ${TIMEOUT}s   Max rounds: ${MAX_ROUNDS}   Attempts: ${MAX_ATTEMPTS}"
echo "Results:    ${RESULTS_DIR}"
echo ""

# Verify LLM server is reachable before spending minutes building.
if ! curl -fsS --max-time 5 "${LLAMA_ENDPOINT}/v1/models" >/dev/null 2>&1; then
    echo "ERROR: LLM server not responding at ${LLAMA_ENDPOINT}" >&2
    exit 1
fi

# ── Build the image ONCE ────────────────────────────────────────────────
echo "Building Docker image (once)..."
if ! docker build -f "${REPO_DIR}/scripts/Dockerfile.benchmark" -t "${IMAGE_NAME}" "${REPO_DIR}" 2>&1 | tail -5; then
    echo "ERROR: image build failed" >&2
    exit 1
fi
echo ""

# ── Arm → config recipe ──────────────────────────────────────────────────
# An "arm" maps to (compaction, gate_context_reset, auto_revert_ast_cascade).
# Plain compaction names use the production defaults (gate on, auto_revert on).
# Special arms hold compaction fixed and vary one knob, for isolated A/Bs.
arm_settings() {  # echoes: "<compaction> <gate> <auto_revert>"
    case "$1" in
        gate_on)  echo "unified true  true" ;;   # gate A/B: gate ON,  auto_revert ON
        gate_off) echo "unified false true" ;;   # gate A/B: gate OFF, auto_revert ON
        # Plain compaction arm → production defaults: gate OFF (A/B-decided),
        # auto_revert ON. Matches src/config ToolsConfig::default after 2026-06-29.
        *)        echo "$1 false true" ;;
    esac
}

# ── Config generator ────────────────────────────────────────────────────
# All providers on (the 6/6 baseline); knobs come from arm_settings so a single
# driver run can do either a compaction A/B or a single-knob A/B (e.g. the gate).
generate_config() {
    local arm="$1"
    local _compaction _gate _autorev
    read -r _compaction _gate _autorev <<< "$(arm_settings "$arm")"
    cat <<TOML
[model]
provider = "llama-cpp"
endpoint = "${LLAMA_ENDPOINT}"
model = "${MODEL}"
context_window = 60000
temperature = ${TEMPERATURE}
max_output_tokens = 8000

[context]
repo_map_budget = 5000
max_rounds = ${MAX_ROUNDS}
pause_after_rounds = 99999
compaction = "${_compaction}"

# Providers mirror the historical 6/6 gemma baseline (run-benchmark-docker.sh
# enables all of these by default). Earlier I left them at their serde defaults,
# which silently disabled profile/guide/project_notes/lessons and dropped the
# 6/6 baseline to 5/6 — see the culprit diff. Only compaction varies per arm.
[context.providers]
profile = true
guide = true
project_notes = true
plan = true
lessons = true
repo_map = false  # off by default — available on demand via code(action='repo_map')
mcp = true
scratchpad = true
usage_guide = true
plan_mode = true

[hardware]
vram_gb = 24.0
vram_reserve_gb = 3.0
ram_budget_gb = 80.0

[web]
search_backend = "serper"
fetch_backend = "jina"

[lsp]
enabled = true
diagnostic_timeout_ms = 2000

[tools]
web_tools = true
plan = true
scratchpad = true
# Knobs from arm_settings. auto_revert + gate default ON (production defaults
# as of this session); a gate A/B varies gate_context_reset while holding the
# rest constant. reactive_debugger/spiral_reset stay OFF (not default-good).
auto_revert_ast_cascade = ${_autorev}
reactive_debugger = false
spiral_reset = false
gate_context_reset = ${_gate}

[logging]
level = "trace"
enabled = true

[validation]
command = "out=\$(cargo build 2>&1) || { echo \"DOES NOT COMPILE:\"; echo \"\$out\" | tail -20; exit 1; }; run=\$(MINISWE_SKIP_VALIDATION=1 ./target/debug/miniswe --system-prompt-override 'Respond only with TOKEN_XYZ and nothing else' --yes hello 2>&1); echo \"\$run\" | grep -q TOKEN_XYZ || { echo \"COMPILES but override NOT consumed. Expected TOKEN_XYZ, GOT: \$run\"; exit 1; }"
timeout_secs = 180
max_retries = 3
TOML
}

# ── Per-run container script (6-check validation) ───────────────────────
# Verbatim mirror of run-benchmark-docker.sh's validated container_script.
read -r -d '' CONTAINER_SCRIPT <<'SCRIPT' || true
#!/bin/bash
set -uo pipefail
SHA="$1"; TASK="$2"; TIMEOUT="$3"; MAX_ATTEMPTS="$4"

cd /work
git -C /repo archive "${SHA}" | tar -x
rm -rf target .miniswe

if grep -q "git-lfs" .gitignore 2>/dev/null; then
    echo -e "target/\n.miniswe\n*.log" > .gitignore
else
    echo ".miniswe" >> .gitignore
fi

mkdir -p /output/miniswe_state
ln -sfn /output/miniswe_state .miniswe
cp /config/config.toml .miniswe/config.toml
if ! miniswe init 2>/output/miniswe_init.txt; then
    echo "ERROR: miniswe init failed:"; cat /output/miniswe_init.txt; exit 1
fi
mkdir -p .miniswe/logs
git init -q && git add -A && git commit -q -m "baseline" 2>/dev/null

START_TIME=$(date +%s); DEADLINE=$((START_TIME + TIMEOUT)); ATTEMPT=0
CURRENT_TASK="${TASK}"; BEST_PASS=0

while [ "$ATTEMPT" -lt "$MAX_ATTEMPTS" ]; do
    ATTEMPT=$((ATTEMPT + 1)); NOW=$(date +%s); REMAINING=$((DEADLINE - NOW))
    if [ "$REMAINING" -le 30 ]; then echo "=== ATTEMPT ${ATTEMPT}: SKIPPED (${REMAINING}s left) ==="; break; fi
    echo "=== ATTEMPT ${ATTEMPT}/${MAX_ATTEMPTS} (${REMAINING}s remaining) ==="

    timeout "${REMAINING}" miniswe --yes "${CURRENT_TASK}" \
        > /output/stdout_attempt${ATTEMPT}.txt 2> /output/stderr_attempt${ATTEMPT}.txt || true

    git diff --name-only > /output/changed_files.txt 2>/dev/null || true
    git ls-files --others --exclude-standard >> /output/changed_files.txt 2>/dev/null || true
    git diff > /output/diff.patch 2>/dev/null || true
    git diff > /output/diff_after_attempt${ATTEMPT}.patch 2>/dev/null || true

    PASS=0; TOTAL=0; ERRORS=""; BINARY="./target/debug/miniswe"; FLAG=""

    TOTAL=$((TOTAL + 1))
    if RUSTFLAGS="-A warnings" cargo check 2> /output/cargo_check.txt; then echo "compile:PASS"; PASS=$((PASS + 1)); else
        echo "compile:FAIL"
        ERRORS="${ERRORS}
COMPILE FAILED:
$(grep -E '^error(\[|:)|^\s*-->|^\s*\|' /output/cargo_check.txt | head -60)"
    fi

    TOTAL=$((TOTAL + 1))
    if [ "$PASS" -ge 1 ]; then
        if RUSTFLAGS="-A warnings" cargo build 2> /output/cargo_build.txt; then echo "build:PASS"; PASS=$((PASS + 1)); else
            echo "build:FAIL"
            ERRORS="${ERRORS}
BUILD FAILED:
$(grep -E '^error(\[|:)|^\s*-->' /output/cargo_build.txt | head -30)"
        fi
    else echo "build:SKIP"; fi

    TOTAL=$((TOTAL + 1))
    if [ -f "${BINARY}" ]; then
        "${BINARY}" --help > /output/help_output.txt 2>&1 || true
        if grep -qiE -- '--[a-z-]*prompt[a-z-]*' /output/help_output.txt; then
            FLAG=$(grep -oE -- '--[a-z-]*prompt[a-z-]*' /output/help_output.txt | head -1); echo "help:PASS(${FLAG})"; PASS=$((PASS + 1))
        else
            echo "help:FAIL"
            ERRORS="${ERRORS}
HELP FAILED: --help has no flag matching '--*prompt*'. Add a CLI flag whose long name contains 'prompt'. Current --help:
$(head -40 /output/help_output.txt)"
        fi
    fi

    TOTAL=$((TOTAL + 1))
    if [ -f "${BINARY}" ] && [ -n "${FLAG}" ]; then
        if "${BINARY}" ${FLAG} "test" --help > /output/parse_output.txt 2>&1; then echo "parse:PASS"; PASS=$((PASS + 1)); else
            echo "parse:FAIL"
            ERRORS="${ERRORS}
PARSE FAILED: \`${BINARY} ${FLAG} \"test\" --help\` rejected. ${FLAG} must take a single string arg. Output:
$(head -20 /output/parse_output.txt)"
        fi
    fi

    TOTAL=$((TOTAL + 1))
    if [ "$PASS" -ge 2 ]; then
        if RUSTFLAGS="-A warnings" cargo test > /output/cargo_test.txt 2>&1; then echo "test:PASS"; PASS=$((PASS + 1)); else
            echo "test:FAIL"
            ERRORS="${ERRORS}
TESTS FAILED:
$(grep -A5 -E 'panicked at|assertion .*failed|^test .* \.\.\. FAILED$|^error(\[|:)' /output/cargo_test.txt | head -40)"
        fi
    fi

    TOTAL=$((TOTAL + 1))
    if [ -f "${BINARY}" ] && [ -n "${FLAG}" ] && [ "$PASS" -ge 4 ]; then
        SMOKE_OVERRIDE='You must respond with exactly the text PONG_42 and nothing else. No explanation, no formatting, just PONG_42.'
        SMOKE_OUTPUT=$(MINISWE_SKIP_VALIDATION=1 timeout 120 "${BINARY}" ${FLAG} "${SMOKE_OVERRIDE}" --yes "ping" 2>/output/smoke_stderr.txt || true)
        echo "${SMOKE_OUTPUT}" > /output/smoke_output.txt
        if echo "${SMOKE_OUTPUT}" | grep -q "PONG_42"; then echo "smoke:PASS"; PASS=$((PASS + 1)); else
            echo "smoke:FAIL"
            ERRORS="${ERRORS}
SMOKE TEST FAILED. Expected PONG_42 in output of \`${BINARY} ${FLAG} \"...\" --yes ping\`. Got:
$(echo "${SMOKE_OUTPUT}" | head -5)
The override is being silently ignored — the feature is incomplete."
        fi
    fi

    echo "=== ATTEMPT ${ATTEMPT} RESULT: ${PASS}/${TOTAL} ==="
    [ "$PASS" -gt "$BEST_PASS" ] && BEST_PASS="$PASS"
    if [ "$PASS" -eq "$TOTAL" ]; then echo "=== PASSED on attempt ${ATTEMPT} ==="; break; fi
    CURRENT_TASK="Your previous changes have these problems:
${ERRORS}
Please fix the issues. The modified files are still on disk."
done

[ "$BEST_PASS" -gt "$PASS" ] && PASS="$BEST_PASS"
echo "=== FINAL: ${PASS}/${TOTAL} after ${ATTEMPT} attempt(s) ==="
SCRIPT

# ── Run one (strategy, run) cell ────────────────────────────────────────
run_cell() {
    local strat="$1" idx="$2"
    local cell_dir="${RESULTS_DIR}/${strat}/run${idx}"
    local cname="miniswe-compaction-${strat}-${idx}-$$"
    mkdir -p "${cell_dir}"

    # Resume: if this cell's docker run already completed (container.log has a
    # FINAL line), skip the (expensive) re-run and only recompute metrics from
    # the on-disk output. Lets a re-launch into the same --results-dir finish a
    # partial matrix without redoing completed cells.
    if grep -q "=== FINAL:" "${cell_dir}/container.log" 2>/dev/null; then
        echo "  (resume) ${strat}/run${idx}: already complete — recomputing metrics"
    else
        generate_config "${strat}" > "${cell_dir}/config.toml"
        local tmp; tmp=$(mktemp); echo "${CONTAINER_SCRIPT}" > "${tmp}"; chmod +x "${tmp}"
        local t0; t0=$(date +%s)
        docker rm -f "${cname}" >/dev/null 2>&1 || true
        ACTIVE_CONTAINER="${cname}"
        docker run --rm --network=host \
            -v "${cell_dir}:/output" \
            -v "${cell_dir}/config.toml:/config/config.toml:ro" \
            -v "${tmp}:/run.sh:ro" \
            -e MINISWE_LLM_DUMP_DIR=/output/llm_dumps \
            --name "${cname}" "${IMAGE_NAME}" \
            bash /run.sh "${BASELINE_SHA}" "${TASK}" "${TIMEOUT}" "${MAX_ATTEMPTS}" \
            > "${cell_dir}/container.log" 2>&1
        echo $(( $(date +%s) - t0 )) > "${cell_dir}/wall_s.txt"
        rm -f "${tmp}"; ACTIVE_CONTAINER=""
    fi

    # Metrics. NB: `grep -c` prints "0" AND exits 1 on no matches, so `|| echo 0`
    # would append a SECOND "0" and break the arithmetic — capture then default.
    local result rounds compactions elided wall r
    result=$(grep -oE "=== FINAL: [0-9]+/[0-9]+" "${cell_dir}/container.log" 2>/dev/null | tail -1 | grep -oE "[0-9]+/[0-9]+" || echo "?/?")
    rounds=0
    for lf in "${cell_dir}"/miniswe_state/logs/*.log; do
        [ -f "${lf}" ] || continue
        r=$(grep -c '\[round ' "${lf}" 2>/dev/null || true)
        rounds=$((rounds + ${r:-0}))
    done
    compactions=$(cat "${cell_dir}"/stderr_attempt*.txt 2>/dev/null | grep -c '\[compaction\] strategy=' || true)
    compactions=${compactions:-0}
    elided=$(cat "${cell_dir}"/stderr_attempt*.txt 2>/dev/null \
        | grep -oE 'elided_tokens=[0-9]+' | grep -oE '[0-9]+' \
        | awk '{s+=$1} END{print s+0}')
    wall=$(cat "${cell_dir}/wall_s.txt" 2>/dev/null || echo "?")

    printf "%s,%s,%s,%s,%s,%s\n" "${strat}" "${idx}" "${result}" "${rounds}" "${compactions}" "${elided}" \
        >> "${RESULTS_DIR}/raw.csv"
    printf "  %-20s run%s: %-5s rounds=%-3s compactions=%-2s elided_tok=%-6s wall=%ss\n" \
        "${strat}" "${idx}" "${result}" "${rounds}" "${compactions}" "${elided}" "${wall}"
}

echo "strategy,run,result,rounds,compactions,elided_tokens" > "${RESULTS_DIR}/raw.csv"

for strat in ${STRATEGIES}; do
    echo "─── ${strat} ───"
    for i in $(seq 1 "${RUNS}"); do
        run_cell "${strat}" "${i}"
    done
    echo ""
done

# ── Aggregate ───────────────────────────────────────────────────────────
echo "================================================================="
echo "  COMPACTION STRATEGY RESULTS  (${MODEL_TAG})"
echo "================================================================="
python3 - "${RESULTS_DIR}" <<'PY'
import csv, json, sys, statistics as st
rd = sys.argv[1]
rows = list(csv.DictReader(open(f"{rd}/raw.csv")))
def pf(x):
    try: return int(x.split('/')[0])
    except Exception: return None
def num(x):
    try: return float(x)
    except Exception: return 0.0
strats, order = {}, []
for r in rows:
    s = r['strategy']
    if s not in strats:
        strats[s] = {'pass': [], 'rounds': [], 'compactions': [], 'elided': [], 'results': []}
        order.append(s)
    p = pf(r['result'])
    strats[s]['pass'].append(p if p is not None else 0)
    strats[s]['results'].append(r['result'])
    strats[s]['rounds'].append(num(r['rounds']))
    strats[s]['compactions'].append(num(r['compactions']))
    strats[s]['elided'].append(num(r['elided_tokens']))

def mean(xs): return round(st.mean(xs), 1) if xs else 0.0
hdr = f"| {'strategy':<20} | {'runs (PASS/6)':<16} | {'mean':<5} | {'rounds':<6} | {'#compact':<8} | {'elided tok':<10} |"
sep = "|" + "-"*22 + "|" + "-"*18 + "|" + "-"*7 + "|" + "-"*8 + "|" + "-"*10 + "|" + "-"*12 + "|"
lines = [hdr, sep]
summary = {}
for s in order:
    d = strats[s]
    runs_str = " ".join(d['results'])
    lines.append(f"| {s:<20} | {runs_str:<16} | {mean(d['pass']):<5} | {mean(d['rounds']):<6} | {mean(d['compactions']):<8} | {round(mean(d['elided'])):<10} |")
    summary[s] = {
        'results': d['results'],
        'mean_pass': mean(d['pass']),
        'mean_rounds': mean(d['rounds']),
        'mean_compactions': mean(d['compactions']),
        'mean_elided_tokens': round(mean(d['elided'])),
    }
table = "\n".join(lines)
print(table)
open(f"{rd}/summary.md","w").write("# Compaction strategy benchmark\n\n" + table + "\n")
json.dump(summary, open(f"{rd}/summary.json","w"), indent=2)

# Flag any strategy where compaction never fired (null comparison).
for s in order:
    if max(strats[s]['compactions']) == 0:
        print(f"\n⚠️  {s}: compaction NEVER fired in any run — task too short to trigger; comparison is null for this arm.")
PY
echo ""
echo "Detailed results: ${RESULTS_DIR}/"
echo "  summary.md / summary.json / raw.csv"
