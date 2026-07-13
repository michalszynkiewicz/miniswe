#!/usr/bin/env bash
# run-opencode-scaffolded-bench.sh — opencode with its own Plan→Build workflow.
#
# Identical to run-opencode-bench.sh (same task, same baseline SHA, same
# Docker image, same 6-check validation) except for HOW opencode is
# invoked per attempt: instead of a single `opencode run` straight to the
# Build agent (opencode's own headless default — no forced planning step),
# this variant explicitly scaffolds two chained calls:
#
#   1. `opencode run ... --agent plan`   — read-only: analyze the codebase,
#      write a concrete step-by-step plan, make no edits (Plan agent can't).
#   2. `opencode run ... --agent build --session <id>` — continues the SAME
#      session (so Build sees the plan from step 1 in context) and does the
#      actual implementation.
#
# The session ID is captured from step 1's `--format json` event stream
# (each event carries a `sessionID` field) and passed to step 2 via
# `--session`. Verified working: a session started under --agent plan
# correctly retains context when continued under --agent build in a
# SEPARATE `opencode run` invocation (tested with a canary phrase before
# writing this script).
#
# This is the closest opencode-native analogue to miniswe's forced
# plan-first ceremony — NOT identical (opencode's Plan agent is a read-only
# analysis pass, not a persistent, checkable plan.md with compile gates),
# but it's the mechanism opencode itself ships for "plan before you build."
# Compare against run-opencode-bench.sh (opencode's own un-scaffolded
# default) to see whether this scaffolding is worth what it costs (2 LLM
# round-trips per attempt instead of 1).
#
# Usage:
#   ./scripts/run-opencode-scaffolded-bench.sh [--timeout 1800] [--max-attempts 3]

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE_NAME="miniswe-opencode-bench"

LLAMA_ENDPOINT="${LLAMA_ENDPOINT:-http://localhost:8464}"

MODEL_ID="$(
    curl -fsS --max-time 3 "${LLAMA_ENDPOINT}/v1/models" 2>/dev/null \
    | python3 -c "import json,sys; r=json.load(sys.stdin); print((r.get('data') or [{}])[0].get('id','?'))" 2>/dev/null
)"
MODEL_TAG="$(
    echo "${MODEL_ID}" | sed -E 's/\.gguf$//; s/[^A-Za-z0-9._-]/_/g' | cut -c1-40
)"
MODEL_TAG="${MODEL_TAG:-unknown}"
RESULTS_DIR="${REPO_DIR}/benchmark_results/opencode_scaffolded_$(date +%Y%m%d_%H%M%S)_${MODEL_TAG}"
BASELINE_SHA="cc34d2626faf32c1b6dd1b8b33af693fb936b098"
ACTIVE_CONTAINER_NAME=""
ACTIVE_TMP_SCRIPT=""

cleanup() {
    set +e
    if [[ -n "${ACTIVE_CONTAINER_NAME}" ]]; then
        docker rm -f "${ACTIVE_CONTAINER_NAME}" >/dev/null 2>&1 || true
    fi
    if [[ -n "${ACTIVE_TMP_SCRIPT}" ]]; then
        rm -f "${ACTIVE_TMP_SCRIPT}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT INT TERM

# Defaults
TIMEOUT=1800
MAX_ATTEMPTS=3
TASK="Add a CLI flag --system-prompt-override (short: -s) that takes a string and replaces the default system prompt with the provided text. When this flag is set, skip all context providers and just use the override text as the system message. Make sure it works for both single-shot and interactive modes."
RUNS=3

while [[ $# -gt 0 ]]; do
    case "$1" in
        --timeout)      TIMEOUT="$2";      shift 2 ;;
        --max-attempts) MAX_ATTEMPTS="$2"; shift 2 ;;
        --task)         TASK="$2";         shift 2 ;;
        --sha)          BASELINE_SHA="$2"; shift 2 ;;
        --runs)         RUNS="$2";         shift 2 ;;
        *) echo "Unknown: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "${RESULTS_DIR}"

echo "=== Docker-isolated opencode (scaffolded plan→build) benchmark ==="
echo "Image:    ${IMAGE_NAME}"
echo "SHA:      ${BASELINE_SHA}"
echo "Model ID: ${MODEL_ID}"
echo "Endpoint: ${LLAMA_ENDPOINT}"
echo "Timeout:  ${TIMEOUT}s"
echo "Attempts: ${MAX_ATTEMPTS}"
echo "Runs:     ${RUNS}"
echo "Results:  ${RESULTS_DIR}"
echo "Task:     ${TASK:0:80}..."
echo ""

if ! curl -fsS --max-time 5 "${LLAMA_ENDPOINT}/v1/models" > /dev/null 2>&1; then
    echo "ERROR: LLM server not responding at ${LLAMA_ENDPOINT}" >&2
    echo "" >&2
    echo "Start a llama-server first (e.g. ./start-gemma4.sh), then re-run." >&2
    exit 1
fi
if [[ -z "${MODEL_ID}" || "${MODEL_ID}" == "?" ]]; then
    echo "ERROR: could not discover model ID from ${LLAMA_ENDPOINT}/v1/models" >&2
    exit 1
fi

echo "Building opencode Docker image (shared with run-opencode-bench.sh)..."
docker build -f "${REPO_DIR}/scripts/Dockerfile.opencode" -t "${IMAGE_NAME}" "${REPO_DIR}" 2>&1 | tail -5
echo ""

generate_opencode_config() {
    cat <<JSON
{
  "\$schema": "https://opencode.ai/config.json",
  "provider": {
    "localllama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Local llama.cpp",
      "options": { "baseURL": "${LLAMA_ENDPOINT}/v1" },
      "models": { "${MODEL_ID}": { "name": "gemma-local" } }
    }
  },
  "model": "localllama/${MODEL_ID}",
  "autoupdate": false,
  "permission": { "*": "allow" }
}
JSON
}

generate_opencode_auth() {
    cat <<JSON
{
  "localllama": { "type": "api", "key": "sk-local-dummy" }
}
JSON
}

run_variant() {
    local name="$1"
    local variant_dir="${RESULTS_DIR}/${name}"
    local container_name="miniswe-opencode-scaffolded-${name}-$$"
    mkdir -p "${variant_dir}"
    echo "--- ${name} ---"

    generate_opencode_config > "${variant_dir}/opencode.json"
    generate_opencode_auth > "${variant_dir}/auth.json"

    local container_script
    container_script=$(cat <<'SCRIPT'
#!/bin/bash
set -uo pipefail
# Note: no set -e — opencode and validation are expected to fail occasionally.

SHA="$1"
TASK="$2"
TIMEOUT="$3"
MAX_ATTEMPTS="$4"
MODEL_REF="$5"

cd /work
git -C /repo archive "${SHA}" | tar -x
rm -rf target

git init -q
git config user.email bench@example.invalid
git config user.name bench
git add -A && git commit -q -m "baseline" 2>/dev/null

# Extract the sessionID from opencode's --format json event stream. Every
# event line carries `sessionID`; grab the first one we see.
extract_session_id() {
    python3 -c "
import json, sys
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
    except Exception:
        continue
    sid = d.get('sessionID')
    if sid:
        print(sid)
        break
"
}

START_TIME=$(date +%s)
DEADLINE=$((START_TIME + TIMEOUT))
ATTEMPT=0
CURRENT_TASK="${TASK}"
BEST_PASS=0

while [ "$ATTEMPT" -lt "$MAX_ATTEMPTS" ]; do
    ATTEMPT=$((ATTEMPT + 1))
    NOW=$(date +%s)
    REMAINING=$((DEADLINE - NOW))
    if [ "$REMAINING" -le 60 ]; then
        echo "=== ATTEMPT ${ATTEMPT}: SKIPPED (${REMAINING}s left) ==="
        break
    fi

    echo "=== ATTEMPT ${ATTEMPT}/${MAX_ATTEMPTS} (${REMAINING}s remaining) ==="

    # Step 1: Plan agent — read-only, analyzes the repo and writes a
    # concrete implementation plan. Half the remaining budget, floor 120s,
    # so a slow plan pass can't starve the build step that has to follow it.
    PLAN_BUDGET=$((REMAINING / 2))
    [ "$PLAN_BUDGET" -lt 120 ] && PLAN_BUDGET=120
    PLAN_PROMPT="Before making any changes, analyze this codebase and write a \
concise, step-by-step implementation plan for the following task. Name the \
specific files you will need to touch and what each change is. Do not make \
any edits yet — this is a planning pass only.

Task: ${CURRENT_TASK}"

    echo "  [plan] spawning plan agent (budget ${PLAN_BUDGET}s)..."
    SESSION_ID=$(timeout "${PLAN_BUDGET}" opencode run "${PLAN_PROMPT}" \
        --model "${MODEL_REF}" \
        --dir /work \
        --agent plan \
        --auto \
        --format json \
        --print-logs \
        2> /output/stderr_attempt${ATTEMPT}_plan.txt \
        | tee /output/stdout_attempt${ATTEMPT}_plan.txt \
        | extract_session_id)

    if [ -z "${SESSION_ID}" ]; then
        echo "  [plan] FAILED to get a session — skipping straight to build with no plan context"
        BUILD_PROMPT="${CURRENT_TASK}"
        SESSION_ARGS=()
    else
        echo "  [plan] session=${SESSION_ID}"
        BUILD_PROMPT="Now implement the plan you just wrote, step by step. Make all \
the necessary code changes to fully complete the task."
        SESSION_ARGS=(--session "${SESSION_ID}")
    fi

    NOW=$(date +%s)
    REMAINING=$((DEADLINE - NOW))
    if [ "$REMAINING" -le 30 ]; then
        echo "=== ATTEMPT ${ATTEMPT}: build step SKIPPED (${REMAINING}s left) ==="
        break
    fi

    # Step 2: Build agent, continuing the plan session (if we have one) —
    # full tool access, does the actual implementation.
    echo "  [build] spawning build agent (budget ${REMAINING}s)..."
    timeout "${REMAINING}" opencode run "${BUILD_PROMPT}" \
        --model "${MODEL_REF}" \
        --dir /work \
        --agent build \
        --auto \
        --print-logs \
        "${SESSION_ARGS[@]}" \
        > /output/stdout_attempt${ATTEMPT}.txt \
        2> /output/stderr_attempt${ATTEMPT}.txt \
        || true

    # Capture diff state
    git add -A
    git diff --cached --name-only > /output/changed_files.txt 2>/dev/null || true
    git diff --cached > /output/diff.patch 2>/dev/null || true
    git diff --cached > /output/diff_after_attempt${ATTEMPT}.patch 2>/dev/null || true

    # === Validate (same 6 checks as run-benchmark-docker.sh / run-opencode-bench.sh) ===
    PASS=0
    TOTAL=0
    ERRORS=""

    TOTAL=$((TOTAL + 1))
    if RUSTFLAGS="-A warnings" cargo check 2> /output/cargo_check.txt; then
        echo "compile:PASS"; PASS=$((PASS + 1))
    else
        echo "compile:FAIL"
        ERRORS="${ERRORS}
COMPILE FAILED:
$(grep -E '^error(\[|:)|^\s*-->|^\s*\|' /output/cargo_check.txt | head -60)"
    fi

    TOTAL=$((TOTAL + 1))
    BINARY="./target/debug/miniswe"
    if [ "$PASS" -ge 1 ]; then
        if RUSTFLAGS="-A warnings" cargo build 2> /output/cargo_build.txt; then
            echo "build:PASS"; PASS=$((PASS + 1))
        else
            echo "build:FAIL"
            ERRORS="${ERRORS}
BUILD FAILED:
$(grep -E '^error(\[|:)|^\s*-->' /output/cargo_build.txt | head -30)"
        fi
    else
        echo "build:SKIP"
    fi

    TOTAL=$((TOTAL + 1))
    FLAG=""
    if [ -f "${BINARY}" ]; then
        "${BINARY}" --help > /output/help_output.txt 2>&1 || true
        if grep -qiE -- '--[a-z-]*prompt[a-z-]*' /output/help_output.txt; then
            FLAG=$(grep -oE -- '--[a-z-]*prompt[a-z-]*' /output/help_output.txt | head -1)
            echo "help:PASS(${FLAG})"; PASS=$((PASS + 1))
        else
            echo "help:FAIL"
            ERRORS="${ERRORS}
HELP FAILED: --help does not contain any '--*prompt*' flag.
$(head -40 /output/help_output.txt)"
        fi
    fi

    TOTAL=$((TOTAL + 1))
    if [ -f "${BINARY}" ] && [ -n "${FLAG}" ]; then
        if "${BINARY}" ${FLAG} "test" --help > /output/parse_output.txt 2>&1; then
            echo "parse:PASS"; PASS=$((PASS + 1))
        else
            echo "parse:FAIL"
            ERRORS="${ERRORS}
PARSE FAILED for ${FLAG}:
$(head -20 /output/parse_output.txt)"
        fi
    fi

    TOTAL=$((TOTAL + 1))
    if [ "$PASS" -ge 2 ]; then
        if RUSTFLAGS="-A warnings" cargo test > /output/cargo_test.txt 2>&1; then
            echo "test:PASS"; PASS=$((PASS + 1))
        else
            echo "test:FAIL"
            ERRORS="${ERRORS}
TESTS FAILED:
$(grep -E '^error(\[|:)|^\s*-->|^test .* \.\.\. FAILED$|panicked at' /output/cargo_test.txt | head -30)"
        fi
    fi

    TOTAL=$((TOTAL + 1))
    if [ -f "${BINARY}" ] && [ -n "${FLAG}" ] && [ "$PASS" -ge 4 ]; then
        SMOKE_OVERRIDE='You must respond with exactly the text PONG_42 and nothing else.'
        SMOKE_OUTPUT=$(timeout 120 "${BINARY}" ${FLAG} "${SMOKE_OVERRIDE}" --yes "ping" 2>/output/smoke_stderr.txt || true)
        echo "${SMOKE_OUTPUT}" > /output/smoke_output.txt
        if echo "${SMOKE_OUTPUT}" | grep -q "PONG_42"; then
            echo "smoke:PASS"; PASS=$((PASS + 1))
        else
            echo "smoke:FAIL"
            ERRORS="${ERRORS}
SMOKE FAILED: expected PONG_42 in stdout but got: $(echo "${SMOKE_OUTPUT}" | head -3)"
        fi
    fi

    echo "=== ATTEMPT ${ATTEMPT} RESULT: ${PASS}/${TOTAL} ==="
    if [ "$PASS" -gt "$BEST_PASS" ]; then BEST_PASS="$PASS"; fi
    if [ "$PASS" -eq "$TOTAL" ]; then
        echo "=== PASSED on attempt ${ATTEMPT} ==="
        break
    fi

    # Next attempt starts a FRESH plan/build session pair (matching how
    # miniswe/aider retries restart their own process fresh too) — but the
    # plan prompt for the retry includes what broke, so the new plan can
    # actually address it instead of repeating the same one blind.
    CURRENT_TASK="${TASK}

Your previous attempt had these problems, which still need fixing:
${ERRORS}
The modified files are still on disk — plan around what's already there,
don't start over from scratch."
done

if [ "$BEST_PASS" -gt "$PASS" ]; then PASS="$BEST_PASS"; fi
echo "=== FINAL: ${PASS}/${TOTAL} after ${ATTEMPT} attempt(s) ==="
SCRIPT
)

    local tmp_script
    tmp_script=$(mktemp)
    echo "${container_script}" > "${tmp_script}"
    chmod +x "${tmp_script}"
    ACTIVE_TMP_SCRIPT="${tmp_script}"

    local start_time
    start_time=$(date +%s)

    docker rm -f "${container_name}" 2>/dev/null || true
    ACTIVE_CONTAINER_NAME="${container_name}"

    docker run --rm \
        --network=host \
        -v "${variant_dir}:/output" \
        -v "${tmp_script}:/run.sh:ro" \
        -v "${variant_dir}/opencode.json:/root/.config/opencode/opencode.json:ro" \
        -v "${variant_dir}/auth.json:/root/.local/share/opencode/auth.json:ro" \
        --name "${container_name}" \
        "${IMAGE_NAME}" \
        bash /run.sh "${BASELINE_SHA}" "${TASK}" "${TIMEOUT}" "${MAX_ATTEMPTS}" "localllama/${MODEL_ID}" \
        2>&1 | tee "${variant_dir}/container.log"

    local end_time
    end_time=$(date +%s)
    echo $((end_time - start_time)) > "${variant_dir}/wall_s.txt"

    rm -f "${tmp_script}"
    ACTIVE_TMP_SCRIPT=""
    ACTIVE_CONTAINER_NAME=""

    local final_line
    final_line=$(grep "=== FINAL:" "${variant_dir}/container.log" 2>/dev/null || echo "FINAL: ?/? after ? attempt(s)")
    local wall_s
    wall_s=$(cat "${variant_dir}/wall_s.txt")
    local attempts
    attempts=$(grep -c "=== ATTEMPT .* remaining" "${variant_dir}/container.log" 2>/dev/null || echo "0")

    echo ""
    echo "    ${final_line}"
    echo "    attempts=${attempts} wall=${wall_s}s"
    grep -E "(compile|build|help|parse|test|smoke):(PASS|FAIL)" "${variant_dir}/container.log" | tail -6 | sed 's/^/    /'
    echo ""

    printf "%s,%s,%s\n" "${name}" "${wall_s}" "${final_line#*: }" >> "${RESULTS_DIR}/raw.csv"
}

echo "run,wall_s,result" > "${RESULTS_DIR}/raw.csv"

echo "═══ opencode scaffolded (plan→build) baseline ═══"
echo ""
for i in $(seq 1 "${RUNS}"); do
    run_variant "run${i}"
done

echo ""
echo "================================================================="
echo "  OPENCODE SCAFFOLDED (PLAN→BUILD) RESULTS"
echo "================================================================="
printf "%-10s %8s %s\n" "Run" "Time" "Result"
echo "-----------------------------------------------------------------"
for d in "${RESULTS_DIR}"/run*/; do
    name=$(basename "$d")
    wall=$(cat "$d/wall_s.txt" 2>/dev/null || echo "?")
    result=$(grep "=== FINAL:" "$d/container.log" 2>/dev/null | grep -oE "[0-9]+/[0-9]+" || echo "?/?")
    printf "%-10s %7ss  %s\n" "$name" "$wall" "$result"
done
echo "================================================================="
echo ""
echo "Detailed results: ${RESULTS_DIR}/"
echo "Compare against run-opencode-bench.sh's un-scaffolded results."
