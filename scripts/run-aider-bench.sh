#!/usr/bin/env bash
# run-aider-bench.sh — Run the same provider benchmark as
# run-benchmark-docker.sh but with aider as the agent instead of miniswe.
#
# Validation suite is identical (6 checks). Same pinned baseline SHA, same
# Docker-isolated container per variant, same /v1/chat/completions endpoint
# on localhost:8464 (so the same llama-server host process can serve both).
#
# Usage:
#   ./scripts/run-aider-bench.sh [--timeout 1800] [--max-attempts 3] \
#                                [--model devstral-small-2]
#
# Honest about scope: aider uses search/replace patches, not OpenAI tool
# calls. That's a real architectural difference, not just a different
# orchestrator — partly why we run this comparison. See
# docs/agent-comparison-design.md.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE_NAME="miniswe-aider-bench"

LLAMA_ENDPOINT="${LLAMA_ENDPOINT:-http://localhost:8464}"
MODEL_TAG="$(
    curl -fsS --max-time 3 "${LLAMA_ENDPOINT}/v1/models" 2>/dev/null \
    | python3 -c "import json,sys; r=json.load(sys.stdin); print((r.get('data') or [{}])[0].get('id','?'))" 2>/dev/null \
    | sed -E 's/\.gguf$//; s/[^A-Za-z0-9._-]/_/g' \
    | cut -c1-40
)"
MODEL_TAG="${MODEL_TAG:-unknown}"
RESULTS_DIR="${REPO_DIR}/benchmark_results/aider_$(date +%Y%m%d_%H%M%S)_${MODEL_TAG}"
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
    docker image rm -f "${IMAGE_NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

# Defaults
TIMEOUT=1800
MAX_ATTEMPTS=3
MODEL="devstral-small-2"
TASK="Add a CLI flag --system-prompt-override (short: -s) that takes a string and replaces the default system prompt with the provided text. When this flag is set, skip all context providers and just use the override text as the system message. Make sure it works for both single-shot and interactive modes."
EDIT_FORMAT="${AIDER_EDIT_FORMAT:-}"   # empty = aider auto-picks

while [[ $# -gt 0 ]]; do
    case "$1" in
        --timeout)      TIMEOUT="$2";      shift 2 ;;
        --max-attempts) MAX_ATTEMPTS="$2"; shift 2 ;;
        --model)        MODEL="$2";        shift 2 ;;
        --task)         TASK="$2";         shift 2 ;;
        --sha)          BASELINE_SHA="$2"; shift 2 ;;
        --edit-format)  EDIT_FORMAT="$2";  shift 2 ;;
        *) echo "Unknown: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "${RESULTS_DIR}"

echo "=== Docker-isolated aider benchmark ==="
echo "Image:    ${IMAGE_NAME}"
echo "SHA:      ${BASELINE_SHA}"
echo "Model:    ${MODEL}"
echo "Endpoint: ${LLAMA_ENDPOINT}"
echo "Timeout:  ${TIMEOUT}s"
echo "Attempts: ${MAX_ATTEMPTS}"
echo "Edit fmt: ${EDIT_FORMAT:-auto}"
echo "Results:  ${RESULTS_DIR}"
echo "Task:     ${TASK:0:80}..."
echo ""

# Verify LLM server is reachable BEFORE the long docker build
if ! curl -fsS --max-time 5 "${LLAMA_ENDPOINT}/v1/models" > /dev/null 2>&1; then
    echo "ERROR: LLM server not responding at ${LLAMA_ENDPOINT}" >&2
    echo "" >&2
    echo "Start a llama-server first (e.g. ./start-mistral-small-4.sh), then re-run." >&2
    exit 1
fi

echo "Building aider Docker image..."
docker build -f "${REPO_DIR}/scripts/Dockerfile.aider" -t "${IMAGE_NAME}" "${REPO_DIR}" 2>&1 | tail -5
echo ""

run_variant() {
    local name="$1"
    local variant_dir="${RESULTS_DIR}/${name}"
    local container_name="miniswe-aider-${name}-$$"
    mkdir -p "${variant_dir}"
    echo "--- ${name} ---"

    local container_script
    container_script=$(cat <<'SCRIPT'
#!/bin/bash
set -uo pipefail
# Note: no set -e — aider and validation are expected to fail occasionally.

SHA="$1"
TASK="$2"
TIMEOUT="$3"
MAX_ATTEMPTS="$4"
EDIT_FORMAT="$5"
LLAMA_ENDPOINT="$6"

cd /work
git -C /repo archive "${SHA}" | tar -x
rm -rf target

# git init + baseline commit for diff capture
git init -q
git config user.email bench@example.invalid
git config user.name bench
git add -A && git commit -q -m "baseline" 2>/dev/null

START_TIME=$(date +%s)
DEADLINE=$((START_TIME + TIMEOUT))
ATTEMPT=0
CURRENT_TASK="${TASK}"
BEST_PASS=0

# Aider edit-format flag (optional)
EDIT_FORMAT_ARG=()
if [ -n "${EDIT_FORMAT}" ]; then
    EDIT_FORMAT_ARG=(--edit-format "${EDIT_FORMAT}")
fi

# Aider needs an API key env var even for local llama-cpp endpoints.
export OPENAI_API_KEY="sk-local-dummy"
export OPENAI_API_BASE="${LLAMA_ENDPOINT}/v1"

# The model name aider sends in the request body. llama-cpp ignores it (it
# only ever serves one model) so we pick a stable label. The "openai/"
# prefix tells LiteLLM (aider's internal LLM client) to use the
# OpenAI-compatible HTTP transport.
AIDER_MODEL="openai/local"

while [ "$ATTEMPT" -lt "$MAX_ATTEMPTS" ]; do
    ATTEMPT=$((ATTEMPT + 1))
    NOW=$(date +%s)
    REMAINING=$((DEADLINE - NOW))
    if [ "$REMAINING" -le 30 ]; then
        echo "=== ATTEMPT ${ATTEMPT}: SKIPPED (${REMAINING}s left) ==="
        break
    fi

    echo "=== ATTEMPT ${ATTEMPT}/${MAX_ATTEMPTS} (${REMAINING}s remaining) ==="

    # Aider in one-shot mode. --yes-always auto-confirms file additions/edits.
    # --no-stream because the harness captures stdout into a file.
    # --no-auto-commits because the harness does its own git tracking and
    # we don't want aider's commits polluting the diff.
    # --no-gitignore prevents aider from rewriting .gitignore on every run.
    timeout "${REMAINING}" aider \
        --model "${AIDER_MODEL}" \
        --yes-always \
        --no-stream \
        --no-auto-commits \
        --no-gitignore \
        --no-show-model-warnings \
        --no-pretty \
        --message "${CURRENT_TASK}" \
        "${EDIT_FORMAT_ARG[@]}" \
        > /output/stdout_attempt${ATTEMPT}.txt \
        2> /output/stderr_attempt${ATTEMPT}.txt \
        || true

    # Capture diff state
    git add -A
    git diff --cached --name-only > /output/changed_files.txt 2>/dev/null || true
    git diff --cached > /output/diff.patch 2>/dev/null || true
    git diff --cached > /output/diff_after_attempt${ATTEMPT}.patch 2>/dev/null || true

    # === Validate (same 6 checks as run-benchmark-docker.sh) ===
    PASS=0
    TOTAL=0
    ERRORS=""

    # 1. cargo check
    TOTAL=$((TOTAL + 1))
    if RUSTFLAGS="-A warnings" cargo check 2> /output/cargo_check.txt; then
        echo "compile:PASS"; PASS=$((PASS + 1))
    else
        echo "compile:FAIL"
        ERRORS="${ERRORS}
COMPILE FAILED:
$(grep -E '^error(\[|:)|^\s*-->|^\s*\|' /output/cargo_check.txt | head -60)"
    fi

    # 2. cargo build
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

    # 3. --help shows prompt override flag
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

    # 4. flag parses
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

    # 5. cargo test
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

    # 6. smoke test
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

    CURRENT_TASK="Your previous changes have these problems:
${ERRORS}
Please fix the issues. The modified files are still on disk."
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
        --name "${container_name}" \
        "${IMAGE_NAME}" \
        bash /run.sh "${BASELINE_SHA}" "${TASK}" "${TIMEOUT}" "${MAX_ATTEMPTS}" "${EDIT_FORMAT}" "${LLAMA_ENDPOINT}" \
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
}

echo "═══ aider baseline ═══"
echo ""
run_variant "00_baseline"

echo ""
echo "================================================================="
echo "  AIDER RESULTS"
echo "================================================================="
printf "%-20s %4s %8s %s\n" "Variant" "Att" "Time" "Result"
echo "-----------------------------------------------------------------"
for d in "${RESULTS_DIR}"/*/; do
    name=$(basename "$d")
    wall=$(cat "$d/wall_s.txt" 2>/dev/null || echo "?")
    attempts=$(grep -c "=== ATTEMPT .* remaining" "$d/container.log" 2>/dev/null || echo "?")
    result=$(grep "=== FINAL:" "$d/container.log" 2>/dev/null | grep -oE "[0-9]+/[0-9]+" || echo "?/?")
    printf "%-20s %4s %7ss  %s\n" "$name" "$attempts" "$wall" "$result"
done
echo "================================================================="
echo ""
echo "Detailed results: ${RESULTS_DIR}/"
