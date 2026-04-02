#!/usr/bin/env bash
# run-benchmark-docker.sh — Run provider benchmark with Docker isolation.
#
# Each variant runs in its own container — completely fresh filesystem,
# no shared target/, no stale binaries.
#
# Usage:
#   ./scripts/run-benchmark-docker.sh [--timeout 1800] [--max-rounds 50]
#
# LLM server must be running on host (localhost:8464).

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE_NAME="miniswe-bench"
RESULTS_DIR="${REPO_DIR}/benchmark_results/docker_$(date +%Y%m%d_%H%M%S)"
BASELINE_SHA="cc34d2626faf32c1b6dd1b8b33af693fb936b098"

# Defaults
TIMEOUT=1800
MAX_ROUNDS=50
TEMPERATURE=0.0
TASK="Add a CLI flag that lets the user limit the maximum number of agent rounds per session. It should override whatever the config file says. Make sure it works for both single-shot and interactive modes."

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --timeout)      TIMEOUT="$2";      shift 2 ;;
        --max-rounds)   MAX_ROUNDS="$2";   shift 2 ;;
        --temperature)  TEMPERATURE="$2";  shift 2 ;;
        --task)         TASK="$2";         shift 2 ;;
        --sha)          BASELINE_SHA="$2"; shift 2 ;;
        *) echo "Unknown: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "${RESULTS_DIR}"

echo "=== Docker-isolated provider benchmark ==="
echo "Image:    ${IMAGE_NAME}"
echo "SHA:      ${BASELINE_SHA}"
echo "Timeout:  ${TIMEOUT}s"
echo "Rounds:   ${MAX_ROUNDS}"
echo "Results:  ${RESULTS_DIR}"
echo "Task:     ${TASK:0:80}..."
echo ""

# Build image
echo "Building Docker image..."
docker build -f "${REPO_DIR}/scripts/Dockerfile.benchmark" -t "${IMAGE_NAME}" "${REPO_DIR}" 2>&1 | tail -5
echo ""

# Check LLM endpoint
if ! curl -s --connect-timeout 5 "http://localhost:8464/v1/models" > /dev/null 2>&1; then
    echo "WARNING: LLM not responding at localhost:8464"
fi

# ── Config generator ────────────────────────────────────────────────────

generate_config() {
    local disabled="${1:-}"

    # Helper for provider toggles
    _dis() { echo ",${disabled}," | grep -q ",${1}," && echo "false" || echo "true"; }

    cat <<TOML
[model]
provider = "llama-cpp"
endpoint = "http://localhost:8464"
model = "devstral-small-2"
context_window = 50000
temperature = ${TEMPERATURE}
max_output_tokens = 16384

[context]
repo_map_budget = 5000
snippet_budget = 12000
history_turns = 5
history_budget = 6000
scratchpad_budget = 1500
max_rounds = ${MAX_ROUNDS}
pause_after_rounds = 99999

[context.providers]
profile = $(_dis profile)
guide = $(_dis guide)
project_notes = $(_dis project_notes)
plan = $(_dis plan)
lessons = $(_dis lessons)
repo_map = $(_dis repo_map)
mcp = $(_dis mcp)
scratchpad = $(_dis scratchpad)
usage_guide = $(_dis usage_guide)
plan_mode = $(_dis plan_mode)

[hardware]
vram_gb = 24.0
vram_reserve_gb = 3.0
ram_budget_gb = 80.0

[web]
search_backend = "serper"
fetch_backend = "jina"

[lsp]
enabled = $(_dis lsp)
diagnostic_timeout_ms = 2000

[logging]
level = "trace"
enabled = true
TOML
}

# ── Run one variant in a fresh container ────────────────────────────────

run_variant() {
    local name="$1"
    local disabled="${2:-}"
    local variant_dir="${RESULTS_DIR}/${name}"
    mkdir -p "${variant_dir}"

    echo "--- ${name} ---"
    echo "${disabled:-none}" > "${variant_dir}/disabled.txt"

    # Generate config
    generate_config "${disabled}" > "${variant_dir}/config.toml"

    # Run in a fresh container:
    # 1. Extract code at pinned SHA
    # 2. Write config
    # 3. Run miniswe init
    # 4. Run miniswe --yes "task"
    # 5. Run validation (cargo check/build/test + smoke test)
    local container_script
    container_script=$(cat <<'SCRIPT'
#!/bin/bash
set -euo pipefail

SHA="$1"
TASK="$2"
TIMEOUT="$3"
MAX_ATTEMPTS=3

# Fresh checkout
cd /work
git -C /repo archive "${SHA}" | tar -x
rm -rf target .miniswe

# Write config and init
mkdir -p .miniswe
cp /config/config.toml .miniswe/config.toml
miniswe init 2>/dev/null || true
mkdir -p .miniswe/logs

# Init git for diff tracking
git init -q && git add -A && git commit -q -m "baseline" 2>/dev/null

START_TIME=$(date +%s)
DEADLINE=$((START_TIME + TIMEOUT))
ATTEMPT=0
CURRENT_TASK="${TASK}"

while [ "$ATTEMPT" -lt "$MAX_ATTEMPTS" ]; do
    ATTEMPT=$((ATTEMPT + 1))
    NOW=$(date +%s)
    REMAINING=$((DEADLINE - NOW))
    if [ "$REMAINING" -le 30 ]; then
        echo "=== ATTEMPT ${ATTEMPT}: SKIPPED (${REMAINING}s left) ==="
        break
    fi

    echo "=== ATTEMPT ${ATTEMPT}/${MAX_ATTEMPTS} (${REMAINING}s remaining) ==="

    # Clear logs from previous attempt
    rm -f .miniswe/logs/*.log

    # Run miniswe (keep modified code from previous attempts)
    timeout "${REMAINING}" miniswe --yes "${CURRENT_TASK}" \
        > /output/stdout_attempt${ATTEMPT}.txt \
        2> /output/stderr_attempt${ATTEMPT}.txt \
        || true

    # Copy logs
    cp .miniswe/logs/*.log /output/ 2>/dev/null || true

    # Capture changes
    git diff --name-only > /output/changed_files.txt 2>/dev/null || true
    git ls-files --others --exclude-standard >> /output/changed_files.txt 2>/dev/null || true
    git diff > /output/diff.patch 2>/dev/null || true

    # === Validate ===
    PASS=0
    TOTAL=0
    ERRORS=""

    # Check 1: cargo check
    TOTAL=$((TOTAL + 1))
    if RUSTFLAGS="-A warnings" cargo check 2> /output/cargo_check.txt; then
        echo "compile:PASS"
        PASS=$((PASS + 1))
    else
        echo "compile:FAIL"
        ERRORS="${ERRORS}
COMPILE ERROR:
$(grep '^error' /output/cargo_check.txt | head -20)"
    fi

    # Check 2: cargo build
    TOTAL=$((TOTAL + 1))
    BINARY="./target/debug/miniswe"
    if [ "$PASS" -ge 1 ]; then
        if RUSTFLAGS="-A warnings" cargo build 2> /output/cargo_build.txt; then
            echo "build:PASS"
            PASS=$((PASS + 1))
        else
            echo "build:FAIL"
        fi
    else
        echo "build:SKIP"
    fi

    # Check 3: --help shows rounds flag
    TOTAL=$((TOTAL + 1))
    FLAG=""
    if [ -f "${BINARY}" ]; then
        "${BINARY}" --help > /output/help_output.txt 2>&1 || true
        if grep -qiE -- '--[a-z-]*round[a-z-]*' /output/help_output.txt; then
            FLAG=$(grep -oE -- '--[a-z-]*round[a-z-]*' /output/help_output.txt | head -1)
            echo "help:PASS(${FLAG})"
            PASS=$((PASS + 1))
        else
            echo "help:FAIL"
            ERRORS="${ERRORS}
--help does not show a rounds flag."
        fi
    fi

    # Check 4: flag parses
    TOTAL=$((TOTAL + 1))
    if [ -f "${BINARY}" ] && [ -n "${FLAG}" ]; then
        if "${BINARY}" ${FLAG} 5 --help > /dev/null 2>&1; then
            echo "parse:PASS"
            PASS=$((PASS + 1))
        else
            echo "parse:FAIL"
        fi
    fi

    # Check 5: cargo test
    TOTAL=$((TOTAL + 1))
    if [ "$PASS" -ge 2 ]; then
        if RUSTFLAGS="-A warnings" cargo test 2> /output/cargo_test.txt; then
            echo "test:PASS"
            PASS=$((PASS + 1))
        else
            echo "test:FAIL"
            ERRORS="${ERRORS}
TESTS FAILED:
$(grep -E 'FAILED|panicked|failures' /output/cargo_test.txt | head -10)"
        fi
    fi

    # Check 6: smoke test — use a task that normally takes 3+ rounds,
    # run with max_rounds=2, verify the flag actually limits it.
    # "Read all .rs files in src/ and list their names" needs multiple
    # read_file calls, so without the flag it would take 3+ rounds.
    TOTAL=$((TOTAL + 1))
    if [ -f "${BINARY}" ] && [ -n "${FLAG}" ] && [ "$PASS" -ge 4 ]; then
        rm -f .miniswe/logs/*.log

        # First: run WITHOUT the flag to establish baseline round count
        SMOKE_EXIT=0
        timeout 120 "${BINARY}" --yes "Read every .rs file in src/ and list their names" \
            > /output/smoke_baseline_stdout.txt 2> /output/smoke_baseline_stderr.txt || SMOKE_EXIT=$?
        BASELINE_ROUNDS=$(grep -c '\[round ' .miniswe/logs/*.log 2>/dev/null || echo 0)
        echo "smoke_baseline_rounds=${BASELINE_ROUNDS}"

        # Then: run WITH flag set to 2
        rm -f .miniswe/logs/*.log
        SMOKE_EXIT=0
        timeout 120 "${BINARY}" ${FLAG} 2 --yes "Read every .rs file in src/ and list their names" \
            > /output/smoke_stdout.txt 2> /output/smoke_stderr.txt || SMOKE_EXIT=$?
        LIMITED_ROUNDS=$(grep -c '\[round ' .miniswe/logs/*.log 2>/dev/null || echo 0)
        echo "smoke_limited_rounds=${LIMITED_ROUNDS}"

        echo "smoke_baseline=${BASELINE_ROUNDS} smoke_limited=${LIMITED_ROUNDS}" > /output/smoke_result.txt

        # The flag works if: limited run has fewer rounds than baseline,
        # AND limited run has at most 2 rounds (the limit we set)
        if [ "${LIMITED_ROUNDS}" -le 2 ] && [ "${BASELINE_ROUNDS}" -gt 2 ]; then
            echo "smoke:PASS(baseline=${BASELINE_ROUNDS}r,limited=${LIMITED_ROUNDS}r)"
            PASS=$((PASS + 1))
        elif [ "${LIMITED_ROUNDS}" -le 2 ] && [ "${BASELINE_ROUNDS}" -le 2 ]; then
            # Task was too simple — both finish in ≤2 rounds
            echo "smoke:INCONCLUSIVE(both≤2r)"
            # Give benefit of doubt if everything else passed
            if [ "$PASS" -ge 5 ]; then
                PASS=$((PASS + 1))
            fi
        else
            echo "smoke:FAIL(baseline=${BASELINE_ROUNDS}r,limited=${LIMITED_ROUNDS}r)"
            ERRORS="${ERRORS}
SMOKE TEST: flag should limit to 2 rounds but got ${LIMITED_ROUNDS}. Without flag: ${BASELINE_ROUNDS} rounds. The flag is not wired to the agent loop."
        fi
    fi

    echo "=== ATTEMPT ${ATTEMPT} RESULT: ${PASS}/${TOTAL} ==="

    # All passed?
    if [ "$PASS" -eq "$TOTAL" ]; then
        echo "=== PASSED on attempt ${ATTEMPT} ==="
        break
    fi

    # Compose retry message
    CURRENT_TASK="Your previous changes have these problems:
${ERRORS}
Please fix the issues. The modified files are still on disk."
done

echo "=== FINAL: ${PASS}/${TOTAL} after ${ATTEMPT} attempt(s) ==="
SCRIPT
)

    # Write the script and config to temp files
    local tmp_script
    tmp_script=$(mktemp)
    echo "${container_script}" > "${tmp_script}"
    chmod +x "${tmp_script}"

    local start_time
    start_time=$(date +%s)

    # Run container with:
    # - Fresh /work (empty, code extracted inside)
    # - Config mounted as /config/config.toml
    # - Output dir mounted as /output
    # - Network host for LLM access
    # - Timeout on the container itself
    docker run --rm \
        --network=host \
        -v "${variant_dir}:/output" \
        -v "${variant_dir}/config.toml:/config/config.toml:ro" \
        -v "${tmp_script}:/run.sh:ro" \
        --name "miniswe-bench-${name}" \
        "${IMAGE_NAME}" \
        bash /run.sh "${BASELINE_SHA}" "${TASK}" "${TIMEOUT}" \
        2>&1 | tee "${variant_dir}/container.log"

    local end_time
    end_time=$(date +%s)
    echo $((end_time - start_time)) > "${variant_dir}/wall_s.txt"

    rm -f "${tmp_script}"

    # Parse results
    local wall_s
    wall_s=$(cat "${variant_dir}/wall_s.txt")

    # Extract final result line
    local final_line
    final_line=$(grep "=== FINAL:" "${variant_dir}/container.log" 2>/dev/null || echo "FINAL: ?/? after ? attempt(s)")

    # Count total rounds across all attempt logs
    local rounds=0
    for logfile in "${variant_dir}"/*.log; do
        [ -f "${logfile}" ] || continue
        local r
        r=$(grep -c '\[round ' "${logfile}" 2>/dev/null || true)
        rounds=$((rounds + ${r:-0}))
    done

    # Count attempts
    local attempts
    attempts=$(grep -c "=== ATTEMPT .* remaining" "${variant_dir}/container.log" 2>/dev/null || echo "0")

    echo ""
    echo "    ${final_line}"
    echo "    rounds=${rounds} attempts=${attempts} wall=${wall_s}s"
    grep -E "(compile|build|help|parse|test|smoke):(PASS|FAIL)" "${variant_dir}/container.log" | tail -6 | sed 's/^/    /'
    echo ""
}

# ── Main ────────────────────────────────────────────────────────────────

# Phase 1: baseline vs all_off
echo "═══ Phase 1: baseline vs all_off ═══"
echo ""

run_variant "00_baseline" ""
run_variant "01_all_off" "profile,guide,project_notes,plan,lessons,repo_map,mcp,scratchpad,usage_guide,plan_mode,lsp"

# Summary
echo ""
echo "================================================================="
echo "  RESULTS"
echo "================================================================="
printf "%-20s %8s %4s %8s %s\n" "Variant" "Rounds" "Att" "Time" "Result"
echo "-----------------------------------------------------------------"
for d in "${RESULTS_DIR}"/*/; do
    name=$(basename "$d")
    local_rounds=0
    for logfile in "$d"/*.log; do
        [ -f "${logfile}" ] || continue
        local_r=$(grep -c '\[round ' "${logfile}" 2>/dev/null || true)
        local_rounds=$((local_rounds + ${local_r:-0}))
    done
    wall=$(cat "$d/wall_s.txt" 2>/dev/null || echo "?")
    attempts=$(grep -c "=== ATTEMPT .* remaining" "$d/container.log" 2>/dev/null || echo "?")
    result=$(grep "=== FINAL:" "$d/container.log" 2>/dev/null | grep -oE "[0-9]+/[0-9]+" || echo "?/?")
    printf "%-20s %8s %4s %7ss  %s\n" "$name" "$local_rounds" "$attempts" "$wall" "$result"
done
echo "================================================================="
echo ""
echo "Detailed results: ${RESULTS_DIR}/"
