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

# Run miniswe
echo "=== MINISWE START ==="
timeout "${TIMEOUT}" miniswe --yes "${TASK}" > /output/stdout.txt 2> /output/stderr.txt || true
echo "=== MINISWE END ==="

# Copy logs
cp .miniswe/logs/*.log /output/ 2>/dev/null || true

# Capture changes
git diff --name-only > /output/changed_files.txt 2>/dev/null || true
git ls-files --others --exclude-standard >> /output/changed_files.txt 2>/dev/null || true
git diff > /output/diff.patch 2>/dev/null || true

# === Validation ===
echo "=== VALIDATION ==="

# Check 1: cargo check
if RUSTFLAGS="-A warnings" cargo check 2> /output/cargo_check.txt; then
    echo "compile:PASS"
else
    echo "compile:FAIL"
    cat /output/cargo_check.txt | grep "^error" | head -20 > /output/compile_errors.txt
fi

# Check 2: cargo build
if RUSTFLAGS="-A warnings" cargo build 2> /output/cargo_build.txt; then
    echo "build:PASS"
else
    echo "build:FAIL"
fi

# Check 3: --help shows rounds flag
BINARY="./target/debug/miniswe"
if [ -f "${BINARY}" ]; then
    "${BINARY}" --help > /output/help_output.txt 2>&1 || true
    if grep -qiE -- '--[a-z-]*round[a-z-]*' /output/help_output.txt; then
        FLAG=$(grep -oE -- '--[a-z-]*round[a-z-]*' /output/help_output.txt | head -1)
        echo "help:PASS(${FLAG})"

        # Check 4: flag parses
        if "${BINARY}" ${FLAG} 5 --help > /dev/null 2>&1; then
            echo "parse:PASS"
        else
            echo "parse:FAIL"
        fi

        # Check 5: cargo test
        if RUSTFLAGS="-A warnings" cargo test 2> /output/cargo_test.txt; then
            echo "test:PASS"
        else
            echo "test:FAIL"
        fi

        # Check 6: smoke test
        rm -f .miniswe/logs/*.log
        if timeout 120 "${BINARY}" ${FLAG} 1 --yes "Say hello" > /output/smoke_stdout.txt 2> /output/smoke_stderr.txt; then
            ROUNDS=$(grep -c '\[round ' .miniswe/logs/*.log 2>/dev/null || echo 0)
            if [ "${ROUNDS}" -ge 1 ] && [ "${ROUNDS}" -le 2 ]; then
                echo "smoke:PASS(${ROUNDS}r)"
            else
                echo "smoke:FAIL(${ROUNDS}r)"
            fi
        else
            echo "smoke:FAIL(exit)"
        fi
    else
        echo "help:FAIL"
    fi
else
    echo "build:NOBINARY"
fi

echo "=== VALIDATION END ==="
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

    # Parse validation results
    local validation_output
    validation_output=$(sed -n '/=== VALIDATION ===/,/=== VALIDATION END ===/p' "${variant_dir}/container.log")
    echo "${validation_output}" > "${variant_dir}/validation_raw.txt"

    local pass=0 total=0
    for check in compile build help parse test smoke; do
        if echo "${validation_output}" | grep -q "${check}:PASS"; then
            (( ++pass ))
        fi
        if echo "${validation_output}" | grep -qE "${check}:(PASS|FAIL)"; then
            (( ++total ))
        fi
    done

    local wall_s
    wall_s=$(cat "${variant_dir}/wall_s.txt")

    # Count rounds from log
    local rounds=0
    if ls "${variant_dir}"/*.log 1>/dev/null 2>&1; then
        rounds=$(grep -c '\[round ' "${variant_dir}"/*.log 2>/dev/null || true)
        rounds=${rounds:-0}
    fi

    echo ""
    echo "    Result: ${pass}/${total} passed, ${rounds} rounds, ${wall_s}s"
    echo "    ${validation_output}" | grep -E "(PASS|FAIL)" | sed 's/^/    /'
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
printf "%-20s %8s %8s\n" "Variant" "Rounds" "Time"
echo "-----------------------------------------------------------------"
for d in "${RESULTS_DIR}"/*/; do
    name=$(basename "$d")
    rounds=$(grep -c '\[round ' "$d"/*.log 2>/dev/null || echo "?")
    wall=$(cat "$d/wall_s.txt" 2>/dev/null || echo "?")
    validation=$(sed -n '/=== VALIDATION ===/,/=== VALIDATION END ===/p' "$d/container.log" 2>/dev/null | grep -cE "PASS" || echo "?")
    printf "%-20s %8s %7ss  %s/6\n" "$name" "$rounds" "$wall" "$validation"
done
echo "================================================================="
echo ""
echo "Detailed results: ${RESULTS_DIR}/"
