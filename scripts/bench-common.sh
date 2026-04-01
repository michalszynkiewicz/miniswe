#!/usr/bin/env bash
# bench-common.sh — Shared functions for provider benchmark scripts.
# Source this, don't run it directly.

# Pinned SHA — provider system was introduced here.
# Benchmark tasks don't exist at this SHA, so tests stay valid.
BASELINE_SHA="e152ca85416f2c8a3ec027c8dfce8dbc42211af6"

# ── Defaults ────────────────────────────────────────────────────────────

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SHA="${BASELINE_SHA}"
LLM_ENDPOINT="${LLM_ENDPOINT:-http://localhost:8464}"
RESULTS_DIR=""
RUN_TIMEOUT=300
MAX_ROUNDS=30
RUNS_PER_VARIANT=1
TEMPERATURE=0.0
PROVIDERS=(profile guide project_notes lessons repo_map scratchpad)

# ── Arg parsing ─────────────────────────────────────────────────────────

parse_common_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --project-dir)  REPO_DIR="$2";          shift 2 ;;
            --sha)          SHA="$2";               shift 2 ;;
            --endpoint)     LLM_ENDPOINT="$2";      shift 2 ;;
            --results-dir)  RESULTS_DIR="$2";       shift 2 ;;
            --timeout)      RUN_TIMEOUT="$2";       shift 2 ;;
            --max-rounds)   MAX_ROUNDS="$2";        shift 2 ;;
            --runs)         RUNS_PER_VARIANT="$2";  shift 2 ;;
            *)
                echo "Unknown option: $1" >&2
                exit 1
                ;;
        esac
    done
}

# ── Config writer ───────────────────────────────────────────────────────

write_config() {
    local out="$1"
    local disabled="${2:-}"

    cat > "$out" <<TOML
[model]
provider = "llama-cpp"
endpoint = "${LLM_ENDPOINT}"
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
pause_after_rounds = ${MAX_ROUNDS}

[context.providers]
profile = $([ "$disabled" = "profile" ] && echo "false" || echo "true")
guide = $([ "$disabled" = "guide" ] && echo "false" || echo "true")
project_notes = $([ "$disabled" = "project_notes" ] && echo "false" || echo "true")
plan = $([ "$disabled" = "plan" ] && echo "false" || echo "true")
lessons = $([ "$disabled" = "lessons" ] && echo "false" || echo "true")
repo_map = $([ "$disabled" = "repo_map" ] && echo "false" || echo "true")
mcp = $([ "$disabled" = "mcp" ] && echo "false" || echo "true")
scratchpad = $([ "$disabled" = "scratchpad" ] && echo "false" || echo "true")
usage_guide = $([ "$disabled" = "usage_guide" ] && echo "false" || echo "true")
plan_mode = $([ "$disabled" = "plan_mode" ] && echo "false" || echo "true")

[hardware]
vram_gb = 24.0
vram_reserve_gb = 3.0
ram_budget_gb = 80.0

[web]
search_backend = "serper"
fetch_backend = "jina"

[logging]
level = "trace"
enabled = true
TOML
}

# ── Work dir management ─────────────────────────────────────────────────

WORK_DIR=""

init_workdir() {
    WORK_DIR=$(mktemp -d "/tmp/miniswe-bench-XXXXXX")
    mkdir -p "${WORK_DIR}"
    cd "${WORK_DIR}" && git init -q && cd - > /dev/null
    prepare_workdir
}

prepare_workdir() {
    # Clean slate from pinned SHA
    cd "${WORK_DIR}"
    git rm -rf --quiet . 2>/dev/null || true
    cd - > /dev/null

    git -C "${REPO_DIR}" archive "${SHA}" | tar -x -C "${WORK_DIR}"

    # Copy .miniswe index data
    rm -rf "${WORK_DIR}/.miniswe"
    mkdir -p "${WORK_DIR}/.miniswe"
    for item in index profile.md guide.md lessons.md; do
        if [ -e "${REPO_DIR}/.miniswe/${item}" ]; then
            cp -r "${REPO_DIR}/.miniswe/${item}" "${WORK_DIR}/.miniswe/"
        fi
    done
    mkdir -p "${WORK_DIR}/.miniswe/logs"

    cd "${WORK_DIR}"
    git add -A && git commit -q --allow-empty -m "pinned at ${SHA}" 2>/dev/null || true
    cd - > /dev/null
}

cleanup_workdir() {
    [ -n "${WORK_DIR}" ] && rm -rf "${WORK_DIR}"
}

# ── Metric extraction ──────────────────────────────────────────────────

extract_metrics() {
    local run_dir="$1"
    local logfile
    logfile=$(ls "${run_dir}"/*.log 2>/dev/null | head -1 || echo "")

    local rounds=0 tool_calls=0 tool_errors=0 context_tokens=0 status="unknown"

    if [ -n "${logfile}" ] && [ -f "${logfile}" ]; then
        rounds=$(grep -c '^\S\+ \[round ' "${logfile}" 2>/dev/null || echo "0")
        tool_calls=$(grep -c '^\S\+ \[tool\]' "${logfile}" 2>/dev/null || echo "0")
        tool_errors=$(grep -c '^\S\+ \[tool\] ✗' "${logfile}" 2>/dev/null || echo "0")

        context_tokens=$(grep '\[context\]' "${logfile}" 2>/dev/null \
            | head -1 \
            | sed 's/.*~\([0-9]*\) tokens.*/\1/' \
            || echo "0")

        if grep -q '\[end\].*status=ok' "${logfile}" 2>/dev/null; then
            status="ok"
        elif grep -q '\[end\].*status=error' "${logfile}" 2>/dev/null; then
            status="error"
        else
            status="timeout"
        fi
    fi

    local files_changed=0
    if [ -f "${run_dir}/changed_files.txt" ]; then
        files_changed=$(grep -c . "${run_dir}/changed_files.txt" 2>/dev/null || echo "0")
    fi

    cat > "${run_dir}/metrics.txt" <<EOF
rounds=${rounds}
tool_calls=${tool_calls}
tool_errors=${tool_errors}
context_tokens=${context_tokens}
status=${status}
files_changed=${files_changed}
EOF
}

average_metrics() {
    local variant_dir="$1"
    local sum_rounds=0 sum_tools=0 sum_errors=0 sum_tokens=0 sum_wall=0
    local count=0 statuses="" validations=""

    for run_dir in "${variant_dir}"/run_*/; do
        [ -f "${run_dir}/metrics.txt" ] || continue
        eval "$(cat "${run_dir}/metrics.txt")"
        sum_rounds=$((sum_rounds + rounds))
        sum_tools=$((sum_tools + tool_calls))
        sum_errors=$((sum_errors + tool_errors))
        sum_tokens=$((sum_tokens + context_tokens))
        sum_wall=$((sum_wall + $(cat "${run_dir}/wall_ms.txt" 2>/dev/null || echo 0)))
        statuses="${statuses}${status},"
        # Read validation result
        local v
        v=$(cat "${run_dir}/validation.txt" 2>/dev/null || echo "?")
        validations="${validations}${v},"
        ((count++))
    done

    [ "$count" -eq 0 ] && count=1
    cat > "${variant_dir}/avg_metrics.txt" <<EOF
avg_rounds=$((sum_rounds / count))
avg_tool_calls=$((sum_tools / count))
avg_tool_errors=$((sum_errors / count))
avg_context_tokens=$((sum_tokens / count))
avg_wall_ms=$((sum_wall / count))
run_count=${count}
statuses=${statuses}
validations=${validations}
EOF
}

# ── Runner ──────────────────────────────────────────────────────────────

# Run miniswe once. Calls validate_result (must be defined by the caller).
# Args: run_name disabled_provider run_number task
run_miniswe() {
    local run_name="$1"
    local disabled="${2:-}"
    local run_num="${3:-1}"
    local task="$4"
    local run_dir="${RESULTS_DIR}/${run_name}/run_${run_num}"
    mkdir -p "${run_dir}"

    echo "--- ${run_name} (run ${run_num}/${RUNS_PER_VARIANT}) ---"

    prepare_workdir
    write_config "${WORK_DIR}/.miniswe/config.toml" "${disabled}"
    cp "${WORK_DIR}/.miniswe/config.toml" "${run_dir}/config.toml"
    echo "${disabled:-none}" > "${run_dir}/disabled_provider.txt"

    local start_time
    start_time=$(date +%s%3N)

    cd "${WORK_DIR}"
    timeout "${RUN_TIMEOUT}" "${MINISWE}" --yes "${task}" \
        > "${run_dir}/stdout.txt" \
        2> "${run_dir}/stderr.txt" \
        || true
    cd - > /dev/null

    local end_time
    end_time=$(date +%s%3N)
    echo $((end_time - start_time)) > "${run_dir}/wall_ms.txt"

    # Capture logs
    if ls "${WORK_DIR}/.miniswe/logs/"*.log 1>/dev/null 2>&1; then
        cp "${WORK_DIR}/.miniswe/logs/"*.log "${run_dir}/"
    fi

    # Capture changes
    cd "${WORK_DIR}"
    git diff --name-only 2>/dev/null > "${run_dir}/changed_files.txt" || true
    git ls-files --others --exclude-standard >> "${run_dir}/changed_files.txt" 2>/dev/null || true
    git diff 2>/dev/null > "${run_dir}/diff.patch" || true
    cd - > /dev/null

    extract_metrics "${run_dir}"

    # Validate — function provided by the task-specific script
    validate_result "${run_dir}" "${WORK_DIR}"

    local wall_s=$(( $(cat "${run_dir}/wall_ms.txt") / 1000 ))
    local validation
    validation=$(cat "${run_dir}/validation.txt" 2>/dev/null || echo "?")
    echo "    ${wall_s}s  valid=${validation}  $(cat "${run_dir}/metrics.txt" | tr '\n' '  ')"
    echo ""
}

# ── Find binary ─────────────────────────────────────────────────────────

find_miniswe_binary() {
    if [ -f "${REPO_DIR}/target/release/miniswe" ]; then
        MINISWE="${REPO_DIR}/target/release/miniswe"
    elif [ -f "${REPO_DIR}/target/debug/miniswe" ]; then
        MINISWE="${REPO_DIR}/target/debug/miniswe"
    else
        echo "Building miniswe..."
        cargo build --release --manifest-path="${REPO_DIR}/Cargo.toml"
        MINISWE="${REPO_DIR}/target/release/miniswe"
    fi
}

# ── Summary tables ──────────────────────────────────────────────────────

print_summary() {
    local task_name="$1"

    echo ""
    echo "================================================================="
    echo "  ${task_name} — RESULTS (SHA: ${SHA:0:12})"
    echo "================================================================="
    printf "%-22s %6s %6s %6s %8s %6s %6s %s\n" \
        "Variant" "Rounds" "Tools" "Errors" "Tokens" "Time" "Valid" "Status"
    echo "-----------------------------------------------------------------"

    for variant_dir in "${RESULTS_DIR}"/*/; do
        [ -f "${variant_dir}/avg_metrics.txt" ] || continue
        variant=$(basename "${variant_dir}")
        eval "$(cat "${variant_dir}/avg_metrics.txt")"

        wall_s=$((avg_wall_ms / 1000))

        # Count validations that passed
        local pass_count=0 total=0
        IFS=',' read -ra vlist <<< "${validations}"
        for v in "${vlist[@]}"; do
            [ -z "$v" ] && continue
            ((total++))
            [ "$v" = "PASS" ] && ((pass_count++))
        done
        local valid_str="${pass_count}/${total}"

        printf "%-22s %6s %6s %6s %8s %5ss %6s %s\n" \
            "${variant}" \
            "${avg_rounds}" \
            "${avg_tool_calls}" \
            "${avg_tool_errors}" \
            "${avg_context_tokens}" \
            "${wall_s}" \
            "${valid_str}" \
            "${statuses}"
    done
    echo "-----------------------------------------------------------------"

    # Delta table
    eval "$(cat "${RESULTS_DIR}/00_baseline/avg_metrics.txt")"
    bl_rounds=$avg_rounds
    bl_tools=$avg_tool_calls
    bl_errors=$avg_tool_errors
    bl_tokens=$avg_context_tokens
    bl_wall=$avg_wall_ms

    echo ""
    echo "Provider impact (delta vs baseline):"
    printf "%-22s %8s %8s %8s %10s %8s\n" \
        "Disabled" "Rounds" "Tools" "Errors" "Tokens" "Time"
    echo "-----------------------------------------------------------------"

    for variant_dir in "${RESULTS_DIR}"/0[1-9]_*/; do
        [ -f "${variant_dir}/avg_metrics.txt" ] || continue
        disabled=$(cat "${variant_dir}"/run_1/disabled_provider.txt 2>/dev/null || echo "?")
        eval "$(cat "${variant_dir}/avg_metrics.txt")"

        dr=$((avg_rounds - bl_rounds))
        dt=$((avg_tool_calls - bl_tools))
        de=$((avg_tool_errors - bl_errors))
        dc=$((avg_context_tokens - bl_tokens))
        dw=$(( (avg_wall_ms - bl_wall) / 1000 ))

        printf "%-22s %+8d %+8d %+8d %+10d %+7ds\n" \
            "${disabled}" "${dr}" "${dt}" "${de}" "${dc}" "${dw}"
    done

    echo "================================================================="
    echo ""
    echo "+ rounds/tools = more work needed without provider (provider helps)"
    echo "- tokens = less context without provider (expected)"
    echo ""
    echo "Logs: ${RESULTS_DIR}/"
}

# ── Orchestrator ────────────────────────────────────────────────────────

# Main entry point. Caller must define: TASK, TASK_NAME, validate_result()
run_benchmark() {
    parse_common_args "$@"

    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    RESULTS_DIR="${RESULTS_DIR:-${REPO_DIR}/benchmark_results/${TASK_NAME}_${timestamp}}"
    mkdir -p "${RESULTS_DIR}"

    find_miniswe_binary
    init_workdir
    trap cleanup_workdir EXIT

    echo "=== miniswe provider benchmark: ${TASK_NAME} ==="
    echo "SHA:        ${SHA}"
    echo "Task:       ${TASK:0:80}..."
    echo "Endpoint:   ${LLM_ENDPOINT}"
    echo "Results:    ${RESULTS_DIR}"
    echo "Timeout:    ${RUN_TIMEOUT}s"
    echo "Max rounds: ${MAX_ROUNDS}"
    echo "Runs/var:   ${RUNS_PER_VARIANT}"
    echo ""

    # Save metadata
    cat > "${RESULTS_DIR}/metadata.txt" <<EOF
task_name=${TASK_NAME}
sha=${SHA}
timestamp=${timestamp}
task=${TASK}
endpoint=${LLM_ENDPOINT}
timeout=${RUN_TIMEOUT}
max_rounds=${MAX_ROUNDS}
runs_per_variant=${RUNS_PER_VARIANT}
temperature=${TEMPERATURE}
providers_tested=${PROVIDERS[*]}
EOF

    # Verify endpoint
    echo "Checking LLM endpoint..."
    if ! curl -s --connect-timeout 5 "${LLM_ENDPOINT}/v1/models" > /dev/null 2>&1; then
        echo "WARNING: LLM endpoint not responding at ${LLM_ENDPOINT}"
        echo ""
    else
        echo "OK"
        echo ""
    fi

    # Baseline
    for run in $(seq 1 "${RUNS_PER_VARIANT}"); do
        run_miniswe "00_baseline" "" "${run}" "${TASK}"
    done
    average_metrics "${RESULTS_DIR}/00_baseline"

    # Ablation
    local i=1
    for provider in "${PROVIDERS[@]}"; do
        variant_name=$(printf "%02d_no_%s" "$i" "$provider")
        for run in $(seq 1 "${RUNS_PER_VARIANT}"); do
            run_miniswe "${variant_name}" "${provider}" "${run}" "${TASK}"
        done
        average_metrics "${RESULTS_DIR}/${variant_name}"
        ((i++))
    done

    print_summary "${TASK_NAME}"
}
