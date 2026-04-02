#!/usr/bin/env bash
# bench-common.sh — Shared functions for provider benchmark scripts.
# Source this, don't run it directly.
#
# Task-specific scripts must define:
#   TASK_NAME  — short identifier
#   TASK       — initial prompt for miniswe
#   validate_result(attempt_dir, work_dir)
#     → writes validation.txt (PASS/PARTIAL/FAIL)
#     → writes validation_errors.txt (human-readable failures for retry)

# SHA of the code the model works on (the task target).
# Pre-LSP codebase with warning fixes and UTF-8 truncation fix.
# Branch: bench-baseline
BASELINE_SHA="cc34d2626faf32c1b6dd1b8b33af693fb936b098"

# ── Defaults ────────────────────────────────────────────────────────────

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SHA="${BASELINE_SHA}"
LLM_ENDPOINT="${LLM_ENDPOINT:-http://localhost:8464}"
RESULTS_DIR=""
RUN_TIMEOUT=900
MAX_ROUNDS=50
MAX_ATTEMPTS=3
RUNS_PER_VARIANT=1
TEMPERATURE=0.0
STRATEGY="full"  # "full" = test every provider, "bisect" = coarse triage then drill down
PROVIDERS=(profile guide project_notes lessons repo_map scratchpad lsp)
# Provider groups for bisect strategy
PROVIDERS_CORE=(repo_map lsp profile)
PROVIDERS_EXTRAS=(guide project_notes lessons scratchpad)

# ── Arg parsing ─────────────────────────────────────────────────────────

parse_common_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --project-dir)   REPO_DIR="$2";          shift 2 ;;
            --sha)           SHA="$2";               shift 2 ;;
            --endpoint)      LLM_ENDPOINT="$2";      shift 2 ;;
            --results-dir)   RESULTS_DIR="$2";       shift 2 ;;
            --timeout)       RUN_TIMEOUT="$2";       shift 2 ;;
            --max-rounds)    MAX_ROUNDS="$2";        shift 2 ;;
            --max-attempts)  MAX_ATTEMPTS="$2";      shift 2 ;;
            --runs)          RUNS_PER_VARIANT="$2";  shift 2 ;;
            --temperature)   TEMPERATURE="$2";       shift 2 ;;
            --strategy)      STRATEGY="$2";          shift 2 ;;
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
    local disabled="${2:-}"  # comma-separated list of disabled providers

    # Helper: check if a provider is in the disabled list
    _is_disabled() {
        echo ",${disabled}," | grep -q ",${1},"
    }

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
profile = $(_is_disabled profile && echo "false" || echo "true")
guide = $(_is_disabled guide && echo "false" || echo "true")
project_notes = $(_is_disabled project_notes && echo "false" || echo "true")
plan = $(_is_disabled plan && echo "false" || echo "true")
lessons = $(_is_disabled lessons && echo "false" || echo "true")
repo_map = $(_is_disabled repo_map && echo "false" || echo "true")
mcp = $(_is_disabled mcp && echo "false" || echo "true")
scratchpad = $(_is_disabled scratchpad && echo "false" || echo "true")
usage_guide = $(_is_disabled usage_guide && echo "false" || echo "true")
plan_mode = $(_is_disabled plan_mode && echo "false" || echo "true")

[hardware]
vram_gb = 24.0
vram_reserve_gb = 3.0
ram_budget_gb = 80.0

[web]
search_backend = "serper"
fetch_backend = "jina"

[lsp]
enabled = $(_is_disabled lsp && echo "false" || echo "true")
diagnostic_timeout_ms = 2000

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

# Reset work dir to pinned SHA (clean slate for a new variant).
prepare_workdir() {
    cd "${WORK_DIR}"
    git rm -rf --quiet . 2>/dev/null || true
    cd - > /dev/null

    git -C "${REPO_DIR}" archive "${SHA}" | tar -x -C "${WORK_DIR}"

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

# ── Metric extraction from a single miniswe invocation ──────────────────

extract_attempt_metrics() {
    local attempt_dir="$1"
    local logfile
    logfile=$(ls "${attempt_dir}"/*.log 2>/dev/null | head -1 || echo "")

    local rounds=0 tool_calls=0 tool_errors=0 context_tokens=0

    if [ -n "${logfile}" ] && [ -f "${logfile}" ]; then
        rounds=$(grep -c '\[round ' "${logfile}" || true)
        rounds=${rounds:-0}
        tool_calls=$(grep -c '\[tool\]' "${logfile}" || true)
        tool_calls=${tool_calls:-0}
        tool_errors=$(grep -c '\[tool\] ✗' "${logfile}" || true)
        tool_errors=${tool_errors:-0}
        context_tokens=$(grep '\[context\]' "${logfile}" | head -1 | sed 's/.*~\([0-9]*\) tokens.*/\1/' || true)
        context_tokens=${context_tokens:-0}
    fi

    cat > "${attempt_dir}/metrics.txt" <<EOF
rounds=${rounds}
tool_calls=${tool_calls}
tool_errors=${tool_errors}
context_tokens=${context_tokens}
EOF
}

# Aggregate metrics across all attempts for a run into run_dir/metrics.txt.
aggregate_run_metrics() {
    local run_dir="$1"
    local total_rounds=0 total_tools=0 total_errors=0 context_tokens=0
    local attempts=0 final_status="unknown"

    for attempt_dir in "${run_dir}"/attempt_*/; do
        [ -f "${attempt_dir}/metrics.txt" ] || continue
        # Source metrics with defaults to avoid empty-var arithmetic errors
        local rounds=0 tool_calls=0 tool_errors=0
        eval "$(cat "${attempt_dir}/metrics.txt")"
        total_rounds=$((total_rounds + ${rounds:-0}))
        total_tools=$((total_tools + ${tool_calls:-0}))
        total_errors=$((total_errors + ${tool_errors:-0}))
        # context_tokens from first attempt (representative)
        if [ "$attempts" -eq 0 ]; then
            context_tokens=${context_tokens:-0}
        fi
        (( ++attempts ))
    done

    local files_changed=0
    if [ -f "${run_dir}/changed_files.txt" ]; then
        files_changed=$(grep -c . "${run_dir}/changed_files.txt" || true)
        files_changed=${files_changed:-0}
    fi

    local validation
    validation=$(cat "${run_dir}/validation.txt" 2>/dev/null || echo "FAIL")

    cat > "${run_dir}/metrics.txt" <<EOF
rounds=${total_rounds}
tool_calls=${total_tools}
tool_errors=${total_errors}
context_tokens=${context_tokens}
attempts=${attempts}
status=${final_status}
validation=${validation}
files_changed=${files_changed}
EOF
}

# Average metrics across repeated runs of the same variant.
average_metrics() {
    local variant_dir="$1"
    local sum_rounds=0 sum_tools=0 sum_errors=0 sum_tokens=0 sum_wall=0 sum_attempts=0
    local count=0 statuses="" validations=""

    for run_dir in "${variant_dir}"/run_*/; do
        [ -f "${run_dir}/metrics.txt" ] || continue
        local rounds=0 tool_calls=0 tool_errors=0 context_tokens=0 attempts=0 validation="?"
        eval "$(cat "${run_dir}/metrics.txt")"
        sum_rounds=$((sum_rounds + ${rounds:-0}))
        sum_tools=$((sum_tools + ${tool_calls:-0}))
        sum_errors=$((sum_errors + ${tool_errors:-0}))
        sum_tokens=$((sum_tokens + ${context_tokens:-0}))
        sum_attempts=$((sum_attempts + ${attempts:-0}))
        local wall_file
        wall_file=$(cat "${run_dir}/wall_ms.txt" 2>/dev/null || echo "0")
        sum_wall=$((sum_wall + ${wall_file:-0}))
        validations="${validations}${validation},"
        (( ++count ))
    done

    [ "$count" -eq 0 ] && count=1
    cat > "${variant_dir}/avg_metrics.txt" <<EOF
avg_rounds=$((sum_rounds / count))
avg_tool_calls=$((sum_tools / count))
avg_tool_errors=$((sum_errors / count))
avg_context_tokens=$((sum_tokens / count))
avg_attempts=$((sum_attempts / count))
avg_wall_ms=$((sum_wall / count))
run_count=${count}
validations=${validations}
EOF
}

# ── Runner with validate-retry loop ─────────────────────────────────────

# Run miniswe with retry on validation failure.
# Does NOT reset workdir between attempts — miniswe sees its own changes.
# Calls validate_result() which must be defined by the task-specific script.
#
# Args: run_name disabled_provider run_number task
run_miniswe() {
    local run_name="$1"
    local disabled="${2:-}"
    local run_num="${3:-1}"
    local task="$4"
    local run_dir="${RESULTS_DIR}/${run_name}/run_${run_num}"
    mkdir -p "${run_dir}"

    echo "--- ${run_name} (run ${run_num}/${RUNS_PER_VARIANT}) ---"

    # Fresh workdir for this run
    prepare_workdir
    write_config "${WORK_DIR}/.miniswe/config.toml" "${disabled}"
    cp "${WORK_DIR}/.miniswe/config.toml" "${run_dir}/config.toml"
    echo "${disabled:-none}" > "${run_dir}/disabled_provider.txt"

    local run_start
    run_start=$(date +%s)
    local deadline=$((run_start + RUN_TIMEOUT))
    local attempt=0
    local current_task="${task}"
    local verdict="FAIL"

    while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
        (( ++attempt ))
        local attempt_dir="${run_dir}/attempt_${attempt}"
        mkdir -p "${attempt_dir}"

        # Time remaining
        local now
        now=$(date +%s)
        local remaining=$((deadline - now))
        if [ "$remaining" -le 30 ]; then
            echo "    attempt ${attempt}: skipped — only ${remaining}s left"
            break
        fi

        echo "    attempt ${attempt}/${MAX_ATTEMPTS} (${remaining}s remaining)"

        # Clear logs from previous attempt
        rm -f "${WORK_DIR}/.miniswe/logs/"*.log

        # Run miniswe — workdir retains modifications from previous attempts
        cd "${WORK_DIR}"
        timeout "${remaining}" "${MINISWE}" --yes "${current_task}" \
            > "${attempt_dir}/stdout.txt" \
            2> "${attempt_dir}/stderr.txt" \
            || true
        cd - > /dev/null

        # Capture logs
        if ls "${WORK_DIR}/.miniswe/logs/"*.log 1>/dev/null 2>&1; then
            cp "${WORK_DIR}/.miniswe/logs/"*.log "${attempt_dir}/"
        fi

        # Extract per-attempt metrics
        extract_attempt_metrics "${attempt_dir}"

        # Validate
        validate_result "${attempt_dir}" "${WORK_DIR}"
        verdict=$(cat "${attempt_dir}/validation.txt" 2>/dev/null || echo "FAIL")

        if [ "${verdict}" = "PASS" ]; then
            echo "    attempt ${attempt}: PASS"
            break
        fi

        # Read errors for retry prompt
        local errors
        errors=$(cat "${attempt_dir}/validation_errors.txt" 2>/dev/null || echo "Unknown errors")
        echo "    attempt ${attempt}: ${verdict}"

        # Compose follow-up message for next attempt
        current_task="Your previous changes have these problems:
${errors}
Please fix the issues. The modified files are still on disk — read them, find the problems, and fix them."
    done

    # Record wall time for the entire run (all attempts)
    local run_end
    run_end=$(date +%s)
    echo $(( (run_end - run_start) * 1000 )) > "${run_dir}/wall_ms.txt"

    # Capture final state of changes
    cd "${WORK_DIR}"
    git diff --name-only 2>/dev/null > "${run_dir}/changed_files.txt" || true
    git ls-files --others --exclude-standard >> "${run_dir}/changed_files.txt" 2>/dev/null || true
    git diff 2>/dev/null > "${run_dir}/diff.patch" || true
    cd - > /dev/null

    # Final validation is from the last attempt
    echo "${verdict}" > "${run_dir}/validation.txt"
    echo "${attempt}" > "${run_dir}/attempts.txt"

    # Aggregate metrics
    aggregate_run_metrics "${run_dir}"

    local wall_s=$(( $(cat "${run_dir}/wall_ms.txt") / 1000 ))
    echo "    RESULT: ${verdict} in ${attempt} attempt(s), ${wall_s}s total"
    eval "$(cat "${run_dir}/metrics.txt")"
    echo "    rounds=${rounds} tools=${tool_calls} errors=${tool_errors} tokens=${context_tokens}"
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
    printf "%-22s %6s %6s %6s %8s %4s %6s %6s\n" \
        "Variant" "Rounds" "Tools" "Errors" "Tokens" "Att" "Time" "Valid"
    echo "-----------------------------------------------------------------"

    for variant_dir in "${RESULTS_DIR}"/*/; do
        [ -f "${variant_dir}/avg_metrics.txt" ] || continue
        variant=$(basename "${variant_dir}")
        eval "$(cat "${variant_dir}/avg_metrics.txt")"

        wall_s=$((avg_wall_ms / 1000))

        # Count pass/total
        local pass_count=0 total=0
        IFS=',' read -ra vlist <<< "${validations}"
        for v in "${vlist[@]}"; do
            [ -z "$v" ] && continue
            (( ++total ))
            [ "$v" = "PASS" ] && (( ++pass_count ))
        done
        local valid_str="${pass_count}/${total}"

        printf "%-22s %6s %6s %6s %8s %4s %5ss %6s\n" \
            "${variant}" \
            "${avg_rounds}" \
            "${avg_tool_calls}" \
            "${avg_tool_errors}" \
            "${avg_context_tokens}" \
            "${avg_attempts}" \
            "${wall_s}" \
            "${valid_str}"
    done
    echo "-----------------------------------------------------------------"

    # Delta table
    eval "$(cat "${RESULTS_DIR}/00_baseline/avg_metrics.txt")"
    bl_rounds=$avg_rounds
    bl_tools=$avg_tool_calls
    bl_errors=$avg_tool_errors
    bl_tokens=$avg_context_tokens
    bl_attempts=$avg_attempts
    bl_wall=$avg_wall_ms

    echo ""
    echo "Provider impact (delta vs baseline):"
    printf "%-22s %8s %8s %8s %10s %6s %8s\n" \
        "Disabled" "Rounds" "Tools" "Errors" "Tokens" "Att" "Time"
    echo "-----------------------------------------------------------------"

    for variant_dir in "${RESULTS_DIR}"/0[1-9]_*/; do
        [ -f "${variant_dir}/avg_metrics.txt" ] || continue
        disabled=$(cat "${variant_dir}"/run_1/disabled_provider.txt 2>/dev/null || echo "?")
        eval "$(cat "${variant_dir}/avg_metrics.txt")"

        dr=$((avg_rounds - bl_rounds))
        dt=$((avg_tool_calls - bl_tools))
        de=$((avg_tool_errors - bl_errors))
        dc=$((avg_context_tokens - bl_tokens))
        da=$((avg_attempts - bl_attempts))
        dw=$(( (avg_wall_ms - bl_wall) / 1000 ))

        printf "%-22s %+8d %+8d %+8d %+10d %+6d %+7ds\n" \
            "${disabled}" "${dr}" "${dt}" "${de}" "${dc}" "${da}" "${dw}"
    done

    echo "================================================================="
    echo ""
    echo "+ rounds/tools/attempts = more work without provider (it helps)"
    echo "- tokens = less context without provider (expected)"
    echo ""
    echo "Logs: ${RESULTS_DIR}/"
}

# ── Score extraction ────────────────────────────────────────────────────

# Get a composite score for a variant: rounds + (attempts * 20) + (errors * 2).
# Higher = worse. Used by bisect to compare variants.
get_variant_score() {
    local variant_dir="$1"
    [ -f "${variant_dir}/avg_metrics.txt" ] || { echo "9999"; return; }
    eval "$(cat "${variant_dir}/avg_metrics.txt")"
    local score=$(( ${avg_rounds:-50} + ${avg_attempts:-3} * 20 + ${avg_tool_errors:-0} * 2 ))
    # Penalize non-passing validation
    local pass_count=0 total=0
    IFS=',' read -ra vlist <<< "${validations:-}"
    for v in "${vlist[@]}"; do
        [ -z "$v" ] && continue
        (( ++total ))
        [ "$v" = "PASS" ] && (( ++pass_count ))
    done
    if [ "$total" -gt 0 ] && [ "$pass_count" -lt "$total" ]; then
        score=$((score + (total - pass_count) * 30))
    fi
    echo "$score"
}

# ── Orchestrator ────────────────────────────────────────────────────────

# Run a single variant and compute its average metrics.
# Args: variant_name disabled_csv
run_variant() {
    local variant_name="$1"
    local disabled="$2"
    for run in $(seq 1 "${RUNS_PER_VARIANT}"); do
        run_miniswe "${variant_name}" "${disabled}" "${run}" "${TASK}"
    done
    average_metrics "${RESULTS_DIR}/${variant_name}"
}

run_benchmark_init() {
    parse_common_args "$@"

    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    RESULTS_DIR="${RESULTS_DIR:-${REPO_DIR}/benchmark_results/${TASK_NAME}_${timestamp}}"
    mkdir -p "${RESULTS_DIR}"

    find_miniswe_binary
    init_workdir
    trap cleanup_workdir EXIT

    echo "=== miniswe provider benchmark: ${TASK_NAME} ==="
    echo "SHA:          ${SHA}"
    echo "Strategy:     ${STRATEGY}"
    echo "Task:         ${TASK:0:80}..."
    echo "Endpoint:     ${LLM_ENDPOINT}"
    echo "Results:      ${RESULTS_DIR}"
    echo "Timeout:      ${RUN_TIMEOUT}s per run"
    echo "Max rounds:   ${MAX_ROUNDS} per attempt"
    echo "Max attempts: ${MAX_ATTEMPTS} per run"
    echo "Runs/variant: ${RUNS_PER_VARIANT}"
    echo ""

    cat > "${RESULTS_DIR}/metadata.txt" <<EOF
task_name=${TASK_NAME}
sha=${SHA}
strategy=${STRATEGY}
timestamp=${timestamp}
task=${TASK}
endpoint=${LLM_ENDPOINT}
timeout=${RUN_TIMEOUT}
max_rounds=${MAX_ROUNDS}
max_attempts=${MAX_ATTEMPTS}
runs_per_variant=${RUNS_PER_VARIANT}
temperature=${TEMPERATURE}
providers_tested=${PROVIDERS[*]}
EOF

    echo "Checking LLM endpoint..."
    if ! curl -s --connect-timeout 5 "${LLM_ENDPOINT}/v1/models" > /dev/null 2>&1; then
        echo "WARNING: LLM endpoint not responding at ${LLM_ENDPOINT}"
    else
        echo "OK"
    fi
    echo ""
}

# Full ablation: test every provider individually.
run_benchmark_full() {
    run_benchmark_init "$@"

    run_variant "00_baseline" ""

    local i=1
    for provider in "${PROVIDERS[@]}"; do
        run_variant "$(printf "%02d_no_%s" "$i" "$provider")" "${provider}"
        (( ++i ))
    done

    print_summary "${TASK_NAME}"
}

# Bisect: coarse triage, then drill into impactful group.
run_benchmark_bisect() {
    run_benchmark_init "$@"

    # ── Phase 1: Coarse triage (3 runs) ──────────────────────────────
    echo "═══ PHASE 1: Coarse triage ═══"
    echo ""

    run_variant "00_baseline" ""
    local baseline_score
    baseline_score=$(get_variant_score "${RESULTS_DIR}/00_baseline")

    # All providers off
    local all_disabled
    all_disabled=$(IFS=,; echo "${PROVIDERS[*]}")
    run_variant "01_all_off" "${all_disabled}"
    local all_off_score
    all_off_score=$(get_variant_score "${RESULTS_DIR}/01_all_off")

    # Core off (repo_map, lsp, profile)
    local core_disabled
    core_disabled=$(IFS=,; echo "${PROVIDERS_CORE[*]}")
    run_variant "02_core_off" "${core_disabled}"
    local core_off_score
    core_off_score=$(get_variant_score "${RESULTS_DIR}/02_core_off")

    echo ""
    echo "── Phase 1 scores (lower = better) ──"
    echo "  baseline:  ${baseline_score}"
    echo "  all_off:   ${all_off_score}"
    echo "  core_off:  ${core_off_score}"
    echo ""

    # Decide whether to continue
    local delta=$((all_off_score - baseline_score))
    if [ "$delta" -lt 5 ] && [ "$delta" -gt -5 ]; then
        echo "No significant impact from any providers (delta=${delta}). Stopping."
        print_summary "${TASK_NAME}"
        return
    fi

    # ── Phase 2: Drill into impactful group ──────────────────────────
    echo "═══ PHASE 2: Drill down ═══"
    echo ""

    local core_delta=$((core_off_score - baseline_score))
    local extras_delta=$((all_off_score - core_off_score))

    echo "  Core providers impact:   ${core_delta}"
    echo "  Extras providers impact: ${extras_delta}"
    echo ""

    # Test individual providers in the group with bigger impact
    local i=10
    if [ "$core_delta" -ge "$extras_delta" ] || [ "$core_delta" -ge 5 ]; then
        echo "Drilling into CORE providers: ${PROVIDERS_CORE[*]}"
        echo ""
        for provider in "${PROVIDERS_CORE[@]}"; do
            run_variant "$(printf "%02d_no_%s" "$i" "$provider")" "${provider}"
            (( ++i ))
        done
    fi

    if [ "$extras_delta" -ge "$core_delta" ] || [ "$extras_delta" -ge 5 ]; then
        echo "Drilling into EXTRAS providers: ${PROVIDERS_EXTRAS[*]}"
        echo ""
        for provider in "${PROVIDERS_EXTRAS[@]}"; do
            run_variant "$(printf "%02d_no_%s" "$i" "$provider")" "${provider}"
            (( ++i ))
        done
    fi

    print_summary "${TASK_NAME}"
}

# Main entry point. Caller must define: TASK, TASK_NAME, validate_result()
run_benchmark() {
    case "${STRATEGY:-full}" in
        full)   run_benchmark_full "$@" ;;
        bisect) run_benchmark_bisect "$@" ;;
        *)
            echo "Unknown strategy: ${STRATEGY}. Use 'full' or 'bisect'." >&2
            exit 1
            ;;
    esac
}
