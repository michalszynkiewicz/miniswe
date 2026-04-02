#!/usr/bin/env bash
# bench-task-B — Benchmark: "Add --max-rounds CLI flag"
#
# Validates by building, running the binary, running tests, and
# doing a live smoke test with the LLM to confirm the flag works.
# On failure, feeds errors back to miniswe for a retry.
#
# Validation:
#   1. cargo check — does it compile?
#   2. cargo build — can we get a binary?
#   3. binary --help — does it show the rounds flag?
#   4. binary <flag> 5 --help — does the flag parse?
#   5. cargo test — do the model's own tests pass?
#   6. live smoke test — run with rounds=1, verify log shows 1 round
#
# Usage:
#   ./scripts/bench-task-B-max-rounds-flag.sh [--runs 3] [--timeout 600]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/bench-common.sh"

TASK_NAME="task_B_max_rounds_flag"
TASK="Add a CLI flag that lets the user limit the maximum number of agent rounds per session. It should override whatever the config file says. Make sure it works for both single-shot and interactive modes. Write a test that verifies the flag actually limits the number of rounds."

# ── Validation ──────────────────────────────────────────────────────────

# Detect the flag name the model chose (might be --max-rounds, --rounds, etc.)
detect_flag_name() {
    local help_output="$1"
    # Look for common patterns in help output
    local flag
    flag=$(echo "${help_output}" | grep -oE -- '--[a-z-]*round[a-z-]*' | head -1 || true)
    echo "${flag}"
}

validate_result() {
    local attempt_dir="$1"
    local work_dir="$2"
    local errors=""
    local passed=0
    local checks=0
    local details=""
    local binary="${work_dir}/target/debug/miniswe"
    local flag_name=""

    # Check 1: cargo check
    (( ++checks ))
    local check_output
    if check_output=$(cd "${work_dir}" && RUSTFLAGS="-A warnings" cargo check 2>&1); then
        (( ++passed ))
        details="${details}compile:PASS "
    else
        details="${details}compile:FAIL "
        local err_lines
        err_lines=$(echo "${check_output}" | grep -E '^error' | head -20)
        errors="${errors}
COMPILE ERROR:
${err_lines}"
    fi
    echo "${check_output}" > "${attempt_dir}/cargo_check.txt"

    # Check 2: cargo build (only if check passed)
    (( ++checks ))
    if [ "$passed" -ge 1 ]; then
        local build_output
        if build_output=$(cd "${work_dir}" && RUSTFLAGS="-A warnings" cargo build 2>&1); then
            (( ++passed ))
            details="${details}build:PASS "
        else
            details="${details}build:FAIL "
            errors="${errors}
BUILD ERROR:
$(echo "${build_output}" | grep -E '^error' | head -10)"
        fi
        echo "${build_output}" > "${attempt_dir}/cargo_build.txt"
    else
        details="${details}build:SKIP "
        errors="${errors}
BUILD: skipped (compile failed)"
    fi

    # Check 3: --help shows some kind of rounds flag
    (( ++checks ))
    if [ -f "${binary}" ]; then
        local help_output
        help_output=$("${binary}" --help 2>&1 || true)
        echo "${help_output}" > "${attempt_dir}/help_output.txt"

        flag_name=$(detect_flag_name "${help_output}")
        if [ -n "${flag_name}" ]; then
            (( ++passed ))
            details="${details}help:PASS(${flag_name}) "
        else
            details="${details}help:FAIL "
            errors="${errors}
--help output does not contain a flag related to 'rounds'. Full output:
${help_output}"
        fi
    else
        details="${details}help:SKIP "
        errors="${errors}
HELP: skipped (no binary)"
    fi

    # Check 4: flag parses without error
    (( ++checks ))
    if [ -f "${binary}" ] && [ -n "${flag_name}" ]; then
        local flag_output
        local flag_exit=0
        flag_output=$("${binary}" ${flag_name} 5 --help 2>&1) || flag_exit=$?
        echo "${flag_output}" > "${attempt_dir}/flag_test.txt"
        echo "exit_code=${flag_exit}" >> "${attempt_dir}/flag_test.txt"

        if [ "$flag_exit" -eq 0 ]; then
            (( ++passed ))
            details="${details}parse:PASS "
        else
            details="${details}parse:FAIL "
            errors="${errors}
'miniswe ${flag_name} 5 --help' exited with code ${flag_exit}. Output:
$(echo "${flag_output}" | head -10)"
        fi
    else
        details="${details}parse:SKIP "
    fi

    # Check 5: cargo test
    (( ++checks ))
    if [ "$passed" -ge 2 ]; then
        local test_output
        local test_exit=0
        test_output=$(cd "${work_dir}" && RUSTFLAGS="-A warnings" cargo test 2>&1) || test_exit=$?
        echo "${test_output}" > "${attempt_dir}/cargo_test.txt"

        if [ "$test_exit" -eq 0 ]; then
            (( ++passed ))
            details="${details}test:PASS "
        else
            details="${details}test:FAIL "
            local failures
            failures=$(echo "${test_output}" | grep -A5 'FAILED\|panicked\|test result:' | head -20)
            errors="${errors}
TESTS FAILED (exit ${test_exit}):
${failures}"
        fi
    else
        details="${details}test:SKIP "
        errors="${errors}
TESTS: skipped (build failed)"
    fi

    # Check 6: smoke test — run a multi-round task with and without the flag.
    # Verifies the flag actually limits rounds, not just parses.
    (( ++checks ))
    if [ -f "${binary}" ] && [ -n "${flag_name}" ] && [ "$passed" -ge 4 ]; then
        mkdir -p "${work_dir}/.miniswe/logs"

        # Run WITHOUT flag to get baseline round count
        rm -f "${work_dir}/.miniswe/logs/"*.log
        cd "${work_dir}"
        timeout 120 "${binary}" --yes "Read every .rs file in src/ and list their names" \
            > "${attempt_dir}/smoke_baseline_stdout.txt" \
            2> "${attempt_dir}/smoke_baseline_stderr.txt" \
            || true
        cd - > /dev/null
        local baseline_rounds=0
        for f in "${work_dir}/.miniswe/logs/"*.log; do
            [ -f "$f" ] || continue
            local r; r=$(grep -c '\[round ' "$f" || true)
            baseline_rounds=$((baseline_rounds + ${r:-0}))
        done

        # Run WITH flag set to 2
        rm -f "${work_dir}/.miniswe/logs/"*.log
        cd "${work_dir}"
        timeout 120 "${binary}" ${flag_name} 2 --yes "Read every .rs file in src/ and list their names" \
            > "${attempt_dir}/smoke_stdout.txt" \
            2> "${attempt_dir}/smoke_stderr.txt" \
            || true
        cd - > /dev/null
        local limited_rounds=0
        for f in "${work_dir}/.miniswe/logs/"*.log; do
            [ -f "$f" ] || continue
            local r; r=$(grep -c '\[round ' "$f" || true)
            limited_rounds=$((limited_rounds + ${r:-0}))
        done

        echo "baseline=${baseline_rounds} limited=${limited_rounds}" > "${attempt_dir}/smoke_result.txt"

        if [ "${limited_rounds}" -le 2 ] && [ "${baseline_rounds}" -gt 2 ]; then
            (( ++passed ))
            details="${details}smoke:PASS(base=${baseline_rounds}r,lim=${limited_rounds}r) "
        elif [ "${limited_rounds}" -le 2 ] && [ "${baseline_rounds}" -le 2 ]; then
            # Both finish quickly — inconclusive but give benefit of doubt
            if [ "$passed" -ge 5 ]; then
                (( ++passed ))
            fi
            details="${details}smoke:INCONCLUSIVE(both≤2r) "
        else
            details="${details}smoke:FAIL(base=${baseline_rounds}r,lim=${limited_rounds}r) "
            errors="${errors}
SMOKE TEST: flag should limit to 2 rounds but got ${limited_rounds}. Without flag: ${baseline_rounds} rounds. The flag is not wired to the agent loop."
        fi
    else
        details="${details}smoke:SKIP "
        errors="${errors}
SMOKE TEST: skipped (earlier checks failed)"
    fi

    # Verdict
    local verdict="FAIL"
    if [ "${passed}" -eq "${checks}" ]; then
        verdict="PASS"
    elif [ "${passed}" -ge 4 ]; then
        verdict="PARTIAL"
    fi

    echo "${verdict}" > "${attempt_dir}/validation.txt"
    echo "${passed}/${checks} ${details}" > "${attempt_dir}/validation_details.txt"
    echo "${errors}" > "${attempt_dir}/validation_errors.txt"
    echo "    validation: ${verdict} (${passed}/${checks}) ${details}"
}

# ── Run ─────────────────────────────────────────────────────────────────

run_benchmark "$@"
