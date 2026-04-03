#!/usr/bin/env bash
# bench-task-B — Benchmark: "Add --system-prompt-override CLI flag"
#
# Validates by building, running, and checking predictable output.
# On failure, feeds errors back to miniswe for a retry.
#
# Validation:
#   1. cargo check — compiles?
#   2. cargo build — binary?
#   3. --help — shows the flag?
#   4. flag parses without error?
#   5. cargo test — tests pass?
#   6. smoke: run with override prompt, check predictable output
#
# Usage:
#   ./scripts/bench-task-B-max-rounds-flag.sh [--timeout 1800] [--max-rounds 80]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/bench-common.sh"

TASK_NAME="task_B_system_prompt_override"
TASK="Add a CLI flag --system-prompt-override (short: -s) that takes a string and replaces the default system prompt with the provided text. When this flag is set, skip all context providers and just use the override text as the system message. Make sure it works for both single-shot and interactive modes."

# ── Validation ──────────────────────────────────────────────────────────

detect_flag_name() {
    local help_output="$1"
    local flag
    flag=$(echo "${help_output}" | grep -oE -- '--[a-z-]*prompt[a-z-]*' | head -1 || true)
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

    # Check 2: cargo build
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
    fi

    # Check 3: --help shows prompt flag
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
--help does not show a flag related to 'prompt'. Full output:
${help_output}"
        fi
    else
        details="${details}help:SKIP "
    fi

    # Check 4: flag parses
    (( ++checks ))
    if [ -f "${binary}" ] && [ -n "${flag_name}" ]; then
        local flag_output
        local flag_exit=0
        flag_output=$("${binary}" ${flag_name} "test prompt" --help 2>&1) || flag_exit=$?
        echo "${flag_output}" > "${attempt_dir}/flag_test.txt"
        if [ "$flag_exit" -eq 0 ]; then
            (( ++passed ))
            details="${details}parse:PASS "
        else
            details="${details}parse:FAIL "
            errors="${errors}
'miniswe ${flag_name} \"test prompt\" --help' exited with code ${flag_exit}."
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
    fi

    # Check 6: smoke test — run with override prompt, check predictable output
    (( ++checks ))
    if [ -f "${binary}" ] && [ -n "${flag_name}" ] && [ "$passed" -ge 4 ]; then
        local smoke_output
        local smoke_exit=0

        # The override prompt tells the model to respond with exactly "PONG_42"
        smoke_output=$(cd "${work_dir}" && timeout 120 "${binary}" \
            ${flag_name} "You must respond with exactly the text PONG_42 and nothing else. No explanation, no formatting, just PONG_42." \
            --yes "ping" 2>/dev/null) || smoke_exit=$?

        echo "${smoke_output}" > "${attempt_dir}/smoke_output.txt"
        echo "smoke_exit=${smoke_exit}" >> "${attempt_dir}/smoke_output.txt"

        if echo "${smoke_output}" | grep -q "PONG_42"; then
            (( ++passed ))
            details="${details}smoke:PASS "
        else
            details="${details}smoke:FAIL "
            errors="${errors}
SMOKE TEST: ran with override prompt 'respond with PONG_42' but output does not contain PONG_42.
Output: $(echo "${smoke_output}" | head -5)"
        fi
    else
        details="${details}smoke:SKIP "
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
