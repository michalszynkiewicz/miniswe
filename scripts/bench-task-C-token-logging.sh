#!/usr/bin/env bash
# bench-task-C — Benchmark: "Add per-round token usage logging"
#
# Validates by building, then checking the binary's behavior would include
# token logging (we can't easily trigger a real LLM call, so we verify the
# code structurally after confirming it compiles and builds).
#
# Validation:
#   1. cargo check — does it compile?
#   2. cargo build — can we get a binary?
#   3. logging.rs has a token/usage method
#   4. llm/mod.rs or llm/types.rs parses usage fields
#   5. Agent loop (run.rs or repl.rs) calls the new logging
#
# Usage:
#   ./scripts/bench-task-C-token-logging.sh [--runs 3] [--timeout 600]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/bench-common.sh"

TASK_NAME="task_C_token_logging"
TASK="Add per-round token usage logging. After each LLM API call, log the number of prompt tokens, completion tokens, and cumulative totals for the session. Steps: 1) In src/llm/mod.rs, parse the 'usage' field from the OpenAI-compatible API response (it contains prompt_tokens, completion_tokens, total_tokens) — both in chat() and chat_stream(). Return it alongside the response. 2) Add a new method to SessionLog in src/logging.rs like token_usage(prompt_tokens, completion_tokens, cumulative_total) that logs at info level as [tokens]. 3) Call it from the agent loop in run.rs and/or repl.rs after each LLM call. Use a running counter for cumulative totals."

# ── Validation ──────────────────────────────────────────────────────────

validate_result() {
    local attempt_dir="$1"
    local work_dir="$2"
    local errors=""
    local passed=0
    local checks=0
    local details=""

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
        errors="${errors}
BUILD: skipped (compile failed)"
    fi

    # Check 3: logging.rs has token/usage method
    (( ++checks ))
    if grep -qE 'fn\s+(token_usage|log_tokens|tokens_used|usage_log)' "${work_dir}/src/logging.rs" 2>/dev/null ||
       grep -qE '\[tokens\]' "${work_dir}/src/logging.rs" 2>/dev/null; then
        (( ++passed ))
        details="${details}log_method:PASS "
    else
        details="${details}log_method:FAIL "
        errors="${errors}
MISSING: src/logging.rs needs a method for logging token usage (e.g. fn token_usage(...) that writes a [tokens] log line)."
    fi

    # Check 4: llm parses usage from response
    (( ++checks ))
    if grep -qE 'usage|prompt_tokens|completion_tokens' "${work_dir}/src/llm/mod.rs" 2>/dev/null ||
       grep -qE 'usage|prompt_tokens|completion_tokens' "${work_dir}/src/llm/types.rs" 2>/dev/null; then
        (( ++passed ))
        details="${details}llm_parse:PASS "
    else
        details="${details}llm_parse:FAIL "
        errors="${errors}
MISSING: src/llm/mod.rs or src/llm/types.rs needs to parse the 'usage' field from the OpenAI API response (prompt_tokens, completion_tokens, total_tokens)."
    fi

    # Check 5: Agent loop calls the logging
    (( ++checks ))
    local found_in_loop=false
    if grep -qE 'token_usage|log_tokens|tokens_used|\.usage' "${work_dir}/src/cli/commands/run.rs" 2>/dev/null; then
        found_in_loop=true
    fi
    if grep -qE 'token_usage|log_tokens|tokens_used|\.usage' "${work_dir}/src/cli/commands/repl.rs" 2>/dev/null; then
        found_in_loop=true
    fi
    if $found_in_loop; then
        (( ++passed ))
        details="${details}loop_call:PASS "
    else
        details="${details}loop_call:FAIL "
        errors="${errors}
MISSING: The agent loop in src/cli/commands/run.rs or repl.rs needs to call the token usage logging after each LLM call."
    fi

    # Verdict
    local verdict="FAIL"
    if [ "${passed}" -eq "${checks}" ]; then
        verdict="PASS"
    elif [ "${passed}" -ge 3 ]; then
        verdict="PARTIAL"
    fi

    echo "${verdict}" > "${attempt_dir}/validation.txt"
    echo "${passed}/${checks} ${details}" > "${attempt_dir}/validation_details.txt"
    echo "${errors}" > "${attempt_dir}/validation_errors.txt"
    echo "    validation: ${verdict} (${passed}/${checks}) ${details}"
}

# ── Run ─────────────────────────────────────────────────────────────────

run_benchmark "$@"
