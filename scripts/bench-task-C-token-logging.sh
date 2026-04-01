#!/usr/bin/env bash
# bench-task-C — Benchmark: "Add per-round token usage logging"
#
# Task requires understanding SSE streaming, logging system, and agent loop
# data flow across 4+ modules. Hardest task — most likely to differentiate
# providers.
#
# Validation checks:
#   1. Does it compile? (cargo check)
#   2. Does logging.rs have a token/usage logging method?
#   3. Does llm/mod.rs parse usage/token fields from API response?
#   4. Is the new logging called from run.rs or repl.rs?
#
# Usage:
#   ./scripts/bench-task-C-token-logging.sh [--runs 3] [--timeout 300]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/bench-common.sh"

TASK_NAME="task_C_token_logging"
TASK="Add per-round token usage logging. After each LLM API call, log the number of prompt tokens, completion tokens, and cumulative totals for the session. Steps: 1) In src/llm/mod.rs, parse the 'usage' field from the OpenAI-compatible API response (it contains prompt_tokens, completion_tokens, total_tokens) — both in chat() and chat_stream(). Return it alongside the response. 2) Add a new method to SessionLog in src/logging.rs like token_usage(prompt_tokens, completion_tokens, cumulative_total) that logs at info level as [tokens]. 3) Call it from the agent loop in run.rs and/or repl.rs after each LLM call. Use a running counter for cumulative totals."

# ── Validation ──────────────────────────────────────────────────────────

validate_result() {
    local run_dir="$1"
    local work_dir="$2"
    local checks=0
    local passed=0
    local details=""

    # Check 1: Does it compile?
    ((checks++))
    if (cd "${work_dir}" && cargo check 2>"${run_dir}/cargo_check.txt"); then
        ((passed++))
        details="${details}compile:PASS "
    else
        details="${details}compile:FAIL "
    fi

    # Check 2: logging.rs has token/usage logging method
    ((checks++))
    if grep -qE 'fn\s+(token_usage|log_tokens|tokens_used|usage)' "${work_dir}/src/logging.rs" 2>/dev/null; then
        ((passed++))
        details="${details}log_method:PASS "
    elif grep -qE '\[tokens\]|token.*usage|prompt_tokens' "${work_dir}/src/logging.rs" 2>/dev/null; then
        ((passed++))
        details="${details}log_method:PASS(pattern) "
    else
        details="${details}log_method:FAIL "
    fi

    # Check 3: llm/mod.rs parses usage from API response
    ((checks++))
    if grep -qE 'usage|prompt_tokens|completion_tokens' "${work_dir}/src/llm/mod.rs" 2>/dev/null; then
        ((passed++))
        details="${details}llm_parse:PASS "
    elif grep -qE 'usage|prompt_tokens|completion_tokens' "${work_dir}/src/llm/types.rs" 2>/dev/null; then
        ((passed++))
        details="${details}llm_parse:PASS(types) "
    else
        details="${details}llm_parse:FAIL "
    fi

    # Check 4: Agent loop calls the new logging
    ((checks++))
    local found_in_loop=false
    if grep -qE 'token_usage|log_tokens|tokens_used|\.usage' "${work_dir}/src/cli/commands/run.rs" 2>/dev/null; then
        found_in_loop=true
    fi
    if grep -qE 'token_usage|log_tokens|tokens_used|\.usage' "${work_dir}/src/cli/commands/repl.rs" 2>/dev/null; then
        found_in_loop=true
    fi
    if $found_in_loop; then
        ((passed++))
        details="${details}loop_call:PASS "
    else
        details="${details}loop_call:FAIL "
    fi

    # Overall verdict
    local verdict="FAIL"
    if [ "${passed}" -eq "${checks}" ]; then
        verdict="PASS"
    elif [ "${passed}" -ge 2 ]; then
        verdict="PARTIAL"
    fi

    echo "${verdict}" > "${run_dir}/validation.txt"
    echo "${passed}/${checks} ${details}" > "${run_dir}/validation_details.txt"
    echo "    validation: ${verdict} (${passed}/${checks}) ${details}"
}

# ── Run ─────────────────────────────────────────────────────────────────

run_benchmark "$@"
