#!/usr/bin/env bash
# bench-task-B — Benchmark: "Add --max-rounds CLI flag"
#
# Task requires wiring a CLI arg through config → agent loops (4 files).
# Tests cross-module navigation where repo_map and profile should help.
#
# Validation checks:
#   1. Does it compile? (cargo check)
#   2. Does cli/mod.rs have a max_rounds arg?
#   3. Is it referenced in run.rs?
#   4. Is it referenced in repl.rs?
#
# Usage:
#   ./scripts/bench-task-B-max-rounds-flag.sh [--runs 3] [--timeout 300]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/bench-common.sh"

TASK_NAME="task_B_max_rounds_flag"
TASK="Add a --max-rounds CLI flag (clap arg, short form: -r, type: Option<u32>) to miniswe that overrides context.max_rounds from config.toml when provided. Steps: 1) Add the arg to Cli struct in src/cli/mod.rs. 2) In src/main.rs or wherever config is passed to commands, apply the override if the flag is Some. 3) Make sure both run.rs and repl.rs use the overridden value. The flag should appear in --help output."

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

    # Check 2: cli/mod.rs has max_rounds arg
    ((checks++))
    if grep -q 'max.rounds\|max_rounds' "${work_dir}/src/cli/mod.rs" 2>/dev/null; then
        ((passed++))
        details="${details}cli_arg:PASS "
    else
        details="${details}cli_arg:FAIL "
    fi

    # Check 3: Referenced in run.rs
    ((checks++))
    if grep -q 'max.rounds\|max_rounds' "${work_dir}/src/cli/commands/run.rs" 2>/dev/null; then
        ((passed++))
        details="${details}run.rs:PASS "
    else
        details="${details}run.rs:FAIL "
    fi

    # Check 4: Referenced in repl.rs
    ((checks++))
    if grep -q 'max.rounds\|max_rounds' "${work_dir}/src/cli/commands/repl.rs" 2>/dev/null; then
        ((passed++))
        details="${details}repl.rs:PASS "
    else
        details="${details}repl.rs:FAIL "
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
