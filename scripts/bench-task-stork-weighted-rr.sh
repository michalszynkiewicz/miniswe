#!/usr/bin/env bash
# bench-task-stork — Benchmark: "Add weighted round-robin load balancer to smallrye-stork"
#
# Tests on a real 250-file Java project where navigation should matter.
# The model must discover the LoadBalancer SPI pattern across multiple
# modules and create a new load balancer following existing conventions.
#
# Validation:
#   1. mvn compile — does it compile?
#   2. WeightedRoundRobin*.java files exist
#   3. @LoadBalancerType annotation present
#   4. LoadBalancer interface implemented
#   5. pom.xml for new module exists
#
# Usage:
#   ./scripts/bench-task-stork-weighted-rr.sh --strategy bisect --timeout 1800

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/bench-common.sh"

TASK_NAME="task_stork_weighted_rr"
TASK="Add a weighted round-robin load balancer to this project. Each service instance should have a configurable weight (integer), and the load balancer should distribute requests proportionally to those weights. Follow the same patterns as the existing load balancer implementations (look at the random or least-requests modules for reference). Create a new Maven module under load-balancer/, implement the LoadBalancer and LoadBalancerProvider interfaces, and use the @LoadBalancerType annotation for registration."

# Stork repo
STORK_REPO="https://github.com/smallrye/smallrye-stork.git"
STORK_SHA="a8293e26a21212a07134b2a511f24cc8bc7d07a7"

# Override prepare_workdir for stork (clone instead of git archive)
prepare_workdir() {
    rm -rf "${WORK_DIR}/src" "${WORK_DIR}/pom.xml"
    cd "${WORK_DIR}"
    # Remove everything except .git
    git rm -rf --quiet . 2>/dev/null || true
    cd - > /dev/null

    # Clone stork at pinned SHA (shallow for speed)
    if [ ! -d "${WORK_DIR}/.stork-cache" ]; then
        git clone --depth 50 "${STORK_REPO}" "${WORK_DIR}/.stork-cache" 2>/dev/null
    fi
    # Copy from cache to work dir
    cd "${WORK_DIR}/.stork-cache"
    git checkout "${STORK_SHA}" 2>/dev/null || git checkout main 2>/dev/null
    cd - > /dev/null

    # Copy source tree (excluding .git of cache)
    rsync -a --exclude='.git' --exclude='.stork-cache' "${WORK_DIR}/.stork-cache/" "${WORK_DIR}/"

    # Init miniswe
    rm -rf "${WORK_DIR}/.miniswe"
    cd "${WORK_DIR}"
    "${MINISWE}" init 2>/dev/null || true
    cd - > /dev/null
    mkdir -p "${WORK_DIR}/.miniswe/logs"

    cd "${WORK_DIR}"
    git add -A && git commit -q --allow-empty -m "stork at ${STORK_SHA}" 2>/dev/null || true
    cd - > /dev/null
}

# Override init_workdir too
init_workdir() {
    WORK_DIR=$(mktemp -d "/tmp/miniswe-bench-XXXXXX")
    mkdir -p "${WORK_DIR}"
    cd "${WORK_DIR}" && git init -q && cd - > /dev/null
    prepare_workdir
}

# ── Validation ──────────────────────────────────────────────────────────

validate_result() {
    local attempt_dir="$1"
    local work_dir="$2"
    local errors=""
    local passed=0
    local checks=0
    local details=""

    # Check 1: mvn compile (just the new module + dependencies)
    (( ++checks ))
    local check_output
    if check_output=$(cd "${work_dir}" && mvn compile -q -pl load-balancer/ -am 2>&1); then
        (( ++passed ))
        details="${details}compile:PASS "
    else
        details="${details}compile:FAIL "
        local err_lines
        err_lines=$(echo "${check_output}" | grep -E '^\[ERROR\]' | head -20)
        errors="${errors}
COMPILE ERROR (mvn compile):
${err_lines}"
    fi
    echo "${check_output}" > "${attempt_dir}/mvn_compile.txt" 2>/dev/null

    # Check 2: WeightedRoundRobin Java files exist
    (( ++checks ))
    local lb_files
    lb_files=$(find "${work_dir}/load-balancer" -name "*eight*ound*obin*.java" -o -name "*weighted*round*robin*.java" -o -name "*WeightedRR*.java" 2>/dev/null | head -5)
    if [ -n "${lb_files}" ]; then
        (( ++passed ))
        details="${details}files:PASS "
    else
        details="${details}files:FAIL "
        errors="${errors}
MISSING: No weighted round-robin Java files found under load-balancer/. Expected files like WeightedRoundRobinLoadBalancer.java."
    fi

    # Check 3: @LoadBalancerType annotation
    (( ++checks ))
    if grep -rq '@LoadBalancerType' "${work_dir}/load-balancer/" --include="*.java" 2>/dev/null | grep -iq "weight"; then
        (( ++passed ))
        details="${details}annotation:PASS "
    elif [ -n "${lb_files}" ] && xargs grep -l '@LoadBalancerType' <<< "${lb_files}" 2>/dev/null | head -1 | grep -q .; then
        (( ++passed ))
        details="${details}annotation:PASS "
    else
        # Broader check — any new @LoadBalancerType in load-balancer/
        local new_annotations
        new_annotations=$(cd "${work_dir}" && git diff --name-only 2>/dev/null | xargs grep -l '@LoadBalancerType' 2>/dev/null | head -1 || true)
        if [ -n "${new_annotations}" ]; then
            (( ++passed ))
            details="${details}annotation:PASS "
        else
            details="${details}annotation:FAIL "
            errors="${errors}
MISSING: No @LoadBalancerType annotation found in new files. The provider must be annotated for registration."
        fi
    fi

    # Check 4: Implements LoadBalancer interface
    (( ++checks ))
    if [ -n "${lb_files}" ] && xargs grep -l 'implements.*LoadBalancer' <<< "${lb_files}" 2>/dev/null | head -1 | grep -q .; then
        (( ++passed ))
        details="${details}interface:PASS "
    elif grep -rq 'implements.*LoadBalancer' "${work_dir}/load-balancer/" --include="*.java" 2>/dev/null; then
        (( ++passed ))
        details="${details}interface:PASS "
    else
        details="${details}interface:FAIL "
        errors="${errors}
MISSING: No class implements the LoadBalancer interface in the new module."
    fi

    # Check 5: New Maven module pom.xml exists
    (( ++checks ))
    local new_pom
    new_pom=$(find "${work_dir}/load-balancer" -mindepth 2 -name "pom.xml" -newer "${work_dir}/load-balancer/pom.xml" 2>/dev/null | head -1)
    if [ -z "${new_pom}" ]; then
        # Check for any pom.xml in a weighted/round-robin directory
        new_pom=$(find "${work_dir}/load-balancer" -path "*eight*" -name "pom.xml" -o -path "*weighted*" -name "pom.xml" 2>/dev/null | head -1)
    fi
    if [ -n "${new_pom}" ]; then
        (( ++passed ))
        details="${details}pom:PASS "
    else
        details="${details}pom:FAIL "
        errors="${errors}
MISSING: No pom.xml found for the new load balancer module. Create a Maven module under load-balancer/."
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
