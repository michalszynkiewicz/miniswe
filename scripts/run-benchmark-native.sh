#!/usr/bin/env bash
# run-benchmark-native.sh — Docker-free benchmark, for a fresh GPU VM.
#
# Same task / config / 6-check validation / best-of-N as run-benchmark-docker.sh,
# but WITHOUT Docker: each run gets a fresh `git archive` checkout in a tempdir,
# driven by a once-built release binary. Isolation comes from the per-run
# tempdir + clean checkout, not a container.
#
# Requires on PATH: cargo/rustc, rust-analyzer (for LSP), git, and an
# OpenAI-compatible LLM server reachable at $LLAMA_ENDPOINT (default :8464).
#
# Usage:
#   scripts/run-benchmark-native.sh --model gemma-4-26B-A4B-it \
#       --timeout 2400 --max-rounds 80 --runs 4
#
# Env knobs (mirror the docker harness):
#   GATE_CONTEXT_RESET (default false — intended ship config)
#   AUTO_REVERT / REACTIVE_DEBUGGER / SPIRAL_RESET (default false)
#   SEED_PATCH=<path>  apply a half-wired seed before the run (optional)
#   LLAMA_ENDPOINT     default http://localhost:8464
set -uo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_ENDPOINT="${LLAMA_ENDPOINT:-http://localhost:8464}"

# Defaults
TIMEOUT=2400
MAX_ROUNDS=80
MAX_ATTEMPTS=3
TEMPERATURE=0.2
RUNS=1
MODEL="gemma-4-26B-A4B-it"
BASELINE_SHA="cc34d2626faf32c1b6dd1b8b33af693fb936b098"
TASK="Add a CLI flag --system-prompt-override (short: -s) that takes a string and replaces the default system prompt with the provided text. When this flag is set, skip all context providers and just use the override text as the system message. Make sure it works for both single-shot and interactive modes."

while [[ $# -gt 0 ]]; do
    case "$1" in
        --timeout)      TIMEOUT="$2";      shift 2 ;;
        --max-rounds)   MAX_ROUNDS="$2";   shift 2 ;;
        --max-attempts) MAX_ATTEMPTS="$2"; shift 2 ;;
        --temperature)  TEMPERATURE="$2";  shift 2 ;;
        --model)        MODEL="$2";        shift 2 ;;
        --runs)         RUNS="$2";         shift 2 ;;
        --task)         TASK="$2";         shift 2 ;;
        --sha)          BASELINE_SHA="$2"; shift 2 ;;
        *) echo "Unknown: $1" >&2; exit 1 ;;
    esac
done

# Resolve optional seed patch to an absolute path.
SEED_PATCH="${SEED_PATCH:-}"
if [[ -n "${SEED_PATCH}" ]]; then
    [[ "${SEED_PATCH}" = /* ]] || SEED_PATCH="${REPO_DIR}/${SEED_PATCH}"
    [[ -f "${SEED_PATCH}" ]] || { echo "SEED_PATCH not found: ${SEED_PATCH}" >&2; exit 1; }
    TASK="${SEED_TASK:-The --system-prompt-override (-s) CLI flag is already partially wired: it parses and the code compiles, but the override text is currently IGNORED. Finish wiring it so the provided text REPLACES the system prompt (skipping all context providers) for both single-shot and interactive modes.}"
fi

MODEL_TAG="$(curl -fsS --max-time 3 "${LLAMA_ENDPOINT}/v1/models" 2>/dev/null \
    | python3 -c "import json,sys;print((json.load(sys.stdin).get('data') or [{}])[0].get('id','?'))" 2>/dev/null \
    | sed -E 's/\.gguf$//; s/[^A-Za-z0-9._-]/_/g' | cut -c1-40)"
MODEL_TAG="${MODEL_TAG:-unknown}"
RESULTS_DIR="${REPO_DIR}/benchmark_results/native_$(date +%Y%m%d_%H%M%S)_${MODEL_TAG}"
mkdir -p "${RESULTS_DIR}"

echo "=== Native (Docker-free) benchmark ==="
echo "Repo:     ${REPO_DIR}"
echo "SHA:      ${BASELINE_SHA}"
echo "Model:    ${MODEL}  (server reports: ${MODEL_TAG})"
echo "Endpoint: ${LLAMA_ENDPOINT}"
echo "Timeout:  ${TIMEOUT}s | Rounds: ${MAX_ROUNDS} | Attempts: ${MAX_ATTEMPTS} | Runs: ${RUNS}"
echo "Seed:     ${SEED_PATCH:-none}"
echo "Results:  ${RESULTS_DIR}"
echo ""

# Preflight
command -v cargo >/dev/null || { echo "ERROR: cargo not on PATH" >&2; exit 1; }
command -v rust-analyzer >/dev/null || echo "WARN: rust-analyzer not on PATH — LSP checks will degrade"
curl -fsS --max-time 5 "${LLAMA_ENDPOINT}/v1/models" >/dev/null 2>&1 \
    || { echo "ERROR: LLM server not responding at ${LLAMA_ENDPOINT}" >&2; exit 1; }

# Build the DRIVER binary once from the current tree (the version under test).
# This is separate from the per-run task workspace, which builds its own
# ./target/debug/miniswe during validation.
echo "Building driver binary (cargo build --release)..."
( cd "${REPO_DIR}" && cargo build --release 2>&1 | tail -3 )
DRIVER="${REPO_DIR}/target/release/miniswe"
[[ -x "${DRIVER}" ]] || { echo "ERROR: driver build failed ($DRIVER missing)" >&2; exit 1; }
echo ""

generate_config() {
cat <<TOML
[model]
provider = "llama-cpp"
endpoint = "${LLAMA_ENDPOINT}"
model = "${MODEL}"
context_window = 60000
temperature = ${TEMPERATURE}
max_output_tokens = 8000

[context]
repo_map_budget = 5000
max_rounds = ${MAX_ROUNDS}
pause_after_rounds = 99999

[context.providers]
repo_map = false

[hardware]
vram_gb = 24.0
vram_reserve_gb = 3.0
ram_budget_gb = 80.0

[lsp]
enabled = true
diagnostic_timeout_ms = 2000

[tools]
auto_revert_ast_cascade = ${AUTO_REVERT:-false}
reactive_debugger = ${REACTIVE_DEBUGGER:-false}
spiral_reset = ${SPIRAL_RESET:-false}
gate_context_reset = ${GATE_CONTEXT_RESET:-false}

[logging]
level = "trace"
enabled = true

[validation]
command = "out=\$(cargo build 2>&1) || { echo \"DOES NOT COMPILE:\"; echo \"\$out\" | tail -20; exit 1; }; run=\$(MINISWE_SKIP_VALIDATION=1 ./target/debug/miniswe --system-prompt-override 'Respond only with TOKEN_XYZ and nothing else' --yes hello 2>&1); echo \"\$run\" | grep -q TOKEN_XYZ || { echo \"COMPILES but override NOT consumed. Expected TOKEN_XYZ, GOT: \$run\"; exit 1; }"
timeout_secs = 180
max_retries = 3
TOML
}

run_one() {
    local idx="$1"
    local vdir="${RESULTS_DIR}/run${idx}"
    mkdir -p "${vdir}"
    local work; work="$(mktemp -d /tmp/miniswe-native-run${idx}.XXXXXX)"

    echo "--- run ${idx} (workdir ${work}) ---"
    # Fresh checkout at the pinned baseline SHA (mirrors the container).
    git -C "${REPO_DIR}" archive "${BASELINE_SHA}" | tar -x -C "${work}"
    ( cd "${work}" && rm -rf target .miniswe )

    # Optional seed patch (half-wired start).
    if [[ -n "${SEED_PATCH}" ]]; then
        if ( cd "${work}" && git -c filter.lfs.smudge=cat -c filter.lfs.process= apply -v "${SEED_PATCH}" 2>/dev/null || git apply -v "${SEED_PATCH}" ); then
            echo "  seed applied"
        else
            echo "  ERROR: seed failed to apply" | tee "${vdir}/SEED_FAILED"
        fi
    fi

    mkdir -p "${work}/.miniswe/logs"
    generate_config > "${work}/.miniswe/config.toml"
    cp "${work}/.miniswe/config.toml" "${vdir}/config.toml"

    ( cd "${work}" && "${DRIVER}" init >/dev/null 2>"${vdir}/miniswe_init.txt" ) \
        || { echo "  init failed"; cat "${vdir}/miniswe_init.txt"; }
    ( cd "${work}" && git init -q && git add -A && git commit -q -m baseline 2>/dev/null )

    local t0; t0=$(date +%s)
    local deadline=$((t0 + TIMEOUT))
    local attempt=0 best=0 current_task="${TASK}"
    : > "${vdir}/run.log"

    while [ "${attempt}" -lt "${MAX_ATTEMPTS}" ]; do
        attempt=$((attempt + 1))
        local now remaining; now=$(date +%s); remaining=$((deadline - now))
        [ "${remaining}" -le 30 ] && { echo "=== ATTEMPT ${attempt}: SKIPPED (${remaining}s left) ===" | tee -a "${vdir}/run.log"; break; }
        echo "=== ATTEMPT ${attempt}/${MAX_ATTEMPTS} (${remaining}s remaining) ===" | tee -a "${vdir}/run.log"

        ( cd "${work}" && timeout "${remaining}" "${DRIVER}" --yes "${current_task}" \
            > "${vdir}/stdout_attempt${attempt}.txt" 2> "${vdir}/stderr_attempt${attempt}.txt" ) || true
        ( cd "${work}" && git diff > "${vdir}/diff_after_attempt${attempt}.patch" 2>/dev/null; git diff > "${vdir}/diff.patch" 2>/dev/null ) || true

        # === 6-check validation ===
        local PASS=0 TOTAL=0 ERRORS="" BINARY="${work}/target/debug/miniswe" FLAG=""
        pushd "${work}" >/dev/null

        TOTAL=$((TOTAL+1))
        if RUSTFLAGS="-A warnings" cargo check 2>"${vdir}/cargo_check.txt"; then echo "compile:PASS" | tee -a "${vdir}/run.log"; PASS=$((PASS+1))
        else echo "compile:FAIL" | tee -a "${vdir}/run.log"; ERRORS="${ERRORS}
COMPILE FAILED:
$(grep -E '^error(\[|:)|^\s*-->' "${vdir}/cargo_check.txt" | head -40)"; fi

        TOTAL=$((TOTAL+1))
        if [ "${PASS}" -ge 1 ] && RUSTFLAGS="-A warnings" cargo build 2>"${vdir}/cargo_build.txt"; then echo "build:PASS" | tee -a "${vdir}/run.log"; PASS=$((PASS+1))
        else echo "build:$([ "${PASS}" -ge 1 ] && echo FAIL || echo SKIP)" | tee -a "${vdir}/run.log"; [ "${PASS}" -ge 1 ] && ERRORS="${ERRORS}
BUILD FAILED:
$(grep -E '^error(\[|:)|^\s*-->' "${vdir}/cargo_build.txt" | head -20)"; fi

        TOTAL=$((TOTAL+1))
        if [ -f "${BINARY}" ]; then "${BINARY}" --help > "${vdir}/help_output.txt" 2>&1 || true
            if grep -qiE -- '--[a-z-]*prompt[a-z-]*' "${vdir}/help_output.txt"; then FLAG=$(grep -oE -- '--[a-z-]*prompt[a-z-]*' "${vdir}/help_output.txt" | head -1); echo "help:PASS(${FLAG})" | tee -a "${vdir}/run.log"; PASS=$((PASS+1))
            else echo "help:FAIL" | tee -a "${vdir}/run.log"; ERRORS="${ERRORS}
HELP FAILED: no --*prompt* flag in --help."; fi
        fi

        TOTAL=$((TOTAL+1))
        if [ -f "${BINARY}" ] && [ -n "${FLAG}" ] && "${BINARY}" ${FLAG} "test" --help >"${vdir}/parse_output.txt" 2>&1; then echo "parse:PASS" | tee -a "${vdir}/run.log"; PASS=$((PASS+1)); else echo "parse:FAIL" | tee -a "${vdir}/run.log"; fi

        TOTAL=$((TOTAL+1))
        if [ "${PASS}" -ge 2 ] && RUSTFLAGS="-A warnings" cargo test >"${vdir}/cargo_test.txt" 2>&1; then echo "test:PASS" | tee -a "${vdir}/run.log"; PASS=$((PASS+1))
        else echo "test:FAIL" | tee -a "${vdir}/run.log"; ERRORS="${ERRORS}
TESTS FAILED:
$(grep -E '^error(\[|:)|^test .* \.\.\. FAILED$|panicked at' "${vdir}/cargo_test.txt" | head -20)"; fi

        TOTAL=$((TOTAL+1))
        if [ -f "${BINARY}" ] && [ -n "${FLAG}" ] && [ "${PASS}" -ge 4 ]; then
            local OUT; OUT=$(MINISWE_SKIP_VALIDATION=1 timeout 120 "${BINARY}" ${FLAG} 'You must respond with exactly the text PONG_42 and nothing else.' --yes ping 2>"${vdir}/smoke_stderr.txt" || true)
            echo "${OUT}" > "${vdir}/smoke_output.txt"
            if echo "${OUT}" | grep -q PONG_42; then echo "smoke:PASS" | tee -a "${vdir}/run.log"; PASS=$((PASS+1)); else echo "smoke:FAIL" | tee -a "${vdir}/run.log"; ERRORS="${ERRORS}
SMOKE FAILED: expected PONG_42, got: $(echo "${OUT}" | head -3)"; fi
        fi
        popd >/dev/null

        echo "=== ATTEMPT ${attempt} RESULT: ${PASS}/${TOTAL} ===" | tee -a "${vdir}/run.log"
        [ "${PASS}" -gt "${best}" ] && best="${PASS}"
        [ "${PASS}" -eq "${TOTAL}" ] && { echo "=== PASSED on attempt ${attempt} ===" | tee -a "${vdir}/run.log"; break; }
        current_task="Your previous changes have these problems:
${ERRORS}
Please fix the issues. The modified files are still on disk."
    done

    echo "=== FINAL: ${best}/6 after ${attempt} attempt(s) ===" | tee -a "${vdir}/run.log"
    echo $(( $(date +%s) - t0 )) > "${vdir}/wall_s.txt"
    # Persist the agent's own logs/plan/scratchpad for later inspection.
    cp -a "${work}/.miniswe" "${vdir}/miniswe_state" 2>/dev/null || true
    rm -rf "${work}"
    echo "    run${idx}: ${best}/6  (wall $(cat "${vdir}/wall_s.txt")s)"
    echo ""
}

for i in $(seq 1 "${RUNS}"); do run_one "${i}"; done

echo "================= NATIVE RESULTS ================="
for d in "${RESULTS_DIR}"/run*/; do
    res=$(grep -oE "=== FINAL: [0-9]+/[0-9]+" "$d/run.log" 2>/dev/null | tail -1 | grep -oE "[0-9]+/[0-9]+" || echo "?/?")
    printf "  %-7s %s  (wall %ss)\n" "$(basename "$d")" "$res" "$(cat "$d/wall_s.txt" 2>/dev/null)"
done
echo "Detailed: ${RESULTS_DIR}/"
