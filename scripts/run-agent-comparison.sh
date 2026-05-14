#!/usr/bin/env bash
# run-agent-comparison.sh — End-to-end comparison harness.
#
# For each MODEL in the list:
#   1. Stop any existing llama-server container (clean slate).
#   2. Start ./start-${MODEL}.sh in the background. The container is
#      named `llama-server-bench` so we can stop it deterministically.
#   3. Wait for /v1/models to respond. Bail if it takes too long.
#   4. Warm-up ping (one tiny completion) so the first agent doesn't
#      eat the cold-cache penalty.
#   5. Run miniswe bench → benchmark_results/comparison_<ts>/${MODEL}/miniswe/
#   6. Run aider bench   → benchmark_results/comparison_<ts>/${MODEL}/aider/
#   7. Stop llama-server, confirm port 8464 is free.
# Then aggregate a summary.tsv.
#
# Usage:
#   ./scripts/run-agent-comparison.sh                    # quick shape check
#   ./scripts/run-agent-comparison.sh --full             # full timeouts
#   ./scripts/run-agent-comparison.sh --models qwen3-coder-next,devstral-small-2
#
# Time budget:
#   --quick (default): 1 attempt × 1800s per agent  → ~1h per model → ~3h total
#   --full:            3 attempts × 3400s per agent → ~6h per model → ~18h total

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_ENDPOINT="${LLAMA_ENDPOINT:-http://localhost:8464}"
LLAMA_CONTAINER="llama-server-bench"

# Defaults
MODELS="qwen3-coder-next,devstral-small-2,gemma4"
MODE="quick"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models) MODELS="$2"; shift 2 ;;
        --quick)  MODE="quick"; shift ;;
        --full)   MODE="full"; shift ;;
        *) echo "Unknown: $1" >&2; exit 1 ;;
    esac
done

case "$MODE" in
    quick) TIMEOUT=1800; MAX_ATTEMPTS=1 ;;
    full)  TIMEOUT=3400; MAX_ATTEMPTS=3 ;;
esac

TS="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${REPO_DIR}/benchmark_results/comparison_${TS}"
mkdir -p "${RESULTS_DIR}"
SUMMARY="${RESULTS_DIR}/summary.tsv"
printf "model\tagent\tattempts\tscore_max\tscore_total\twall_s\n" > "${SUMMARY}"

echo "=== Agent comparison harness ==="
echo "Models:   ${MODELS}"
echo "Mode:     ${MODE} (timeout=${TIMEOUT}s, max_attempts=${MAX_ATTEMPTS})"
echo "Results:  ${RESULTS_DIR}"
echo ""

# ── llama-server lifecycle helpers ──────────────────────────────────────

stop_llama_server() {
    docker stop "${LLAMA_CONTAINER}" >/dev/null 2>&1 || true
    docker rm   "${LLAMA_CONTAINER}" >/dev/null 2>&1 || true
    # Wait for port to free
    for _ in {1..30}; do
        if ! curl -fsS --max-time 1 "${LLAMA_ENDPOINT}/v1/models" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    echo "WARNING: port still in use after stop attempt" >&2
}

start_llama_server() {
    local model="$1"
    local launcher="${REPO_DIR}/start-${model}.sh"
    if [ ! -x "${launcher}" ]; then
        echo "ERROR: launcher not found: ${launcher}" >&2
        return 1
    fi
    echo "  -> launching ${launcher}"
    # Override the container name in run-llama-cuda.sh so we can stop it
    # deterministically. Run in background; redirect output to a log.
    LLAMA_CONTAINER_NAME="${LLAMA_CONTAINER}" \
        nohup "${launcher}" > "${RESULTS_DIR}/${model}_llama-server.log" 2>&1 &
    # Don't track $! here because nohup forks; the docker container name is
    # what we'll stop with.
}

wait_for_server_ready() {
    local timeout_s=${1:-180}
    local elapsed=0
    while [ "$elapsed" -lt "$timeout_s" ]; do
        if curl -fsS --max-time 2 "${LLAMA_ENDPOINT}/v1/models" >/dev/null 2>&1; then
            echo "  -> server up (after ${elapsed}s)"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    echo "ERROR: server didn't come up within ${timeout_s}s" >&2
    return 1
}

warmup_completion() {
    # Single tiny completion so the first real bench request isn't slowed
    # by lazy-loaded weights / cold KV cache buffers.
    curl -fsS --max-time 60 "${LLAMA_ENDPOINT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d '{"model":"local","messages":[{"role":"user","content":"hi"}],"max_tokens":4}' \
        >/dev/null 2>&1 || true
}

# ── Per-model run ──────────────────────────────────────────────────────

run_model() {
    local model="$1"
    local model_dir="${RESULTS_DIR}/${model}"
    mkdir -p "${model_dir}"

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  MODEL: ${model}"
    echo "════════════════════════════════════════════════════════════════"

    stop_llama_server
    start_llama_server "${model}" || return 1
    if ! wait_for_server_ready 240; then
        echo "  -> giving up on ${model}"
        return 1
    fi
    warmup_completion

    # ── miniswe bench ────────────────────────────────────────────────────
    local miniswe_results_root_before
    miniswe_results_root_before=$(ls -1d "${REPO_DIR}/benchmark_results/docker_"* 2>/dev/null | wc -l)

    echo ""
    echo "── miniswe ──"
    "${REPO_DIR}/scripts/run-benchmark-docker.sh" \
        --timeout "${TIMEOUT}" \
        --max-rounds 600 \
        --max-attempts "${MAX_ATTEMPTS}" \
        --model "${model}" \
        || true

    # The miniswe bench writes to its own timestamped dir. Move the latest
    # one under our comparison dir so everything is in one place.
    local newest_miniswe
    newest_miniswe=$(ls -1td "${REPO_DIR}/benchmark_results/docker_"* 2>/dev/null | head -1)
    if [ -n "${newest_miniswe}" ] && \
       [ "$(ls -1d "${REPO_DIR}/benchmark_results/docker_"* 2>/dev/null | wc -l)" -gt "${miniswe_results_root_before}" ]; then
        mv "${newest_miniswe}" "${model_dir}/miniswe"
    fi

    # ── aider bench ──────────────────────────────────────────────────────
    local aider_results_root_before
    aider_results_root_before=$(ls -1d "${REPO_DIR}/benchmark_results/aider_"* 2>/dev/null | wc -l)

    echo ""
    echo "── aider ──"
    "${REPO_DIR}/scripts/run-aider-bench.sh" \
        --timeout "${TIMEOUT}" \
        --max-attempts "${MAX_ATTEMPTS}" \
        --model "${model}" \
        || true

    local newest_aider
    newest_aider=$(ls -1td "${REPO_DIR}/benchmark_results/aider_"* 2>/dev/null | head -1)
    if [ -n "${newest_aider}" ] && \
       [ "$(ls -1d "${REPO_DIR}/benchmark_results/aider_"* 2>/dev/null | wc -l)" -gt "${aider_results_root_before}" ]; then
        mv "${newest_aider}" "${model_dir}/aider"
    fi

    stop_llama_server

    # ── record into summary.tsv ──────────────────────────────────────────
    for agent in miniswe aider; do
        local agent_dir="${model_dir}/${agent}/00_baseline"
        [ -d "${agent_dir}" ] || continue
        local container_log="${agent_dir}/container.log"
        [ -f "${container_log}" ] || continue

        local final_line score score_total attempts wall_s
        final_line=$(grep "=== FINAL:" "${container_log}" 2>/dev/null | tail -1)
        score=$(echo "${final_line}" | grep -oE "[0-9]+/[0-9]+" | head -1 | cut -d/ -f1)
        score_total=$(echo "${final_line}" | grep -oE "[0-9]+/[0-9]+" | head -1 | cut -d/ -f2)
        attempts=$(grep -c "=== ATTEMPT .* remaining" "${container_log}" 2>/dev/null || echo "0")
        wall_s=$(cat "${agent_dir}/wall_s.txt" 2>/dev/null || echo "?")

        printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
            "${model}" "${agent}" "${attempts:-0}" \
            "${score:-0}" "${score_total:-?}" "${wall_s:-?}" >> "${SUMMARY}"
    done
}

# ── Main loop ───────────────────────────────────────────────────────────

IFS=',' read -ra MODEL_LIST <<< "${MODELS}"
for model in "${MODEL_LIST[@]}"; do
    run_model "${model}"
done

# Defensive: make sure the server is stopped at the end.
stop_llama_server

# ── Summary ─────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  COMPARISON SUMMARY"
echo "════════════════════════════════════════════════════════════════"
column -t -s $'\t' "${SUMMARY}"
echo ""
echo "Detailed results: ${RESULTS_DIR}/"
echo "TSV: ${SUMMARY}"
