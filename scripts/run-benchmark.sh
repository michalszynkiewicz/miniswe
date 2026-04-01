#!/usr/bin/env bash
# run-benchmark.sh — Build and run provider benchmarks in Docker.
#
# Usage:
#   ./scripts/run-benchmark.sh B              # task B only
#   ./scripts/run-benchmark.sh C              # task C only
#   ./scripts/run-benchmark.sh all            # both tasks
#   ./scripts/run-benchmark.sh B --runs 3     # 3 runs per variant
#
# Or run locally without Docker:
#   ./scripts/bench-task-B-max-rounds-flag.sh
#   ./scripts/bench-task-C-token-logging.sh
#
# LLM server must be running on the host (localhost:8464).
# Docker uses --network=host to reach it.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="${REPO_DIR}/benchmark_results"
IMAGE_NAME="miniswe-bench"

TASK="${1:-all}"
shift || true

echo "=== Building benchmark image ==="
docker build \
    -f "${REPO_DIR}/scripts/Dockerfile.benchmark" \
    -t "${IMAGE_NAME}" \
    "${REPO_DIR}"

echo ""
echo "=== Running benchmark: ${TASK} ==="
mkdir -p "${RESULTS_DIR}"

docker run --rm \
    --network=host \
    -v "${RESULTS_DIR}:/results" \
    "${IMAGE_NAME}" \
    "${TASK}" \
    "$@"

echo ""
echo "Results: ${RESULTS_DIR}/"
