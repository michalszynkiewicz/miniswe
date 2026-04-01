#!/usr/bin/env bash
# docker-entrypoint.sh — Run one or both benchmark tasks.
# Usage: docker run ... miniswe-bench [B|C|all] [--runs N] [--timeout S] ...
set -euo pipefail

TASK="${1:-all}"
shift || true

COMMON_ARGS=(--project-dir /repo --results-dir /results "$@")

case "${TASK}" in
    B|b)
        echo "=== Running Task B: --max-rounds CLI flag ==="
        /bench/scripts/bench-task-B-max-rounds-flag.sh "${COMMON_ARGS[@]}"
        ;;
    C|c)
        echo "=== Running Task C: per-round token logging ==="
        /bench/scripts/bench-task-C-token-logging.sh "${COMMON_ARGS[@]}"
        ;;
    all|ALL)
        echo "=== Running Task B: --max-rounds CLI flag ==="
        /bench/scripts/bench-task-B-max-rounds-flag.sh "${COMMON_ARGS[@]}"
        echo ""
        echo "========================================================"
        echo ""
        echo "=== Running Task C: per-round token logging ==="
        /bench/scripts/bench-task-C-token-logging.sh "${COMMON_ARGS[@]}"
        ;;
    *)
        echo "Unknown task: ${TASK}. Use B, C, or all." >&2
        exit 1
        ;;
esac
