#!/usr/bin/env bash
# Cycle gemma4, devstral-small-2, qwen3-coder-next through the simple-tier
# modality probe. Reuses run-agent-comparison.sh's proven llama lifecycle.
set -uo pipefail

REPO="/home/michal/dev/miniswe"
LLAMA_CONTAINER="llama-server-bench"
ENDPOINT="http://localhost:8464"
WORK="/tmp/simple-tier-probe"
export TRIALS=6 MAX_ROUNDS=7

MODELS=("gemma4" "devstral-small-2" "qwen3-coder-next")

stop_server() {
    docker stop "$LLAMA_CONTAINER" >/dev/null 2>&1 || true
    docker rm   "$LLAMA_CONTAINER" >/dev/null 2>&1 || true
    for _ in $(seq 1 30); do
        curl -fsS --max-time 1 "$ENDPOINT/v1/models" >/dev/null 2>&1 || return 0
        sleep 1
    done
    echo "WARN: port still in use after stop" >&2
}
trap stop_server EXIT INT TERM

wait_ready() {
    local t=0 lim=${1:-360}
    while [ "$t" -lt "$lim" ]; do
        if curl -fsS --max-time 2 "$ENDPOINT/v1/models" >/dev/null 2>&1; then
            echo "  server up after ${t}s ($(curl -fsS --max-time 2 "$ENDPOINT/v1/models" \
                | python3 -c 'import json,sys;print((json.load(sys.stdin).get("data") or [{}])[0].get("id","?"))' 2>/dev/null))"
            return 0
        fi
        sleep 5; t=$((t+5))
    done
    echo "  ERROR: server not ready in ${lim}s" >&2
    return 1
}

for model in "${MODELS[@]}"; do
    echo "════════ MODEL: $model ════════"
    stop_server
    launcher="$REPO/start-${model}.sh"
    if [ ! -x "$launcher" ]; then echo "  no launcher $launcher, skip"; continue; fi
    LLAMA_CONTAINER_NAME="$LLAMA_CONTAINER" \
        nohup "$launcher" > "$WORK/${model}_server.log" 2>&1 &
    if ! wait_ready 420; then
        echo "  giving up on $model"; tail -5 "$WORK/${model}_server.log" 2>/dev/null; continue
    fi
    # warmup
    curl -fsS --max-time 60 "$ENDPOINT/v1/chat/completions" -H 'Content-Type: application/json' \
        -d '{"model":"local","messages":[{"role":"user","content":"hi"}],"max_tokens":4}' >/dev/null 2>&1 || true
    rm -f "$WORK/results.jsonl" "$WORK/run.log"
    echo "  running probe (TRIALS=$TRIALS MAX_ROUNDS=$MAX_ROUNDS)..."
    python3 "$WORK/harness.py" > "$WORK/run.log" 2>&1
    cp "$WORK/results.jsonl" "$WORK/results_${model}.jsonl" 2>/dev/null || true
    cp "$WORK/run.log" "$WORK/run_${model}.log" 2>/dev/null || true
    echo "  --- $model summary ---"
    tail -4 "$WORK/run_${model}.log"
    stop_server
done

echo "════════ ALL DONE ════════"
for model in "${MODELS[@]}"; do
    echo "── $model ──"
    tail -4 "$WORK/run_${model}.log" 2>/dev/null || echo "  (no results)"
done
