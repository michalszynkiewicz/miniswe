#!/bin/bash
# Start llama-server with Gemma 4 26B-A4B MoE for the miniswe benchmark.
#
# First-time setup (downloads ~15 GB):
#   mkdir -p $HOME/models
#   huggingface-cli download unsloth/gemma-4-26B-A4B-it-GGUF \
#       --include "*Q4_K_M*" \
#       --local-dir $HOME/models/gemma-4-26B-A4B-it-GGUF

set -euo pipefail

MODEL_DIR="$HOME/models/gemma-4-26B-A4B-it-GGUF"
# KV cache quantization. Only 6 of 30 layers are full attention (24 are
# sliding, 1024-token window), so the cache is small: at 60K context q4_0 is
# ~0.9 GB, q8_0 ~1.7 GB, f16 ~3.0 GB. Measured 2026-08-22: server at 17.2 GB
# with q4_0 under a ~20 GB budget, so q8_0 fits. q8_0 is the default since
# 2026-08-22 (was q4_0 for every run before that): q4 KV is a suspected
# contributor to the loop pathologies (the cache_prompt cold-prefill hack
# exists for it). Override with MINISWE_KV_TYPE=q4_0 to reproduce old runs.
KV_TYPE="${MINISWE_KV_TYPE:-q8_0}"
MODEL_FILE="$MODEL_DIR/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"

if [ ! -f "$MODEL_FILE" ]; then
    # Handle sharded downloads (larger quants ship as *-00001-of-0000N.gguf)
    SHARD=$(ls "$MODEL_DIR"/gemma-4-26B-A4B-it-Q4_K_M*-00001-of-*.gguf 2>/dev/null | head -1 || true)
    if [ -n "$SHARD" ]; then
        MODEL_FILE="$SHARD"
    else
        echo "Model file not found under $MODEL_DIR" >&2
        echo "Run the huggingface-cli download command from the header comment." >&2
        exit 1
    fi
fi

exec "$(dirname "$0")/scripts/run-llama-cuda.sh" \
    --jinja \
    --reasoning-budget 2000 \
    -m "$MODEL_FILE" \
    -c 60000 \
    --cache-type-k "$KV_TYPE" \
    --cache-type-v "$KV_TYPE" \
    -ngl 99 \
    --flash-attn on \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 64 \
    -np 1 \
    --port 8464
