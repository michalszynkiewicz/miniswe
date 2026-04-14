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

exec llama-server \
    --jinja \
    -m "$MODEL_FILE" \
    --port 8464 \
    -c 60000
