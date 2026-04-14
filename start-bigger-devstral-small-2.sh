#!/usr/bin/env bash
# Start llama.cpp server with Devstral Small 2 for miniswe.
#
# Hardware target: RTX 3090 (24GB VRAM, ~21GB usable) + 128GB RAM
# Model: Q6_K (~19GB) + q4_0 KV cache (~2GB at 50K ctx) = ~21GB VRAM
# The "bigger" variant trades KV precision (q8_0 → q4_0) for higher-quality
# weights (UD-Q4_K_XL → Q6_K) at the same 50K context.
#
# Download the model first:
#   mkdir -p models
#   hf download unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF \
#     --include "Devstral-Small-2-24B-Instruct-2512-Q6_K.gguf" \
#     --local-dir models/

set -euo pipefail

MODEL="${MINISWE_MODEL:-$HOME/models/devstral-small-2/Devstral-Small-2-24B-Instruct-2512-Q6_K.gguf}"
PORT="${MINISWE_PORT:-8464}"
CTX_SIZE="${MINISWE_CTX_SIZE:-50000}"
THREADS="${MINISWE_THREADS:-8}"

if [ ! -f "$MODEL" ]; then
    echo "Model not found: $MODEL"
    echo ""
    echo "Download it with:"
    echo "  mkdir -p models"
    echo "  hf download unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF \\"
    echo "    --include 'Devstral-Small-2-24B-Instruct-2512-Q6_K.gguf' \\"
    echo "    --local-dir models/"
    echo ""
    echo "Or set MINISWE_MODEL to point to your GGUF file."
    exit 1
fi

echo "Starting Devstral Small 2 for miniswe..."
echo "  Model:   $MODEL"
echo "  Context: $CTX_SIZE tokens"
echo "  KV:      q4_0"
echo "  Port:    $PORT"
echo ""

exec llama-server \
    --model "$MODEL" \
    --ctx-size "$CTX_SIZE" \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    --n-gpu-layers 99 \
    --flash-attn on \
    --threads "$THREADS" \
    --temp 0.15 \
    -np 1 \
    --port "$PORT" \
    --metrics
