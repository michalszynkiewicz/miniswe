#!/usr/bin/env bash
# Start llama.cpp server with Devstral Small 2 for miniswe.
#
# Hardware target: RTX 3090 (24GB VRAM, ~21GB usable) + 128GB RAM
# Model: UD-Q4_K_XL (~15GB) + Q8_0 KV cache (~3.1GB at 50K ctx) = ~18GB VRAM
#
# Download the model first (Unsloth Dynamic quant — best quality per bit):
#   mkdir -p models
#   hf download unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF \
#     --include "Devstral-Small-2-24B-Instruct-2512-UD-Q4_K_XL.gguf" \
#     --local-dir models/

set -euo pipefail

MODEL="${MINISWE_MODEL:-models/Devstral-Small-2-24B-Instruct-2512-UD-Q4_K_XL.gguf}"
PORT="${MINISWE_PORT:-8464}"
CTX_SIZE="${MINISWE_CTX_SIZE:-50000}"
THREADS="${MINISWE_THREADS:-8}"

if [ ! -f "$MODEL" ]; then
    echo "Model not found: $MODEL"
    echo ""
    echo "Download it with:"
    echo "  mkdir -p models"
    echo "  hf download unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF \\"
    echo "    --include 'Devstral-Small-2-24B-Instruct-2512-UD-Q4_K_XL.gguf' \\"
    echo "    --local-dir models/"
    echo ""
    echo "Or set MINISWE_MODEL to point to your GGUF file."
    exit 1
fi

echo "Starting Devstral Small 2 for miniswe..."
echo "  Model:   $MODEL"
echo "  Context: $CTX_SIZE tokens"
echo "  KV:      Q8_0"
echo "  Port:    $PORT"
echo ""

exec llama-server \
    --model "$MODEL" \
    --ctx-size "$CTX_SIZE" \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --n-gpu-layers 99 \
    --flash-attn on \
    --threads "$THREADS" \
    --port "$PORT" \
    --metrics
