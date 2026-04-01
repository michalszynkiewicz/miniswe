#!/usr/bin/env bash
# Start llama.cpp server with Qwen 3 235B for miniswe planning/reasoning.
#
# Hardware target: AMD Ryzen 9 9950X3D (16C/32T) + 128GB RAM — CPU only
# Model: UD-Q3_K_XL (~97GB across 3 shards) — too large for 24GB VRAM
# KV cache: Q4_0 to save RAM (~97GB model + KV must fit in 128GB)
#
# Download the model first (Unsloth Dynamic quant):
#   mkdir -p models/qwen3-235b
#   hf download unsloth/Qwen3-235B-A22B-Thinking-2507-GGUF \
#     --include "UD-Q3_K_XL/*" \
#     --local-dir models/qwen3-235b/

set -euo pipefail

MODEL_DIR="${MINISWE_QWEN_DIR:-$HOME/models/qwen3-235b/UD-Q3_K_XL}"
MODEL="${MODEL_DIR}/Qwen3-235B-A22B-Thinking-2507-UD-Q3_K_XL-00001-of-00003.gguf"
PORT="${MINISWE_QWEN_PORT:-8465}"
CTX_SIZE="${MINISWE_QWEN_CTX:-32768}"
THREADS="${MINISWE_QWEN_THREADS:-16}"

if [ ! -f "$MODEL" ]; then
    echo "Model not found: $MODEL"
    echo ""
    echo "Download it with:"
    echo "  mkdir -p models/qwen3-235b"
    echo "  hf download unsloth/Qwen3-235B-A22B-Thinking-2507-GGUF \\"
    echo "    --include 'UD-Q3_K_XL/*' \\"
    echo "    --local-dir models/qwen3-235b/"
    echo ""
    echo "Or set MINISWE_QWEN_DIR to point to your UD-Q3_K_XL directory."
    exit 1
fi

echo "Starting Qwen 3 235B for miniswe (planning/reasoning)..."
echo "  Model:   $MODEL"
echo "  Shards:  3 (auto-detected from first shard)"
echo "  Context: $CTX_SIZE tokens"
echo "  KV:      Q4_0 (RAM-saving mode)"
echo "  Threads: $THREADS"
echo "  Port:    $PORT"
echo "  GPU:     none (CPU only — ~97GB in RAM)"
echo ""

exec llama-server \
    --model "$MODEL" \
    --model-draft "$HOME/models/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf" \
    --n-gpu-layers-draft 99 \
    --draft-max 16 \
    --ctx-size "$CTX_SIZE" \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    --n-gpu-layers 15 \
    --threads "$THREADS" \
    --threads-batch 32 \
    --port "$PORT" \
    --batch-size 4096 \
    --mlock \
    --flash-attn on \
    --metrics
