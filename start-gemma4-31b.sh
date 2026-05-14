#!/usr/bin/env bash
# Start llama-server with Gemma 4 31B IT (Q4) for miniswe.
#
# Hardware target: RTX 3090 (24GB VRAM, ~21GB usable) + 128GB RAM
# Model: Gemma 4 31B IT (dense, 30.7B params, 256K max context)
# VRAM budget (verified against Unsloth's 17-20 GB Q4 recommendation):
#   UD-Q4_K_XL weights  18.8 GB
#   q4_0 KV + buffers   ~2   GB  (at 32K ctx, flash-attn on)
#   total               ~21  GB
#
# Download the model first (18.8 GB):
#   mkdir -p $HOME/models
#   hf download unsloth/gemma-4-31B-it-GGUF \
#     --include "*UD-Q4_K_XL*" \
#     --local-dir $HOME/models/gemma-4-31B-it-GGUF
#
# Smaller quants if it OOMs: Q4_K_M (18.3 GB), Q4_K_S (17.4 GB), IQ4_XS (16.4 GB).

set -euo pipefail

MODEL_DIR="${MINISWE_MODEL_DIR:-$HOME/models/gemma-4-31B-it-GGUF}"
PORT="${MINISWE_PORT:-8464}"
CTX_SIZE="${MINISWE_CTX_SIZE:-32768}"
THREADS="${MINISWE_THREADS:-8}"

MODEL="${MINISWE_MODEL:-}"
if [ -z "$MODEL" ]; then
    MODEL=$(ls "$MODEL_DIR"/gemma-4-31B-it-UD-Q4_K_XL*.gguf 2>/dev/null | head -1 || true)
    [ -z "$MODEL" ] && MODEL=$(ls "$MODEL_DIR"/gemma-4-31B-it-Q4_K_M*.gguf 2>/dev/null | head -1 || true)
    # Sharded downloads ship as *-00001-of-0000N.gguf; llama.cpp follows the rest.
    [ -z "$MODEL" ] && MODEL=$(ls "$MODEL_DIR"/gemma-4-31B-it-*Q4*-00001-of-*.gguf 2>/dev/null | head -1 || true)
fi

if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "Model not found under $MODEL_DIR" >&2
    echo "" >&2
    echo "Download it with:" >&2
    echo "  hf download unsloth/gemma-4-31B-it-GGUF \\" >&2
    echo "    --include '*UD-Q4_K_XL*' \\" >&2
    echo "    --local-dir $MODEL_DIR" >&2
    exit 1
fi

echo "Starting Gemma 4 31B IT for miniswe..."
echo "  Model:   $MODEL"
echo "  Context: $CTX_SIZE tokens"
echo "  KV:      q4_0"
echo "  Port:    $PORT"
echo ""

exec "$(dirname "$0")/scripts/run-llama-cuda.sh" \
    --jinja \
    --model "$MODEL" \
    --ctx-size "$CTX_SIZE" \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    --n-gpu-layers 99 \
    --flash-attn on \
    --threads "$THREADS" \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 64 \
    -np 1 \
    --port "$PORT" \
    --metrics
