#!/usr/bin/env bash
# Start llama-server with Mistral Small 3.1 24B Instruct for miniswe.
#
# Hardware target: RTX 3090 (24GB VRAM, ~21GB usable) + 128GB RAM
# Model: Mistral-Small-3.1-24B-Instruct (dense, 24B params, 128K max context)
# Useful as a non-Devstral Mistral baseline — same chat template family
# (with [TOOL_CALLS] / [ARGS]) but a different post-training mix, so it
# helps separate "the leak / parser bug is Mistral-tool-format-related"
# from "it's specifically Devstral Small 2."
#
# VRAM budget at UD-Q4_K_XL + q4_0 KV cache + flash-attn:
#   weights              ~14 GB
#   KV cache (60K ctx)   ~2-3 GB
#   buffers              ~1 GB
#   total                ~17-18 GB  (plenty of headroom)
#
# Smaller quants if you want more KV ctx: Q4_K_M, IQ4_XS.
#
# Download the model first:
#   mkdir -p $HOME/models
#   hf download unsloth/Mistral-Small-3.1-24B-Instruct-2503-GGUF \
#     --include "*UD-Q4_K_XL*" \
#     --local-dir $HOME/models/Mistral-Small-3.1-24B-Instruct-GGUF

set -euo pipefail

MODEL_DIR="${MINISWE_MODEL_DIR:-$HOME/models/Mistral-Small-3.1-24B-Instruct-GGUF}"
PORT="${MINISWE_PORT:-8464}"
CTX_SIZE="${MINISWE_CTX_SIZE:-60000}"
THREADS="${MINISWE_THREADS:-8}"

MODEL="${MINISWE_MODEL:-}"
if [ -z "$MODEL" ]; then
    MODEL=$(ls "$MODEL_DIR"/Mistral-Small-3.1-24B-Instruct-*UD-Q4_K_XL*.gguf 2>/dev/null | head -1 || true)
    [ -z "$MODEL" ] && MODEL=$(ls "$MODEL_DIR"/Mistral-Small-3.1-24B-Instruct-*Q4_K_M*.gguf 2>/dev/null | head -1 || true)
    # Sharded downloads ship as *-00001-of-0000N.gguf; llama.cpp follows the rest.
    [ -z "$MODEL" ] && MODEL=$(ls "$MODEL_DIR"/Mistral-Small-3.1-24B-Instruct-*Q4*-00001-of-*.gguf 2>/dev/null | head -1 || true)
fi

if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "Model not found under $MODEL_DIR" >&2
    echo "" >&2
    echo "Download it with:" >&2
    echo "  hf download unsloth/Mistral-Small-3.1-24B-Instruct-2503-GGUF \\" >&2
    echo "    --include '*UD-Q4_K_XL*' \\" >&2
    echo "    --local-dir $MODEL_DIR" >&2
    exit 1
fi

echo "Starting Mistral Small 3.1 24B for miniswe..."
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
    --temp 0.15 \
    -np 1 \
    --port "$PORT" \
    --metrics
