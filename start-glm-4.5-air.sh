#!/usr/bin/env bash
# Start llama-server with GLM-4.5-Air for miniswe.
#
# Hardware target: RTX 3090 (24GB VRAM, ~21GB usable) + 128GB RAM
# Model: GLM-4.5-Air (Zhipu's hybrid MoE, ~12B activated of ~106B total)
# Tops several open-weight coding leaderboards in early 2026; uses a
# distinct chat template from Devstral / Qwen so it's a useful third
# data point alongside Gemma when isolating template-driven failure
# modes (e.g., the [TOOL_CALLS] leak).
#
# VRAM budget at UD-Q4_K_XL + q4_0 KV cache + flash-attn:
#   weights              ~17-18 GB  (Q4 over 106B sparse params)
#   KV cache (32K ctx)   ~2 GB
#   buffers              ~1 GB
#   total                ~20-21 GB
#
# Larger context (60K) bumps KV to ~4 GB — back off to Q4_K_S or IQ4_XS
# if you OOM. Air is meant for 32-64K typical agent loops, not 200K+.
#
# Download the model first:
#   mkdir -p $HOME/models
#   hf download unsloth/GLM-4.5-Air-GGUF \
#     --include "*UD-Q4_K_XL*" \
#     --local-dir $HOME/models/GLM-4.5-Air-GGUF

set -euo pipefail

MODEL_DIR="${MINISWE_MODEL_DIR:-$HOME/models/GLM-4.5-Air-GGUF}"
PORT="${MINISWE_PORT:-8464}"
CTX_SIZE="${MINISWE_CTX_SIZE:-32768}"
THREADS="${MINISWE_THREADS:-8}"

MODEL="${MINISWE_MODEL:-}"
if [ -z "$MODEL" ]; then
    MODEL=$(ls "$MODEL_DIR"/GLM-4.5-Air-*UD-Q4_K_XL*.gguf 2>/dev/null | head -1 || true)
    [ -z "$MODEL" ] && MODEL=$(ls "$MODEL_DIR"/GLM-4.5-Air-*Q4_K_M*.gguf 2>/dev/null | head -1 || true)
    # Sharded downloads ship as *-00001-of-0000N.gguf; llama.cpp follows the rest.
    [ -z "$MODEL" ] && MODEL=$(ls "$MODEL_DIR"/GLM-4.5-Air-*Q4*-00001-of-*.gguf 2>/dev/null | head -1 || true)
fi

if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "Model not found under $MODEL_DIR" >&2
    echo "" >&2
    echo "Download it with:" >&2
    echo "  hf download unsloth/GLM-4.5-Air-GGUF \\" >&2
    echo "    --include '*UD-Q4_K_XL*' \\" >&2
    echo "    --local-dir $MODEL_DIR" >&2
    exit 1
fi

echo "Starting GLM-4.5-Air for miniswe..."
echo "  Model:   $MODEL"
echo "  Context: $CTX_SIZE tokens"
echo "  KV:      q4_0"
echo "  Port:    $PORT"
echo ""

# GLM uses temperature 0.6 + top_p 0.95 in Zhipu's recipe; the bench
# scripts override per-request, so these only apply when a client
# doesn't specify them.
exec "$(dirname "$0")/scripts/run-llama-cuda.sh" \
    --jinja \
    --model "$MODEL" \
    --ctx-size "$CTX_SIZE" \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    --n-gpu-layers 99 \
    --flash-attn on \
    --threads "$THREADS" \
    --temp 0.6 \
    --top-p 0.95 \
    -np 1 \
    --port "$PORT" \
    --metrics
