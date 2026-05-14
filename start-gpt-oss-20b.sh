#!/usr/bin/env bash
# Start llama-server with GPT-OSS-20B for miniswe.
#
# Hardware target: RTX 3090 (24GB VRAM, ~21GB usable) + 128GB RAM
# Model: GPT-OSS-20B (OpenAI's open-weight 20B, dense, native MXFP4
# weights — Q4 territory at full quality, no extra quant loss).
# Different post-training and tool-call format from Mistral/Qwen/Gemma,
# so it's the cleanest "is the bug template-specific" probe.
#
# VRAM budget:
#   weights              ~12 GB  (MXFP4 native)
#   KV cache (60K ctx)   ~2-3 GB
#   buffers              ~1 GB
#   total                ~15-16 GB  (lots of headroom; can push ctx to 128K)
#
# Download the model first:
#   mkdir -p $HOME/models
#   hf download unsloth/gpt-oss-20b-GGUF \
#     --include "*F16*" \
#     --local-dir $HOME/models/gpt-oss-20b-GGUF
#
# Note: unsloth ships gpt-oss in MXFP4 + F16-wrapped variants — the F16
# include filter pulls the standard layout. If that fails, pass
# `--include "*"` and pick the file you want manually.

set -euo pipefail

MODEL_DIR="${MINISWE_MODEL_DIR:-$HOME/models/gpt-oss-20b-GGUF}"
PORT="${MINISWE_PORT:-8464}"
CTX_SIZE="${MINISWE_CTX_SIZE:-60000}"
THREADS="${MINISWE_THREADS:-8}"

MODEL="${MINISWE_MODEL:-}"
if [ -z "$MODEL" ]; then
    # MXFP4 is the native format; F16 wrappers are the common llama.cpp ones.
    MODEL=$(ls "$MODEL_DIR"/gpt-oss-20b-*F16*.gguf 2>/dev/null | head -1 || true)
    [ -z "$MODEL" ] && MODEL=$(ls "$MODEL_DIR"/gpt-oss-20b-*MXFP4*.gguf 2>/dev/null | head -1 || true)
    [ -z "$MODEL" ] && MODEL=$(ls "$MODEL_DIR"/gpt-oss-20b-*Q4*.gguf 2>/dev/null | head -1 || true)
    [ -z "$MODEL" ] && MODEL=$(ls "$MODEL_DIR"/gpt-oss-20b-*-00001-of-*.gguf 2>/dev/null | head -1 || true)
fi

if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "Model not found under $MODEL_DIR" >&2
    echo "" >&2
    echo "Download it with:" >&2
    echo "  hf download unsloth/gpt-oss-20b-GGUF \\" >&2
    echo "    --include '*F16*' \\" >&2
    echo "    --local-dir $MODEL_DIR" >&2
    exit 1
fi

echo "Starting GPT-OSS-20B for miniswe..."
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
