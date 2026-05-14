#!/bin/bash
# Start llama-server with Qwen3-Coder-30B-A3B-Instruct (MoE, 30B total / 3B active)
# for the miniswe benchmark.
#
# Released April 2026; benchmark-leading on SWE-Bench / LiveCodeBench among
# open-weight models that fit on 24 GB.
#
# Hardware target: RTX 3090 (24GB VRAM, ~21GB usable) + 128GB RAM
# VRAM budget at UD-Q4_K_XL + q4_0 KV cache + flash-attn:
#   weights              17.7 GB
#   KV cache (60K ctx)   ~2-3 GB
#   buffers              ~1 GB
#   total                ~21 GB
#
# Smaller quants if it OOMs: Q4_K_M (18.6 GB), Q4_K_S (17.5 GB), IQ4_XS (16.4 GB).
#
# First-time setup (downloads ~18 GB):
#   mkdir -p $HOME/models && hg download unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF --include "*UD-Q4_K_XL*" --local-dir $HOME/models/Qwen3-Coder-30B-A3B-Instruct-GGUF

set -euo pipefail

MODEL_DIR="$HOME/models/Qwen3-Coder-30B-A3B-Instruct-GGUF"
MODEL_FILE="$MODEL_DIR/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf"

if [ ! -f "$MODEL_FILE" ]; then
    # Handle sharded downloads (larger quants ship as *-00001-of-0000N.gguf)
    SHARD=$(ls "$MODEL_DIR"/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL*-00001-of-*.gguf 2>/dev/null | head -1 || true)
    [ -z "$SHARD" ] && SHARD=$(ls "$MODEL_DIR"/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M*.gguf 2>/dev/null | head -1 || true)
    [ -z "$SHARD" ] && SHARD=$(ls "$MODEL_DIR"/Qwen3-Coder-30B-A3B-Instruct-IQ4_XS*.gguf 2>/dev/null | head -1 || true)
    if [ -n "$SHARD" ]; then
        MODEL_FILE="$SHARD"
    else
        echo "Model file not found under $MODEL_DIR" >&2
        echo "Run the huggingface-cli download command from the header comment." >&2
        exit 1
    fi
fi

# Sampling defaults follow the Qwen team's recommendations for
# Qwen3-Coder-30B-A3B-Instruct. The miniswe benchmark scripts override
# `temperature` and other knobs per-request; these only apply when a
# client doesn't specify them.
exec "$(dirname "$0")/scripts/run-llama-cuda.sh" \
    --jinja \
    -m "$MODEL_FILE" \
    -c 60000 \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    -ngl 99 \
    --flash-attn on \
    --temp 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --repeat-penalty 1.05 \
    -np 1 \
    --port 8464
