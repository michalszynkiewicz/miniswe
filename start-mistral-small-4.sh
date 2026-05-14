#!/usr/bin/env bash
# Start llama-server with Mistral Small 4 for miniswe.
#
# Released March 2026 — Mistral's unified MoE replacing the separate
# Magistral (reasoning), Pixtral (vision), and Devstral (agentic coding)
# lines into a single model. Apache 2.0, 256K context, multimodal.
#
# Architecture: MoE, 128 experts, 4 active per token.
#   119B total params / 6B active per forward (8B incl. embeddings).
#
# Hardware target: RTX 3090 (24GB VRAM, ~21GB usable) + 128GB RAM
# Sparse activation makes CPU-offload practical — only 6B params hit
# per token, so the experts can sit in RAM while attention + active
# layers run on the GPU. Expected throughput on this rig: ~15-30 tok/s
# (slower than Devstral 24B which runs fully on GPU, but usable for
# bench runs you're willing to wait on).
#
# Q4 footprint ~60GB on disk → ~60GB in RAM. Comfortably fits with
# the 128GB budget alongside KV cache.
#
# Download the model first (~60 GB at Q4):
#   mkdir -p $HOME/models
#   hf download unsloth/Mistral-Small-4-119B-2603-GGUF \
#     --include "*UD-Q4_K_XL*" \
#     --local-dir $HOME/models/Mistral-Small-4-119B-GGUF
#
# Smaller quants if RAM is tight: Q4_K_M (~58 GB), Q3_K_M (~46 GB).
#
# Note: Mistral Small 4 exposes a reasoning_effort parameter
# (none/high). Bench scripts can override per-request; this server
# starts with reasoning_effort left to client default.

set -euo pipefail

MODEL_DIR="${MINISWE_MODEL_DIR:-$HOME/models/Mistral-Small-4-119B-GGUF}"
PORT="${MINISWE_PORT:-8464}"
CTX_SIZE="${MINISWE_CTX_SIZE:-60000}"
THREADS="${MINISWE_THREADS:-16}"

MODEL="${MINISWE_MODEL:-}"
if [ -z "$MODEL" ]; then
    # Unsloth ships large multi-shard quants under a per-quant
    # subdirectory (e.g. UD-Q4_K_XL/<shards>.gguf), so search up to 2
    # levels deep. Single-shard quants land in the top-level dir.
    # Always pass the first shard (-00001-of-*) — llama.cpp follows the rest.
    MODEL=$(find "$MODEL_DIR" -maxdepth 2 -name 'Mistral-Small-4-119B-*UD-Q4_K_XL*-00001-of-*.gguf' 2>/dev/null | head -1)
    [ -z "$MODEL" ] && MODEL=$(find "$MODEL_DIR" -maxdepth 2 -name 'Mistral-Small-4-119B-*UD-Q4_K_XL*.gguf' ! -name '*-of-*' 2>/dev/null | head -1)
    [ -z "$MODEL" ] && MODEL=$(find "$MODEL_DIR" -maxdepth 2 -name 'Mistral-Small-4-119B-*Q4_K_M*-00001-of-*.gguf' 2>/dev/null | head -1)
    [ -z "$MODEL" ] && MODEL=$(find "$MODEL_DIR" -maxdepth 2 -name 'Mistral-Small-4-119B-*Q4_K_M*.gguf' ! -name '*-of-*' 2>/dev/null | head -1)
    [ -z "$MODEL" ] && MODEL=$(find "$MODEL_DIR" -maxdepth 2 -name 'Mistral-Small-4-119B-*-00001-of-*.gguf' 2>/dev/null | head -1)
fi

if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "Model not found under $MODEL_DIR" >&2
    echo "" >&2
    echo "Download it with:" >&2
    echo "  hf download unsloth/Mistral-Small-4-119B-2603-GGUF \\" >&2
    echo "    --include '*UD-Q4_K_XL*' \\" >&2
    echo "    --local-dir $MODEL_DIR" >&2
    exit 1
fi

echo "Starting Mistral Small 4 119B (MoE, 6B active) for miniswe..."
echo "  Model:    $MODEL"
echo "  Context:  $CTX_SIZE tokens"
echo "  KV:       q4_0"
echo "  Threads:  $THREADS"
echo "  Port:     $PORT"
echo "  Strategy: GPU does attention + hot layers; CPU holds the experts."
echo "  Expect:   ~15-30 tok/s (much slower than Devstral 24B, but smarter)."
echo ""

# --override-tensor keeps the experts on CPU while attention layers and
# small tensors run on the GPU. This is the standard MoE-on-consumer-GPU
# idiom for llama.cpp — much faster than naive partial offload because
# the hot path (attention per token) stays accelerated and only the
# sparse expert mat-muls hit RAM bandwidth.
exec "$(dirname "$0")/scripts/run-llama-cuda.sh" \
    --jinja \
    --model "$MODEL" \
    --ctx-size "$CTX_SIZE" \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    --n-gpu-layers 99 \
    --override-tensor "([0-9]+).ffn_.*_exps.=CPU" \
    --flash-attn on \
    --threads "$THREADS" \
    --threads-batch 32 \
    --batch-size 2048 \
    --mlock \
    -np 1 \
    --port "$PORT" \
    --metrics
