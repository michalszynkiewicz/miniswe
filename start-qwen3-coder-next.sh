#!/usr/bin/env bash
# Start llama-server with Qwen3-Coder-Next for miniswe.
#
# Released February 2026 — Qwen's purpose-built agentic-coding model
# designed for local use. Hybrid MoE/attention architecture (12×(3×DeltaNet→MoE)
# → Gated Attention → MoE) with 512 experts, 10 routed + 1 shared per token.
#
# Architecture: MoE, 80B total / 3B active per forward.
#   Native 256K context, non-thinking mode only.
#   Qwen claims comparable performance to models with 10–20× more active params.
#
# Hardware target: RTX 3090 (24GB VRAM, ~21GB usable) + 128GB RAM
# Same offload strategy as Mistral Small 4: experts on CPU, attention +
# hot layers on GPU. With only 3B active per token and ~46GB weights at
# UD-Q4_K_XL, this should run noticeably faster than Mistral Small 4
# (which has 6B active and ~60GB weights).
#
# Expected throughput on this rig: ~25–40 tok/s.
#
# Download the model first (~46 GB at Q4):
#   mkdir -p $HOME/models
#   hf download unsloth/Qwen3-Coder-Next-GGUF \
#     --include "*UD-Q4_K_XL*" \
#     --local-dir $HOME/models/Qwen3-Coder-Next-GGUF
#
# Smaller quants if RAM is tight: Q4_K_M (~44 GB), Q3_K_M (~36 GB).
#
# Recommended sampling (Qwen team): temperature=1.0, top_p=0.95, top_k=40,
# repeat_penalty=1.1. Bench scripts override per-request; this server starts
# with the recommended defaults so chat/manual use behaves sensibly.

set -euo pipefail

MODEL_DIR="${MINISWE_MODEL_DIR:-$HOME/models/Qwen3-Coder-Next-GGUF}"
PORT="${MINISWE_PORT:-8464}"
CTX_SIZE="${MINISWE_CTX_SIZE:-60000}"
THREADS="${MINISWE_THREADS:-16}"

MODEL="${MINISWE_MODEL:-}"
if [ -z "$MODEL" ]; then
    # Without this check, `find` against a missing directory exits non-zero
    # under `set -euo pipefail`, killing the script before the helpful
    # "Model not found" branch below can print the download instructions.
    if [ ! -d "$MODEL_DIR" ]; then
        echo "Model directory not found: $MODEL_DIR" >&2
        echo "" >&2
        echo "Download it with:" >&2
        echo "  hf download unsloth/Qwen3-Coder-Next-GGUF \\" >&2
        echo "    --include '*UD-Q4_K_XL*' \\" >&2
        echo "    --local-dir $MODEL_DIR" >&2
        echo "" >&2
        echo "Or set MINISWE_MODEL_DIR to point to an existing directory." >&2
        exit 1
    fi

    # Unsloth ships larger quants under a per-quant subdirectory; smaller
    # quants land in the top-level dir. Search up to 2 levels deep, always
    # passing the first shard so llama.cpp follows the rest.
    MODEL=$(find "$MODEL_DIR" -maxdepth 2 -name 'Qwen3-Coder-Next-*UD-Q4_K_XL*-00001-of-*.gguf' 2>/dev/null | head -1)
    [ -z "$MODEL" ] && MODEL=$(find "$MODEL_DIR" -maxdepth 2 -name 'Qwen3-Coder-Next-*UD-Q4_K_XL*.gguf' ! -name '*-of-*' 2>/dev/null | head -1)
    [ -z "$MODEL" ] && MODEL=$(find "$MODEL_DIR" -maxdepth 2 -name 'Qwen3-Coder-Next-*Q4_K_M*-00001-of-*.gguf' 2>/dev/null | head -1)
    [ -z "$MODEL" ] && MODEL=$(find "$MODEL_DIR" -maxdepth 2 -name 'Qwen3-Coder-Next-*Q4_K_M*.gguf' ! -name '*-of-*' 2>/dev/null | head -1)
    [ -z "$MODEL" ] && MODEL=$(find "$MODEL_DIR" -maxdepth 2 -name 'Qwen3-Coder-Next-*-00001-of-*.gguf' 2>/dev/null | head -1)
fi

if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "No Qwen3-Coder-Next GGUF found under $MODEL_DIR" >&2
    echo "" >&2
    echo "Download it with:" >&2
    echo "  hf download unsloth/Qwen3-Coder-Next-GGUF \\" >&2
    echo "    --include '*UD-Q4_K_XL*' \\" >&2
    echo "    --local-dir $MODEL_DIR" >&2
    exit 1
fi

echo "Starting Qwen3-Coder-Next 80B (MoE, 3B active) for miniswe..."
echo "  Model:    $MODEL"
echo "  Context:  $CTX_SIZE tokens"
echo "  KV:       q4_0"
echo "  Threads:  $THREADS"
echo "  Port:     $PORT"
echo "  Sampling: temp=1.0 top_p=0.95 top_k=40 repeat_penalty=1.1 (Qwen-recommended)"
echo "  Strategy: GPU does attention + hot layers; CPU holds the experts."
echo "  Expect:   ~25-40 tok/s (faster than Mistral Small 4: 3B vs 6B active)."
echo ""

# --override-tensor keeps the experts on CPU while attention layers and
# small tensors run on the GPU. Standard MoE-on-consumer-GPU idiom — keeps
# the hot per-token path accelerated and only sparse expert mat-muls hit
# RAM bandwidth.
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
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 40 \
    --repeat-penalty 1.1 \
    -np 1 \
    --port "$PORT" \
    --metrics
