#!/usr/bin/env bash
# Start llama-server with Gemma 4 31B IT (dense, Q4) for miniswe.
#
# Hardware: RTX 3090 (24 GB) + 128 GB RAM + Ryzen 9950X3D (16 cores).
# Sized for a ~20 GB VRAM budget (rest of the card reserved for other work)
# at 60K context — which does NOT fit fully on-GPU, so part of the model is
# offloaded to CPU:
#   UD-Q4_K_XL weights  ~18.8 GB
#   q4_0 KV @60K        ~3   GB   (10 global layers grow with ctx; 50 sliding-
#                                  window layers stay capped at the 1024 window)
#   compute overhead    ~2   GB
#   total               ~24 GB   →  trimmed to ~20 GB via --n-gpu-layers
#
# The 31B is DENSE (no cheap MoE expert offload), so whole layers go to CPU.
# Default 48/60 on GPU → 12 on CPU (~20%), ~19 GB. Each offloaded layer takes
# its KV to CPU too, so -ngl eases both weight and KV pressure on the GPU.
#
# Tuning (the knob): watch `nvidia-smi` on load + during the first prefill —
#   headroom  → raise NGL toward 50 (fewer CPU layers = faster)
#   OOM       → lower NGL (overhead spikes during prefill, so leave margin)
#   Override:  MINISWE_NGL=50 ./start-gemma4-31b.sh
#
# Expected speed on this box: ~10-12 tok/s decode — ~3x slower than fully-GPU
# Devstral-Small-2 Q4, ~8x slower than the gemma-4-26B MoE (measured 95 tok/s).
# Prefill takes the bigger hit from the CPU layers.
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
CTX_SIZE="${MINISWE_CTX_SIZE:-60000}"
THREADS="${MINISWE_THREADS:-16}"      # physical cores, for the CPU-resident layers
NGL="${MINISWE_NGL:-42}"              # GPU layers of 60; 18 on CPU (~30%). Measured on
                                      # this box (RTX 3090, ~5.5 GB desktop baseline):
                                      # model footprint ~14.4 GB, ~3.8 GB VRAM free for
                                      # other work ("X"), decode ~8.6 tok/s. Raise toward
                                      # 48 (~16 GB, ~11 tok/s) if you don't need the
                                      # headroom; lower if X needs more or it OOMs.

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

echo "Starting Gemma 4 31B IT (dense) for miniswe..."
echo "  Model:   $MODEL"
echo "  Context: $CTX_SIZE tokens"
echo "  KV:      q4_0"
echo "  Layers:  $NGL/60 on GPU, $((60 - NGL)) on CPU"
echo "  Port:    $PORT"
echo ""

exec "$(dirname "$0")/scripts/run-llama-cuda.sh" \
    --jinja \
    --model "$MODEL" \
    --ctx-size "$CTX_SIZE" \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    --n-gpu-layers "$NGL" \
    --flash-attn on \
    --threads "$THREADS" \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 64 \
    -np 1 \
    --port "$PORT" \
    --metrics
