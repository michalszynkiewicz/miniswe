#!/usr/bin/env bash
# Start llama-server with Poolside Laguna XS 2.1 (33B-A3B MoE, agentic) for miniswe.
#
# Laguna XS 2.1 (poolside, Aug 2026): sparse MoE, ~33B total / ~3B ACTIVE per
# token — 256 routed experts + 1 shared, 8 routed active (moe_intermediate
# 512). 40 layers in a [global, sliding, sliding, sliding] pattern: 10
# full-attention layers + 30 sliding-window layers (512-token window); layer 0
# is dense MLP, the other 39 are MoE. 8 KV heads x 128 dim, 262,144-token
# context (YaRN x32 on the global layers). Verified from
# poolside/Laguna-XS-2.1 config.json on 2026-08-23. Licence OpenMDW-1.1.
#
# Hardware: RTX 3090 (24 GB) + 128 GB RAM + Ryzen 9950X3D (16 cores).
# Sized for a ~20 GB VRAM budget (~4 GB of the card reserved for other work).
#
# VRAM MATH (IQ4_XS, 18.2 GB on disk):
#   K+V per token per layer = 8 heads x 128 x 2 = 2 KB at f16. Only the 10
#   global layers cache the whole context; the 30 sliding layers are capped
#   at 512 tokens (~30 MB total). At 60K context:
#     KV f16   10 x 60000 x 2 KB ≈ 1.2 GB      q8_0 ≈ 0.65 GB
#   Experts are 0.43 GB per MoE layer (16.7 GB of the 18.2 GB total) but only
#   8/256 are touched per token (~13 MB/layer/token), so --n-cpu-moe is nearly
#   free: each layer pushed to CPU saves ~0.43 GB of VRAM for ~0.2 ms/token of
#   RAM streaming. Default below keeps 6 MoE layers' experts on CPU:
#     weights on GPU     ~15.6 GB   (18.2 - 6 x 0.43)
#     KV 60K q8_0         ~0.65 GB
#     CUDA ctx + buffers  ~1.2 GB
#     total               ~17.5 GB  → comfortably inside the budget
#   Tune: MINISWE_NCMOE=0 puts everything on the GPU (~20 GB, at the line);
#   raise it if nvidia-smi shows a spill. Do NOT use -ngl for this model —
#   whole-layer offload moves attention too and costs ~5 ms/token per layer.
#
# Expected decode: ~3B active → gemma-4 26B-A4B ballpark (~70 tok/s fully
# resident). Measure on the first run.
#
# THINKING: the chat template keys on the `enable_thinking` kwarg (template
# default: false → it prefills `</think>` so the model answers directly;
# true → leaves `<think>` open). miniswe sends
# `chat_template_kwargs: {"enable_thinking": false}` on instruct requests and
# `true` when `model.thinking` is on; the server default below covers clients
# that send no kwargs. CAVEAT for a thinking arm: Laguna is trained with
# "preserved thinking" — it expects prior turns' reasoning echoed back in
# `reasoning_content`. miniswe discards reasoning, so the thinking arm runs
# in a slightly off-distribution regime; run instruct first.
#
# TOOL CALLS use the GLM-style `<tool_call>name<arg_key>k</arg_key>
# <arg_value>v</arg_value>…</tool_call>` format; llama.cpp parses it natively
# (Laguna support merged 2026-07-22, PR #25165; our server-cuda13 image is
# build 10524, the bartowski quants were made with b10087).
#
# SAMPLING per the model card: temp 1.0, top-k 20, top-p 1.0. miniswe
# overrides `temperature` per request (bench config 0.2; 0.6 when
# `model.thinking` is on); top-k/top-p come from these flags.
#
# Download the model first (~18.2 GB, single file, imatrix quant):
#   mkdir -p $HOME/models
#   hf download bartowski/Laguna-XS-2.1-GGUF \
#     --include "Laguna-XS-2.1-IQ4_XS.gguf" \
#     --local-dir $HOME/models/Laguna-XS-2.1-GGUF
#
# Other 4-bit quants in that repo: Q4_0 19.2 GB, Q4_K_S 19.8 GB, Q4_K_M
# 20.6 GB, Q4_K_L 20.7 GB — all fit only with more experts on CPU
# (MINISWE_NCMOE=10-12). poolside/Laguna-XS-2.1-GGUF ships an official
# Q4_K_M (same size class) and BF16.

set -euo pipefail

MODEL_DIR="${MINISWE_MODEL_DIR:-$HOME/models/Laguna-XS-2.1-GGUF}"
PORT="${MINISWE_PORT:-8464}"
CTX_SIZE="${MINISWE_CTX_SIZE:-60000}"
KV_TYPE="${MINISWE_KV_TYPE:-q8_0}"      # f16 also fits (+0.6 GB); q4_0 is a suspected loop-pathology contributor
THREADS="${MINISWE_THREADS:-16}"        # physical cores; used for CPU-resident expert tensors
NCMOE="${MINISWE_NCMOE:-6}"             # MoE layers (of 39) whose experts live on CPU; 0 = all on GPU

MODEL="${MINISWE_MODEL:-}"
if [ -z "$MODEL" ]; then
    # Preference order: the budget-safe IQ4_XS, then the larger 4-bit quants.
    for pat in 'Laguna-XS-2.1-IQ4_XS' 'Laguna-XS-2.1-Q4_K_S' 'Laguna-XS-2.1-Q4_0' 'Laguna-XS-2.1-Q4_K_M'; do
        MODEL=$(ls "$MODEL_DIR"/${pat}*.gguf 2>/dev/null | grep -v -- '-00002-of-' | head -1 || true)
        [ -n "$MODEL" ] && break
    done
fi

if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "Model not found under $MODEL_DIR" >&2
    echo "" >&2
    echo "Download it with:" >&2
    echo "  hf download bartowski/Laguna-XS-2.1-GGUF \\" >&2
    echo "    --include 'Laguna-XS-2.1-IQ4_XS.gguf' \\" >&2
    echo "    --local-dir $MODEL_DIR" >&2
    exit 1
fi

ARGS=(
    --jinja
    --chat-template-kwargs '{"enable_thinking":false}'
    --model "$MODEL"
    --ctx-size "$CTX_SIZE"
    --cache-type-k "$KV_TYPE"
    --cache-type-v "$KV_TYPE"
    --n-gpu-layers 999
    --n-cpu-moe "$NCMOE"
    --flash-attn on
    --threads "$THREADS"
    --temp 1.0
    --top-k 20
    --top-p 1.0
    -np 1
    --port "$PORT"
    --metrics
)

echo "Starting Laguna XS 2.1 (33B-A3B MoE) for miniswe..."
echo "  Model:     $MODEL"
echo "  Context:   $CTX_SIZE tokens, KV $KV_TYPE (10 global layers cache full context)"
echo "  Experts:   $NCMOE/39 MoE layers' experts on CPU (0 = all on GPU)"
echo "  Thinking:  off by default (enable_thinking=false); per-request kwargs override"
echo "  Sampling:  temp 1.0, top-k 20, top-p 1.0 (miniswe overrides temp per request)"
echo "  Port:      $PORT"
echo ""

exec "$(dirname "$0")/scripts/run-llama-cuda.sh" "${ARGS[@]}"
