#!/usr/bin/env bash
# Start llama-server with Cohere North Mini Code 1.0 (30B-A3B MoE, agentic
# coding) for miniswe.
#
# North Mini Code 1.0 (CohereLabs, Jun 2026): sparse MoE, ~30B total / ~3B
# ACTIVE per token — 128 routed experts, 8 active (sigmoid gating, no shared
# expert), expert intermediate 768. 49 layers in a [global, sliding x3]
# pattern: 13 full-attention + 36 sliding-window (4096-token window); layer 0
# is dense. 4 KV heads x 128 dim; 500K-token context, rope theta 50000, no
# rope scaling. Verified from CohereLabs/North-Mini-Code-1.0 config.json on
# 2026-08-23. Apache 2.0. Claims SWE-bench Verified 67.6%.
#
# Hardware: RTX 3090 (24 GB) + 128 GB RAM + Ryzen 9950X3D (16 cores).
# Sized for a ~20 GB VRAM budget (~4 GB of the card reserved for other work).
#
# VRAM MATH (unsloth UD-Q4_K_M, 19.2 GB on disk):
#   K+V per token per layer = 4 heads x 128 x 2 x 2 = 2 KB at f16. Only the
#   13 global layers cache the whole context; the 36 sliding layers hold 4096
#   tokens each (~0.3 GB). At 60K context:
#     KV f16   13 x 60000 x 2 KB ≈ 1.6 GB      q8_0 ≈ 0.8 GB    (+0.3 GB sliding)
#   Experts are ~0.34 GB per MoE layer (16.3 GB of the 19.2 GB) but only 8/128
#   are touched per token, so --n-cpu-moe is nearly free (same trick as
#   start-laguna-xs.sh). Default below keeps 10 MoE layers' experts on CPU:
#     weights on GPU     ~15.8 GB   (19.2 - 10 x 0.34)
#     KV 60K q8_0         ~1.1 GB
#     CUDA ctx + buffers  ~1.3 GB
#     total               ~18.2 GB  → inside the budget with ~2 GB to spare
#   Tune: lower MINISWE_NCMOE to pull experts back onto the GPU (each ≈ 0.34
#   GB), raise it if nvidia-smi shows a spill. Do NOT use -ngl here — whole-
#   layer offload moves attention too and costs ~5 ms/token per layer.
#
# Expected decode: ~3B active, Laguna/gemma-26B class (~100+ tok/s). Measure.
#
# THINKING ("interleaved thinking", Cohere-style): the chat template's
# `reasoning` kwarg (default TRUE) decides whether the generation prompt opens
# `<|START_THINKING|>` or prefills an empty `<|START_THINKING|><|END_THINKING|>`
# block; `reasoning_effort: "none"` is equivalent to `reasoning: false`. The
# template has NO `enable_thinking` variable, so miniswe's per-request
# `{"enable_thinking": false/true}` is inert for this model — the server
# default below (MINISWE_REASONING, default false = instruct arm) is what
# counts. The server merges request kwargs over the defaults key-by-key, so
# miniswe's kwarg neither helps nor hurts. A thinking arm = MINISWE_REASONING=
# true + `model.thinking = true` in miniswe (for temp 0.6) — with the caveat
# that the card says the model "works best" when prior turns' reasoning is
# passed back in history; miniswe discards `reasoning_content`, so that arm is
# off-distribution (llama.cpp's --reasoning-preserve may help; untested).
#
# TOOL CALLS use Cohere's `<|START_ACTION|>[{"tool_call_id":…,"tool_name":…,
# "parameters":{…}}]<|END_ACTION|>` JSON-list format (llama.cpp's Command-R7B
# style parser). llama.cpp `cohere2moe` support merged 2026-06-13 (PR
# #24260); our server-cuda13 image is build 10524. Verify with a tool-call
# round trip before the first bench run.
#
# SAMPLING per the model card: temp 1.0, top-p 0.95 (benchmarks run with
# exactly those). miniswe overrides `temperature` per request (bench config
# 0.2; 0.6 when `model.thinking` is on); top-p comes from this flag.
#
# Download the model first (~19.2 GB, single file):
#   mkdir -p $HOME/models
#   hf download unsloth/North-Mini-Code-1.0-GGUF \
#     --include "North-Mini-Code-1.0-UD-Q4_K_M.gguf" \
#     --local-dir $HOME/models/North-Mini-Code-1.0-GGUF
#
# Other 4-bit quants in that repo: UD-Q4_K_S 18.05 GB, MXFP4_MOE 18.66 GB,
# UD-Q4_K_XL 19.25 GB, UD-IQ4_XS 15.2 GB — K_S/MXFP4 need ~3 fewer CPU expert
# layers; IQ4_XS fits fully on the GPU.

set -euo pipefail

MODEL_DIR="${MINISWE_MODEL_DIR:-$HOME/models/North-Mini-Code-1.0-GGUF}"
PORT="${MINISWE_PORT:-8464}"
CTX_SIZE="${MINISWE_CTX_SIZE:-60000}"
KV_TYPE="${MINISWE_KV_TYPE:-q8_0}"      # f16 also fits (+0.8 GB); q4_0 is a suspected loop-pathology contributor
THREADS="${MINISWE_THREADS:-16}"        # physical cores; used for CPU-resident expert tensors
NCMOE="${MINISWE_NCMOE:-10}"            # MoE layers (of 48) whose experts live on CPU; 0 = all on GPU
REASONING="${MINISWE_REASONING:-false}" # template kwarg: false = instruct arm (empty think block), true = interleaved thinking

case "$REASONING" in
    true|false) ;;
    *) echo "MINISWE_REASONING must be true|false (got '$REASONING')" >&2; exit 1 ;;
esac

MODEL="${MINISWE_MODEL:-}"
if [ -z "$MODEL" ]; then
    for pat in 'North-Mini-Code-1.0-UD-Q4_K_M' 'North-Mini-Code-1.0-UD-Q4_K_S' 'North-Mini-Code-1.0-MXFP4_MOE' 'North-Mini-Code-1.0-UD-Q4_K_XL' 'North-Mini-Code-1.0-UD-IQ4_XS'; do
        MODEL=$(ls "$MODEL_DIR"/${pat}*.gguf 2>/dev/null | grep -v -- '-00002-of-' | head -1 || true)
        [ -n "$MODEL" ] && break
    done
fi

if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "Model not found under $MODEL_DIR" >&2
    echo "" >&2
    echo "Download it with:" >&2
    echo "  hf download unsloth/North-Mini-Code-1.0-GGUF \\" >&2
    echo "    --include 'North-Mini-Code-1.0-UD-Q4_K_M.gguf' \\" >&2
    echo "    --local-dir $MODEL_DIR" >&2
    exit 1
fi

ARGS=(
    --jinja
    --chat-template-kwargs "{\"reasoning\":$REASONING}"
    --model "$MODEL"
    --ctx-size "$CTX_SIZE"
    --cache-type-k "$KV_TYPE"
    --cache-type-v "$KV_TYPE"
    --n-gpu-layers 999
    --n-cpu-moe "$NCMOE"
    --flash-attn on
    --threads "$THREADS"
    --temp 1.0
    --top-p 0.95
    -np 1
    --port "$PORT"
    --metrics
)

echo "Starting North Mini Code 1.0 (30B-A3B MoE) for miniswe..."
echo "  Model:     $MODEL"
echo "  Context:   $CTX_SIZE tokens, KV $KV_TYPE (13 global layers cache full context)"
echo "  Experts:   $NCMOE/48 MoE layers' experts on CPU (0 = all on GPU)"
echo "  Thinking:  reasoning=$REASONING (template kwarg; miniswe's enable_thinking is inert for this model)"
echo "  Sampling:  temp 1.0, top-p 0.95 (miniswe overrides temp per request)"
echo "  Port:      $PORT"
echo ""

exec "$(dirname "$0")/scripts/run-llama-cuda.sh" "${ARGS[@]}"
