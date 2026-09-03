#!/usr/bin/env bash
# Start llama-server with Qwen3.8-27B (dense) for miniswe.
#
# Qwen3.8 (2026-08) builds on the Qwen3.5 architecture: model_type qwen3_5,
# 64 layers in a 3:1 hybrid — 48 Gated-DeltaNet (linear attention) layers and
# 16 full-attention layers (full_attention_interval=4), 4 KV heads x 256 dim.
# Verified from Qwen/Qwen3.8-27B config.json on 2026-08-22.
#
# Hardware: RTX 3090 (24 GB) + 128 GB RAM + Ryzen 9950X3D (16 cores).
# Sized for a ~20 GB VRAM budget (~4 GB of the card reserved for other work).
#
# WHY THIS FITS FULLY ON THE GPU (unlike the dense gemma-4-31B, which needs
# ~30% of its layers on CPU and decodes at 8.6 tok/s): only the 16 attention
# layers keep a KV cache, and they are small (4 KV heads x 256 dim). At 60K
# context with q4_0 KV that is ~1.1 GB, vs ~4.5 GB for a conventional 64-layer
# dense model. The 48 GDN layers carry a fixed-size recurrent state instead.
#
#   UD-Q4_K_M weights    16.5 GB  (token embeddings ~0.8 GB of that stay on CPU)
#   KV cache 60K q4_0    ~1.1 GB
#   CUDA ctx + buffers   ~1.3 GB
#   total                ~18 GB   → fully resident, ~2 GB under the budget
#
#   UD-Q4_K_XL (17.6 GB) is ~19.2 GB total — right at the line. Try it with
#   MINISWE_MODEL=... and watch nvidia-smi; if it spills, lower MINISWE_NGL.
#
# Expected decode: weights-bandwidth-bound at ~17 GB/token → ~20-25 tok/s
# fully resident. Each layer pushed to CPU costs roughly 5 ms/token.
#
# The knob (MINISWE_NGL): layers on the GPU, of 64. Default 999 = all.
#   OOM at load, or VRAM over budget → MINISWE_NGL=60 (each CPU layer ≈ 0.27 GB).
#
# THINKING: Qwen3.8's chat template thinks by default (reasoning_effort=xhigh).
# miniswe sends `chat_template_kwargs: {"enable_thinking": false}` on
# instruct requests (and `true` when `model.thinking` is on), which the
# template honors: false prefills `<think>\n</think>\n\n`, true leaves
# `<think>\n` open. The server-side default below covers clients that send
# no kwargs (REPL via curl, etc). Do NOT pass reasoning_effort=none — the
# template only accepts xhigh|medium|low and raises on anything else; depth
# tuning for the thinking arm is `{"enable_thinking": true, "reasoning_effort": "low"}`.
#
# SAMPLING follows Qwen's instruct-mode recommendation (temp 0.7, top-p 0.8,
# top-k 20, min-p 0). miniswe overrides `temperature` per request (bench
# config uses 0.2); top-p/top-k/penalties come from these flags. Qwen also
# recommends presence_penalty 1.5 in instruct mode to curb repetition; it is
# OFF here by default so A/B runs stay comparable with the other launchers
# (none use it) and structured tool-call output isn't biased against
# repeated identifiers. Enable it with MINISWE_PRESENCE_PENALTY=1.5.
#
# Download the model first (~16.5 GB):
#   mkdir -p $HOME/models
#   hf download unsloth/Qwen3.8-27B-GGUF \
#     --include "Qwen3.8-27B-UD-Q4_K_M.gguf" \
#     --local-dir $HOME/models/Qwen3.8-27B-GGUF
#
# Other 4-bit quants in that repo: UD-Q4_K_XL 17.6 GB, UD-Q4_K_S 15.4 GB,
# UD-IQ4_XS 14.3 GB. The mmproj-*.gguf files are the vision tower — not
# needed for text-only use and not loaded here. The repo also ships an MTP/
# draft head for speculative decoding (untested with this launcher).

set -euo pipefail

MODEL_DIR="${MINISWE_MODEL_DIR:-$HOME/models/Qwen3.8-27B-GGUF}"
PORT="${MINISWE_PORT:-8464}"
CTX_SIZE="${MINISWE_CTX_SIZE:-60000}"
THREADS="${MINISWE_THREADS:-16}"      # physical cores; used for CPU-resident tensors
NGL="${MINISWE_NGL:-999}"             # GPU layers of 64 (999 = all). Lower if VRAM spills.
PRESENCE_PENALTY="${MINISWE_PRESENCE_PENALTY:-0}"

MODEL="${MINISWE_MODEL:-}"
if [ -z "$MODEL" ]; then
    # Preference order: the budget-safe K_M, then K_XL, then the smaller quants.
    for pat in 'Qwen3.8-27B-UD-Q4_K_M' 'Qwen3.8-27B-UD-Q4_K_XL' 'Qwen3.8-27B-UD-Q4_K_S' 'Qwen3.8-27B-UD-IQ4_XS' 'Qwen3.8-27B-Q4_K_M'; do
        MODEL=$(ls "$MODEL_DIR"/${pat}*.gguf 2>/dev/null | grep -v -- '-00002-of-' | head -1 || true)
        [ -n "$MODEL" ] && break
    done
fi

if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "Model not found under $MODEL_DIR" >&2
    echo "" >&2
    echo "Download it with:" >&2
    echo "  hf download unsloth/Qwen3.8-27B-GGUF \\" >&2
    echo "    --include 'Qwen3.8-27B-UD-Q4_K_M.gguf' \\" >&2
    echo "    --local-dir $MODEL_DIR" >&2
    exit 1
fi

ARGS=(
    --jinja
    --chat-template-kwargs '{"enable_thinking":false}'
    --model "$MODEL"
    --ctx-size "$CTX_SIZE"
    --cache-type-k q4_0
    --cache-type-v q4_0
    --n-gpu-layers "$NGL"
    --flash-attn on
    --threads "$THREADS"
    --temp 0.7
    --top-p 0.8
    --top-k 20
    --min-p 0.0
    --presence-penalty "$PRESENCE_PENALTY"
    -np 1
    --port "$PORT"
    --metrics
)

echo "Starting Qwen3.8-27B (dense, GDN/attention hybrid) for miniswe..."
echo "  Model:     $MODEL"
echo "  Context:   $CTX_SIZE tokens"
echo "  KV:        q4_0 (only the 16 attention layers cache KV)"
echo "  Layers:    $NGL/64 on GPU (999 = all)"
echo "  Thinking:  off by default (enable_thinking=false); per-request kwargs override"
echo "  Sampling:  instruct mode (temp 0.7, top-p 0.8, top-k 20, min-p 0, presence ${PRESENCE_PENALTY})"
echo "  Port:      $PORT"
echo ""

exec "$(dirname "$0")/scripts/run-llama-cuda.sh" "${ARGS[@]}"
