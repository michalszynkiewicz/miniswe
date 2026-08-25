#!/usr/bin/env bash
# Start llama-server with NVIDIA Nemotron 3.5 Lightning 30B-A3B for miniswe.
#
# Released 2026-08-11. Hybrid Mamba-Transformer MoE: 30B total, ~3B ACTIVE per
# token, agentic-trained ("self-healing tool calling"), up to 1M context.
#
# Hardware: RTX 3090 (24 GB) + 128 GB RAM + Ryzen 9950X3D (16 cores).
# Sized for a ~20 GB VRAM budget (rest of the card reserved for other work).
#
# KEY DIFFERENCE from start-gemma4-31b.sh: that model is DENSE, so it offloads
# whole LAYERS to CPU with -ngl. This one is MoE, so the cheap win is to keep
# the attention/Mamba/router path on the GPU and push only the EXPERT tensors
# to CPU with --n-cpu-moe. Only ~3B params are active per token, so the CPU
# only has to stream the active experts — the same trick that ran your gemma-4
# 26B MoE fast. This is far better than treating it like a dense model.
#
#   UD-Q4_K_XL weights   ~20 GB total  (most of it is rarely-touched experts)
#   with all experts on CPU, GPU holds only attention + KV + active path
#   → VRAM footprint drops to a few GB, leaving lots of room for "X"
#
# The knob (MINISWE_NCMOE): number of layers whose MoE experts live on CPU.
#   Default 99 = ALL experts on CPU (llama.cpp caps it at the real layer
#   count). Safest: guaranteed to fit the 20 GB budget with headroom, starts
#   reliably regardless of the exact layer geometry, relies on the fast CPU +
#   3B-active for decode speed.
#   To go FASTER: LOWER MINISWE_NCMOE to pull expert layers back onto the GPU,
#   watching `nvidia-smi` on load until you approach your 20 GB budget. Each
#   layer you pull off CPU costs VRAM but speeds decode.
#     Override:  MINISWE_NCMOE=24 ./start-nemotron35-30b.sh
#
# Expected speed on this box: decode should land well above the dense 31B's
# 8.6 tok/s (only ~3B active vs 30.7B dense) — likely in the 26B-MoE ballpark.
# Measure on first run and tune NCMOE from there.
#
# Optional MTP speculative decoding (MINISWE_MTP=1): the UD GGUF ships with a
# Multi-Token-Prediction draft head; --spec-type draft-mtp uses it as a
# built-in draft model (~1.3-1.5x decode, no separate draft model). Requires a
# recent llama.cpp build (b10362+) AND an MTP-enabled GGUF (NOT the *-noMTP-*
# variants). Left OFF by default so an older cuda image still starts; flip it
# on once you've confirmed the server binary recognizes the flag.
#
# Download the model first (~20 GB):
#   mkdir -p $HOME/models
#   hf download unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF \
#     --include "*UD-Q4_K_XL*" \
#     --local-dir $HOME/models/Nemotron-3.5-Lightning-30B-A3B-GGUF
#
# Smaller quants if you want more headroom: Q4_K_M, Q4_K_S, IQ4_XS.
# The *-noMTP-NVFP4-* variant is smaller but strips the MTP draft head.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${MINISWE_MODEL_DIR:-$HOME/models/Nemotron-3.5-Lightning-30B-A3B-GGUF}"

# Chat-template fix: the GGUF's shipped template (identical to NVIDIA's HF
# original) prefills `<|im_start|>assistant\n<think></think>` with NO newline
# after the closed think tag in non-thinking mode. The model's trained format
# has exactly one \n there; withholding it makes generation degenerate into
# full-max_tokens newline floods at prose→tool-call junctures (7/8 flood on
# exact bench-request replays; +1 newline = 0/8 and instant clean tool calls).
# The patched copy adds that single newline and nothing else.
TEMPLATE_FIX="${SCRIPT_DIR}/scripts/templates/nemotron35-lightning-think-nl-fix.jinja"
PORT="${MINISWE_PORT:-8464}"
CTX_SIZE="${MINISWE_CTX_SIZE:-60000}"
THREADS="${MINISWE_THREADS:-16}"      # physical cores, for the CPU-resident experts
NCMOE="${MINISWE_NCMOE:-99}"          # layers whose MoE experts sit on CPU (99 = all).
                                      # LOWER it to pull experts onto the GPU for speed
                                      # as VRAM allows; RAISE (or keep 99) to stay within
                                      # the 20 GB budget / leave room for other work.

MODEL="${MINISWE_MODEL:-}"
if [ -z "$MODEL" ]; then
    MODEL=$(ls "$MODEL_DIR"/*Nemotron-3.5-Lightning-30B-A3B-UD-Q4_K_XL*.gguf 2>/dev/null | grep -v -- '-00002-of-' | head -1 || true)
    [ -z "$MODEL" ] && MODEL=$(ls "$MODEL_DIR"/*Nemotron-3.5-Lightning-30B-A3B-Q4_K_M*.gguf 2>/dev/null | head -1 || true)
    # Sharded downloads ship as *-00001-of-0000N.gguf; llama.cpp follows the rest.
    [ -z "$MODEL" ] && MODEL=$(ls "$MODEL_DIR"/*Nemotron-3.5-Lightning-30B-A3B-*Q4*-00001-of-*.gguf 2>/dev/null | head -1 || true)
fi

if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "Model not found under $MODEL_DIR" >&2
    echo "" >&2
    echo "Download it with:" >&2
    echo "  hf download unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF \\" >&2
    echo "    --include '*UD-Q4_K_XL*' \\" >&2
    echo "    --local-dir $MODEL_DIR" >&2
    exit 1
fi

if [ ! -f "$TEMPLATE_FIX" ]; then
    echo "Patched chat template not found: $TEMPLATE_FIX" >&2
    echo "Refusing to start without it — the stock template causes newline floods." >&2
    echo "(Regenerate: fetch /props chat_template and add \\n after '<think></think>'" >&2
    echo " in the add_generation_prompt branch.)" >&2
    exit 1
fi

# Mount the template into the container at the same path so the flag Just Works.
export LLAMA_EXTRA_MOUNT="${LLAMA_EXTRA_MOUNT:-} -v ${SCRIPT_DIR}/scripts/templates:${SCRIPT_DIR}/scripts/templates:ro"

# Build args as an array so the optional MTP flag can be appended conditionally.
ARGS=(
    --jinja
    --chat-template-file "$TEMPLATE_FIX"
    --model "$MODEL"
    --ctx-size "$CTX_SIZE"
    --cache-type-k q4_0
    --cache-type-v q4_0
    --n-gpu-layers 999
    --n-cpu-moe "$NCMOE"
    --flash-attn on
    --threads "$THREADS"
    --temp 0.2
    --top-p 0.95
    --min-p 0.01
    -np 1
    --port "$PORT"
    --metrics
)

MTP_NOTE="off"
if [ "${MINISWE_MTP:-0}" = "1" ]; then
    ARGS+=(--spec-type draft-mtp)
    MTP_NOTE="on (--spec-type draft-mtp)"
fi

echo "Starting NVIDIA Nemotron 3.5 Lightning 30B-A3B (MoE) for miniswe..."
echo "  Model:     $MODEL"
echo "  Context:   $CTX_SIZE tokens"
echo "  KV:        q4_0"
echo "  Experts:   $NCMOE layers on CPU (99 = all); attention/active path on GPU"
echo "  Sampling:  instruct mode (temp 0.2, top-p 0.95, min-p 0.01)"
echo "  Template:  ${TEMPLATE_FIX##*/} (newline-after-</think> flood fix)"
echo "  MTP spec:  $MTP_NOTE"
echo "  Port:      $PORT"
echo ""

exec "$(dirname "$0")/scripts/run-llama-cuda.sh" "${ARGS[@]}"
