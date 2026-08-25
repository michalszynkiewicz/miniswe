#!/usr/bin/env bash
# Start llama-server with Muse Glimmer-30B (dense, agentic, always-thinking)
# for miniswe.
#
# Muse Glimmer-30B (meta-models, Aug 2026): ~29.6B dense incl. a 1.8B vision
# encoder that lives in the separate mmproj (not loaded here — text only).
# 52 layers in a [local, local, local, global] pattern: 39 sliding-window
# layers (2048-token window) + 13 full-attention layers; 2 KV heads x 128
# dim; 131,072-token context. Verified from meta-models/Muse-Glimmer-30B
# config.json on 2026-08-22. Claims SWE-bench Verified 76.0%.
#
# Hardware: RTX 3090 (24 GB) + 128 GB RAM + Ryzen 9950X3D (16 cores).
# Sized for a ~20 GB VRAM budget (~4 GB of the card reserved for other work).
#
# WHY THE FULL 131K CONTEXT FITS ON THE GPU AT f16 KV: K+V per token per
# layer is 2 heads x 128 x 2 = 1 KB. Only the 13 global layers cache the
# whole context (13 x 131072 x 1 KB = 1.75 GB); the 39 sliding layers are
# capped at 2048 tokens (80 MB). So:
#
#   KQuant-17GB-Q4_K_M weights   16.8 GB
#   KV cache 131K, f16            ~1.8 GB   (q8_0 → ~1.0 GB if VRAM is tight)
#   CUDA ctx + compute buffers   ~1.0-1.5 GB
#   total                        ~19.5-20 GB  → fully resident, no CPU offload
#
# No q4_0 KV here on purpose: the cache is small enough for f16, and q4 KV is
# a suspected contributor to the loop pathologies we fight on other models.
# If nvidia-smi shows it spilling over budget: MINISWE_KV_TYPE=q8_0 first,
# MINISWE_CTX_SIZE=100000 second, MINISWE_NGL=50 last (~5 ms/token per CPU layer).
#
# Expected decode ~20 tok/s (dense, weights-bandwidth-bound). The repo ships a
# DFlash draft model for speculative decoding (1.6 GB) — MINISWE_DRAFT=1 loads
# it if present; untested, watch VRAM.
#
# THINKING: reasoning CANNOT be disabled — the template always opens the
# thinking channel; miniswe's `enable_thinking` kwarg is ignored by it (the
# server merges request kwargs over the defaults below key-by-key, so
# reasoning_strength survives). Intensity is the template kwarg
# `reasoning_strength` = low | medium | high | xhigh (vendor recommends
# high/xhigh for agentic work). Start with MEDIUM: the bench gives ~50-80 s
# per round, i.e. ~1000-1500 generated tokens at 20 tok/s, thinking included
# — xhigh would spend that many times over on one step. Ladder: medium →
# high → xhigh, each a server restart with MINISWE_REASONING_STRENGTH.
#
# --reasoning-budget N is the safety net: the server force-closes the think
# block after N reasoning tokens so the answer always has room inside
# miniswe's max_tokens (bench: 8000). Without it a long think returns empty
# content with finish_reason=length (the failure that made us disable gemma
# thinking). Raise the budget together with max_output_tokens when moving up
# the ladder (suggested: high → 4000/8000, xhigh → 6000/10000).
#
# Reasoning text arrives as `reasoning_content`; miniswe discards it (never
# re-sent), so thinking does not grow the conversation — context pressure
# comes from history only. miniswe's own `model.context_window` (bench:
# 60000) governs compaction; raise it separately to actually use more.
#
# SAMPLING per the model card: temp 1.0, top-p 0.95, top-k 64. miniswe
# overrides `temperature` per request (0.2 instruct, 0.6 when
# `model.thinking` is on); top-p/top-k come from these flags.
#
# Needs llama.cpp >= b10353 (Glimmer support, merged 2026-08-10); our
# server-cuda13 image is build 10524.
#
# Download the model first (~16.8 GB):
#   mkdir -p $HOME/models
#   hf download meta-models/Muse-Glimmer-30B-GGUF \
#     --include "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf" \
#     --local-dir $HOME/models/Muse-Glimmer-30B-GGUF
#
# Optional draft model for MINISWE_DRAFT=1 (1.6 GB):
#   hf download meta-models/Muse-Glimmer-30B-GGUF \
#     --include "dflash-Muse-Glimmer-30B-Q4_K_M.gguf" \
#     --local-dir $HOME/models/Muse-Glimmer-30B-GGUF
#
# The other text quant, KQuant-Dynamic-Q4_K_XL (19.7 GB), does not fit the
# 20 GB budget with any KV; mmproj-*.gguf is the vision tower, not needed.

set -euo pipefail

MODEL_DIR="${MINISWE_MODEL_DIR:-$HOME/models/Muse-Glimmer-30B-GGUF}"
PORT="${MINISWE_PORT:-8464}"
CTX_SIZE="${MINISWE_CTX_SIZE:-131072}"
KV_TYPE="${MINISWE_KV_TYPE:-f16}"       # f16 fits; q8_0 if VRAM is tight
THREADS="${MINISWE_THREADS:-16}"        # physical cores; used for CPU-resident tensors
NGL="${MINISWE_NGL:-99}"                # GPU layers of 52 (99 = all)
REASONING_STRENGTH="${MINISWE_REASONING_STRENGTH:-medium}"   # low|medium|high|xhigh
REASONING_BUDGET="${MINISWE_REASONING_BUDGET:-2000}"         # think tokens before forced close; -1 = unlimited
DRAFT="${MINISWE_DRAFT:-0}"             # 1 = load the DFlash drafter for speculative decoding

case "$REASONING_STRENGTH" in
    low|medium|high|xhigh) ;;
    *) echo "MINISWE_REASONING_STRENGTH must be low|medium|high|xhigh (got '$REASONING_STRENGTH')" >&2; exit 1 ;;
esac

MODEL="${MINISWE_MODEL:-}"
if [ -z "$MODEL" ]; then
    for pat in 'Muse-Glimmer-30B-KQuant-17GB-Q4_K_M' 'Muse-Glimmer-30B-Q4_K_M' 'Muse-Glimmer-30B-UD-Q4_K_M'; do
        MODEL=$(ls "$MODEL_DIR"/${pat}*.gguf 2>/dev/null | grep -v -- '-00002-of-' | head -1 || true)
        [ -n "$MODEL" ] && break
    done
fi

if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "Model not found under $MODEL_DIR" >&2
    echo "" >&2
    echo "Download it with:" >&2
    echo "  hf download meta-models/Muse-Glimmer-30B-GGUF \\" >&2
    echo "    --include 'Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf' \\" >&2
    echo "    --local-dir $MODEL_DIR" >&2
    exit 1
fi

ARGS=(
    --jinja
    --chat-template-kwargs "{\"reasoning_strength\":\"$REASONING_STRENGTH\"}"
    --reasoning-budget "$REASONING_BUDGET"
    --model "$MODEL"
    --ctx-size "$CTX_SIZE"
    --cache-type-k "$KV_TYPE"
    --cache-type-v "$KV_TYPE"
    --n-gpu-layers "$NGL"
    --flash-attn on
    --threads "$THREADS"
    --temp 1.0
    --top-p 0.95
    --top-k 64
    -np 1
    --port "$PORT"
    --metrics
)

DRAFT_MODEL=""
if [ "$DRAFT" = "1" ]; then
    DRAFT_MODEL=$(ls "$MODEL_DIR"/dflash-Muse-Glimmer-30B*.gguf 2>/dev/null | head -1 || true)
    if [ -z "$DRAFT_MODEL" ]; then
        echo "MINISWE_DRAFT=1 but no dflash-Muse-Glimmer-30B*.gguf under $MODEL_DIR" >&2
        exit 1
    fi
    ARGS+=(-md "$DRAFT_MODEL" -ngld 99)
fi

echo "Starting Muse Glimmer-30B (dense, always-thinking) for miniswe..."
echo "  Model:      $MODEL"
echo "  Context:    $CTX_SIZE tokens, KV $KV_TYPE (13 global layers cache full context)"
echo "  Layers:     $NGL/52 on GPU (99 = all)"
echo "  Reasoning:  strength=$REASONING_STRENGTH, budget=$REASONING_BUDGET tokens (cannot be disabled)"
echo "  Sampling:   temp 1.0, top-p 0.95, top-k 64 (miniswe overrides temp per request)"
[ -n "$DRAFT_MODEL" ] && echo "  Draft:      $DRAFT_MODEL (speculative decoding)"
echo "  Port:       $PORT"
echo ""

exec "$(dirname "$0")/scripts/run-llama-cuda.sh" "${ARGS[@]}"
