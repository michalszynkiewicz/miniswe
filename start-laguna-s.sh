#!/usr/bin/env bash
# Laguna S 2.1 (poolside) — 118B total / 8B active MoE, for miniswe.
#
# SIZE. Near-twin of Mistral Small 4 (119B-A6B): UD-Q4_K_XL is 73.4 GB vs
# Mistral's 70 GB. The difference that matters is ACTIVE params — 8B vs 6B,
# so ~33% more weight streamed per token. Expect it to be the slower of the two.
#
# ARCHITECTURE (from config.json):
#   48 layers, of which only 12 are full-attention; the other 36 are
#   sliding-window (window=512). 256 routed experts + 1 shared, top-10.
#   num_key_value_heads=8, head_dim=128.
#
# CONTEXT IS ALMOST FREE. Only the 12 global layers scale with context; the
# 36 SWA layers cost a flat ~40 MB no matter how long the prompt is.
#   KV/token = 12 x 2 x 8 x 128
#     q8_0 -> ~26.1 KB/token  (1 GB per ~39K tokens)
#     q4_0 -> ~13.8 KB/token  (1 GB per ~72K tokens)
#   60K ctx at q8_0 is ~1.6 GB. You could run 250K+ before VRAM complains.
#   Verify against llama-server's own KV line at load rather than this comment.
#
# VRAM BUDGET on a 24 GB 3090 (200 W cap). Experts are ~68 GB of the 73 GB
# file, ~1.45 GB per layer. Everything you don't spend on KV buys experts:
#   total VRAM                                       24.0 GB
#   desktop/display already resident (MEASURED)      -4.9 GB   <- do not forget this
#   non-expert weights (attn + shared + embeddings)  -2.5 GB
#   KV 60K q8_0                                      -1.6 GB
#   CUDA context + buffers                           -1.4 GB
#   -> ~13.6 GB left = ~9 layers of experts resident
# Default below keeps 40 of 48 MoE layers' experts on CPU (8 resident, ~11.6 GB,
# ~2 GB slack). An OOM costs a full restart cycle and 8-vs-9 resident layers is
# a ~2% change in CPU work, so this is deliberately conservative: LOWER NCMOE
# toward 38 if nvidia-smi shows real headroom, raise it if it spills.
# NOTE: the 4.9 GB desktop floor is why this is 40 and not 36 -- an earlier
# estimate budgeted the full 24 GB and would have OOMed on first boot.
#
# EXPECTED DECODE: 10/256 experts x ~1.43 GB/layer = ~55 MB/layer/token;
# 36 CPU layers = ~2 GB/token off RAM. Dual-channel DDR5 at a realistic
# 60-70 GB/s puts the ceiling near 30 tok/s -> plan on 12-20 tok/s.
# Needs ~68 GB of free RAM for the offloaded experts.
#
# LOAD MODE. llama.cpp warns at load that this exact config is a bad pairing:
# "tensor overrides to CPU are used with mmap enabled - consider using
# --load-mode none for better performance". It is not cosmetic. On the 08-28
# demo-e2e-task run the model was 69 GB on disk but only 55.2 GB resident
# (RssFile), with 1.9 GB of llama-server swapped out -- ~14 GB of the CPU-side
# experts were being faulted from disk on a model that reads those experts on
# EVERY token, which taxes both prefill (165 tok/s) and decode (19.3 tok/s).
# `none` drops mmap and loads the tensors outright, so the experts land in
# anonymous memory the page cache cannot evict. Costs a slower cold start (a
# real 69 GB read) and shows up as ~69 GB anon rather than as page cache --
# budget that against the k3d cluster the pkg-mcp e2e deploys on the same box.
# MINISWE_LOAD_MODE=mmap+mlock pins the mmap instead (run-llama-cuda.sh already
# passes --ulimit memlock=-1 --cap-add IPC_LOCK); =auto restores pre-08-30.
#
# THINKING: same chat-template contract as Laguna XS — the template keys on
# the `enable_thinking` kwarg (default false -> prefills `</think>` so the
# model answers directly). miniswe sends the kwarg per request. CAVEAT:
# Laguna is trained with "preserved thinking" and expects prior turns'
# reasoning echoed back in `reasoning_content`; miniswe discards reasoning,
# so a thinking arm runs slightly off-distribution. Run instruct first.
# NOTE: miniswe's thinking flag also raises temperature 0.2 -> 0.6.
#
# TOOL CALLS use the GLM-style <tool_call> format, parsed natively by
# llama.cpp (same as Laguna XS 2.1).
#
# SAMPLING per the model card: temp 1.0, top-k 20, top-p 1.0. miniswe
# overrides temperature per request; these are the server-side defaults.
#
# Download (~73 GB, 3 shards):
#   hf download unsloth/Laguna-S-2.1-GGUF --include "UD-Q4_K_XL/*" \
#       --local-dir ~/models/Laguna-S-2.1-GGUF
set -euo pipefail

MODEL_DIR="${MINISWE_MODEL_DIR:-$HOME/models/Laguna-S-2.1-GGUF}"
CTX_SIZE="${MINISWE_CTX_SIZE:-60000}"
KV_TYPE="${MINISWE_KV_TYPE:-q8_0}"
NCMOE="${MINISWE_NCMOE:-40}"
LOAD_MODE="${MINISWE_LOAD_MODE:-none}"
THREADS="${MINISWE_THREADS:-16}"
PORT="${MINISWE_PORT:-8464}"

MODEL="${MINISWE_MODEL:-}"
if [ -z "$MODEL" ]; then
    MODEL=$(find "$MODEL_DIR" -maxdepth 2 -name '*UD-Q4_K_XL*00001-of-*.gguf' 2>/dev/null | head -1 || true)
    [ -z "$MODEL" ] && MODEL=$(find "$MODEL_DIR" -maxdepth 2 -name '*.gguf' 2>/dev/null | grep -v -- '-0000[2-9]-of-' | head -1 || true)
fi
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "Model not found under $MODEL_DIR" >&2
    echo "  hf download unsloth/Laguna-S-2.1-GGUF --include \"UD-Q4_K_XL/*\" \\" >&2
    echo "      --local-dir $MODEL_DIR" >&2
    exit 1
fi

echo "Starting Laguna S 2.1 (118B-A8B MoE) for miniswe..."
echo "  Model:     $MODEL"
echo "  Context:   $CTX_SIZE tokens, KV $KV_TYPE (12 of 48 layers cache full context)"
echo "  Experts:   $NCMOE/48 MoE layers' experts on CPU (0 = all on GPU)"
echo "  Threads:   $THREADS"
echo "  Load mode: $LOAD_MODE (none = no mmap, experts stay resident; slower start)"
echo "  Port:      $PORT"
echo "  Expect:    ~12-20 tok/s; needs ~68 GB free RAM for offloaded experts."
echo ""

exec "$(dirname "$0")/scripts/run-llama-cuda.sh" \
    --jinja \
    --chat-template-kwargs '{"enable_thinking":false}' \
    --model "$MODEL" \
    --ctx-size "$CTX_SIZE" \
    --cache-type-k "$KV_TYPE" \
    --cache-type-v "$KV_TYPE" \
    --n-gpu-layers 999 \
    --n-cpu-moe "$NCMOE" \
    --load-mode "$LOAD_MODE" \
    --flash-attn on \
    --threads "$THREADS" \
    --threads-batch 32 \
    --batch-size 2048 \
    --temp 1.0 \
    --top-k 20 \
    --top-p 1.0 \
    -np 1 \
    --port "$PORT" \
    --metrics
