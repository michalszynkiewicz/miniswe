#!/usr/bin/env bash
# Start llama-server with Muse Glimmer-30B + the DFlash drafter (speculative
# decoding) for miniswe.
#
# This is start-muse-glimmer-30b.sh with ONE variable changed: the drafter is
# loaded. Model, quant, context, KV type, flash-attn, threads, sampling,
# reasoning strength and reasoning budget are all deliberately identical, so
# the resulting row is a clean A/B against the plain Glimmer row and any wall
# time difference is attributable to speculative decoding alone. If you change
# a setting here, change it in the base script too or the comparison is dead.
#
# WHY: decode dominates this model's runs. Measured over the three Glimmer
# slots of round_20260905_160709 (server.log timings):
#
#   slot          prefill   decode    max ctx reached
#   06-glimmer       97 s    409 s     36,668 tokens
#   13-glimmer       94 s    330 s     35,818 tokens
#   20-glimmer       88 s    368 s     30,651 tokens
#
# ~80% of server time and ~65% of wall clock is token generation at 20 tok/s.
# That is exactly the cost speculative decoding attacks, and Glimmer is good
# material for it: its output is mostly structured tool-call JSON and code,
# which a drafter predicts well, and it makes the fewest edits in the field so
# there is little thrash to mispredict.
#
# CONTEXT SIZING — why 80000 and not 60000. miniswe's `context_window` is
# 60000 but `estimate_tokens` is `text.len() / 4` (src/context/mod.rs:297), a
# chars-per-token heuristic, not a tokenizer. Measured against llama-server's
# real counts on the three Glimmer slots of round_20260905_160709:
#
#   slot          miniswe estimate   server real   ratio
#   06-glimmer              29,249        36,668   1.25x
#   13-glimmer              28,324        35,818   1.26x
#   20-glimmer              24,459        30,651   1.25x
#
# So miniswe's 60000 budget is ~75,000 REAL tokens at the server. A server
# --ctx-size of 60000 is overflowed by any run that fills miniswe's own
# budget, and the server truncates silently — miniswe has no idea it happened.
# This is not hypothetical: in that same round, every slot that recorded a
# `truncated = 1` event was running at ctx 50176 or 60160 (gemma x4,
# laguna-xs x3, north x4, devstral x1), and all three Glimmer slots at 131072
# recorded zero. 80000 clears the ~75k ceiling with margin.
#
# 80000 also differs from the base arm's 131072, but inertly: allocated KV
# does not affect decode or prefill speed (attention cost scales with actual
# sequence length), and the only behavioural effect of ctx size is truncation,
# which cannot occur below ~75k real tokens. The worst prompt ever observed on
# this model was 36,668.
#
# VRAM — measured, not estimated. The drafter's geometry was read out of the
# GGUF header (general.architecture=dflash, size_label 2.6B):
#
#   dflash.block_count                5
#   dflash.attention.head_count_kv    8
#   dflash.attention.key_length     128   (value_length likewise)
#   dflash.attention.sliding_window  2048
#   dflash.attention.sliding_window_pattern  [1,1,1,1,1]   <- ALL layers local
#   dflash.target_layers            [2, 14, 26, 38, 50]
#
# Every DFlash layer is sliding-window, so its KV is pinned to the 2048-token
# window and does NOT scale with the context: 5 layers x 8 kv heads x
# (128+128) x 2 B x 2048 = 42 MB. The draft context is effectively free, which
# is just as well — it is no longer separately sizable and simply inherits the
# target's.
#
#   main weights, KQuant-17GB-Q4_K_M          16.80 GB
#   main KV @ 80K f16 (13 global + 39 local)   1.14 GB
#   dflash weights (HF content-length)         1.63 GB
#   dflash KV (window-capped, see above)       0.04 GB
#   CUDA ctx + compute buffers                ~1.0-1.5 GB
#   total                                   ~20.6-21.1 GB of 24 GB
#
# That leaves ~3 GB spare. NO CPU OFFLOAD IS NEEDED: keep MINISWE_NGL=99 (all
# 52 target layers) and MINISWE_DRAFT_NGL=99 (all 5 draft layers). A
# CPU-resident layer costs ~5 ms/token and would swamp any gain the drafter
# makes, so offload is the wrong lever even if you are tight — shrink the main
# KV with MINISWE_KV_TYPE=q8_0 (~1.14 GB -> ~0.6 GB) before touching layers.
# MINISWE_DRAFT_KV_TYPE exists for symmetry but saves ~20 MB; it is not a lever.
#
# NOTE: dflash.target_layers means this is an EAGLE-style drafter that reads
# the target model's hidden states, not a standalone small model. It is
# therefore tightly coupled to Glimmer specifically. If the llama.cpp build
# does not recognise the `dflash` architecture the server fails at load — a
# loud error, not a silent fallback.
#
# SLIDING WINDOW. Glimmer is 39 SWA layers (2048-token window) + 13 global.
# Speculative decoding rewinds the target KV on a rejected draft, and this
# repo has a whole fix (`has_narrow_attention_window`, fcd8378) about how
# llama.cpp cannot roll back past a sliding window without a full re-prefill.
# That cliff does not apply here: rewinds are bounded by the draft block size
# (15 tokens), two orders of magnitude inside the 2048-token window.
#
# OUTPUT EQUIVALENCE. llama.cpp's acceptance test preserves the target
# model's distribution, so this arm samples from the same distribution as the
# base arm but not the same realized trajectories. Expect scores to be
# statistically indistinguishable and wall time to move. If acceptance turns
# out to be poor, speculative decoding is SLOWER than no drafter at all --
# that is a real possible outcome of this arm, not a misconfiguration.
#
# FLAG NAMES. llama.cpp renamed the whole drafter namespace to --spec-*.
# --draft-max and --draft-min were REMOVED outright (now --spec-draft-n-max /
# --spec-draft-n-min), and --ctx-size-draft / -cd is gone with no replacement:
# common_params_speculative_draft has no context field at all any more, the
# drafter inherits the target's. -md, -ngld and --draft-p-min still work as
# aliases, but the canonical spellings are used below so the next rename fails
# loudly instead of drifting.
#
# BLOCK SIZE. The drafter's `dflash.block_size` is 16 and DFlash denoises in
# place from [id_last, <mask> x 15], so it yields at most block_size - 1 = 15
# draft tokens. Asking for more is clamped with a warning; 15 is the ceiling.
#
# NO --spec-type NEEDED. The server reads general.architecture=dflash out of
# the draft GGUF and selects draft-dflash itself. Confirm it actually engaged —
# a drafter that quietly fails to attach is exactly the silent null result this
# arm is exposed to — by grepping the server log for:
#
#   auto-detected speculative type 'draft-dflash' from the draft model metadata
#   ...: - n_max=15, n_min=0, p_min=0.00
#   ...: - block_size=16, mask_token_id=..., n_extract=5, ...
#
# n_extract=5 is the count of target layers tapped ([2,14,26,38,50]).
#
# Download the drafter first (1.6 GB):
#   hf download meta-models/Muse-Glimmer-30B-GGUF \
#     --include "dflash-Muse-Glimmer-30B-Q4_K_M.gguf" \
#     --local-dir $HOME/models/Muse-Glimmer-30B-GGUF
#
# See start-muse-glimmer-30b.sh for the rest of the model's notes (reasoning
# strength ladder, why reasoning cannot be disabled, KV math).

set -euo pipefail

MODEL_DIR="${MINISWE_MODEL_DIR:-$HOME/models/Muse-Glimmer-30B-GGUF}"
PORT="${MINISWE_PORT:-8464}"
CTX_SIZE="${MINISWE_CTX_SIZE:-80000}"     # see the CONTEXT SIZING note above; do NOT drop to 60000
KV_TYPE="${MINISWE_KV_TYPE:-f16}"       # f16 fits; q8_0 if VRAM is tight
THREADS="${MINISWE_THREADS:-16}"        # physical cores; used for CPU-resident tensors
NGL="${MINISWE_NGL:-99}"                # GPU layers of 52 (99 = all)
REASONING_STRENGTH="${MINISWE_REASONING_STRENGTH:-medium}"   # low|medium|high|xhigh
REASONING_BUDGET="${MINISWE_REASONING_BUDGET:-2000}"         # think tokens before forced close; -1 = unlimited

# --- speculative decoding knobs (this script's reason to exist) -------------
DRAFT_NGL="${MINISWE_DRAFT_NGL:-99}"      # drafter GPU layers; a CPU drafter defeats the purpose
DRAFT_MAX="${MINISWE_DRAFT_MAX:-15}"      # tokens drafted per step; block_size - 1 is the hard ceiling
DRAFT_MIN="${MINISWE_DRAFT_MIN:-0}"       # upstream default; a shorter draft than this is dropped entirely
DRAFT_P_MIN="${MINISWE_DRAFT_P_MIN:-0}"   # upstream default: draft the whole block, let the target verify it.
                                          # Above 0 the block is truncated at the first position whose top-1
                                          # probability falls below it. That saves drafter work but not target
                                          # work — one verify pass costs about the same for 4 tokens as for 15
                                          # — so it trades away accepted tokens for nothing. Leave it at 0.
DRAFT_KV_TYPE="${MINISWE_DRAFT_KV_TYPE:-}"  # empty = server default; saves only ~20 MB, not a real lever

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

DRAFT_MODEL="${MINISWE_DRAFT_MODEL:-}"
if [ -z "$DRAFT_MODEL" ]; then
    DRAFT_MODEL=$(ls "$MODEL_DIR"/dflash-Muse-Glimmer-30B*.gguf 2>/dev/null | head -1 || true)
fi

if [ -z "$DRAFT_MODEL" ] || [ ! -f "$DRAFT_MODEL" ]; then
    echo "Draft model not found under $MODEL_DIR" >&2
    echo "" >&2
    echo "This script exists to run the drafter — without it, use" >&2
    echo "start-muse-glimmer-30b.sh instead. Download it with:" >&2
    echo "  hf download meta-models/Muse-Glimmer-30B-GGUF \\" >&2
    echo "    --include 'dflash-Muse-Glimmer-30B-Q4_K_M.gguf' \\" >&2
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
    # --- drafter (see FLAG NAMES above; the --draft-* spellings are removed) ---
    --spec-draft-model "$DRAFT_MODEL"
    --spec-draft-ngl "$DRAFT_NGL"
    --spec-draft-n-max "$DRAFT_MAX"
    --spec-draft-n-min "$DRAFT_MIN"
    --spec-draft-p-min "$DRAFT_P_MIN"
)

# There is deliberately no draft context size here: the drafter has none of its
# own, and its KV is window-capped at ~42 MB regardless (see VRAM above).

# Opt-in only: quantizing a 42 MB cache saves ~20 MB, so the flag is passed
# just when explicitly asked for.
if [ -n "$DRAFT_KV_TYPE" ]; then
    ARGS+=(--spec-draft-type-k "$DRAFT_KV_TYPE" --spec-draft-type-v "$DRAFT_KV_TYPE")
fi

echo "Starting Muse Glimmer-30B + DFlash drafter (speculative decoding) for miniswe..."
echo "  Model:      $MODEL"
echo "  Draft:      $DRAFT_MODEL"
echo "  Context:    $CTX_SIZE tokens, KV $KV_TYPE (13 global layers cache full context)"
echo "  Layers:     $NGL/52 on GPU, drafter $DRAFT_NGL (99 = all)"
echo "  Spec:       n-max=$DRAFT_MAX n-min=$DRAFT_MIN p-min=$DRAFT_P_MIN${DRAFT_KV_TYPE:+, draft KV $DRAFT_KV_TYPE} (draft ctx inherits the target's)"
echo "  Reasoning:  strength=$REASONING_STRENGTH, budget=$REASONING_BUDGET tokens (cannot be disabled)"
echo "  Sampling:   temp 1.0, top-p 0.95, top-k 64 (miniswe overrides temp per request)"
echo "  Port:       $PORT"
echo ""
echo "  Budget: ~20.6-21.1 GB of 24 GB (draft KV is window-capped at ~42 MB)."
echo ""

exec "$(dirname "$0")/scripts/run-llama-cuda.sh" "${ARGS[@]}"
