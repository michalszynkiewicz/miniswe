#!/usr/bin/env bash
# bootstrap-vast.sh — run ON the Vast.ai VM. Sets up the toolchain, fetches the
# model(s), starts a native CUDA llama-server, and runs the Docker-free bench.
#
# Assumes: a CUDA *devel* base image (nvcc present) with apt, plus the miniswe
# repo already synced to the current directory (the launcher rsyncs it here).
#
# Usage (from the repo root on the VM):
#   scripts/vast/bootstrap-vast.sh <gemma|qwen|both> [runs]
#
# Env:
#   HF_TOKEN          optional HuggingFace token (avoids download rate limits)
#   LLAMA_SERVER_BIN  skip building llama.cpp if a CUDA llama-server exists
#   MODELS_DIR        default $HOME/models
set -uo pipefail

WHICH="${1:-both}"
RUNS="${2:-4}"
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
PORT=8464
mkdir -p "${MODELS_DIR}"

log() { echo "[bootstrap $(date +%H:%M:%S)] $*"; }

# ── 1. System deps + rust + rust-analyzer ───────────────────────────────
log "Installing system packages..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq \
    build-essential cmake git curl pkg-config libssl-dev python3-pip libgomp1 ca-certificates >/dev/null

if ! command -v cargo >/dev/null; then
    log "Installing rust..."
    curl -fsSL https://sh.rustup.rs | sh -s -- -y --default-toolchain stable >/dev/null
fi
# shellcheck disable=SC1091
source "$HOME/.cargo/env" 2>/dev/null || export PATH="$HOME/.cargo/bin:$PATH"

if ! command -v rust-analyzer >/dev/null; then
    log "Installing rust-analyzer..."
    curl -fsSL https://github.com/rust-lang/rust-analyzer/releases/latest/download/rust-analyzer-x86_64-unknown-linux-gnu.gz \
        | gunzip -c > /usr/local/bin/rust-analyzer && chmod +x /usr/local/bin/rust-analyzer
fi

# ── 2. CUDA llama-server (use prebuilt if present, else build) ───────────
LLAMA_BIN="${LLAMA_SERVER_BIN:-}"
if [[ -z "${LLAMA_BIN}" ]] && command -v llama-server >/dev/null; then
    LLAMA_BIN="$(command -v llama-server)"
fi
if [[ -z "${LLAMA_BIN}" ]]; then
    log "Building llama.cpp with CUDA (this takes a few minutes)..."
    if [[ ! -d "$HOME/llama.cpp" ]]; then
        git clone --depth 1 https://github.com/ggml-org/llama.cpp "$HOME/llama.cpp" >/dev/null 2>&1
    fi
    ( cd "$HOME/llama.cpp" \
        && cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null \
        && cmake --build build --config Release -j --target llama-server >/dev/null )
    LLAMA_BIN="$HOME/llama.cpp/build/bin/llama-server"
fi
[[ -x "${LLAMA_BIN}" ]] || { log "ERROR: no llama-server binary (set LLAMA_SERVER_BIN)"; exit 1; }
log "llama-server: ${LLAMA_BIN}"

# ── 3. Model download helper ────────────────────────────────────────────
pip install -q -U "huggingface_hub[cli]" >/dev/null 2>&1 || true
hf_dl() {  # repo, include-glob, dest
    local repo="$1" inc="$2" dest="$3"
    log "Downloading ${repo} (${inc}) → ${dest}"
    if command -v hf >/dev/null; then
        HF_TOKEN="${HF_TOKEN:-}" hf download "${repo}" --include "${inc}" --local-dir "${dest}"
    else
        HF_TOKEN="${HF_TOKEN:-}" huggingface-cli download "${repo}" --include "${inc}" --local-dir "${dest}"
    fi
}

GEMMA_DIR="${MODELS_DIR}/gemma-4-26B-A4B-it-GGUF"
QWEN_DIR="${MODELS_DIR}/Qwen3-Coder-Next-GGUF"

find_gguf() {  # dir, then patterns... → echoes first matching first-shard/single file
    local dir="$1"; shift
    local p f
    for p in "$@"; do
        f=$(find "${dir}" -maxdepth 2 -name "${p}" 2>/dev/null | sort | head -1)
        [[ -n "${f}" ]] && { echo "${f}"; return; }
    done
}

# ── 4. Server start/stop ────────────────────────────────────────────────
SERVER_PID=""
stop_server() { [[ -n "${SERVER_PID}" ]] && kill "${SERVER_PID}" 2>/dev/null; SERVER_PID=""; sleep 3; }
trap stop_server EXIT

wait_ready() {
    local i=0
    until curl -fsS --max-time 3 "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q '"id"'; do
        i=$((i+1)); [ "$i" -gt 180 ] && { log "ERROR: server not ready after ~15min"; return 1; }
        sleep 5
    done
    log "server ready: $(curl -s http://localhost:${PORT}/v1/models | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)"
}

start_gemma() {
    local m; m=$(find_gguf "${GEMMA_DIR}" 'gemma-4-26B-A4B-it-UD-Q4_K_M*-00001-of-*.gguf' 'gemma-4-26B-A4B-it-UD-Q4_K_M*.gguf' 'gemma-4-26B-A4B-it*.gguf')
    [[ -n "${m}" ]] || { log "ERROR: no gemma gguf in ${GEMMA_DIR}"; return 1; }
    log "starting gemma server: ${m}"
    "${LLAMA_BIN}" --jinja --reasoning-budget 2000 -m "${m}" -c 60000 \
        --cache-type-k q4_0 --cache-type-v q4_0 -ngl 99 --flash-attn on \
        --temp 1.0 --top-p 0.95 --top-k 64 -np 1 --port "${PORT}" \
        > "$HOME/llama-gemma.log" 2>&1 &
    SERVER_PID=$!; wait_ready
}

start_qwen() {
    local m; m=$(find_gguf "${QWEN_DIR}" 'Qwen3-Coder-Next-*UD-Q4_K_XL*-00001-of-*.gguf' 'Qwen3-Coder-Next-*UD-Q4_K_XL*.gguf' 'Qwen3-Coder-Next-*-00001-of-*.gguf' 'Qwen3-Coder-Next-*.gguf')
    [[ -n "${m}" ]] || { log "ERROR: no qwen gguf in ${QWEN_DIR}"; return 1; }
    log "starting qwen server (experts on CPU): ${m}"
    "${LLAMA_BIN}" --jinja --model "${m}" --ctx-size 60000 \
        --cache-type-k q4_0 --cache-type-v q4_0 --n-gpu-layers 99 \
        --override-tensor "([0-9]+).ffn_.*_exps.=CPU" --flash-attn on \
        --threads 16 --threads-batch 32 --batch-size 2048 --mlock \
        --temp 1.0 --top-p 0.95 --top-k 40 --repeat-penalty 1.1 \
        -np 1 --port "${PORT}" --metrics \
        > "$HOME/llama-qwen.log" 2>&1 &
    SERVER_PID=$!; wait_ready
}

run_bench() {  # model-label, timeout, max-rounds
    log "running native bench: model=$1 timeout=$2 rounds=$3 runs=${RUNS}"
    ( cd "${REPO_DIR}" && GATE_CONTEXT_RESET=false bash scripts/run-benchmark-native.sh \
        --model "$1" --timeout "$2" --max-rounds "$3" --runs "${RUNS}" )
}

# ── 5. Drive the requested model(s) ─────────────────────────────────────
do_gemma() { hf_dl unsloth/gemma-4-26B-A4B-it-GGUF '*UD-Q4_K_M*' "${GEMMA_DIR}"; start_gemma && run_bench gemma-4-26B-A4B-it 2400 80; stop_server; }
do_qwen()  { hf_dl unsloth/Qwen3-Coder-Next-GGUF '*UD-Q4_K_XL*' "${QWEN_DIR}";   start_qwen  && run_bench qwen3-coder-next 6000 600; stop_server; }

case "${WHICH}" in
    gemma) do_gemma ;;
    qwen)  do_qwen ;;
    both)  do_gemma; do_qwen ;;
    *) log "unknown target '${WHICH}' (use gemma|qwen|both)"; exit 1 ;;
esac

log "DONE. Results under ${REPO_DIR}/benchmark_results/"
