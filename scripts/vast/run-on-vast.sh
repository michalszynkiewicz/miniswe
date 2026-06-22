#!/usr/bin/env bash
# run-on-vast.sh — provision a Vast.ai 3090, run the bench, fetch results, destroy.
#
# Cost-safety is layered (see docs/vast-setup.md):
#   1. PREPAID balance on your account = absolute hard cap (load only what a run costs).
#   2. trap … EXIT here destroys the instance on success/error/Ctrl-C.
#   3. On-VM dead-man's switch destroys the instance after MAX_HOURS even if THIS
#      machine dies (it calls the Vast API from inside the box).
#
# Prereqs (see docs/vast-setup.md): `pip install vastai`, `vastai set api-key <KEY>`,
# an SSH key registered on your Vast account, and VAST_API_KEY exported.
#
# Usage:
#   export VAST_API_KEY=...            # also used by the on-VM dead-man's switch
#   scripts/vast/run-on-vast.sh both   # or: gemma | qwen
set -uo pipefail

# ── Config (override via env) ───────────────────────────────────────────
WHICH="${1:-both}"
RUNS="${RUNS:-4}"
GPU_NAME="${GPU_NAME:-RTX_3090}"
MIN_RAM_GB="${MIN_RAM_GB:-64}"          # qwen's CPU-expert offload needs ~50GB
DISK_GB="${DISK_GB:-180}"               # models (~61GB) + builds + headroom
MIN_RELIABILITY="${MIN_RELIABILITY:-0.985}"
MAX_DPH="${MAX_DPH:-0.50}"              # don't auto-rent above $/hr
MAX_HOURS="${MAX_HOURS:-12}"           # on-VM dead-man's switch
# CUDA *devel* image (nvcc present) so bootstrap can build CUDA llama.cpp.
IMAGE="${IMAGE:-nvidia/cuda:12.4.1-devel-ubuntu22.04}"
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
HF_TOKEN="${HF_TOKEN:-}"

command -v vastai >/dev/null || { echo "ERROR: vastai CLI not found — pip install vastai" >&2; exit 1; }
[[ -n "${VAST_API_KEY:-}" ]] || { echo "ERROR: export VAST_API_KEY=... first" >&2; exit 1; }

INSTANCE_ID=""
destroy() {
    if [[ -n "${INSTANCE_ID}" ]]; then
        echo ">>> destroying instance ${INSTANCE_ID}"
        vastai destroy instance "${INSTANCE_ID}" 2>/dev/null || true
        INSTANCE_ID=""
    fi
}
trap destroy EXIT INT TERM

# ── 1. Find the cheapest qualifying VERIFIED host ───────────────────────
# NOTE (first run): sanity-check these field names against `vastai search offers --help`
# — the CLI query DSL occasionally changes. cpu_ram is GB; gpu_ram is MB.
QUERY="gpu_name=${GPU_NAME} num_gpus=1 cpu_ram>=${MIN_RAM_GB} disk_space>=${DISK_GB} reliability>=${MIN_RELIABILITY} verified=true rentable=true"
echo ">>> searching: ${QUERY}"
OFFER_JSON="$(vastai search offers "${QUERY}" -o 'dph_total' --raw 2>/dev/null)"
# JSON on stdin (via the pipe), MAX_DPH as argv[1] — no stdin collision.
read -r OFFER_ID OFFER_DPH <<<"$(printf '%s' "${OFFER_JSON}" | python3 -c '
import json,sys
maxd=float(sys.argv[1])
data=json.load(sys.stdin)
best=None
for o in data:
    dph=o.get("dph_total") or 1e9
    if dph<=maxd and (best is None or dph<best["dph_total"]):
        best=o
print(str(best["id"])+" "+str(best["dph_total"]) if best else "")
' "${MAX_DPH}")"
[[ -n "${OFFER_ID:-}" ]] || { echo "ERROR: no verified ${GPU_NAME} with >=${MIN_RAM_GB}GB RAM under \$${MAX_DPH}/hr. Loosen MAX_DPH/MIN_RAM_GB." >&2; exit 1; }
echo ">>> chosen offer ${OFFER_ID} @ \$${OFFER_DPH}/hr"

# ── 2. Create the instance ──────────────────────────────────────────────
CREATE_JSON="$(vastai create instance "${OFFER_ID}" --image "${IMAGE}" --disk "${DISK_GB}" --ssh --direct --raw 2>/dev/null)"
INSTANCE_ID="$(echo "${CREATE_JSON}" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("new_contract") or d.get("id") or "")')"
[[ -n "${INSTANCE_ID}" ]] || { echo "ERROR: create failed: ${CREATE_JSON}" >&2; exit 1; }
echo ">>> created instance ${INSTANCE_ID} (image ${IMAGE}, ${DISK_GB}GB)"

# ── 3. Wait for SSH ─────────────────────────────────────────────────────
echo ">>> waiting for instance to come up..."
SSH_HOST=""; SSH_PORT=""
for _ in $(seq 1 120); do
    # Filter `show instances` by our id (a singular `show instance` isn't reliable across CLI versions).
    read -r STATUS SSH_HOST SSH_PORT <<<"$(vastai show instances --raw 2>/dev/null | python3 -c '
import json,sys
iid=int(sys.argv[1]);
data=json.load(sys.stdin)
for o in data:
    if int(o.get("id",-1))==iid:
        print(o.get("actual_status",""), o.get("ssh_host","") or "", o.get("ssh_port","") or "")
        break
' "${INSTANCE_ID}")"
    [[ "${STATUS}" == "running" && -n "${SSH_HOST}" && -n "${SSH_PORT}" ]] && break
    sleep 10
done
[[ -n "${SSH_HOST}" && -n "${SSH_PORT}" ]] || { echo "ERROR: SSH never came up" >&2; exit 1; }
SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p ${SSH_PORT} root@${SSH_HOST}"
echo ">>> ssh ready: root@${SSH_HOST}:${SSH_PORT}"
# settle SSHd
for _ in $(seq 1 30); do ${SSH} true 2>/dev/null && break; sleep 5; done

# ── 4. Sync the repo (current tree, incl. .git for the pinned-SHA checkout) ──
echo ">>> syncing repo..."
rsync -az --delete \
    --exclude target --exclude benchmark_results --exclude 'benchmark_results_vast' \
    -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p ${SSH_PORT}" \
    "${REPO_DIR}/" "root@${SSH_HOST}:/root/miniswe/"

# ── 5. On-VM dead-man's switch (independent of this machine) ─────────────
# Destroys the instance via the Vast REST API after MAX_HOURS no matter what.
# NOTE (first run): verify the destroy endpoint against current Vast API docs.
${SSH} "cat > /root/selfdestruct.sh" <<EOF
#!/bin/bash
sleep \$(( ${MAX_HOURS} * 3600 ))
curl -s -X DELETE "https://console.vast.ai/api/v0/instances/${INSTANCE_ID}/" \
     -H "Authorization: Bearer ${VAST_API_KEY}" >/dev/null 2>&1
EOF
${SSH} "chmod +x /root/selfdestruct.sh && nohup /root/selfdestruct.sh >/dev/null 2>&1 &" || true
echo ">>> dead-man's switch armed (${MAX_HOURS}h)"

# ── 6. Run the bench detached (survives SSH drops), poll for completion ──
echo ">>> launching bench (${WHICH}, ${RUNS} runs each)..."
${SSH} "cd /root/miniswe && HF_TOKEN='${HF_TOKEN}' nohup bash scripts/vast/bootstrap-vast.sh ${WHICH} ${RUNS} > /root/bench.log 2>&1 & echo started"
echo ">>> tailing /root/bench.log (Ctrl-C is safe — instance still tears down)"
while true; do
    ${SSH} "tail -n 5 /root/bench.log 2>/dev/null; echo ---; grep -q 'DONE. Results' /root/bench.log && echo BENCH_DONE || true" 2>/dev/null | sed 's/^/[vm] /'
    ${SSH} "grep -q 'DONE. Results' /root/bench.log" 2>/dev/null && break
    sleep 60
done

# ── 7. Pull results back, then destroy (trap also covers this) ───────────
echo ">>> fetching results..."
mkdir -p "${REPO_DIR}/benchmark_results_vast"
rsync -az -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p ${SSH_PORT}" \
    "root@${SSH_HOST}:/root/miniswe/benchmark_results/" "${REPO_DIR}/benchmark_results_vast/" || true
${SSH} "tail -n 40 /root/bench.log" 2>/dev/null | sed 's/^/[vm] /'

echo ">>> done — results in ${REPO_DIR}/benchmark_results_vast/"
# trap destroys the instance now.
