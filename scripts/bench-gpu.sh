#!/usr/bin/env bash
# bench-gpu.sh — GPU telemetry + llama-server provenance for bench drivers.
# Source this, don't run it directly.
#
# Why this exists (docs/gpu-hardening.md items 2 and 5):
#
#   The Xid 79 that killed the 2026-08-27 run was undiagnosable. Nothing
#   recorded the card's state during the run — no power, clock, or
#   temperature trace existed for the hours before it fell off the bus, so
#   there was no way to tell a thermal event from a driver fault from a
#   power-delivery fault. Every bench driver should leave that evidence.
#
#   Separately, CLAUDE.md requires a fresh llama-server per run, because a
#   long-lived one makes server uptime an uncontrolled variable (GPU memory
#   fragmentation, thermal state) across arms compared to each other. The
#   drivers cannot start the server, but they can record which instance
#   served the run and shout when the same one already served an earlier one.
#
# Usage, in a driver that has already created RESULTS_DIR:
#
#     source "$(dirname "$0")/bench-gpu.sh"
#     gpu_bench_start "${RESULTS_DIR}"     # after the server-reachable check
#     ...
#     gpu_bench_finish "${RESULTS_DIR}"    # in the summary block
#
#   and add `gpu_telemetry_stop` to the driver's cleanup trap so an
#   interrupted run doesn't leave the sampler behind.
#
# Env knobs:
#   GPU_SAMPLE_INTERVAL     seconds between samples (default 10; 0 = off)
#   LLAMA_CONTAINER_FILTER  docker name filter (default: llama-server-)

GPU_TELEMETRY_PID=""
GPU_RUN_START_STAMP=""
GPU_SAMPLE_INTERVAL="${GPU_SAMPLE_INTERVAL:-10}"

# Stop the sampler. Safe to call repeatedly and from a trap.
gpu_telemetry_stop() {
    [[ -n "${GPU_TELEMETRY_PID}" ]] || return 0
    kill "${GPU_TELEMETRY_PID}" >/dev/null 2>&1 || true
    wait "${GPU_TELEMETRY_PID}" 2>/dev/null || true
    GPU_TELEMETRY_PID=""
    return 0
}

# Begin sampling and record which llama-server is about to serve this run.
#   $1 — results directory (must already exist)
gpu_bench_start() {
    local results_dir="$1"
    local repo_dir="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
    local endpoint="${LLAMA_ENDPOINT:-${LLM_ENDPOINT:-http://localhost:8464}}"

    GPU_RUN_START_STAMP="$(date '+%Y-%m-%d %H:%M:%S')"

    if [[ "${GPU_SAMPLE_INTERVAL}" == "0" ]]; then
        echo "GPU:      telemetry disabled (GPU_SAMPLE_INTERVAL=0)"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        # One-shot full dump: driver version, power limit, persistence mode,
        # ECC state — the static context you want when reading the trace back.
        nvidia-smi -q > "${results_dir}/gpu-info.txt" 2>&1 || true
        nvidia-smi \
            --query-gpu=timestamp,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,fan.speed,utilization.gpu,memory.used,clocks_event_reasons.active \
            --format=csv,nounits -l "${GPU_SAMPLE_INTERVAL}" \
            > "${results_dir}/gpu-telemetry.csv" 2>/dev/null &
        GPU_TELEMETRY_PID=$!
        echo "GPU:      sampling every ${GPU_SAMPLE_INTERVAL}s -> gpu-telemetry.csv"
    else
        echo "GPU:      nvidia-smi not found — NO telemetry for this run" >&2
    fi

    local filter container started uptime prior
    filter="${LLAMA_CONTAINER_FILTER:-llama-server-}"
    container="$(docker ps --filter "name=${filter}" --format '{{.Names}}' 2>/dev/null | head -1 || true)"
    if [[ -z "${container}" ]]; then
        echo "Server:   no container matching '${filter}' — provenance not recorded" >&2
        return 0
    fi

    started="$(docker inspect -f '{{.State.StartedAt}}' "${container}" 2>/dev/null || echo '')"
    uptime="$(( $(date +%s) - $(date -d "${started:-now}" +%s 2>/dev/null || date +%s) ))"
    {
        echo "container: ${container}"
        echo "started:   ${started}"
        echo "uptime_s:  ${uptime}"
        echo "endpoint:  ${endpoint}"
        echo "model:     ${MODEL_TAG:-unknown}"
    } > "${results_dir}/llama-server.txt"
    echo "Server:   ${container} (up ${uptime}s)"

    prior="$(grep -l "^container: ${container}$" \
        "${repo_dir}"/benchmark_results/*/llama-server.txt 2>/dev/null \
        | grep -vF "${results_dir}/" | head -3 || true)"
    if [[ -n "${prior}" ]]; then
        echo "" >&2
        echo "!! WARNING: llama-server '${container}' already served:" >&2
        echo "${prior}" | sed 's|^|     |' >&2
        echo "   CLAUDE.md requires a restart between runs — otherwise server" >&2
        echo "   uptime is an uncontrolled variable across the comparison." >&2
        echo "" >&2
    fi
    # Both this function and gpu_bench_finish end in an `if`. Without an
    # explicit success return, a false condition is the function's exit
    # status and `set -e` in the caller aborts the whole run.
    return 0
}

# Stop sampling and print what the trace and the kernel log say.
#   $1 — results directory
gpu_bench_finish() {
    local results_dir="$1"
    gpu_telemetry_stop

    if [[ -s "${results_dir}/gpu-telemetry.csv" ]]; then
        echo ""
        # Bit values are NVML's clocksEventReasons; the ones that explain a
        # slow or dead run are the SW power cap and the three HW slowdowns.
        awk -F', *' '
            NR > 1 && NF >= 10 && $1 !~ /^timestamp/ {
                n++
                if ($5 + 0 > maxp) maxp = $5 + 0
                if ($6 + 0 > maxt) maxt = $6 + 0
                if ($4 + 0 > maxm) maxm = $4 + 0
                ev = strtonum($10)
                if (and(ev, 0x4))  sw_cap++
                if (and(ev, 0x8))  hw_slow++
                if (and(ev, 0x40)) hw_therm++
                if (and(ev, 0x80)) hw_brake++
            }
            END {
                if (n == 0) { print "GPU:  telemetry file has no samples"; exit }
                printf "GPU:  %d samples | peak %.0f W | peak %d C | peak mem %d MHz\n", n, maxp, maxt, maxm
                if (sw_cap)   printf "      SW power cap active in %d samples\n", sw_cap
                if (hw_slow)  printf "      !! HW slowdown in %d samples\n", hw_slow
                if (hw_therm) printf "      !! HW thermal slowdown in %d samples\n", hw_therm
                if (hw_brake) printf "      !! HW power brake in %d samples\n", hw_brake
            }
        ' "${results_dir}/gpu-telemetry.csv"
    fi

    # The 2026-08-27 failure left its only trace in the kernel log, which
    # nothing was reading. Surface it next to the run it belongs to.
    [[ -n "${GPU_RUN_START_STAMP}" ]] || return 0
    local xids
    xids="$(journalctl -k --since "${GPU_RUN_START_STAMP}" 2>/dev/null \
        | grep -iE 'NVRM.*Xid|fallen off the bus' || true)"
    if [[ -n "${xids}" ]]; then
        echo "${xids}" > "${results_dir}/gpu-xid.txt"
        echo ""
        echo "!! NVIDIA Xid / bus errors during this run (see gpu-xid.txt):"
        echo "${xids}" | tail -10 | sed 's|^|   |'
    fi
    return 0
}
