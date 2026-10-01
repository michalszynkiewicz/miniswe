#!/usr/bin/env bash
# run-model-round.sh — one comparable benchmark round across the whole model
# field: start each llama-server, run the bench, probe idle decode, tear the
# server down, repeat.
#
# The point of a "round" is comparability: every run in it uses the same
# miniswe binary, the same shipped defaults, a fresh llama-server, and the
# same task/timeout. That is what makes a headline table honest. Anything
# that would break that (a dirty tree mid-round, a dead GPU, the wrong model
# answering on 8464) aborts instead of quietly producing a row.
#
# NEVER EDIT THIS FILE WHILE A ROUND IS IN FLIGHT. bash re-reads a running
# script by byte offset; an edit mid-run corrupts the parse (this already
# killed one run — see .plan/final-round-tally.md).
#
# Usage:
#   ./scripts/run-model-round.sh [options]
#
#   --dry-run             print the queue and exit (no server, no bench)
#   --only a,b,c          restrict to these model keys (see the table below)
#   --skip a,b            drop these model keys
#   --order pass|group    pass  = round-robin: every model once, then again
#                                 (default; spreads thermal drift, and a
#                                 partial night still covers the field)
#                         group = all N runs of a model back to back
#   --runs-promising N    runs per promising model  (default 3)
#   --runs-other N        runs per dropped model    (default 1)
#   --timeout SECS        bench deadline per run    (default 3400)
#   --max-rounds N        agent round cap           (default 600)
#   --cooldown SECS       idle gap between runs     (default 60)
#   --require-clean       refuse to start if compiled sources are uncommitted
#                         (default: warn, record the diff, and run anyway)
#   --resume DIR          append to an existing round dir instead of a new one
#
# Output: benchmark_results/round_<stamp>/
#   queue.tsv        one row per planned run, filled in as it completes
#   round.md         human-readable tally, rewritten after every run
#   provenance.txt   HEAD, tree state, agent binary sha256, llama image, host
#   <NN>-<key>/      per-run: build.log, binary.sha256, server.log, props.json,
#                    probe.txt, bench.log
#
# The bench result dirs themselves stay where run-benchmark-docker.sh puts
# them (benchmark_results/docker_<stamp>_<model>/); queue.tsv links to them.

set -uo pipefail   # NOT -e: one failed run must not kill the queue

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_DIR}"

ENDPOINT="${LLAMA_ENDPOINT:-http://localhost:8464}"

# ── Model registry ──────────────────────────────────────────────────────
#
# tier:     promising = earns the multi-run treatment (has produced 6/6)
#           other     = on the bench to confirm a known negative
# thinking: the arm this row runs. Most models get exactly one arm, because a
#           thinking/instruct split is a SECOND variable (miniswe raises
#           temperature to 0.6 with thinking, src/config/model.rs:236) — but
#           gemma4 and Laguna XS are the only two models where BOTH arms are
#           known-good, so they are the only place that question can be
#           answered at all, and they get a row each way.
# expect:   substring the server's /v1/models id must contain. Guards against
#           a launcher quant fallback or a stale server on 8464.
# loadsecs: how long to wait for /health before calling it a failed start.
# env:      extra launcher environment.
#
# key          tier       launcher                     thinking expect                  loadsecs env
MODELS=(
"gemma4      | promising | ./start-gemma4.sh           | false | gemma-4-26B-A4B        | 600  |"
"laguna-xs   | promising | ./start-laguna-xs.sh        | false | Laguna-XS-2.1          | 600  |"
"gemma4-think| promising | ./start-gemma4.sh           | true  | gemma-4-26B-A4B        | 600  |"
"lagxs-think | promising | ./start-laguna-xs.sh        | true  | Laguna-XS-2.1          | 600  |"
"devstral    | promising | ./start-devstral-small-2.sh | false | Devstral-Small-2       | 600  |"
"glimmer     | promising | ./start-muse-glimmer-30b.sh | true  | Muse-Glimmer-30B       | 600  |"
"glim-dflash | promising | ./start-muse-glimmer-30b-dflash.sh | true | Muse-Glimmer-30B  | 600  |"
"laguna-s    | promising | ./start-laguna-s.sh         | false | Laguna-S-2.1           | 1800 |"
"gpt-oss     | other     | ./start-gpt-oss-20b.sh      | false | gpt-oss-20b            | 600  |"
"north       | other     | ./start-north-mini-code.sh  | true  | North-Mini-Code        | 900  | MINISWE_REASONING=true"
"nemotron    | other     | ./start-nemotron35-30b.sh   | false | Nemotron-3.5-Lightning | 900  |"
"mistral4    | other     | ./start-mistral-small-4.sh  | false | Mistral-Small-4        | 1800 |"
)

# Notes on the non-obvious arms:
#   *-think  — gemma4 and Laguna XS both have a clean thinking record: gemma
#              5 runs 5x 6/6 (456-1214s, 4 first-attempt), Laguna XS 4 runs
#              4x 6/6 (481/516s post-cliff-fix, i.e. FASTER than its own
#              instruct arm at 514-1550s). They are cheap, and n=3 vs n=3 on
#              one binary is the closest this bench can get to answering
#              "does thinking help" — the temp 0.2->0.6 coupling still rides
#              along, so report it as a two-variable result.
#   glimmer  — reasoning cannot be disabled in its template, so THINKING=true
#              is the only self-consistent arm (thinking=false runs asked for
#              temp 0.2 on a model that thought anyway: 3431s/3427s vs
#              571s/657s properly configured).
#   glim-dflash — the SAME arm as glimmer with the DFlash drafter attached
#              (speculative decoding). Its launcher differs from glimmer's by
#              exactly five --spec-draft-* flags plus ctx 80000 instead of
#              131072, so the pair is a clean A/B on decode speed: scores
#              should be statistically indistinguishable and only wall time
#              should move. It is tier=promising, so a plain full round now
#              runs it 3x — pass --skip glim-dflash for the old field.
#   north    — instruct scored 0/6 (235 reads, zero edits). MINISWE_REASONING
#              is the launcher's template kwarg; THINKING is miniswe's own
#              flag. Both are needed; neither alone switches the model.
#   nemotron — instruct only. The patched chat template that stops the
#              newline flood exists for the non-thinking branch; the one
#              thinking run drowned in a 13,005-line flood.

field() { echo "$1" | awk -F'|' -v n="$2" '{gsub(/^[ \t]+|[ \t]+$/,"",$n); print $n}'; }
lookup() {
    local key="$1" n="$2" row
    for row in "${MODELS[@]}"; do
        [[ "$(field "$row" 1)" == "$key" ]] && { field "$row" "$n"; return 0; }
    done
    return 1
}

# ── Options ─────────────────────────────────────────────────────────────
DRY_RUN=0; ONLY=""; SKIP=""; ORDER="pass"
RUNS_PROMISING=3; RUNS_OTHER=1
TIMEOUT=3400; MAX_ROUNDS=600; COOLDOWN=60
REQUIRE_CLEAN=0; ROUND_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)          DRY_RUN=1; shift ;;
        --only)             ONLY="$2"; shift 2 ;;
        --skip)             SKIP="$2"; shift 2 ;;
        --order)            ORDER="$2"; shift 2 ;;
        --runs-promising)   RUNS_PROMISING="$2"; shift 2 ;;
        --runs-other)       RUNS_OTHER="$2"; shift 2 ;;
        --timeout)          TIMEOUT="$2"; shift 2 ;;
        --max-rounds)       MAX_ROUNDS="$2"; shift 2 ;;
        --cooldown)         COOLDOWN="$2"; shift 2 ;;
        --require-clean)    REQUIRE_CLEAN=1; shift ;;
        --resume)           ROUND_DIR="$2"; shift 2 ;;
        -h|--help)          sed -n '2,45p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

[[ "$ORDER" == "pass" || "$ORDER" == "group" ]] || { echo "--order must be pass|group" >&2; exit 2; }

in_list() { local needle="$1" list="$2"; [[ ",$list," == *",$needle,"* ]]; }

SELECTED=()
for row in "${MODELS[@]}"; do
    key="$(field "$row" 1)"
    [[ -n "$ONLY" ]] && ! in_list "$key" "$ONLY" && continue
    [[ -n "$SKIP" ]] &&   in_list "$key" "$SKIP" && continue
    SELECTED+=("$key")
done
[[ ${#SELECTED[@]} -gt 0 ]] || { echo "no models selected" >&2; exit 2; }

# ── Build the queue ─────────────────────────────────────────────────────
# Promising models first in every pass; the dropped models go last, because
# they reliably burn the full timeout and are the runs you can afford to
# lose if the card falls off the bus overnight (Xid 79 has hit twice, both
# times 6-8h into sustained MoE load).
QUEUE=()
promising=(); other=()
for key in "${SELECTED[@]}"; do
    if [[ "$(lookup "$key" 2)" == "promising" ]]; then promising+=("$key"); else other+=("$key"); fi
done

if [[ "$ORDER" == "pass" ]]; then
    for ((p=1; p<=RUNS_PROMISING; p++)); do
        for key in "${promising[@]}"; do QUEUE+=("$key"); done
    done
    for ((p=1; p<=RUNS_OTHER; p++)); do
        for key in "${other[@]}"; do QUEUE+=("$key"); done
    done
else
    for key in "${promising[@]}"; do
        for ((p=1; p<=RUNS_PROMISING; p++)); do QUEUE+=("$key"); done
    done
    for key in "${other[@]}"; do
        for ((p=1; p<=RUNS_OTHER; p++)); do QUEUE+=("$key"); done
    done
fi

if [[ "$DRY_RUN" == "1" ]]; then
    echo "=== Round plan (${#QUEUE[@]} runs, order=${ORDER}) ==="
    printf '%-4s %-12s %-10s %-9s %s\n' "#" "model" "tier" "thinking" "launcher"
    i=0
    for key in "${QUEUE[@]}"; do
        i=$((i+1))
        printf '%-4s %-12s %-10s %-9s %s\n' "$i" "$key" \
            "$(lookup "$key" 2)" "$(lookup "$key" 4)" "$(lookup "$key" 3)"
    done
    echo
    echo "Worst case: ${#QUEUE[@]} x ${TIMEOUT}s bench + load/cooldown"
    echo "            = $(( (${#QUEUE[@]} * (TIMEOUT + COOLDOWN + 180)) / 3600 ))h if every run hits the timeout"
    exit 0
fi

# ── Pre-flight ──────────────────────────────────────────────────────────
fail() { echo "" >&2; echo "PRE-FLIGHT FAILED: $*" >&2; exit 1; }

command -v docker      >/dev/null || fail "docker not found"
command -v nvidia-smi  >/dev/null || fail "nvidia-smi not found — GPU health cannot be checked"
nvidia-smi -L >/dev/null 2>&1     || fail "nvidia-smi -L failed — the GPU is not healthy right now"

# Uncommitted changes are recorded, not forbidden. Only files cargo actually
# compiles can change the benched binary — a dirty posts/*.md or .plan/ note
# cannot, and refusing to start over one is just in the way.
DIRTY="$(git status --porcelain)"
CODE_DIRTY="$(echo "$DIRTY" | grep -E '^.{2} *(src/|Cargo\.toml|Cargo\.lock|build\.rs|scripts/Dockerfile\.benchmark)' || true)"
if [[ -n "$CODE_DIRTY" ]]; then
    echo ""
    echo "!!! WARNING: uncommitted CODE changes — the benched binary is not HEAD."
    echo "$CODE_DIRTY" | sed 's/^/    /'
    echo "    (Dockerfile.benchmark does COPY . . and builds from the working tree.)"
    echo "    Full diff saved to the round dir as uncommitted.patch."
    echo ""
    if [[ "$REQUIRE_CLEAN" == "1" ]]; then
        echo "--require-clean was passed; stopping." >&2
        exit 1
    fi
fi

for key in "${SELECTED[@]}"; do
    launcher="$(lookup "$key" 3)"
    [[ -x "$launcher" ]] || fail "launcher not executable: $launcher (model $key)"
done

STAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -n "$ROUND_DIR" ]]; then
    # Resuming: keep the original round's identity in the tally header.
    STAMP="$(basename "$ROUND_DIR" | sed 's/^round_//')"
else
    ROUND_DIR="${REPO_DIR}/benchmark_results/round_${STAMP}"
fi
mkdir -p "$ROUND_DIR"
QUEUE_TSV="${ROUND_DIR}/queue.tsv"
[[ -f "$QUEUE_TSV" ]] || printf 'idx\tmodel\ttier\tthinking\tstarted\twall_s\tresult\tattempts\tdecode1\tdecode2\trundir\n' > "$QUEUE_TSV"

{
    echo "round      ${STAMP}"
    echo "host       $(hostname)"
    echo "head       $(git rev-parse HEAD)  $(git log -1 --format=%s)"
    echo "tree       $([[ -n "$CODE_DIRTY" ]] && echo "DIRTY (code)" || { [[ -n "$DIRTY" ]] && echo "dirty (non-code only)" || echo clean; })"
    echo "llama img  ${LLAMA_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda13}"
    echo "timeout    ${TIMEOUT}s   max-rounds ${MAX_ROUNDS}   cooldown ${COOLDOWN}s"
    echo "order      ${ORDER}   promising x${RUNS_PROMISING}   other x${RUNS_OTHER}"
    echo "queue      ${QUEUE[*]}"
    echo "gpu        $(nvidia-smi --query-gpu=name,power.limit --format=csv,noheader)"
} > "${ROUND_DIR}/provenance.txt"
# Provenance, not a gate: record exactly what the binary was built from.
# `git diff HEAD` misses untracked files entirely, and an untracked src/*.rs
# is precisely the kind of change that silently redefines a round — so grab
# those separately.
if [[ -n "$DIRTY" ]]; then
    echo "$DIRTY" > "${ROUND_DIR}/git-status.txt"
    git diff HEAD > "${ROUND_DIR}/uncommitted.patch" 2>/dev/null
    untracked="$(git ls-files --others --exclude-standard -- src Cargo.toml Cargo.lock build.rs)"
    if [[ -n "$untracked" ]]; then
        echo "$untracked" | tar -czf "${ROUND_DIR}/untracked-src.tgz" -T - 2>/dev/null
    fi
fi

# ── Server lifecycle ────────────────────────────────────────────────────
SERVER_PID=""
SERVER_NAME=""

kill_servers() {
    [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" >/dev/null 2>&1
    docker ps --format '{{.Names}}' 2>/dev/null \
        | grep -E 'llama-server-' \
        | xargs -r docker rm -f >/dev/null 2>&1
    SERVER_PID=""; SERVER_NAME=""
    return 0
}

on_exit() { echo ""; echo "[round] cleaning up..."; kill_servers; }
trap on_exit EXIT INT TERM

# start_server <key> <logfile>  → 0 ready, 1 failed
start_server() {
    local key="$1" log="$2"
    local launcher expect loadsecs extra_env
    launcher="$(lookup "$key" 3)"
    expect="$(lookup "$key" 5)"
    loadsecs="$(lookup "$key" 6)"
    extra_env="$(lookup "$key" 7)"

    kill_servers
    sleep 3   # let the driver release VRAM before the next allocation

    SERVER_NAME="llama-server-round-$$-$(date +%s)"
    # shellcheck disable=SC2086  # extra_env is intentionally word-split
    env $extra_env LLAMA_CONTAINER_NAME="$SERVER_NAME" "$launcher" > "$log" 2>&1 &
    SERVER_PID=$!

    local waited=0
    while (( waited < loadsecs )); do
        if curl -fsS --max-time 3 "${ENDPOINT}/health" >/dev/null 2>&1; then
            local id
            id="$(curl -fsS --max-time 5 "${ENDPOINT}/v1/models" \
                  | python3 -c 'import json,sys; print((json.load(sys.stdin).get("data") or [{}])[0].get("id",""))' 2>/dev/null)"
            if [[ "$id" != *"$expect"* ]]; then
                echo "[round] WRONG MODEL on ${ENDPOINT}: expected '*${expect}*', got '${id}'" | tee -a "$log"
                return 1
            fi
            curl -fsS --max-time 5 "${ENDPOINT}/props" > "${log%.log}-props.json" 2>/dev/null
            echo "[round] ${key} ready after ${waited}s — ${id}"
            return 0
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[round] launcher for ${key} exited before serving; tail of ${log}:" >&2
            tail -20 "$log" >&2
            return 1
        fi
        sleep 5; waited=$((waited + 5))
    done
    echo "[round] ${key} did not answer /health within ${loadsecs}s" >&2
    tail -20 "$log" >&2
    return 1
}

# idle_probe → "decode1 decode2" (tok/s), measured before the server dies.
# Same prompt and n_predict as every historical probe in the tally, so the
# numbers stay comparable to the ones already published.
idle_probe() {
    local out=""
    for _ in 1 2; do
        local v
        v="$(curl -fsS --max-time 120 "${ENDPOINT}/completion" \
             -d '{"prompt":"Write a detailed explanation of how a hash map works, covering buckets, collisions, and resizing.","n_predict":200,"temperature":0.2}' \
             | python3 -c 'import sys,json; t=json.load(sys.stdin)["timings"]; print(f"{t[\"predicted_per_second\"]:.1f}")' 2>/dev/null)"
        out="${out}${v:-?} "
    done
    echo "$out"
}

# ── Tally rendering ─────────────────────────────────────────────────────
render_round_md() {
    {
        echo "# Model round ${STAMP}"
        echo
        sed 's/^/    /' "${ROUND_DIR}/provenance.txt"
        echo
        echo "| # | model | arm | result | wall | attempts | decode | run dir |"
        echo "|---|---|---|---|---|---|---|---|"
        tail -n +2 "$QUEUE_TSV" | while IFS=$'\t' read -r idx model tier think started wall result att d1 d2 rundir; do
            local_arm="instruct"; [[ "$think" == "true" ]] && local_arm="thinking"
            echo "| ${idx} | ${model} | ${local_arm} | ${result} | ${wall}s | ${att} | ${d1}/${d2} | \`$(basename "${rundir}")\` |"
        done
    } > "${ROUND_DIR}/round.md"
}

# ── The queue ───────────────────────────────────────────────────────────
echo "=== Model round ${STAMP}: ${#QUEUE[@]} runs, order=${ORDER} ==="
echo "Round dir: ${ROUND_DIR}"
echo ""

IDX=0
ROUND_BINARY=""
for key in "${QUEUE[@]}"; do
    IDX=$((IDX + 1))
    tier="$(lookup "$key" 2)"
    think="$(lookup "$key" 4)"
    slot="$(printf '%02d' "$IDX")-${key}"
    slotdir="${ROUND_DIR}/${slot}"
    mkdir -p "$slotdir"

    echo "──────────────────────────────────────────────────────────────"
    echo "[$IDX/${#QUEUE[@]}] ${key} (${tier}, thinking=${think})  $(date '+%F %T')"

    # The card has fallen off the bus twice, both times hours into sustained
    # MoE load. Every further run after that is garbage, so stop the queue.
    if ! nvidia-smi -L >/dev/null 2>&1; then
        echo "[round] GPU IS GONE (nvidia-smi -L failed). Stopping the queue." | tee "${ROUND_DIR}/ABORTED.txt"
        nvidia-smi >> "${ROUND_DIR}/ABORTED.txt" 2>&1
        break
    fi

    # Build the bench image HERE, before the model is loaded.
    #
    # run-benchmark-docker.sh builds it itself, but only after the server is
    # up — so a cold `cargo build --release` burns minutes with 17 GB of
    # weights sitting resident on the card. Building first makes that a
    # cache hit by the time it gets there.
    #
    # It also pins the round. Dockerfile.benchmark is `COPY . .` + `cargo
    # build --release`, so ANY change to the build context between runs
    # invalidates the layer and recompiles — and if the change was under
    # src/, the later runs are a DIFFERENT AGENT than the earlier ones, with
    # nothing in the results saying so. Hash the binary each time and shout
    # if it moves.
    build_log="${slotdir}/build.log"
    if ! docker build -f "${REPO_DIR}/scripts/Dockerfile.benchmark" \
            -t miniswe-bench "${REPO_DIR}" > "$build_log" 2>&1; then
        echo "[round] docker build FAILED; tail of ${build_log}:" >&2
        tail -20 "$build_log" >&2
        printf '%s\t%s\t%s\t%s\t%s\t-\tBUILD-FAIL\t-\t-\t-\t-\n' \
            "$IDX" "$key" "$tier" "$think" "$(date '+%F %T')" >> "$QUEUE_TSV"
        render_round_md
        continue
    fi
    binsha="$(docker run --rm miniswe-bench sha256sum /usr/local/bin/miniswe 2>/dev/null | awk '{print $1}')"
    echo "${binsha}" > "${slotdir}/binary.sha256"
    if [[ -z "${ROUND_BINARY}" ]]; then
        ROUND_BINARY="$binsha"
        echo "round bin  ${binsha}" >> "${ROUND_DIR}/provenance.txt"
        echo "[round] agent binary ${binsha:0:12} (pinned for this round)"
    elif [[ "$binsha" != "$ROUND_BINARY" ]]; then
        echo "[round] !!! BINARY DRIFT: run ${IDX} is ${binsha:0:12}, round started on ${ROUND_BINARY:0:12}" \
            | tee -a "${ROUND_DIR}/BINARY-DRIFT.txt"
        echo "[round]     the repo changed mid-round; runs before and after this point are NOT comparable" \
            | tee -a "${ROUND_DIR}/BINARY-DRIFT.txt"
        git -C "${REPO_DIR}" status --porcelain >> "${ROUND_DIR}/BINARY-DRIFT.txt"
    fi

    started="$(date '+%F %T')"
    if ! start_server "$key" "${slotdir}/server.log"; then
        printf '%s\t%s\t%s\t%s\t%s\t-\tSERVER-FAIL\t-\t-\t-\t-\n' \
            "$IDX" "$key" "$tier" "$think" "$started" >> "$QUEUE_TSV"
        render_round_md
        kill_servers
        continue
    fi

    THINKING="$think" ./scripts/run-benchmark-docker.sh \
        --timeout "$TIMEOUT" --max-rounds "$MAX_ROUNDS" \
        2>&1 | tee "${slotdir}/bench.log"

    rundir="$(grep -m1 '^Results:' "${slotdir}/bench.log" | awk '{print $2}')"
    result="?"; wall="?"; att="?"
    if [[ -n "$rundir" && -d "$rundir" ]]; then
        final="$(grep -m1 '=== FINAL:' "${rundir}/00_baseline/container.log" 2>/dev/null)"
        result="$(echo "$final" | grep -oE '[0-9]+/[0-9]+' | head -1)"
        att="$(echo "$final" | grep -oE 'after [0-9]+' | grep -oE '[0-9]+')"
        wall="$(cat "${rundir}/00_baseline/wall_s.txt" 2>/dev/null)"
        # config.toml is the receipt: verify the arm we asked for is the arm
        # that ran, before the row is written down as evidence.
        cfg_think="$(grep -m1 '^thinking' "${rundir}/00_baseline/config.toml" 2>/dev/null | awk '{print $3}')"
        if [[ "$cfg_think" != "$think" ]]; then
            echo "[round] ARM MISMATCH: asked thinking=${think}, config.toml says ${cfg_think}" \
                | tee -a "${slotdir}/bench.log"
            result="${result}!ARM"
        fi
        cp "${rundir}/00_baseline/config.toml" "${slotdir}/config.toml" 2>/dev/null
    fi

    # Probe BEFORE the server dies — this is the only chance to get a decode
    # number on the exact instance that served the run.
    probe="$(idle_probe)"
    echo "$probe" > "${slotdir}/probe.txt"
    d1="$(echo "$probe" | awk '{print $1}')"; d2="$(echo "$probe" | awk '{print $2}')"

    kill_servers

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$IDX" "$key" "$tier" "$think" "$started" \
        "${wall:-?}" "${result:-?}" "${att:-?}" "${d1:-?}" "${d2:-?}" "${rundir:-?}" >> "$QUEUE_TSV"
    render_round_md

    echo "[$IDX/${#QUEUE[@]}] ${key}: ${result:-?} in ${wall:-?}s, decode ${probe}"
    echo "                cooldown ${COOLDOWN}s"
    sleep "$COOLDOWN"
done

echo ""
echo "=== Round complete ==="
column -t -s$'\t' "$QUEUE_TSV" 2>/dev/null || cat "$QUEUE_TSV"
echo ""
echo "Tally:  ${ROUND_DIR}/round.md"
