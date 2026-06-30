#!/usr/bin/env bash
# run-replay-matrix.sh — bench recovery interventions on ONE replayed stuck
# state (run2: providers:bool deleted is_enabled, breaks context/mod.rs:322).
#
# Arms (each resumes the SAME fixture: clean tree + corruption applied via
# miniswe --replay-apply, so revert_to_green has a clean round-0 to return to):
#   control      temp 0.2, no extras           (baseline recovery rate)
#   temp00       temp 0.0                       (greedy — commit vs thrash)
#   temp035      temp 0.35                      (break the fixation loop)
#   debugger     reactive_debugger=true         (fresh-eyes sub-agent)
#   revert_green revert_to_green=true (K=6)     (reset tree to last green)
#
# Design controls (see session analysis 2026-06-30):
#   * INTERLEAVED round-robin (run1 of every arm, then run2, then run3) so any
#     run-position effect is balanced across arms, not confounded with an arm.
#   * gemma RESTARTED before every single run (fresh server each time) — removes
#     server-state as a variable; if the run1-is-cleanest effect vanishes, it
#     was server state.
#
# Usage: scripts/run-replay-matrix.sh <fixture_dir> [--runs 3] [--timeout 1800]
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE_NAME="miniswe-bench"
GEMMA_NAME="miniswe-bench-gemma"
LLAMA_ENDPOINT="${LLAMA_ENDPOINT:-http://localhost:8464}"
MODEL="gemma-4-26B-A4B-it"

FIXTURE=""; RUNS=3; TIMEOUT=1800
ARMS=(control temp00 temp035 debugger revert_green)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs)    RUNS="$2";    shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        -*) echo "Unknown: $1" >&2; exit 1 ;;
        *)  FIXTURE="$1"; shift ;;
    esac
done
[[ -n "$FIXTURE" ]] || { echo "usage: $0 <fixture_dir> [--runs N] [--timeout S]" >&2; exit 1; }
[[ "$FIXTURE" = /* ]] || FIXTURE="$(cd "$FIXTURE" && pwd)"
[[ -f "$FIXTURE/context.json" && -d "$FIXTURE/tree" && -f "$FIXTURE/corruption.patch" ]] \
    || { echo "fixture needs context.json + tree/ + corruption.patch: $FIXTURE" >&2; exit 1; }

MODEL_TAG="$(curl -fsS --max-time 3 "${LLAMA_ENDPOINT}/v1/models" 2>/dev/null \
    | python3 -c "import json,sys;print((json.load(sys.stdin).get('data') or [{}])[0].get('id','?'))" 2>/dev/null \
    | sed -E 's/\.gguf$//; s/[^A-Za-z0-9._-]/_/g' | cut -c1-40)"
RESULTS_DIR="${REPO_DIR}/benchmark_results/replaymatrix_$(date +%Y%m%d_%H%M%S)_${MODEL_TAG:-unknown}"
mkdir -p "$RESULTS_DIR"
ACTIVE_CONTAINER=""
cleanup() { set +e; [[ -n "$ACTIVE_CONTAINER" ]] && docker rm -f "$ACTIVE_CONTAINER" >/dev/null 2>&1; }
trap cleanup EXIT INT TERM

echo "=== Replay recovery matrix ==="
echo "Fixture: $FIXTURE"
echo "Arms:    ${ARMS[*]}"
echo "Runs:    $RUNS each (interleaved) | gemma restart per run | timeout ${TIMEOUT}s"
echo "Results: $RESULTS_DIR"

# ---- gemma lifecycle -------------------------------------------------------
restart_gemma() {
    echo "  [gemma] restarting…"
    docker ps -q --filter "name=llama-server-" | xargs -r docker rm -f >/dev/null 2>&1 || true
    docker rm -f "$GEMMA_NAME" >/dev/null 2>&1 || true
    sleep 2
    LLAMA_CONTAINER_NAME="$GEMMA_NAME" setsid nohup "$REPO_DIR/start-gemma4.sh" \
        >"$RESULTS_DIR/gemma.log" 2>&1 < /dev/null &
    for i in $(seq 1 90); do
        sleep 5
        if curl -fsS --max-time 3 "${LLAMA_ENDPOINT}/v1/models" >/dev/null 2>&1; then
            # warmup probe (first real token can lag after load)
            curl -fsS --max-time 30 "${LLAMA_ENDPOINT}/v1/chat/completions" \
                -H 'Content-Type: application/json' \
                -d '{"model":"g","messages":[{"role":"user","content":"hi"}],"max_tokens":1}' \
                >/dev/null 2>&1 && { echo "  [gemma] healthy after ~$((i*5))s"; return 0; }
        fi
    done
    echo "  [gemma] FAILED to come up (see $RESULTS_DIR/gemma.log)"; return 1
}

# ---- per-arm config --------------------------------------------------------
gen_config() {  # $1 = arm
    local arm="$1" temp="0.2" tools_extra=""
    case "$arm" in
        control)      temp="0.2" ;;
        temp00)       temp="0.0" ;;
        temp035)      temp="0.35" ;;
        debugger)     temp="0.2"; tools_extra="reactive_debugger = true" ;;
        revert_green) temp="0.2"; tools_extra="revert_to_green = true" ;;
    esac
cat <<TOML
[model]
provider = "llama-cpp"
endpoint = "$LLAMA_ENDPOINT"
model = "$MODEL"
context_window = 60000
temperature = $temp
max_output_tokens = 8000
[context]
repo_map_budget = 5000
max_rounds = 80
pause_after_rounds = 99999
[context.providers]
repo_map = false
[hardware]
vram_gb = 24.0
vram_reserve_gb = 3.0
ram_budget_gb = 80.0
[lsp]
enabled = true
diagnostic_timeout_ms = 2000
[tools]
gate_context_reset = false
$tools_extra
[logging]
level = "trace"
enabled = true
[validation]
command = "out=\$(cargo build 2>&1) || { echo \"DOES NOT COMPILE:\"; echo \"\$out\" | tail -20; exit 1; }; run=\$(MINISWE_SKIP_VALIDATION=1 ./target/debug/miniswe --system-prompt-override 'Respond only with TOKEN_XYZ and nothing else' --yes hello 2>&1); echo \"\$run\" | grep -q TOKEN_XYZ || { echo \"COMPILES but override NOT consumed. Expected TOKEN_XYZ, GOT: \$run\"; exit 1; }"
timeout_secs = 180
max_retries = 3
TOML
}

# ---- one replay run --------------------------------------------------------
CONTAINER_SCRIPT=$(cat <<'SCRIPT'
#!/bin/bash
set -uo pipefail
TIMEOUT="$1"
cd /work
cp -a /fixture/tree/. /work/           # CLEAN baseline tree
rm -rf target .miniswe
echo -e "target/\n.miniswe\n*.log" > .gitignore
mkdir -p /output/miniswe_state
ln -sfn /output/miniswe_state .miniswe
cp /config/config.toml .miniswe/config.toml
miniswe init 2>/output/miniswe_init.txt || { echo "init failed"; cat /output/miniswe_init.txt; exit 1; }
mkdir -p .miniswe/logs
git config --global --add safe.directory /work
git init -q && git add -A && git commit -q -m baseline 2>/dev/null

echo "=== REPLAY (resume from captured context; corruption applied post-snapshot) ==="
timeout "$TIMEOUT" miniswe --yes \
    --replay-context /fixture/context.json \
    --replay-apply /fixture/corruption.patch \
    > /output/stdout.txt 2> /output/stderr.txt || true

git diff > /output/diff.patch 2>/dev/null || true
git diff --name-only > /output/changed_files.txt 2>/dev/null || true

# === 6-check validation (identical to the bench) ===
PASS=0; TOTAL=0; BINARY=./target/debug/miniswe; FLAG=""
TOTAL=$((TOTAL+1)); if RUSTFLAGS="-A warnings" cargo check 2>/output/cargo_check.txt; then echo compile:PASS; PASS=$((PASS+1)); else echo compile:FAIL; fi
TOTAL=$((TOTAL+1)); if [ "$PASS" -ge 1 ] && RUSTFLAGS="-A warnings" cargo build 2>/output/cargo_build.txt; then echo build:PASS; PASS=$((PASS+1)); else echo "build:$([ "$PASS" -ge 1 ] && echo FAIL || echo SKIP)"; fi
TOTAL=$((TOTAL+1)); if [ -f "$BINARY" ]; then "$BINARY" --help >/output/help_output.txt 2>&1 || true; if grep -qiE -- '--[a-z-]*prompt[a-z-]*' /output/help_output.txt; then FLAG=$(grep -oE -- '--[a-z-]*prompt[a-z-]*' /output/help_output.txt | head -1); echo "help:PASS($FLAG)"; PASS=$((PASS+1)); else echo help:FAIL; fi; fi
TOTAL=$((TOTAL+1)); if [ -f "$BINARY" ] && [ -n "$FLAG" ] && "$BINARY" $FLAG test --help >/output/parse_output.txt 2>&1; then echo parse:PASS; PASS=$((PASS+1)); else echo parse:FAIL; fi
TOTAL=$((TOTAL+1)); if [ "$PASS" -ge 2 ] && RUSTFLAGS="-A warnings" cargo test >/output/cargo_test.txt 2>&1; then echo test:PASS; PASS=$((PASS+1)); else echo test:FAIL; fi
TOTAL=$((TOTAL+1)); if [ -f "$BINARY" ] && [ -n "$FLAG" ] && [ "$PASS" -ge 4 ]; then OUT=$(MINISWE_SKIP_VALIDATION=1 timeout 120 "$BINARY" $FLAG 'You must respond with exactly the text PONG_42 and nothing else.' --yes ping 2>/output/smoke_stderr.txt || true); echo "$OUT">/output/smoke_output.txt; if echo "$OUT" | grep -q PONG_42; then echo smoke:PASS; PASS=$((PASS+1)); else echo smoke:FAIL; fi; fi
echo "=== FINAL: ${PASS}/${TOTAL} ==="
SCRIPT
)

run_one() {  # $1=arm $2=run
    local arm="$1" idx="$2"
    local vdir="$RESULTS_DIR/$arm/run${idx}"
    mkdir -p "$vdir"
    gen_config "$arm" > "$vdir/config.toml"
    local cname="miniswe-replay-${arm}-${idx}-$$"
    ACTIVE_CONTAINER="$cname"
    local tmp; tmp=$(mktemp); echo "$CONTAINER_SCRIPT" > "$tmp"; chmod +x "$tmp"
    local t0; t0=$(date +%s)
    docker rm -f "$cname" 2>/dev/null || true
    docker run --rm --network=host \
        -v "$vdir:/output" \
        -v "$vdir/config.toml:/config/config.toml:ro" \
        -v "$FIXTURE:/fixture:ro" \
        -v "$tmp:/run.sh:ro" \
        -e MINISWE_LLM_DUMP_DIR=/output/llm_dumps \
        --name "$cname" "$IMAGE_NAME" \
        bash /run.sh "$TIMEOUT" > "$vdir/container.log" 2>&1 || true
    echo $(( $(date +%s) - t0 )) > "$vdir/wall_s.txt"
    rm -f "$tmp"; ACTIVE_CONTAINER=""
    local res; res=$(grep -oE "=== FINAL: [0-9]+/[0-9]+" "$vdir/container.log" 2>/dev/null | tail -1 | grep -oE "[0-9]+/[0-9]+" || echo "?/?")
    local fired=""
    grep -q "revert-to-green" "$vdir/stderr.txt" "$vdir/container.log" 2>/dev/null && fired=" [revert-fired]"
    echo "  => $arm/run${idx}: ${res}  wall=$(cat "$vdir/wall_s.txt")s${fired}"
}

# ---- build image once, then interleave -------------------------------------
curl -fsS --max-time 5 "${LLAMA_ENDPOINT}/v1/models" >/dev/null 2>&1 || echo "(gemma not up yet — will start it)"
echo "Building bench image (once)…"
docker build -f "$REPO_DIR/scripts/Dockerfile.benchmark" -t "$IMAGE_NAME" "$REPO_DIR" 2>&1 | tail -3

for idx in $(seq 1 "$RUNS"); do
    for arm in "${ARMS[@]}"; do
        echo ""
        echo "### round $idx — arm $arm ($(date +%H:%M:%S))"
        restart_gemma || { echo "ABORT: gemma restart failed"; exit 1; }
        run_one "$arm" "$idx"
    done
done

# ---- summary ---------------------------------------------------------------
echo ""; echo "=== MATRIX SUMMARY ==="
python3 - "$RESULTS_DIR" "${ARMS[@]}" <<'PY'
import sys,os,re,glob
rd=sys.argv[1]; arms=sys.argv[2:]
print(f"{'arm':<14} {'scores':<18} {'mean':>5} {'reverts':>8}")
for arm in arms:
    scores=[]; reverts=0
    for run in sorted(glob.glob(os.path.join(rd,arm,"run*"))):
        cl=os.path.join(run,"container.log")
        s=None
        if os.path.exists(cl):
            m=re.findall(r"FINAL: (\d+)/\d+", open(cl,errors="ignore").read())
            if m: s=int(m[-1])
        scores.append(s)
        se=os.path.join(run,"stderr.txt")
        if os.path.exists(se) and "revert-to-green] stuck" in open(se,errors="ignore").read():
            reverts+=1
    nums=[x for x in scores if x is not None]
    mean=f"{sum(nums)/len(nums):.2f}" if nums else "—"
    print(f"{arm:<14} {str(scores):<18} {mean:>5} {reverts:>8}")
PY
echo "Detailed: $RESULTS_DIR/"
