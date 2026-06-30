#!/usr/bin/env bash
# run-replay-docker.sh — replay a real "before first fix" fixture vs the live model.
#
# Unlike run-benchmark-docker.sh (fresh checkout + task), this drops the agent
# into the EXACT state a real run was in right before its first fix: the
# captured working tree (fixture/tree/) AND the captured conversation context
# (fixture/context.json), then runs `miniswe --replay-context` and applies the
# same 6-check validation. See docs/replay-mode-design.md.
#
# Build a fixture first:
#   scripts/replay/extract-fixture.py <run>/00_baseline <fixture_dir>
#
# Usage:
#   scripts/run-replay-docker.sh <fixture_dir> [--timeout 1800] [--model gemma-4-26B-A4B-it] [--runs 1]
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE_NAME="miniswe-bench"   # reuse the benchmark image (built from current tree)

FIXTURE=""
TIMEOUT=1800
MODEL="gemma-4-26B-A4B-it"
RUNS=1
LLAMA_ENDPOINT="${LLAMA_ENDPOINT:-http://localhost:8464}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --model)   MODEL="$2";   shift 2 ;;
        --runs)    RUNS="$2";    shift 2 ;;
        -*)        echo "Unknown: $1" >&2; exit 1 ;;
        *)         FIXTURE="$1"; shift ;;
    esac
done

[[ -n "$FIXTURE" ]] || { echo "usage: $0 <fixture_dir> [--timeout N] [--model M] [--runs N]" >&2; exit 1; }
[[ "$FIXTURE" = /* ]] || FIXTURE="$(cd "$FIXTURE" && pwd)"
[[ -f "$FIXTURE/context.json" && -d "$FIXTURE/tree" ]] || { echo "fixture missing context.json or tree/: $FIXTURE" >&2; exit 1; }

MODEL_TAG="$(curl -fsS --max-time 3 "${LLAMA_ENDPOINT}/v1/models" 2>/dev/null \
    | python3 -c "import json,sys;print((json.load(sys.stdin).get('data') or [{}])[0].get('id','?'))" 2>/dev/null \
    | sed -E 's/\.gguf$//; s/[^A-Za-z0-9._-]/_/g' | cut -c1-40)"
MODEL_TAG="${MODEL_TAG:-unknown}"
RESULTS_DIR="${REPO_DIR}/benchmark_results/replay_$(date +%Y%m%d_%H%M%S)_${MODEL_TAG}"
ACTIVE_CONTAINER=""
cleanup() { set +e; [[ -n "$ACTIVE_CONTAINER" ]] && docker rm -f "$ACTIVE_CONTAINER" >/dev/null 2>&1; docker image rm -f "$IMAGE_NAME" >/dev/null 2>&1; }
trap cleanup EXIT INT TERM

mkdir -p "$RESULTS_DIR"
echo "=== Replay benchmark ==="
echo "Fixture:  $FIXTURE"
echo "Model:    $MODEL"
echo "Endpoint: $LLAMA_ENDPOINT"
echo "Timeout:  ${TIMEOUT}s | Runs: $RUNS"
echo "Manifest: $(python3 -c "import json;m=json.load(open('$FIXTURE/manifest.json'));print('round',m.get('round'),'| nmsgs',m.get('n_messages'),'| from',m.get('source_run','?').split('/')[-2] if '/' in m.get('source_run','') else '?')" 2>/dev/null)"
echo "Results:  $RESULTS_DIR"

curl -fsS --max-time 5 "${LLAMA_ENDPOINT}/v1/models" >/dev/null 2>&1 || { echo "ERROR: LLM server not at $LLAMA_ENDPOINT" >&2; exit 1; }
echo "Building image..."; docker build -f "$REPO_DIR/scripts/Dockerfile.benchmark" -t "$IMAGE_NAME" "$REPO_DIR" 2>&1 | tail -3

# Config: fast mode, refactor on, done-gate on, gate_context_reset OFF (so a
# mid-loop reset can't clobber the seeded context).
gen_config() {
cat <<TOML
[model]
provider = "llama-cpp"
endpoint = "$LLAMA_ENDPOINT"
model = "$MODEL"
context_window = 60000
temperature = 0.2
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
[logging]
level = "trace"
enabled = true
[validation]
command = "out=\$(cargo build 2>&1) || { echo \"DOES NOT COMPILE:\"; echo \"\$out\" | tail -20; exit 1; }; run=\$(MINISWE_SKIP_VALIDATION=1 ./target/debug/miniswe --system-prompt-override 'Respond only with TOKEN_XYZ and nothing else' --yes hello 2>&1); echo \"\$run\" | grep -q TOKEN_XYZ || { echo \"COMPILES but override NOT consumed. Expected TOKEN_XYZ, GOT: \$run\"; exit 1; }"
timeout_secs = 180
max_retries = 3
TOML
}

run_one() {
    local idx="$1"
    local vdir="$RESULTS_DIR/run${idx}"
    mkdir -p "$vdir"
    gen_config > "$vdir/config.toml"
    local cname="miniswe-replay-${idx}-$$"
    ACTIVE_CONTAINER="$cname"
    local script
    script=$(cat <<'SCRIPT'
#!/bin/bash
set -uo pipefail
TIMEOUT="$1"
cd /work
# Restore the captured working tree (the half-built, pre-first-fix code state).
cp -a /fixture/tree/. /work/
rm -rf target .miniswe
echo -e "target/\n.miniswe\n*.log" > .gitignore
mkdir -p /output/miniswe_state
ln -sfn /output/miniswe_state .miniswe
cp /config/config.toml .miniswe/config.toml
miniswe init 2>/output/miniswe_init.txt || { echo "init failed"; cat /output/miniswe_init.txt; exit 1; }
mkdir -p .miniswe/logs
# Fixture files keep the host uid (bind mount); git runs as root → "dubious
# ownership". Allow it so the baseline commit (and thus diff capture) works.
git config --global --add safe.directory /work
git init -q && git add -A && git commit -q -m baseline 2>/dev/null

echo "=== REPLAY (single resume from captured context) ==="
timeout "$TIMEOUT" miniswe --yes --replay-context /fixture/context.json \
    > /output/stdout.txt 2> /output/stderr.txt || true

git diff > /output/diff.patch 2>/dev/null || true
git diff --name-only > /output/changed_files.txt 2>/dev/null || true

# === 6-check validation (same as the bench) ===
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
    local tmp; tmp=$(mktemp); echo "$script" > "$tmp"; chmod +x "$tmp"
    local t0; t0=$(date +%s)
    docker rm -f "$cname" 2>/dev/null || true
    docker run --rm --network=host \
        -v "$vdir:/output" \
        -v "$vdir/config.toml:/config/config.toml:ro" \
        -v "$FIXTURE:/fixture:ro" \
        -v "$tmp:/run.sh:ro" \
        -e MINISWE_LLM_DUMP_DIR=/output/llm_dumps \
        --name "$cname" "$IMAGE_NAME" \
        bash /run.sh "$TIMEOUT" 2>&1 | tee "$vdir/container.log"
    echo $(( $(date +%s) - t0 )) > "$vdir/wall_s.txt"
    rm -f "$tmp"; ACTIVE_CONTAINER=""
    local res; res=$(grep -oE "=== FINAL: [0-9]+/[0-9]+" "$vdir/container.log" | tail -1 | grep -oE "[0-9]+/[0-9]+" || echo "?/?")
    echo "  run${idx}: ${res}  wall=$(cat "$vdir/wall_s.txt")s"
}

for i in $(seq 1 "$RUNS"); do run_one "$i"; done

echo ""; echo "=== REPLAY RESULTS ==="
for d in "$RESULTS_DIR"/run*/; do
    res=$(grep -oE "=== FINAL: [0-9]+/[0-9]+" "$d/container.log" 2>/dev/null | tail -1 | grep -oE "[0-9]+/[0-9]+" || echo "?/?")
    printf "  %s: %s  (wall %ss)\n" "$(basename "$d")" "$res" "$(cat "$d/wall_s.txt" 2>/dev/null)"
done
echo "Detailed: $RESULTS_DIR/"
