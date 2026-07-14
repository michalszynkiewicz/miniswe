#!/usr/bin/env bash
# E2E test for long-running-action support (shell promotion + `jobs` tool)
# against a REAL model. Two scenarios:
#
#   happy — a fake deploy that takes ~110s and succeeds. The model must let
#           it get promoted, monitor with jobs(wait, check=./check-deploy.sh)
#           watching READY counts rise, observe FINISHED, and report success.
#   stuck — a fake deploy that HANGS forever while the probe reports a
#           terminal ImagePullBackOff. The only correct move is jobs(kill)
#           + reporting the root cause. The job itself never exits.
#
# Both are self-contained (no k8s/docker): the "deploy" prints phases by
# wall-clock elapsed time, and the probe computes its report from the same
# start-timestamp file — so probe output CHANGES between monitoring cycles.
#
# Usage: scripts/run-jobs-e2e.sh [happy|stuck|both]   (default: both)
#
# Grading is mechanical (grep + filesystem ground truth), printed as
# PASS/FAIL per criterion. Known-risk criterion: repeated identical
# jobs(wait) calls can trip the period-1 loop-detector nudge — reported
# as INFO, not failure (evidence for a possible jobs exemption).
set -uo pipefail

SCENARIO="${1:-both}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MINISWE="${MINISWE:-$REPO_DIR/target/debug/miniswe}"
LLAMA_ENDPOINT="${LLAMA_ENDPOINT:-http://localhost:8464}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_DIR/benchmark_results/jobs_e2e_$(date +%Y%m%d_%H%M%S)}"
RUN_TIMEOUT=600

if [ ! -x "$MINISWE" ]; then
    echo "miniswe binary not found at $MINISWE (cargo build first, or set MINISWE=)" >&2
    exit 1
fi
if ! curl -fsS --max-time 3 "$LLAMA_ENDPOINT/v1/models" >/dev/null 2>&1; then
    echo "model server not responding at $LLAMA_ENDPOINT" >&2
    exit 1
fi

# ── Workspace scaffolding ────────────────────────────────────────────────

write_common_scripts() { # $1 = workspace dir
    local ws="$1"
    # Probe: reports "cluster state" derived from elapsed time since the
    # deploy recorded its start. Scenario-specific phase table is appended
    # by the caller as check-phases.sh.
    cat > "$ws/check-deploy.sh" <<'EOS'
#!/usr/bin/env bash
# Report deployment status (like `kubectl get pods` would).
if [ ! -f .deploy-started ]; then
    echo "no deployment in progress"
    exit 0
fi
elapsed=$(( $(date +%s) - $(cat .deploy-started) ))
source ./check-phases.sh "$elapsed"
EOS
    chmod +x "$ws/check-deploy.sh"
}

write_happy_workspace() { # $1 = workspace dir
    local ws="$1"
    write_common_scripts "$ws"
    cat > "$ws/fake-deploy.sh" <<'EOS'
#!/usr/bin/env bash
# Fake deploy: ~110s, then succeeds.
date +%s > .deploy-started
echo "$(( $(cat .deploy-runs 2>/dev/null || echo 0) + 1 ))" > .deploy-runs
echo "Loading package app-with-deps"
echo "Deploying component 'app' (this takes a few minutes)"
for i in $(seq 1 11); do
    sleep 10
    echo "waiting for pods to become ready ($((i * 10))s)"
done
echo "Deployment complete."
touch deployed.ok
EOS
    chmod +x "$ws/fake-deploy.sh"
    cat > "$ws/check-phases.sh" <<'EOS'
elapsed="$1"
if   [ "$elapsed" -lt 30 ];  then echo "READY 1/5 — pulling images"
elif [ "$elapsed" -lt 60 ];  then echo "READY 2/5 — containers creating"
elif [ "$elapsed" -lt 90 ];  then echo "READY 4/5 — almost there"
elif [ "$elapsed" -lt 110 ]; then echo "READY 5/5 — finalizing"
else echo "READY 5/5 — deployment healthy"
fi
EOS
}

write_stuck_workspace() { # $1 = workspace dir
    local ws="$1"
    write_common_scripts "$ws"
    cat > "$ws/fake-deploy-stuck.sh" <<'EOS'
#!/usr/bin/env bash
# Fake deploy that HANGS: waits forever on readiness that never comes.
# Its own output looks slow, not doomed — the probe carries the bad news.
date +%s > .deploy-started
echo "$(( $(cat .deploy-runs 2>/dev/null || echo 0) + 1 ))" > .deploy-runs
echo "Loading package app-with-deps"
echo "Deploying component 'app'"
n=0
while true; do
    sleep 15
    n=$((n + 1))
    echo "still waiting for pods to become ready (attempt $n)"
done
EOS
    chmod +x "$ws/fake-deploy-stuck.sh"
    cat > "$ws/check-phases.sh" <<'EOS'
elapsed="$1"
if [ "$elapsed" -lt 10 ]; then
    echo "READY 0/5 — containers creating"
else
    restarts=$(( elapsed / 10 ))
    echo "pod app-0: ImagePullBackOff — image \"app:9.9.9\" not found (${restarts} restarts)"
    echo "READY 0/5 — deployment cannot progress"
fi
EOS
}

init_workspace() { # $1 = workspace dir
    local ws="$1"
    (cd "$ws" && "$MINISWE" init >/dev/null 2>&1) || {
        echo "miniswe init failed in $ws" >&2
        return 1
    }
    # Promotion at 20s instead of 60 keeps a full run at ~3-5 min. Trace
    # logging turns on llm_dumps — grading reads tool results from there
    # (the TUI trace truncates them to one line).
    python3 - "$ws/.miniswe/config.toml" "$LLAMA_ENDPOINT" <<'EOF'
import re, sys
path, endpoint = sys.argv[1], sys.argv[2]
s = open(path).read()
s = re.sub(r'(?m)^endpoint\s*=.*$', f'endpoint = "{endpoint}"', s, count=1)
if "[shell]" in s:
    s = re.sub(r'(?ms)(\[shell\].*?default_timeout_secs\s*=\s*)\d+', r'\g<1>20', s, count=1)
else:
    s += '\n[shell]\ndefault_timeout_secs = 20\n'
if "[logging]" in s:
    s = re.sub(r'(?ms)(\[logging\].*?level\s*=\s*)"[^"]*"', r'\g<1>"trace"', s, count=1)
    s = re.sub(r'(?ms)(\[logging\].*?enabled\s*=\s*)\w+', r'\g<1>true', s, count=1)
else:
    s += '\n[logging]\nlevel = "trace"\nenabled = true\n'
open(path, "w").write(s)
EOF
}

# ── Grading helpers ──────────────────────────────────────────────────────

PASS_COUNT=0
FAIL_COUNT=0
check() { # $1 = label, $2 = 0/1 success
    if [ "$2" -eq 0 ]; then
        echo "  ✓ $1"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo "  ✗ $1"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

run_scenario() { # $1 = happy|stuck
    local scenario="$1"
    local ws
    ws=$(mktemp -d /tmp/jobs-e2e-XXXXXX)
    local out="$RESULTS_ROOT/$scenario"
    mkdir -p "$out"
    echo "═══ scenario: $scenario  (workspace $ws) ═══"

    local task deploy_script
    if [ "$scenario" = "happy" ]; then
        write_happy_workspace "$ws"
        deploy_script="fake-deploy.sh"
        task="Deploy the app by running ./fake-deploy.sh and confirm the deployment completes successfully. \
It takes a few minutes. Monitor progress with ./check-deploy.sh — the deployment is done when it \
reports READY 5/5 and the file deployed.ok exists. Do not modify any files; just deploy, monitor, and report."
    else
        write_stuck_workspace "$ws"
        deploy_script="fake-deploy-stuck.sh"
        task="Deploy the app by running ./fake-deploy-stuck.sh. Monitor progress with ./check-deploy.sh. \
If a pod is in ImagePullBackOff the deployment can never succeed — in that case abort the deployment \
and report the root cause. Do not modify any files; just deploy, monitor, and report the outcome."
    fi

    init_workspace "$ws" || return 1

    local start
    start=$(date +%s)
    (cd "$ws" && MINISWE_LLM_DUMP_DIR="$out/llm_dumps" timeout "$RUN_TIMEOUT" "$MINISWE" --yes "$task") \
        > "$out/stdout.txt" 2> "$out/stderr.txt"
    local exit_code=$?
    local wall=$(( $(date +%s) - start ))
    echo "  run: exit=$exit_code wall=${wall}s (timeout=$RUN_TIMEOUT)"
    cp -r "$ws/.miniswe/logs" "$out/miniswe-logs" 2>/dev/null
    cp -r "$ws/.miniswe/llm_dumps" "$out/llm_dumps" 2>/dev/null

    local trace="$out/stderr.txt"
    local clean
    clean="$out/stderr-clean.txt"
    sed 's/\x1b\[[0-9;]*m//g' "$trace" > "$clean"
    # Model-visible content: tool results live in the llm dumps (the TUI
    # trace truncates them); fall back to the trace if dumps are absent.
    local dumps="$out/llm_dumps"
    model_saw() { # $1 = extended regex
        if [ -d "$dumps" ] && [ -n "$(ls -A "$dumps" 2>/dev/null)" ]; then
            grep -lqE "$1" "$dumps"/*.json 2>/dev/null
        else
            grep -qE "$1" "$clean"
        fi
    }

    # ── Common criteria ──
    check "run finished within timeout (no hang)" "$([ "$exit_code" -ne 124 ]; echo $?)"
    grep -qE "started as background job|promoted to background job" "$clean"
    check "deploy ran as a background job (explicit or promoted)" $?
    grep -qE "→ jobs\(" "$clean"
    check "model called the jobs tool" $?
    model_saw "READY [0-9]/5|ImagePullBackOff"
    check "monitoring probe output reached the model" $?
    local runs
    runs=$(cat "$ws/.deploy-runs" 2>/dev/null || echo 0)
    check "deploy executed exactly once (no blind re-run), got $runs" "$([ "$runs" = "1" ]; echo $?)"

    # ── Scenario-specific criteria ──
    if [ "$scenario" = "happy" ]; then
        [ -f "$ws/deployed.ok" ]
        check "ground truth: deployed.ok exists" $?
        model_saw "FINISHED"
        check "model saw the FINISHED job report" $?
        grep -qiE "success|complete|deployed" "$out/stdout.txt"
        check "final answer reports success" $?
    else
        grep -qE "→ jobs\(.*kill|job [0-9]+ killed" "$clean" || model_saw "job [0-9]+ killed"
        check "model KILLED the stuck job via jobs(kill)" $?
        sleep 1
        ! pgrep -f "$deploy_script" >/dev/null
        check "no orphan deploy process after the run" $?
        grep -qi "ImagePullBackOff" "$out/stdout.txt"
        check "final answer cites the root cause (ImagePullBackOff)" $?
        ! grep -qiE "deployed successfully|deployment (succeeded|completed successfully)" "$out/stdout.txt"
        check "final answer does NOT claim success" $?
    fi

    # ── Known-risk INFO (not graded) ──
    local nudges
    nudges=$(grep -c "Repeated read" "$clean" 2>/dev/null)
    echo "  ℹ loop-detector nudges during monitoring: ${nudges:-0} (evidence for a jobs exemption if >0)"

    # Cleanup: kill any stragglers, drop workspace.
    pkill -f "$deploy_script" 2>/dev/null
    rm -rf "$ws"
    echo
}

echo "results: $RESULTS_ROOT"
mkdir -p "$RESULTS_ROOT"
case "$SCENARIO" in
    happy) run_scenario happy ;;
    stuck) run_scenario stuck ;;
    both)  run_scenario happy; run_scenario stuck ;;
    *) echo "usage: $0 [happy|stuck|both]" >&2; exit 1 ;;
esac

echo "═══ TOTAL: $PASS_COUNT passed, $FAIL_COUNT failed ═══"
[ "$FAIL_COUNT" -eq 0 ]
