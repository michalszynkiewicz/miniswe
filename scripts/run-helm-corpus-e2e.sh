#!/usr/bin/env bash
# Manual e2e for the fast-mode Helm template check (src/tools/fast/helm.rs):
# shallow-clones real chart repos to /tmp and runs every chart template
# through `parse_check`. Any failure is a false `[ast] broken` on a valid
# template. Too slow and network-bound for CI; run it by hand after touching
# helm.rs or the YAML check in ast.rs.
#
# Usage: scripts/run-helm-corpus-e2e.sh [--refresh]
#   --refresh   delete the cached clones and download them again
#
# Env: HELM_CORPUS_DIR (default /tmp/miniswe-helm-corpus) — where the
# charts are cloned; existing clones are reused.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CORPUS="${HELM_CORPUS_DIR:-/tmp/miniswe-helm-corpus}"

REPOS=(
    "https://github.com/bitnami/charts.git bitnami"
    "https://github.com/prometheus-community/helm-charts.git prometheus-community"
    "https://github.com/grafana/helm-charts.git grafana"
)

if [ "${1:-}" = "--refresh" ]; then
    rm -rf "$CORPUS"
fi
mkdir -p "$CORPUS"

for entry in "${REPOS[@]}"; do
    read -r url name <<<"$entry"
    if [ -d "$CORPUS/$name/.git" ]; then
        echo "using cached $CORPUS/$name ($(git -C "$CORPUS/$name" log -1 --format='%h %cs'))"
    else
        echo "cloning $url -> $CORPUS/$name"
        git clone --quiet --depth 1 "$url" "$CORPUS/$name"
    fi
done

cd "$REPO_DIR"
HELM_CORPUS_DIR="$CORPUS" cargo test --test helm_templates_realdata -- --nocapture
