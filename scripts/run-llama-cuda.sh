#!/usr/bin/env bash
# Wrapper around `docker run ghcr.io/ggml-org/llama.cpp:server-cuda13`.
#
# All args after the image are forwarded verbatim to llama-server inside
# the container, so callers (the start-*.sh scripts) keep their existing
# CLI shape — just swap `exec llama-server` for `exec .../run-llama-cuda.sh`.
#
# Why Docker?
#   The official llama.cpp project ships CUDA binaries only via Docker on
#   Linux (no Linux-CUDA tarball as of b9133). Homebrew's llama.cpp is
#   Vulkan-only. This wrapper is the cleanest path to CUDA + the MoE
#   offload knobs (--override-tensor) we use for sparse models.
#
# What it does:
#   * `--gpus all`            — expose every NVIDIA GPU to the container
#   * `--network=host`        — keep host-port semantics so existing bench
#                               scripts (and any --port the caller passes)
#                               keep working unchanged
#   * `--ulimit memlock=-1`   — allow --mlock to pin model weights
#   * mounts $HOME/models RO  — at the same path inside the container, so
#                               model paths in callers Just Work
#   * `--cap-add IPC_LOCK`    — backup for --mlock on stricter Docker hosts
#   * `--log-driver=journald` — keep the server's stdout after the container
#                               exits. The default json-file driver stores logs
#                               under the container dir, which `--rm` deletes,
#                               so a crashed server's last output is lost right
#                               when it matters most (see docs/gpu-hardening.md
#                               item 2a). Retrieve with:
#                                 journalctl CONTAINER_NAME=llama-server-<pid>
#
# Override knobs (rarely needed):
#   LLAMA_IMAGE          — pin to a specific image tag for reproducibility
#   LLAMA_CONTAINER_NAME — name the container (default: llama-server-<pid>)
#   LLAMA_EXTRA_MOUNT    — additional `-v` arg, e.g. for models stored
#                          outside ~/models. Example:
#                            LLAMA_EXTRA_MOUNT="-v /data/models:/data/models:ro"
#   LLAMA_ALLOW_EXISTING — start even if a llama-server container is already
#                          running (see the guard below)

set -euo pipefail

IMAGE="${LLAMA_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda13}"
CONTAINER_NAME="${LLAMA_CONTAINER_NAME:-llama-server-$$}"
EXTRA_MOUNT="${LLAMA_EXTRA_MOUNT:-}"

# Refuse to stack a second server on top of a running one.
#
# CLAUDE.md: "always restart the llama-server between bench runs" — a
# long-running instance makes uptime an uncontrolled variable (GPU memory
# fragmentation, thermal throttling) that confounds arms run at different
# points in its lifetime. That rule had no mechanical guard; this is it.
#
# It is also a usability fix: with --network=host a second instance dies on
# the port bind, but only after the CUDA context and model load are already
# under way, buried in llama-server's output. Fail here instead, with the
# command to fix it.
existing="$(docker ps --filter 'name=llama-server-' --format '{{.Names}}\t{{.Status}}' 2>/dev/null || true)"
if [[ -n "${existing}" && -z "${LLAMA_ALLOW_EXISTING:-}" ]]; then
    echo "Error: a llama-server container is already running:" >&2
    echo "${existing}" | sed 's/^/  /' >&2
    echo "" >&2
    echo "CLAUDE.md requires a fresh server between bench runs — otherwise server" >&2
    echo "uptime becomes an uncontrolled variable across the comparison." >&2
    echo "" >&2
    echo "Stop it with:" >&2
    echo "  docker rm -f $(echo "${existing}" | cut -f1 | tr '\n' ' ')" >&2
    echo "" >&2
    echo "Or set LLAMA_ALLOW_EXISTING=1 to start anyway (expect a port conflict)." >&2
    exit 1
fi

# shellcheck disable=SC2086  # EXTRA_MOUNT is intentionally word-split
exec docker run --rm \
    --log-driver=journald \
    --gpus all \
    --network=host \
    --ulimit memlock=-1 \
    --cap-add IPC_LOCK \
    -v "$HOME/models:$HOME/models:ro" \
    $EXTRA_MOUNT \
    --name "$CONTAINER_NAME" \
    "$IMAGE" \
    "$@"
