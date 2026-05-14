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
#
# Override knobs (rarely needed):
#   LLAMA_IMAGE          — pin to a specific image tag for reproducibility
#   LLAMA_CONTAINER_NAME — name the container (default: llama-server-<pid>)
#   LLAMA_EXTRA_MOUNT    — additional `-v` arg, e.g. for models stored
#                          outside ~/models. Example:
#                            LLAMA_EXTRA_MOUNT="-v /data/models:/data/models:ro"

set -euo pipefail

IMAGE="${LLAMA_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda13}"
CONTAINER_NAME="${LLAMA_CONTAINER_NAME:-llama-server-$$}"
EXTRA_MOUNT="${LLAMA_EXTRA_MOUNT:-}"

# shellcheck disable=SC2086  # EXTRA_MOUNT is intentionally word-split
exec docker run --rm \
    --gpus all \
    --network=host \
    --ulimit memlock=-1 \
    --cap-add IPC_LOCK \
    -v "$HOME/models:$HOME/models:ro" \
    $EXTRA_MOUNT \
    --name "$CONTAINER_NAME" \
    "$IMAGE" \
    "$@"
