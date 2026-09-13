#!/bin/bash
# ARD 启动脚本
set -euo pipefail
VERSION=$(git describe --tags --abbrev=0 2>/dev/null || echo "1.0.0")
IMAGE="${ARD_IMAGE:-ard:${VERSION}}"

# Run from the repository root so the relative mounts below resolve even when
# the script is invoked from another directory.
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO_ROOT="${PWD}"

# Parse --image-dir from CLI args to auto-mount.
#
# `docker -v` needs an absolute host path: handed a relative one it treats the
# value as a *named volume* and fails with "includes invalid characters for a
# local volume name". Paths that already live inside the repository need no
# extra mount at all — the repository is mounted at /app, so the relative path
# stays valid inside the container. Any other path is made absolute and mounted
# at /data/images. The container-side argument is rewritten to match.
IMAGE_DIR=""
PASSTHRU_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --image-dir)
            if [[ $# -lt 2 ]]; then
                echo "run.sh: --image-dir requires a directory argument" >&2
                exit 2
            fi
            if [[ ! -d "$2" ]]; then
                echo "run.sh: image directory not found: $2" >&2
                exit 2
            fi
            IMAGE_DIR_ABS="$(cd "$2" && pwd)"
            if [[ "${IMAGE_DIR_ABS}" == "${REPO_ROOT}/"* ]]; then
                IMAGE_DIR_IN_CONTAINER="${IMAGE_DIR_ABS#"${REPO_ROOT}/"}"
            else
                IMAGE_DIR_IN_CONTAINER="/data/images"
                IMAGE_DIR="${IMAGE_DIR_ABS}"
            fi
            PASSTHRU_ARGS+=("--image-dir" "${IMAGE_DIR_IN_CONTAINER}")
            shift 2
            ;;
        *)
            PASSTHRU_ARGS+=("$1")
            shift
            ;;
    esac
done

# Allow ARD_IMAGE_DIR env var to override
if [[ -n "${ARD_IMAGE_DIR:-}" ]]; then
    IMAGE_DIR="${ARD_IMAGE_DIR}"
fi

IMAGE_DIR_MOUNT=()
if [[ -n "${IMAGE_DIR}" ]]; then
    IMAGE_DIR_MOUNT=(-v "${IMAGE_DIR}:/data/images:ro")
fi

docker run --rm --network=host \
    -v "${REPO_ROOT}/configs:/app/configs:ro" \
    -v "${REPO_ROOT}/ontology:/app/ontology:ro" \
    -v "${REPO_ROOT}/examples:/app/examples:ro" \
    -v "${REPO_ROOT}/.local:/app/.local:ro" \
    -v "${REPO_ROOT}/outputs:/app/outputs" \
    "${IMAGE_DIR_MOUNT[@]}" \
    "${IMAGE}" "${PASSTHRU_ARGS[@]}"
