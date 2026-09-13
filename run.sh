#!/bin/bash
# ARD 启动脚本
set -euo pipefail
VERSION=$(git describe --tags --abbrev=0 2>/dev/null || echo "1.0.0")
IMAGE="${ARD_IMAGE:-ard:${VERSION}}"

# Parse --image-dir from CLI args to auto-mount
IMAGE_DIR="/tmp"
PASSTHRU_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --image-dir)
            IMAGE_DIR="$2"
            PASSTHRU_ARGS+=("$1" "/data/images")
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
    IMAGE_DIR="$ARD_IMAGE_DIR"
fi

docker run --rm --network=host \
    -v "${PWD}/configs:/app/configs:ro" \
    -v "${PWD}/ontology:/app/ontology:ro" \
    -v "${PWD}/examples:/app/examples:ro" \
    -v "${PWD}/.local:/app/.local:ro" \
    -v "${PWD}/outputs:/app/outputs" \
    -v "${IMAGE_DIR}:/data/images:ro" \
    "${IMAGE}" "${PASSTHRU_ARGS[@]}"