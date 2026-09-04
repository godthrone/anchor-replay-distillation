#!/bin/bash
# ARD 启动脚本
set -euo pipefail
VERSION=$(git describe --tags --abbrev=0 2>/dev/null || echo "1.0.0")
IMAGE="${ARD_IMAGE:-ard:${VERSION}}"
docker run --rm \
    -v "${PWD}/configs:/app/configs:ro" \
    -v "${PWD}/data:/app/data:ro" \
    -v "${PWD}/examples:/app/examples:ro" \
    -v "${PWD}/.local:/app/.local:ro" \
    -v "${PWD}/outputs:/app/outputs" \
    -v "${ARD_IMAGE_DIR:-/tmp}:/data/images:ro" \
    "${IMAGE}" "$@"