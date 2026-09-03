#!/bin/bash
# ARD v1.0.0 启动脚本
set -euo pipefail
IMAGE="${ARD_IMAGE:-ard:1.0.0}"
docker run --rm \
    -v "${PWD}/configs:/app/configs:ro" \
    -v "${PWD}/outputs:/app/outputs" \
    -v "${ARD_IMAGE_DIR:-/tmp}:/data/images:ro" \
    "${IMAGE}" "$@"