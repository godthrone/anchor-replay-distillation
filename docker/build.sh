#!/bin/bash
set -euo pipefail
VERSION=$(git describe --tags --abbrev=0 2>/dev/null || echo "1.0.0")
IMAGE_NAME="${IMAGE_NAME:-ard:${VERSION}}"
docker build \
    --build-arg HTTP_PROXY="${HTTP_PROXY:-}" \
    --build-arg HTTPS_PROXY="${HTTPS_PROXY:-}" \
    --build-arg NO_PROXY="${NO_PROXY:-}" \
    --build-arg VERSION="${VERSION}" \
    -t "${IMAGE_NAME}" \
    -f docker/Dockerfile \
    .
echo "Built: ${IMAGE_NAME}"