#!/bin/bash
set -euo pipefail
VERSION=$(git describe --tags --abbrev=0 2>/dev/null || echo "1.0.0")
IMAGE_NAME="${IMAGE_NAME:-ard:${VERSION}}"

# §14.3 取值规则: in a class A / public repository an intranet proxy address is
# §15.1 confidential.  It must NOT travel as `--build-arg` — that writes the
# literal value into `docker history`, which `docker history --no-trunc` shows
# in full.  It is passed as a BuildKit secret instead, mounted only on the
# Dockerfile `RUN` steps that need the network (§14.3 正例一).
#
# Each host variable maps to its own secret id (the three values differ; see the
# Dockerfile header).  A variable that is unset or empty contributes no flag at
# all: the corresponding mount is then absent, the build step sees no proxy
# variable and connects directly — a restricted network passes the secrets, an
# open one passes nothing, and neither errors out.
SECRET_ARGS=()
if [ -n "${HTTP_PROXY:-}" ]; then
    SECRET_ARGS+=(--secret "id=http_proxy,env=HTTP_PROXY")
fi
if [ -n "${HTTPS_PROXY:-}" ]; then
    SECRET_ARGS+=(--secret "id=https_proxy,env=HTTPS_PROXY")
fi
if [ -n "${NO_PROXY:-}" ]; then
    SECRET_ARGS+=(--secret "id=no_proxy,env=NO_PROXY")
fi

docker build \
    "${SECRET_ARGS[@]}" \
    --build-arg VERSION="${VERSION}" \
    -t "${IMAGE_NAME}" \
    -f docker/Dockerfile \
    .
echo "Built: ${IMAGE_NAME}"
