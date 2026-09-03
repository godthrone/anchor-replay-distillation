FROM python:3.11-slim@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534
WORKDIR /app

# Build args for proxy (not persisted in image)
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

# Layer 1: dependencies
COPY uv.lock pyproject.toml .python-version README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv pip install uv && uv sync --no-install-project --frozen --no-dev

# Layer 2: source code
COPY src ./src
COPY configs ./configs
ARG VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION}
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

ENTRYPOINT ["uv", "run", "ard"]