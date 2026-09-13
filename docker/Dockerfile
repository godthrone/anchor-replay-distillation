FROM python:3.11-slim
WORKDIR /app

# Build args for proxy (not persisted in image)
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

# Version injected by build.sh via --build-arg VERSION=<git describe>.
# Declared *before* the first `uv sync` so every metadata build in this file
# sees it: SETUPTOOLS_SCM_PRETEND_VERSION must be in the environment of the
# sync that installs the project, otherwise the dist-info directory keeps the
# fallback version instead of the released one.
ARG VERSION=0.0.0

# Layer 1: dependencies
COPY uv.lock pyproject.toml .python-version README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv pip install uv && uv sync --no-install-project --frozen --no-dev

# Layer 2: source code
COPY src ./src
COPY configs ./configs
COPY ontology ./ontology
COPY examples ./examples
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION}
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"
ENTRYPOINT ["python", "-m", "ard"]