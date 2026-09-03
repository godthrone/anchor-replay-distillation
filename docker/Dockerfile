FROM python:3.11-slim
WORKDIR /app

# Build args for proxy (not persisted in image)
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

# Layer 1: dependencies
COPY uv.lock pyproject.toml .python-version README.md LICENSE ./
RUN pip install uv && uv sync --no-install-project --frozen --no-dev

# Layer 2: source code
COPY src ./src
COPY configs ./configs
ARG VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION}
RUN uv sync --frozen --no-dev

ENTRYPOINT ["uv", "run", "ard"]