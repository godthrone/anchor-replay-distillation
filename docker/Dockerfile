# Patch-level tag, not the floating python:3.11-slim: a minor tag is re-pointed
# whenever a new patch is released, so two builds of the same commit could run
# on different interpreters (§14.3, §6).  3.11.15 is the patch the project's
# uv-managed development interpreter resolves to.
FROM python:3.11.15-slim
WORKDIR /app

# Proxy access — BuildKit secret, never ARG/ENV (§14.3 取值规则).  This is a
# class A / public repository, so an intranet proxy or mirror address is §15.1
# confidential.  `--build-arg` would write the literal value into `docker
# history` (visible via `docker history --no-trunc`); `ENV` would additionally
# write it into `Config.Env`, leaving a stale proxy that breaks the runtime
# (§14.3 代理不得污染运行时, never relaxed).  A secret enters neither.
#
# A secret is visible only to the `RUN` that mounts it, so each network step
# below mounts the ones it needs; when a secret is not supplied the mount is
# simply absent, the variable stays unset and the step connects directly — a
# restricted network passes them via build.sh, an open one passes nothing, and
# neither errors.  The three variables carry different values, so each has its
# own secret id (one value bound to three names would give NO_PROXY the proxy's
# value).  Any future mirror/index value takes the same route.

# Version injected by build.sh via --build-arg VERSION=<git describe>.
# Declared *before* the first `uv sync` so every metadata build in this file
# sees it: SETUPTOOLS_SCM_PRETEND_VERSION must be in the environment of the
# sync that installs the project, otherwise the dist-info directory keeps the
# fallback version instead of the released one.
ARG VERSION=0.0.0

# Layer 1: dependencies
COPY uv.lock pyproject.toml .python-version README.md LICENSE ./
# uv is the only package manager (§14.2) and the tool that resolves `uv.lock`,
# so its version is pinned: 0.11.7 generated the lock in this repository and is
# the version the development environment runs (`uv --version`).  A floating
# `uv` would let a future release of the resolver decide what `--frozen` means
# for this lock.  Pinning it is not possible via `uv.lock`, which records the
# resolution rather than the resolver.
RUN --mount=type=secret,id=http_proxy,env=HTTP_PROXY \
    --mount=type=secret,id=https_proxy,env=HTTPS_PROXY \
    --mount=type=secret,id=no_proxy,env=NO_PROXY \
    --mount=type=cache,target=/root/.cache/uv \
    pip install uv==0.11.7 && uv sync --no-install-project --frozen --no-dev

# Layer 2: source code
COPY src ./src
COPY configs ./configs
COPY ontology ./ontology
COPY examples ./examples
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION}
RUN --mount=type=secret,id=http_proxy,env=HTTP_PROXY \
    --mount=type=secret,id=https_proxy,env=HTTPS_PROXY \
    --mount=type=secret,id=no_proxy,env=NO_PROXY \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"
ENTRYPOINT ["python", "-m", "ard"]