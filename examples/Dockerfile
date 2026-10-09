ARG LITELLM_IMAGE=ghcr.io/berriai/litellm:main-stable
ARG TPS_VERSION=v0.1.0

FROM ghcr.io/astral-sh/uv:0.11.7 AS uv

FROM ${LITELLM_IMAGE}
ARG TPS_VERSION
RUN --mount=from=uv,source=/uv,target=/bin/uv \
    uv pip install --python /app/.venv/bin/python --no-cache \
    "solesonic-litellm-tokens-per-second @ https://github.com/isaacpeel/solesonic-litellm-tokens-per-second/archive/refs/tags/${TPS_VERSION}.tar.gz"
