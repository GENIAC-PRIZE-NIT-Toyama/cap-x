# Shared image for the default PerceptionAPI profile: sam3, graspnet, pyroki.
# Each server is launched by overriding `command:` per docker-compose service,
# so the three services can each be started independently, pinned to a
# different GPU, restarted independently, etc.
#
# No CUDA toolkit / nvcc is required: neither capx/third_party/sam3 nor
# capx/third_party/contact_graspnet_pytorch ship custom CUDA extensions, and
# the torch wheel (cu128) bundles its own CUDA runtime libraries. The host
# only needs an NVIDIA driver + nvidia-container-toolkit.
#
# capx/third_party/sam3 and capx/third_party/contact_graspnet_pytorch are
# git submodules. .dockerignore excludes .git, so the build context relies
# on them already being checked out on the host:
#   git submodule update --init capx/third_party/sam3 \
#       capx/third_party/contact_graspnet_pytorch
# (a bare `git clone` of this repo leaves those dirs empty).
FROM python:3.10-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# pyproject.toml declares `capx` itself as the installable project, so the
# full (dockerignore-filtered) source tree must be present before `uv sync`
# runs -- it installs capx as an editable package, not just its dependencies.
COPY pyproject.toml uv.lock README.md ./
COPY capx ./capx

# --extra contactgraspnet pulls in contact_graspnet_pytorch; sam3 and pyroki
# are already base dependencies. --frozen requires uv.lock to already match
# pyproject.toml (it does, this repo commits its lockfile).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra contactgraspnet

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/hf-cache

EXPOSE 8114 8115 8116
