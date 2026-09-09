# syntax=docker/dockerfile:1.7
# The line above must stay the very first line of the file: it pins the
# BuildKit Dockerfile frontend version that understands the `RUN <<'PY'`
# heredoc used below to patch contact_graspnet_pytorch.

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
# git submodules. Dockerfile.dockerignore (next to this file) excludes .git,
# so the build context relies on them already being checked out on the host
# -- run this from the repo root before building:
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

# contact_graspnet_pytorch/data.py imports SceneRenderer (pyrender ->
# pyglet) at module load time, purely as a side effect of sharing a file
# with the point-cloud utility functions GraspEstimator actually needs.
# pyglet's window backend probes a real X server at import time and
# crashes without one, which breaks the serving path in this headless
# image. SceneRenderer itself is only used by PointCloudReader (training-
# data generation), which this image never touches.
#
# contact_graspnet_pytorch is a git submodule pointing at someone else's
# fork (capx/third_party/contact_graspnet_pytorch, see .gitmodules) that
# this project has no push access to, so the fix is applied here at build
# time instead of as a submodule commit. The assert makes the build fail
# loudly (rather than silently no-op) if upstream ever changes this line.
RUN python3 - <<'PY'
path = "capx/third_party/contact_graspnet_pytorch/contact_graspnet_pytorch/data.py"
old = "from contact_graspnet_pytorch.scene_renderer import SceneRenderer\n"
new = (
    "try:\n"
    "    from contact_graspnet_pytorch.scene_renderer import SceneRenderer\n"
    "except Exception:\n"
    "    SceneRenderer = None  # pyrender/pyglet unavailable in this headless image\n"
)
with open(path) as f:
    src = f.read()
assert src.count(old) == 1, (
    f"expected exactly one occurrence of the SceneRenderer import in {path}; "
    "upstream contact_graspnet_pytorch may have changed, update this patch"
)
with open(path, "w") as f:
    f.write(src.replace(old, new, 1))
print("patched:", path)
PY

# --extra contactgraspnet pulls in contact_graspnet_pytorch; sam3 and pyroki
# are already base dependencies. --frozen requires uv.lock to already match
# pyproject.toml (it does, this repo commits its lockfile).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra contactgraspnet

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/hf-cache

EXPOSE 8114 8115 8116
