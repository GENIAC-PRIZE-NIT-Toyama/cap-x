# syntax=docker/dockerfile:1.7
#
# 1 セッション 1 コンテナの worker。参加者の生成コードが実行されるのはここだけ。
# backend（capx.remote_env.server）が隔離フラグ付きで `docker run` する
# （--cap-drop ALL、--internal ネットワーク、資源の上限）。隔離は起動時に
# かけるので、このファイルは動くイメージを作ることだけを担う。
#
# ビルド前に submodule が要る（リポジトリ直下で）:
#   git submodule update --init capx/third_party/sam3 capx/third_party/robosuite
#
# ビルド（リポジトリ直下で）:
#   docker build -f docker/worker/Dockerfile -t capx-worker:latest .
FROM python:3.10-slim-bookworm

# libegl1/libopengl0/libglvnd0: MuJoCo の EGL オフスクリーン描画（MUJOCO_GL=egl）
# に GLVND の loader が要る。NVIDIA 側の実体（libEGL_nvidia.so）は、実行時に
# nvidia-container-toolkit が --runtime=nvidia から差し込む。
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libegl1 \
    libopengl0 \
    libglvnd0 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# pyproject.toml は capx 自身をインストール対象にしているので、`uv sync` の前に
# ソースが揃っている必要がある（editable で入る）。
COPY pyproject.toml uv.lock README.md ./
COPY capx ./capx
COPY env_configs ./env_configs

# --frozen は uv.lock のとおりに入れる。lock が pyproject.toml より古いと、
# 足した依存（pyzmq など）が入らないまま起動して落ちる。
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra robosuite

# capx/integrations/motion/pyroki_context.py は Panda の URDF を
# robot_descriptions 経由で取得し、キャッシュが無いと github から git clone する。
# 実行時のコンテナは外へ出られない（--internal）ので、clone は必ず失敗する。
# ネットワークのあるビルド時に温めておく。HOME を /tmp に向ける（下）と
# 既定のキャッシュ先が消えるので、HOME の外に置く。
ENV ROBOT_DESCRIPTIONS_CACHE=/app/.cache/robot_descriptions
RUN .venv/bin/python -c "from robot_descriptions.loaders.yourdfpy import load_robot_description; load_robot_description('panda_description')" \
 && test -d /app/.cache/robot_descriptions/example-robot-data

# 設定やキャッシュを書くライブラリ（mujoco、fontconfig、matplotlib、robosuite の
# private macros）が、イメージの書き込み層ではなく tmpfs に書くようにする。
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=egl \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib

EXPOSE 19500

# 引数は backend が渡す: <config_path> --port --host --session-id
ENTRYPOINT ["python", "-m", "capx.remote_env.worker.server"]
