# CaP-X BEHAVIOR Dockerビルド


## 0. クローン

```bash
git clone --recurse-submodules https://github.com/GENIAC-PRIZE-NIT-Toyama/cap-x && cd cap-x
```

## 1. Dockerビルド

`Dockerfile.b1k` を使用してDockerイメージをビルドします。

```bash
docker build -f docker/Dockerfile.b1k -t capx-b1k:latest .
```

- cuRoboの `lerp` 競合エラー (C++20で発生) は、sedコマンドでソースコードから独自実装のlerp関数を削除します。
- 30GB超の巨大なデータセット (BEHAVIOR-1K) はビルドに含めず、コンテナ実行時にホスト側からマウントします。

## 2. コンテナ起動

```bash
docker run --rm -it \
  --runtime=nvidia \
  --gpus '"device=0"' \
  --ipc=host \
  --network=host \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v $(pwd)/env_configs:/workspace/cap-x/env_configs \
  -v $(pwd)/capx/third_party/b1k/datasets:/workspace/cap-x/capx/third_party/b1k/datasets \
  -v $(pwd)/outputs:/workspace/cap-x/outputs \
  -e TMPDIR=/workspace/cap-x/capx/third_party/b1k/datasets/tmp \
  -e HF_HOME=/workspace/cap-x/capx/third_party/b1k/datasets/hf_cache \
  capx-b1k:latest /bin/bash
```

---

## 3. データセットのダウンロード（初回のみ）

Dockerイメージには30GB超のデータセットを含めていないため、**初回起動時のみ**コンテナ内でデータセットをダウンロードし、ホスト側のマウント先（`datasets`フォルダ）に保存します。

```bash
source /workspace/cap-x/.venv/bin/activate

# OmniGibson / B1K のアセットをダウンロード
python -m omnigibson.utils.asset_utils \
    --download_omnigibson_robot_assets \
    --download_behavior_1k_assets \
    --download_2025_challenge_task_instances \
    --accept_license

# R1Pro用のURDFファイルを配置
mkdir -p capx/third_party/b1k/datasets/omnigibson-robot-assets/models/r1pro/urdf
cp capx/third_party/b1k/assets/r1pro_ik.urdf \
   capx/third_party/b1k/datasets/omnigibson-robot-assets/models/r1pro/urdf/r1pro_ik.urdf
```

---

## 4. 実行コマンド

IsaacSimの`libcrypto.so.3`により`openssl`がクラッシュし、PyTorchコンパイルに失敗します。ここではPyTorch JITコンパイルを無効化します。

```bash
source .envrc
source /workspace/cap-x/.venv/bin/activate

TORCH_COMPILE_DISABLE=1 \
OMNI_KIT_ACCEPT_EULA=YES \
OMNIGIBSON_HEADLESS=1 \
python capx/envs/launch.py \
  --config-path env_configs/r1pro/r1pro_pick_up_trash.yaml \
  --model "nvidia/Gemma-4-31B-IT-NVFP4"
```