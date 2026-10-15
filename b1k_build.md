# CaP-X BEHAVIOR (b1k) ビルド

## 1. 必要ハードウェア
- GPU
  - Ada Lovelace (IsaacSim 4.5.0の場合)
  - Driver 580
  - CUDA Toolkit 12.8
- VRAM: 21GiB

## 2. エラー / 解決

### IsaacSim のグラフィッククラッシュ
- 原因: 特定のNVIDIAドライババージョンと IsaacSim 4.5.0 の間に非互換性がある。
- 解決策: ドライバのバージョンを動作確認済みのもの (580.系) に合わせる。

### ReadmeのBEHAVIOR (IsaacSim)セクション
- 原因: `source capx/third_party/b1k/.venv/bin/activate`を実行しても`b1k/.venv`は存在しない
- 解決策: ルートの.venv環境で進めてOK

### cuRobo の `lerp` 競合エラー
- 原因: C++20の `std::lerp` とcuRobo内部の独自実装 `lerp` (helper_math.h) が名前衝突を起こしてコンパイルエラーになる。  
  - Readme通りの`capx/third_party/b1k/uv_install.sh`実行ではエラーが出ず気づけない
  - コンパイルでC++20が指定されておりそのままでは回避不可能
- 解決策: `src/curobo/curobolib/cpp/helper_math.h`内の独自実装lerp関数を削除する
  - 参考: [NVIDIA Isaac Lab Documentation, Step 2: Install cuRobo](https://isaac-sim.github.io/IsaacLab/develop/source/features/imitation-learning/skillgen.html)

### JITコンパイルエラー
- 原因: 実行時にcuRoboがPyTorch C++拡張をJITコンパイルしようとするが、ビルドツール(`ninja`)やCUDAライブラリへのパスが不足している。
- 解決策:
  ```bash
  export LD_LIBRARY_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/lib:$LD_LIBRARY_PATH 
  ```

### 補足: CUDA Toolkit 12.8以外 & sudoがない場合
- 原因: CUDA Toolkit 12.8以外が入っている場合、PyTorchがcuRoboのコンパイルをブロックする。
- 解決策: PyTorchのエラーチェックを強制的に`print`へ書き換えて突破する。
  ```bash
  sed -i 's/raise RuntimeError(CUDA_MISMATCH_MESSAGE/print(CUDA_MISMATCH_MESSAGE/g' $(python -c "import sysconfig; print(sysconfig.get_path('purelib'))")/torch/utils/cpp_extension.py
  ```


## 3. 修正済みBEHAVIOR Installation
CaP-X install
```bash
git clone --recurse-submodules https://github.com/GENIAC-PRIZE-NIT-Toyama/cap-x && cd cap-x
uv python install 3.10 && uv venv -p 3.10
uv sync
source .venv/bin/activate
uv pip install ninja
```

Env settings
```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export OMNI_KIT_ACCEPT_EULA=YES

# Primitive APIで .envrc を使う場合
source .envrc

# GPUを指定する場合
nvidia-smi -L
export CUDA_VISIBLE_DEVICES="GPU-xxxx-xxxx-xxxx..."
```

b1k setup
```bash
cd capx/third_party/b1k
sed -i '/uv pip install.*curobo/s/^/#/' uv_install.sh
./uv_install.sh --dataset
```

cuRobo fix
```bash
cd ../curobo
sed -i '/^inline __device__ __host__ float lerp(float a, float b, float t)$/,+3d' src/curobo/curobolib/cpp/helper_math.h
uv pip install -e . --no-build-isolation
cd ../../..
```

Run BEHAVIOR task
```bash
OMNIGIBSON_HEADLESS=1 \
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/r1pro/r1pro_pick_up_trash_oracle.yaml

OMNIGIBSON_HEADLESS=1 \
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/r1pro/r1pro_pick_up_trash.yaml \
    --model "google/gemini-3.1-pro-preview"
```
