


# CaP-X

### A Framework for Benchmarking and Improving Coding Agents for Robot Manipulation

[Project Page](https://capgym.github.io/) &ensp;|&ensp; [Paper](https://arxiv.org/abs/2603.22435) 

**Max Fu<sup>&#42;,1,2</sup>, Justin Yu<sup>&#42;,2</sup>, Karim El-Refai<sup>&#42;,2</sup>, Ethan Kou<sup>&#42;,2</sup>, Haoru Xue<sup>&#42;,1,2</sup>,
Huang Huang<sup>3</sup>, Wenli Xiao<sup>4</sup>, Guanzhi Wang<sup>1</sup>, Fei-Fei Li<sup>3</sup>, Guanya Shi<sup>4</sup>, Jiajun Wu<sup>3</sup>,
Shankar Sastry<sup>2</sup>, Yuke Zhu<sup>1</sup>, Ken Goldberg<sup>&dagger;,2</sup>, Jim Fan<sup>&dagger;,1</sup>**

<sup>1</sup>NVIDIA &ensp; <sup>2</sup>UC Berkeley &ensp; <sup>3</sup>Stanford University &ensp; <sup>4</sup>Carnegie Mellon University

<sup>&#42;</sup>Equal contribution &ensp; <sup>&dagger;</sup>Equal advising



---
**CaP-X** is an open-access framework for systematically studying Code-as-Policy agents in robot manipulation. It consists of four components:


| Component      | What it does                                                                                                                                                                                   |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **CaP-Gym**    | Interactive Gymnasium environments where agents control robots by generating Python code that composes perception and control primitives. 39 tasks across Robosuite, LIBERO-PRO, and BEHAVIOR. |
| **CaP-Bench**  | Systematic benchmark evaluating coding agents across abstraction levels, interaction modes, and visual grounding modalities. 8 tiers (S1-S4 single-turn, M1-M4 multi-turn).                    |
| **CaP-Agent0** | Training-free agentic framework with multi-turn visual differencing, auto-synthesized skill libraries, and parallel ensembled reasoning.                                                       |
| **CaP-RL**     | Reinforcement learning on the coding agent via GRPO, using environment rewards to post-train language models. Transfers from sim to real with minimal gap.                                     |


---

## Installation

CaP-X uses [uv](https://docs.astral.sh/uv/) for dependency management. Requires **Python 3.10** and a **CUDA-capable GPU**.

```bash
git clone --recurse-submodules https://github.com/capgym/cap-x && cd cap-x

# Or if already cloned without --recurse-submodules:
git submodule update --init --recursive

# Install uv (if not present)
curl -LsSf https://astral.sh/uv/install.sh | sh

uv python install 3.10 && uv venv -p 3.10

# Base install
uv sync
```

### Simulator-specific setup

Pick **one** simulator family to install, as Robosuite (1.5.0) and LIBERO (`robosuite==1.4.0`) would be in conflict.

#### Robosuite

```bash
uv sync --extra robosuite
```

#### LIBERO-PRO

LIBERO requires a **separate virtual environment**.

```bash
uv venv .venv-libero --python 3.12
source .venv-libero/bin/activate
uv sync --active --extra libero --extra contactgraspnet
```

See [docs/libero-tasks.md](docs/libero-tasks.md) for running any of 130+ LIBERO tasks.

#### BEHAVIOR (Isaac Sim)

BEHAVIOR tasks run on NVIDIA Isaac Sim via OmniGibson. Requires Python 3.10 and CUDA 12.x.

```bash
cd capx/third_party/b1k
./uv_install.sh --dataset          # installs OmniGibson, Isaac Sim, BDDL, cuRobo, and downloads assets
cd ../../..                        # back to repo root

# Post-install fix — copy cuRobo JIT headers (run with b1k venv active)
source capx/third_party/b1k/.venv/bin/activate
cp capx/third_party/curobo/src/curobo/curobolib/cpp/*.h \
   $(python -c "import sysconfig; print(sysconfig.get_path('purelib'))")/curobo/curobolib/cpp/
```

> The `--dataset` flag downloads robot assets, BEHAVIOR-1K scene/object assets, and 2025 challenge task instances. You will be prompted to accept the NVIDIA Isaac Sim EULA and BEHAVIOR dataset license. To auto-accept, add `--accept-dataset-tos`.

For headless servers:
```bash
sudo apt-get update && sudo apt-get install -y libegl1 libgl1
# Remove duplicate Vulkan ICD if present (causes segfault on multi-GPU systems)
sudo rm -f /usr/share/vulkan/icd.d/nvidia_icd.json
```

See [docs/behavior-tasks.md](docs/behavior-tasks.md) for task details and expected baselines.

### Optional extras

```bash
uv sync --extra verl             # RL training with VeRL/GRPO
uv sync --extra contactgraspnet  # Contact-GraspNet grasp planning
uv sync --extra curobo           # cuRobo GPU-accelerated IK & motion planning (requires CUDA)
```

## Quick Start

### 1. Perception servers (auto-launched)

Perception servers (SAM3, ContactGraspNet, PyRoKi) are **auto-launched** by the YAML config when you run an evaluation. No manual setup required for most configs.

> **SAM3 authentication:** SAM3 weights require HuggingFace access. Request access at the [SAM3 repo](https://github.com/facebookresearch/sam3), then authenticate locally with `huggingface-cli login`. Weights are cached after first download.

To pre-launch servers (e.g. for sharing across multiple eval runs):

```bash
# Start SAM3 + GraspNet + PyRoKi with automatic GPU allocation
uv run --no-sync --active capx/serving/launch_servers.py --profile default
```

Use `--dry-run` to preview the allocation. Other profiles:

```bash
--profile full      # All perception servers (SAM3, GraspNet, PyRoKi, OWL-ViT, SAM2)
--profile minimal   # PyRoKi only (for oracle/privileged evals)
```

### 2. Set up an LLM proxy

The evaluation harness queries an LLM through a local proxy that exposes an OpenAI-compatible API.

```bash
# OpenRouter (get a key at openrouter.ai/keys)
echo "sk-or-v1-your-key-here" > .openrouterkey
uv run --no-sync --active capx/serving/openrouter_server.py --key-file .openrouterkey --port 8110
```

> **Note:** `.openrouterkey` are git-ignored. The default server URL in configs is `http://127.0.0.1:8110/chat/completions`. 

See [docs/configuration.md](docs/configuration.md) for all provider options (OpenRouter, vLLM, custom).

### 3. Run evaluation

```bash
# Robosuite: single-turn benchmark (100 trials, 12 parallel workers)
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml \
    --model "google/gemini-3.1-pro-preview"

# Robosuite: multi-turn with visual differencing
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack_multiturn_vdm.yaml \
    --model "google/gemini-3.1-pro-preview"

# LIBERO-PRO: spatial task (requires .venv-libero)
source .venv-libero/bin/activate
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/libero/franka_libero_spatial_0.yaml \
    --model "google/gemini-3.1-pro-preview"

# BEHAVIOR: R1Pro radio pickup (20 trials) — requires b1k venv
source capx/third_party/b1k/.venv/bin/activate
OMNI_KIT_ACCEPT_EULA=YES OMNIGIBSON_HEADLESS=1 \
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/r1pro/r1pro_pick_up_radio.yaml \
    --model "google/gemini-3.1-pro-preview"

# Interactive Web UI
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml \
    --web-ui True
# Open http://localhost:8200

# Regression tests
./scripts/regression_test.sh quick    # 10-trial smoke test (~30s)
./scripts/regression_test.sh test1    # Full single-turn (~3 min)
```

> **Tip (BEHAVIOR):** Isaac Sim uses `OMNIGIBSON_GPU_ID` (not `CUDA_VISIBLE_DEVICES`) to select the GPU. For best performance on multi-GPU systems, run perception servers on a separate GPU (e.g. `OMNIGIBSON_GPU_ID=0` for the eval, and pre-launch SAM3/GraspNet with `CUDA_VISIBLE_DEVICES=1`). Set `OMNI_KIT_ACCEPT_EULA=YES` and `OMNIGIBSON_HEADLESS=1` for headless evaluation.

---

## Perception API / PyRoKi / OpenAI 互換 API のセットアップ（日本語）

CaP-X は Perception/Motion サーバー（SAM3, ContactGraspNet, PyRoKi, SAM2, OWL-ViT）と、コード生成に使う LLM を OpenAI 互換 API 経由で呼び出します。どちらの接続先も環境変数（`.envrc`）で指定するのが唯一の情報源です。`env_configs/**/*.yaml` の `api_servers[]` には `host`/`port` は書かず、`_target_` と追加パラメータ（`device` など）のみを持ちます。

### 1. `.envrc` を用意する

```bash
cp .envrc.example .envrc
```

`.envrc` を編集して必要な値を設定します。[direnv](https://direnv.net/) を使っていれば `direnv allow .` で `cd` するたびに自動読み込みされ、使っていなければ都度 `source .envrc` してください。

```bash
# --- OpenAI 互換 API (LLM) ---
export CAPX_LLM_WIRE=responses                     # "chat" (Chat Completions) か "responses" (Responses API)
export OPENAI_BASE_URL=https://api.openai.com/v1    # 末尾に /chat/completions は付けない
export OPENAI_API_KEY=sk-...

# --- Perception / PyRoKi サーバー ---
export SAM3_SERVICE_URL=http://127.0.0.1:8114
export GRASPNET_SERVICE_URL=http://127.0.0.1:8115
export PYROKI_SERVICE_URL=http://127.0.0.1:8116
export SAM2_SERVICE_URL=http://127.0.0.1:8113
export OWLVIT_SERVICE_URL=http://127.0.0.1:8117

# サーバーが未起動のときにローカルへ自動起動したい場合のみ有効化（デフォルトOFF）
# export CAPX_AUTO_LAUNCH_SERVERS=1
```

> `capx/envs/launch.py` は評価を始める前に、上記の `*_SERVICE_URL` が指すホストへ接続できるか確認します。接続できない場合は **デフォルトではエラーで停止**します（実機やシミュレータのセットアップに入る前に fail-fast させるため）。ローカルに未起動のサーバーがあれば自動的に起動してほしい場合のみ `CAPX_AUTO_LAUNCH_SERVERS=1` を設定してください（`*_SERVICE_URL` がリモートホストを指している場合、この設定をしても自動起動はされずエラーのままです）。

### 2. Perception API / PyRoKi サーバーを起動する

`.envrc` の URL が指すホスト・ポートでサーバーを起動しておきます。

```bash
# SAM3 + ContactGraspNet + PyRoKi をまとめて起動（GPU自動割り当て）
uv run --no-sync --active capx/serving/launch_servers.py --profile default

# PyRoKi のみでよい場合（oracle/privileged 評価など、GPU不要）
uv run --no-sync --active capx/serving/launch_servers.py --profile minimal

# OWL-ViT・SAM2 も含めてすべて起動
uv run --no-sync --active capx/serving/launch_servers.py --profile full

# 起動せずに割り当て内容だけ確認
uv run --no-sync --active capx/serving/launch_servers.py --profile default --dry-run
```

個別に1サーバーだけ起動したい場合は、直接呼び出すこともできます（ポートは `.envrc` の値と合わせてください）。

```bash
uv run --no-sync --active python -m capx.serving.launch_pyroki_server --port 8116 --host 127.0.0.1
```

> **SAM3 の認証:** SAM3 の重みは HuggingFace のアクセス権が必要です。[SAM3 リポジトリ](https://github.com/facebookresearch/sam3) でアクセスをリクエストし、`huggingface-cli login` でログインしてください（初回ダウンロード後はキャッシュされます）。

> **別ノードで起動している場合:** すでに別マシン（GPUサーバー等）で Perception/PyRoKi サーバーが起動済みなら、この手順は不要です。`.envrc` の該当する `*_SERVICE_URL` をそのホストのアドレスに向けるだけで、`launch.py` はローカルで起動せずそのサーバーをそのまま使います。

### 3. 評価を実行する

`.envrc` を読み込んだ状態（`direnv` 有効、または `source .envrc` 済み）で、通常どおり `launch.py` を実行するだけです。

```bash
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml \
    --model "google/gemini-3.1-pro-preview"
```

より詳しい設定項目は [docs/configuration.md](docs/configuration.md) を参照してください。

---

## Documentation

| Guide | Contents |
| ----- | -------- |
| [Adding Environments](docs/adding-environments.md) | Creating simulators, task environments, YAML configs |
| [Adding APIs](docs/adding-apis.md) | Implementing and registering new robot control APIs |
| [Configuration](docs/configuration.md) | YAML format, CLI flags, LLM provider setup |
| [LIBERO-PRO Tasks](docs/libero-tasks.md) | Setup, running any of 130+ LIBERO tasks, suite reference |
| [BEHAVIOR Tasks](docs/behavior-tasks.md) | Setup, R1Pro tasks, expected baselines, environment variables |
| [Development](docs/development.md) | Testing, linting, LIBERO/GraspNet setup, checkpoints, known issues |
| [Real-World Franka Panda Bringup](docs/real-franka.md) | Bringup with robots_realtime, real-robot QuickStart |
| [RL Training](docs/rl-training.md) | CaP-RL with GRPO/VeRL, sim-to-real transfer |
| [Skill Library Compilation](scripts/skill_library_compilation/README.md) | Analyze eval outputs, compile reusable skill libraries |

---

## Citation

```bibtex
@article{fu2025capx,
  title     = {{CaP-X}: A Framework for Benchmarking and Improving Coding Agents for Robot Manipulation},
  author    = {Fu, Max and Yu, Justin and El-Refai, Karim and Kou, Ethan and Xue, Haoru and Huang, Huang and Xiao, Wenli and Wang, Guanzhi and Li, Fei-Fei and Shi, Guanya and Wu, Jiajun and Sastry, Shankar and Zhu, Yuke and Goldberg, Ken and Fan, Jim},
  journal   = {arXiv preprint arXiv:2603.22435},
  year      = {2025},
  url       = {https://arxiv.org/abs/2603.22435}
}
```

## License

This project is released under the [MIT License](LICENSE).
