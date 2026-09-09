# Configuration Reference

## CLI flags

Override any YAML config field from the command line:

```bash
uv run --no-sync --active capx/envs/launch.py \
    --config-path <config.yaml> \
    --model google/gemini-3.1-pro-preview \
    --server-url http://127.0.0.1:8110/chat/completions \
    --temperature 1.0 \
    --total-trials 100 \
    --num-workers 12 \
    --record-video True
```

| Flag                | Default                                  | Description                           |
| ------------------- | ---------------------------------------- | ------------------------------------- |
| `--config-path`     | *(required)*                             | Path to YAML task config              |
| `--model`           | `google/gemini-3.1-pro-preview`          | Model name                            |
| `--server-url`      | `http://127.0.0.1:8110/chat/completions` | LLM endpoint                          |
| `--temperature`     | `1.0`                                    | Sampling temperature                  |
| `--total-trials`    | from YAML                                | Number of evaluation trials           |
| `--num-workers`     | from YAML                                | Parallel worker count                 |
| `--web-ui`          | `False`                                  | Launch interactive web UI             |
| `--use-oracle-code` | `False`                                  | Run human-written reference solutions |

## YAML config format

```yaml
# env_configs/my_task/my_task.yaml
env:
  _target_: capx.envs.tasks.my_robot.my_task.MyTaskCodeEnv
  cfg:
    _target_: capx.envs.tasks.base.CodeExecEnvConfig
    low_level: my_sim_env
    privileged: false
    apis:
      - FrankaControlApi

record_video: true
output_dir: ./outputs/my_task
trials: 100
num_workers: 12
```

The `_target_` keys enable Hydra-style lazy instantiation via `capx.envs.configs.instantiate()`.

### Perception servers (api_servers)

YAML configs declare which perception/motion servers a task needs via `api_servers`:

```yaml
api_servers:
  - _target_: capx.serving.launch_sam3_server.main
    device: cuda

  - _target_: capx.serving.launch_contact_graspnet_server.main

  - _target_: capx.serving.launch_pyroki_server.main
    robot: panda_description
    target_link: panda_hand
```

Entries only carry `_target_` plus extra kwargs (`device`, `robot`, ...) — **not**
`host`/`port`. The `*_SERVICE_URL` env vars below are the single source of truth for
where each server actually lives, so the client that calls it and the check that
verifies it's running always agree on the same endpoint:

| Server | Env var | Default |
|---|---|---|
| SAM3 | `SAM3_SERVICE_URL` | `http://127.0.0.1:8114` |
| ContactGraspNet | `GRASPNET_SERVICE_URL` | `http://127.0.0.1:8115` |
| PyRoKi | `PYROKI_SERVICE_URL` | `http://127.0.0.1:8116` |
| SAM2 | `SAM2_SERVICE_URL` | `http://127.0.0.1:8113` |
| OWL-ViT | `OWLVIT_SERVICE_URL` | `http://127.0.0.1:8117` |

Before running trials, `launch.py` checks that every server listed in `api_servers`
is reachable at its resolved `*_SERVICE_URL`:
- **Reachable**: skipped, assumed already running (e.g. started externally).
- **Not reachable (default)**: raises a `RuntimeError` and exits immediately, before
  any environment/robot setup happens. Start the missing server(s) first — see
  `launch_servers.py` below — or point the env var at one that's already running.
- **Not reachable, with `CAPX_AUTO_LAUNCH_SERVERS=1` set**: auto-launches it as a
  local subprocess instead of erroring. This only kicks in when the resolved host is
  local (`127.0.0.1`/`localhost`/`0.0.0.0`) — a remote `*_SERVICE_URL` always fails
  validation instead, since launching a local process can't satisfy a remote URL.
  Auto-launch is **off by default**.

If you prefer to manage servers separately (e.g. for sharing across multiple eval runs), use `launch_servers.py`:

```bash
uv run --no-sync --active capx/serving/launch_servers.py --profile default
```

| Profile | Servers | GPU Required |
|---------|---------|-------------|
| `default` | SAM3 (8114) + ContactGraspNet (8115) + PyRoKi (8116) | Yes (~5 GB VRAM) |
| `full` | default + OWL-ViT (8117) + SAM2 (8113) | Yes (~14 GB VRAM) |
| `minimal` | PyRoKi (8116) only | No (CPU-only) |

## Adding new LLM providers

CaP-X queries language models through a local proxy server that exposes an OpenAI-compatible `/chat/completions` endpoint.

### OpenRouter (recommended for getting started)

1. Get an API key at [openrouter.ai/keys](https://openrouter.ai/keys)
2. Save it to a file in the project root:
   ```bash
   echo "sk-or-v1-your-key-here" > .openrouterkey
   ```
3. Start the proxy (supports automatic key rotation across multiple keys):
   ```bash
   uv run --no-sync --active capx/serving/openrouter_server.py --key-file .openrouterkey --port 8110
   ```

OpenRouter provides access to Gemini, GPT, Claude, DeepSeek, Qwen, and other models through a single API key.

### Option B: vLLM (local models)

```bash
uv run python -m capx.serving.vllm_server --model Qwen/Qwen2.5-Coder-7B-Instruct --port 8080 --tensor-parallel-size 4
```

### Option C: Custom providers

Providers live under `capx/serving/providers/` and implement a simple `generate_code` method. Extend to Gemini/Claude/Bedrock by adding new provider classes.

> **Note:** `.openrouterkey` is git-ignored. Never commit API keys to the repository.
