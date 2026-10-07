"""上級者向けの入り口。シミュレータを同じマシンで回す。

    uv run --no-sync --active local/trial.py \\
        --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml \\
        --model "google/gemini-3.1-pro-preview"

自分の Agent を差し込む:

    uv run --no-sync --active local/trial.py \\
        --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml \\
        --agent local/agents/my_agent.py \\
        --model "google/gemini-3.1-pro-preview"

中身は `capx.bench` にある。ここは引数を受けて呼ぶだけで、Bench の実装は
`remote/trial.py` と共有する——指標が 2 つに分かれないようにするため。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tyro  # noqa: E402

from capx.bench import load_agent, make_agent_env, run_trial  # noqa: E402
from capx.bench.args import AgentArgs, BenchArgs, agent_args_from_config  # noqa: E402


def main(
    config_path: str,
    /,
    *,
    agent: str = "capx.baselines.capagent0.CaPAgent0",
    model: str = "",
    total_trials: int | None = None,
    output_dir: str | None = None,
    record_video: bool | None = None,
    use_oracle_code: bool | None = None,
    server_url: str | None = None,
    api_key: str | None = None,
    execution_time_s: float = 1000.0,
    max_steps: int = 10,
    debug: bool = False,
) -> None:
    """1 つの設定で trial を回す。

    Args:
        config_path: 環境を定義する YAML。
        agent: Agent の指定。`local/agents/my_agent.py` か `module.Class`。
        model: LLM のモデル名。Agent に渡る。
        total_trials: 試行回数。省略時は YAML の `trials`。
        output_dir: 成果物の出力先。
        record_video: 動画を録る（`output_dir` が要る）。
        use_oracle_code: oracle を回す。LLM を使わない疎通確認用。
        server_url: LLM のエンドポイント。省略時は `OPENAI_BASE_URL`。
        api_key: LLM の API キー。省略時は `OPENAI_API_KEY`。
        execution_time_s: `step()` に使える累計時間。LLM 待ちは含まない。
        max_steps: `step()` を呼べる回数。
    """
    from capx.envs.configs.loader import DictLoader
    from capx.envs.runner import _start_api_servers, _stop_api_servers
    from capx.utils.launch_utils import _print_and_save_summary

    bench = BenchArgs(
        config_path=config_path,
        agent=agent,
        total_trials=total_trials,
        output_dir=output_dir,
        record_video=record_video,
        use_oracle_code=use_oracle_code,
        execution_time_s=execution_time_s,
        max_steps=max_steps,
        debug=debug,
    )
    cli_agent = AgentArgs(model=model, server_url=server_url, api_key=api_key)

    configs_dict = DictLoader.load([os.path.expanduser(config_path)])
    if "env" not in configs_dict:
        raise ValueError(f"{config_path} に `env` がない")

    env_factory = configs_dict["env"]
    agent_args = agent_args_from_config(configs_dict, overrides=cli_agent)

    config = _bench_config(bench, configs_dict)
    if config["record_video"] and not config["output_dir"]:
        raise ValueError("--record-video には --output-dir が要る")
    if config["output_dir"]:
        Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)

    use_oracle = bool(config["use_oracle_code"])
    agent_spec = "capx.baselines.oracle.OracleAgent" if use_oracle else bench.agent

    server_procs = _start_api_servers(configs_dict.get("api_servers"))
    started = time.time()
    summaries = []
    try:
        env = make_agent_env(
            env_factory=env_factory,
            budget=bench.budget(),
            record_video=bool(config["record_video"]),
            wrist_camera=agent_args.use_wrist_camera,
        )
        runner = load_agent(agent_spec, ctx=agent_args.to_context(bench.debug))
        print(f"Agent: {type(runner).__name__}  ({agent_spec})")

        try:
            for trial in range(1, config["total_trials"] + 1):
                summaries.append(
                    run_trial(
                        env,
                        runner,
                        trial,
                        config,
                        budget=bench.budget(),
                        allow_reference_code=use_oracle,
                    )
                )
        finally:
            env.close()

        _print_and_save_summary(summaries, _SummaryArgs(bench, agent_args), config, started)
    finally:
        _stop_api_servers(server_procs)


class _SummaryArgs:
    """`_print_and_save_summary()` が読む属性だけを持つ薄い器。

    あちらは `LaunchArgs` を前提に `args.model` などを読む。Bench と Agent に
    分けた引数をそのまま渡せないので、必要な分だけ見せる。
    """

    def __init__(self, bench: BenchArgs, agent: AgentArgs) -> None:
        self.model = "oracle" if bench.use_oracle_code else (agent.model or "unknown")
        self.config_path = bench.config_path
        self.visual_differencing_model = agent.visual_differencing_model


def _bench_config(bench: BenchArgs, configs_dict: dict[str, Any]) -> dict[str, Any]:
    """`bench:` ブロックと旧来の最上位キーを合わせて 1 つの dict にする。

    優先順位は CLI > `bench:` > 最上位（旧形式）> 既定値。
    """
    block = configs_dict.get("bench") or {}

    def pick(name: str, default: Any) -> Any:
        cli = getattr(bench, name, None)
        if cli is not None:
            return cli
        if name in block:
            return block[name]
        legacy = {"total_trials": "trials"}.get(name, name)
        return configs_dict.get(legacy, default)

    return {
        "total_trials": pick("total_trials", 10),
        "num_workers": pick("num_workers", 1),
        "record_video": pick("record_video", False),
        "output_dir": pick("output_dir", None),
        "use_oracle_code": pick("use_oracle_code", False),
        # 既存の成果物保存が読むキー。Agent 側の設定だが config 経由で渡る。
        "use_img_differencing": False,
        "save_multiturn_prompts": configs_dict.get("save_multiturn_prompts", False),
    }


if __name__ == "__main__":
    tyro.cli(main)
