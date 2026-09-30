"""自分の Agent を GPU マシンのシミュレータで動かす。

    cd remote
    uv run --env-file .env trial.py --task cube_stack --agent agents/example_agent.py

やること: Agent ファイルを読み、GPU マシンに環境を用意してもらい、Agent を 1 回
走らせて、結果を表示する。中身は `capx.bench` にあり、ここは引数を受けて呼ぶだけ。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import tyro

from capx.agent_api import EnvUnavailable
from capx.bench import load_agent, make_agent_env, run_trial
from capx.bench.args import AgentArgs
from capx.bench.errors import explain, explain_load, highlight_generated_error

HERE = Path(__file__).resolve().parent


def main(
    *,
    agent: str = "agents/example_agent.py",
    task: str = "cube_stack",
    total_trials: int = 1,
    seed: int | None = None,
    model: str = os.environ.get("OPENAI_BASE_MODEL", ""),
    output_dir: str | None = None,
    endpoint: str | None = None,
) -> None:
    """Agent を 1 回（または --total-trials 回）走らせる。

    Args:
        agent: Agent ファイル。`remote/` からの相対パスでも、実行した場所からでもよい。
        task: タスク名。GPU マシンが用意している名前から選ぶ。
        total_trials: 試行回数。
        seed: 初期配置の seed。省略すると試行番号（1 始まり）。
        model: LLM のモデル名。省略すると環境変数 OPENAI_BASE_MODEL。
        output_dir: 実行したコードやログの保存先。
        endpoint: worker に直接つなぐ（`tcp://host:19500`）。通常は使わない。
    """
    agent_path = _resolve_agent(agent)
    server_url = os.environ.get("CAPX_ENV_SERVER_URL")
    if not endpoint and not server_url:
        sys.exit(
            "GPU マシンの場所が分からない。環境変数 CAPX_ENV_SERVER_URL を設定する"
            "（remote/.env.example をコピーして .env にし、"
            " `uv run --env-file .env trial.py` で実行する）"
        )

    print(f"Agent : {agent_path}")
    ctx = AgentArgs(model=model).to_context()
    try:
        runner = load_agent(str(agent_path), ctx=ctx)
    except Exception as exc:
        sys.exit(str(explain_load(exc, str(agent_path))))

    env = _connect(endpoint, server_url, task)
    config = {"output_dir": output_dir, "record_video": False}
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        summaries = [
            run_trial(env, runner, trial, config, seed=seed)
            for trial in range(1, total_trials + 1)
        ]
    except EnvUnavailable as exc:
        sys.exit(str(explain(exc)))
    finally:
        steps = list(env.recorded_steps)
        env.close()

    _show(summaries, steps)


def _resolve_agent(spec: str) -> Path:
    """`remote/` の中でも外でも動くよう、両方を探す。解決後のパスを見せる。"""
    for candidate in (Path(spec), HERE / spec):
        if candidate.exists():
            return candidate.resolve()
    sys.exit(f"Agent ファイルが見つからない: {spec}\n  探した場所: {Path(spec).resolve()} と {HERE / spec}")


def _connect(endpoint: str | None, server_url: str | None, task: str):
    try:
        if endpoint:
            from capx.remote_env.client import RemoteAgentEnv

            return RemoteAgentEnv(endpoint=endpoint)
        return make_agent_env(server_url=server_url, task_id=task)
    except Exception as exc:
        sys.exit(f"{explain(exc)}\n  接続先: {endpoint or server_url}")


def _show(summaries, steps=()) -> None:
    """結果を人間向けに出す。Agent には渡らない（採点は Bench が読む）。"""
    print()
    for s in summaries:
        print(f"trial {s.trial}")
        print(f"  task_completed : {s.task_completed}")
        print(f"  reward         : {s.reward:.4f}")
        print(f"  steps_used     : {s.num_code_blocks}")
        if not s.success:
            print("  (コードの実行でエラーが出ました)")
            failed = next((st for st in reversed(steps) if not st.ok), None)
            hint = highlight_generated_error(failed.code, failed.stderr) if failed else None
            print(f"  {hint}" if hint else s.log)
    if len(summaries) > 1:
        done = sum(bool(s.task_completed) for s in summaries)
        mean = sum(s.reward for s in summaries) / len(summaries)
        print(f"\n{len(summaries)} trial: task_completed {done} 件 / reward 平均 {mean:.4f}")


if __name__ == "__main__":
    tyro.cli(main)
