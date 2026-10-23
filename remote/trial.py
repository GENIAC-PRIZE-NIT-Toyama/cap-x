"""自分の Agent を GPU マシンのシミュレータで動かす。

    cd remote
    uv run --env-file .env trial.py --task cube_stack --agent agents/example_agent.py

やること: Agent ファイルを読み、GPU マシンに環境を用意してもらい、Agent を 1 回
走らせて、結果を表示する。中身は `capx.bench` にあり、ここは引数を受けて呼ぶだけ。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import tyro

from capx.agent_api import EnvUnavailable
from capx.bench import load_agent, make_agent_env, run_trial
from capx.bench.args import AgentArgs
from capx.bench.errors import explain, explain_load, highlight_generated_error

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:  # `tracing` を、どこから読み込まれても見つけられるように
    sys.path.insert(0, str(HERE))

import tracing  # noqa: E402


def main(
    *,
    agent: str = "agents/example_agent.py",
    task: str = "cube_stack",
    total_trials: int = 1,
    seed: int | None = None,
    model: str = os.environ.get("OPENAI_BASE_MODEL", ""),
    output_dir: str | None = None,
    save: bool = True,
    record_video: bool | None = None,
    trace: bool = True,
    trace_step_videos: bool = False,
    endpoint: str | None = None,
) -> None:
    """Agent を 1 回（または --total-trials 回）走らせる。

    Args:
        agent: Agent ファイル。`remote/` からの相対パスでも、実行した場所からでもよい。
        task: タスク名。GPU マシンが用意している名前から選ぶ。
        total_trials: 試行回数。
        seed: 初期配置の seed。省略すると試行番号（1 始まり）。
        model: LLM のモデル名。省略すると環境変数 OPENAI_BASE_MODEL。
        output_dir: 結果を置く場所。省略すると `remote/outputs/`（どこから実行しても同じ）。
            その下の `<モデル名>/<タスク名>/<実行した日時>/` に、試行ごとのフォルダ
            （実行したコード・ログ・LLM の入出力・動画・画像・result.json）を作る。
            実行ごとに日時のフォルダが分かれるので、前の結果と混ざらない。
        save: `--no-save` で結果を保存しない。
        record_video: 動画を受け取って保存する。省略すると、結果を保存するときは保存する
            （動画の書き出しで各ステップが少し長くなる）。
        trace: `--no-trace` で Langfuse に送らない。キー（LANGFUSE_PUBLIC_KEY /
            LANGFUSE_SECRET_KEY）が無ければ、付けなくても送らない。
        trace_step_videos: `env.step` の記録に、その step の動画も付ける。Langfuse の容量が
            試行あたり約 2 倍になるので、失敗した試行を詳しく見たいときだけ使う。
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

    run_dir = _make_run_dir(output_dir, model, task) if save else None
    if run_dir:
        print(f"保存先: {run_dir}")
    if record_video is None:
        record_video = save
    config = {"output_dir": str(run_dir) if run_dir else None, "record_video": record_video}
    meta = {"agent_spec": str(agent_path), "model": model, "endpoint_task": task}
    session = tracing.session_id(run_dir, model, task)
    if tracing.setup(trace, step_videos=trace_step_videos):
        print(f"トレース: Langfuse の session {session}")

    holder = {"env": _connect(endpoint, server_url, task, record_video)}
    summaries: list = []
    steps: list = []
    try:
        for trial in range(1, total_trials + 1):
            summary = _run_with_retries(
                holder, runner, trial, config, seed, meta,
                lambda: _connect(endpoint, server_url, task, record_video),
                session=session,
            )
            summaries.append(summary)
            steps = list(holder["env"].recorded_steps)
    except EnvUnavailable as exc:
        sys.exit(str(explain(exc)))
    finally:
        # やり直しの途中（閉じてから繋ぎ直すまで）に止まると None のまま
        if holder["env"] is not None:
            holder["env"].close()
        tracing.flush()

    _show(summaries, steps)


#: GPU マシン側の障害（接続断など）で trial をやり直す回数。
MAX_INFRA_RETRIES = 3

#: やり直すまでの待ち。失敗のたびに倍にする（10 秒 → 20 秒 → 40 秒）。続けて失敗する
#: ときは GPU マシン側が不調なので、すぐ作り直し続けて負荷をかけない。
RETRY_BASE_WAIT_S = 10.0


def _run_with_retries(
    holder, runner, trial, config, seed, meta, reconnect, sleep=time.sleep, session: str = ""
):
    """環境側の障害なら、新しいセッションを作り直して同じ trial をやり直す。

    Agent の責任（予算超過・Agent のバグ）ではリトライしない。中断した後の環境は
    状態が分からないので、使い回さず作り直す。やり直した回数は結果に残る。
    attempt ごとに 1 つトレースを作る（Langfuse が有効なとき）。
    """

    def attempt(retries):
        if holder["env"] is None:
            holder["env"] = reconnect()
        with tracing.trial_trace(
            session=session,
            task=meta.get("endpoint_task", ""),
            trial=trial,
            seed=seed if seed is not None else trial,
            attempt=retries + 1,
            model=meta.get("model", ""),
            agent_path=meta.get("agent_spec", ""),
            tags=["trial"],
            infra_errors=(EnvUnavailable,),
        ) as trace:
            summary = run_trial(
                trace.wrap(holder["env"]), runner, trial, config, seed=seed, meta=meta,
                infrastructure_retries=retries,
            )
            trace.finish(summary)
            return summary

    def drop_env(exc, retries, gave_up):
        if not gave_up:  # 使い切ったときは呼び出し元が閉じる
            holder["env"].close()
            holder["env"] = None

    return _retry_infra(attempt, on_failure=drop_env, action="新しいセッションでやり直す", sleep=sleep)


def _retry_infra(
    attempt,
    *,
    retryable=(EnvUnavailable,),
    on_failure=None,
    action="やり直す",
    describe=lambda exc: str(explain(exc)),
    sleep=time.sleep,
):
    """`attempt(やり直した回数)` を呼び、インフラの失敗なら待ってやり直す。

    やり直しの決まり（回数と待ち時間）はここだけに置く。`suite.py` も使う。
    やり直す前に、失敗のたびに倍にした時間（10 秒、20 秒、40 秒）だけ待つ。

    Args:
        retryable: やり直す例外の型。
        on_failure: 失敗のたびに `(例外, やり直した回数, 使い切ったか)` で呼ぶ。
            後片付けや記録に使う。
        action: 待つときの表示（「〜秒待って、<action>」）。
        describe: 例外を表示用の文に直す。

    Raises:
        やり直しを使い切ったら、最後の例外をそのまま投げる。
    """
    retries = 0
    while True:
        try:
            return attempt(retries)
        except retryable as exc:
            gave_up = retries >= MAX_INFRA_RETRIES
            if on_failure is not None:
                on_failure(exc, retries, gave_up)
            if gave_up:
                raise
            retries += 1
            wait_s = RETRY_BASE_WAIT_S * 2 ** (retries - 1)
            print(f"{describe(exc)}\n  {wait_s:.0f} 秒待って、{action}（{retries}/{MAX_INFRA_RETRIES}）")
            sleep(wait_s)


#: 結果の既定の置き場所。ローカル実行の `outputs/`（リポジトリ直下）とは分ける。
DEFAULT_OUTPUT_ROOT = HERE / "outputs"


def _make_run_dir(root: str | None, model: str, task: str, now: float | None = None) -> Path:
    """`<root>/<モデル名>/<タスク名>/<日時>/` を作って返す。

    モデル名の `/` は `_` にする（ローカル実行と同じ）。同じ秒に 2 回実行したときは
    `_2` などを付けて、前の結果に重ねない。`suite.py` はタスク名の代わりに
    `suite_<名前>` を渡す。
    """
    base = (Path(root) if root else DEFAULT_OUTPUT_ROOT) / (model.replace("/", "_") or "model_unset") / task
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
    base.mkdir(parents=True, exist_ok=True)
    for n in range(1, 1000):
        run_dir = base / (stamp if n == 1 else f"{stamp}_{n}")
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            continue
    raise RuntimeError(f"保存先を作れない: {base / stamp}")


def _resolve_agent(spec: str) -> Path:
    """`remote/` の中でも外でも動くよう、両方を探す。解決後のパスを見せる。"""
    for candidate in (Path(spec), HERE / spec):
        if candidate.exists():
            return candidate.resolve()
    sys.exit(f"Agent ファイルが見つからない: {spec}\n  探した場所: {Path(spec).resolve()} と {HERE / spec}")


def _connect(endpoint: str | None, server_url: str | None, task: str, record_video: bool = False):
    try:
        return _open_env(endpoint, server_url, task, record_video)
    except Exception as exc:
        sys.exit(f"{explain(exc)}\n  接続先: {endpoint or server_url}")


def _open_env(endpoint: str | None, server_url: str | None, task: str, record_video: bool = False):
    """`_connect` の、失敗しても終了しない版。`suite.py` はやり直しの判断に使う。"""
    if endpoint:
        from capx.remote_env.client import RemoteAgentEnv

        return RemoteAgentEnv(endpoint=endpoint, record_video=record_video)
    return make_agent_env(server_url=server_url, task_id=task, record_video=record_video)


def _show(summaries, steps=()) -> None:
    """結果を人間向けに出す。Agent には渡らない（採点は Bench が読む）。"""
    print()
    for s in summaries:
        print(f"trial {s.trial}")
        print(f"  task_completed : {s.task_completed}")
        print(f"  reward         : {s.reward:.4f}")
        print(f"  steps_used     : {s.num_code_blocks}")
        if s.result is not None and s.result.failure:
            print(f"  failure        : {s.result.failure['message']}")
        if not s.success:
            print("  (最後のステップのコードがエラーで終わりました)")
            failed = next((st for st in reversed(steps) if not st.ok), None)
            hint = highlight_generated_error(failed.code, failed.stderr) if failed else None
            print(f"  {hint}" if hint else s.log)
    if len(summaries) > 1:
        done = sum(bool(s.task_completed) for s in summaries)
        mean = sum(s.reward for s in summaries) / len(summaries)
        ok = sum(bool(s.success) for s in summaries)
        print(
            f"\n{len(summaries)} trial: task_completion_rate {done}/{len(summaries)}"
            f" / reward 平均 {mean:.4f} / exec_ok_rate {ok}/{len(summaries)}"
        )


if __name__ == "__main__":
    tyro.cli(main)
