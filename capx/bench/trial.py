"""1 trial を回す。Bench の中心。

やることは 4 つだけ。

    reset -> agent.run -> evaluate -> 保存

LLM もプロンプトも multi-turn の判断もここには無い。それらは Agent の責務で、
`AgentEnv` 越しにしか env に触れない。Bench は Agent が何を返そうと、
`step()` を通ったコードを自分で記録する——自己申告には頼らない。

Local と Remote の分岐は `make_agent_env()` の 1 回だけで、この関数の中には
入らない。`AgentEnv` が `step` / `render` しか公開しないので、そもそも
「どちらか」を問い合わせる手段がない。
"""

from __future__ import annotations

import gc
import time
from typing import TYPE_CHECKING, Any

from capx.agent_api import Agent, AgentResult, Budget, BudgetExceeded, EnvUnavailable
from capx.bench.artifacts import _build_log_lines
from capx.utils.launch_utils import TrialSummary, _save_trial_artifacts

if TYPE_CHECKING:
    from capx.local_env import LocalAgentEnv


def run_trial(
    env: LocalAgentEnv,
    agent: Agent,
    trial: int,
    config: dict[str, Any],
    budget: Budget | None = None,
    *,
    allow_reference_code: bool = False,
) -> TrialSummary:
    """1 trial を回して `TrialSummary` を返す。

    Args:
        env: `AgentEnv` の実体。Bench は `reset_for_trial()` と `evaluate()` を
            使うが、Agent に渡すときは `step` / `render` しか見えない。
        agent: `run(env, task, budget)` を持つもの。
        trial: trial 番号。1 始まり。seed にもなる。
        config: 出力先・録画などの Bench 設定。
        budget: 予算。`env` 側で強制される。
        allow_reference_code: True なら `TaskSpec.reference_code` に oracle を
            入れる。`OracleAgent` を回すときだけ。
    """
    started = time.time()
    budget = budget or Budget()

    task = env.reset_for_trial(trial)
    if allow_reference_code:
        oracle = getattr(env.inner, "oracle_code", None)
        if oracle:
            task = _with_reference_code(task, oracle)

    failure: str | None = None
    try:
        result = agent.run(env, task, budget)
    except BudgetExceeded as exc:
        # Agent の責任。リトライせず、その時点の状態で採点する。
        failure = f"budget: {exc}"
        result = AgentResult()
    except EnvUnavailable:
        # 環境側の障害。Bench の呼び出し元が新しいセッションで作り直す。
        raise
    except Exception as exc:  # Agent のバグ。trial は失敗として記録する。
        failure = f"agent_error: {exc!r}"
        result = AgentResult()

    outcome = env.evaluate()
    steps = env.recorded_steps

    final_code = "\n".join(
        f"# Code block {i}\n{s.code}" for i, s in enumerate(steps)
    )
    stderr = steps[-1].stderr if steps else ""
    if failure:
        stderr = f"{stderr}\n{failure}".strip()

    info_step = {
        "sandbox_rc": outcome.sandbox_rc,
        "stdout": steps[-1].stdout if steps else "",
        "stderr": stderr,
        "task_completed": outcome.task_completed,
    }

    notes = result.notes or {}
    num_regenerations = int(notes.get("num_regenerations", 0))
    num_finishes = int(notes.get("num_finishes", 0))

    log_lines = _build_log_lines(
        final_code,
        info_step,
        outcome.reward,
        outcome.terminated,
        outcome.truncated,
        num_regenerations,
        num_finishes,
        len(steps),
        stderr_override=stderr,
    )

    code_path = None
    if config.get("output_dir"):
        code_path = _save_trial_artifacts(
            config,
            trial,
            outcome.sandbox_rc,
            outcome.reward,
            bool(outcome.task_completed),
            final_code,
            None,
            [],
            log_lines,
            [],
        )

    _save_videos(env, config, trial, info_step, outcome, steps)

    print(f"Trial {trial} took {time.time() - started:.2f} seconds")
    gc.collect()

    return TrialSummary(
        trial=trial,
        success=outcome.sandbox_rc == 0,
        reward=outcome.reward,
        terminated=outcome.terminated,
        truncated=outcome.truncated,
        sandbox_rc=outcome.sandbox_rc,
        log="\n".join(log_lines),
        task_completed=outcome.task_completed,
        code_path=code_path,
        num_regenerations=num_regenerations,
        num_finishes=num_finishes,
        num_code_blocks=len(steps),
    )


def _with_reference_code(task, code: str):
    """`TaskSpec` は frozen なので差し替えた新しいものを作る。"""
    from dataclasses import replace

    return replace(task, reference_code=code)


def _save_videos(env, config, trial, info_step, outcome, steps) -> None:
    """録画していれば、ターンごとと結合の動画を書く。

    フレーム範囲は `LocalAgentEnv` が `step()` ごとに控えたもの。Agent が
    何ターン回したかに関わらず、通った分だけが残る。
    """
    if not config.get("record_video") or not steps:
        return

    from capx.bench.artifacts import _save_trial_video, _save_turn_and_combined_videos

    ranges = [s.frame_range for s in steps]
    if any(end > start for start, end in ranges):
        _save_turn_and_combined_videos(
            env.inner, config, trial, info_step, outcome.reward, ranges
        )
    else:
        _save_trial_video(
            env.inner, config, trial, info_step, outcome.reward, len(steps)
        )
