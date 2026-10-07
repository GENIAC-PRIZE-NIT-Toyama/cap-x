"""CaP-Agent0: 現行 `trial.py` の Agent 部分を `Agent` プロトコルにしたもの。

やることは今までと同じ——初期画像を見て、LLM にコードを書かせ、実行し、
multi-turn なら差分を見て「やり直す」か「終わる」かを決める。

**変わったのは env の触り方だけ。** `CodeExecutionEnvBase` を直接呼ぶのをやめ、
`AgentEnv.step()` 越しにした。その結果、この Agent からは reward も
task_completed も見えない。採点は Bench が `evaluate()` で行う。

参加者が書く Agent はこれを継承しない。`run(env, task, budget)` を持つ
クラスであれば何でもよく、これは比較対象のベースラインとして同梱する。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from capx.agent_api import (
    AgentEnv,
    AgentResult,
    Budget,
    BudgetExceeded,
    StepResult,
    TaskSpec,
)
from capx.agent_api.components.code_extract import extract_code


@dataclass
class AgentContext:
    """Bench が Agent に渡す接続設定。

    `--model` などの CLI 引数がここに入る。Agent は使っても無視してもよい
    （モデル名をファイルに直書きする自作 Agent も許容する）。
    """

    model: str = ""
    server_url: str | None = None
    api_key: str | None = None
    wire: str | None = None
    temperature: float | None = None
    max_tokens: int = 2048 * 10
    reasoning_effort: str | None = None
    debug: bool = False

    # multi-turn / VDM の設定。空なら single-turn。
    multi_turn_prompt: str | None = None
    use_visual_feedback: bool = False
    use_img_differencing: bool = False
    use_video_differencing: bool = False
    use_wrist_camera: bool = False
    use_parallel_ensemble: bool = False
    use_multimodel: bool = False
    use_legacy_multi_turn_decision_prompt: bool = False
    visual_differencing_model: str | None = None
    visual_differencing_model_server_url: str | None = None
    visual_differencing_model_api_key: str | None = None
    visual_differencing_wire: str | None = None

    extra: dict[str, Any] = field(default_factory=dict)


class CaPAgent0:
    """multi-turn + visual differencing + ensemble のベースライン Agent."""

    def __init__(self, ctx: AgentContext | None = None) -> None:
        self.ctx = ctx or AgentContext()
        self._llm_calls = 0

    # -- Agent プロトコル --------------------------------------------------

    def run(self, env: AgentEnv, task: TaskSpec, budget: Budget) -> AgentResult:
        from capx.llm.client import ModelQueryArgs

        ctx = self.ctx
        query_args = ModelQueryArgs(
            model=ctx.model,
            server_url=ctx.server_url,
            api_key=ctx.api_key,
            wire=ctx.wire,
            temperature=ctx.temperature,
            max_tokens=ctx.max_tokens,
            reasoning_effort=ctx.reasoning_effort,
            debug=ctx.debug,
        )

        code_blocks = list(self._initial_code(query_args, task))
        block_idx = 0
        last_result: StepResult | None = None
        regenerations = 0
        finishes = 0

        while block_idx < len(code_blocks) and block_idx < budget.max_steps:
            code = code_blocks[block_idx]
            block_idx += 1

            try:
                last_result = env.step(
                    code, capture_video=ctx.use_video_differencing
                )
            except BudgetExceeded:
                # 予算切れは Agent の責任。その時点の状態で Bench が採点する。
                break

            if not ctx.multi_turn_prompt:
                continue

            # 現行と同じ: 終了済みエピソードへの操作が出たら打ち切る
            if "terminated episode" in last_result.stderr:
                break

            decision, new_code = self._decide(
                query_args, env, task, code_blocks, block_idx, last_result
            )

            if decision == "regenerate":
                new_blocks = extract_code(new_code or "")
                del code_blocks[block_idx:]
                code_blocks.extend(new_blocks)
                regenerations += 1
            elif decision == "finish":
                finishes += 1
                break

        return AgentResult(
            notes={
                "num_regenerations": regenerations,
                "num_finishes": finishes,
                "num_code_blocks": len(code_blocks),
                "last_ok": last_result.ok if last_result else False,
            },
            llm_calls=self._llm_calls,
        )

    # -- 内部 --------------------------------------------------------------

    def _initial_code(self, query_args: Any, task: TaskSpec) -> list[str]:
        """最初のコードを書かせる。

        プロンプトは `task.default_prompt`（現行 `obs["full_prompt"]` 相当）を
        そのまま使う。自作 Agent はここを自分で組み立ててよい。
        """
        from capx.llm.client import (
            query_model,
            query_model_ensemble,
            query_single_model_ensemble,
        )

        prompt = copy.deepcopy(task.default_prompt)
        ctx = self.ctx

        if ctx.use_parallel_ensemble:
            if ctx.use_multimodel:
                out = query_model_ensemble(query_args, prompt, is_multiturn=False)
            else:
                out = query_single_model_ensemble(
                    query_args, prompt, ctx.model, is_multiturn=False
                )
        else:
            out = query_model(query_args, prompt)

        self._llm_calls += 1
        return extract_code(out["content"])

    def _decide(
        self,
        query_args: Any,
        env: AgentEnv,
        task: TaskSpec,
        code_blocks: list[str],
        block_idx: int,
        result: StepResult,
    ) -> tuple[str, str | None]:
        """やり直すか終わるかを決める。

        現行の `_handle_multi_turn_step()` と同じ判断をするが、環境の状態は
        `AgentEnv.render()` と `StepResult` からしか取らない——reward は見ない。
        """
        from capx.llm.client import query_model
        from capx.utils.launch_utils import (
            _build_multi_turn_decision_prompt,
            _build_multi_turn_decision_prompt_legacy,
            _parse_multi_turn_decision,
        )

        ctx = self.ctx
        executed = "\n".join(code_blocks[:block_idx])
        remaining = "\n".join(code_blocks[block_idx:])
        filled = (ctx.multi_turn_prompt or "").format(
            executed_code_blocks=executed,
            remaining_code_blocks=remaining,
            stdout=result.stdout,
            stderr=result.stderr,
        )

        # `AgentEnv` は observation を返さないので、プロンプト組み立てに必要な
        # 形だけをここで作る。現行の組み立て関数は obs["full_prompt"] を期待する。
        obs_like = {"full_prompt": copy.deepcopy(task.default_prompt)}

        builder = (
            _build_multi_turn_decision_prompt_legacy
            if ctx.use_legacy_multi_turn_decision_prompt
            else _build_multi_turn_decision_prompt
        )
        image_b64 = None
        if ctx.use_visual_feedback or ctx.use_img_differencing:
            import base64

            raw = env.render()
            if raw:
                image_b64 = (
                    "data:image/jpeg;base64," + base64.b64encode(raw).decode("utf-8")
                )

        decision_prompt = builder(obs_like, filled, image_b64, None)
        content = query_model(query_args, decision_prompt)
        self._llm_calls += 1
        return _parse_multi_turn_decision(content["content"])
