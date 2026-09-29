"""人が書いた正解コードを 1 回実行するだけの Agent。

LLM を呼ばないので、環境と API が動いているかの確認に使える。失敗したら
Agent ではなく環境側の問題、という切り分けができる。

正解コードは `TaskSpec.reference_code` から受け取る。Bench が明示的に許可した
ときだけそこに入るので、Agent が勝手に取れるわけではない。
"""

from __future__ import annotations

from typing import Any

from capx.agent_api import AgentEnv, AgentResult, Budget, TaskSpec


class OracleAgent:
    """`task.reference_code` をそのまま 1 回 `step()` に渡す。"""

    def __init__(self, ctx: Any = None) -> None:
        self.ctx = ctx

    def run(self, env: AgentEnv, task: TaskSpec, budget: Budget) -> AgentResult:
        if not task.reference_code:
            raise ValueError(
                f"{task.task_id} に oracle コードが無い。"
                " Bench が reference_code を渡していないか、タスクが持っていない"
            )
        result = env.step(task.reference_code)
        return AgentResult(
            notes={"ok": result.ok, "truncation_reason": result.truncation_reason},
            llm_calls=0,
        )
