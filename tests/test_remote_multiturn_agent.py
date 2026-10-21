"""`remote/agents/multiturn_agent.py`。LLM と env は偽物に差し替える。"""

from __future__ import annotations

from pathlib import Path

import pytest

from capx.agent_api import Budget, StepResult, TaskSpec
from capx.bench import load_agent

AGENT = Path(__file__).resolve().parent.parent / "remote" / "agents" / "multiturn_agent.py"
TASK = TaskSpec(task_id="t", seed=1, instruction="キューブを積む", api_docs="open_gripper()")


class _Env:
    def __init__(self) -> None:
        self.codes: list[str] = []

    def step(self, code, *, capture_video=False):
        self.codes.append(code)
        return StepResult(request_id="r", ok=True, stdout=f"out{len(self.codes)}", stderr="")


@pytest.fixture
def agent_with(monkeypatch):
    def make(replies):
        agent = load_agent(str(AGENT))
        prompts = []

        def fake_query(args, messages):
            prompts.append([dict(m) for m in messages])
            return {"content": replies.pop(0)}

        # load_agent は sys.modules に登録しないので、モジュールの globals を直接差し替える
        monkeypatch.setitem(type(agent).run.__globals__, "query_model", fake_query)
        return agent, prompts

    return make


def test_feeds_results_back_until_the_llm_says_done(agent_with) -> None:
    agent, prompts = agent_with(["```python\na()\n```", "```python\nb()\n```", "完了"])
    env = _Env()

    result = agent.run(env, TASK, Budget(max_steps=5))

    assert env.codes == ["a()", "b()"]
    assert result.notes == {"turns": 2, "finished": True}
    assert "キューブを積む" in prompts[0][1]["content"]
    assert "out1" in prompts[1][-1]["content"], "前の実行結果が次のプロンプトに入る"
    assert len(prompts[2]) == 6, "会話が積み上がる（system + 往復）"


def test_stops_at_max_steps(agent_with) -> None:
    agent, _ = agent_with(["```python\nx()\n```"] * 3)
    env = _Env()

    result = agent.run(env, TASK, Budget(max_steps=3))

    assert len(env.codes) == 3
    assert result.notes == {"turns": 3, "finished": False}


def test_asks_again_when_there_is_no_code(agent_with) -> None:
    agent, prompts = agent_with(["考え中です", "```python\nok()\n```", "完了"])
    env = _Env()

    agent.run(env, TASK, Budget(max_steps=5))

    assert env.codes == ["ok()"]
    assert "コードブロックが見つかりません" in prompts[1][-1]["content"]
