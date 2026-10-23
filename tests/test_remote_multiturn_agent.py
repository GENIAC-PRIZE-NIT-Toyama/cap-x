"""`remote/agents/multiturn_agent.py`。LLM と env は偽物に差し替える。"""

from __future__ import annotations

import copy
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

    def render(self, camera="main"):
        return b"\xff\xd8JPEG"


def _text(message: dict) -> str:
    content = message["content"]
    return content if isinstance(content, str) else content[0]["text"]


def _images(messages: list[dict]) -> int:
    return sum(
        1
        for m in messages
        if isinstance(m["content"], list)
        for part in m["content"]
        if part["type"] == "image_url"
    )


@pytest.fixture
def agent_with(monkeypatch):
    def make(replies):
        agent = load_agent(str(AGENT))
        prompts = []

        def fake_query(args, messages):
            # Agent は前の発言から画像を外すので、呼ばれた時点の中身を写しておく
            prompts.append(copy.deepcopy(messages))
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
    assert "キューブを積む" in _text(prompts[0][1])
    assert "out1" in _text(prompts[1][-1]), "前の実行結果が次のプロンプトに入る"
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
    assert "コードブロックが見つかりません" in _text(prompts[1][-1])


def test_only_the_latest_image_is_sent(agent_with) -> None:
    agent, prompts = agent_with(["```python\na()\n```", "```python\nb()\n```", "完了"])

    agent.run(_Env(), TASK, Budget(max_steps=5))

    assert [_images(p) for p in prompts] == [1, 1, 1], "画像は毎回いちばん新しい 1 枚だけ"
    assert prompts[2][-1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "省略" in _text(prompts[2][1]), "古い発言の画像は外して、そう書いておく"


def test_images_can_be_turned_off(agent_with) -> None:
    agent, prompts = agent_with(["完了"])
    type(agent).run.__globals__["USE_IMAGES"] = False
    try:
        agent.run(_Env(), TASK, Budget(max_steps=5))
    finally:
        type(agent).run.__globals__["USE_IMAGES"] = True
    assert _images(prompts[0]) == 0


def test_no_render_after_the_last_step(agent_with) -> None:
    class Counting(_Env):
        renders = 0

        def render(self, camera="main"):
            Counting.renders += 1
            return super().render(camera)

    agent, _ = agent_with(["```python\nx()\n```"] * 2)
    agent.run(Counting(), TASK, Budget(max_steps=2))
    assert Counting.renders == 2, "最初と 1 回目の結果のときだけ撮る。最後の step の後は聞かないので撮らない"


def test_falls_back_to_text_without_an_image(agent_with) -> None:
    class NoCamera(_Env):
        def render(self, camera="main"):
            return b""

    agent, prompts = agent_with(["完了"])
    agent.run(NoCamera(), TASK, Budget(max_steps=5))
    assert isinstance(prompts[0][1]["content"], str), "空の画像は送らない"


def test_old_image_note_is_replaced(agent_with) -> None:
    agent, prompts = agent_with(["```python\na()\n```", "完了"])
    agent.run(_Env(), TASK, Budget(max_steps=5))
    old = _text(prompts[1][1])
    assert old.endswith("（このときの画像は省略）") and "添えます" not in old
