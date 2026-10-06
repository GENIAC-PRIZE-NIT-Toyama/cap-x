"""参加者が書いた Agent ファイルを読み込めるかを固定する。

これが壊れると、参加者は自分の Agent を動かせない。ワークショップの入口に
あたる部分なので、規約（`class Agent` + `run(env, task, budget)`）を
テストで押さえる。

simulator は要らないので手元PC でも走る。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from capx.bench import load_agent

AGENT_WITH_CTX = '''
class Agent:
    def __init__(self, ctx):
        self.ctx = ctx

    def run(self, env, task, budget):
        return None
'''

AGENT_WITHOUT_INIT = '''
class Agent:
    def run(self, env, task, budget):
        return None
'''

NAMED_DIFFERENTLY = '''
class MyCleverAgent:
    def run(self, env, task, budget):
        return None
'''

TWO_CANDIDATES = '''
class FirstAgent:
    def run(self, env, task, budget):
        return None


class SecondAgent:
    def run(self, env, task, budget):
        return None
'''

NO_AGENT = '''
def helper():
    return 1
'''


def _write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def test_loads_from_file_path_and_passes_ctx(tmp_path: Path) -> None:
    path = _write(tmp_path, "a.py", AGENT_WITH_CTX)
    agent = load_agent(str(path), ctx={"model": "x"})
    assert agent.ctx == {"model": "x"}


def test_agent_without_init_is_allowed(tmp_path: Path) -> None:
    """`__init__` を書かない Agent も許容する。

    `object.__init__` の署名は引数を取るように読めるので、素直に
    `inspect.signature` で判定すると `TypeError: Agent() takes no arguments`
    になる。モデル名を直書きする最小の Agent が書けなくなるので、ここを守る。
    """
    path = _write(tmp_path, "b.py", AGENT_WITHOUT_INIT)
    assert load_agent(str(path)) is not None


def test_single_run_class_is_found_even_if_named_differently(tmp_path: Path) -> None:
    """`Agent` という名前でなくても、run() を持つクラスが 1 つなら使う。"""
    path = _write(tmp_path, "c.py", NAMED_DIFFERENTLY)
    assert type(load_agent(str(path))).__name__ == "MyCleverAgent"


def test_ambiguous_file_says_what_to_do(tmp_path: Path) -> None:
    """候補が複数なら、黙ってどれかを選ばず、直し方を伝えて落ちる。"""
    path = _write(tmp_path, "d.py", TWO_CANDIDATES)
    with pytest.raises(AttributeError, match="Agent"):
        load_agent(str(path))


def test_file_without_agent_says_what_to_do(tmp_path: Path) -> None:
    path = _write(tmp_path, "e.py", NO_AGENT)
    with pytest.raises(AttributeError, match="run"):
        load_agent(str(path))


def test_missing_file_reports_the_resolved_path(tmp_path: Path) -> None:
    """相対パスで迷わないよう、解決後の絶対パスを見せる。"""
    with pytest.raises(FileNotFoundError, match=str(tmp_path)):
        load_agent(str(tmp_path / "nope.py"))


def test_loads_from_import_path() -> None:
    agent = load_agent("capx.baselines.oracle.OracleAgent")
    assert type(agent).__name__ == "OracleAgent"


def test_step_output_is_clipped_to_the_budget() -> None:
    """生成コードが大量に print しても、返す stdout / stderr は上限まで。"""
    from types import SimpleNamespace

    from capx.agent_api import Budget
    from capx.local_env import LocalAgentEnv

    class Env:
        low_level_env = SimpleNamespace(_sim_step_count=0)

        def step(self, code):
            info = {"sandbox_rc": 0, "stdout": "x" * 5000, "stderr": "ok"}
            return None, 0.0, False, False, info

    env = LocalAgentEnv(Env(), budget=Budget(max_output_bytes=100))
    result = env.step("print('x')")
    assert len(result.stdout.encode()) < 200
    assert "切り捨て" in result.stdout
    assert result.stderr == "ok", "上限内はそのまま"


def _load_source(tmp_path, source: str):
    from capx.bench import load_agent

    path = tmp_path / "agent_under_test.py"
    path.write_text(source, encoding="utf-8")
    return load_agent(str(path), ctx="CTX")


def test_an_agent_that_inherits_base_agent_is_loaded(tmp_path) -> None:
    agent = _load_source(
        tmp_path,
        "from capx.agent_api import AgentResult, BaseAgent\n"
        "class Agent(BaseAgent):\n"
        "    def run(self, env, task, budget):\n"
        "        return AgentResult()\n",
    )
    assert type(agent).__name__ == "Agent"


def test_forgetting_run_is_caught_before_connecting(tmp_path) -> None:
    """`run` を書き忘れたら、読み込みの時点で何が足りないかを言う。"""
    import pytest

    with pytest.raises(TypeError, match="run"):
        _load_source(
            tmp_path,
            "from capx.agent_api import BaseAgent\n"
            "class Agent(BaseAgent):\n"
            "    def runn(self, env, task, budget):\n"
            "        pass\n",
        )


def test_ctx_reaches_an_init_defined_on_a_parent_class(tmp_path) -> None:
    """共通処理を親クラスにまとめ、`__init__(ctx)` をそこに書いても ctx が届く。"""
    agent = _load_source(
        tmp_path,
        "from capx.agent_api import AgentResult, BaseAgent\n"
        "class MyBase(BaseAgent):\n"
        "    def __init__(self, ctx):\n"
        "        self.ctx = ctx\n"
        "class Agent(MyBase):\n"
        "    def run(self, env, task, budget):\n"
        "        return AgentResult()\n",
    )
    assert agent.ctx == "CTX"


def test_a_plain_class_without_inheritance_still_loads(tmp_path) -> None:
    agent = _load_source(
        tmp_path,
        "class Agent:\n"
        "    def run(self, env, task, budget):\n"
        "        return None\n",
    )
    assert type(agent).__name__ == "Agent"
