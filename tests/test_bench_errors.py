"""失敗の種類分けと、生成コードの失敗行の取り出し。"""

from __future__ import annotations

import openai

from capx.agent_api import BudgetExceeded, CapacityFull, EnvStartFailed, EnvUnavailable
from capx.bench.errors import explain, explain_load, highlight_generated_error


def _openai_error(cls):
    """種類だけを見るので、`__init__`（HTTP クライアントの型に依存する）は通さない。"""
    return cls.__new__(cls)


def test_infrastructure_kinds_are_told_apart() -> None:
    assert explain(BudgetExceeded("execution_budget")).kind == "budget"
    assert explain(CapacityFull("上限")).kind == "capacity_full"
    assert explain(EnvStartFailed("起動失敗")).kind == "env_startup"
    assert explain(EnvUnavailable("切れた")).kind == "infrastructure"


def test_llm_failures_point_at_the_env_file() -> None:
    assert explain(_openai_error(openai.AuthenticationError)).kind == "llm_auth"
    assert explain(_openai_error(openai.APITimeoutError)).kind == "llm_timeout"
    unreachable = explain(_openai_error(openai.APIConnectionError))
    assert unreachable.kind == "llm_unreachable" and "OPENAI_BASE_URL" in unreachable.message


def test_an_agent_bug_says_where_in_the_agent_file() -> None:
    try:
        exec(compile("raise KeyError('x')", "/home/me/my_agent.py", "exec"))
    except KeyError as exc:
        result = explain(exc)
    assert result.kind == "agent_error"
    assert "my_agent.py:1" in result.message


def test_a_load_failure_is_its_own_kind() -> None:
    assert explain_load(ImportError("no"), "agents/a.py").kind == "agent_import"


def test_the_failing_line_of_the_generated_code_is_shown() -> None:
    code = "x = 1\ny = undefined_name\nz = 3"
    stderr = (
        "Traceback (most recent call last):\n"
        '  File "/app/x.py", line 5, in run\n'
        '  File "<string>", line 2, in <module>\n'
        "NameError: name 'undefined_name' is not defined\n"
    )
    hint = highlight_generated_error(code, stderr)
    assert "2 行目" in hint and "y = undefined_name" in hint and "NameError" in hint


def test_no_hint_when_the_error_is_not_in_the_generated_code() -> None:
    assert highlight_generated_error("x = 1", "some other failure") is None
