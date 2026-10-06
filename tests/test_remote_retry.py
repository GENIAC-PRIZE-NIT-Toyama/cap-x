"""インフラ障害でやり直すときの待ち時間。失敗のたびに倍にする。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from capx.agent_api import EnvUnavailable

TRIAL = Path(__file__).resolve().parent.parent / "remote" / "trial.py"


@pytest.fixture
def trial_module(monkeypatch):
    spec = importlib.util.spec_from_file_location("remote_trial_under_test", TRIAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Env:
    def close(self) -> None:
        pass


def _holder():
    return {"env": _Env()}


def test_waits_double_with_each_failure(trial_module, monkeypatch) -> None:
    attempts = []

    def flaky(env, runner, trial, config, **kw):
        attempts.append(kw["infrastructure_retries"])
        if len(attempts) <= 3:
            raise EnvUnavailable("切れた")
        return "done"

    monkeypatch.setattr(trial_module, "run_trial", flaky)
    waits: list[float] = []

    result = trial_module._run_with_retries(
        _holder(), None, 1, {}, None, {}, _Env, sleep=waits.append
    )

    assert result == "done"
    assert waits == [10.0, 20.0, 40.0]
    assert attempts == [0, 1, 2, 3], "やり直した回数が結果に渡る"


def test_gives_up_after_the_limit_without_waiting_again(trial_module, monkeypatch) -> None:
    def always(*a, **kw):
        raise EnvUnavailable("切れた")

    monkeypatch.setattr(trial_module, "run_trial", always)
    waits: list[float] = []

    with pytest.raises(EnvUnavailable):
        trial_module._run_with_retries(_holder(), None, 1, {}, None, {}, _Env, sleep=waits.append)
    assert waits == [10.0, 20.0, 40.0], "上限に達した失敗のあとは待たずに終わる"


def test_no_wait_when_nothing_fails(trial_module, monkeypatch) -> None:
    monkeypatch.setattr(trial_module, "run_trial", lambda *a, **kw: "ok")
    waits: list[float] = []
    trial_module._run_with_retries(_holder(), None, 1, {}, None, {}, _Env, sleep=waits.append)
    assert waits == []
